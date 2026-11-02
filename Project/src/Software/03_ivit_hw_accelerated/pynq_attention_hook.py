#!/usr/bin/env python3
"""
===============================================================================
Module:         pynq_attention_hook.py
Description:    PYNQ Hardware Offload Engine for I-ViT Multi-Head Attention.
Target SoC:     AMD Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
Bus Protocol:   128-bit AXI4-Stream via AXI DMA (MM2S & S2MM)
Control Bus:    AXI4-Lite CSR Register Map (axi_lite_regs.sv)

Features:
  - Persistent zero-copy CMA DDR4 buffer pre-allocation (in_buffer, out_buffer)
  - Time-Multiplexed Sequential Head Processing (H=4 heads processed sequentially
    on the single-head 8-DSP accelerator core)
  - Full hardware acceleration for Stage 2 (N=256, dk=16, H=4) @ 150 MHz
  - Software fallback for Stages 3 (N=64, dk=20) and 4 (N=16, dk=24)
  - Detailed telemetry (DMA dispatch latency, PL kernel compute time)

Author:         Lead Embedded HW/SW Co-Design Engineer
Project:        MobileViT-XXS + I-ViT Co-Design on AMD Kria KV260
===============================================================================
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn

# PYNQ Runtime Imports (optional import for host/simulation portability)
try:
    from pynq import Overlay, allocate, Device
    PYNQ_AVAILABLE = True
except (ImportError, RuntimeError):
    PYNQ_AVAILABLE = False
    Overlay = None
    allocate = None
    Device = None


# -----------------------------------------------------------------------------
# AXI4-Lite CSR Register Offsets (Strictly aligned with axi_lite_regs.sv)
# -----------------------------------------------------------------------------
ADDR_CTRL      = 0x00  # Bit 0: START, Bit 1: CLEAR, Bit 2: RESET
ADDR_STATUS    = 0x04  # Core Status [0: busy, 1: done, 2: error]
ADDR_NBYTES    = 0x08  # Total QKV packet size in bytes (12,288 for Stage 2 single head)
ADDR_SEQ_LEN   = 0x10  # Sequence length N (256 for Stage 2)
ADDR_HEAD_DIM  = 0x14  # Head dimension dk (16 for Stage 2)
ADDR_NUM_HEADS = 0x18  # Number of heads H (1 for single-head active core)
ADDR_SHIFT_VAL = 0x1C  # ASR shift value
ADDR_SCALE_M   = 0x20  # Dyadic Mantissa M (27640 for Stage 2)
ADDR_SCALE_E   = 0x24  # Dyadic Exponent E (22 for Stage 2)


class PynqAttentionAccelerator:
    """
    Hardware Interface managing the Kria KV260 Attention Core PL Overlay.
    Processes one attention head per stream transaction (Time-Multiplexed).
    """
    def __init__(
        self,
        bitstream_path: str,
        seq_len: int = 256,
        head_dim: int = 16,
        num_heads: int = 1,
        dyadic_m: int = 27640,
        dyadic_e: int = 22
    ):
        self.bitstream_path = bitstream_path
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dyadic_m = dyadic_m
        self.dyadic_e = dyadic_e

        self.qkv_bytes = seq_len * head_dim * 3        # 256 * 16 * 3 = 12,288 Bytes per head
        self.out_bytes = seq_len * head_dim * 1        # 256 * 16 * 1 = 4,096 Bytes per head

        self.ol: Optional[Overlay] = None
        self.dma = None
        self.csr = None
        self.in_buffer = None
        self.out_buffer = None

        self.total_invocations = 0
        self.last_kernel_latency_us = 0.0

        if not PYNQ_AVAILABLE:
            print("[WARN] PYNQ runtime not available in this environment. Hardware offload disabled.")
            return

        self._initialize_hardware()

    def _initialize_hardware(self):
        """Load bitstream, find DMA and CSR IP blocks, and pre-allocate CMA DDR4 buffers."""
        if not os.path.exists(self.bitstream_path):
            raise FileNotFoundError(f"[ERROR]: Bitstream not found at '{self.bitstream_path}'")

        print(f"[PYNQ] Loading Attention Core Overlay: '{self.bitstream_path}'...")
        self.ol = Overlay(self.bitstream_path)

        # Locate AXI DMA
        self.dma = getattr(self.ol, "axi_dma_0", None)
        if self.dma is None:
            raise RuntimeError("[ERROR]: AXI DMA 'axi_dma_0' not found in Overlay!")

        # Locate AXI4-Lite CSR
        for name in ["mha_core_0", "attention_core_top_2A_0", "attention_core_top_0", "axi_lite_regs_0"]:
            if hasattr(self.ol, name):
                csr_block = getattr(self.ol, name)
                self.csr = getattr(csr_block, "s_axi", csr_block)
                print(f"  -> Identified CSR IP block: '{name}'")
                break

        if self.csr is None:
            # Fallback search through IP dict
            for k in self.ol.ip_dict.keys():
                if "mha" in k.lower() or "attention" in k.lower() or "axi_lite" in k.lower():
                    csr_block = getattr(self.ol, k)
                    self.csr = getattr(csr_block, "s_axi", csr_block)
                    print(f"  -> Identified CSR IP block via IP dict: '{k}'")
                    break

        if self.csr is None:
            raise RuntimeError("[ERROR]: Could not locate AXI4-Lite CSR IP block in overlay!")

        # Pre-allocate zero-copy contiguous memory (CMA) DDR4 buffers
        print(f"[PYNQ] Pre-allocating zero-copy CMA DDR4 buffers (IN={self.qkv_bytes} B, OUT={self.out_bytes} B)...")
        self.in_buffer = allocate(shape=(self.qkv_bytes,), dtype=np.int8)
        self.out_buffer = allocate(shape=(self.out_bytes,), dtype=np.int8)
        print(f"  -> CMA Ingress Buffer  : {hex(self.in_buffer.physical_address)}")
        print(f"  -> CMA Egress Buffer   : {hex(self.out_buffer.physical_address)}")

        # Configure static CSR parameters
        self._configure_csr()

    def _configure_csr(self):
        """Write hardware sequence length, head dimension, and dyadic scaling into CSR."""
        self.csr.write(ADDR_NBYTES,    self.qkv_bytes)
        self.csr.write(ADDR_SEQ_LEN,   self.seq_len)
        self.csr.write(ADDR_HEAD_DIM,  self.head_dim)
        self.csr.write(ADDR_NUM_HEADS, self.num_heads)
        self.csr.write(ADDR_SCALE_M,   self.dyadic_m)
        self.csr.write(ADDR_SCALE_E,   self.dyadic_e)

    def execute_attention_stage2(
        self, 
        q_tensor: torch.Tensor, 
        k_tensor: torch.Tensor, 
        v_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Stream Q, K, V for ONE head through 128-bit AXI DMA, pulse START, and retrieve Context output.
        """
        # 1. Flatten and pack into contiguous DMA buffer
        # Shape: [1, 256, 16] or [256, 16] INT8 -> 4,096 bytes each
        q_np = q_tensor.detach().cpu().numpy().astype(np.int8).flatten()
        k_np = k_tensor.detach().cpu().numpy().astype(np.int8).flatten()
        v_np = v_tensor.detach().cpu().numpy().astype(np.int8).flatten()

        q_size = len(q_np)
        k_size = len(k_np)
        v_size = len(v_np)

        self.in_buffer[0 : q_size] = q_np
        self.in_buffer[q_size : q_size + k_size] = k_np
        self.in_buffer[q_size + k_size : q_size + k_size + v_size] = v_np

        # 2. Arm DMA & Fire Hardware Accelerator
        t0 = time.perf_counter()

        # Arm receiver first to prevent buffer overrun
        self.dma.recvchannel.transfer(self.out_buffer)
        # Transmit QKV ingress packet
        self.dma.sendchannel.transfer(self.in_buffer)

        # Pulse START bit
        self.csr.write(ADDR_CTRL, 0x01)

        # Await DMA transfers
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        t1 = time.perf_counter()
        self.last_kernel_latency_us = (t1 - t0) * 1e6
        self.total_invocations += 1

        # Return a copy of the output array
        return self.out_buffer.copy()


# -----------------------------------------------------------------------------
# PyTorch Forward Monkey-Patch Hook
# -----------------------------------------------------------------------------
class HardwareAcceleratedSelfAttention(nn.Module):
    """
    Drop-in replacement for IViTSelfAttention in Stage 2.
    Implements Time-Multiplexed Sequential Head Processing:
    Processes all H heads sequentially through the single-head PL Core via AXI DMA.
    Routes other stages to standard I-ViT CPU execution.
    """
    def __init__(self, original_attn_module: nn.Module, accelerator: Optional[PynqAttentionAccelerator]):
        super().__init__()
        self.attn = original_attn_module
        self.accelerator = accelerator
        self.stage2_invocations = 0

    def forward(self, x: torch.Tensor, act_scaling_factor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        num_heads = self.attn.num_attention_heads
        head_dim = C // num_heads

        # Check if this layer matches our PL accelerator configuration (Stage 2: N=256, dk=16)
        can_offload = (
            self.accelerator is not None 
            and self.accelerator.dma is not None 
            and N == 256 
            and head_dim == 16 
        )

        if can_offload:
            # -----------------------------------------------------------------
            # 1. PS Execution: QKV Projection & Activation Quantization
            # -----------------------------------------------------------------
            x, act_scaling_factor = self.attn.qkv(x, act_scaling_factor)
            x, act_scaling_factor_1 = self.attn.qact1(x, act_scaling_factor)

            qkv = x.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, num_heads, N, head_dim]

            # -----------------------------------------------------------------
            # 2. PL Acceleration: Time-Multiplexed Sequential Head Processing
            #    Processes each head h sequentially through the 1-head PL core.
            # -----------------------------------------------------------------
            context_heads = []
            for h in range(num_heads):
                q_h = q[:, h, :, :]  # [B, N, dk]
                k_h = k[:, h, :, :]
                v_h = v[:, h, :, :]

                out_h_raw = self.accelerator.execute_attention_stage2(q_h, k_h, v_h)
                self.stage2_invocations += 1

                # Convert single head back to PyTorch Tensor [B, 1, N, head_dim]
                out_h_tensor = torch.from_numpy(out_h_raw.reshape(B, 1, N, head_dim)).to(
                    device=x.device, dtype=torch.float32
                )
                context_heads.append(out_h_tensor)

            # Concatenate all H heads along head dimension -> [B, H, N, dk]
            context_all = torch.cat(context_heads, dim=1)  # [B, 4, N, 16]
            x = context_all.transpose(1, 2).reshape(B, N, C)  # [B, N, 64]

            # Context scaling factor alignment
            act_scaling_factor = act_scaling_factor_1 * 1.0

            # -----------------------------------------------------------------
            # 3. PS Execution: Post-Attention Output Projection & Residual
            # -----------------------------------------------------------------
            x, act_scaling_factor = self.attn.qact2(x, act_scaling_factor)
            x, act_scaling_factor = self.attn.proj(x, act_scaling_factor)
            x, act_scaling_factor = self.attn.qact3(x, act_scaling_factor)

            return x, act_scaling_factor
        else:
            # Software Fallback: Execute standard I-ViT forward pass on ARM CPU
            return self.attn(x, act_scaling_factor)


def inject_hardware_attention_hook(
    model: nn.Module, 
    accelerator: Optional[PynqAttentionAccelerator]
) -> int:
    """
    Traverse the MobileViT model and wrap Stage 2 IViTSelfAttention blocks
    with HardwareAcceleratedSelfAttention.
    """
    hooks_injected = 0
    # In surgered MobileViT:
    # model.mobilevit.encoder.layer[2] is MobileViT Block 1 (Stage 2)
    try:
        stage2_block = model.mobilevit.encoder.layer[2].transformer.layer[0]
        if hasattr(stage2_block, "attn"):
            orig_attn = stage2_block.attn
            stage2_block.attn = HardwareAcceleratedSelfAttention(orig_attn, accelerator)
            hooks_injected += 1
            print(f"[HOOK] Successfully wrapped Stage 2 IViTSelfAttention with PL Accelerator Hook (Time-Multiplexed H=4).")
    except Exception as e:
        print(f"[WARN] Failed to wrap Stage 2: {e}")

    return hooks_injected
