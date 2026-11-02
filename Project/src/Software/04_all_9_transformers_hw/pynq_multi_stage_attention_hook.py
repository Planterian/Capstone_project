#!/usr/bin/env python3
"""
===============================================================================
Module:         pynq_multi_stage_attention_hook.py
Description:    Multi-Stage PYNQ Hardware Offload Engine for ALL 9 Transformer
                layers across Stage 2, Stage 3, and Stage 4 in MobileViT-XXS.
Target SoC:     AMD Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
Bus Protocol:   128-bit AXI4-Stream via AXI DMA (MM2S & S2MM)
Control Bus:    AXI4-Lite CSR Register Map (axi_lite_regs.sv)

Features:
  - Time-Multiplexed Sequential Head Processing (H=4 heads per layer)
  - Dynamic runtime CSR reconfiguration per transformer layer:
      * Stage 2 (2 layers): N=256, dk=16 (M=27640, E=22) -> Ingress=12288 B, Egress=4096 B per head
      * Stage 3 (4 layers): N=64,  dk=20 (M=21936, E=22) -> Ingress=3840 B,  Egress=1280 B per head
      * Stage 4 (3 layers): N=16,  dk=24 (M=18810, E=22) -> Ingress=1152 B,  Egress=384 B per head
  - Pre-allocated zero-copy CMA DDR4 buffer (12,288 B max) reused across all calls

Author:         Lead Embedded HW/SW Co-Design Engineer
Project:        MobileViT-XXS + I-ViT Co-Design on AMD Kria KV260
===============================================================================
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

import torch
import torch.nn as nn

# PYNQ Runtime Imports (with graceful simulation fallback)
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
ADDR_NBYTES    = 0x08  # Total QKV packet size in bytes per head
ADDR_SEQ_LEN   = 0x10  # Sequence length N
ADDR_HEAD_DIM  = 0x14  # Head dimension dk
ADDR_NUM_HEADS = 0x18  # Number of heads H (1 for single active head on PL)
ADDR_SHIFT_VAL = 0x1C  # ASR shift value
ADDR_SCALE_M   = 0x20  # Dyadic Mantissa M
ADDR_SCALE_E   = 0x24  # Dyadic Exponent E


# Stage Dyadic Scaling Calibrations (aligned with hw_scale_params.svh)
STAGE_CONFIGS = {
    # (N, head_dim): (M, E, Stage Name)
    (256, 16): {"M": 27640, "E": 22, "stage": "Stage 2"},
    (64,  20): {"M": 21936, "E": 22, "stage": "Stage 3"},
    (16,  24): {"M": 18810, "E": 22, "stage": "Stage 4"},
}


class PynqMultiStageAccelerator:
    """
    Hardware Interface managing multi-stage runtime CSR re-configuration
    and AXI DMA transfers for all 9 Transformer layers on Kria KV260.
    """
    def __init__(self, bitstream_path: str):
        self.bitstream_path = bitstream_path
        self.ol: Optional[Overlay] = None
        self.dma = None
        self.csr = None
        self.in_buffer = None
        self.out_buffer = None

        # Max allocation based on Stage 2 peak: 256 * 16 * 3 = 12,288 B per head
        self.max_in_bytes = 256 * 16 * 3   # 12,288 Bytes
        self.max_out_bytes = 256 * 16 * 1  # 4,096 Bytes

        # Profiling Telemetry
        self.total_invocations = 0
        self.stage_invocations = defaultdict(int)
        self.stage_latencies_us = defaultdict(list)
        self.last_kernel_latency_us = 0.0

        if not PYNQ_AVAILABLE:
            print("[WARN] PYNQ runtime not available. Multi-stage hardware offload disabled.")
            return

        self._initialize_hardware()

    def _initialize_hardware(self):
        """Load overlay and pre-allocate unified maximum CMA DDR4 buffers."""
        if not os.path.exists(self.bitstream_path):
            raise FileNotFoundError(f"[ERROR]: Bitstream not found at '{self.bitstream_path}'")

        print(f"[PYNQ] Loading Multi-Stage Attention Overlay: '{self.bitstream_path}'...")
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
            for k in self.ol.ip_dict.keys():
                if "mha" in k.lower() or "attention" in k.lower() or "axi_lite" in k.lower():
                    csr_block = getattr(self.ol, k)
                    self.csr = getattr(csr_block, "s_axi", csr_block)
                    print(f"  -> Identified CSR IP block via IP dict: '{k}'")
                    break

        if self.csr is None:
            raise RuntimeError("[ERROR]: Could not locate AXI4-Lite CSR IP block in overlay!")

        # Pre-allocate zero-copy contiguous memory (CMA) DDR4 buffers
        print(f"[PYNQ] Pre-allocating unified zero-copy CMA DDR4 buffers (MAX_IN={self.max_in_bytes} B, MAX_OUT={self.max_out_bytes} B)...")
        self.in_buffer = allocate(shape=(self.max_in_bytes,), dtype=np.int8)
        self.out_buffer = allocate(shape=(self.max_out_bytes,), dtype=np.int8)
        print(f"  -> Unified CMA Ingress : {hex(self.in_buffer.physical_address)}")
        print(f"  -> Unified CMA Egress  : {hex(self.out_buffer.physical_address)}")

    def execute_attention_single_head(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        seq_len: int,
        head_dim: int
    ) -> np.ndarray:
        """
        Dynamically reprogram CSR for one head of this specific stage, stream Q, K, V via AXI DMA,
        and retrieve Context output.
        """
        cfg = STAGE_CONFIGS.get((seq_len, head_dim), {"M": 27640, "E": 22, "stage": "Custom"})
        dyadic_m = cfg["M"]
        dyadic_e = cfg["E"]
        stage_name = cfg["stage"]

        qkv_bytes = seq_len * head_dim * 3
        out_bytes = seq_len * head_dim * 1

        # 1. Dynamic CSR Re-Programming
        self.csr.write(ADDR_NBYTES,    qkv_bytes)
        self.csr.write(ADDR_SEQ_LEN,   seq_len)
        self.csr.write(ADDR_HEAD_DIM,  head_dim)
        self.csr.write(ADDR_NUM_HEADS, 1)  # Single active head
        self.csr.write(ADDR_SCALE_M,   dyadic_m)
        self.csr.write(ADDR_SCALE_E,   dyadic_e)

        # 2. Flatten and slice into pre-allocated contiguous DMA buffer
        q_np = q_tensor.detach().cpu().numpy().astype(np.int8).flatten()
        k_np = k_tensor.detach().cpu().numpy().astype(np.int8).flatten()
        v_np = v_tensor.detach().cpu().numpy().astype(np.int8).flatten()

        q_size = len(q_np)
        k_size = len(k_np)
        v_size = len(v_np)

        self.in_buffer[0 : q_size] = q_np
        self.in_buffer[q_size : q_size + k_size] = k_np
        self.in_buffer[q_size + k_size : q_size + k_size + v_size] = v_np

        # 3. Trigger DMA and Measure Execution Time
        t0 = time.perf_counter()

        self.dma.recvchannel.transfer(self.out_buffer[:out_bytes])
        self.dma.sendchannel.transfer(self.in_buffer[:qkv_bytes])

        self.csr.write(ADDR_CTRL, 0x01)

        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        t1 = time.perf_counter()
        dur_us = (t1 - t0) * 1e6

        self.last_kernel_latency_us = dur_us
        self.total_invocations += 1
        self.stage_invocations[stage_name] += 1
        self.stage_latencies_us[stage_name].append(dur_us)

        return self.out_buffer[:out_bytes].copy()


# -----------------------------------------------------------------------------
# PyTorch Forward Multi-Stage Monkey-Patch Hook
# -----------------------------------------------------------------------------
class MultiStageHardwareAcceleratedSelfAttention(nn.Module):
    """
    Drop-in replacement for IViTSelfAttention wrapping ANY of the 9 transformer
    layers across Stage 2, Stage 3, and Stage 4.
    Implements Time-Multiplexed Sequential Head Processing across all H heads.
    """
    def __init__(
        self, 
        original_attn_module: nn.Module, 
        accelerator: Optional[PynqMultiStageAccelerator],
        layer_idx: int,
        stage_name: str
    ):
        super().__init__()
        self.attn = original_attn_module
        self.accelerator = accelerator
        self.layer_idx = layer_idx
        self.stage_name = stage_name
        self.invocations = 0

    def forward(self, x: torch.Tensor, act_scaling_factor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        num_heads = self.attn.num_attention_heads
        head_dim = C // num_heads

        can_offload = (
            self.accelerator is not None 
            and self.accelerator.dma is not None 
        )

        if can_offload:
            # 1. PS Execution: QKV Projection & Activation Quantization
            x, act_scaling_factor = self.attn.qkv(x, act_scaling_factor)
            x, act_scaling_factor_1 = self.attn.qact1(x, act_scaling_factor)

            qkv = x.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, num_heads, N, head_dim]

            # 2. PL Acceleration: Time-Multiplexed Sequential Head Processing
            context_heads = []
            for h in range(num_heads):
                q_h = q[:, h, :, :]  # [B, N, dk]
                k_h = k[:, h, :, :]
                v_h = v[:, h, :, :]

                out_h_raw = self.accelerator.execute_attention_single_head(
                    q_h, k_h, v_h, seq_len=N, head_dim=head_dim
                )
                self.invocations += 1

                out_h_tensor = torch.from_numpy(out_h_raw.reshape(B, 1, N, head_dim)).to(
                    device=x.device, dtype=torch.float32
                )
                context_heads.append(out_h_tensor)

            # Concatenate all H heads along head dimension -> [B, H, N, dk]
            context_all = torch.cat(context_heads, dim=1)  # [B, num_heads, N, dk]
            x = context_all.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

            # Scaling factor alignment
            act_scaling_factor = act_scaling_factor_1 * 1.0

            # 3. PS Execution: Post-Attention Output Projection & Residual
            x, act_scaling_factor = self.attn.qact2(x, act_scaling_factor)
            x, act_scaling_factor = self.attn.proj(x, act_scaling_factor)
            x, act_scaling_factor = self.attn.qact3(x, act_scaling_factor)

            return x, act_scaling_factor
        else:
            # Software Fallback
            return self.attn(x, act_scaling_factor)


def inject_all_9_hardware_attention_hooks(
    model: nn.Module, 
    accelerator: Optional[PynqMultiStageAccelerator]
) -> int:
    """
    Traverse the entire MobileViT model and inject the hardware accelerator
    hook into ALL 9 Transformer layers across Stage 2, Stage 3, and Stage 4.
    """
    hooks_injected = 0
    stage_mapping = [
        (2, "Stage 2", 2),
        (3, "Stage 3", 4),
        (4, "Stage 4", 3),
    ]

    for enc_idx, stage_name, expected_layers in stage_mapping:
        try:
            mvit_block = model.mobilevit.encoder.layer[enc_idx]
            transformer_layers = mvit_block.transformer.layer
            for l_idx in range(len(transformer_layers)):
                t_layer = transformer_layers[l_idx]
                if hasattr(t_layer, "attn"):
                    orig_attn = t_layer.attn
                    t_layer.attn = MultiStageHardwareAcceleratedSelfAttention(
                        original_attn_module=orig_attn,
                        accelerator=accelerator,
                        layer_idx=l_idx,
                        stage_name=stage_name
                    )
                    hooks_injected += 1
                    print(f"  -> Wrapped [{stage_name}] Transformer Layer #{l_idx} with Multi-Stage PL Hook (Time-Multiplexed H=4).")
        except Exception as e:
            print(f"  [WARN] Failed to wrap layers in encoder layer[{enc_idx}] ({stage_name}): {e}")

    print(f"[HOOK] Total Hardware Attention Hooks Injected: {hooks_injected} / 9 layers.")
    return hooks_injected
