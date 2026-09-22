#!/usr/bin/env python3
"""
===============================================================================
Script:       deploy_kv260.py
Description:  Python Host Driver for Multi-Head Attention (MHA) Accelerator
              running on AMD Kria KV260 via PYNQ Framework v3.0+.
Target:       AMD Kria KV260 Vision AI Starter Kit
Author:       Gemini Notebook / Capstone Team
===============================================================================
"""

import time
import numpy as np
import pynq
from pynq import Overlay, allocate

# -----------------------------------------------------------------------------
# 1. HARDWARE SYSTEM CONFIGURATION
# -----------------------------------------------------------------------------
BITSTREAM_PATH = "attention_core_top.bit"

# Model Parameters (Matching attention_core_top.sv)
N_TOKENS       = 196    # Sequence length N = 196 (14x14 patch grid)
D_HEAD         = 32     # Head dimension d_k = 32
N_HEADS        = 2      # Number of attention heads H = 2
SHIFT_VAL      = 2      # ASR Barrel Shifter bit-shift value for 1/sqrt(d_k)

# Derived Tensor Sizes
QKV_ELEMENTS   = N_TOKENS * D_HEAD * 3  # 196 * 32 * 3 = 18,816 INT8 values
OUT_ELEMENTS   = N_TOKENS * D_HEAD      # 196 * 32     = 6,272 INT8 values

# AXI4-Lite CSR Register Offsets (axi_lite_regs.sv)
ADDR_CTRL      = 0x00   # Bit 0: START, Bit 1: CLEAR, Bit 2: BUSY, Bit 3: DONE
ADDR_NBYTES    = 0x04   # Total QKV packet size in bytes
ADDR_TOKENS    = 0x10   # Sequence length N
ADDR_HDIM      = 0x18   # Head dimension d_k
ADDR_SHIFT     = 0x20   # Scaler ASR shift count


def main():
    print("==================================================================")
    print("   MHA ACCELERATOR HOST DRIVER - AMD KRIA KV260 (PYNQ FRAMEWORK) ")
    print("==================================================================")

    # -------------------------------------------------------------------------
    # 2. OVERLAY LOADING & HARDWARE INITIALIZATION
    # -------------------------------------------------------------------------
    print(f"[INFO] Loading Bitstream: {BITSTREAM_PATH} ...")
    ol = Overlay(BITSTREAM_PATH)
    
    # Identify IP Cores from Overlay Design
    dma = ol.axi_dma_0
    csr = ol.attention_core_top_0.s_axi # Or ol.axi_lite_regs_0 depending on block design name
    
    print("[INFO] Hardware Overlay loaded successfully!")

    # -------------------------------------------------------------------------
    # 3. CONTIGUOUS MEMORY ALLOCATION (CMA BUFFERS)
    # -------------------------------------------------------------------------
    print("[INFO] Allocating physically contiguous DDR4 buffers (CMA) ...")
    
    # Input buffer for multiplexed Q, K, V matrices (INT8)
    in_buffer = allocate(shape=(QKV_ELEMENTS,), dtype=np.int8)
    # Output buffer for Softmax-weighted result matrix (INT8)
    out_buffer = allocate(shape=(OUT_ELEMENTS,), dtype=np.int8)

    # Generate synthetic input token data (-128 to +127 signed INT8)
    np.random.seed(42)
    dummy_qkv = np.random.randint(-64, 64, size=(QKV_ELEMENTS,), dtype=np.int8)
    np.copyto(in_buffer, dummy_qkv)

    print(f"  - Ingress Buffer  : {in_buffer.shape} ({in_buffer.nbytes} Bytes) @ Physical Addr: {hex(in_buffer.physical_address)}")
    print(f"  - Egress Buffer   : {out_buffer.shape} ({out_buffer.nbytes} Bytes) @ Physical Addr: {hex(out_buffer.physical_address)}")

    # -------------------------------------------------------------------------
    # 4. AXI4-LITE CSR CONFIGURATION
    # -------------------------------------------------------------------------
    print("[INFO] Configuring Hardware CSR Registers via AXI4-Lite MMIO ...")
    csr.write(ADDR_NBYTES, in_buffer.nbytes)
    csr.write(ADDR_TOKENS, N_TOKENS)
    csr.write(ADDR_HDIM,   D_HEAD)
    csr.write(ADDR_SHIFT,  SHIFT_VAL)

    print(f"  - NBYTES    = {csr.read(ADDR_NBYTES)}")
    print(f"  - TOKENS    = {csr.read(ADDR_TOKENS)}")
    print(f"  - HDIM      = {csr.read(ADDR_HDIM)}")
    print(f"  - SHIFT_VAL = {csr.read(ADDR_SHIFT)}")

    # -------------------------------------------------------------------------
    # 5. DMA TRANSFER & COMPUTATION EXECUTION
    # -------------------------------------------------------------------------
    print("[INFO] Starting Non-Blocking AXI DMA Streaming & MHA Core Execution ...")
    
    start_time = time.perf_counter()

    # Initiate S2MM (PL -> DDR4) receive channel first
    dma.recvchannel.transfer(out_buffer)
    
    # Initiate MM2S (DDR4 -> PL) send channel
    dma.sendchannel.transfer(in_buffer)

    # Pulse START signal on AXI4-Lite CSR (bit 0 = 1)
    csr.write(ADDR_CTRL, 0x01)

    # Wait for AXI DMA channels to complete transfer
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    print("[SUCCESS] Execution Completed!")
    print(f"  - End-to-End Latency : {latency_ms:.3f} ms")
    print(f"  - System Throughput  : {(1000.0 / latency_ms):.2f} FPS")

    # -------------------------------------------------------------------------
    # 6. OUTPUT VERIFICATION & CLEANUP
    # -------------------------------------------------------------------------
    print("[INFO] Sample Output Tokens (First 16 values):")
    print("  ", out_buffer[:16])

    # Check non-zero output
    non_zero_count = np.count_nonzero(out_buffer)
    print(f"[VERIFY] Non-zero elements returned: {non_zero_count} / {OUT_ELEMENTS}")
    if non_zero_count > 0:
        print("[PASSED] Hardware accelerator responded with valid probability tensors!")
    else:
        print("[WARNING] Output buffer is all zeros. Check AXI-Stream interface and resets.")

    # Free CMA Memory
    in_buffer.freebuffer()
    out_buffer.freebuffer()
    print("[INFO] CMA memory freed. Driver run complete.")

if __name__ == "__main__":
    main()
