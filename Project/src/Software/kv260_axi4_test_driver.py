#!/usr/bin/env python3
"""
===============================================================================
Script:       kv260_axi4_test_driver.py
Description:  Hardware AXI4 Test Driver for Attention Core Top (Variant 2A)
              Targeting AMD Kria KV260 Vision AI Starter Kit via PYNQ Runtime.
Features:
  - 128-bit AXI4-Stream Ingress / Egress via AXI DMA (16 INT8 tokens/beat)
  - Full AXI4-Lite CSR Register Map Alignment with axi_lite_regs.sv
  - Automated Dyadic Scaling (M=27640, E=22) and Core Trigger
===============================================================================
"""

import os
import sys
import time
import numpy as np
from pynq import Overlay, allocate

# -----------------------------------------------------------------------------
# 1. AXI4-LITE CSR REGISTER MAP (Strictly aligned with axi_lite_regs.sv)
# -----------------------------------------------------------------------------
ADDR_CTRL      = 0x00  # Bit 0: START, Bit 1: CLEAR, Bit 2: RESET
ADDR_STATUS    = 0x04  # Core Status [busy, done, error, bank_sel, head_idx]
ADDR_NBYTES    = 0x08  # Total QKV packet size in bytes
ADDR_SEQ_LEN   = 0x10  # Sequence length N (default: 256)
ADDR_HEAD_DIM  = 0x14  # Head dimension dk (16, 20, 24, 32)
ADDR_NUM_HEADS = 0x18  # Number of heads H (1, 2, 4)
ADDR_SHIFT_VAL = 0x1C  # ASR shift value
ADDR_SCALE_M   = 0x20  # Dyadic Scale Mantissa M (16-bit signed, Stage 2: 27640)
ADDR_SCALE_E   = 0x24  # Dyadic Scale Exponent E (5-bit unsigned, Stage 2: 22)
ADDR_OUT_M     = 0x28  # Output Mantissa
ADDR_OUT_E     = 0x2C  # Output Exponent
ADDR_LUT_ADDR  = 0x40  # Softmax Exp BRAM LUT Address
ADDR_LUT_DATA  = 0x44  # Softmax Exp BRAM LUT Data

# Model Parameters for Stage 2 Default
SEQ_LEN   = 256
HEAD_DIM  = 16
NUM_HEADS = 1

# Derived Tensor Sizes
QKV_BYTES = SEQ_LEN * HEAD_DIM * 3        # 256 * 16 * 3 = 12,288 Bytes
OUT_BYTES = SEQ_LEN * HEAD_DIM * NUM_HEADS # 256 * 16 * 1 = 4,096 Bytes
BITSTREAM = "attention_core_top_2A.bit"

def main():
    print("==================================================================")
    print("  KRIA KV260 AXI4 HARDWARE TEST DRIVER: VARIANT 2A (DSP PACKING) ")
    print("==================================================================")

    if not os.path.exists(BITSTREAM):
        print(f"[ERROR]: Bitstream '{BITSTREAM}' not found in current directory!")
        print("Please ensure attention_core_top_2A.bit and .hwh are placed here.")
        sys.exit(1)

    print(f"[1/5] Loading Bitstream Overlay '{BITSTREAM}'...")
    ol = Overlay(BITSTREAM)
    
    # Locate IP Cores
    dma = getattr(ol, "axi_dma_0", None)
    if dma is None:
        raise RuntimeError("AXI DMA 'axi_dma_0' not found in Overlay!")

    # Locate CSR (try common block design naming conventions)
    csr = None
    for name in ["attention_core_top_2A_0", "attention_core_top_0", "axi_lite_regs_0"]:
        if hasattr(ol, name):
            csr_block = getattr(ol, name)
            csr = getattr(csr_block, "s_axi", csr_block)
            break

    if csr is None:
        print("[WARN] CSR specific IP alias not directly identified. Searching IP dict...")
        for k in ol.ip_dict.keys():
            if "attention" in k or "axi_lite" in k:
                csr = getattr(ol, k).s_axi
                print(f"  -> Found CSR IP block: {k}")
                break

    print("[INFO] Hardware Overlay loaded successfully!")

    # -------------------------------------------------------------------------
    # 2. CONTIGUOUS MEMORY ALLOCATION (CMA)
    # -------------------------------------------------------------------------
    print(f"[2/5] Allocating CMA DDR4 buffers: IN={QKV_BYTES} B, OUT={OUT_BYTES} B...")
    in_buffer  = allocate(shape=(QKV_BYTES,), dtype=np.int8)
    out_buffer = allocate(shape=(OUT_BYTES,), dtype=np.int8)

    # Load input data from binary file or numpy file
    if os.path.exists("input_qkv.bin"):
        print("  -> Loading raw binary 'input_qkv.bin' from Laptop...")
        raw_bytes = np.fromfile("input_qkv.bin", dtype=np.int8)
        np.copyto(in_buffer, raw_bytes[:QKV_BYTES])
    elif os.path.exists("input_qkv.npy"):
        print("  -> Loading 'input_qkv.npy' from Laptop transmission...")
        in_raw = np.load("input_qkv.npy")
        np.copyto(in_buffer, in_raw.flatten()[:QKV_BYTES])
    else:
        print("  -> [NOTICE] No input file found. Using reproducible random stimulus...")
        np.random.seed(42)
        dummy_qkv = np.random.randint(-32, 32, size=(QKV_BYTES,), dtype=np.int8)
        np.copyto(in_buffer, dummy_qkv)

    print(f"  - Ingress Buffer Physical Address : {hex(in_buffer.physical_address)}")
    print(f"  - Egress Buffer Physical Address  : {hex(out_buffer.physical_address)}")

    # -------------------------------------------------------------------------
    # 3. CONFIGURE AXI4-LITE CSR
    # -------------------------------------------------------------------------
    print("[3/5] Configuring AXI4-Lite CSR Registers...")
    csr.write(ADDR_NBYTES,    QKV_BYTES)
    csr.write(ADDR_SEQ_LEN,   SEQ_LEN)
    csr.write(ADDR_HEAD_DIM,  HEAD_DIM)
    csr.write(ADDR_NUM_HEADS, NUM_HEADS)
    csr.write(ADDR_SCALE_M,   27640)
    csr.write(ADDR_SCALE_E,   22)

    # Readback Verification
    rb_nbytes = csr.read(ADDR_NBYTES)
    rb_seq    = csr.read(ADDR_SEQ_LEN)
    rb_hdim   = csr.read(ADDR_HEAD_DIM)
    rb_m      = csr.read(ADDR_SCALE_M)
    rb_e      = csr.read(ADDR_SCALE_E)

    print(f"  - Verified CSR: NBYTES={rb_nbytes}, N={rb_seq}, dk={rb_hdim}, M={rb_m}, E={rb_e}")
    if rb_seq != SEQ_LEN or rb_hdim != HEAD_DIM or rb_m != 27640:
        print("[ERROR]: CSR Readback mismatch! AXI4-Lite bus communication failure.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. STREAM VIA AXI DMA (128-BIT NATIVE BUS)
    # -------------------------------------------------------------------------
    print("[4/5] Initiating DMA Stream & Firing Accelerator...")
    t0 = time.perf_counter()

    # Arm S2MM receiver
    dma.recvchannel.transfer(out_buffer)
    # Initiate MM2S transmitter
    dma.sendchannel.transfer(in_buffer)

    # Pulse START bit
    csr.write(ADDR_CTRL, 0x01)

    # Await completion
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    t1 = time.perf_counter()
    duration_us = (t1 - t0) * 1e6
    throughput_fps = 1e6 / duration_us

    print(f"[5/5] DMA Transfers Finished! Roundtrip PL Latency: {duration_us:.2f} us ({throughput_fps:.1f} FPS)")
    print(f"  - Output Samples Received : {len(out_buffer)} bytes")
    print(f"  - First 16 Output Tokens  : {out_buffer[:16].tolist()}")

    # Save output for SCP retrieval (both binary and npy)
    out_buffer.tofile("output_context.bin")
    np.save("output_context.npy", out_buffer)
    print("[SUCCESS] Output successfully written to 'output_context.bin' & 'output_context.npy'.")

    # Cleanup
    in_buffer.freebuffer()
    out_buffer.freebuffer()

if __name__ == "__main__":
    main()

