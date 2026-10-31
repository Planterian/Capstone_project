#!/usr/bin/env python3
"""
===============================================================================
Script:       test_client_laptop.py
Description:  Laptop-Side Host Automation Script to test AXI4 Interconnect
              and MobileViT Variant 2A Accelerator on AMD Kria KV260.
Features:
  - 100% Pure Standard Python (NO NumPy or external packages required!)
  - Built-in SSH Proxy Jump Support (-J ktmt@100.124.168.26)
  - Automatic Packaging of Q, K, V Hex Vectors into Binary
  - Automated Bit-Exact Comparison against PyTorch Golden References
===============================================================================
"""

import os
import sys
import argparse
import subprocess

def load_hex_bytes(filepath):
    """Parses 2-digit hex values from PyTorch golden .hex vector file into raw bytes."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing vector file: {filepath}")
    with open(filepath, "r") as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return vals

def to_signed8(val):
    """Converts uint8 (0..255) to signed int8 (-128..127)."""
    return val - 256 if val >= 128 else val

def main():
    parser = argparse.ArgumentParser(description="Laptop Host Test Client for KV260 MHA Accelerator")
    parser.add_argument("--ip", type=str, default="192.168.1.179", help="Kria KV260 IP address")
    parser.add_argument("--user", type=str, default="ubuntu", help="SSH username (default: ubuntu)")
    parser.add_argument("--jump", type=str, default="ktmt@100.124.168.26", help="SSH Jump host (e.g. ktmt@100.124.168.26)")
    parser.add_argument("--remote-dir", type=str, default="/home/ubuntu/mha_2a_test", help="Target remote folder on KV260")
    args = parser.parse_args()

    kv260_target = f"{args.user}@{args.ip}"

    # Build Jump Host flags if provided
    jump_opt = f"-J {args.jump} " if args.jump else ""

    print("==================================================================")
    print("   MHA VARIANT 2A LAPTOP-TO-KV260 SSH/AXI4 TEST RUNNER")
    print(f"   Target Board IP  : {args.ip} (User: {args.user})")
    if args.jump:
        print(f"   SSH Jump Proxy   : {args.jump}")
    print(f"   Remote Directory : {args.remote_dir}")
    print("==================================================================")

    # 1. Package Q, K, V Hex vectors into raw binary format
    print("\n[Step 1] Loading local Golden Vectors (Q, K, V)...")
    try:
        q_vals = load_hex_bytes("q_matrix.hex")
        k_vals = load_hex_bytes("k_matrix.hex")
        v_vals = load_hex_bytes("v_matrix.hex")
        gold_vals = load_hex_bytes("golden_context_out.hex")
    except Exception as e:
        print(f"[ERROR]: Could not load hex files: {e}")
        sys.exit(1)

    packed_bytes = bytearray(q_vals + k_vals + v_vals)
    with open("input_qkv.bin", "wb") as f:
        f.write(packed_bytes)

    print(f"  -> Successfully packed raw QKV tensor: {len(packed_bytes)} bytes (N=256, dk=16, 3 matrices)")

    # 2. Check SSH connectivity via Jump Host
    print(f"\n[Step 2] Testing SSH connection to {kv260_target}...")
    test_ssh_cmd = f"ssh {jump_opt}-o ConnectTimeout=8 {kv260_target} 'echo [KV260]: SSH Connection OK!'"
    print(f"  -> Executing: {test_ssh_cmd}")
    ret = os.system(test_ssh_cmd)
    if ret != 0:
        print(f"\n[ERROR]: Cannot connect to KV260 at {args.ip} via Jump Host {args.jump}.")
        print("Please verify your network, VPN/Tailscale connection, and SSH credentials.")
        sys.exit(1)

    # 3. Create remote directory and sync Overlay files, Driver, and Data
    print(f"\n[Step 3] Syncing Overlay Bitstream, Driver, and Data to KV260 via SCP...")
    os.system(f"ssh {jump_opt}{kv260_target} 'mkdir -p {args.remote_dir}'")

    bit_file = "attention_core_top_2A.bit"
    hwh_file = "attention_core_top_2A.hwh"
    if not os.path.exists(bit_file) or not os.path.exists(hwh_file):
        print(f"[ERROR]: {bit_file} or {hwh_file} not found in current directory.")
        sys.exit(1)

    files_to_send = f"input_qkv.bin kv260_axi4_test_driver.py {bit_file} {hwh_file}"
    scp_cmd = f"scp {jump_opt}{files_to_send} {kv260_target}:{args.remote_dir}/"
    print(f"  -> Executing: {scp_cmd}")
    ret = os.system(scp_cmd)
    if ret != 0:
        print("[ERROR]: SCP file transfer failed!")
        sys.exit(1)
    print("  -> Transferred Overlay (.bit, .hwh), Driver, and 'input_qkv.bin' successfully.")

    # 4. Trigger remote hardware test on KV260 via SSH
    print(f"\n[Step 4] Triggering AXI4 Acceleration on KV260 via SSH...")
    run_cmd = f"ssh {jump_opt}{kv260_target} 'cd {args.remote_dir} && python3 kv260_axi4_test_driver.py'"
    print(f"  -> Executing: {run_cmd}\n")
    ret = os.system(run_cmd)
    if ret != 0:
        print("[ERROR]: Execution on KV260 encountered an error.")
        sys.exit(1)

    # 5. Fetch output from KV260
    print(f"\n[Step 5] Retrieving hardware execution result from KV260...")
    fetch_cmd = f"scp {jump_opt}{kv260_target}:{args.remote_dir}/output_context.bin ./"
    ret = os.system(fetch_cmd)
    if ret != 0:
        print("[ERROR]: Failed to fetch 'output_context.bin' from KV260!")
        sys.exit(1)

    # 6. Verify against Golden Reference
    print("\n[Step 6] Verifying Hardware Output against PyTorch Golden Model...")
    with open("output_context.bin", "rb") as f:
        hw_raw = list(f.read())

    if len(hw_raw) != len(gold_vals):
        print(f"[FAIL]: Output size mismatch! Expected {len(gold_vals)} bytes, got {len(hw_raw)} bytes.")
        sys.exit(1)

    gold_signed = [to_signed8(x) for x in gold_vals]
    hw_signed   = [to_signed8(x) for x in hw_raw]

    mismatches = sum(1 for h, g in zip(hw_signed, gold_signed) if h != g)
    total_samples = len(gold_signed)
    match_rate = ((total_samples - mismatches) / total_samples) * 100.0

    print("==================================================================")
    print("              HARDWARE AXI4 VERIFICATION REPORT                   ")
    print("==================================================================")
    print(f"  Target Device           : AMD Kria KV260 (xck26-sfvc784-2LV-c)")
    print(f"  Total Output Samples    : {total_samples}")
    print(f"  Exact Bit-Matched       : {total_samples - mismatches}")
    print(f"  Mismatched Elements     : {mismatches}")
    print(f"  Bit-Accuracy Rate       : {match_rate:.2f}%")
    print("------------------------------------------------------------------")

    if mismatches == 0:
        print("  >>> VERIFICATION STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<")
    else:
        print("  >>> VERIFICATION STATUS: MISMATCH DETECTED! [FAIL] <<<")
        print("  First 10 Mismatched Samples Comparison:")
        count = 0
        for i in range(total_samples):
            if hw_signed[i] != gold_signed[i]:
                print(f"    Index {i:04d}: HW = {hw_signed[i]:4d} | Golden = {gold_signed[i]:4d}")
                count += 1
                if count >= 10:
                    break
    print("==================================================================\n")

if __name__ == "__main__":
    main()
