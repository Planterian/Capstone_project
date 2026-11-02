#!/bin/bash
# Runner script executed on Kria KV260 with root permissions & preserved XRT/PYNQ environment
cd "$(dirname "$0")"

# 1. Unload any active Kria SOM apps to free FPGA manager for PYNQ
echo capstone2026 | sudo -S xmutil unloadapp 2>/dev/null || true

# 2. Source all profile and XRT setups
[ -f /etc/profile ] && source /etc/profile 2>/dev/null
for f in /etc/profile.d/*.sh; do
    [ -f "$f" ] && source "$f" 2>/dev/null
done
for xrt_path in /opt/xilinx/xrt/setup.sh /tools/Xilinx/xrt/setup.sh; do
    [ -f "$xrt_path" ] && source "$xrt_path" 2>/dev/null
done
[ -f /usr/local/share/pynq-venv/bin/activate ] && source /usr/local/share/pynq-venv/bin/activate 2>/dev/null

# 3. Run with sudo -E (preserves XRT and virtualenv environment)
echo capstone2026 | sudo -S -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" XILINX_XRT="$XILINX_XRT" python3 kv260_axi4_test_driver.py

# 4. Ensure permissions on output context for retrieval
sudo chmod 666 output_context.bin 2>/dev/null || true
sudo chown ubuntu:ubuntu output_context.bin 2>/dev/null || true
