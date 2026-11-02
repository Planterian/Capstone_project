#!/bin/bash
# =============================================================================
# Automated Benchmark Runner: Phase 4 — All 9 Transformers Accelerated (KV260)
# Target Platform: AMD Kria KV260 (AArch64 Ubuntu 22.04 + PYNQ v3.0.1)
# =============================================================================
set -e

cd "$(dirname "$0")"

# 1. Unload default Kria SOM starter kit application to free the FPGA manager
echo "[INFO] Unloading active Kria SOM starter app..."
echo capstone2026 | sudo -S xmutil unloadapp 2>/dev/null || true

# 2. Lock ARM Cortex-A53 CPU to Maximum Nominal Frequency (Performance Governor)
echo "[INFO] Configuring CPU scaling governor to 'performance'..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true

# 3. Source XRT and PYNQ Runtime Environments
[ -f /etc/profile ] && source /etc/profile 2>/dev/null || true
for xrt_path in /opt/xilinx/xrt/setup.sh /tools/Xilinx/xrt/setup.sh; do
    [ -f "$xrt_path" ] && source "$xrt_path" 2>/dev/null || true
done
if [ -f /usr/local/share/pynq-venv/bin/activate ]; then
    source /usr/local/share/pynq-venv/bin/activate
fi

# 4. Target Image Path
IMAGE_PATH="${1:-../00_common/test_images/sample_infer.jpg}"

# 5. Execute with Sudo Privileges (Required for PYNQ CMA memory mmap & DMA access)
echo "[INFO] Running Phase 4 (9 Transformers Accelerated) on: $IMAGE_PATH..."
echo capstone2026 | sudo -S -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" XILINX_XRT="$XILINX_XRT" \
    python3 run_all_transformers_hw_kv260.py \
        --image "$IMAGE_PATH" \
        --bitstream overlay/attention_core_top_2A.bit \
        --checkpoint weights/mobilevit_ivit_qat_best.pth \
        --iterations 30 \
        --warmup 10 \
        --threads 4 \
        --output-json phase4_metrics.json

# 6. Ensure permissions on generated JSON artifact
sudo chmod 666 phase4_metrics.json 2>/dev/null || true
sudo chown ubuntu:ubuntu phase4_metrics.json 2>/dev/null || true

echo "[DONE] Phase 4 benchmark complete. Results stored in phase4_metrics.json."

