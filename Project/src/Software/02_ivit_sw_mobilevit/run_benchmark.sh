#!/bin/bash
# =============================================================================
# Automated Benchmark Runner: Phase 2 — I-ViT Quantized Software Model (CPU)
# Target Platform: AMD Kria KV260 (AArch64 Ubuntu 22.04)
# =============================================================================
set -e

cd "$(dirname "$0")"

# 1. Lock ARM Cortex-A53 CPU to Maximum Nominal Frequency (Performance Governor)
echo "[INFO] Configuring CPU scaling governor to 'performance'..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true

# 2. Source PYNQ Virtual Environment (if installed)
if [ -f /usr/local/share/pynq-venv/bin/activate ]; then
    source /usr/local/share/pynq-venv/bin/activate
fi

# 3. Target Image Path
IMAGE_PATH="${1:-../00_common/test_images/sample_infer.jpg}"

# 4. Execute Benchmark with 4 CPU Threads pinned across all 4 Cortex-A53 cores
echo "[INFO] Running Phase 2 I-ViT Quantized SW Benchmark on: $IMAGE_PATH..."
OMP_NUM_THREADS=4 taskset -c 0-3 python3 run_ivit_sw_kv260.py \
    --image "$IMAGE_PATH" \
    --checkpoint weights/mobilevit_ivit_qat_best.pth \
    --iterations 30 \
    --warmup 10 \
    --threads 4 \
    --output-json phase2_metrics.json

echo "[DONE] Phase 2 benchmark complete. Results stored in phase2_metrics.json."

