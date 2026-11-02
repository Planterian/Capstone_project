#!/bin/bash
# =============================================================================
# Automated Benchmark Runner: Phase 1 — Baseline MobileViT-XXS (FP32 Software)
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
echo "[INFO] Running Phase 1 Baseline Benchmark on image: $IMAGE_PATH..."
OMP_NUM_THREADS=4 taskset -c 0-3 python3 run_baseline_kv260.py \
    --image "$IMAGE_PATH" \
    --iterations 30 \
    --warmup 10 \
    --threads 4 \
    --output-json phase1_metrics.json

echo "[DONE] Phase 1 benchmark complete. Results stored in phase1_metrics.json."

