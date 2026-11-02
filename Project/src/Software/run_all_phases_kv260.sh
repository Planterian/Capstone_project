#!/bin/bash
# =============================================================================
# MASTER BENCHMARK RUNNER: 4-PHASE COMPARATIVE EVALUATION ON STANDALONE KV260
# Target Platform: AMD Kria KV260 Vision AI Starter Kit
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_PATH="${1:-$SCRIPT_DIR/00_common/test_images/sample_infer.jpg}"

echo "================================================================================"
echo "    AMD KRIA KV260: END-TO-END MOBILEVIT 4-PHASE EVALUATION HARNESS"
echo "================================================================================"
echo "[INFO] Test Image Target: $IMAGE_PATH"
echo "[INFO] Timestamp        : $(date)"
echo "--------------------------------------------------------------------------------"

# -----------------------------------------------------------------------------
# 1. Execute Phase 1: Pure FP32 Software Baseline
# -----------------------------------------------------------------------------
echo ""
echo ">>> [1/4] EXECUTING PHASE 1: FP32 SOFTWARE BASELINE (HUGGING FACE) <<<"
cd "$SCRIPT_DIR/01_baseline_mobilevit"
./run_benchmark.sh "$IMAGE_PATH"

# -----------------------------------------------------------------------------
# 2. Execute Phase 2: Integer-Quantized I-ViT Software (CPU)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [2/4] EXECUTING PHASE 2: I-ViT QUANTIZED SOFTWARE (CPU EMULATION) <<<"
cd "$SCRIPT_DIR/02_ivit_sw_mobilevit"
./run_benchmark.sh "$IMAGE_PATH"

# -----------------------------------------------------------------------------
# 3. Execute Phase 3: Partitioned HW/SW Co-Design (Stage 2 PL Accelerated)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [3/4] EXECUTING PHASE 3: PARTITIONED HW/SW CO-DESIGN (STAGE 2 PL) <<<"
cd "$SCRIPT_DIR/03_ivit_hw_accelerated"
./run_benchmark.sh "$IMAGE_PATH"

# -----------------------------------------------------------------------------
# 4. Execute Phase 4: Full-Network Acceleration (All 9 Transformers on PL)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [4/4] EXECUTING PHASE 4: FULL-NETWORK ACCELERATION (ALL 9 TRANSFORMERS ON PL) <<<"
cd "$SCRIPT_DIR/04_all_9_transformers_hw"
./run_benchmark.sh "$IMAGE_PATH"

# -----------------------------------------------------------------------------
# 5. Generate Comparative Metrics Summary Table
# -----------------------------------------------------------------------------
cd "$SCRIPT_DIR"
python3 -c "
import json
import os

phases = [
    ('Phase 1 (FP32 Baseline)',     '01_baseline_mobilevit/phase1_metrics.json'),
    ('Phase 2 (I-ViT SW Only)',     '02_ivit_sw_mobilevit/phase2_metrics.json'),
    ('Phase 3 (Stage 2 HW)',        '03_ivit_hw_accelerated/phase3_metrics.json'),
    ('Phase 4 (All 9 Transformers)','04_all_9_transformers_hw/phase4_metrics.json')
]

data = []
for p, fpath in phases:
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            d = json.load(f)
            data.append((p, d))

if not data:
    print('[WARN] No metrics JSON files found.')
    exit(0)

base_lat = data[0][1]['end_to_end_stats'].get('mean_ms', 1.0)

print('\n' + '=' * 115)
print('                        STANDALONE KRIA KV260 4-PHASE COMPARATIVE BENCHMARK REPORT')
print('=' * 115)
header = f'| {\"Evaluation Phase\":<27} | {\"Precision\":<14} | {\"Mean Lat (ms)\":<13} | {\"Throughput\":<11} | {\"Speedup\":<9} | {\"Peak RAM\":<10} | {\"Top-1 Prediction\":<16} |'
print(header)
print('|' + '-' * 29 + '|' + '-' * 16 + '|' + '-' * 15 + '|' + '-' * 13 + '|' + '-' * 11 + '|' + '-' * 12 + '|' + '-' * 20 + '|')

rows = []
for phase_name, d in data:
    prec = d.get('precision', 'N/A')
    stats = d.get('end_to_end_stats', {})
    mean_lat = stats.get('mean_ms', 0.0)
    fps = stats.get('throughput_fps', 0.0)
    speedup = f'{base_lat / mean_lat:.2f}x' if mean_lat > 0 else 'N/A'
    ram = f'{d.get(\"peak_rss_mb\", 0.0):.1f} MB'
    preds = d.get('top5_predictions', [])
    top1_str = f'ID {preds[0][\"class_id\"]} ({preds[0][\"confidence_percent\"]})' if preds else 'N/A'
    row_str = f'| {phase_name:<27} | {prec:<14} | {mean_lat:<13.2f} | {fps:<11.1f} | {speedup:<9} | {ram:<10} | {top1_str:<18} |'
    print(row_str)
    rows.append({
        'phase': phase_name,
        'precision': prec,
        'mean_lat_ms': mean_lat,
        'throughput_fps': fps,
        'speedup': speedup,
        'peak_ram_mb': ram,
        'top1_pred': top1_str
    })
print('=' * 115 + '\n')

# Export to Markdown summary
with open('BENCHMARK_SUMMARY.md', 'w') as mf:
    mf.write('# Standalone AMD Kria KV260 4-Phase Comparative Benchmark Report\n\n')
    mf.write('| Evaluation Phase | Precision | Mean Latency (ms) | Throughput (FPS) | Speedup vs Baseline | Peak RAM (MB) | Top-1 Prediction |\n')
    mf.write('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n')
    for r in rows:
        mf.write(f'| **{r[\"phase\"]}** | {r[\"precision\"]} | {r[\"mean_lat_ms\"]:.2f} | {r[\"throughput_fps\"]:.1f} | {r[\"speedup\"]} | {r[\"peak_ram_mb\"]} | {r[\"top1_pred\"]} |\n')

print('[SUCCESS] Consolidated summary exported to BENCHMARK_SUMMARY.md.')
"

echo "[COMPLETE] 4-Phase Master evaluation pipeline finished successfully!"
