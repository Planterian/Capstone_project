# Phase 1: Baseline MobileViT-XXS (FP32 Software) on AMD Kria KV260

This folder contains the complete runtime harness for the **unmodified FP32 software baseline** of Apple's MobileViT-XXS.

## Quickstart on Standalone KV260

### Run with default sample image:
```bash
./run_benchmark.sh
```

### Run on a custom image from your dataset:
```bash
python3 run_baseline_kv260.py --image /home/ubuntu/test_images/my_image.jpg --iterations 50
```

## Architectural Characteristics
- **Network:** `apple/mobilevit-xx-small` (~1.27M parameters)
- **Precision:** FP32 (Single Precision IEEE 754)
- **Compute Target:** Quad-core ARM Cortex-A53 @ 1.33 GHz
- **Expected Latency:** ~180 – 240 ms / image (4 – 5.5 FPS)
- **Output Artifacts:** `phase1_metrics.json`

