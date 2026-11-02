# Phase 2: I-ViT Quantized Software Model (CPU Only) on AMD Kria KV260

This folder contains the complete runtime harness for the **Integer-Only Quantized MobileViT-XXS** model running purely on the ARM Cortex-A53 CPU of the KV260.

## Quickstart on Standalone KV260

### Run with default sample image:
```bash
./run_benchmark.sh
```

### Run on a custom image:
```bash
python3 run_ivit_sw_kv260.py --image /home/ubuntu/test_images/my_image.jpg --iterations 50
```

## Architectural Characteristics
- **Network:** MobileViT-XXS + I-ViT Integer Surgery (Shiftmax, ShiftGELU, IntLayerNorm, QuantLinear)
- **Trained Checkpoint:** `weights/mobilevit_ivit_qat_best.pth` (17.7 MB, 67.28% Top-1 Accuracy)
- **Precision:** INT8 Weights, INT8 Activations, INT16/INT32 Accumulators
- **Compute Target:** Quad-core ARM Cortex-A53 @ 1.33 GHz
- **Output Artifacts:** `phase2_metrics.json`

