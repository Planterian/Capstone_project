#!/usr/bin/env python3
"""
===============================================================================
Script:         run_baseline_kv260.py
Phase:          PHASE 1 — BASELINE MOBILEVIT-XXS (FP32 PURE SOFTWARE)
Target SoC:     AMD Kria KV260 Vision AI Starter Kit (Quad ARM Cortex-A53 PS)
Description:
  Executes the unmodified pre-trained Apple MobileViT-XXS model (FP32) from
  Hugging Face Hub in pure CPU software on the Kria KV260.
  Measures ground-truth accuracy, end-to-end latency, and Stage 2 Attention
  latency to establish the scientific baseline for subsequent phases.

Author:         Lead Embedded HW/SW Co-Design Engineer
Project:        MobileViT-XXS + I-ViT Co-Design on AMD Kria KV260
===============================================================================
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List

# Ensure 00_common utilities are discoverable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "00_common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import torch
from transformers import MobileViTForImageClassification, MobileViTConfig
import benchmark_utils as bu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Hugging Face MobileViT-XXS FP32 Software Baseline on KV260"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=os.path.join(COMMON_DIR, "test_images", "sample_infer.jpg"),
        help="Path to input test image (.jpg/.png) from ImageNet/COCO."
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="apple/mobilevit-xx-small",
        help="Hugging Face model repository ID or local saved checkpoint directory."
    )
    parser.add_argument(
        "--local-weights",
        type=str,
        default=os.path.join(SCRIPT_DIR, "weights", "mobilevit_xxs_fp32.pt"),
        help="Path to locally cached FP32 model checkpoint (for offline KV260 operation)."
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=30,
        help="Number of timed benchmark measurement iterations (default: 30)."
    )
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=10,
        help="Number of untimed warmup iterations to stabilize caches & CPU governor (default: 10)."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of OpenMP/PyTorch CPU worker threads (default: 4 for Cortex-A53 quad-core)."
    )
    parser.add_argument(
        "--output-json", 
        type=str, 
        default=os.path.join(SCRIPT_DIR, "phase1_metrics.json"),
        help="Destination path for telemetry and latency report JSON."
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Timing Hook for Stage 2 Transformer Attention Layer
# -----------------------------------------------------------------------------
class AttentionTimingHook:
    """
    Forward hook that measures the duration of Stage 2 Transformer Attention specifically.
    """
    def __init__(self):
        self.stage2_latencies_ms: List[float] = []
        self._start_ns = 0

    def pre_hook(self, module, inputs):
        self._start_ns = time.perf_counter_ns()

    def post_hook(self, module, inputs, outputs):
        elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self.stage2_latencies_ms.append(elapsed_ms)


def load_baseline_model(model_id: str, local_weights_path: str, device: torch.device) -> MobileViTForImageClassification:
    """
    Load the MobileViT model either from local saved weights or Hugging Face repository.
    """
    print(f"[INFO] Initializing MobileViT-XXS (FP32)...")
    if os.path.exists(local_weights_path):
        print(f"  -> Loading pre-cached weights from: {local_weights_path}")
        model = torch.load(local_weights_path, map_location=device)
    else:
        print(f"  -> Loading weights from Hugging Face: '{model_id}'...")
        model = MobileViTForImageClassification.from_pretrained(model_id)
        # Cache locally for offline execution on KV260
        os.makedirs(os.path.dirname(local_weights_path), exist_ok=True)
        try:
            torch.save(model, local_weights_path)
            print(f"  -> Successfully cached model locally to: {local_weights_path}")
        except Exception as e:
            print(f"  [WARN] Could not write cache file: {e}")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    bu.print_benchmark_header("Phase 1: Baseline MobileViT-XXS FP32 Software Evaluation (KV260)")

    # 1. Environment & Thread Configuration
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    print(f"[ENV] Host CPU Target       : ARM Cortex-A53 (Quad-Core)")
    print(f"[ENV] PyTorch Version       : {torch.__version__}")
    print(f"[ENV] Active CPU Threads    : {torch.get_num_threads()}")
    print(f"[ENV] Precision Engine      : IEEE 754 Single-Precision (FP32)")

    # 2. Ingest & Preprocess Input Image
    print(f"\n[DATA] Ingesting test image : {args.image}")
    input_tensor = bu.preprocess_image(args.image, image_size=256).to(device)
    print(f"  -> Input Tensor Shape     : {list(input_tensor.shape)} (B, C, H, W)")

    # 3. Model Instantiation & Profiling Hook Attachment
    model = load_baseline_model(args.model_id, args.local_weights, device)
    
    # Attach timing hook to Stage 2 Attention Layer
    # In HuggingFace MobileViT:
    # model.mobilevit.encoder.layer[1] represents Stage 2 (N=256, dk=16)
    hook = AttentionTimingHook()
    try:
        stage2_block = model.mobilevit.encoder.layer[1]
        stage2_block.register_forward_pre_hook(hook.pre_hook)
        stage2_block.register_forward_hook(hook.post_hook)
        print("  -> Successfully attached telemetry hook to Stage 2 Attention Block.")
    except Exception as e:
        print(f"  [WARN] Could not attach hook to Stage 2: {e}")

    # 4. Warmup Phase (Prime CPU caches and governors)
    print(f"\n[EXEC] Executing {args.warmup} warmup iterations (untimed)...")
    with torch.no_grad():
        for i in range(args.warmup):
            _ = model(input_tensor)
    hook.stage2_latencies_ms.clear()

    # 5. Benchmarking Measurement Phase
    print(f"[EXEC] Executing {args.iterations} measurement iterations (nanosecond timer)...")
    e2e_latencies_ms: List[float] = []

    with torch.no_grad():
        for it in range(args.iterations):
            with bu.PrecisionTimer() as timer:
                outputs = model(input_tensor)
            e2e_latencies_ms.append(timer.duration_ms)

    # 6. Aggregate Telemetry & Statistics
    e2e_stats = bu.compute_statistics(e2e_latencies_ms)
    stage2_stats = bu.compute_statistics(hook.stage2_latencies_ms) if hook.stage2_latencies_ms else {}
    peak_rss_mb = bu.get_peak_memory_mb()

    # 7. Classification Predictions
    class_map = bu.load_imagenet_classes()
    top5_preds = bu.extract_topk_predictions(outputs.logits, class_map, topk=5)

    # 8. Output Reporting
    bu.print_prediction_table(top5_preds)
    bu.print_metrics_summary(e2e_stats, peak_rss_mb)

    if stage2_stats:
        print("[STAGE 2 ATTENTION KERNEL PROFILE]")
        print(f"  - Mean Stage 2 Latency    : {stage2_stats.get('mean_ms', 0.0):.2f} ms")
        print(f"  - Fraction of Total Time  : {(stage2_stats.get('mean_ms', 0.0) / e2e_stats.get('mean_ms', 1.0)) * 100.0:.1f} %")
        print("=" * 76 + "\n")

    # 9. Save JSON Artifact
    report_data = {
        "phase": "Phase 1 - Baseline MobileViT-XXS",
        "precision": "FP32",
        "device": "ARM Cortex-A53 (4 Cores)",
        "num_threads": args.threads,
        "input_resolution": [256, 256],
        "image_tested": os.path.basename(args.image),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "end_to_end_stats": e2e_stats,
        "stage2_attention_stats": stage2_stats,
        "peak_rss_mb": peak_rss_mb,
        "top5_predictions": top5_preds,
        "notes": "Unmodified Apple MobileViT-XXS FP32 baseline."
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Telemetry report exported -> {args.output_json}")


if __name__ == "__main__":
    main()

