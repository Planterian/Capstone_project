#!/usr/bin/env python3
"""
===============================================================================
Script:         run_ivit_sw_kv260.py
Phase:          PHASE 2 — I-ViT QUANTIZED SOFTWARE MODEL (ARM CPU ONLY)
Target SoC:     AMD Kria KV260 Vision AI Starter Kit (Quad ARM Cortex-A53 PS)
Description:
  Executes the Integer-Only Quantized MobileViT-XXS model (I-ViT architecture
  surgery with fine-tuned QAT weights) in pure CPU software on Kria KV260.
  Uses patch_ivit_device to bypass hardcoded .cuda() calls.
  Evaluates the latency overhead of CPU-simulated integer math vs Phase 1 FP32,
  and establishes the direct predecessor to Phase 3 PL hardware offload.

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

# Ensure local directories and 00_common are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "00_common"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

# Apply device patch BEFORE importing I-ViT modules to eliminate .cuda() calls
import patch_ivit_device  # noqa: F401
from mobilevit_ivit_surgery import replace_mobilevit_blocks
from transformers import MobileViTForImageClassification
import torch
import torch.nn as nn
import benchmark_utils as bu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: I-ViT Quantized Software MobileViT-XXS Evaluation on KV260"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=os.path.join(COMMON_DIR, "test_images", "sample_infer.jpg"),
        help="Path to input test image (.jpg/.png)."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=os.path.join(SCRIPT_DIR, "weights", "mobilevit_ivit_qat_best.pth"),
        help="Path to fine-tuned QAT weights (mobilevit_ivit_qat_best.pth)."
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
        help="Number of untimed warmup iterations (default: 10)."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of OpenMP/PyTorch CPU worker threads (default: 4 for Cortex-A53)."
    )
    parser.add_argument(
        "--output-json", 
        type=str, 
        default=os.path.join(SCRIPT_DIR, "phase2_metrics.json"),
        help="Destination path for telemetry and latency report JSON."
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Robust Weight Loader with Dynamic Buffer Adaptation
# -----------------------------------------------------------------------------
def load_qat_weights(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load fine-tuned QAT weights with buffer shape adaptation for activation scales.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR]: QAT checkpoint not found at: '{checkpoint_path}'")

    print(f"[LOAD] Loading QAT checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model_state = model.state_dict()
    modules_dict = dict(model.named_modules())

    adapted_buffers = 0
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            tokens = k.split(".")
            mod_name = ".".join(tokens[:-1])
            buf_name = tokens[-1]
            if mod_name in modules_dict:
                setattr(modules_dict[mod_name], buf_name, v.clone())
                adapted_buffers += 1

    model.load_state_dict(state_dict)
    model.to(device)
    print(f"  -> Successfully loaded {len(state_dict)} state keys ({adapted_buffers} calibrated buffers adapted).")


# -----------------------------------------------------------------------------
# Timing Hook for Stage 2 Transformer Attention Layer
# -----------------------------------------------------------------------------
class AttentionTimingHook:
    def __init__(self):
        self.stage2_latencies_ms: List[float] = []
        self._start_ns = 0

    def pre_hook(self, module, inputs):
        self._start_ns = time.perf_counter_ns()

    def post_hook(self, module, inputs, outputs):
        elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self.stage2_latencies_ms.append(elapsed_ms)


def main():
    args = parse_args()
    bu.print_benchmark_header("Phase 2: I-ViT Quantized Software Model Evaluation (KV260 CPU)")

    # 1. Environment & Thread Configuration
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    print(f"[ENV] Host CPU Target       : ARM Cortex-A53 (Quad-Core)")
    print(f"[ENV] Active CPU Threads    : {torch.get_num_threads()}")
    print(f"[ENV] Arithmetic Engine     : Integer-Only Emulation (I-ViT Shiftmax / ShiftGELU)")

    # 2. Ingest & Preprocess Input Image
    print(f"\n[DATA] Ingesting test image : {args.image}")
    input_tensor = bu.preprocess_image(args.image, image_size=256).to(device)
    print(f"  -> Input Tensor Shape     : {list(input_tensor.shape)} (B, C, H, W)")

    # 3. Model Surgery & QAT Weights Ingestion
    print(f"\n[SURGERY] Applying I-ViT Integer Surgery to MobileViT-XXS...")
    base_model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-xx-small", low_cpu_mem_usage=True
    )
    model, replaced_blocks = replace_mobilevit_blocks(base_model)
    print(f"  -> Replaced {len(replaced_blocks)} Transformer layers with I-ViT integer blocks.")

    load_qat_weights(model, args.checkpoint, device)
    model.eval()

    # 4. Attach Telemetry Hook to Stage 2 Attention Layer
    hook = AttentionTimingHook()
    try:
        # In surgered MobileViT, layer[2] corresponds to MobileViT Block 1 (Stage 2)
        stage2_attn = model.mobilevit.encoder.layer[2].transformer.layer[0].attn
        stage2_attn.register_forward_pre_hook(hook.pre_hook)
        stage2_attn.register_forward_hook(hook.post_hook)
        print("  -> Telemetry hook attached to Stage 2 I-ViT Attention block.")
    except Exception as e:
        print(f"  [WARN] Could not attach hook to Stage 2 attention: {e}")

    # 5. Warmup Phase
    print(f"\n[EXEC] Executing {args.warmup} warmup iterations (untimed)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_tensor)
    hook.stage2_latencies_ms.clear()

    # 6. Benchmarking Measurement Phase
    print(f"[EXEC] Executing {args.iterations} measurement iterations (nanosecond timer)...")
    e2e_latencies_ms: List[float] = []

    with torch.no_grad():
        for it in range(args.iterations):
            with bu.PrecisionTimer() as timer:
                outputs = model(input_tensor)
            e2e_latencies_ms.append(timer.duration_ms)

    # 7. Aggregate Telemetry & Statistics
    e2e_stats = bu.compute_statistics(e2e_latencies_ms)
    stage2_stats = bu.compute_statistics(hook.stage2_latencies_ms) if hook.stage2_latencies_ms else {}
    peak_rss_mb = bu.get_peak_memory_mb()

    # 8. Classification Predictions
    class_map = bu.load_imagenet_classes()
    top5_preds = bu.extract_topk_predictions(outputs.logits, class_map, topk=5)

    # 9. Output Reporting
    bu.print_prediction_table(top5_preds)
    bu.print_metrics_summary(e2e_stats, peak_rss_mb)

    if stage2_stats:
        print("[STAGE 2 I-ViT ATTENTION KERNEL PROFILE (CPU EMULATION)]")
        print(f"  - Mean Attention Latency  : {stage2_stats.get('mean_ms', 0.0):.2f} ms")
        print(f"  - Fraction of Total Time  : {(stage2_stats.get('mean_ms', 0.0) / e2e_stats.get('mean_ms', 1.0)) * 100.0:.1f} %")
        print("=" * 76 + "\n")

    # 10. Save JSON Artifact
    report_data = {
        "phase": "Phase 2 - I-ViT Quantized Software",
        "precision": "INT8 / INT16 / INT32 (Integer-Only)",
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
        "notes": "Quantized MobileViT-XXS executing via CPU integer emulation."
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Telemetry report exported -> {args.output_json}")


if __name__ == "__main__":
    main()

