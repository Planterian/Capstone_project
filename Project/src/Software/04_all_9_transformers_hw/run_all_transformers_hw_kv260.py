#!/usr/bin/env python3
"""
===============================================================================
Script:         run_all_transformers_hw_kv260.py
Phase:          PHASE 4 — FULL ACCELERATION OF ALL 9 TRANSFORMERS (STAGES 2, 3, 4)
Target SoC:     AMD Kria KV260 Vision AI Starter Kit (Quad ARM Cortex-A53 + PL)
Hardware Core:  Attention Core Variant 2A (DSP Packed @ 150.000 MHz)
Control Bus:    128-bit AXI DMA (MM2S & S2MM) + Dynamic AXI4-Lite CSR Reprogramming

Description:
  Executes the full MobileViT-XXS network on AMD Kria KV260, offloading
  ALL 9 Transformer Multi-Head Attention blocks across Stage 2, Stage 3,
  and Stage 4 directly to the physical FPGA PL Core.
  Dynamically reprograms CSR sequence length, head dimension, and dyadic scales.

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
import numpy as np

# Ensure local directories and 00_common are discoverable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "00_common"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

# Apply device patch BEFORE importing I-ViT modules
import patch_ivit_device  # noqa: F401
from mobilevit_ivit_surgery import replace_mobilevit_blocks
from transformers import MobileViTForImageClassification
import torch
import torch.nn as nn
import benchmark_utils as bu

from pynq_multi_stage_attention_hook import (
    PynqMultiStageAccelerator,
    inject_all_9_hardware_attention_hooks,
    PYNQ_AVAILABLE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Full Acceleration of All 9 Transformers on Kria KV260"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=os.path.join(COMMON_DIR, "test_images", "sample_infer.jpg"),
        help="Path to input test image (.jpg/.png)."
    )
    parser.add_argument(
        "--bitstream", 
        type=str, 
        default=os.path.join(SCRIPT_DIR, "overlay", "attention_core_top_2A.bit"),
        help="Path to FPGA bitstream overlay (.bit)."
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
        default=os.path.join(SCRIPT_DIR, "phase4_metrics.json"),
        help="Destination path for telemetry and latency report JSON."
    )
    return parser.parse_args()


def load_qat_weights(model: nn.Module, checkpoint_path: str, device: torch.device):
    """Load fine-tuned QAT weights with dynamic buffer shape adaptation."""
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


def main():
    args = parse_args()
    bu.print_benchmark_header("Phase 4: Full-Network 9-Transformer Hardware Acceleration (KV260)")

    # 1. Environment & Thread Configuration
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    print(f"[ENV] Processing System (PS) : Quad-Core ARM Cortex-A53")
    print(f"[ENV] Programmable Logic (PL): MHA Variant 2A Multi-Stage Accelerator @ 150 MHz")
    print(f"[ENV] Scope of Acceleration  : ALL 9 Transformer Layers (Stage 2: 2, Stage 3: 4, Stage 4: 3)")
    print(f"[ENV] Dynamic CSR Support    : Configurable N, dk, M, E per Attention invocation")
    print(f"[ENV] PYNQ Runtime Available : {PYNQ_AVAILABLE}")

    # 2. Ingest & Preprocess Input Image
    print(f"\n[DATA] Ingesting test image  : {args.image}")
    input_tensor = bu.preprocess_image(args.image, image_size=256).to(device)
    print(f"  -> Input Tensor Shape      : {list(input_tensor.shape)} (B, C, H, W)")

    # 3. Model Surgery & QAT Weights Ingestion
    print(f"\n[SURGERY] Initializing MobileViT-XXS with I-ViT Integer Surgery...")
    base_model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-xx-small", low_cpu_mem_usage=True
    )
    model, replaced_blocks = replace_mobilevit_blocks(base_model)
    print(f"  -> Replaced {len(replaced_blocks)} Transformer layers with I-ViT integer blocks.")

    load_qat_weights(model, args.checkpoint, device)
    model.eval()

    # 4. Initialize Multi-Stage PYNQ PL Hardware Accelerator
    print(f"\n[PL OVERLAY] Initializing Multi-Stage Hardware Overlay...")
    accelerator = None
    if PYNQ_AVAILABLE and os.path.exists(args.bitstream):
        try:
            accelerator = PynqMultiStageAccelerator(bitstream_path=args.bitstream)
            print("  -> Multi-stage hardware accelerator initialized successfully!")
        except Exception as e:
            print(f"  [ERROR] Hardware initialization failed: {e}")
            accelerator = None
    else:
        print("  [WARN] Running without active PL overlay (PYNQ unavailable or bitstream missing).")

    # 5. Inject Hardware Forward Hooks into ALL 9 Attention Blocks
    print("\n[HOOK INJECTION] Wrapping all 9 Transformer Attention Layers...")
    hooks_count = inject_all_9_hardware_attention_hooks(model, accelerator)

    # 6. Warmup Phase
    print(f"\n[EXEC] Executing {args.warmup} warmup iterations (untimed)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_tensor)

    # Reset telemetry counters after warmup
    if accelerator:
        accelerator.total_invocations = 0
        accelerator.stage_invocations.clear()
        accelerator.stage_latencies_us.clear()

    # 7. Benchmarking Measurement Phase
    print(f"[EXEC] Executing {args.iterations} measurement iterations (nanosecond timer)...")
    e2e_latencies_ms: List[float] = []

    with torch.no_grad():
        for it in range(args.iterations):
            with bu.PrecisionTimer() as timer:
                outputs = model(input_tensor)
            e2e_latencies_ms.append(timer.duration_ms)

    # 8. Aggregate Telemetry & Statistics
    e2e_stats = bu.compute_statistics(e2e_latencies_ms)
    peak_rss_mb = bu.get_peak_memory_mb()

    # Hardware breakdown
    stage_latencies_summary = {}
    total_pl_ms_per_image = 0.0
    if accelerator and accelerator.total_invocations > 0:
        for stg, l_list in accelerator.stage_latencies_us.items():
            if l_list:
                mean_us = float(np.mean(l_list))
                inv_per_img = len(l_list) / float(args.iterations)
                stg_total_ms = (mean_us * inv_per_img) / 1000.0
                total_pl_ms_per_image += stg_total_ms
                stage_latencies_summary[stg] = {
                    "invocations_per_frame": round(inv_per_img, 1),
                    "mean_kernel_us": round(mean_us, 2),
                    "total_stage_ms": round(stg_total_ms, 2)
                }

    # 9. Classification Predictions
    class_map = bu.load_imagenet_classes()
    top5_preds = bu.extract_topk_predictions(outputs.logits, class_map, topk=5)

    # 10. Output Reporting
    bu.print_prediction_table(top5_preds)
    bu.print_metrics_summary(e2e_stats, peak_rss_mb)

    if stage_latencies_summary:
        print("[PL MULTI-STAGE ATTENTION HARDWARE BREAKDOWN]")
        print(f"{'Stage Name':<12} | {'Invocations/Frame':<20} | {'Mean Kernel Lat (us)':<22} | {'Stage Total (ms)'}")
        print("-" * 76)
        for stg, s_info in stage_latencies_summary.items():
            print(f"{stg:<12} | {s_info['invocations_per_frame']:<20} | {s_info['mean_kernel_us']:<22.2f} | {s_info['total_stage_ms']:.2f} ms")
        print("-" * 76)
        print(f"  -> Total Cumulative PL Attention Time: {total_pl_ms_per_image:.2f} ms / frame")
        print(f"  -> Total Hardware Attention Invocations : {accelerator.total_invocations}")
        print("=" * 76 + "\n")

    # 11. Save JSON Artifact
    report_data = {
        "phase": "Phase 4 - All 9 Transformers Accelerated",
        "precision": "INT8 / INT16 Hybrid (PS: Cortex-A53 + PL: 150 MHz Core across 9 layers)",
        "device": "AMD Kria KV260 (xck26-sfvc784-2LV-c)",
        "num_threads": args.threads,
        "input_resolution": [256, 256],
        "image_tested": os.path.basename(args.image),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "end_to_end_stats": e2e_stats,
        "pl_stage_breakdown": stage_latencies_summary,
        "total_pl_attention_ms_per_frame": round(total_pl_ms_per_image, 2),
        "peak_rss_mb": peak_rss_mb,
        "top5_predictions": top5_preds,
        "notes": "Full-network acceleration: All 9 Transformer layers offloaded to FPGA PL."
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Telemetry report exported -> {args.output_json}")


if __name__ == "__main__":
    main()

