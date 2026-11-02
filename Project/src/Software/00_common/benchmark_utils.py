#!/usr/bin/env python3
"""
===============================================================================
Module:         benchmark_utils.py
Description:    Standardized Performance Benchmarking & Telemetry Utilities
                Targeting Embedded Evaluation on AMD Kria KV260 (AArch64).
Features:
  - Nanosecond-precision monotonic execution timing (time.perf_counter_ns)
  - Statistical aggregation (Mean, Median, Std, Min, Max, P95, Throughput/FPS)
  - Resident Set Size (Peak RSS) physical memory tracking via POSIX resource
  - ImageNet-1k preprocessing pipeline (256x256, bicubic, standard normalization)
  - Human-readable Top-1 and Top-5 class extraction & report formatting
===============================================================================
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

try:
    import resource
except ImportError:
    resource = None  # Windows fallback

try:
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    Image = None
    transforms = None


# -----------------------------------------------------------------------------
# 1. High-Resolution Memory & Timing Telemetry
# -----------------------------------------------------------------------------

def get_peak_memory_mb() -> float:
    """
    Query the peak Resident Set Size (RSS) memory consumed by the current process.
    On Linux AArch64 (KV260), ru_maxrss is reported in Kilobytes.
    On macOS / BSD, ru_maxrss is reported in Bytes.
    """
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        else:
            # Linux reports in KB
            return float(usage) / 1024.0
    return 0.0


class PrecisionTimer:
    """
    Context manager for high-resolution nanosecond timing.
    """
    def __init__(self):
        self.start_ns = 0
        self.end_ns = 0
        self.duration_ms = 0.0

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_ns = time.perf_counter_ns()
        self.duration_ms = (self.end_ns - self.start_ns) / 1_000_000.0


def compute_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Compute statistical moments across measured iteration latencies.
    """
    if not latencies_ms:
        return {}
    
    arr = np.array(latencies_ms, dtype=np.float64)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p95_val = float(np.percentile(arr, 95))
    fps = 1000.0 / mean_val if mean_val > 0 else 0.0

    return {
        "mean_ms": round(mean_val, 2),
        "median_ms": round(median_val, 2),
        "std_ms": round(std_val, 2),
        "min_ms": round(min_val, 2),
        "max_ms": round(max_val, 2),
        "p95_ms": round(p95_val, 2),
        "throughput_fps": round(fps, 1),
    }


# -----------------------------------------------------------------------------
# 2. Image Preprocessing & Category Mappings
# -----------------------------------------------------------------------------

def build_imagenet_transform(image_size: int = 256):
    """
    Build standard ImageNet evaluation preprocessing pipeline for MobileViT:
    Resize(image_size) -> CenterCrop(image_size) -> ToTensor() -> Normalize
    """
    if transforms is None:
        raise ImportError("torchvision.transforms is required for image preprocessing.")
        
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image_path: str, image_size: int = 256) -> Any:
    """
    Load an image from disk, convert to RGB, and return preprocessed tensor [1, 3, H, W].
    """
    if Image is None:
        raise ImportError("PIL (Pillow) is required to load images.")
        
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    transform = build_imagenet_transform(image_size)
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)  # Shape: [1, 3, 256, 256]


def load_imagenet_classes(classes_path: Optional[str] = None) -> Dict[int, str]:
    """
    Load ImageNet-1k 1000 human-readable class names from JSON file.
    """
    if classes_path is None or not os.path.exists(classes_path):
        # Search common fallback paths
        cand_paths = [
            os.path.join(os.path.dirname(__file__), "imagenet_classes.json"),
            os.path.join(os.path.dirname(__file__), "..", "00_common", "imagenet_classes.json"),
            "/home/ubuntu/mha_eval/00_common/imagenet_classes.json",
            "imagenet_classes.json"
        ]
        for p in cand_paths:
            if os.path.exists(p):
                classes_path = p
                break

    if classes_path and os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Keys can be strings or ints
            return {int(k): v for k, v in data.items()}
            
    # Fallback to generic class labels if file missing
    return {i: f"ImageNet_Class_{i}" for i in range(1000)}


def extract_topk_predictions(
    logits_or_probs: Any, 
    class_map: Dict[int, str], 
    topk: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract Top-K predicted classes, probabilities, and class names from model output.
    """
    if torch is None:
        raise ImportError("PyTorch is required for prediction tensor extraction.")

    if isinstance(logits_or_probs, torch.Tensor):
        probs = torch.softmax(logits_or_probs.squeeze(0), dim=-1)
        top_probs, top_indices = torch.topk(probs, topk)
        top_probs = top_probs.detach().cpu().numpy().tolist()
        top_indices = top_indices.detach().cpu().numpy().tolist()
    else:
        # Numpy array fallback
        probs = np.exp(logits_or_probs) / np.sum(np.exp(logits_or_probs))
        top_indices = np.argsort(probs)[-topk:][::-1]
        top_probs = probs[top_indices].tolist()

    results = []
    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        results.append({
            "rank": rank,
            "class_id": int(idx),
            "class_name": class_map.get(int(idx), f"Class_{idx}"),
            "confidence": round(float(prob), 4),
            "confidence_percent": f"{float(prob) * 100.0:.2f}%"
        })
    return results


# -----------------------------------------------------------------------------
# 3. Formatted Terminal Reporting
# -----------------------------------------------------------------------------

def print_benchmark_header(title: str):
    """Print an eye-catching terminal banner."""
    print("=" * 76)
    print(f"  {title.upper()}")
    print("=" * 76)


def print_prediction_table(predictions: List[Dict[str, Any]]):
    """Print structured table of Top-K predictions."""
    print("\n[PREDICTION RESULTS]")
    print(f"{'Rank':<6} | {'Class ID':<10} | {'Confidence':<12} | {'Class Name'}")
    print("-" * 76)
    for p in predictions:
        print(f"#{p['rank']:<5} | {p['class_id']:<10} | {p['confidence_percent']:<12} | {p['class_name']}")


def print_metrics_summary(stats: Dict[str, float], peak_rss_mb: float):
    """Print structured table of latency and memory metrics."""
    print("\n[PERFORMANCE TELEMETRY]")
    print(f"  - Mean Inference Latency  : {stats.get('mean_ms', 0.0):.2f} ms")
    print(f"  - Median Inference Latency: {stats.get('median_ms', 0.0):.2f} ms")
    print(f"  - Standard Deviation      : {stats.get('std_ms', 0.0):.2f} ms")
    print(f"  - Min / Max Latency       : {stats.get('min_ms', 0.0):.2f} ms / {stats.get('max_ms', 0.0):.2f} ms")
    print(f"  - Tail Latency (P95)      : {stats.get('p95_ms', 0.0):.2f} ms")
    print(f"  - System Throughput       : {stats.get('throughput_fps', 0.0):.1f} FPS")
    print(f"  - Peak Memory RSS (RAM)   : {peak_rss_mb:.2f} MB")
    print("=" * 76 + "\n")

