"""
I-ViT Post-Training Quantization (PTQ) Calibration for MobileViT
=================================================================

Phase 2 of the hardware-software co-design pipeline.

This script:
  1. Loads the surgered MobileViT model (from Phase 1) containing I-ViT
     integer-only Transformer layers (QuantLinear, QuantAct, IntLayerNorm,
     IntSoftmax / Shiftmax, IntGELU / ShiftGELU).

  2. Runs 1,024 representative calibration images through the network
     under `torch.no_grad()` to collect activation statistics.  The I-ViT
     `QuantAct` nodes use exponential moving average (momentum = 0.95)
     to track running min/max per activation channel.

  3. Freezes the activation ranges by calling `freeze_model()`, which
     sets `QuantAct.running_stat = False` on every observer.  After
     freezing, the scaling factors are locked:

         act_scaling_factor = max(|min_val|, |max_val|) / (2^(bit-1) - 1)

  4. Decomposes every scaling factor into dyadic form:

         S = M x 2^{-E}        (integer multiplier M, bit-shift E)

     This is the representation used by the FPGA fixed-point datapath
     via `batch_frexp()`.

  5. Saves the calibrated state_dict to `mobilevit_ivit_calibrated.pth`.

Prerequisites:
  - Phase 1 surgery script must be importable.
  - `transformers`, `torch`, `tqdm`, `Pillow` must be installed.
  - An ImageNet validation set (or any image folder) for calibration data.
    If ImageNet is unavailable, the script falls back to generating
    synthetic calibration tensors (useful for CI / testing).

Usage:
    python -m ivit_surgery.mobilevit_ivit_ptq_calibration \\
        --data-dir /path/to/imagenet/val \\
        [--num-images 1024] [--batch-size 32] [--device cuda]

Author : Capstone Team
"""

# ===================================================================
# 0. Imports & Path Bootstrap
# ===================================================================

import sys
import os
import copy
import argparse
import json
from pathlib import Path
from collections import OrderedDict

# Ensure I-ViT root is on sys.path (same setup as surgery script)
IVIT_ROOT = os.environ.get("IVIT_ROOT")
if not IVIT_ROOT or not os.path.exists(IVIT_ROOT):
    cand_rel = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "I-ViT"))
    if os.path.exists(cand_rel):
        IVIT_ROOT = cand_rel
    else:
        for cand in ["/content/I-ViT", os.path.abspath("I-ViT"), os.path.abspath("../I-ViT"), os.path.expanduser("~/I-ViT")]:
            if os.path.exists(cand):
                IVIT_ROOT = cand
                break
        else:
            IVIT_ROOT = cand_rel

if IVIT_ROOT and IVIT_ROOT not in sys.path:
    sys.path.insert(0, IVIT_ROOT)

# Surgery package path
_SURGERY_DIR = os.path.dirname(os.path.abspath(__file__))
_NOTEBOOK_DIR = os.path.dirname(_SURGERY_DIR)
if _NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, _NOTEBOOK_DIR)

# Apply device-compatibility patches BEFORE any I-ViT imports
from ivit_surgery import patch_ivit_device  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from tqdm import tqdm

# I-ViT quantisation primitives
from models.quantization_utils import QuantAct, QuantLinear, QuantMatMul
from models.quantization_utils.quant_utils import (
    symmetric_linear_quantization_params,
    batch_frexp,
)

# --------------------------------------------------------------------------
# Safe freeze / unfreeze helpers
# --------------------------------------------------------------------------
# NOTE: We do NOT use I-ViT's built-in `freeze_model()` / `unfreeze_model()`
# from `models.model_utils` because they walk the model using `dir()` +
# `getattr()`.  HuggingFace `PreTrainedModel` exposes a `base_model`
# *property* that returns `self`, causing infinite recursion when the walker
# calls `getattr(model, 'base_model')`.
#
# Our safe versions use `model.named_modules()`, which is an iterator that
# yields (name, module) pairs without triggering property descriptors.
# --------------------------------------------------------------------------

def _safe_freeze_model(model: nn.Module) -> None:
    """Freeze activation ranges: QuantAct.running_stat = False."""
    for _, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.fix()   # sets running_stat = False

def _safe_unfreeze_model(model: nn.Module) -> None:
    """Unfreeze activation ranges: QuantAct.running_stat = True."""
    for _, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.unfix()  # sets running_stat = True

# Phase 1 surgery
from ivit_surgery.mobilevit_ivit_surgery import replace_mobilevit_blocks


# ===================================================================
# 1. Calibration Dataset Loader
# ===================================================================

class ImageFolderCalibrationDataset(Dataset):
    """
    Lightweight dataset that reads images from a directory tree and
    applies the exact preprocessing used by MobileViTImageProcessor:
      - Resize shortest edge to 288
      - CenterCrop to 256x256
      - Convert to RGB float tensor  [0, 1]
      - Normalize with ImageNet mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]

    We use torchvision transforms to match HuggingFace's processor
    pipeline exactly, avoiding any dependency on the specific HF
    processor version.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    # MobileViT-XX-Small expects 256x256 input
    CROP_SIZE = 256
    RESIZE_SIZE = 288

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root_dir: str, max_images: int = 1024):
        """
        Parameters
        ----------
        root_dir : str
            Path to image directory (e.g., ImageNet val/).  Images can be
            in subdirectories (ImageNet-style class folders).
        max_images : int
            Maximum number of images to load for calibration.
        """
        from torchvision import transforms

        self.root = Path(root_dir)
        assert self.root.is_dir(), f"Calibration data directory not found: {root_dir}"

        # Collect all image paths (recursive)
        all_images = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.IMAGE_EXTENSIONS and p.is_file()
        ])

        if len(all_images) == 0:
            raise FileNotFoundError(
                f"No images found in {root_dir}.  "
                f"Expected extensions: {self.IMAGE_EXTENSIONS}"
            )

        # Take at most `max_images`
        self.image_paths = all_images[:max_images]
        print(f"  [DATA] Found {len(all_images)} images, "
              f"using {len(self.image_paths)} for calibration.")

        # Build the transform pipeline matching Apple MobileViTImageProcessor:
        # Scale to [0, 1] and flip RGB -> BGR (NO ImageNet normalization!)
        self.transform = transforms.Compose([
            transforms.Resize(self.RESIZE_SIZE),
            transforms.CenterCrop(self.CROP_SIZE),
            transforms.ToTensor(),               # -> [C, H, W] float32 in [0, 1]
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :]),  # RGB -> BGR expected by Apple MobileViT
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class SyntheticCalibrationDataset(Dataset):
    """
    Fallback dataset that generates random tensors matching the
    expected input distribution.  Used when no real image directory
    is available (e.g., CI or quick smoke tests).

    WARNING: Synthetic data will NOT produce meaningful activation
    ranges.  Always prefer real images for production calibration.
    """

    def __init__(self, num_images: int = 1024, image_size: int = 256):
        self.num_images = num_images
        self.image_size = image_size
        # Pre-generate a fixed random seed for reproducibility
        self.rng = torch.Generator().manual_seed(42)
        print(f"  [DATA] Using SYNTHETIC calibration data ({num_images} images). "
              f"Results will NOT be production-grade.")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Uniform [0, 1] distribution matching raw image tensors
        return torch.rand(3, 256, 256)


def build_calibration_loader(data_dir: str | None,
                             num_images: int = 1024,
                             batch_size: int = 32,
                             num_workers: int = 4) -> DataLoader:
    """
    Build a DataLoader for PTQ calibration.

    Parameters
    ----------
    data_dir : str or None
        Path to calibration images.  If None or non-existent, falls
        back to synthetic data.
    num_images : int
        Number of calibration images to use (default: 1024).
    batch_size : int
        Batch size for the calibration loop (default: 32).
    num_workers : int
        DataLoader workers (default: 4).

    Returns
    -------
    DataLoader
    """
    if data_dir and Path(data_dir).is_dir():
        dataset = ImageFolderCalibrationDataset(data_dir, max_images=num_images)
    else:
        if data_dir:
            print(f"  [WARN] Data directory '{data_dir}' not found. "
                  f"Falling back to synthetic data.")
        dataset = SyntheticCalibrationDataset(num_images=num_images)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # Deterministic ordering for reproducibility
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ===================================================================
# 2. Calibration State Management
# ===================================================================

def enable_calibration_mode(model: nn.Module) -> None:
    """
    Put the model into calibration mode.

    This recursively walks the model and calls `unfix()` on every
    `QuantAct` module, which sets `running_stat = True`.  With
    running_stat enabled, each forward pass updates the exponential
    moving average of activation min/max values:

        min_val = momentum * min_val + (1 - momentum) * batch_min
        max_val = momentum * max_val + (1 - momentum) * batch_max

    These running statistics are used to compute the symmetric
    quantization scaling factor:

        act_scaling_factor = max(|min|, |max|) / (2^(bit-1) - 1)

    IMPORTANT: The model must be in eval() mode so that CNN BatchNorm
    layers use their running mean/variance (not batch statistics).
    Only the I-ViT QuantAct observers should be "unfrozen".
    """
    model.eval()  # CNN BatchNorm -> running stats, dropout disabled

    # Use our safe recursive unfreezer (avoids HF base_model recursion)
    # This calls QuantAct.unfix() -> self.running_stat = True
    _safe_unfreeze_model(model)

    # Count how many QuantAct observers were enabled
    n_observers = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            assert module.running_stat, \
                f"Observer {name} not unfrozen after unfreeze_model()!"
            n_observers += 1

    print(f"  [CALIB] Enabled calibration mode: "
          f"{n_observers} QuantAct observers are now tracking activations.")
    print(f"  [CALIB] Observer momentum: "
          f"{0.95} (exponential moving average)")


def disable_calibration_mode(model: nn.Module) -> dict:
    """
    Freeze calibration and compute final dyadic scales.

    This function:
      1. Calls `freeze_model()` which sets `QuantAct.running_stat = False`
         on all observers.  The min/max values are now LOCKED.

      2. Runs one more symbolic forward pass (zero tensor) so that every
         `QuantAct.forward()` computes the final `act_scaling_factor`
         from the locked min/max values via:

             act_scaling_factor = symmetric_linear_quantization_params(
                 bit_width, min_val, max_val
             )

      3. Collects all scaling factors and decomposes them into dyadic
         form S = M x 2^{-E} using `batch_frexp()`.

    Parameters
    ----------
    model : nn.Module
        The calibrated model.

    Returns
    -------
    scale_report : dict
        Dictionary mapping module names to their dyadic scale parameters:
        {name: {"scaling_factor": float, "mantissa": int, "exponent": int,
                "bit_width": int, "min_val": float, "max_val": float}}
    """
    # Step 1: Freeze all QuantAct observers
    # ----------------------------------------------------------
    # After this call, QuantAct.running_stat = False for all observers.
    # The forward() method will still compute act_scaling_factor from the
    # now-frozen min_val / max_val, but will NO LONGER update them.
    _safe_freeze_model(model)

    n_frozen = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            assert not module.running_stat, \
                f"Observer {name} not frozen after freeze_model()!"
            n_frozen += 1

    print(f"  [CALIB] Disabled calibration mode: "
          f"{n_frozen} QuantAct observers FROZEN.")

    # Step 2: Collect dyadic scale decompositions
    # ----------------------------------------------------------
    # For each QuantAct, the act_scaling_factor was computed during
    # the calibration forward passes.  We now decompose it:
    #
    #     S = M x 2^{-E}
    #
    # where M is an integer mantissa and E is the bit-shift exponent.
    # This is the representation used by the FPGA integer datapath.
    scale_report = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            sf = module.act_scaling_factor.detach().clone()

            # Ensure scaling factor is valid (non-zero after calibration)
            if sf.abs().sum().item() < 1e-15:
                print(f"  [WARN] {name}: scaling_factor is zero -- "
                      f"this observer may not have seen any data.")
                continue

            # Dyadic decomposition: S = M x 2^{-E}
            # batch_frexp returns (mantissa_int, exponent) such that
            #   scaling_factor ~= mantissa_int x 2^{-(31 - exponent)}
            # but we use the standard frexp convention here.
            try:
                mantissa, exponent = batch_frexp(sf)
                scale_report[name] = {
                    "scaling_factor": sf.item() if sf.numel() == 1 else sf.cpu().tolist(),
                    "mantissa_M": mantissa.item() if mantissa.numel() == 1 else mantissa.cpu().tolist(),
                    "exponent_E": exponent.item() if exponent.numel() == 1 else exponent.cpu().tolist(),
                    "bit_width": module.activation_bit,
                    "min_val": module.min_val.item() if hasattr(module.min_val, 'item') else float(module.min_val),
                    "max_val": module.max_val.item() if hasattr(module.max_val, 'item') else float(module.max_val),
                }
            except Exception as e:
                print(f"  [WARN] Could not decompose scale for {name}: {e}")
                scale_report[name] = {
                    "scaling_factor": sf.item() if sf.numel() == 1 else sf.cpu().tolist(),
                    "error": str(e),
                }

    print(f"  [CALIB] Collected dyadic scales for "
          f"{len(scale_report)} QuantAct nodes.")

    return scale_report


# ===================================================================
# 3. The Calibration Loop
# ===================================================================

def run_calibration(model: nn.Module,
                    dataloader: DataLoader,
                    device: torch.device) -> None:
    """
    Run the calibration loop: forward-pass 1,024 images through the
    model under torch.no_grad() to populate the QuantAct observers.

    During each forward pass, every QuantAct node updates its running
    min/max statistics via exponential moving average:

        if first_batch:
            min_val = batch_min
            max_val = batch_max
        else:
            min_val = 0.95 * min_val + 0.05 * batch_min
            max_val = 0.95 * max_val + 0.05 * batch_max

    These statistics are tracked at ALL quantisation points:
      - Q/K/V inputs           (qact_input, qact1)
      - Attention scores       (qact_attn1, qact_softmax)
      - MLP activations        (qact_gelu, mlp.qact1, mlp.qact2)
      - Residual additions     (qact_res1, qact_res2)
      - Self-attention output  (attn.qact1, attn.qact2, attn.qact3)

    Parameters
    ----------
    model : nn.Module
        The surgered model with calibration mode ENABLED.
    dataloader : DataLoader
        Calibration data loader.
    device : torch.device
        Device to run calibration on.
    """
    model.eval()
    total_images = 0

    print(f"\n{'-'*60}")
    print(f"  PTQ Calibration Loop")
    print(f"  Batches: {len(dataloader)}  |  "
          f"Batch size: {dataloader.batch_size}  |  "
          f"Device: {device}")
    print(f"{'-'*60}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc="  Calibrating",
            unit="batch",
            ncols=80,
        )):
            # The dataset returns plain tensors (no labels needed for PTQ)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            # ------------------------------------------------------
            # Forward pass -- this is where QuantAct observers inside
            # the I-ViT blocks accumulate their running min/max stats.
            #
            # The HuggingFace model wrapper expects `pixel_values=`.
            # The output logits are discarded; we only care about the
            # side-effect of updating observer statistics.
            # ------------------------------------------------------
            _ = model(pixel_values=images)

            total_images += images.shape[0]

    print(f"\n  [CALIB] Calibration complete: {total_images} images processed.")


# ===================================================================
# 4. Scale Finalization & Saving
# ===================================================================

def save_calibrated_model(model: nn.Module,
                          scale_report: dict,
                          save_dir: str,
                          model_filename: str = "mobilevit_ivit_calibrated.pth",
                          report_filename: str = "calibration_scale_report.json") -> None:
    """
    Save the calibrated model state_dict and the dyadic scale report.

    The state_dict includes:
      - All original CNN (MobileNetV2) weights
      - All I-ViT Transformer weights (QuantLinear)
      - All QuantAct buffers:  act_scaling_factor, min_val, max_val
      - All QuantLinear buffers:  fc_scaling_factor, weight_integer, bias_integer
      - IntLayerNorm buffers:  norm_scaling_factor, bias_integer

    Parameters
    ----------
    model : nn.Module
        The calibrated model (after freeze_model).
    scale_report : dict
        Dyadic scale decomposition from disable_calibration_mode().
    save_dir : str
        Directory to save the checkpoint and report.
    model_filename : str
        Filename for the state_dict checkpoint.
    report_filename : str
        Filename for the JSON scale report.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # -- Save model state_dict --
    model_path = save_path / model_filename
    torch.save({
        "model_state_dict": model.state_dict(),
        "calibration_config": {
            "num_calibration_images": 1024,
            "observer_momentum": 0.95,
            "activation_bit": 8,
            "weight_bit": 8,
            "quantization_mode": "symmetric",
        },
    }, model_path)
    print(f"  [SAVE] Calibrated model -> {model_path}")
    print(f"         State dict keys: {len(model.state_dict())}")

    # -- Save dyadic scale report (human-readable JSON) --
    report_path = save_path / report_filename
    with open(report_path, "w") as f:
        json.dump(scale_report, f, indent=2, default=str)
    print(f"  [SAVE] Scale report    -> {report_path}")
    print(f"         QuantAct nodes:   {len(scale_report)}")


def print_scale_summary(scale_report: dict) -> None:
    """Pretty-print a summary of the calibrated dyadic scales."""
    print(f"\n{'='*70}")
    print(f"  Dyadic Scale Summary:  S = M x 2^{{-E}}")
    print(f"{'='*70}")
    print(f"  {'Module':<50} {'S':>10}  {'M':>12}  {'E':>6}  {'bit':>3}")
    print(f"  {'-'*50} {'-'*10}  {'-'*12}  {'-'*6}  {'-'*3}")

    for name, info in scale_report.items():
        if "error" in info:
            print(f"  {name:<50} ERROR: {info['error']}")
            continue

        sf = info["scaling_factor"]
        M = info["mantissa_M"]
        E = info["exponent_E"]
        bit = info["bit_width"]

        # Format for scalar values
        if isinstance(sf, (int, float)):
            print(f"  {name:<50} {sf:>10.6f}  {M:>12}  {E:>6.0f}  {bit:>3}")
        else:
            print(f"  {name:<50} [tensor]     [tensor]     [..]   {bit:>3}")

    print(f"{'='*70}\n")


# ===================================================================
# 5. Main Entry Point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: PTQ Calibration for I-ViT MobileViT"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to calibration images (e.g., ImageNet val/).  "
             "Falls back to synthetic data if not provided.",
    )
    parser.add_argument(
        "--num-images", type=int, default=1024,
        help="Number of calibration images (default: 1024).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for calibration loop (default: 32).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (default: auto-detect cuda/cpu).",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save calibrated model.  "
             "Defaults to the ivit_surgery package directory.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0 for Windows compatibility).",
    )
    args = parser.parse_args()

    # -- Device selection --
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Save directory --
    if args.save_dir is None:
        args.save_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*60}")
    print(f"  Phase 2 -- I-ViT PTQ Calibration for MobileViT-XX-Small")
    print(f"  Device          : {device}")
    print(f"  Calibration imgs: {args.num_images}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Data directory  : {args.data_dir or '(synthetic)'}")
    print(f"  Save directory  : {args.save_dir}")
    print(f"{'='*60}\n")

    # ==================================================================
    # Step 1: Load and surgery the model  (reuse Phase 1 pipeline)
    # ==================================================================
    print("[1/5] Loading apple/mobilevit-xx-small and applying I-ViT surgery ...")
    from transformers import MobileViTForImageClassification

    model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-xx-small", low_cpu_mem_usage=True
    )

    model, replaced_prefixes = replace_mobilevit_blocks(
        model,
        num_attention_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        attn_drop=0.0,
        hidden_drop=0.0,       # No dropout during calibration
        layer_norm_eps=1e-5,
    )

    model.to(device)
    model.eval()  # CRITICAL: CNN BatchNorm uses running stats, NOT batch stats

    print(f"\n  Transformer layers replaced: {len(replaced_prefixes)}")
    for pfx in replaced_prefixes:
        print(f"    * {pfx}")

    # ==================================================================
    # Step 2: Build calibration data loader
    # ==================================================================
    print(f"\n[2/5] Building calibration data loader ...")
    cal_loader = build_calibration_loader(
        data_dir=args.data_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ==================================================================
    # Step 3: Enable calibration mode (unfreeze QuantAct observers)
    # ==================================================================
    print(f"\n[3/5] Enabling calibration mode (unfreezing QuantAct observers) ...")
    enable_calibration_mode(model)

    # ==================================================================
    # Step 4: Run the calibration loop
    # ==================================================================
    print(f"\n[4/5] Running calibration loop ...")
    run_calibration(model, cal_loader, device)

    # ==================================================================
    # Step 5: Freeze observers, compute dyadic scales, and save
    # ==================================================================
    print(f"\n[5/5] Freezing observers and computing dyadic scales ...")

    # -- Lock the activation ranges --
    # After this call, QuantAct.running_stat = False and min_val/max_val
    # are frozen.  All subsequent forward passes will use these locked
    # ranges to compute act_scaling_factor.
    scale_report = disable_calibration_mode(model)

    # -- Print summary --
    print_scale_summary(scale_report)

    # -- Save calibrated checkpoint --
    save_calibrated_model(model, scale_report, save_dir=args.save_dir)

    # ==================================================================
    # Verification: Quick forward pass with frozen scales
    # ==================================================================
    print(f"\n{'-'*60}")
    print(f"  Verification: forward pass with frozen scales")
    print(f"{'-'*60}")

    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=device)
        out = model(pixel_values=dummy)
        logits = out.logits
        print(f"  Output shape : {logits.shape}")
        print(f"  Logit range  : [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        top5 = logits.topk(5).indices[0].tolist()
        print(f"  Top-5 classes: {top5}")

    print(f"\n{'='*60}")
    print(f"  Phase 2 PTQ Calibration -- COMPLETE [OK]")
    print(f"{'='*60}\n")

    return model, scale_report


if __name__ == "__main__":
    main()
