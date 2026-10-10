"""
I-ViT Quantization-Aware Training (QAT) for MobileViT-XX-Small
==============================================================

Phase 3 of the hardware-software co-design pipeline for deployment
on the Xilinx Kria KV260 FPGA.

This script performs QAT fine-tuning on the surgered and PTQ-calibrated
MobileViT model to recover accuracy loss introduced by integer arithmetic
approximations (Shiftmax, ShiftGELU, IntLayerNorm, IntSoftmax, QuantLinear).

Key Pipeline Features:
  1. Loads calibrated checkpoint (`mobilevit_ivit_calibrated.pth`) from Phase 2.
  2. Sets model into QAT mode (`enable_qat_mode`), locking activation observers
     (QuantAct.running_stat = False) while keeping Straight-Through Estimators (STE)
     active across all integer arithmetic boundaries.
  3. Differential learning rate strategy:
     - CNN stem / inverted residuals: protected with microscopic LR (e.g., 1e-6)
       or completely frozen.
     - I-ViT Transformer blocks: fine-tuned at standard LR (e.g., 1e-4).
     - Classifier head: fine-tuned at standard LR (e.g., 1e-4).
  4. AdamW optimizer with Cosine Annealing learning rate schedule & linear warmup.
  5. Colab-optimized DataLoader supporting ImageNet (ImageFolder) with fallback
     to synthetic data for testing / CI.
  6. Tracks validation Top-1 & Top-5 accuracy and checkpoints best weights to
     `mobilevit_ivit_qat_best.pth`.

Author : Capstone Team
"""

# ===================================================================
# 0. Imports & Path Bootstrap
# ===================================================================

import sys
import os
import math
import copy
import time
import argparse
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Dict, Any, Optional

# Ensure I-ViT root is on sys.path
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
from torch.utils.data import DataLoader, Dataset
try:
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    transforms = None
    ImageFolder = None

from tqdm import tqdm

# I-ViT quantisation primitives
from models.quantization_utils import (
    QuantAct,
    QuantLinear,
    QuantMatMul,
    IntLayerNorm,
    IntSoftmax,
    IntGELU,
)

# Phase 1 surgery module
from ivit_surgery.mobilevit_ivit_surgery import replace_mobilevit_blocks


# ===================================================================
# Default Hyperparameters (Configurable at script top & via CLI)
# ===================================================================

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRANSFORMER_LR = 1e-4
DEFAULT_CNN_LR = 1e-6
DEFAULT_HEAD_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_MIN_LR_RATIO = 0.01
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_FREEZE_CNN = False   # If True: freeze CNN stem parameters completely


# ===================================================================
# 1. Model Loading & State Adaptation
# ===================================================================

def build_surgered_mobilevit(
    pretrained_model_name: str = "apple/mobilevit-xx-small",
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, list]:
    """
    Instantiate HuggingFace MobileViT and apply Phase 1 I-ViT surgery.
    """
    from transformers import MobileViTForImageClassification

    print(f"  [MODEL] Loading base model: '{pretrained_model_name}' ...")
    model = MobileViTForImageClassification.from_pretrained(
        pretrained_model_name, low_cpu_mem_usage=True
    )

    print(f"  [SURGERY] Replacing FP32 Transformer layers with I-ViT blocks ...")
    model, replaced_prefixes = replace_mobilevit_blocks(
        model,
        num_attention_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        attn_drop=0.0,
        hidden_drop=0.0,
        layer_norm_eps=1e-5,
    )
    model.to(device)
    print(f"  [SURGERY] Successfully replaced {len(replaced_prefixes)} Transformer layers.")
    return model, replaced_prefixes


def load_calibrated_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load calibrated weights and activation ranges from Phase 2 checkpoint.

    Handles buffer shape adaptation:
    During PTQ calibration, `QuantAct.act_scaling_factor` (0-dim scalar) and
    `IntLayerNorm.norm_scaling_factor` (1D tensor matching hidden dimension)
    are dynamically assigned in memory. When loading into a freshly created
    model, this function dynamically adapts buffer shapes so PyTorch's
    `load_state_dict()` succeeds without shape mismatch errors.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Calibrated checkpoint not found at: '{checkpoint_path}'. "
            f"Please run Phase 2 calibration first."
        )

    print(f"  [LOAD] Loading calibrated weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model_state = model.state_dict()
    modules_dict = dict(model.named_modules())

    adapted_buffers = 0
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            tokens = k.split(".")
            mod_name = ".".join(tokens[:-1])
            buf_name = tokens[-1]
            if mod_name in modules_dict:
                # Re-assign buffer with the calibrated shape
                setattr(modules_dict[mod_name], buf_name, v.clone())
                adapted_buffers += 1

    model.load_state_dict(state_dict)
    model.to(device)
    print(f"  [LOAD] Successfully loaded state dict ({len(state_dict)} keys, "
          f"{adapted_buffers} calibrated buffers dynamically adapted).")
    return checkpoint


# ===================================================================
# 2. QAT Mode Configuration
# ===================================================================

def enable_qat_mode(
    model: nn.Module,
    freeze_cnn: bool = False,
) -> Dict[str, int]:
    """
    Configure the surgered MobileViT model for Quantization-Aware Training (QAT).

    Rules enforced:
      1. Observers Locked: Calls `fix()` on all `QuantAct` modules (`running_stat = False`).
         Activation ranges (min_val, max_val) and scaling factors (act_scaling_factor)
         are NOT recalculated during QAT.
      2. STE Active: Fake-quantization operators (`SymmetricQuantFunction`,
         `fixedpoint_mul`, `floor_ste`, `round_ste`) are enabled to simulate integer
         bounds during forward pass and pass Straight-Through gradients during backward pass.
      3. CNN Stem Protection:
         - If `freeze_cnn=True`: sets `requires_grad = False` on all CNN parameters
           and locks BatchNorm layers in `eval()` mode.
         - If `freeze_cnn=False`: keeps CNN parameters trainable (to be fine-tuned
           with microscopic LR).
      4. Transformer Blocks: All I-ViT weights (`QuantLinear`, `IntLayerNorm`) remain
         fully trainable with `requires_grad = True`.

    Returns:
      stats: Summary dictionary of module counts and trainable parameter counts.
    """
    n_quant_act = 0
    n_quant_linear = 0
    n_int_norm = 0
    n_int_act = 0

    # Step 1: Lock all QuantAct observers
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.fix()   # sets module.running_stat = False
            assert not module.running_stat, f"Observer {name} must be fixed for QAT!"
            n_quant_act += 1
        elif isinstance(module, QuantLinear):
            n_quant_linear += 1
        elif isinstance(module, IntLayerNorm):
            n_int_norm += 1
        elif isinstance(module, (IntSoftmax, IntGELU)):
            n_int_act += 1

    # Step 2: Handle CNN freezing if requested
    n_cnn_params = 0
    n_transformer_params = 0
    n_head_params = 0

    for name, param in model.named_parameters():
        if "transformer" in name:
            param.requires_grad = True
            n_transformer_params += param.numel()
        elif "classifier" in name:
            param.requires_grad = True
            n_head_params += param.numel()
        else:
            # CNN stem / inverted residual block
            if freeze_cnn:
                param.requires_grad = False
            else:
                param.requires_grad = True
            n_cnn_params += param.numel()

    # Step 3: Put model into training mode
    model.train()

    # CRITICAL: Always keep CNN BatchNorm in eval mode to prevent destroying
    # Apple's pre-trained running statistics on mini-datasets!
    for name, module in model.named_modules():
        if "transformer" not in name and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"\n  [QAT MODE] Activated:")
    print(f"    * QuantAct observers locked (running_stat=False) : {n_quant_act}")
    print(f"    * QuantLinear layers in STE mode                 : {n_quant_linear}")
    print(f"    * IntLayerNorm modules                           : {n_int_norm}")
    print(f"    * Integer activation modules (Shiftmax/ShiftGELU): {n_int_act}")
    print(f"    * CNN Stem status                                : {'FROZEN' if freeze_cnn else 'MICRO-LR'}")
    print(f"    * Trainable parameters                           : {trainable_count:,} / {total_count:,}")

    return {
        "quant_act": n_quant_act,
        "quant_linear": n_quant_linear,
        "int_norm": n_int_norm,
        "int_act": n_int_act,
        "trainable_params": trainable_count,
        "total_params": total_count,
    }


def enforce_qat_training_state(model: nn.Module, freeze_cnn: bool = False) -> None:
    """
    Helper to be called at the start of each training epoch.
    Ensures model.train() does not unfreeze observers or alter CNN BatchNorm state.
    """
    model.train()

    # Verify observers are still fixed
    for _, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.fix()

    # CRITICAL: Always keep CNN BatchNorm in eval mode to prevent destroying
    # Apple's pre-trained running statistics on mini-datasets!
    for name, module in model.named_modules():
        if "transformer" not in name and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()


# ===================================================================
# 3. Optimizer & Learning Rate Scheduler
# ===================================================================

def build_optimizer_and_scheduler(
    model: nn.Module,
    epochs: int,
    transformer_lr: float = DEFAULT_TRANSFORMER_LR,
    cnn_lr: float = DEFAULT_CNN_LR,
    head_lr: float = DEFAULT_HEAD_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    min_lr_ratio: float = DEFAULT_MIN_LR_RATIO,
    freeze_cnn: bool = False,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Build AdamW optimizer with differential parameter groups and
    Cosine Annealing schedule with linear warmup.

    Parameter Groups:
      1. Transformer blocks: lr = transformer_lr
      2. CNN stem (if not frozen): lr = cnn_lr
      3. Classifier head: lr = head_lr
      (Each group separates weight decay for multi-dim weights vs 1D/biases).
    """
    transformer_decay = []
    transformer_no_decay = []
    cnn_decay = []
    cnn_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = param.ndim < 2 or "bias" in name or "norm" in name

        if "transformer" in name:
            if is_no_decay:
                transformer_no_decay.append(param)
            else:
                transformer_decay.append(param)
        elif "classifier" in name:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if is_no_decay:
                cnn_no_decay.append(param)
            else:
                cnn_decay.append(param)

    param_groups = []

    # Group 1: Transformer
    if transformer_decay:
        param_groups.append({
            "params": transformer_decay,
            "lr": transformer_lr,
            "weight_decay": weight_decay,
            "name": "transformer_decay",
        })
    if transformer_no_decay:
        param_groups.append({
            "params": transformer_no_decay,
            "lr": transformer_lr,
            "weight_decay": 0.0,
            "name": "transformer_no_decay",
        })

    # Group 2: Classifier Head
    if head_decay:
        param_groups.append({
            "params": head_decay,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head_decay",
        })
    if head_no_decay:
        param_groups.append({
            "params": head_no_decay,
            "lr": head_lr,
            "weight_decay": 0.0,
            "name": "head_no_decay",
        })

    # Group 3: CNN Stem (if trainable)
    if not freeze_cnn:
        if cnn_decay:
            param_groups.append({
                "params": cnn_decay,
                "lr": cnn_lr,
                "weight_decay": weight_decay,
                "name": "cnn_decay",
            })
        if cnn_no_decay:
            param_groups.append({
                "params": cnn_no_decay,
                "lr": cnn_lr,
                "weight_decay": 0.0,
                "name": "cnn_no_decay",
            })

    optimizer = torch.optim.AdamW(param_groups, eps=1e-8)

    # Cosine Annealing with Linear Warmup schedule
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  [OPTIMIZER] Initialized AdamW with {len(param_groups)} parameter groups.")
    print(f"              Transformer LR: {transformer_lr:.2e} | CNN LR: {cnn_lr if not freeze_cnn else 0.0:.2e}")
    print(f"  [SCHEDULER] Cosine Annealing: {epochs} epochs ({warmup_epochs} warmup).")

    return optimizer, scheduler


# ===================================================================
# 4. Colab-Friendly Data Loaders
# ===================================================================

class SyntheticQATDataset(Dataset):
    """
    Synthetic dataset for CI / local testing when full ImageNet is not present.
    Generates deterministic pseudo-random images and labels.
    """
    def __init__(self, num_samples: int = 256, image_size: int = 256, num_classes: int = 1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(idx)
        # Scaled [0, 1] uniform distribution matching MobileViT inputs
        image = torch.rand(3, self.image_size, self.image_size, generator=g)
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return image, label


def get_mobilevit_transforms(image_size: int = 256):
    """
    Standard Apple MobileViT data preprocessing pipeline:
      - Raw [0, 1] scaling (ToTensor)
      - Channel flip from RGB to BGR (do_flip_channel_order = True)
      - NO ImageNet mean/std normalization! (Apple MobileViT operates on raw [0, 1] BGR)
    """
    if not HAS_TORCHVISION:
        return None, None

    # Apple MobileViT was trained on raw [0, 1] BGR tensors.
    # We flip RGB -> BGR via lambda x: x[[2, 1, 0], :, :]
    flip_bgr = transforms.Lambda(lambda x: x[[2, 1, 0], :, :])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        flip_bgr,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 288 / 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        flip_bgr,
    ])

    return train_transform, val_transform


def build_qat_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    image_size: int = 256,
    synthetic_samples: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.

    Searches `data_dir` for `train/` and `val/` (or `validation/`).
    If `data_dir` is not provided or folders do not exist, falls back cleanly
    to `SyntheticQATDataset` to allow smoke testing on any machine.
    """
    # Windows safeguard: workers > 0 can trigger multiprocessing spawn issues
    if sys.platform == "win32" and num_workers > 0:
        print("  [WARN] Windows OS detected. Adjusting num_workers to 0 for stability.")
        num_workers = 0

    use_synthetic = True
    train_dataset = None
    val_dataset = None

    if data_dir is not None and os.path.isdir(data_dir):
        if not HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required to load ImageFolder datasets. "
                "Please run 'pip install torchvision' or run on Google Colab where it is pre-installed."
            )

        train_transform, val_transform = get_mobilevit_transforms(image_size=image_size)

        # Look for train & val subdirectories
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        if not os.path.isdir(val_path):
            val_path = os.path.join(data_dir, "validation")

        if os.path.isdir(train_path) and os.path.isdir(val_path):
            print(f"  [DATA] Found ImageNet dataset at: {data_dir}")
            print(f"         Train dir: {train_path}")
            print(f"         Val dir  : {val_path}")
            train_dataset = ImageFolder(train_path, transform=train_transform)
            val_dataset = ImageFolder(val_path, transform=val_transform)
            use_synthetic = False
        else:
            print(f"  [WARN] '{data_dir}' does not contain 'train' and 'val' subdirectories.")

    if use_synthetic:
        print(f"  [DATA] Using SYNTHETIC dataset ({synthetic_samples} samples). "
              f"Provide --data-dir /path/to/imagenet for real ImageNet training.")
        train_dataset = SyntheticQATDataset(num_samples=synthetic_samples, image_size=image_size)
        val_dataset = SyntheticQATDataset(num_samples=max(32, synthetic_samples // 4), image_size=image_size)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) > batch_size else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"  [DATA] Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
          f"(Batch size: {batch_size}, Pin memory: {pin_memory})")

    return train_loader, val_loader


# ===================================================================
# 5. Accuracy & Metrics
# ===================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def compute_topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[float, float]:
    """Computes Top-1 and Top-5 accuracy percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0], res[1]


# ===================================================================
# 6. Training & Validation Epoch Loops
# ===================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    freeze_cnn: bool = False,
) -> Tuple[float, float, float]:
    """
    Train model for one QAT epoch.
    Straight-Through Estimator (STE) passes gradients back to weights.
    """
    enforce_qat_training_state(model, freeze_cnn=freeze_cnn)

    loss_meter = AverageMeter("Loss", ":.4f")
    top1_meter = AverageMeter("Acc@1", ":6.2f")
    top5_meter = AverageMeter("Acc@5", ":6.2f")

    pbar = tqdm(
        train_loader,
        desc=f"  Train Epoch {epoch:02d}",
        unit="batch",
        ncols=95,
        leave=False,
    )

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass through fake-quantized network
        outputs = model(pixel_values=images)
        logits = outputs.logits

        loss = criterion(logits, targets)

        # Backward pass via STE
        loss.backward()

        # Gradient clipping
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

        optimizer.step()

        # Metrics
        acc1, acc5 = compute_topk_accuracy(logits, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1, images.size(0))
        top5_meter.update(acc5, images.size(0))

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "top1": f"{top1_meter.avg:.2f}%",
            "top5": f"{top5_meter.avg:.2f}%",
        })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate quantized model on validation dataset.
    """
    model.eval()

    # Ensure observers remain fixed in eval mode
    for _, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.fix()

    loss_meter = AverageMeter("Val Loss", ":.4f")
    top1_meter = AverageMeter("Val Acc@1", ":6.2f")
    top5_meter = AverageMeter("Val Acc@5", ":6.2f")

    pbar = tqdm(
        val_loader,
        desc="  Validating",
        unit="batch",
        ncols=95,
        leave=False,
    )

    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            loss = criterion(logits, targets)

            acc1, acc5 = compute_topk_accuracy(logits, targets, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1, images.size(0))
            top5_meter.update(acc5, images.size(0))

            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{top1_meter.avg:.2f}%",
            })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# ===================================================================
# 7. Checkpointing
# ===================================================================

def save_qat_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_acc1: float,
    val_acc5: float,
    save_path: str,
    is_best: bool = False,
    extra_config: Optional[dict] = None,
) -> None:
    """Save training checkpoint with metadata."""
    checkpoint = {
        "epoch": epoch,
        "val_acc1": val_acc1,
        "val_acc5": val_acc5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": extra_config or {},
    }

    torch.save(checkpoint, save_path)
    tag = "[BEST CHECKPOINT]" if is_best else "[CHECKPOINT]"
    print(f"  {tag} Saved: {save_path} (Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%)")


# ===================================================================
# 8. Main Training Pipeline
# ===================================================================

def train_qat(
    calibrated_ckpt_path: str,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    transformer_lr: float = DEFAULT_TRANSFORMER_LR,
    cnn_lr: float = DEFAULT_CNN_LR,
    head_lr: float = DEFAULT_HEAD_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    freeze_cnn: bool = DEFAULT_FREEZE_CNN,
    device_str: Optional[str] = None,
    num_workers: int = 4,
) -> Tuple[nn.Module, float]:
    """
    Full Quantization-Aware Training Pipeline execution.
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = _SURGERY_DIR
    os.makedirs(save_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(save_dir, "mobilevit_ivit_qat_best.pth")
    latest_checkpoint_path = os.path.join(save_dir, "mobilevit_ivit_qat_latest.pth")

    print(f"\n{'='*70}")
    print(f"  Phase 3: I-ViT Quantization-Aware Training (QAT)")
    print(f"  Device              : {device}")
    print(f"  Calibrated Checkpoint: {calibrated_ckpt_path}")
    print(f"  Epochs              : {epochs}")
    print(f"  Batch size          : {batch_size}")
    print(f"  Transformer LR      : {transformer_lr:.2e}")
    print(f"  CNN LR              : {cnn_lr:.2e} (freeze_cnn={freeze_cnn})")
    print(f"  Warmup Epochs       : {warmup_epochs}")
    print(f"  Data Directory      : {data_dir or '(synthetic fallback)'}")
    print(f"  Best Model Output   : {best_checkpoint_path}")
    print(f"{'='*70}\n")

    # Step 1: Instantiate surgered model
    print("[1/5] Building surgered MobileViT-XX-Small ...")
    model, replaced_prefixes = build_surgered_mobilevit(device=device)

    # Step 2: Load calibrated weights
    print("\n[2/5] Loading PTQ calibrated checkpoint ...")
    load_calibrated_weights(model, calibrated_ckpt_path, device=device)

    # Step 3: Activate QAT mode
    print("\n[3/5] Activating QAT mode (locking observers, enabling STE) ...")
    stats = enable_qat_mode(model, freeze_cnn=freeze_cnn)

    # Step 4: Build DataLoaders
    print("\n[4/5] Building DataLoaders ...")
    train_loader, val_loader = build_qat_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        synthetic_samples=128 if data_dir is None else None,
    )

    # Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        epochs=epochs,
        transformer_lr=transformer_lr,
        cnn_lr=cnn_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        freeze_cnn=freeze_cnn,
    )

    # Loss function (CrossEntropy with Label Smoothing)
    if label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Initial validation before training
    print("\n[5/5] Running pre-training evaluation (baseline calibrated state) ...")
    val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
    print(f"  [BASELINE] Calibrated Model Accuracy: Top-1 = {val_acc1:.2f}%, Top-5 = {val_acc5:.2f}%\n")

    best_acc1 = val_acc1
    start_time = time.time()

    print(f"{'-'*70}")
    print(f"  Beginning QAT Training Loop ({epochs} epochs)")
    print(f"{'-'*70}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            grad_clip=grad_clip,
            freeze_cnn=freeze_cnn,
        )

        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Step LR scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        is_best = (epoch == 1) or (val_acc1 > best_acc1)
        if is_best:
            best_acc1 = max(best_acc1, val_acc1)
            save_qat_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_acc1=val_acc1,
                val_acc5=val_acc5,
                save_path=best_checkpoint_path,
                is_best=True,
                extra_config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "transformer_lr": transformer_lr,
                    "freeze_cnn": freeze_cnn,
                },
            )

        # Always save latest
        save_qat_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_acc1=val_acc1,
            val_acc5=val_acc5,
            save_path=latest_checkpoint_path,
            is_best=False,
        )

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc@1: {train_acc1:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc@1: {val_acc1:.2f}% (Best: {best_acc1:.2f}%) | "
            f"LR: {current_lr:.2e}"
        )

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  QAT Training Complete in {total_time/60:.1f} min!")
    print(f"  Best Validation Acc@1: {best_acc1:.2f}%")
    print(f"  Saved Best Model Checkpoint: {best_checkpoint_path}")
    print(f"{'='*70}\n")

    return model, best_acc1


# ===================================================================
# 9. CLI Entry Point
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: Quantization-Aware Training (QAT) for MobileViT-XX-Small"
    )
    parser.add_argument(
        "--calibrated-ckpt", type=str,
        default=os.path.join(_SURGERY_DIR, "mobilevit_ivit_calibrated.pth"),
        help="Path to Phase 2 calibrated checkpoint (.pth).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to ImageNet dataset root (containing train/ and val/). "
             "Falls back to synthetic data if not specified.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=_SURGERY_DIR,
        help="Directory to save QAT checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of QAT epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--transformer-lr", type=float, default=DEFAULT_TRANSFORMER_LR,
        help=f"Learning rate for Transformer blocks (default: {DEFAULT_TRANSFORMER_LR}).",
    )
    parser.add_argument(
        "--cnn-lr", type=float, default=DEFAULT_CNN_LR,
        help=f"Learning rate for CNN stem (default: {DEFAULT_CNN_LR}).",
    )
    parser.add_argument(
        "--head-lr", type=float, default=DEFAULT_HEAD_LR,
        help=f"Learning rate for classifier head (default: {DEFAULT_HEAD_LR}).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay (default: {DEFAULT_WEIGHT_DECAY}).",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS,
        help=f"Warmup epochs (default: {DEFAULT_WARMUP_EPOCHS}).",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING,
        help=f"Label smoothing factor (default: {DEFAULT_LABEL_SMOOTHING}).",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
        help=f"Gradient clipping norm (default: {DEFAULT_GRAD_CLIP}).",
    )
    parser.add_argument(
        "--freeze-cnn", action="store_true", default=DEFAULT_FREEZE_CNN,
        help="Freeze CNN stem parameters completely during QAT.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use ('cuda' or 'cpu'). Auto-detects if None.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader num_workers (default: 4, auto 0 on Windows).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_qat(
        calibrated_ckpt_path=args.calibrated_ckpt,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        transformer_lr=args.transformer_lr,
        cnn_lr=args.cnn_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        freeze_cnn=args.freeze_cnn,
        device_str=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
