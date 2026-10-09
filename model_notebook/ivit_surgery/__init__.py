# I-ViT Surgery package for MobileViT Architecture Surgery
# Phase 1: Replace FP32 Transformer blocks with I-ViT integer-only equivalents
# Phase 2: PTQ Calibration — collect activation statistics and compute dyadic scales
# Phase 3: QAT Fine-Tuning — recovery of accuracy drop via STE on fake-quantized model

from .mobilevit_ivit_surgery import (
    IViTMobileViTEncoderLayer,
    IViTMobileViTTransformerLayer,
    IViTSelfAttention,
    IViTMlp,
    replace_mobilevit_blocks,
    transfer_weights,
    build_weight_map,
    fuse_qkv_weights,
)

from .mobilevit_ivit_ptq_calibration import (
    enable_calibration_mode,
    disable_calibration_mode,
    run_calibration,
    build_calibration_loader,
    save_calibrated_model,
)

from .mobilevit_ivit_qat import (
    enable_qat_mode,
    load_calibrated_weights,
    build_qat_dataloaders,
    build_optimizer_and_scheduler,
    train_one_epoch,
    validate,
    save_qat_checkpoint,
    train_qat,
)

__all__ = [
    # Phase 1 — Surgery
    "IViTMobileViTEncoderLayer",
    "IViTMobileViTTransformerLayer",
    "IViTSelfAttention",
    "IViTMlp",
    "replace_mobilevit_blocks",
    "transfer_weights",
    "build_weight_map",
    "fuse_qkv_weights",
    # Phase 2 — PTQ Calibration
    "enable_calibration_mode",
    "disable_calibration_mode",
    "run_calibration",
    "build_calibration_loader",
    "save_calibrated_model",
    # Phase 3 — QAT Fine-Tuning
    "enable_qat_mode",
    "load_calibrated_weights",
    "build_qat_dataloaders",
    "build_optimizer_and_scheduler",
    "train_one_epoch",
    "validate",
    "save_qat_checkpoint",
    "train_qat",
]
