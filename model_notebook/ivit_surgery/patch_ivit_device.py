"""
Device-Compatibility Patch for I-ViT
=====================================

The original I-ViT codebase hardcodes `.cuda()` in several places inside
`quant_utils.py` and `quant_modules.py`.  This module monkey-patches those
files at import time so they work on **any** device (CPU, CUDA, MPS, etc.).

Usage:
    import ivit_surgery.patch_ivit_device   # just import — patches apply automatically

What gets patched:
    1. quant_utils.SymmetricQuantFunction.forward  — `torch.tensor(0.).cuda()`
    2. quant_utils.batch_frexp                     — `.cuda()` on numpy conversions
    3. quant_modules.IntLayerNorm.forward           — `torch.sqrt(n).cuda()`
    4. quant_modules.IntGELU.forward                — `torch.Tensor([...]).cuda()`
    5. quant_modules.IntSoftmax.forward             — `torch.Tensor([...]).cuda()`
"""

import sys, os

# Ensure I-ViT is importable
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

import torch
import numpy as np
from decimal import Decimal
import decimal

# Import the modules we need to patch
from models.quantization_utils import quant_utils as _qu
from models.quantization_utils import quant_modules as _qm


# -----------------------------------------------------------------------
# Patch 1: SymmetricQuantFunction.forward  —  zero_point device
# -----------------------------------------------------------------------
_orig_sqf_forward = getattr(_qu.SymmetricQuantFunction.forward, "__func__", _qu.SymmetricQuantFunction.forward)

@staticmethod
def _patched_sqf_forward(ctx, x, k, specified_scale, is_weight):
    scale = specified_scale
    zero_point = torch.tensor(0., device=x.device)       # ← patched
    n = 2 ** (k - 1) - 1
    new_quant_x = _qu.linear_quantize(x, scale, zero_point, is_weight=is_weight)
    new_quant_x = torch.clamp(new_quant_x, -n - 1, n)
    ctx.scale = scale
    ctx.is_weight = is_weight
    return new_quant_x

_qu.SymmetricQuantFunction.forward = _patched_sqf_forward


# -----------------------------------------------------------------------
# Patch 2: batch_frexp  —  numpy→tensor device
# -----------------------------------------------------------------------
def _patched_batch_frexp(inputs, max_bit=31):
    shape_of_input = inputs.size()
    device = inputs.device                                # ← patched
    inputs_flat = inputs.view(-1)
    output_m, output_e = np.frexp(inputs_flat.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(float(m * (2 ** max_bit))).quantize(
            Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    output_e = float(max_bit) - output_e
    return (torch.from_numpy(output_m).to(device).view(shape_of_input),  # ← patched
            torch.from_numpy(output_e).to(device).view(shape_of_input))

_qu.batch_frexp = _patched_batch_frexp


# -----------------------------------------------------------------------
# Patch 3: IntLayerNorm.forward  —  dim_sqrt device
# -----------------------------------------------------------------------
_orig_intln_forward = _qm.IntLayerNorm.forward

def _patched_intln_forward(self, x, scaling_factor=None):
    if self.dim_sqrt is None:
        n = torch.tensor(x.shape[2], dtype=torch.float)
        self.dim_sqrt = torch.sqrt(n).to(x.device)       # ← patched
    return _orig_intln_forward(self, x, scaling_factor)

_qm.IntLayerNorm.forward = _patched_intln_forward


# -----------------------------------------------------------------------
# Patch 4: IntGELU.forward  —  sigmoid_scaling_factor device
# -----------------------------------------------------------------------
_orig_intgelu_forward = _qm.IntGELU.forward

def _patched_intgelu_forward(self, x, scaling_factor=None):
    pre_x_int = x / scaling_factor
    scaling_factor_sig = scaling_factor * 1.702

    x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
    x_int = pre_x_int - x_int_max

    exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
    exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)
    exp_int_sum = exp_int + exp_int_max

    exp_int_sum.clamp_max_(2**31 - 1)
    factor = _qu.floor_ste.apply((2**31 - 1) / exp_int_sum)
    sigmoid_int = _qu.floor_ste.apply(
        exp_int * factor / 2 ** (31 - self.output_bit + 1))
    sigmoid_scaling_factor = torch.Tensor(
        [1 / 2 ** (self.output_bit - 1)]).to(x.device)   # ← patched

    x_int = pre_x_int * sigmoid_int
    scaling_factor = scaling_factor * sigmoid_scaling_factor
    self.act_scaling_factor = scaling_factor
    return x_int * scaling_factor, scaling_factor

_qm.IntGELU.forward = _patched_intgelu_forward


# -----------------------------------------------------------------------
# Patch 5: IntSoftmax.forward  —  scaling_factor device
# -----------------------------------------------------------------------
_orig_intsoftmax_forward = _qm.IntSoftmax.forward

def _patched_intsoftmax_forward(self, x, scaling_factor):
    x_int = x / scaling_factor
    x_int_max, _ = x_int.max(dim=-1, keepdim=True)
    x_int = x_int - x_int_max

    exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
    exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

    exp_int_sum.clamp_max_(2**31 - 1)
    factor = _qu.floor_ste.apply((2**31 - 1) / exp_int_sum)
    exp_int = _qu.floor_ste.apply(
        exp_int * factor / 2 ** (31 - self.output_bit + 1))
    scaling_factor = torch.Tensor(
        [1 / 2 ** (self.output_bit - 1)]).to(x.device)   # ← patched

    self.act_scaling_factor = scaling_factor
    return exp_int * scaling_factor, scaling_factor

_qm.IntSoftmax.forward = _patched_intsoftmax_forward


# -----------------------------------------------------------------------
# Patch 6: QuantAct.forward — preserve calibrated act_scaling_factor
# -----------------------------------------------------------------------
# In original I-ViT, `self.act_scaling_factor = symmetric_linear_quantization_params`
# was placed outside `if self.running_stat:`. When running_stat is False
# (fixed / QAT mode), it recomputed scale from uninitialized min/max (0, 0),
# crushing act_scaling_factor to float32 eps (~1.19e-7) and exploding gradients.
# We move the scale calculation inside `if self.running_stat:` so calibrated
# scales are strictly preserved during QAT and inference.
_orig_quantact_forward = _qm.QuantAct.forward

def _patched_quantact_forward(self, x, pre_act_scaling_factor=None, identity=None, identity_scaling_factor=None):
    with torch.no_grad():
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if len(x_act.shape) == 4:
                x_act = x_act.permute(0, 2, 3, 1)
            v = x_act.reshape(-1, x_act.shape[-1])
            v = v.transpose(0, 1)

            cur_min = v.min(axis=1).values
            cur_max = v.max(axis=1).values
            if torch.eq(self.min_val, self.max_val).all():
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                self.min_val = self.min_val * self.act_range_momentum + \
                             cur_min * (1 - self.act_range_momentum)
                self.max_val = self.max_val * self.act_range_momentum + \
                               cur_max * (1 - self.act_range_momentum)
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

            self.act_scaling_factor = _qu.symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val)

    if pre_act_scaling_factor is None:
        quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor, False)
    else:
        quant_act_int = _qu.fixedpoint_mul.apply(
            x, pre_act_scaling_factor,
            self.activation_bit, self.quant_mode,
            self.act_scaling_factor,
            identity, identity_scaling_factor)

    correct_output_scale = self.act_scaling_factor.view(-1)
    return quant_act_int * correct_output_scale, self.act_scaling_factor

_qm.QuantAct.forward = _patched_quantact_forward


print("[PATCH] I-ViT device-compatibility patches applied (CPU/CUDA agnostic).")

