# I-ViT Surgery — Deep-Dive Variable Walkthrough
## Every Variable Explained, Phase by Phase

---

## 0. MobileViT-XXS Architecture Constants

Before diving in, let's nail down the **model constants** that appear everywhere:

| Variable Name | Value | Where It Comes From | What It Means |
|---|---|---|---|
| `hidden_sizes` | `[64, 80, 96]` | HuggingFace config | The width (number of features) of each Transformer stage |
| `num_attention_heads` ($H$) | `4` | HuggingFace config | Number of parallel attention heads |
| `mlp_ratio` | `2.0` | HuggingFace config | MLP intermediate size = `hidden_size × 2.0` |
| `qkv_bias` | `True` | HuggingFace config | Whether Q/K/V projections have a bias term |
| `hidden_act` | `silu` | HuggingFace config | Activation function in MLP (replaced by ShiftGELU) |
| `layer_norm_eps` | `1e-5` | HuggingFace config | Epsilon for LayerNorm numerical stability |
| `patch_size` | `2` | HuggingFace config | Patches are 2×2 pixels → $16 \times 16 = 256$ tokens |

### Derived Variables (computed from the above)

| Variable | Formula | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| `hidden_size` ($C$) | `hidden_sizes[i]` | **64** | **80** | **96** |
| `attention_head_size` ($d_k$) | $C / H$ | **16** | **20** | **24** |
| `all_head_size` | $H \times d_k$ | 64 | 80 | 96 |
| `intermediate_size` | $C \times 2.0$ | **128** | **160** | **192** |
| $N$ (sequence length) | $16 \times 16$ | **256** | **256** | **256** |
| `scale` ($1/\sqrt{d_k}$) | `attention_head_size ** -0.5` | 0.25 | 0.2236 | 0.2041 |

> [!IMPORTANT]
> These numbers ripple through **everything** — the Python model shapes, the calibration report, and the hardware CSR registers. When you see `hidden_size=64` in Phase 1, that's the same `64` that becomes `HEAD_DIM=16` (since $64/4=16$) in the FPGA's `hw_scale_params.svh`.

---

## Phase 1 — Architecture Surgery: Variable-by-Variable

### 1.1 What Gets Replaced and Why

The goal is to swap out floating-point modules for **integer-arithmetic** versions that can run on fixed-point FPGA hardware. Here's every single replacement:

```
Original HF Module                    →  I-ViT Replacement
─────────────────────────────────────────────────────────────
nn.LayerNorm(64)                       →  IntLayerNorm(64)
nn.Linear(64, 64) [query]  ─┐
nn.Linear(64, 64) [key]    ─┼─ FUSED → QuantLinear(64, 192)  [one big QKV]
nn.Linear(64, 64) [value]  ─┘
F.softmax(dim=-1)                      →  IntSoftmax(16)      [Shiftmax]
nn.Linear(64, 64) [proj]              →  QuantLinear(64, 64)
nn.LayerNorm(64)                       →  IntLayerNorm(64)
nn.Linear(64, 128) [fc1]              →  QuantLinear(64, 128)
silu activation                        →  IntGELU()            [ShiftGELU]
nn.Linear(128, 64) [fc2]              →  QuantLinear(128, 64)
```

*(Above is for Stage 2 where `hidden_size=64`. Stages 3/4 use 80/96.)*

### 1.2 The Quantization Context Bridge — What is `act_scaling_factor`?

This is the variable that confuses most people. Here's the deal:

**Normal PyTorch:** Every layer takes a tensor and returns a tensor.
```python
x = layer(x)   # x is just a tensor, e.g. shape [B, N, C]
```

**I-ViT:** Every layer takes a **tuple** `(tensor, scaling_factor)` and returns a **tuple**.
```python
x, act_scaling_factor = layer(x, act_scaling_factor)
```

The `act_scaling_factor` is a **small scalar tensor** (often just one number) that tells you:
> "To convert this floating-point tensor to its integer representation, divide by this number."

$$x_{\text{int}} = \text{round}\left(\frac{x_{\text{float}}}{\text{act\_scaling\_factor}}\right)$$

For example, if `act_scaling_factor = 0.0107` and `x_float = 1.36`:
$$x_{\text{int}} = \text{round}(1.36 / 0.0107) = \text{round}(127.1) = 127$$

This integer `127` is what the FPGA actually computes with (fits in INT8: range $[-128, 127]$).

### 1.3 The Bridge Problem and Solution

HuggingFace's MobileViT CNN code knows nothing about `act_scaling_factor`. So the replacement block must:

1. **Entry** — Create a scaling factor from nothing:
   ```python
   # qact_input takes a raw tensor and produces (quantized_tensor, scaling_factor)
   x_quant, act_scaling_factor = self.qact_input(hidden_states)
   # hidden_states shape: [B*P, 256, 64]  (plain tensor from CNN)
   # x_quant shape:       [B*P, 256, 64]  (fake-quantized tensor)
   # act_scaling_factor:  scalar tensor, e.g. tensor(0.5345)
   ```

2. **Inside** — Pass the tuple through all I-ViT layers (they all expect it)

3. **Exit** — Strip the scaling factor and return plain tensor:
   ```python
   return x   # just the tensor, drop act_scaling_factor
   ```

### 1.4 Full Forward Pass — Every Variable, Every Shape

Let's trace a single Transformer layer (Stage 2, `hidden_size=64`):

```python
# INPUT: hidden_states shape [B*P, 256, 64] (from HF CNN pipeline)

# ──── STEP 1: Bootstrap quantization context ────
x_1, act_sf_1 = self.qact_input(hidden_states)
# x_1:      [B*P, 256, 64]    (fake-quantized version of input)
# act_sf_1: scalar ~0.534     (scaling factor for this activation)

# ──── STEP 2: Pre-Norm ────
x, act_sf = self.norm1(x_1, act_sf_1)
# IntLayerNorm: x = (x - mean) / std, but in integer arithmetic
# x:      [B*P, 256, 64]
# act_sf: scalar ~0.0107

x, act_sf = self.qact1(x, act_sf)
# Re-quantize after norm (may adjust scale)
# x:      [B*P, 256, 64]
# act_sf: scalar ~0.0107

# ──── STEP 3: Self-Attention ────
x, act_sf = self.attn(x, act_sf)
# (Details below in section 1.5)
# x:      [B*P, 256, 64]
# act_sf: scalar ~4.8e-5

# ──── STEP 4: Residual Connection (attention) ────
x_2, act_sf_2 = self.qact_res1(x, act_sf, x_1, act_sf_1)
# This computes:  x_2 = x + x_1  (the residual add)
# But both x and x_1 have DIFFERENT scaling factors!
# qact_res1 re-quantizes the sum to a common scale.
# x_2:     [B*P, 256, 64]    (16-bit precision for accumulation)
# act_sf_2: scalar ~0.002

# ──── STEP 5: Pre-Norm + MLP ────
x, act_sf = self.norm2(x_2, act_sf_2)
x, act_sf = self.qact3(x, act_sf)
x, act_sf = self.mlp(x, act_sf)
# (Details in section 1.6)
# x:      [B*P, 256, 64]
# act_sf: scalar ~1.06e-4

# ──── STEP 6: Residual Connection (MLP) ────
x, act_sf = self.qact_res2(x, act_sf, x_2, act_sf_2)
# x:      [B*P, 256, 64]
# act_sf: scalar ~0.002

# ──── STEP 7: Strip and return ────
return x   # plain tensor [B*P, 256, 64] back to HF pipeline
```

### 1.5 Inside Self-Attention — The QKV Fusion

```python
# INPUT: x shape [B*P, 256, 64], act_sf = scalar

# ── Fused QKV Projection ──
x, act_sf = self.qkv(x, act_sf)
# QuantLinear(in=64, out=192)  because 192 = 3 × 64
# x shape: [B*P, 256, 192]   ← Q, K, V are concatenated!
# act_sf:  updated scalar

x, act_sf_1 = self.qact1(x, act_sf)
# x shape: [B*P, 256, 192]

# ── Reshape into Q, K, V ──
qkv = x.reshape(B, 256, 3, 4, 16).permute(2, 0, 3, 1, 4)
# qkv shape: [3, B, 4, 256, 16]
#             ↑  ↑  ↑   ↑    ↑
#             Q/K/V  heads tokens head_dim
q, k, v = qkv[0], qkv[1], qkv[2]
# q shape: [B, 4, 256, 16]    (4 heads, 256 tokens, 16 features each)
# k shape: [B, 4, 256, 16]
# v shape: [B, 4, 256, 16]

# ── Q·K^T (Attention Scores) ──
attn, act_sf = self.matmul_1(q, act_sf_1, k.transpose(-2,-1), act_sf_1)
# Matrix multiply: [B,4,256,16] × [B,4,16,256] = [B,4,256,256]
# attn shape: [B, 4, 256, 256]    ← each token attends to all 256 tokens
# act_sf:     updated (product of two scaling factors)

# ── Scale by 1/√dk ──
attn = attn * self.scale       # self.scale = 1/√16 = 0.25
act_sf = act_sf * self.scale   # scaling factor tracks the same multiplication!

attn, act_sf = self.qact_attn1(attn, act_sf)
# Re-quantize attention scores to 8-bit range

# ── Shiftmax (Integer Softmax) ──
attn, act_sf = self.int_softmax(attn, act_sf)
# Converts scores to probabilities using integer-only arithmetic
# Uses log2-based shift approximation of exp()
# attn shape: [B, 4, 256, 256]   (probabilities, fake-quantized)

# ── Attn × V (Context) ──
x, act_sf = self.matmul_2(attn, act_sf, v, act_sf_1)
# [B,4,256,256] × [B,4,256,16] = [B,4,256,16]
x = x.transpose(1,2).reshape(B, 256, 64)
# x shape: [B, 256, 64]   (heads concatenated back)

# ── Output Projection ──
x, act_sf = self.qact2(x, act_sf)
x, act_sf = self.proj(x, act_sf)     # QuantLinear(64, 64)
x, act_sf = self.qact3(x, act_sf)    # 16-bit output quantization
# x shape: [B*P, 256, 64]
# act_sf:  scalar ~4.8e-5
```

### 1.6 Inside MLP

```python
# INPUT: x shape [B*P, 256, 64], act_sf = scalar

x, act_sf = self.fc1(x, act_sf)       # QuantLinear(64, 128)  expand
# x shape: [B*P, 256, 128]

x, act_sf = self.qact_gelu(x, act_sf) # re-quantize before activation
x, act_sf = self.act(x, act_sf)       # IntGELU (ShiftGELU)
# Approximates GELU using integer shift operations
# x shape: [B*P, 256, 128]

x, act_sf = self.qact1(x, act_sf)     # re-quantize after activation
x, act_sf = self.fc2(x, act_sf)       # QuantLinear(128, 64)  shrink
# x shape: [B*P, 256, 64]

x, act_sf = self.qact2(x, act_sf)     # 16-bit output
# x shape: [B*P, 256, 64]
# act_sf:  scalar ~1.06e-4
```

### 1.7 QKV Weight Fusion — The Actual Numbers

When transferring weights from HuggingFace to I-ViT, three separate weight matrices get concatenated:

```python
# Original HF model has:
#   query.weight: shape [64, 64]    (64 output features, 64 input features)
#   key.weight:   shape [64, 64]
#   value.weight: shape [64, 64]

# I-ViT needs:
#   qkv.weight:   shape [192, 64]   (192 = 3 × 64)
#                  rows 0:64   = query.weight
#                  rows 64:128 = key.weight
#                  rows 128:192 = value.weight

# Same for biases:
#   query.bias: [64]  ─┐
#   key.bias:   [64]  ─┼─ cat → qkv.bias: [192]
#   value.bias: [64]  ─┘
```

---

## Phase 2 — PTQ Calibration: Every Number Explained

### 2.1 What is `QuantAct` Doing?

`QuantAct` is a **monitoring node** placed between layers. During calibration, it watches activations flow through and records their range. Think of it as a voltmeter measuring the signal range.

**Key variables inside each QuantAct:**

| Variable | Type | What It Stores |
|---|---|---|
| `running_stat` | `bool` | `True` = actively tracking, `False` = locked |
| `min_val` | `tensor` | Smallest activation value seen (smoothed) |
| `max_val` | `tensor` | Largest activation value seen (smoothed) |
| `activation_bit` | `int` | 8 or 16 (quantization bit width) |
| `act_range_momentum` | `float` | 0.95 (EMA smoothing factor) |
| `act_scaling_factor` | `tensor` | The computed quantization scale |

### 2.2 EMA Update — Concrete Example

Suppose we're at `qact1` in Stage 2, layer 0. The observer tracks post-LayerNorm activations:

```
Batch 1 (first batch):
  batch_min = -1.35    batch_max = 1.42
  → min_val = -1.35    (directly assigned on first batch)
  → max_val = 1.42

Batch 2:
  batch_min = -1.28    batch_max = 1.38
  → min_val = 0.95 × (-1.35) + 0.05 × (-1.28) = -1.2825 + -0.064 = -1.3465
  → max_val = 0.95 × 1.42    + 0.05 × 1.38    = 1.349   + 0.069  = 1.418

Batch 3:
  batch_min = -1.31    batch_max = 1.40
  → min_val = 0.95 × (-1.3465) + 0.05 × (-1.31) = -1.27918 + -0.0655 = -1.3447
  → max_val = 0.95 × 1.418     + 0.05 × 1.40    = 1.3471   + 0.07    = 1.4171

...after 32 batches (1024 images)...

Final values (from your calibration_scale_report.json):
  min_val = -1.2687
  max_val = 1.3608
```

### 2.3 Scale Computation — The Formula

Once calibration finishes, `min_val` and `max_val` are locked. The scaling factor is computed:

$$\text{act\_scaling\_factor} = \frac{\max(|\text{min\_val}|, |\text{max\_val}|)}{2^{b-1} - 1}$$

For 8-bit quantization ($b = 8$):

$$\text{act\_scaling\_factor} = \frac{\max(1.2687, 1.3608)}{2^7 - 1} = \frac{1.3608}{127} = 0.01072$$

This is **exactly** the `0.010715` you see in your [`calibration_scale_report.json`](file:///c:/Users/ADMIN/OneDrive/Desktop/Capstone_project_git/model_notebook/ivit_surgery/calibration_scale_report.json) for `qact1` of layer 0!

For 16-bit quantization ($b = 16$), the denominator is $2^{15} - 1 = 32767$:

$$\text{act\_scaling\_factor}_{\text{16-bit}} = \frac{\max(65.65, 64.79)}{32767} = \frac{65.65}{32767} = 0.002004$$

This matches `qact_res1`'s value of `0.002004` in the JSON.

### 2.4 Dyadic Decomposition — What Are `M` and `E`?

The FPGA can't do floating-point division. So every scaling factor gets decomposed into:

$$S = M \times 2^{-E}$$

where:
- $M$ = integer mantissa (fits in ~31 bits)
- $E$ = shift amount (how many bits to right-shift)

The function `batch_frexp()` does this. Let's work through a real example:

```
For qact1 (layer 0):
  act_scaling_factor = 0.010715

  Step 1: numpy.frexp(0.010715) returns:
    mantissa = 0.5487 (normalized to [0.5, 1.0))
    exponent = -6     (so 0.010715 ≈ 0.5487 × 2^(-6))

  Step 2: Scale mantissa to integer:
    M = round(0.5487 × 2^31) = round(1,178,599,424) ≈ 1,472,696,704

  Step 3: Compute E:
    E = 31 - (-6) = 37

  Verification:
    M × 2^(-E) = 1,472,696,704 × 2^(-37) = 0.01072 ✓
```

So in your JSON: `"mantissa_M": 1472696704, "exponent_E": 37.0` — this is exactly what the FPGA uses!

### 2.5 The 12 Observers Per Layer — What Each One Watches

```
┌─────────────────────────────────────────────────────────┐
│  Input from CNN                                         │
│    ↓                                                    │
│  ① qact_input (8-bit) ← "How big are the raw inputs?"  │
│    ↓                                                    │
│  IntLayerNorm                                           │
│    ↓                                                    │
│  ② qact1 (8-bit) ← "How big are post-norm values?"     │
│    ↓                                                    │
│  ┌── Self-Attention ──────────────────────────────────┐ │
│  │  QKV projection                                    │ │
│  │    ↓                                               │ │
│  │  ③ attn.qact1 (8-bit) ← "Q/K/V activation range"  │ │
│  │    ↓                                               │ │
│  │  Q·K^T matmul + scale                              │ │
│  │    ↓                                               │ │
│  │  ④ attn.qact_attn1 (8-bit) ← "Score range"         │ │
│  │    ↓                                               │ │
│  │  Shiftmax (Integer Softmax)                        │ │
│  │    ↓                                               │ │
│  │  Attn × V matmul + reshape                         │ │
│  │    ↓                                               │ │
│  │  ⑤ attn.qact2 (8-bit) ← "Context vector range"     │ │
│  │    ↓                                               │ │
│  │  Output projection                                 │ │
│  │    ↓                                               │ │
│  │  ⑥ attn.qact3 (16-bit) ← "Projected output range"  │ │
│  └────────────────────────────────────────────────────┘ │
│    ↓                                                    │
│  ⑦ qact_res1 (16-bit) ← "After residual add"           │
│    ↓  (+x_1 residual)                                   │
│  IntLayerNorm                                           │
│    ↓                                                    │
│  ⑧ qact3 (8-bit) ← "Pre-MLP range"                     │
│    ↓                                                    │
│  ┌── MLP ─────────────────────────────────────────────┐ │
│  │  fc1 (expand 64→128)                               │ │
│  │    ↓                                               │ │
│  │  ⑨ mlp.qact_gelu (8-bit) ← "Pre-activation range"  │ │
│  │    ↓                                               │ │
│  │  ShiftGELU                                         │ │
│  │    ↓                                               │ │
│  │  ⑩ mlp.qact1 (8-bit) ← "Post-activation range"     │ │
│  │    ↓                                               │ │
│  │  fc2 (shrink 128→64)                               │ │
│  │    ↓                                               │ │
│  │  ⑪ mlp.qact2 (16-bit) ← "MLP output range"         │ │
│  └────────────────────────────────────────────────────┘ │
│    ↓                                                    │
│  ⑫ qact_res2 (16-bit) ← "After MLP residual add"       │
│    ↓  (+x_2 residual)                                   │
│  Output to next layer                                   │
└─────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Why are some 16-bit and others 8-bit?** The residual-add points (⑦, ⑫) and projection outputs (⑥, ⑪) use 16-bit because accumulation results have wider range. In the FPGA, the systolic array produces INT32 accumulators that get scaled down. The 16-bit observers preserve more precision at these critical boundaries.

---

## Phase 3 — QAT: The Bug and the Fix

### 3.1 The QuantAct Scale Overwrite Bug — Step by Step

This is the most critical engineering story. Here's exactly what happened:

**Original I-ViT code** (`QuantAct.forward()`, line 191):
```python
def forward(self, x, ...):
    # ... observer update code (only runs if self.running_stat == True) ...

    # BUG: This line runs UNCONDITIONALLY, even when running_stat=False!
    self.act_scaling_factor = symmetric_linear_quantization_params(
        self.activation_bit, self.min_val, self.max_val
    )
```

**The disaster sequence:**

```
1. Phase 2 calibration finishes.
   - qact1.act_scaling_factor = 0.0107 ✓ (correct!)
   - qact1.min_val = -1.27  ✓
   - qact1.max_val = 1.36   ✓

2. Phase 3 QAT starts. Load calibrated checkpoint.
   But min_val and max_val are NOT in state_dict!
   (They were plain Python attributes, not registered buffers)
   - qact1.min_val = 0.0  ← RESET TO ZERO!
   - qact1.max_val = 0.0  ← RESET TO ZERO!
   - qact1.act_scaling_factor = 0.0107  ← loaded correctly from checkpoint

3. First forward pass (Batch 1):
   The bug line computes:
   symmetric_linear_quantization_params(8, min=0.0, max=0.0)
   → max_abs = max(|0|, |0|) = 0.0
   → clamp to eps = 1.19e-7
   → act_scaling_factor = 1.19e-7 / 127 = 9.37e-10

   The carefully calibrated 0.0107 is OVERWRITTEN with 9.37e-10!

4. Forward pass uses this tiny scale:
   x_int = x / 9.37e-10 ≈ x × 1,000,000,000
   → Values explode to ~10^9

5. Backward pass:
   Gradients are proportional to (1/scale)³ through the chain
   → Gradient ≈ (1 / 9.37e-10)³ ≈ 10^{27}
   → float32 overflows at ~3.4 × 10^{38}
   → Gradient = inf → NaN

6. Batch 2: "ValueError: cannot convert NaN to integer" 💥
```

**The fix** (in `patch_ivit_device.py`):
```python
def _patched_quantact_forward(self, x, ...):
    if self.running_stat:         # ← ONLY update when observing
        # ... update min_val, max_val ...
        self.act_scaling_factor = symmetric_linear_quantization_params(...)
    # When running_stat=False, act_scaling_factor is PRESERVED from checkpoint
```

### 3.2 What is STE (Straight-Through Estimator)?

During QAT, the forward pass does fake quantization:

$$x_q = \text{round}(x / s) \times s$$

The `round()` function has zero gradient everywhere (it's a step function). STE solves this by pretending the gradient of `round()` is 1:

$$\frac{\partial x_q}{\partial x} \approx 1 \quad \text{(STE assumption)}$$

This lets gradients flow back through quantization points during backpropagation, so the weights can learn to be "quantization-friendly."

### 3.3 Differential Learning Rates — Why?

| Parameter Group | LR | Rationale |
|---|---|---|
| Transformer weights | `1e-4` | These are the newly quantized layers. They need to adapt to the quantization noise introduced by Shiftmax, ShiftGELU, etc. |
| CNN stem | `1e-6` | Apple's pretrained CNN features are already good. A tiny LR prevents "catastrophic forgetting" — destroying good features while fixing quantization issues. |
| Classifier head | `1e-4` | The final classification layer needs to adjust to the slightly different feature distributions coming from the quantized Transformer. |

---

## Phase 4 — RTL Hardware: Every Signal Traced

### 4.1 What the FPGA Actually Computes

The FPGA **only** accelerates the attention core math:

$$\text{Context} = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

Everything else (LayerNorm, QKV projection, MLP, residual adds) runs on the ARM CPU.

### 4.2 Hardware Variables — Signal by Signal

#### In `scale_unit.sv` — The Dyadic Rescaler

```
┌────────────────────────────────────────────────────────────┐
│  INPUTS:                                                    │
│    in_score [31:0]  ← INT32 accumulator result from GEMM   │
│                       (e.g., result of Q·K^T dot product)  │
│    scale_m [15:0]   ← 16-bit signed mantissa M             │
│                       Loaded from AXI-Lite CSR by ARM CPU  │
│                       Stage 2: 27640                        │
│                       Stage 3: 21936                        │
│                       Stage 4: 18810                        │
│    scale_e [4:0]    ← 5-bit shift amount E                 │
│                       Stage 2: 22                           │
│                       Stage 3: 22                           │
│                       Stage 4: 22                           │
│    in_valid         ← 1-bit: "input data is ready"         │
│                                                             │
│  PIPELINE STAGE 1 (1 clock cycle):                          │
│    prod_stage [47:0] = in_score × scale_m                  │
│    (48-bit to prevent overflow: 32-bit × 16-bit = 48-bit)  │
│                                                             │
│    Example (Stage 2, dk=16):                                │
│      in_score = 5000 (from Q·K^T accumulation)             │
│      scale_m = 27640                                        │
│      prod_stage = 5000 × 27640 = 138,200,000               │
│                                                             │
│  PIPELINE STAGE 2 (1 clock cycle):                          │
│    shifted_score [47:0] = prod_stage >>> scale_e            │
│    (arithmetic right shift = divide by 2^E)                 │
│                                                             │
│      shifted_score = 138,200,000 >>> 22                     │
│                    = 138,200,000 / 4,194,304                │
│                    = 32  (integer division)                  │
│                                                             │
│    clamped_score [15:0] = clamp(shifted_score, -32768, 32767)│
│      32 fits in INT16, so clamped_score = 32                │
│                                                             │
│  OUTPUT:                                                    │
│    scaled_score [15:0] = 32 (INT16)                         │
│    out_valid            = 1                                  │
└────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **Why 48-bit intermediate?** Because $\text{INT32} \times \text{INT16} = \text{INT48}$. If you used only 32 bits, the multiplication would overflow. After the shift, the result fits back into INT16.

#### In `softmax_lut.sv` — The Integer Softmax

```
INPUT:  scaled_score [15:0] for each of 256 tokens (from scale_unit)

STEP 1: Find max score
  max_score = max(score[0], score[1], ..., score[255])
  (Needed to prevent overflow in exp())

STEP 2: Subtract max and look up exp() from ROM
  For each token i:
    diff[i] = max_score - score[i]    (always ≥ 0)
    exp_val[i] = ROM[diff[i]]         (256-entry lookup table)

  ROM contents:
    ROM[0]   = round(65535 × e^(0/32))     = 65535  (= max probability)
    ROM[1]   = round(65535 × e^(-1/32))    = 63521
    ROM[32]  = round(65535 × e^(-32/32))   = 24109  (= e^(-1) × 65535)
    ROM[255] = round(65535 × e^(-255/32))  = 0      (≈ 0, negligible)

STEP 3: Sum all exp values
  sum_exp = exp_val[0] + exp_val[1] + ... + exp_val[255]
  (can be up to 256 × 65535 = 16,776,960)

STEP 4: Divide (24-step non-restoring iterative divider)
  For each token i:
    prob[i] = (exp_val[i] × 256) / sum_exp
  Result is UINT8 (0..255), where 255 ≈ probability 1.0

OUTPUT: prob[0..255], each UINT8
```

#### In `pe_slice_dsp_packing.sv` — The DSP Multiply Unit

This is where INT8 × INT8 multiplication happens inside the systolic array:

```
INPUTS:
  operand_a [7:0]  ← INT8 signed (e.g., Q value = -43)
  operand_b [7:0]  ← INT8 signed (e.g., K value = 17)
  operand_c [7:0]  ← INT8 signed (e.g., second K value, packed)

SIGN EXTENSION (critical fix):
  operand_a_extended [17:0] = {10{operand_a[7]}, operand_a}
  If operand_a = -43 (binary: 11010101)
    bit [7] = 1 (negative)
    extended = 11_1111_1111_1101_0101 (18-bit signed = still -43)

  (The OLD code did {{10{operand_a}}} which replicated ALL 8 bits
   10 times — completely wrong for sign extension!)

DSP PACKING:
  Two multiplications in one DSP48E2 slice:
  packed = (operand_b << 18) + operand_c

  dsp_product = operand_a_extended × packed
  
  result_ab = upper bits → A × B
  result_ac = lower bits → A × C

  This gives 2 INT8 multiplies per DSP slice per clock cycle!
```

#### In `systolic_mac_array.sv` — The Matrix Engine

```
ARRAY_SIZE = 16 (for Stage 2 with dk=16)

The array is a 16×16 grid of PE slices:
  
  Q tokens flow LEFT → RIGHT (one per cycle, staggered)
  K tokens flow TOP → BOTTOM (one per cycle, staggered)

  Each PE computes: accumulator += Q[i] × K[j]
  
  After 16 cycles: accumulator[r][c] = Σ Q[r,k] × K[c,k] for k=0..15
                                      = dot product of row r of Q with row c of K
                                      = one element of the Q·K^T matrix!

  Result: acc_matrix [31:0] for each (row, col) pair
  
  THE FIX: Old code truncated to [7:0] (lost upper 24 bits!)
  New code keeps all 32 bits → feeds into scale_unit
```

### 4.3 CSR Registers — What the ARM CPU Programs

Before triggering the FPGA, the ARM CPU writes configuration via AXI-Lite:

| Register | Address | Stage 2 Value | What It Sets |
|---|---|---|---|
| `CSR_SEQ_LEN` | `0x10` | 256 | Number of tokens ($N$) |
| `CSR_HEAD_DIM` | `0x14` | 16 | Features per head ($d_k$) |
| `CSR_NUM_HEAD` | `0x18` | 4 | Number of heads ($H$) |
| `CSR_SCALE_M` | `0x20` | 27640 | Dyadic mantissa for $1/\sqrt{d_k}$ scale |
| `CSR_SCALE_E` | `0x24` | 22 | Dyadic shift for $1/\sqrt{d_k}$ scale |
| `CSR_CTRL` | `0x00` | `0x01` | START pulse — triggers the FSM |

Then the ARM streams Q, K, V matrices (each $256 \times d_k$ INT8 values) via AXI-Stream.

### 4.4 Sequential Head FSM — Why Loop Instead of Parallel

The KV260 has limited resources. Four parallel attention heads would need:

| Resource | Per Head | × 4 Heads | KV260 Available | Usage |
|---|---|---|---|---|
| DSP48E2 slices | ~40 | 160 | 1248 | 12.8% |
| BRAM36K | 12 | 48 | 144 | **33.3%** |
| LUT | ~8000 | 32000 | 117120 | 27.3% |

With the sequential approach (1 head, looped 4 times):

| Resource | 1 Head | KV260 Available | Usage |
|---|---|---|---|
| DSP48E2 | ~40 | 1248 | **3.2%** |
| BRAM36K | 12 | 144 | **8.3%** |
| LUT | ~8000 | 117120 | **6.8%** |

The tradeoff is 4× latency for one attention layer, but **70%+ resource savings**, leaving room for the rest of the SoC design.

---

## Complete Variable Glossary

### Python-Side (Phases 1-3)

| Variable | Where | Type | Example Value | Meaning |
|---|---|---|---|---|
| `hidden_size` / $C$ | Surgery | int | 64, 80, 96 | Feature dimension per stage |
| `num_attention_heads` / $H$ | Surgery | int | 4 | Parallel attention heads |
| `attention_head_size` / $d_k$ | Surgery | int | 16, 20, 24 | Features per head ($C/H$) |
| `intermediate_size` | Surgery | int | 128, 160, 192 | MLP hidden dimension ($C \times 2$) |
| `act_scaling_factor` | QuantAct | tensor | 0.0107 | Float-to-integer conversion scale |
| `min_val` | QuantAct | tensor | -1.269 | Observed activation minimum |
| `max_val` | QuantAct | tensor | 1.361 | Observed activation maximum |
| `activation_bit` | QuantAct | int | 8 or 16 | Quantization bit width |
| `act_range_momentum` | QuantAct | float | 0.95 | EMA smoothing factor |
| `mantissa_M` / $M$ | Calibration | int | 1,472,696,704 | Dyadic integer multiplier |
| `exponent_E` / $E$ | Calibration | float | 37.0 | Dyadic bit-shift amount |
| `running_stat` | QuantAct | bool | True/False | Observer active or locked |
| `fc_scaling_factor` | QuantLinear | tensor | varies | Weight quantization scale |
| `weight_integer` | QuantLinear | tensor | INT8 values | Quantized weight matrix |

### Hardware-Side (Phase 4)

| Signal | Module | Width | Example | Meaning |
|---|---|---|---|---|
| `in_score` | scale_unit | 32-bit signed | 5000 | Raw GEMM accumulator output |
| `scale_m` | scale_unit | 16-bit signed | 27640 | Dyadic mantissa from CSR |
| `scale_e` | scale_unit | 5-bit unsigned | 22 | Dyadic shift from CSR |
| `prod_stage` | scale_unit | 48-bit signed | 138,200,000 | Intermediate product |
| `shifted_score` | scale_unit | 48-bit signed | 32 | After arithmetic right shift |
| `clamped_score` | scale_unit | 16-bit signed | 32 | Clamped to INT16 range |
| `cnt_reg` | softmax_lut | 9-bit | 0–255 | Token counter |
| `exp_val` | softmax_lut | 16-bit unsigned | 0–65535 | Exponential lookup value |
| `sum_exp` | softmax_lut | 32-bit unsigned | ~16M | Sum of all exp values |
| `prob` | softmax_lut | 8-bit unsigned | 0–255 | Softmax probability (UINT8) |
| `operand_a` | pe_slice | 8-bit signed | -43 | First INT8 operand (Q or S) |
| `operand_b` | pe_slice | 8-bit signed | 17 | Second INT8 operand (K or V) |
| `pe_acc_matrix` | systolic | 32-bit signed | varies | Per-PE accumulator |
| `head_cnt` | attention_core | 3-bit | 0–3 | Current head index in FSM |
| `core_done` | attention_core | 1-bit | 0/1 | Interrupt flag to ARM CPU |
