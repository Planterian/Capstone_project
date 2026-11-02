# TECHNICAL SPECIFICATION: PHASE 2 — I-ViT QUANTIZED SOFTWARE MODEL (CPU ONLY)

**Document ID:** `02_PHASE2_IVIT_SW_QUANTIZED_SPEC.md`  
**Phase:** 2 of 3 (Quantized Software Benchmark)  
**Model Architecture:** MobileViT-XXS with I-ViT Integer Architecture Surgery  
**Trained Weights:** `mobilevit_ivit_qat_best.pth` (17.7 MB, Top-1 Accuracy: 67.28% on ImageNet-1k-mini)  
**Arithmetic Precision:** Integer-Only (INT8 Weights, INT8 Activations, INT16/INT32 Accumulators)  
**Execution Target:** AMD Kria KV260 PS (Quad-core ARM Cortex-A53 @ 1.33 GHz)  
**Software Runtime:** PyTorch 2.x CPU Engine with `patch_ivit_device` monkey-patching  
**Location:** [`Project/src/Software/02_ivit_sw_mobilevit/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/02_ivit_sw_mobilevit/)  

---

## 1. Overview & Objective

The primary objective of **Phase 2** is to benchmark the software-based execution of the integer-only quantized Vision Transformer (I-ViT) integrated into MobileViT-XXS on the ARM Cortex-A53 CPU.

This phase isolates the algorithmic impact of **Quantization-Aware Training (QAT)** from physical hardware acceleration:
1. **Accuracy Integrity Check:** Verifies that the model retains its 67.28% Top-1 accuracy capability on real-world test images without numerical drift or catastrophic degradation.
2. **CPU Quantization Emulation Profile:** Measures the latency of simulating integer quantization arithmetic (bit-shifts, dyadic scaling, LUT lookups) using PyTorch CPU tensors.
3. **Reference for HW Co-Design:** Serves as the immediate predecessor to Phase 3, allowing us to compute the exact speedup gained when offloading the attention math from ARM Cortex-A53 software to the FPGA PL Core.

---

## 2. Integer Architecture Surgery & Mathematical Primitives

In standard MobileViT, the transformer blocks rely on FP32 floating-point matrix multiplications, LayerNorm (which requires floating-point division and $\sqrt{\cdot}$), and Softmax (which requires floating-point $\exp(\cdot)$).

In **I-ViT MobileViT**, all operations inside the Transformer blocks are replaced with integer-only equivalents:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               I-ViT INTEGER REPLACEMENT MODULES (INSIDE TRANSFORMER)                   │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Standard FP32 Module     │ I-ViT Integer Replacement Module                            │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ nn.Linear                │ QuantLinear (INT8 weight x INT8 act -> INT32 accum -> INT8) │
│ nn.LayerNorm             │ IntLayerNorm (Integer mean, bit-shift sqrt, dyadic scaling) │
│ nn.Softmax               │ IntSoftmax / Shiftmax (Base-2 integer exp via LUT + shift)  │
│ nn.GELU                  │ IntGELU / ShiftGELU (Polynomial integer approximation)      │
│ MatMul (Q · K^T, Attn · V)│ QuantMatMul (INT8 x INT8 -> INT32 -> Dyadic Scaled INT8)   │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### 2.1 Dyadic Scaling Formulation
In I-ViT, real-valued scaling factors $S$ are decomposed into integer multipliers (Mantissa $M$) and right-shift exponents ($E$):

$$S \approx M \cdot 2^{-E}, \quad M \in [0, 32767], \quad E \in [0, 31]$$

For Stage 2 ($N=256, d_k=16$), the calibrated dyadic parameters are:
$$M = 27640, \quad E = 22$$

### 2.2 ARM CPU Device Portability Bridge (`patch_ivit_device.py`)
Original academic I-ViT implementations contain hardcoded `.cuda()` calls inside CUDA C++ kernels. To enable execution on the ARM Cortex-A53 CPU of the KV260, we employ the dynamic device monkey-patcher:
- Replaces all `.cuda()` invocations in `SymmetricQuantFunction`, `batch_frexp`, `IntLayerNorm`, `IntGELU`, and `IntSoftmax` with dynamic `to(x.device)` routing.
- Automatically selects `device = torch.device("cpu")` on Kria Linux.

---

## 3. Implementation Details & Script Structure

The Phase 2 directory [`02_ivit_sw_mobilevit/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/02_ivit_sw_mobilevit/) contains:

### 3.1 `run_ivit_sw_kv260.py`
* **Model Builder:**
  - Loads baseline `apple/mobilevit-xx-small`.
  - Executes `replace_mobilevit_blocks(model)` to inject I-ViT integer layers into Stages 2, 3, and 4.
  - Loads fine-tuned checkpoint `mobilevit_ivit_qat_best.pth` (17.7 MB).
* **Execution & Timing:**
  - Performs inference on sample test images from `00_common/test_images/`.
  - Warmup: 10 cycles.
  - Measurement: 30 iterations.
  - Records total inference latency ($T_{\text{QAT\_SW}}$) and Stage 2 Attention latency ($T_{\text{attn\_SW}}$).
* **Prediction Extraction:**
  - Evaluates Top-1 and Top-5 class IDs and confirms concordance with the Phase 1 FP32 baseline.

### 3.2 `run_benchmark.sh`
* Automated shell wrapper setting CPU frequency governor to `performance`.
* Runs memory profiling (`Peak RSS`) and generates `phase2_metrics.json`.

---

## 4. Expected Performance & Insights on KV260

| Metric | Target Value | Notes |
| :--- | :---: | :--- |
| **Model Size on Disk** | **17.7 MB** | Quantized checkpoint |
| **Total Parameters** | **1.27 Million** | Fully preserved topology |
| **Total Inference Latency ($T_{\text{QAT\_SW}}$)** | $\approx 220 - 310\text{ ms}$ | Emulated integer math on ARM CPU |
| **Stage 2 Attention Latency ($T_{\text{attn\_SW}}$)** | $\approx 25 - 45\text{ ms}$ | Target for FPGA acceleration in Phase 3 |
| **Top-1 Accuracy Retention** | **$\ge 98\%$ of FP32** | Validated by QAT fine-tuning (67.28%) |
| **Peak RAM Consumption (RSS)** | $\approx 310\text{ MB}$ | Well within 4 GB DDR4 |

> [!NOTE]
> **Why Software Quantization on CPU Can Be Slower than Native FP32:**  
> When executing on general-purpose CPUs without dedicated INT8 matrix-multiplication hardware (such as ARMv8.2-A DotProd, which Cortex-A53 lacks), PyTorch emulates quantization by converting between integer types, performing manual bit-shifts, and applying dynamic clamping.  
> This overhead demonstrates precisely why an **FPGA domain-specific accelerator (Phase 3)** is essential to unlock the true potential of integer quantization!

---

## 5. Tracking & Future Considerations (`TODO`)

- [ ] **TODO: [Arm Compute Library (ACL)]** Explore whether integrating ARM Compute Library (ACL) INT8 GEMM provides hardware acceleration on Cortex-A53 NEON pipelines.
- [ ] **TODO: [Layer-by-Layer Timing Dump]** Add a flag `--dump-layer-timings` to produce an exact JSON breakdown of time spent in `IntLayerNorm`, `QuantLinear`, `Shiftmax`, and `IntGELU`.
- [ ] **TODO: [Quantization Error Analysis]** Compute Cosine Similarity and Mean Absolute Error (MAE) between Phase 1 FP32 logits and Phase 2 INT8 logits on the same input image.

