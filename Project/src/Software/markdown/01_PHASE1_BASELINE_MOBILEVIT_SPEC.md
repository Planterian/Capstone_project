# TECHNICAL SPECIFICATION: PHASE 1 — BASELINE MOBILEVIT-XXS (FP32 SOFTWARE)

**Document ID:** `01_PHASE1_BASELINE_MOBILEVIT_SPEC.md`  
**Phase:** 1 of 3 (Software Baseline)  
**Model Architecture:** `apple/mobilevit-xx-small` (Hugging Face)  
**Arithmetic Precision:** IEEE 754 Floating-Point 32-bit (FP32)  
**Execution Target:** AMD Kria KV260 PS (Quad-core ARM Cortex-A53 @ 1.33 GHz)  
**Software Runtime:** PyTorch 2.x CPU Engine (OpenMP multi-threading, ARM NEON SIMD)  
**Location:** [`Project/src/Software/01_baseline_mobilevit/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/01_baseline_mobilevit/)  

---

## 1. Overview & Objective

The primary objective of **Phase 1** is to establish the reference ground-truth performance baseline. The baseline model is the unmodified, pre-trained `apple/mobilevit-xx-small` published on Hugging Face.

By executing this model in pure FP32 software on the ARM Cortex-A53 CPU of the Kria KV260, we capture:
1. **Reference Classification Accuracy:** Top-1 and Top-5 class IDs and softmax probability distribution on standard ImageNet-1k images.
2. **Computational Latency Ceiling ($T_{\text{FP32}}$):** Total end-to-end execution time per image, establishing the benchmark against which quantization (Phase 2) and FPGA hardware acceleration (Phase 3) will be evaluated.
3. **Stage-by-Stage Latency Breakdown:** Isolating the runtime of the CNN stem, the inverted residual blocks, the transformer attention layers, and the classification head.
4. **Memory Footprint:** Peak Resident Set Size (RSS) during FP32 graph evaluation.

---

## 2. Model Architecture & Pipeline Specifications

### 2.1 Model Parameters
- **Architecture Name:** MobileViT-XXS (Extra Extra Small)
- **Total Parameters:** ~1.27 Million
- **Input Resolution:** $3 \times 256 \times 256$ (RGB)
- **Number of Classes:** 1,000 (ImageNet-1k standard)
- **Pretrained Checkpoint:** Cached locally for 100% offline execution on KV260.

### 2.2 Deep Architectural Hierarchy

```text
Input Image: [1, 3, 256, 256]
  │
  ├── [Stem]: Conv-3x3, Stride=2 (Output: [1, 16, 128, 128])
  │
  ├── [Stage 1]: MobileNetV2 Inverted Residuals (Output: [1, 24, 64, 64])
  │
  ├── [Stage 2 - MobileViT Block 1]:
  │     ├── Local Rep: Conv-3x3 -> Conv-1x1
  │     ├── Unfold Tokens: Patches 2x2 -> N=256 tokens, hidden_dim=64 (H=1, dk=16)
  │     ├── Global Rep: 2x MobileViT Transformer Layers (Self-Attention + MLP)
  │     │     └── Self-Attention: Q, K, V [256 x 16], Softmax, Attn x V
  │     ├── Fold Tokens -> Feature Map
  │     └── Fusion: Conv-1x1 + Residual Add (Output: [1, 48, 32, 32])
  │
  ├── [Stage 3 - MobileViT Block 2]:
  │     ├── Local Rep: Conv-3x3 -> Conv-1x1
  │     ├── Unfold Tokens: N=64 tokens, hidden_dim=80 (H=1, dk=20)
  │     ├── Global Rep: 4x MobileViT Transformer Layers
  │     └── Fusion: Conv-1x1 + Residual Add (Output: [1, 64, 16, 16])
  │
  ├── [Stage 4 - MobileViT Block 3]:
  │     ├── Local Rep: Conv-3x3 -> Conv-1x1
  │     ├── Unfold Tokens: N=16 tokens, hidden_dim=96 (H=1, dk=24)
  │     ├── Global Rep: 3x MobileViT Transformer Layers
  │     └── Fusion: Conv-1x1 + Residual Add (Output: [1, 80, 8, 8])
  │
  ├── [Penultimate Head]: Conv-1x1 -> 320 channels + Global Average Pooling (Output: [1, 320])
  └── [Linear Classifier]: Linear(320 -> 1000) (Output: [1, 1000] Logits)
```

---

## 3. Implementation Details & Script Structure

The Phase 1 directory [`01_baseline_mobilevit/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/01_baseline_mobilevit/) contains:

### 3.1 `run_baseline_kv260.py`
* **Model Loader:**
  - Automatically checks for a local serialized checkpoint (`saved_model/` or `mobilevit_xxs_fp32.pt`).
  - If absent and internet access is available, downloads from `apple/mobilevit-xx-small` and serializes it locally for future offline runs.
* **Pre-processing:**
  - Implements torchvision transforms matching MobileViT ImageNet-1k evaluation protocol:
    $$\text{Resize}(256) \rightarrow \text{CenterCrop}(256) \rightarrow \text{ToTensor}() \rightarrow \text{Normalize}(\mu, \sigma)$$
* **Profiling Hooks:**
  - Uses `torch.profiler` or high-resolution `time.perf_counter_ns()` with custom layer hooks to record:
    * $T_{\text{stem}}$: CNN input convolutions
    * $T_{\text{stage2\_attn}}$: Stage 2 Multi-Head Attention layers (the hardware target)
    * $T_{\text{total}}$: Full network latency
* **Warmup & Measurement:**
  - Warmup: 10 cycles (ensuring instruction cache and branch predictors stabilize).
  - Measurement: 30 iterations (calculates Mean, Median, Min, Max, P95, and Std Dev).
* **Classification Output:**
  - Extracts Top-5 predicted class indices, confidence probabilities, and human-readable names from `00_common/imagenet_classes.json`.

### 3.2 `run_benchmark.sh`
* Sets Linux CPU governor to `performance`:
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```
* Binds Python execution to all 4 ARM cores using `taskset -c 0-3`:
  ```bash
  OMP_NUM_THREADS=4 taskset -c 0-3 python3 run_baseline_kv260.py --image ../00_common/test_images/sample_cat.jpg
  ```

---

## 4. Expected Performance Targets on KV260

| Metric | Target Value | Notes |
| :--- | :---: | :--- |
| **Input Shape** | $1 \times 3 \times 256 \times 256$ | Batch Size = 1 (Real-time edge inference) |
| **Total Inference Latency ($T_{\text{FP32}}$)** | $\approx 180 - 240\text{ ms}$ | Quad-core ARM Cortex-A53 @ 1.33 GHz |
| **Throughput** | $\approx 4.0 - 5.5\text{ FPS}$ | Pure software baseline |
| **Peak RAM Consumption (RSS)** | $< 350\text{ MB}$ | Well within the 4 GB DDR4 capacity |
| **Top-1 Class Confidence** | $> 80\%$ | On clear sample images (e.g. Tabby Cat) |

---

## 5. Tracking & Future Considerations (`TODO`)

- [ ] **TODO: [Optimization]** Evaluate PyTorch `torch.jit.trace` / TorchScript optimization on ARM Cortex-A53 to assess if graph fusion provides additional speedup.
- [ ] **TODO: [Batching]** Add support for batch sizes $B \in \{1, 2, 4, 8\}$ to observe DDR4 memory bandwidth saturation on KV260.
- [ ] **TODO: [Dataset Sweep]** Add automated evaluation over a 100-image validation mini-batch to compute aggregate Top-1 validation accuracy.

