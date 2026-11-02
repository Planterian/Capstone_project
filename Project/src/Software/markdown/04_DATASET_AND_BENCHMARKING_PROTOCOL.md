# TECHNICAL SPECIFICATION: DATASET PREPARATION & BENCHMARKING PROTOCOL ON STANDALONE KV260

**Document ID:** `04_DATASET_AND_BENCHMARKING_PROTOCOL.md`  
**Purpose:** Standardized Image Ingestion, Class Mapping, and Benchmarking Harness for Standalone KV260  
**Target Hardware:** AMD Kria KV260 Vision AI Starter Kit  
**Location:** [`Project/src/Software/markdown/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/markdown/)  

---

## 1. Overview & Image Repository Structure

To evaluate the MobileViT models across all three phases (FP32 Baseline, I-ViT Quantized Software, and HW/SW Co-Design), a standardized image data pipeline is deployed on the KV260.

On the KV260 board, the test images reside in a dedicated directory:
```text
/home/ubuntu/test_images/
├── sample_cat.jpg              # ImageNet class: tabby, tabby cat (ID: 281)
├── sample_dog.jpg              # ImageNet class: Golden Retriever (ID: 207)
├── sample_sports_car.jpg       # ImageNet class: sports car (ID: 817)
└── imagenet_mini_val/          # (Optional) Mini-validation batch for statistical evaluation
```

Locally in the repository, these assets are staged in:
[`Project/src/Software/00_common/test_images/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/00_common/test_images/)

---

## 2. Image Preprocessing & Normalization Protocol

To ensure 100% numerical consistency across all three phases, input images undergo identical preprocessing:

```text
Raw Image (.jpg / .png)
       │
       ▼
 [RGB Conversion]  (PIL Image.convert("RGB"))
       │
       ▼
 [Resize & Crop]   (Resize to 256x256 via Bilinear Interpolation, CenterCrop 256x256)
       │
       ▼
 [Tensor Scaling]  (Scale pixel values to [0.0, 1.0])
       │
       ▼
 [Normalization]   (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
       │
       ▼
 [Batch Dimension] (Unsqueeze to shape [1, 3, 256, 256])
```

Mathematical formulation:
$$X_{\text{norm}}[c, h, w] = \frac{\frac{X_{\text{raw}}[c, h, w]}{255.0} - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}$$

---

## 3. ImageNet-1k Class Label Integration

The 1,000 class names are indexed in [`00_common/imagenet_classes.json`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/00_common/imagenet_classes.json):
```json
{
  "0": "tench, Tinca tinca",
  "207": "golden retriever",
  "281": "tabby, tabby cat",
  "817": "sports car, sport car",
  "...": "..."
}
```

Every inference runner extracts:
- **Top-1 Prediction:** Class ID, Human-readable Name, Softmax Probability / Logit.
- **Top-5 Predictions:** Top 5 highest scoring classes to evaluate model degradation across quantization boundaries.

---

## 4. Benchmarking Methodology & Statistical Rigor

### 4.1 Environmental Control on KV260 Linux
Before running benchmarks, dynamic CPU frequency scaling must be locked to prevent frequency throttling:
```bash
# Force ARM Cortex-A53 CPU to maximum nominal clock (1.33 GHz)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

### 4.2 Standardized Profiling Protocol (`benchmark_utils.py`)
Each benchmark adheres to the following sequence:

1. **Model Initialization:**
   - Load weights, initialize PYNQ overlays, and pre-allocate DDR4 CMA buffers.
2. **Warmup Phase ($N_{\text{warmup}} = 10$):**
   - Run 10 inference passes without recording metrics.
   - Purpose: Prime L1/L2 instruction and data caches, warm up DRAM controllers, and trigger PyTorch JIT execution graphs.
3. **Timed Measurement Phase ($N_{\text{runs}} = 30$):**
   - Record start timestamp using high-resolution monotonic timer:
     $$T_{\text{start}} = \text{time.perf\_counter\_ns}()$$
   - Execute forward pass.
   - Record stop timestamp:
     $$T_{\text{end}} = \text{time.perf\_counter\_ns}()$$
   - Calculate latency:
     $$\Delta t_i = \frac{T_{\text{end}} - T_{\text{start}}}{10^6} \quad (\text{ms})$$
4. **Statistical Aggregation:**
   - **Mean Latency:** $\mu = \frac{1}{N} \sum_{i=1}^N \Delta t_i$
   - **Median Latency:** $\tilde{\mu} = \text{median}(\{\Delta t_i\})$
   - **Standard Deviation:** $\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (\Delta t_i - \mu)^2}$
   - **P95 Latency:** 95th percentile (tail latency)
   - **Throughput:** $\text{FPS} = \frac{1000}{\mu_{\text{ms}}}$
5. **Memory Footprint:**
   - **Peak RSS:** $\text{RAM}_{\text{peak}} = \frac{\text{ru\_maxrss}}{1024} \quad (\text{MB})$

---

## 5. Automated Reporting Formats

### 5.1 JSON Output (`phaseX_metrics.json`)
```json
{
  "phase": "Phase 1 - Baseline FP32",
  "model": "apple/mobilevit-xx-small",
  "device": "ARM Cortex-A53 (4 Cores)",
  "precision": "FP32",
  "num_iterations": 30,
  "mean_latency_ms": 210.45,
  "median_latency_ms": 209.80,
  "p95_latency_ms": 218.12,
  "throughput_fps": 4.75,
  "peak_rss_mb": 312.4,
  "top1_class_id": 281,
  "top1_class_name": "tabby, tabby cat",
  "top1_confidence": 0.8924
}
```

### 5.2 Consolidated Comparison Table (`BENCHMARK_SUMMARY.md`)
Generated automatically by `run_all_phases_kv260.sh` to provide thesis-ready data.

