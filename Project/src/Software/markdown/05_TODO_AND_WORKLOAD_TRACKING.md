# WORKLOAD TRACKING, FEATURE BACKLOG & TODO MATRIX

**Document ID:** `05_TODO_AND_WORKLOAD_TRACKING.md`  
**Purpose:** Comprehensive tracking of core deliverables, nice-to-have extensions, and technical backlog  
**Project:** End-to-End HW/SW Co-Design MobileViT-XXS on AMD Kria KV260  
**Location:** [`Project/src/Software/markdown/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/markdown/)  
**Updated:** November 2026  

---

## 1. Master Workload Tracking Matrix

| Feature / Workload | Phase | Priority | Status | Target File(s) | Description |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Comprehensive Markdown Specifications** | All | **High** | **COMPLETED** | `markdown/*.md` (00 to 06) | Architectural blueprints for all 4 phases |
| **Common Dataset & ImageNet Map** | Common | **High** | **COMPLETED** | `00_common/` | ImageNet 1k classes, test image, benchmarking utils |
| **Phase 1: FP32 Baseline Runner** | 1 | **High** | **COMPLETED** | `01_baseline_mobilevit/` | Pure PyTorch Hugging Face FP32 CPU runner |
| **Phase 2: I-ViT Quantized SW Runner** | 2 | **High** | **COMPLETED** | `02_ivit_sw_mobilevit/` | I-ViT integer surgery + `mobilevit_ivit_qat_best.pth` |
| **Phase 3: Stage 2 HW Co-Design** | 3 | **High** | **COMPLETED** | `03_ivit_hw_accelerated/` | PYNQ DMA + 150 MHz PL Core on Stage 2 |
| **Phase 4: All 9 Transformers Accelerated** | 4 | **High** | **COMPLETED** | `04_all_9_transformers_hw/` | Dynamic CSR reprogram across all 9 layers (Stages 2, 3, 4) |
| **Automated 4-Phase Comparative Runner** | All | **High** | **COMPLETED** | `run_all_phases_kv260.sh` | One-command execution generating 4-phase report table |
| **One-Click KV260 Deployment Suite** | All | **High** | **COMPLETED** | `deploy_all_to_kv260.sh` | Automated sync to `/home/ubuntu/mha_eval/` on KV260 |

---

## 2. Technical Backlog & "Nice-to-Have" Extensions (`TODO` Tracker)

### 2.1 Multi-Head Attention Scaling ($H > 1$)
- [ ] **TODO: [Multi-Head Support ($H=2, H=4$)]**
  - *Context:* Current core accelerates single-head attention ($H=1$).
  - *Action:* The hardware RTL already includes `bank_sel` ping-pong and head loops. Update driver to stream multiple heads sequentially or synthesize multi-core PL instances.

### 2.2 Asynchronous DMA Pipeline
- [ ] **TODO: [Asynchronous DMA Overlapping]**
  - *Context:* Cortex-A53 waits synchronously for DMA completion (`dma.recvchannel.wait()`).
  - *Action:* Fire DMA non-blocking, compute non-dependent CPU operations, and synchronize only before context matrix reshaping.

### 2.3 Video Streaming & Camera Deployment
- [ ] **TODO: [Live Video Ingestion (USB / MIPI IAS Camera)]**
  - *Context:* Current evaluation focuses on static images from ImageNet/COCO.
  - *Action:* Add `video_demo_kv260.py` utilizing OpenCV / V4L2 to capture live 30 FPS video frames, perform real-time classification, and display FPS on HDMI or web stream.

### 2.4 Power & Energy Efficiency Measurement
- [ ] **TODO: [On-Chip Power Monitoring (PMBus / INA260)]**
  - *Context:* Comparing FPS alone does not capture energy efficiency.
  - *Action:* Read `/sys/class/hwmon/hwmon*/power1_input` on KV260 during inference to record average Wattage and compute Energy per Frame ($\text{mJ/frame}$).

### 2.5 Full Dataset Validation Sweeps
- [ ] **TODO: [ImageNet-1k Validation Mini-Batch (1,000 images)]**
  - *Context:* Single image testing proves functionality and latency.
  - *Action:* Add `eval_dataset_accuracy.py` to compute aggregate Top-1 and Top-5 accuracy percentages across all 4 phases over 1,000 images.
