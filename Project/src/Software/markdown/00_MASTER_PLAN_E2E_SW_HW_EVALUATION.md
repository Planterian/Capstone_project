# MASTER PLAN: END-TO-END SW-HW CO-DESIGN EVALUATION ON STANDALONE AMD KRIA KV260

**Document ID:** `00_MASTER_PLAN_E2E_SW_HW_EVALUATION.md`  
**Author:** Lead Embedded HW/SW Co-Design & VLSI Verification Engineer  
**Project:** MobileViT-XXS + I-ViT Multi-Head Attention Hardware Accelerator  
**Target Hardware:** AMD Kria KV260 Vision AI Starter Kit (`xck26-sfvc784-2LV-c`)  
**Processing System (PS):** Quad-core ARM Cortex-A53 @ 1.33 GHz, 4 GB 64-bit DDR4-2400  
**Programmable Logic (PL):** Attention Core Variant 2A (`attention_core_top_2A`) @ 150.000 MHz  
**Software Stack:** Ubuntu 22.04 LTS (AArch64), Linux Kernel 5.15, PYNQ v3.0.1, PyTorch 2.x  
**Location:** [`Project/src/Software/markdown/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/markdown/)  
**Updated:** November 2026 (Added Phase 4: Full 9-Transformer Network Acceleration)  

---

## 1. Executive Summary & Research Objectives

In modern embedded AI systems, evaluating a domain-specific accelerator (DSA) requires a rigorous comparative benchmark against pure software baselines running on the **exact same physical silicon**. This experimental setup eliminates confounding variables such as differences in host CPU microarchitecture, memory bus frequencies, compiler optimizations, or operating system daemons.

This master plan establishes a complete, automated, **four-phase evaluation framework** deployed directly on the standalone AMD Kria KV260:

```text
==================================================================================================
                 STANDALONE AMD KRIA KV260 FOUR-PHASE EVALUATION SPECTRUM
==================================================================================================

  [PHASE 1: FP32 SOFTWARE BASELINE]
  Hugging Face Apple MobileViT-XXS (FP32)
  ├── Execution: 100% ARM Cortex-A53 CPU (PS)
  ├── Math Engine: PyTorch standard FP32 floating point (OpenMP / NEON SIMD)
  └── Purpose: Establish ground-truth classification accuracy and baseline latency ceiling.

  [PHASE 2: INT8/INT16 QUANTIZED SOFTWARE (I-ViT)]
  MobileViT-XXS with I-ViT Transformer Surgery + QAT Checkpoint (mobilevit_ivit_qat_best.pth)
  ├── Execution: 100% ARM Cortex-A53 CPU (PS) via CPU device patching
  ├── Math Engine: Integer-only arithmetic (Shiftmax, ShiftGELU, IntLayerNorm, QuantLinear)
  └── Purpose: Measure the speedup and accuracy impact of quantization ALONE without FPGA hardware.

  [PHASE 3: PARTITIONED HW/SW CO-DESIGN (STAGE 2 PL ACCELERATED)]
  MobileViT-XXS with Stage 2 Attention Offloaded to PL Core
  ├── Execution: Hybrid PS (ARM Cortex-A53) + PL (FPGA Accelerator @ 150 MHz)
  ├── Interconnect: 128-bit AXI DMA (MM2S & S2MM) + AXI4-Lite CSR Control Bus
  ├── Math Engine:
  │   ├── Stage 2 Attention (N=256, dk=16) -> Accelerated on FPGA PL Core (150 MHz)
  │   └── Stages 3 & 4 Attention -> Executed via Software Fallback on ARM PS
  └── Purpose: Quantify the isolated hardware acceleration factor of the primary Stage 2 bottleneck.

  [PHASE 4: FULL-NETWORK ATTENTION ACCELERATION (ALL 9 TRANSFORMERS ON PL)]
  MobileViT-XXS with ALL 9 Transformer Layers Offloaded to PL Core
  ├── Execution: Hybrid PS (ARM Cortex-A53) + PL (FPGA Accelerator @ 150 MHz)
  ├── Interconnect: Dynamic CSR Reprogramming per layer + Unified Zero-Copy CMA DDR4 Buffers
  ├── Math Engine:
  │   ├── Stage 2 (2 layers): N=256, dk=16 (M=27640, E=22) -> Offloaded to PL Core
  │   ├── Stage 3 (4 layers): N=64,  dk=20 (M=22559, E=22) -> Offloaded to PL Core
  │   └── Stage 4 (3 layers): N=16,  dk=24 (M=19947, E=22) -> Offloaded to PL Core
  └── Purpose: Achieve maximum Amdahl speedup across the entire 9-transformer network topology.
==================================================================================================
```

---

## 2. Directory Architecture in `Project/src/Software/`

```text
Project/src/Software/
├── markdown/                                       # Master documentation, specifications, and reports
│   ├── 00_MASTER_PLAN_E2E_SW_HW_EVALUATION.md      # This master blueprint
│   ├── 01_PHASE1_BASELINE_MOBILEVIT_SPEC.md        # Phase 1 technical specification
│   ├── 02_PHASE2_IVIT_SW_QUANTIZED_SPEC.md         # Phase 2 technical specification
│   ├── 03_PHASE3_IVIT_HW_ACCELERATED_SPEC.md       # Phase 3 technical specification
│   ├── 04_DATASET_AND_BENCHMARKING_PROTOCOL.md     # Image loading, metrics & test protocol
│   ├── 05_TODO_AND_WORKLOAD_TRACKING.md            # Future feature tracking & backlog
│   └── 06_PHASE4_ALL_9_TRANSFORMERS_ACCELERATED_SPEC.md # Phase 4 technical specification
│
├── 00_common/                                      # Shared dataset assets, label dictionaries & helpers
│   ├── test_images/                                # Sample images (ImageNet-1k / COCO)
│   │   └── sample_infer.jpg
│   ├── imagenet_classes.json                       # 1,000 ImageNet-1k human-readable class names
│   └── benchmark_utils.py                          # High-precision timer, RSS memory, warmup & reporting
│
├── 01_baseline_mobilevit/                          # PHASE 1: Pure Hugging Face FP32 Baseline
│   ├── run_baseline_kv260.py
│   ├── run_benchmark.sh
│   └── README.md
│
├── 02_ivit_sw_mobilevit/                           # PHASE 2: Software Quantized I-ViT Model (CPU)
│   ├── weights/mobilevit_ivit_qat_best.pth
│   ├── models/quantization_utils/
│   ├── run_ivit_sw_kv260.py
│   ├── run_benchmark.sh
│   └── README.md
│
├── 03_ivit_hw_accelerated/                         # PHASE 3: Partitioned Co-Design (Stage 2 PL)
│   ├── overlay/attention_core_top_2A.bit (.hwh)
│   ├── pynq_attention_hook.py
│   ├── run_ivit_hw_kv260.py
│   ├── run_benchmark.sh
│   └── README.md
│
├── 04_all_9_transformers_hw/                       # PHASE 4: Full Network Acceleration (All 9 Transformers)
│   ├── overlay/attention_core_top_2A.bit (.hwh)
│   ├── pynq_multi_stage_attention_hook.py
│   ├── run_all_transformers_hw_kv260.py
│   ├── run_benchmark.sh
│   └── README.md
│
├── deploy_all_to_kv260.sh                          # One-click deployment script (Laptop -> KV260 via SSH/SCP)
└── run_all_phases_kv260.sh                         # Master execution script generating 4-phase comparative reports
```

---

## 3. Comparative Evaluation Matrix Across All 4 Phases

| Metric | Phase 1 (Baseline) | Phase 2 (I-ViT SW) | Phase 3 (Stage 2 HW) | Phase 4 (All 9 HW) |
| :--- | :---: | :---: | :---: | :---: |
| **Model** | Apple MobileViT-XXS | I-ViT MobileViT-XXS | I-ViT MobileViT-XXS | I-ViT MobileViT-XXS |
| **Arithmetic** | FP32 | INT8 / INT16 | Hybrid PS/PL | Hybrid PS/PL |
| **Attention Target** | Cortex-A53 (4 Cores) | Cortex-A53 (4 Cores) | Stage 2 -> PL (150 MHz) | **All 9 Layers -> PL (150 MHz)** |
| **PL Invocations / Img** | 0 | 0 | 1 | **9 Invocations** |
| **Stage 2 ($N=256, d_k=16$)** | Software FP32 | Software Integer | **PL Core (~2.08 ms)** | **PL Core (~2.08 ms)** |
| **Stage 3 ($N=64, d_k=20$)** | Software FP32 | Software Integer | Software Integer | **PL Core (~0.52 ms)** |
| **Stage 4 ($N=16, d_k=24$)** | Software FP32 | Software Integer | Software Integer | **PL Core (~0.15 ms)** |
| **Speedup vs Baseline** | $1.00\times$ (Ref) | Amdahl bound | Measured on KV260 | **Maximum Network Speedup** |
