# TECHNICAL SPECIFICATION: PHASE 4 — FULL ACCELERATION OF ALL 9 TRANSFORMERS ACROSS 3 STAGES

**Document ID:** `06_PHASE4_ALL_9_TRANSFORMERS_ACCELERATED_SPEC.md`  
**Phase:** 4 of 4 (Comprehensive Full-Network Attention Acceleration)  
**System Target:** AMD Kria KV260 Vision AI Starter Kit (`xck26-sfvc784-2LV-c`)  
**Scope:** 100% Offload of ALL 9 Multi-Head Attention Blocks in MobileViT-XXS  
**Hardware Accelerator:** Attention Core Variant 2A (`attention_core_top_2A`) @ 150.000 MHz  
**Data Bus:** 128-bit AXI DMA (MM2S & S2MM) with Dynamic CSR Parameterization  
**Location:** [`Project/src/Software/04_all_9_transformers_hw/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/04_all_9_transformers_hw/)  

---

## 1. Executive Summary & Objective

In **Phase 3**, hardware acceleration was applied selectively to Stage 2 Attention ($N=256, d_k=16$), while Stages 3 and 4 executed on the ARM Cortex-A53 CPU via software fallback.

In **Phase 4**, we achieve full-network attention acceleration by offloading **all 9 Transformer layers across all 3 MobileViT stages** directly to the physical FPGA PL Core:

```text
==================================================================================================
        MOBILEVIT-XXS 9 TRANSFORMER LAYERS HIERARCHY ACCELERATED ON KRIA KV260 PL
==================================================================================================

  [STAGE 2 - MobileViT Block 1]: 2 Transformer Layers (N=256 tokens, dk=16, H=1)
  ├── Layer 0 Attention -> Offloaded to PL Core (12,288 B DMA Ingress -> 4,096 B Egress)
  └── Layer 1 Attention -> Offloaded to PL Core (12,288 B DMA Ingress -> 4,096 B Egress)
      CSR Config: N=256, dk=16, M=27640, E=22

  [STAGE 3 - MobileViT Block 2]: 4 Transformer Layers (N=64 tokens, dk=20, H=1)
  ├── Layer 0 Attention -> Offloaded to PL Core (3,840 B DMA Ingress -> 1,280 B Egress)
  ├── Layer 1 Attention -> Offloaded to PL Core (3,840 B DMA Ingress -> 1,280 B Egress)
  ├── Layer 2 Attention -> Offloaded to PL Core (3,840 B DMA Ingress -> 1,280 B Egress)
  └── Layer 3 Attention -> Offloaded to PL Core (3,840 B DMA Ingress -> 1,280 B Egress)
      CSR Config: N=64, dk=20, M=22559, E=22

  [STAGE 4 - MobileViT Block 3]: 3 Transformer Layers (N=16 tokens, dk=24, H=1)
  ├── Layer 0 Attention -> Offloaded to PL Core (1,152 B DMA Ingress -> 384 B Egress)
  ├── Layer 1 Attention -> Offloaded to PL Core (1,152 B DMA Ingress -> 384 B Egress)
  └── Layer 2 Attention -> Offloaded to PL Core (1,152 B DMA Ingress -> 384 B Egress)
      CSR Config: N=16, dk=24, M=19947, E=22
==================================================================================================
  TOTAL HARDWARE ACCELERATED INVOCATIONS PER INFERENCE PASS: 9 ATTENTION CORES
==================================================================================================
```

---

## 2. Multi-Stage Dimensional & Parameter Mapping

The 9 Transformer layers vary in sequence length $N$, head dimension $d_k$, and dyadic scaling $(M, E)$ derived from the QAT fine-tuned checkpoint (`mobilevit_ivit_qat_best.pth`):

| Network Stage | Transformer Layers | Sequence Length ($N$) | Head Dim ($d_k$) | Ingress Payload ($N \times d_k \times 3$) | Egress Context ($N \times d_k$) | Dyadic Scale ($M, E$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 2** | Layer 0, 1 (2 layers) | **$N = 256$** | **$d_k = 16$** | **12,288 Bytes** | **4,096 Bytes** | $M = 27640, E = 22$ |
| **Stage 3** | Layer 0, 1, 2, 3 (4 layers) | **$N = 64$** | **$d_k = 20$** | **3,840 Bytes** | **1,280 Bytes** | $M = 22559, E = 22$ |
| **Stage 4** | Layer 0, 1, 2 (3 layers) | **$N = 16$** | **$d_k = 24$** | **1,152 Bytes** | **384 Bytes** | $M = 19947, E = 22$ |

---

## 3. Dynamic CSR Parameterization & Zero-Copy DMA Architecture

### 3.1 Dynamic CSR Reconfiguration Protocol
Before each of the 9 attention invocations, the host driver reprograms the AXI4-Lite CSR registers:
1. `ADDR_NBYTES` $\leftarrow N \times d_k \times 3$
2. `ADDR_SEQ_LEN` $\leftarrow N$
3. `ADDR_HEAD_DIM` $\leftarrow d_k$
4. `ADDR_NUM_HEADS` $\leftarrow 1$
5. `ADDR_SCALE_M` $\leftarrow M_{\text{stage}}$
6. `ADDR_SCALE_E` $\leftarrow E_{\text{stage}}$

### 3.2 Unified Zero-Copy Memory Management
Instead of allocating 9 separate buffers per image:
- Pre-allocates **one maximum-sized contiguous memory (CMA) DDR4 buffer** at startup:
  $$\text{CMA Ingress Buffer} = 12,288\text{ Bytes (Stage 2 peak)}$$
  $$\text{CMA Egress Buffer} = 4,096\text{ Bytes (Stage 2 peak)}$$
- For Stage 3 ($3,840\text{ B}$) and Stage 4 ($1,152\text{ B}$), the driver transfers the exact sub-slice of the pre-allocated buffer:
  `dma.sendchannel.transfer(in_buffer[:payload_bytes])`
  `dma.recvchannel.transfer(out_buffer[:context_bytes])`
- **Result:** $0$ dynamic OS memory allocations during the entire inference pass, minimizing kernel trap overhead.

---

## 4. Latency Formulation & Speedup Analysis

The end-to-end latency in Phase 4 is:

$$T_{\text{E2E\_Phase4}} = T_{\text{CNN\_PS}} + T_{\text{MLP\_PS}} + \sum_{i=1}^{9} \left( T_{\text{DMA\_in}} + T_{\text{PL\_core}} + T_{\text{DMA\_out}} \right)_i$$

Where:
- Stage 2 (2 layers): $2 \times \approx 2.08\text{ ms} = 4.16\text{ ms}$
- Stage 3 (4 layers): $4 \times \approx 0.52\text{ ms} = 2.08\text{ ms}$
- Stage 4 (3 layers): $3 \times \approx 0.15\text{ ms} = 0.45\text{ ms}$
- **Total PL Attention Time (9 Layers):** $\approx 6.69\text{ ms}$!

Compared to executing all 9 attention layers in pure software on the ARM Cortex-A53 CPU (which takes $\approx 75 - 110\text{ ms}$ in emulated integer math), Phase 4 provides massive kernel acceleration across the entire network.

---

## 5. Implementation Roadmap for Phase 4

1. **`pynq_multi_stage_attention_hook.py`:**
   - Multi-stage aware hook inspecting $(N, d_k)$ dynamically and setting CSR parameters for all 9 layers.
2. **`run_all_transformers_hw_kv260.py`:**
   - Full-network benchmark recording overall latency and per-stage PL latency breakdowns.
3. **`run_benchmark.sh`:**
   - Automated root wrapper with `xmutil unloadapp` and XRT environment sourcing.
4. **Update Master Harness:**
   - Integrate Phase 4 into `run_all_phases_kv260.sh` to generate a 4-row comparative executive report.

