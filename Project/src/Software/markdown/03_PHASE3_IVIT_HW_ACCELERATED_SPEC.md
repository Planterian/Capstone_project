# TECHNICAL SPECIFICATION: PHASE 3 — END-TO-END HW/SW CO-DESIGN (PS + PL ACCELERATION)

**Document ID:** `03_PHASE3_IVIT_HW_ACCELERATED_SPEC.md`  
**Phase:** 3 of 3 (Hardware-Accelerated Deployment)  
**System Architecture:** Hybrid HW/SW Partitioned Co-Design (PS: Cortex-A53 + PL: Attention Core)  
**Hardware Accelerator:** Attention Core Variant 2A (`attention_core_top_2A`) @ 150.000 MHz  
**Data Bus:** 128-bit AXI4-Stream via AXI DMA (`axi_dma_0`), AXI SmartConnect CSR (`axi_smc_ctrl`)  
**Host Controller:** AMD Kria KV260 PS via PYNQ v3.0.1 Embedded Linux Overlay  
**Location:** [`Project/src/Software/03_ivit_hw_accelerated/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Software/03_ivit_hw_accelerated/)  

---

## 1. Overview & Objective

**Phase 3** represents the pinnacle of this capstone research: deploying the full **End-to-End HW/SW Partitioned MobileViT-XXS** on physical silicon.

In this architecture:
* The general-purpose CNN feature extraction (Stem, Inverted Residuals), LayerNorms, Feed-Forward MLPs, and Classifier Heads execute on the **ARM Cortex-A53 CPU (Processing System - PS)**.
* The most compute-intensive, memory-bound bottleneck of the model — the **Stage 2 Multi-Head Attention kernel ($N=256, d_k=16$)** — is offloaded to the custom **FPGA Attention Accelerator (Programmable Logic - PL)** running at 150 MHz.

---

## 2. Hardware/Software Partitioning Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             FULL NETWORK EXECUTION FLOW: PARTITIONED BETWEEN PS AND PL                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [Input Image: 1x3x256x256]                                                          │
│        │                                                                               │
│        ▼ (Executed on PS - Quad-Core Cortex-A53)                                       │
│   [CNN Stem & Stage 1 Inverted Residuals]                                              │
│        │                                                                               │
│        ▼ Feature Map: [1, 24, 64, 64]                                                  │
│   [Stage 2: MobileViT Block 1]                                                         │
│   ├── Local Rep: Conv-3x3 -> Conv-1x1                                                  │
│   ├── Token Unfolding -> N=256 tokens, dk=16                                           │
│   ├── QKV Linear Projection (Fused)                                                    │
│   │                                                                                    │
│   │   ══════════════════════════════════════════════════════════════════════════       │
│   │   >>> HARDWARE BOUNDARY: OFF-LOAD TO FPGA PL CORE VIA PYNQ DMA <<<                 │
│   │                                                                                    │
│   │   1. Copy Q, K, V (12,288 Bytes) to Contiguous DDR4 Buffer (pynq.allocate)         │
│   │   2. Configure AXI4-Lite CSR: N=256, dk=16, M=27640, E=22, NBYTES=12288            │
│   │   3. AXI DMA MM2S: 128-bit Stream Burst (16 tokens/beat) into PL                   │
│   │   4. PL Accelerator Computes:                                                      │
│   │        - GEMM Q · K^T (INT8 x INT8 DSP Packed) -> INT32                            │
│   │        - Dyadic Scale (M=27640, E=22) -> INT16                                     │
│   │        - Shiftmax Softmax Exp LUT BRAM -> Probabilities [0, 32767]                 │
│   │        - Context GEMM Prob x V (INT16 x INT8 DSP Packed)                           │
│   │        - Hardware Execution Latency: 2.08 ms (481.3 FPS!)                          │
│   │   5. AXI DMA S2MM: Stream 4,096 Bytes Output Context back to PS                    │
│   │   ══════════════════════════════════════════════════════════════════════════       │
│   │                                                                                    │
│   ├── Inject 4,096-Byte Context Tensor back into PyTorch Execution Graph               │
│   ├── Transformer MLPs (ShiftGELU, QuantLinear) -> Executed on PS                      │
│   └── Token Folding & Residual Fusion                                                  │
│        │                                                                               │
│        ▼ (Executed on PS with Software Fallback / Multi-stage Support)                 │
│   [Stage 3: MobileViT Block 2 (N=64, dk=20)]                                           │
│   [Stage 4: MobileViT Block 3 (N=16, dk=24)]                                           │
│   [Penultimate Conv-1x1 + Global Average Pooling + Classifier Head]                    │
│        │                                                                               │
│        ▼                                                                               │
│   [Output Top-5 Predictions & Hardware Performance Metrics]                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. PYNQ Hardware Control & CSR Register Specification

### 3.1 CSR Memory Map (Aligned with `axi_lite_regs.sv`)

| Offset | Register Name | R/W | Bit Description | Configured Value |
| :---: | :--- | :---: | :--- | :--- |
| `0x00` | `ADDR_CTRL` | R/W | Bit 0: START, Bit 1: CLEAR, Bit 2: RESET | Pulse `0x01` to fire |
| `0x04` | `ADDR_STATUS` | RO | Core status: `[0]=busy`, `[1]=done`, `[2]=error` | Monitored by driver |
| `0x08` | `ADDR_NBYTES` | R/W | Total QKV payload size in bytes | `12288` |
| `0x10` | `ADDR_SEQ_LEN` | R/W | Sequence length $N$ | `256` |
| `0x14` | `ADDR_HEAD_DIM` | R/W | Head dimension $d_k$ | `16` |
| `0x18` | `ADDR_NUM_HEADS`| R/W | Number of attention heads $H$ | `1` |
| `0x20` | `ADDR_SCALE_M` | R/W | Dyadic Mantissa $M$ (16-bit signed) | `27640` |
| `0x24` | `ADDR_SCALE_E` | R/W | Dyadic Exponent $E$ (5-bit unsigned) | `22` |

### 3.2 Contiguous Memory (CMA) DMA Allocation
- **Ingress Buffer (QKV):**
  $$\text{Size} = 256 \times 16 \times 3 = 12,288\text{ Bytes} \quad (\text{INT8})$$
- **Egress Buffer (Context):**
  $$\text{Size} = 256 \times 16 \times 1 = 4,096\text{ Bytes} \quad (\text{INT8})$$
- **DMA Channel:**
  - `dma.sendchannel.transfer(in_buffer)`
  - `dma.recvchannel.transfer(out_buffer)`
  - Native 128-bit streaming over AXI4-Stream at 150 MHz ($150 \times 16\text{ B/s} = 2.4\text{ GB/s}$ peak bandwidth).

---

## 4. Multi-Stage Architectural Policy (Stages 2, 3, 4)

In MobileViT-XXS, the three transformer stages possess varying token dimensions:
* **Stage 2:** $N=256, d_k=16$ (Accounts for **$> 65\%$** of total transformer attention compute).
* **Stage 3:** $N=64, d_k=20$ (Accounts for $\approx 25\%$ of transformer compute).
* **Stage 4:** $N=16, d_k=24$ (Accounts for $< 10\%$ of transformer compute).

### Current Policy:
1. **Stage 2 ($N=256, d_k=16$):** Fully accelerated on the synthesized PL core (`attention_core_top_2A.bit`), achieving **100% bit-exact accuracy** ($MAE = 0.0$) and a hardware latency of **2.08 ms**.
2. **Stages 3 & 4:** Executed via clean software fallback using the Phase 2 I-ViT integer engine on the ARM CPU. This hybrid execution ensures full end-to-end network functionality while isolating the benchmark of the Stage 2 accelerator.

---

## 5. Performance Equations & Latency Breakdown

The total end-to-end latency of Phase 3 is decomposed as:

$$T_{\text{E2E\_HW}} = T_{\text{PS\_backbone}} + T_{\text{DMA\_overhead}} + T_{\text{PL\_kernel}}$$

Where:
- $T_{\text{PL\_kernel}} \approx 2.08\text{ ms}$ (measured on silicon at 150 MHz).
- $T_{\text{DMA\_overhead}} = T_{\text{alloc}} + T_{\text{cache\_flush}} + T_{\text{scatter/gather}}$.
- $T_{\text{PS\_backbone}}$ = CNN Stem, Stage 1, Stage 2 MLPs, Stage 3, Stage 4, Classifier.

---

## 6. Tracking & Future Considerations (`TODO`)

- [ ] **TODO: [Zero-Copy DMA]** Pre-allocate CMA DDR4 buffers once during model initialization to eliminate per-inference allocation overhead.
- [ ] **TODO: [Multi-Stage Reconfigurable LUT]** Extend the PL driver to update the exp LUT and CSRs dynamically for Stage 3 ($N=64, d_k=20$) and Stage 4 ($N=16, d_k=24$).
- [ ] **TODO: [Asynchronous DMA Overlapping]** Pipeline the DMA transfer of Stage 2 Attention with the concurrent execution of subsequent convolution layers.
- [ ] **TODO: [Power & Energy Measurement]** Integrate PMBus / INA260 power sensors on KV260 to measure energy per inference ($\text{mJ/frame}$).

