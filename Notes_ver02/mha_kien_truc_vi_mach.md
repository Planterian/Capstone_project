# Kiến trúc Vi mạch MHA v2.0 (MobileViT-XXS Attention Accelerator)

> **Dự án:** Bộ tăng tốc phần cứng Scaled-Dot Attention Core cho Apple MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật đồng bộ hóa Bit-Exact với I-ViT QAT Model và RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mha_kien_truc_vi_mach.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mha_kien_truc_vi_mach.md)

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG TOP-LEVEL (attention_core_top)

Sơ đồ kết nối phân định ranh giới giữa **Processing System (ARM Cortex-A53 PS)** và **Programmable Logic (FPGA PL)** trên bo mạch AMD Kria KV260:

```text
========================================================================================================================
                                     TOP-LEVEL RTL ARCHITECTURE: attention_core_top (v2.0)
========================================================================================================================

    ARM PS (Host / PYNQ Driver / Jupyter Lab)
   +---------------------------------------+
   |  Memory-Mapped AXI-Lite               |
   |  Control, Status & BRAM Reload Regs   |
   +-------------------+-------------------+
                       |
                       | (AWADDR, WDATA, ARADDR, RDATA / AXI4-Lite 32-bit)
                       v
   +-------------------------------------------------------------------------------------------------------------------+
   | AXI4-Lite Slave Register File & Central Controller (axi_lite_regs.sv)                                             |
   |   - Reg 0x00: CONTROL [0: START, 1: CLEAR_BUF, 2: SOFT_RESET]  - Reg 0x18: NUM_HEADS (H = 4)                     |
   |   - Reg 0x04: STATUS  [0: BUSY, 1: DONE, 2: ERROR, 3: PP_BANK] - Reg 0x20: SCALE_M (Dyadic Mantissa M = 27640)   |
   |   - Reg 0x10: SEQ_LEN (N = 256)                                - Reg 0x24: SCALE_E (Dyadic Exponent E = 22)      |
   |   - Reg 0x14: HEAD_DIM (d_k = 16, 20, 24)                      - Reg 0x40/0x44: BRAM LUT Reload Window (Step 2.5) |
   +---------------------------------------------------+---------------------------------------------------------------+
                                                       | (Control Signals: start_pulse, cfg_lut_*, reg_scale_*)
                                                       v
   +-------------------------------------------------------------------------------------------------------------------+
   |                                            PIPELINED ATTENTION DATAPATH                                           |
   |                                                                                                                   |
   |  AXI DMA MM2S (DDR4 -> PL)                                                                                        |
   |  +--------------------+                                                                                           |
   |  | AXI4-Stream Ingress|=== (64-bit S_AXIS_TDATA / TVALID / TREADY / TLAST) ===+                                   |
   |  +--------------------+                                                       |                                   |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           | Skid Buffer & Multi-Bank Ingress BRAM  |              |
   |                                                           | (FSM nạp tuần tự Q -> K -> V INT8)     |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               |                                   |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 1: QK^T GEMM Streaming Engine   |              |
   |                                                           |  (Nhân song song d_k chiều -> INT32)   |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT32 Raw Scores: 1 score/cycle) |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 2: Dyadic Scaler Unit (HW-04)   |              |
   |                                                           |  ((Score x M) >>> E -> INT16 Sat Clamped)|            |
   |                                                           +-------------------+--------------------+              |
   |                                                           | (INT16 Scaled Scores [-32768, 32767])                 |
   |                                                           v                                                       |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 3: Reconfigurable Softmax Engine|              |
   |                                                           |  (Max-Sub + True Dual-Port 1024w BRAM  |              |
   |                                                           |   + 32-step Restoring Div -> UINT16)   |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (UINT16 Probabilities [0, 32767])  |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 4: Score x V Context GEMM       |              |
   |                                                           |  (Tích lũy 256 tokens, >>> 15, Sat INT8|              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT8 Context Vectors)            |
   |                                                                               v                                   |
   |  AXI DMA S2MM (PL -> DDR4)                                                    |                                   |
   |  +--------------------+                                                       |                                   |
   |  | AXI4-Stream Egress |<== (64-bit M_AXIS_TDATA / TVALID / TREADY / TLAST) ===+                                   |
   |  +--------------------+                                                                                           |
   +-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. LUỒNG DỮ LIỆU PIPELINE 4 TẦNG (PIPELINE DATAPATH)

Sơ đồ chi tiết luồng xử lý đồng bộ chu kỳ xung nhịp giữa các khối chức năng:

```text
========================================================================================================================
                                     MHA CORE PIPELINE & DATAFLOW ARCHITECTURE
========================================================================================================================

   AXI4-Stream RX  
   S_AXIS_TDATA [63:0]  
   ======/======================================================================================================+  
         |                                                                                                      |  
         v                                                                                                      v  
   +------------------+     +-------------------+     +-------------------+     +------------------+     +------------------+
   |   INPUT BUFFER   |     |  STAGE 1: QK^T    |     |  STAGE 2: DYADIC  |     |  STAGE 3: BRAM   |     |  STAGE 4: SCORE*V|
   |   SKID & BRAM    |     |  DOT-PRODUCT      |     |  SCALER UNIT      |     |  SHIFTMAX ENGINE |     |  CONTEXT GEMM    |
   |                  |     |                   |     |                   |     |                  |     |                  |
   |  +------------+  |     |  +-------------+  |     |  +-------------+  |     |  +------------+  |     |  +------------+  |
   |  | Q_MEM      |  |     |  | Parallel d_k|  |     |  | 48-bit Mult |  |     |  | Max-Sub    |  |     |  | 32-bit Acc |  |
   |  | 256x32 INT8|  |     |  | Multipliers |  |     |  | (Score x M) |  |     |  | Buffer 256 |  |     |  | 32 Chans   |  |
   |  +------------+  |     |  +------+------+  |     |  +------+------+  |     |  +-----+------+  |     |  +-----+------+  |
   |  | K_MEM      |  |     |         |         |     |         |         |     |        |         |     |        |         |
   |  | 256x32 INT8|  |     |         v         |     |         v         |     |        v         |     |        v         |
   |  +------------+  |     |  +-------------+  |     |  +-------------+  |     |  +------------+  |     |  +------------+  |
   |  | V_MEM      |  |     |  | Reduction   |  |     |  | ASR >>> E   |  |     |  | 1024w BRAM |  |     |  | >>> 15      |  |
   |  | 256x32 INT8|  |     |  | Adder Tree  |  |     |  | Sat Clamp   |  |     |  | Exp ROM    |  |     |  | Sat Clamp   |  |
   |  +------------+  |     |  +------+------+  |     |  +------+------+  |     |  +-----+------+  |     |  +-----+------+  |
   |                  |     |         |         |     |         |         |     |        |         |     |        |         |
   |                  |     |         |         |     |         |         |     |        v         |     |        |         |
   |                  |     |         |         |     |         |         |     |  +------------+  |     |        |         |
   |                  |     |         |         |     |         |         |     |  | 32-step Div|  |     |        |         |
   |                  |     |         |         |     |         |         |     |  | 2^31 / Sum |  |     |        |         |
   |                  |     |         |         |     |         |         |     |  +-----+------+  |     |        |         |
   |                  |     |         |         |     |         |         |     |        |         |     |        |         |
   +--------+---------+     +---------+---------+     +---------+---------+     +--------+---------+     +--------+---------+
            | (Read Port)             |                         |                        |                        |
            |                         | INT32 Raw Score         | INT16 Scaled Score     | UINT16 Prob [0..32767] | 64-bit Beats
            +-------------------------+-------------------------+------------------------+------------------------+
                                                                                                                  |
                                                                                                                  v
                                                                                                          AXI4-Stream TX
                                                                                                          M_AXIS_TDATA [63:0]
                                                                                                          (512 Beats / Head)
```

---

## 3. THUẬT TOÁN ĐIỀU KHIỂN BỘ TĂNG TỐC PHẦN CỨNG (HARDWARE ACCELERATION PSEUDO-ALGORITHMS)

Nhằm mô tả chuẩn xác cơ chế vận hành vi kiến trúc theo quy chuẩn bài báo khoa học (Academic Algorithm Specifications), toàn bộ luồng xử lý phần cứng của Attention Core được phân rã thành **5 khung thuật toán độc lập theo từng tầng Pipeline và khối chức năng**:
1. **Thuật toán 1 (Top-Level Orchestration):** Bộ điều phối trung tâm FSM cấp chip, quản lý luồng nạp DMA Ingress, kích hoạt tuần tự các tầng tính toán và xuất kết quả DMA Egress.
2. **Thuật toán 2 (Stage 1 --- $QK^T$ GEMM Engine):** Chi tiết hóa khối GEMM 1 tổng quát và ánh xạ vi kiến trúc trên 3 biến thể phần cứng (1A, 2A, 3B).
3. **Thuật toán 3 (Stage 2 --- Pipelined Dyadic Scaler):** Khối phần cứng thu phóng số học điểm cố định bão hòa $(S \times M) \ggg E \to \text{INT16}$.
4. **Thuật toán 4 (Stage 3 --- Hardware IntSoftmax Engine):** Bộ xử lý Softmax phần cứng 3-pass (Max-Search, Exp BRAM-LUT, Bộ chia khôi phục số dư 32 chu kỳ, Chuẩn hóa xác suất $\text{UINT16}$).
5. **Thuật toán 5 (Stage 4 --- Context $P \times V$ GEMM Engine):** Khối tích lũy vector ngữ cảnh, dịch bit và kẹp bão hòa $\text{INT8}$.

---

### 3.1 Thuật toán 1: Bộ Điều Phối Toàn Trình Top-Level (`attention_core_top`)

```text
========================================================================================================================
Algorithm 1: Top-Level Multi-Head Attention Accelerator Orchestration (attention_core_top)
========================================================================================================================
Require: Matrices Q, K, V in INT8 format (N x d_k per head) stored in off-chip DDR4 memory;
         Dyadic scaling parameters: Mantissa M (INT18), Exponent E (UINT5);
         Softmax Exp-LUT parameters (1024 words x 20-bit fixed-point coefficients in DDR4);
         Constants: Sequence length N = 256, Head dimension d_k in {16, 20, 24}, Number of heads H = 4;
         AXI4-Stream bus width W_bus in {64, 128} bits;
         Control flags: start_pulse, update_lut_cfg, clear_buf, soft_reset.
Ensure:  Context output matrix O in INT8 format (N x d_k per head) written back to DDR4 memory;
         Core status flag DONE asserted upon completion.
------------------------------------------------------------------------------------------------------------------------
 1: if update_lut_cfg is true then
 2:    Load 1024-word Exp-LUT coefficients from DDR into True Dual-Port Softmax BRAM via AXI-Lite MMIO (Reg 0x40/0x44)
 3: end if
 4: for each attention head h in [0, H) do
 5:    DMA_MM2S_Transfer(Q[h], K[h], V[h]) from DDR into on-chip Ping-Pong BRAM (Q_MEM, K_MEM, V_MEM)
 6:    for each query token row i in [0, N) do
 7:       s_raw[i, :]   <- Stage1_QKT_GEMM(Q_MEM[i, :], K_MEM, N, d_k)                      // Call Algorithm 2
 8:       s_scaled[i, :]<- Stage2_Dyadic_Scaler(s_raw[i, :], M, E, N)                       // Call Algorithm 3
 9:       P[i, :]       <- Stage3_Hardware_IntSoftmax(s_scaled[i, :], N)                    // Call Algorithm 4
10:       o_i           <- Stage4_Context_GEMM(P[i, :], V_MEM, N, d_k)                      // Call Algorithm 5
11:       Store token vector O[i, :] <- o_i into Egress Ping-Pong Buffer
12:    end for
13:    DMA_S2MM_Transfer(O) to DDR4 via AXI4-Stream Egress (W_bus beats, tlast asserted at final beat)
14: end for
15: Assert core DONE interrupt and update STATUS register (Reg 0x04[1] <- 1)
========================================================================================================================
```

---

### 3.2 Thuật toán 2: Stage 1 --- Khối $QK^T$ GEMM & Các Biến Thể Vi Kiến Trúc

```text
========================================================================================================================
Algorithm 2: Stage 1 --- Generalized QK^T GEMM Row Computation & Microarchitecture Mapping
========================================================================================================================
Require: Query row vector q_i = Q[i, 0 : d_k-1] in INT8;
         Key matrix K[0 : N-1, 0 : d_k-1] in INT8 stored in on-chip Ping-Pong BRAM (K_MEM);
         Parameters: Sequence length N = 256, Head dimension d_k in {16, 20, 24}, Bus width W_bus in {64, 128} bits;
         Target Variant in {Variant_1A, Variant_2A, Variant_3B}.
Ensure:  Raw Attention Score row vector s_raw[i, 0 : N-1] in INT32.
------------------------------------------------------------------------------------------------------------------------
    // --- Case 1: Variant 1A --- Pure LUT Streaming with N=32 Parallel Adder Tree (pipelined_adder_tree) ---
 1: if Target Variant == Variant_1A then
 2:    Latch query row vector q_i into local query registers (held stationary across all N key tokens)
 3:    for each key token index j in [0, N) do
 4:       Fetch key token vector K[j, 0 : d_k-1] from BRAM (1 beat / token on 128-bit bus, zero-padded to 32 bytes)
 5:       for each channel c in [0, 32) parallel do
 6:          prod[c] <- (c < d_k) ? (q_i[c] * K[j, c]) : 0                                  // 32 INT8 multipliers in LUT
 7:       end for
 8:       s_raw[i, j] <- Pipelined_Adder_Tree_5_Stages(prod[0 : 31])                        // 5-stage adder tree, II = 1 cycle
 9:    end for

    // --- Case 2: Variant 2A --- Interleaved DSP-Packed Streaming (dsp_packed_adder_tree, 8 DSPs, Dual 3-Stage Trees) ---
10: else if Target Variant == Variant_2A then
11:    Latch query row vector q_i into local query registers (held stationary across all key token pairs)
12:    num_beats <- ceil(d_k / 8)                                                           // Token sub-beats: 2 beats for d_k=16; 3 beats for d_k=20, 24
13:    for each key token pair p in [0, N/2) do                                             // Stream 128 pairs of Even (j_e = 2p) and Odd (j_o = 2p+1) tokens
14:       j_e <- 2*p,  j_o <- 2*p + 1
15:       Initialize beat accumulators: acc_e <- 0, acc_o <- 0
16:       for each beat b in [0, num_beats) do
17:          Fetch slices q_i[8*b : 8*b+7], K[j_e, 8*b : 8*b+7], K[j_o, 8*b : 8*b+7] from BRAM
18:          for each DSP slice d in [0, 8) parallel do                                     // 8 DSP48E2 packed MAC units (27x18-bit)
19:             prod_e[d] <- q_i[8*b + d] * K[j_e, 8*b + d]                                // Port A x Port B (K_even product)
20:             prod_o[d] <- q_i[8*b + d] * K[j_o, 8*b + d]                                // Port A x Port C (K_odd product)
21:          end for
22:          acc_e <- acc_e + Adder_Tree_3_Stages(prod_e[0 : 7])                            // 3-stage binary tree (8 -> 4 -> 2 -> 1)
23:          acc_o <- acc_o + Adder_Tree_3_Stages(prod_o[0 : 7])                            // Parallel tree for odd token
24:       end for
25:       {s_raw[i, j_e], s_raw[i, j_o]} <- Zero_Bubble_Interleaver(acc_e, acc_o)          // Sustained throughput II = 1 cycle
26:    end for

    // --- Case 3: Variant 3B --- 2D Output-Stationary Systolic MAC Array (systolic_mac_array, 4x4 Grid, 16 DSPs) ---
27: else if Target Variant == Variant_3B then
28:    Constants: T_row = 4 (tile rows), T_col = 8 (tile columns via 4x4 PE grid with dual-INT8 packing)
29:    for each column block j_blk in [0, N) with step T_col do
30:       current_block_col <- min(T_col, N - j_blk)
31:       for each row block i_blk in [0, N) with step T_row do
32:          current_block_row <- min(T_row, N - i_blk)
33:          Initialize systolic local accumulator tile: localC[0 : T_row-1, 0 : T_col-1] <- 0
34:          for each dimension slice k_0 in [0, d_k) with step 1 do
35:             Skew & stream local Q slice Q[i_blk : i_blk + T_row - 1, k_0] into PE row inputs
36:             Skew & stream local K slice K[j_blk : j_blk + T_col - 1, k_0] into PE column inputs
37:             localC <- localC + localQ * localK^T                                        // 4x4 PEs x 2 INT8 = 32 MACs/cycle
38:          end for
39:          Sequentially drain tile localC (32 items) into Reorder Buffer at offset (i_blk, j_blk)
40:       end for
41:    end for
42:    Drain raw score row vector from Reorder Buffer: s_raw[i, 0 : N-1] <- Reorder_Buffer[i, 0 : N-1]
43: end if
========================================================================================================================
```

---

### 3.3 Thuật toán 3: Stage 2 --- Khối Thu Phóng Pipelined Dyadic Scaler Unit (`scale_unit`)

```text
========================================================================================================================
Algorithm 3: Stage 2 --- Pipelined Dyadic Scaler Unit (scale_unit)
========================================================================================================================
Require: Raw Attention Score row vector s_raw[i, 0 : N-1] in INT32;
         Dyadic scaling parameters: Mantissa M (INT18), Exponent E (UINT5);
         Sequence length N = 256.
Ensure:  Scaled Attention Score row vector s_scaled[i, 0 : N-1] in INT16 saturated to [-32768, 32767].
------------------------------------------------------------------------------------------------------------------------
 1: for each key token index j in [0, N) do
 2:    P_mult <- s_raw[i, j] * M                                                            // DSP Mult: INT32 x INT18 -> INT50
 3:    S_shifted <- P_mult >>> E                                                            // Arithmetic Right Shift >>> E
 4:    s_scaled[i, j] <- Clamp_INT16(S_shifted)                                             // Saturation Clamp to [-32768, 32767]
 5: end for
 6: return s_scaled[i, 0 : N-1]
========================================================================================================================
```

---

### 3.4 Thuật toán 4: Stage 3 --- Bộ Xử Lý Phần Cứng IntSoftmax 3 Passes (`softmax_lut`)

```text
========================================================================================================================
Algorithm 4: Stage 3 --- Reconfigurable 3-Pass Hardware IntSoftmax Engine (softmax_lut)
========================================================================================================================
Require: Scaled Attention Score row vector s_scaled[i, 0 : N-1] in INT16;
         True Dual-Port Exp-LUT BRAM table EXP_BRAM_LUT[0 : 1023] (1024 words x 20-bit);
         Sequence length N = 256.
Ensure:  Normalized Attention Probability row vector P[i, 0 : N-1] in UINT16 where P[i, j] in [0, 32767].
------------------------------------------------------------------------------------------------------------------------
    // --- Pass 1: Streaming Row Maximum Search ---
 1: m_i <- -32768                                                                           // Khởi tạo giá trị cực tiểu INT16
 2: for each key token index j in [0, N) do
 3:    m_i <- max(m_i, s_scaled[i, j])                                                      // Bộ so sánh dò đỉnh cực đại
 4: end for
    // --- Pass 2: Exponentiation via BRAM LUT & Denominator Summation ---
 5: S_i <- 0                                                                                // Bộ tích lũy mẫu số UINT24
 6: for each key token index j in [0, N) do
 7:    d_ij <- s_scaled[i, j] - m_i                                                         // Hiệu số d_ij <= 0
 8:    lut_addr <- Clamp_INT(-1023, 0, d_ij) + 1023                                         // Kẹp địa chỉ trong [0, 1023]
 9:    e_ij <- EXP_BRAM_LUT[lut_addr]                                                       // Tra giá trị hàm mũ định dạng UFix20_19
10:    S_i <- S_i + e_ij                                                                    // Tích lũy tổng sum(e^d_ij)
11: end for
    // --- Intermediate: 32-Step Restoring Integer Division ---
12: R_i <- 32_Step_Restoring_Divider(2^31, S_i)                                             // Nghịch đảo: floor(2^31 / S_i) (UFix32_31)
    // --- Pass 3: Probability Normalization & Saturation ---
13: for each key token index j in [0, N) do
14:    M_prob <- e_ij * R_i                                                                 // Nhân: 20-bit x 32-bit -> 52-bit
15:    P[i, j] <- Clamp_UINT16(M_prob >>> 16)                                               // Xác suất chuẩn hóa trong [0, 32767]
16: end for
17: return P[i, 0 : N-1]
========================================================================================================================
```

---

### 3.5 Thuật toán 5: Stage 4 --- Khối Nhân Ma Trận Context $P \times V$ (`score_v_gemm`)

```text
========================================================================================================================
Algorithm 5: Stage 4 --- Context Score x Value GEMM Engine (score_v_gemm)
========================================================================================================================
Require: Normalized Probability row vector P[i, 0 : N-1] in UINT16;
         Value matrix V in INT8 stored in on-chip Ping-Pong BRAM (V_MEM);
         Constants: Sequence length N = 256, Head dimension d_k in {16, 20, 24}.
Ensure:  Context output token vector o_i = O[i, 0 : d_k-1] in INT8.
------------------------------------------------------------------------------------------------------------------------
 1: for each feature channel d in [0, d_k) parallel do
 2:    acc_out[d] <- 0                                                                      // Khởi tạo bộ tích lũy 32-bit
 3:    for each token index j in [0, N) do
 4:       acc_out[d] <- acc_out[d] + (P[i, j] * V_MEM[j, d])                                // Nhân UINT16 x INT8 tích lũy 32-bit
 5:    end for
 6:    S_proj <- acc_out[d] >>> 15                                                          // Thu phóng điểm cố định ASR 15
 7:    O[i, d] <- Clamp_INT8(S_proj)                                                        // Kẹp bão hòa trong [-128, 127]
 8: end for
 9: return O[i, 0 : d_k-1]
========================================================================================================================
```

---

### 3.6 Bảng Ánh Xạ Biến Số Thuật Toán Sang Tín Hiệu & Module Phần Cứng RTL

| Ký hiệu thuật toán | Tên phần cứng / Module RTL | Tín hiệu / Cổng giao tiếp | Chuẩn dữ liệu | Chức năng vi kiến trúc |
|---|---|---|---|---|
| $Q, K, V$ (DDR4) | AXI DMA MM2S | `S_AXIS_TDATA [127:0]` | INT8 (vector) | Luồng nạp dữ liệu từ RAM ngoài DDR4 vào PL qua AXI4-Stream |
| $Q\_\text{MEM}, K\_\text{MEM}, V\_\text{MEM}$ | `ping_pong_bram_buffer.sv` | `q_mem_addr`, `q_mem_rdata` | INT8 (on-chip) | Bộ đệm SRAM on-chip True Dual-Port, hỗ trợ Ping-Pong song song hóa |
| `COMPUTE_QKT_ROW_GEMM` | Stage 1 GEMM Engine:<br/>- `pipelined_adder_tree.sv` (1A)<br/>- `dsp_packed_adder_tree.sv` (2A)<br/>- `systolic_mac_array.sv` (3B) | `score_valid`, `raw_score [31:0]` | INT32 | Tính tích vô hướng ma trận điểm số chú ý thô giữa truy vấn và khóa |
| $s_{\text{scaled}}[i, j]$ | `scale_unit.sv` (Stage 2) | `scaled_score [15:0]`, `valid` | INT16 | Dyadic Rescaling: $(S \times M) \ggg E$, bão hòa trong $[-32768, 32767]$ |
| $m_i$ | `softmax_lut.sv` (Pass 1) | `max_val [15:0]`, `max_valid` | INT16 | Tìm giá trị cực đại hàng của vector điểm số phục vụ trừ max |
| $e_{i, j}$ | `softmax_lut.sv` (Pass 2) | `exp_lut_rdata [19:0]` | UFix20_19 | Tra giá trị hàm mũ từ True Dual-Port BRAM (1024 từ $\times$ 20-bit) |
| $S_i$ | `softmax_lut.sv` (Pass 2) | `sum_exp [23:0]` | UINT24 | Tích lũy tổng mẫu số $\sum e^{s_j - m_i}$ qua 256 tokens |
| $R_i$ | `softmax_lut.sv` (Divider) | `reciprocal [31:0]` | UFix32_31 | Bộ chia phục hồi số dư 32 chu kỳ: $R_i = \lfloor 2^{31} / S_i \rfloor$ |
| $P[i, j]$ | `softmax_lut.sv` (Pass 3) | `prob_out [15:0]`, `prob_valid`| UINT16 | Nhân chuẩn hóa xác suất: $(e_{i, j} \times R_i) \ggg 16 \in [0, 32767]$ |
| $\text{acc\_out}, O[i, d]$| `score_v_gemm.sv` (Stage 4) | `out_tdata [127:0]`, `tvalid` | INT8 | Tích lũy $P \times V$, dịch phải 15 bit và kẹp bão hòa giá trị INT8 |
| $O$ (DDR4) | AXI DMA S2MM | `M_AXIS_TDATA [127:0]` | INT8 (vector) | Luồng xuất kết quả chú ý về DDR4 (`tlast` bật ở beat cuối cùng) |
| `update_lut_cfg` | `axi_lite_regs.sv` | `cfg_lut_we`, `cfg_lut_addr` | AXI-Lite MMIO | Ghi đè hệ số hàm mũ qua cửa sổ CSR (Reg `0x40`/`0x44`) |
| `start_pulse`, `DONE` | `axi_lite_regs.sv` | `reg_control[0]`, `reg_status[1]`| Pulse / Flag | Lệnh kích hoạt thực thi từ ARM CPU và cờ báo hoàn thành xử lý |

---

## 4. BẢNG ĐỐI CHIẾU SỬA ĐỔI KIẾN TRÚC TỪ v1.0 SANG v2.0

| Hạng mục thiết kế | Phiên bản cũ (`Notes/`) | Phiên bản chuẩn hóa hiện tại (`Notes_ver02/`) | Lý do kỹ thuật / Căn cứ sửa đổi |
|---|---|---|---|
| **Mô hình mục tiêu** | ViT tiêu chuẩn ($N=196, d_k=32, H=12$) | **Apple MobileViT-XXS** ($N=256, d_k \in \{16, 20, 24\}, H=4$) | Khớp cấu trúc MobileViT trích xuất từ Hugging Face |
| **Cơ chế thu phóng (Scaler)** | Dịch phải thuần túy $\text{ASR} \gg 2$ hoặc $3$ (0 DSPs) | **Dyadic Rescaler:** $(S \times M) \ggg E$ với $M=27640, E=22$ | $1/\sqrt{20}$ và $1/\sqrt{24}$ không thể biểu diễn bằng phép dịch bit |
| **Độ phân giải Softmax** | UINT8 $[0, 255]$ (bộ chia xấp xỉ tổ hợp) | **UINT16 $[0, 32767]$** (Bộ chia lặp 32 chu kỳ khôi phục số dư) | Khớp 100% Bit-Exact với toán tử `IntSoftmax` của I-ViT |
| **Bảng tra hàm mũ Exp ROM** | ROM tĩnh 256 từ (gán cứng $e^x$) | **True Dual-Port BRAM 1024 từ $\times$ 20-bit** có nạp lại qua CSR | Hỗ trợ 3 Stages linh hoạt thông qua phần mềm cấu hình |
| **Tích lũy Systolic MAC** | Gộp chung `col0 + col1` vào 1 thanh ghi | **Tách độc lập 2 bộ tích lũy `pe_acc_col0` và `col1`** | Bảo toàn tính độc lập của 2 cột đặc trưng trong DSP packing |
| **Khối Context GEMM** | Chỉ tính 1 token đầu tiên (thiếu 99.6%) | **Tính đủ 256 tokens, 512 beats AXI-Stream, `tlast` beat 511** | Xuất trọn vẹn 4,096 bytes ma trận Context Output |
| **Vòng lặp Head (Multi-Head)** | 4 lõi song song hoặc FSM lặp Head 0 | **Single-Head Core tuần tự hóa 4 Heads** qua FSM trung tâm | Tiết kiệm >70% tài nguyên DSP và BRAM trên chip Kria KV260 |
| **Mã đọc tệp `$readmemh`** | Nằm trong Top synthesizable RTL | **Loại bỏ 100% trong synthesizable RTL**, nạp qua AXI-Stream | Đảm bảo khả năng tổng hợp phần cứng trên bo mạch vật lý |

