# Kiến trúc Vi mạch MHA

Dưới đây là **Bộ sơ đồ khối kiến trúc vi mạch RTL (RTL Block Diagrams & Pipeline Datapath)** hoàn chỉnh cho **Multi-Head Attention (MHA) Core** và các mô-đun thành phần, được tổng hợp bằng **ASCII Art** theo đúng quy chuẩn thiết kế vi mạch chuyên sâu.

---

### 1. TOP-LEVEL SYSTEM ARCHITECTURE (attention_core_top)

Sơ đồ thể hiện giao diện giữa **Processing System (ARM PS)** và **Programmable Logic (FPGA PL)** qua các bus AXI, cùng FSM điều khiển trung tâm và bộ đệm BRAM Ping-Pong.

```text
========================================================================================================================
                                     TOP-LEVEL RTL ARCHITECTURE: attention_core_top
========================================================================================================================

    ARM PS (Host / PYNQ Driver)
   +---------------------------+
   |  Memory-Mapped AXI-Lite   |
   |  Control & Status Regs    |
   +-------------+-------------+
                 |
                 | (AWADDR, WDATA, ARADDR, RDATA / AXI-Lite Handshake)
                 v
   +-------------------------------------------------------------------------------------------------------------------+
   | AXI4-Lite Slave Register File & Control FSM (HW-01 / axi_lite_regs.sv)                                            |
   |   - Reg 0x00: CONTROL [0: START, 1: DONE, 2: ERROR]      - Reg 0x10: SEQ_LEN (N = 196)                            |
   |   - Reg 0x0C: CORE_ID (0x56495432)                       - Reg 0x18: HEAD_DIM (d_k = 32)                          |
   +---------------------------------------------------+---------------------------------------------------------------+
                                                       | (Control Signals: start_pulse, tile_config, clear_buffers)
                                                       v
   +-------------------------------------------------------------------------------------------------------------------+
   |                                            PIPELINED ATTENTION DATAPATH                                           |
   |                                                                                                                   |
   |  AXI DMA (MM2S)                                                                                                   |
   |  +-------------------+                                                                                            |
   |  | AXI4-Stream RX    |=== (64-bit S_AXIS_TDATA / TVALID / TREADY / TLAST) ===+                                    |
   |  +-------------------+                                                        |                                   |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           | Input Ping-Pong BRAM Buffers (HW-02)   |              |
   |                                                           |   [ Q_Buffer ] [ K_Buffer ] [ V_Buffer]|              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               |                                   |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 1: QK^T GEMM Engine (HW-03)      |             |
   |                                                           |  (Systolic MAC Array / DSP Packing)    |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT32 Partial Sums)              |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 2: Scaler Unit (HW-04)          |              |
   |                                                           |  (Arithmetic Right-Shift ASR 1/sqrt(d_k)|             |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT16 Scaled Scores)             |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 3: Hardware Softmax (HW-05)     |              |
   |                                                           |  (Max-Sub + Exp LUT + Inverse Div)     |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT8 Probabilities)              |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 4: Score x V GEMM Engine (HW-07)|              |
   |                                                           |  (Systolic MAC Array / Saturating)     |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT8 Output Tokens)              |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           | Output Ping-Pong Buffer (HW-09)        |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               |                                   |
   |  AXI DMA (S2MM)                                                               v                                   |
   |  +-------------------+                                                        |                                   |
   |  | AXI4-Stream TX    |<== (64-bit M_AXIS_TDATA / TVALID / TREADY / TLAST) ====+=                                  |
   |  +-------------------+                                                                                            |
   +-------------------------------------------------------------------------------------------------------------------+
```

---

### 2. MULTI-HEAD ATTENTION CORE PIPELINE DATAPATH

Sơ đồ chi tiết luồng dữ liệu 5 giai đoạn (5-Stage Execution Pipeline) truyền qua các thanh ghi Pipelining, hỗ trợ cờ valid_pipe và bắt tay Backpressure (TREADY/TVALID).

```text
========================================================================================================================
                                     MHA CORE PIPELINE & DATAFLOW ARCHITECTURE
========================================================================================================================

  AXI4-Stream RX  
  S_AXIS_TDATA [63:0]  
  ======/=======================================================================================================+  
        |                                                                                                       |  
        v                                                                                                       v  
  +------------------+     +-------------------+     +-------------------+     +-------------------+     +-------------------+
  |   INPUT BRAM     |     |  STAGE 1: QK^T    |     |  STAGE 2: SCALER  |     |  STAGE 3: SOFTMAX |     |  STAGE 4: SCORE*V |
  |   PING-PONG      |     |  MATMUL (MAC)     |     |  (ASR SHIFT)      |     |  ENGINE (LUT/PWL) |     |  AGGREGATION      |
  |                  |     |                   |     |                   |     |                   |     |                   |
  |  +------------+  |     |  +-------------+  |     |  +-------------+  |     |  +-------------+  |     |  +-------------+  |
  |  | Q_RAM      |  | INT8|  | DSP48E2     |  |INT32|  | ASR Shift   |  |INT16|  | Max-Sub     |  | INT8|  | DSP48E2     |  |
  |  | (N x d_k)  |==+====>|  | MAC Array   |==+====>|  | Bit-shifter |==+====>|  | Exp LUT     |==+====>|  | MAC Array   |==+===+
  |  +------------+  | Q,K |  | (Unrolled)  |  |Score|  | (1/sqrt(d_k)|  |Score|  | Div/Inverse |  | Prob|  | (Saturating)|  |   |
  |  | K_RAM      |  |     |  +-------------+  |     |  +-------------+  |     |  +-------------+  |     |  +-------------+  |   |
  |  | (d_k x N)  |  |     |                   |     |                   |     |                   |     |                   |   |
  |  +------------+  |     |  Reg Stage 1      |     |  Reg Stage 2      |     |  Reg Stage 3      |     |  Reg Stage 4      |   |
  |  | V_RAM      |  |     |  [32-bit Acc]     |     |  [16-bit Scaled]  |     |  [8-bit Prob]     |     |  [8-bit Clamped]  |   |
  |  | (N x d_k)  |  |     +---------+---------+     +---------+---------+     +---------+---------+     +---------+---------+   |
  |  +------------+  |               |                         |                         |                         |             |
  +------------------+               |                         |                         |                         |             |
        |                            v                         v                         v                         v             |
        |                      +-----------+             +-----------+             +-----------+             +-----------+       |
        +--------------------->| v_stage1  |------------>| v_stage2  |------------>| v_stage3  |------------>| v_stage4  |       |
          in_valid             +-----------+             +-----------+             +-----------+             +-----------+       |
                                                                                                                   |             |
                                                                                                                   v             |
                                                                                                             out_valid           |
                                                                                                                                 |
  AXI4-Stream TX                                                                                                                 |
  M_AXIS_TDATA [63:0] <==========================================================================================================+
```

---

### 3. MICRO-ARCHITECTURE OF CORE MODULES

#### Sơ đồ A: Processing Element (PE) với DSP Packing $18 \times 27$-bit (pe_slice_dsp_packing.sv)

Kỹ thuật đóng gói 2 phép nhân **INT8** ($A \times B$ và $A \times C$) vào duy nhất 1 khối phần cứng **DSP48E2** ($18 \times 27$-bit multiplier) trên chip UltraScale+.

```text
========================================================================================================================
                                PE SLICE MODULE: DSP48E2 2x INT8 PACKING ARCHITECTURE
========================================================================================================================

    Operand A [7:0]          Operand B [7:0]         Operand C [7:0]
      (Signed INT8)            (Signed INT8)           (Signed INT8)
            |                        |                       |
            v                        v                       v
     +--------------+         +--------------+        +--------------+
     | Sign-Extend  |         | Sign-Extend  |        | Sign-Extend  |
     |  (to 18-bit) |         |  (to 27-bit) |        |  (to 27-bit) |
     +------+-------+         +------+-------+        +------+-------+
            |                        |                       |
            |                        v                       |
            |                 +--------------+               |
            |                 | Shift <<< 18 |               |
            |                 +------+-------+               |
            |                        |                       |
            |                        +----------+------------+
            |                                   |
            |                                   v
            |                         +------------------+
            |                         | Add (B<<18) + C  | ==> Packed Operand [26:0]
            |                         +---------+--------+
            |                                   |
            +-------------------+---------------+
                                |
                                v
                     +---------------------+
                     | DSP48E2 Multiplier  |
                     |  (18-bit x 27-bit)  |
                     +----------+----------+
                                |
                                v
                     P_45 [44:0] = (A x B) * 2^18 + (A x C)
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
   Bits [33:18] (Raw A x B)                        Bits [15:0] (Raw A x C)
        |                                               |
        v                                               v
  +-----------------------+                        +-----------------------+
  | Sign-Correction Logic |                        |     Direct Output     |
  | High_Corr = P[33:18]  |                        |      Extraction       |
  |          + P[15]      |                        | (Guarded by P[17:16]) |        
  +---------+-------------+                        +-----------+-----------+
            |                                                  |
            v                                                  v
     Result_AB [15:0]                                   Result_AC [15:0]
   (A x B Product INT16)                              (A x C Product INT16)
```

---

#### Sơ đồ B: Cấu trúc Cây cộng Pipelined Adder Tree (pipelined_adder_tree.sv)

Mạch giảm dần nhị phân (Binary Tree Reduction) tính tích vô hướng $N$ phần tử trong 1 chu kỳ clock hiệu dụng ($II = 1$), quản lý Bit-Growth và chèn thanh ghi Flip-Flop.

```text
========================================================================================================================
                                PIPELINED ADDER TREE ARCHITECTURE (N = 16 Inputs)
========================================================================================================================

  Inputs: Vector A[0..15] (INT8), Vector B[0..15] (INT8)
    |
    v (Stage 0: Parallel Element-wise Multipliers)
  +-----+  +-----+  +-----+  +-----+         +-----+  +-----+  +-----+  +-----+
  | P0  |  | P1  |  | P2  |  | P3  |  . . .  | P12 |  | P13 |  | P14 |  | P15 |   (16x INT16 Products)
  +--+--+  +--+--+  +--+--+  +--+--+         +--+--+  +--+--+  +--+--+  +--+--+
     |        |        |        |               |        |        |        |
     +---+----+        +---+----+               +---+----+        +---+----+
         |                 |                        |                 |
         v (Reg Stage 0)   v (Reg Stage 0)          v (Reg Stage 0)   v (Reg Stage 0)
     +-------+         +-------+                +-------+         +-------+
     | Adder |         | Adder |      . . .     | Adder |         | Adder |    (Stage 1: 8 Adders, 17-bit)
     +---+---+         +---+---+                +---+---+         +---+---+
         |                 |                        |                 |
         +--------+--------+                        +--------+--------+
                  |                                          |
                  v (Reg Stage 1)                            v (Reg Stage 1)
              +-------+                                  +-------+
              | Adder |              . . .               | Adder |            (Stage 2: 4 Adders, 18-bit)
              +---+---+                                  +---+---+
                  |                                          |
                  +--------------------+---------------------+
                                       |
                                       v (Reg Stage 2)
                                   +-------+
                                   | Adder |                                  (Stage 3: 2 Adders, 19-bit)
                                   +---+---+
                                       |
                                       +----------+
                                                  |
                                                  v (Reg Stage 3)
                                              +-------+
                                              | Adder |                       (Stage 4: Root Adder, 20-bit)
                                              +---+---+
                                                  |
                                                  v
                                      Dot_Product [31:0] (INT32)
```

---

#### Sơ đồ C: Mạch Activation ShiftGELU 5-Stage Pipeline (shift_gelu_pipeline.sv)

Mạch GELU thuần số nguyên **0 DSPs**, sử dụng bộ dịch bit Barrel Shifter xấp xỉ hệ số $1.702$ và đoạn tính Sigmoid tuyến tính.

```text
========================================================================================================================
                                SHIFTGELU ACTIVATION PIPELINE ARCHITECTURE (0 DSPs)
========================================================================================================================

  Input Sample I_in [7:0] (INT8)
    |
    |-----> [ Delay FIFO (5-Stage Delay Line) ] ----------------------------------------------+
    |                                                                                         |
    v (Stage 0: 1.703125 * I_in via LUT-only Bit-Shifts)                                      |
  +---------------------------------------------------------+                                 |
  | I_p = I_in + (I_in >>> 1) + (I_in >>> 3) + (I_in >>> 4) |                                 |
  +----------------------------+----------------------------+                                 |
                               |                                                              |
                               v (Stage 1: Delta Computation)                                 |
  +---------------------------------------------------------+                                 |
  | I_delta = I_p - I_MAX_CONST (16'sd216)                  |                                 |
  +----------------------------+----------------------------+                                 |
                               |                                                              |
                               v (Stage 2: Piecewise INT8 Sigmoid Approx)                     |
  +---------------------------------------------------------+                                 |
  | if (I_delta >= 0)        Sigmoid = 255                  |                                 |
  | else if (I_delta < -300) Sigmoid = 0                    |                                 |
  | else                     Sigmoid = (I_delta+300)*255/300|                                 |
  +----------------------------+----------------------------+                                 |
                               |                                                              |
                               v Sigmoid [7:0] (UINT8)                                        v Delayed I_in [7:0]
  +-------------------------------------------------------------------------------------------+----------------+
  | Stage 3: Multiply & Rescale                                                                                |
  | Prod_16 = Delayed_I_in * Sigmoid  ===>  Clamped_INT8 = Clamp_INT8( Prod_16 >>> 8 )                         |
  +--------------------------------------------+---------------------------------------------------------------+
                                               |
                                               v
                                    I_out [7:0] (INT8 Output)
```

---

#### Sơ đồ D: Khối Re-quantization & Residual Adder (requant_residual_add.sv)

Mạch cộng đường tắt (Shortcut Connection) xử lý lệch Scale Factor ($S_F \neq S_X$) thông qua hai luồng nhân Dyadic và kẹp ngưỡng bão hòa.

```text
========================================================================================================================
                            RE-QUANTIZATION & RESIDUAL ADDER ARCHITECTURE
========================================================================================================================

   MSA Stream I_F [7:0]     Dyadic Scale M_F, e_F      Shortcut Stream I_X [7:0]      Dyadic Scale M_X, e_X
           |                            |                            |                            |
           v                            v                            v                            v
   +---------------+            +---------------+            +---------------+            +---------------+
   | Register Reg0 |            | Register Reg0 |            | Register Reg0 |            | Register Reg0 |
   +-------+-------+            +-------+-------+            +-------+-------+            +-------+-------+
           |                            |                            |                            |
           +--------------+-------------+                            +--------------+-------------+
                          |                                                         |
                          v (Stage 0: 40-bit Dyadic Mult)                           v (Stage 0: 40-bit Dyadic Mult)
               +----------------------+                                  +----------------------+
               | Prod_F = I_F * M_F   |                                  | Prod_X = I_X * M_X   |
               +----------+-----------+                                  +----------+-----------+
                          |                                                         |
                          v (Stage 1: Shift >>> e_F)                                v (Stage 1: Shift >>> e_x)
               +----------------------+                                  +----------------------+
               | Scaled_F = Prod_F>>e_F|                                 | Scaled_X = Prod_X>>e_X|
               +----------+-----------+                                  +----------+-----------+
                          |                                                         |
                          +-------------------------+-------------------------------+
                                                    |
                                                    v (Stage 2: Full Add & Saturation)
                                         +---------------------+
                                         | Sum_Full = F + X    |
                                         | Saturating Clamp    |
                                         |   [-128, +127]      |
                                         +----------+----------+
                                                    |
                                                    v
                                         I_out [7:0] (INT8 Output)
```

---

#### Sơ đồ E: Structure of Hardware Softmax Engine (softmax_lut.sv)

Mạch Softmax phần cứng gồm logic tìm Max, BRAM ROM chứa bảng tra hàm mũ $e^x$, bộ tích lũy mẫu số INT32, và khối chia/dịch bit.

```text
========================================================================================================================
                                    HARDWARE SOFTMAX ENGINE ARCHITECTURE
========================================================================================================================

   Raw Attention Scores INT16 (Row of N elements)
          |
          v
   +------------------------------------+
   | 1. Max Finder Logic                |  ===>  Finds S_max = Max(Score[0..N-1])
   +-----------------+------------------+
                     |
                     v
   +------------------------------------+
   | 2. Subtraction Unit                |  ===>  Shifted_Score[i] = Score[i] - S_max  (<= 0)
   +-----------------+------------------+
                     |
                     v
   +------------------------------------+
   | 3. BRAM Exp LUT ROM                |  ===>  Exp_Val[i] = Exp_Table[ Shifted_Score[i] ]
   +-----------------+------------------+
                     |
                     +----------------------------------+
                     |                                  |
                     v                                  v
   +------------------------------------+    +------------------------------------+
   | 4. Denominator Accumulator         |    | 5. Pipeline Delay Line             |
   |    Sum_Exp = Sum( Exp_Val[0..N-1] )|    |    Buffers Exp_Val[i]              |
   +-----------------+------------------+    +------------------+-----------------+
                     |                                          |
                     v                                          |
   +------------------------------------+                       |
   | 6. Dyadic Reciprocal Unit          |                       |
   |    Inv_Sum = (1 / Sum_Exp) * 2^e   |                       |
   +-----------------+------------------+                       |
                     |                                          |
                     +-------------------+----------------------+
                                         |
                                         v
   +------------------------------------------------------------------------------+
   | 7. Normalization Multiplier & Quantizer                                      |
   |    Prob_INT8[i] = Clamp_UINT8( (Exp_Val[i] * Inv_Sum) >>> e )                |
   +-------------------------------------+----------------------------------------+
                                         |
                                         v
                              Attention Probabilities INT8 [7:0] (to Score x V GEMM)
```

##### 💡 Tóm tắt đặc tính vi mạch (RTL Specifications Summary):
* **Băng thông dữ liệu (Data Widths)** : Dữ liệu đầu vào/đầu ra INT8 $[-128, +127]$, bộ tích lũy MAC INT32, điểm Score INT16.
* **Tài nguyên DSP** : Mạch DSP Packing giúp chạy **2 MACs / DSP / Cycle** trên cổng $18 \times 27$-bit của DSP48E2.
* **Tài nguyên LUT** : Mạch **ShiftGELU** và **Scaler** tiêu tốn **0 DSPs** (hoàn toàn dùng logic LUT và Barrel Shifter).
* **Bắt tay giao thức** : Chuẩn hóa AXI4-Stream (TVALID/TREADY/TLAST) xử lý backpressure an toàn và AXI4-Lite Slave cho thanh ghi FSM.
