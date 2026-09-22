# Pipeline Phi Tuyến - MHA Level 3: Non-Linear Pipeline & Activation Stages

Dưới đây là **LEVEL 3 — NON-LINEAR PIPELINE & ACTIVATION STAGES** hoàn tất chuỗi phân rã vi mạch theo cấu trúc Phân cấp Từ trên xuống (Top-Down Hierarchy).
Phần này đi sâu vào chi tiết các mô-đun xử lý phi tuyến và chuẩn hóa: **Đơn vị Scaler (scale_unit.sv)**, **Bộ tính Softmax Phần cứng (softmax_lut.sv)** và **Khối Re-quantization & Residual Adder (requant_residual_add.sv)**.

---

### 1. Top-Level Non-Linear Sub-System Block Diagram (nonlinear_pipeline_top.sv)

```text
===================================================================================================================================
                        LEVEL 3: NON-LINEAR PIPELINE & ACTIVATION SUB-SYSTEM (nonlinear_pipeline_top.sv)
===================================================================================================================================

  Raw GEMM Score INT32 (From Stage 1 Systolic Array)
            │
            v
  +-----------------------------------------------------------------------------------------------------------------------------+
  | 1. SCALER UNIT (scale_unit.sv)                                                                                              |
  |    - Receives INT32 Accumulator Output                                                                                     |
  |    - Applies Arithmetic Right-Shift (ASR) via Barrel Shifter: Scaled_Score = Raw_Score >>> shift_val (0 DSPs)               |
  |    - Clamps Output to INT16 d_k Scaling Range                                                                                |
  +--------------------------------------------------------------+--------------------------------------------------------------+
                                                                 │
                                                                 │ INT16 Scaled Scores
                                                                 v
  +-----------------------------------------------------------------------------------------------------------------------------+
  | 2. HARDWARE SOFTMAX ENGINE (softmax_lut.sv)                                                                                 |
  |                                                                                                                             |
  |   +-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+   |
  |   | Row Max Finder        |──►  | Subtraction Unit      |──►  | BRAM Exp LUT ROM      |──►  | Normalization & Div   |   |
  |   | S_max = Max(S_0..N-1) |     | Shifted = S_ij - S_max|     | Lookup e^(Shifted)    |     | Prob = Exp / Sum_Exp  |   |
  |   +-----------------------+     +-----------------------+     +-----------------------+     +-----------+-----------+   |
  +---------------------------------------------------------------------------------------------------------│-------------------+
                                                                                                            │
                                                                                                            │ INT8 Probabilities
                                                                                                            v
                                                                                           [ Stage 4: Score x V GEMM Engine ]
                                                                                                            │
                                                                                                            │ INT8 MSA Output (I_F)
                                                                                                            v
  +-----------------------------------------------------------------------------------------------------------------------------+
  | 3. RE-QUANTIZATION & RESIDUAL ADDER (requant_residual_add.sv)                                                              |
  |                                                                                                                             |
  |   MSA Stream I_F (INT8) ──────► [ Dyadic Rescaler: M_F * 2^(-e_F) ] ──┐                                                     |
  |                                                                      ├──► Full Adder ──► Saturating Clamp ──► Out (INT8)    |
  |   Shortcut Stream I_X (INT8) ──► [ Dyadic Rescaler: M_X * 2^(-e_X) ] ──┘                   [-128, +127]                    |
  +-----------------------------------------------------------------------------------------------------------------------------+
```

---

### 2. Bảng Mô Tả Tín Hiệu & Routing Component (LEVEL 3 Signal Table)

| Tên Tín hiệu (Signal) | Hướng | Độ rộng bit | Mô tả Chức năng & Định tuyến Routing |
| ------ | ------ | ------ | ------ |
| clk / rst_n | Input | 1 / 1 bit | Clock hệ thống 200 MHz và Reset bất đồng bộ mức thấp. |
| raw_score_in | Input | 32 bits | Điểm tích vô hướng thô $\text{INT32}$ từ mảng Systolic Stage 1. |
| shift_val | Input | 4 bits | Số bit dịch phải đại số $\text{ASR}$ để chuẩn hóa $1/\sqrt{d_k}$ (mặc định $= 2$ hoặc $3$). |
| scaled_score_out | Internal | 16 bits | Điểm chú ý đã thu phóng $\text{INT16}$ cấp cho Softmax Engine. |
| softmax_start | Internal | 1 bit | Xung kích hoạt chu kỳ tính Softmax cho dòng token mới. |
| lut_rom_addr | Internal | 10 bits | Địa chỉ tra bảng BRAM ROM chứa giá trị hàm mũ $e^x$. |
| lut_rom_data | Internal | 16 bits | Dữ liệu hàm mũ UINT16 đọc ra từ BRAM ROM. |
| sum_exp_acc | Internal | 32 bits | Bộ tích lũy mẫu số $\sum e^{\Delta S}$ dạng UINT32. |
| prob_out | Output | 8 bits | Xác suất Attention đã chuẩn hóa UINT8 cấp cho Stage 4 GEMM. |
| i_f_data / i_x_data | Input | 8 / 8 bits | Tín hiệu luồng chính MSA ($I_F$) và luồng đường tắt Shortcut ($I_X$) dạng INT8. |
| m_f / e_f | Input | 16 / 5 bits | Hệ số nhân Dyadic $M_F$ và số bit dịch $e_F$ cho luồng $I_F$. |
| m_x / e_x | Input | 16 / 5 bits | Hệ số nhân Dyadic $M_X$ và số bit dịch $e_X$ cho luồng $I_X$. |
| res_add_out | Output | 8 bits | Kết quả cộng đường tắt đã kẹp bão hòa $\text{INT8} \in [-128, +127]$. |

---

### 3. Micro-Architecture của Đơn Vị Scaler (scale_unit.sv)

Mạch dịch bit đại số Barrel Shifter tiêu tốn **0 khối DSP**, chuẩn hóa chuỗi dữ liệu trong 1 chu kỳ clock:

```text
========================================================================================================================
                                    SCALER UNIT PIPELINE (scale_unit.sv)
========================================================================================================================

    Raw Accumulator Score: In_Score [31:0] (INT32)        Shift Value: Shift_Val [3:0] (0..15)
              │                                                     │
              v                                                     v
    +-----------------------------------------------------------------------------------+
    | Arithmetic Right Barrel Shifter (ASR)                                             |
    | Shifted_Score = In_Score >>> Shift_Val                                           |
    +-----------------------------------------┬-----------------------------------------+
                                              │
                                              v
    +-----------------------------------------------------------------------------------+
    | Saturating Clamp to 16-bit Signed Range                                           |
    |   if (Shifted_Score > 32767)       Clamped_Score = 32767                          |
    |   else if (Shifted_Score < -32768)  Clamped_Score = -32768                         |
    |   else                             Clamped_Score = Shifted_Score[15:0]            |
    +-----------------------------------------┬-----------------------------------------+
                                              │
                                              v
                                 Scaled_Score_Out [15:0] (INT16)
```

---

### 4. Kiến Trúc Chi Tiết Hardware Softmax Engine (softmax_lut.sv)

Sơ đồ FSMD 4 giai đoạn xử lý Softmax số nguyên bảo toàn độ chính xác với BRAM Exp ROM và bộ chia Dyadic Reciprocal:

```text
========================================================================================================================
                                HARDWARE SOFTMAX ENGINE DATAPATH (softmax_lut.sv)
========================================================================================================================

  Stage 1: Max Finder         Stage 2: Shift & BRAM ROM      Stage 3: Denominator Acc      Stage 4: Dyadic Div
  +-------------------+       +-------------------+          +-------------------+         +-------------------+
  | Scan Row Score[i] |======>| Sub = S_i - S_max |=========>| Exp_Val = ROM[Sub]|========>| Inv_Sum = 2^e/Sum |
  | Find S_max        |       | (Guaranteed <= 0) |          | Sum += Exp_Val    |         | Prob = Exp*Inv>>>e|
  +-------------------+       +-------------------+          +-------------------+         +---------┬---------+
                                                                                                     │
                                                                                                     v
                                                                                        Attention Prob [7:0] (UINT8)
```

#### ASMD Chart Điều Khiển Softmax Engine (softmax_lut_fsm.sv):

```text
                               +-------------------+
                               |     ST_SM_IDLE    | <--------------------+
                               +---------+---------+                      |
                                         |                                |
                                (softmax_start == 1)                      |
                                 /               \                        |
                               YES                NO                      |
                               /                    \                     |
                              v                      +--------------------+
                     +-------------------+
                     |   ST_FIND_MAX     |
                     | S_max <= Max(S_i) |
                     | cnt   <= 0        |
                     +---------+---------+
                               |
                               v
                     (cnt == N - 1?)
                      /                               YES            NO
                    /                                   v                  v
         +-------------------+   (cnt <= cnt + 1)
         |   ST_EXP_ACC      |
         | addr <= S_i-S_max |
         | Sum  <= Sum + ROM |
         +---------+---------+
                   |
                   v
         (cnt == N - 1?)
          /                   YES            NO
        /                       v                  v
 +---------------+   (cnt <= cnt + 1)
 | ST_DYADIC_DIV |
 | Inv <= 2^e/Sum|
 +-------+-------+
         |
         v
 +---------------+
 | ST_NORM_OUT   |
 | P_i<=Exp*Inv  |
 +-------+-------+
         |
         v
     ST_SM_IDLE
```

---

### 5. Vi Mạch Re-quantization & Residual Adder (requant_residual_add.sv)

Mạch cộng đường tắt (Shortcut Connection) xử lý lệch Scale Factor ($S_F \neq S_X$) bằng phép nhân Dyadic Rescaling và kẹp ngưỡng bão hòa $\text{INT8}$:

```text
========================================================================================================================
                          RE-QUANTIZATION & RESIDUAL ADDER ARCHITECTURE
========================================================================================================================

   MSA Stream I_F [7:0]     Dyadic Scale M_F, e_F      Shortcut Stream I_X [7:0]      Dyadic Scale M_X, e_X
           │                            │                            │                            │
           v                            v                            v                            v
   +---------------+            +---------------+            +---------------+            +---------------+
   | Register Reg0 |            | Register Reg0 |            | Register Reg0 |            | Register Reg0 |
   +-------┬-------+            +-------┬-------+            +-------┬-------+            +-------┬-------+
           │                            │                            │                            │
           +--------------┬-------------+                            +--------------┬-------------+
                          │                                                         │
                          v (40-bit Dyadic Product)                                 v (40-bit Dyadic Product)
               +----------------------+                                  +----------------------+
               | Prod_F = I_F * M_F   |                                  | Prod_X = I_X * M_X   |
               +----------┬-----------+                                  +----------┬-----------+
                          │                                                         │
                          v (Shift >>> e_F)                                         v (Shift >>> e_X)
               +----------------------+                                  +----------------------+
               | Scaled_F = Prod_F>>e_F|                                 | Scaled_X = Prod_X>>e_X|
               +----------┬-----------+                                  +----------┬-----------+
                          │                                                         │
                          +-------------------------┬-------------------------------+
                                                    │
                                                    v (Full Add & Saturating Clamp)
                                         +---------------------+
                                         | Sum_Full = F + X    |
                                         | Clamp_INT8          |
                                         |   [-128, +127]      |
                                         +----------┬----------+
                                                    |
                                                    v
                                         Res_Out [7:0] (INT8)
```

```verilog
module requant_residual_add (
    input  logic        clk,
    input  logic        rst_n,
    // Input Streams
    input  logic signed [7:0]  i_f_data,   // Main Feature Stream (MSA/MLP)
    input  logic signed [7:0]  i_x_data,   // Residual Shortcut Stream
    // Dyadic Scale Parameters
    input  logic signed [15:0] m_f,
    input  logic        [4:0]  e_f,
    input  logic signed [15:0] m_x,
    input  logic        [4:0]  e_x,
    // Output
    output logic signed [7:0]  res_out
);

    logic signed [23:0] prod_f, prod_x;
    logic signed [23:0] scaled_f, scaled_x;
    logic signed [24:0] sum_full;
    logic signed [7:0]  clamped_out;

    // Pipeline Stage 1: Dyadic Multiplication
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_f <= '0;
            prod_x <= '0;
        end else begin
            prod_f <= 24'(i_f_data) * 24'(m_f);
            prod_x <= 24'(i_x_data) * 24'(m_x);
        end
    end

    // Pipeline Stage 2: Arithmetic Right Shift & Addition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_full <= '0;
        end else begin
            scaled_f <= prod_f >>> e_f;
            scaled_x <= prod_x >>> e_x;
            sum_full <= 25'(scaled_f) + 25'(scaled_x);
        end
    end

    // Pipeline Stage 3: Saturating Clamp to INT8 [-128, +127]
    always_comb begin
        if (sum_full > 25'sd127)
            clamped_out = 8'sd127;
        else if (sum_full < -25'sd128)
            clamped_out = -8'sd128;
        else
            clamped_out = sum_full[7:0];
    end

    assign res_out = clamped_out;

endmodule
```

---

### 🏆 TỔNG KẾT TOÀN BỘ BỘ SƠ ĐỒ RTL HIERARCHY

Toàn bộ hệ thống vi mạch **Multi-Head Attention (MHA) Accelerator** đã được phân rã thành công từ tổng quan đến chi tiết vi mô qua 4 cấp độ:

1. **LEVEL 0**: Vỏ bọc hệ thống `attention_core_top.sv`, Giao diện bus AXI4-Lite / AXI4-Stream, Sơ đồ ASMD Chart trung tâm và Mô hình Băng thông Bus.
2. **LEVEL 1**: Sub-system bộ nhớ `memory_streaming_subsys.sv`, Mạch chống nghẽn Skid Buffer `axis_adapter.sv`, và FSM tráo đệm BRAM Ping-Pong `ping_pong_bram_buffer.sv`.
3. **LEVEL 2**: Mảng tính toán GEMM `systolic_mac_array.sv` (Output-Stationary), Mạch PE Slice DSP Packing 2x INT8 `pe_slice_dsp_packing.sv`, và Cây cộng đường ống `pipelined_adder_tree.sv`.
4. **LEVEL 3**: Khối xử lý phi tuyến `nonlinear_pipeline_top.sv` gồm Scaler Unit `scale_unit.sv` (0 DSPs), Softmax Engine `softmax_lut.sv` (Max-sub + Exp ROM), và Re-quantization Residual Adder `requant_residual_add.sv`.
