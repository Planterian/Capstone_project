# Khối Scaler Unit

Khối **Scaler Unit (scale_unit.sv)** đóng vai trò thu phóng điểm số chú ý (Attention Score) trong tầng **Multi-Head Attention (MHA) Engine**, nằm ngay giữa mảng nhân GEMM Stage 1 ($QK^T$) và khối Hardware Softmax.

---

##### 1. Vai Trò Kỹ Thuật & Nguyên Lý Tối Ưu (0 DSPs)
* **Mục đích toán học**: Trong cơ chế Scaled Dot-Product Attention, các điểm số tích vô hướng $QK^T$ cần được thu phóng theo hệ số $\frac{1}{\sqrt{d_k}}$ để kiểm soát phương sai, tránh đẩy hàm Softmax vào vùng triệt tiêu gradient (vanishing gradient).
* **Tối ưu phần cứng FPGA**: Trong số học dấu phẩy động, đây là phép chia cho $\sqrt{d_k}$. Tuy nhiên trên phần cứng FPGA, phép chia số thực rất tốn kém tài nguyên logic/FPU. Nhờ cố định $d_k$ ở các giá trị chuẩn phần cứng (như $d_k = 32$ với $\sqrt{32} \approx 5.65 \implies 2^2$ hoặc $2^3$), khối Scaler Unit biến đổi toàn bộ phép chia thành **phép dịch bit phải đại số (Arithmetic Right-Shift - ASR)**.
* **Hiệu năng phần cứng**: Tiêu tốn **0 khối DSP**, đạt trễ **1 chu kỳ clock** với nhịp khởi tạo **$II = 1$**.

---

##### 2. Sơ Đồ Kiến Trúc Vi Mạch (Micro-Architecture Diagram)
Khối Scaler Unit bao gồm 2 công đoạn chính: **Barrel Shifter đại số (ASR)** và **Mạch kẹp ngưỡng bão hòa (Saturating Clamp)**:

```text
========================================================================================================================
                                    SCALER UNIT PIPELINE (scale_unit.sv)
========================================================================================================================

    Raw Accumulator Score: in_score [31:0] (Signed INT32)     Shift Value: shift_val [3:0] (0..15)
              │                                                         │
              v                                                         v
    +-----------------------------------------------------------------------------------+
    | Arithmetic Right Barrel Shifter (ASR)                                             |
    | shifted_score = in_score >>> shift_val (Bảo toàn bit dấu negative)                |
    +-----------------------------------------┬-----------------------------------------+
                                              │
                                              v  [31:0]
    +-----------------------------------------------------------------------------------+
    | Saturating Clamp to 16-bit Signed Range                                           |
    |   if (shifted_score > 32767)       clamped_score = 32767                          |
    |   else if (shifted_score < -32768)  clamped_score = -32768                        |
    |   else                             clamped_score = shifted_score[15:0]            |
    +-----------------------------------------┬-----------------------------------------+
                                              │
                                              v
                                 scaled_score_out [15:0] (Signed INT16)
```

---

##### 3. Bảng Tín Hiệu & Định Tuyến Routing (Interface Signals)

| Tên Tín hiệu (Signal) | Hướng | Độ rộng bit | Mô tả Chức năng & Routing |
| ------ | ------ | ------ | ------ |
| clk | Input | 1 bit | Xung clock hệ thống PL (danh định 200 MHz). |
| rst_n | Input | 1 bit | Reset bất đồng bộ tích cực mức thấp. |
| in_valid | Input | 1 bit | Cờ báo điểm tích vô hướng INT32 từ GEMM Stage 1 hợp lệ. |
| in_score | Input | 32 bits | Điểm tích vô hướng thô **Signed INT32** từ bộ tích lũy Accumulator. |
| shift_val | Input | 4 bits | Số bit dịch phải đại số $\text{ASR}$ (mặc định = 2 hoặc 3 tùy thuộc $\sqrt{d_k}$). |
| out_valid | Output | 1 bit | Cờ báo điểm chú ý đã thu phóng hợp lệ cấp cho Softmax Engine. |
| scaled_score | Output | 16 bits | Điểm chú ý thu phóng **Signed INT16** đã kẹp ngưỡng bão hòa. |

---

##### 4. Mã Nguồn RTL SystemVerilog Chuẩn (scale_unit.sv)

```verilog
`timescale 1ns / 1ps
// ============================================================================
// Module Name:   scale_unit
// Description:  Arithmetic Right Shift (ASR) Scaler Unit converting INT32
//               GEMM scores into INT16 inputs for Softmax with 0 DSPs.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+
// ============================================================================

module scale_unit #(
    parameter int IN_WIDTH  = 32,   // Độ rộng bit điểm tích lũy thô (INT32)
    parameter int OUT_WIDTH = 16    // Độ rộng bit điểm đầu ra cho Softmax (INT16)
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,
    input  logic signed [IN_WIDTH-1:0]  in_score,
    input  logic        [3:0]       shift_val,  // Số bit dịch (ví dụ: 2 hoặc 3)

    output logic                    out_valid,
    output logic signed [OUT_WIDTH-1:0] scaled_score
);

    logic signed [IN_WIDTH-1:0] shifted_score;
    logic signed [OUT_WIDTH-1:0] clamped_score;
    logic                        valid_reg;

    assign shifted_score = in_score >>> shift_val;

    always_comb begin
        if (shifted_score > 32'sd32767)
            clamped_score = 16'sd32767;
        else if (shifted_score < -32'sd32768)
            clamped_score = -16'sd32768;
        else
            clamped_score = shifted_score[OUT_WIDTH-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_score <= '0;
            out_valid    <= 1'b0;
        end else begin
            scaled_score <= clamped_score;
            out_valid    <= in_valid;
        end
    end

endmodule
```

---

##### 5. Điểm Mạnh Kiến Trúc
1. **Bảo toàn bit dấu (Sign Extension)**: Phép dịch $signed(in_score) >>> shift_val$ giữ nguyên bit MSB (bit dấu), đảm bảo chính xác các giá trị âm sau khi nhân tích vô hướng.
2. **Khóa tràn dải (Saturating Logic)**: Giúp điểm số khi thu phóng từ INT32 về INT16 không bị hiện tượng lật dấu (wrap-around) làm hỏng phân bố xác suất của Softmax Engine.
3. **Triệt tiêu tài nguyên DSP**: Hoàn toàn dùng khối dịch bit Barrel Shifter trên logic fabric (LUT/CLB), giải phóng $100\%$ khối phần cứng DSP48E2 cho mảng nhân Systolic MAC Array.
