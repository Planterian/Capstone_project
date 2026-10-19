# Khối Scaler Unit v2.0 (Dyadic Rescaler)

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/khoi_scaler_unit.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/khoi_scaler_unit.md)

---

## 1. VAI TRÒ KỸ THUẬT & SỰ THAY THẾ TỪ ASR SANG DYADIC SCALING

Trong thiết kế ban đầu (`Notes/khoi_scaler_unit.md`), việc thu phóng điểm số chú ý $1/\sqrt{d_k}$ được thực hiện bằng phép dịch phải đại số đơn giản (ASR). Tuy nhiên:
1. Đối với **Apple MobileViT-XXS**, các tầng Transformer có $d_k \in \{16, 20, 24\}$:
   - $d_k = 16 \implies \sqrt{16} = 4 = 2^2$ (có thể dịch phải 2 bit).
   - $d_k = 20 \implies \sqrt{20} \approx 4.472$ (không thể biểu diễn bằng dịch bit).
   - $d_k = 24 \implies \sqrt{24} \approx 4.899$ (không thể biểu diễn bằng dịch bit).
2. Khi áp dụng lượng tử hóa số nguyên **I-ViT (Integer-only Vision Transformer)**, các ma trận $Q, K$ và điểm Score đều có các hệ số tỷ lệ lượng tử hóa riêng biệt ($S_Q, S_K, S_{\text{score}}$). Tỷ lệ hiệu dụng thực tế là:
   $$S_{\text{effective}} = \frac{S_Q \cdot S_K}{S_{\text{score}} \cdot \sqrt{d_k}}$$
   Hệ số này là một số thực bất kỳ (ví dụ $0.006590$ cho Stage 2). Do đó, phần cứng bắt buộc phải sử dụng **Phép nhân tỷ lệ Dyadic (Dyadic Rescaler)**:
   $$S_{\text{effective}} \approx M \times 2^{-E}$$
   trong đó $M$ là phần định trị nguyên 16-bit ($M \in [0, 32767]$) và $E$ là số mũ dịch bit ($E \in [0, 31]$).

---

## 2. SƠ ĐỒ KIẾN TRÚC VI MẠCH (scale_unit.sv)

```text
========================================================================================================================
                                     DYADIC SCALER UNIT PIPELINE (scale_unit.sv)
========================================================================================================================

    Raw Score: in_score [31:0] (INT32)              Mantissa: scale_m [15:0] (INT16)
               │                                                    │
               v                                                    v
    +-----------------------------------------------------------------------------------+
    | Pipeline Stage 1: 48-bit Signed Multiplication                                    |
    | prod_stage = 48'($signed(in_score)) * 48'($signed(scale_m))                        |
    +-----------------------------------------┬-----------------------------------------+
                                              │
                                              v  [47:0] Intermediate Product
                                              │                                 Exponent: scale_e [4:0] (0..31)
                                              │                                                 │
                                              v                                                 v
    +-----------------------------------------------------------------------------------+
    | Pipeline Stage 2: Arithmetic Right Shift & Saturated Clamping                     |
    | shifted_score = prod_stage >>> scale_e                                            |
    |   if (shifted_score > 32767)       clamped_score = 16'sd32767                     |
    |   else if (shifted_score < -32768)  clamped_score = -16'sd32768                   |
    |   else                             clamped_score = 16'(shifted_score)             |
    +-----------------------------------------┬-----------------------------------------+
                                              │
                                              v
                                 scaled_score [15:0] (Signed INT16)
```

---

## 3. THAM SỐ DYADIC CHUẨN CHO 3 STAGES TRÊN MOBILEVIT-XXS

| Transformer Stage | Kích thước $d_k$ | Tỷ lệ hiệu dụng $S_{\text{eff}}$ | Mantissa $M$ (Thập phân) | Mantissa $M$ (Hex) | Exponent $E$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Stage 2** | 16 | 0.006590 | **27,640** | `0x6BF8` | 22 |
| **Stage 3** | 20 | 0.005230 | **21,936** | `0x55B0` | 22 |
| **Stage 4** | 24 | 0.004485 | **18,810** | `0x497A` | 22 |

Các tham số này được CPU ARM PS nạp động vào các thanh ghi AXI-Lite CSR tại địa chỉ `0x20` (SCALE_M) và `0x24` (SCALE_E), cho phép cùng một phần cứng xử lý linh hoạt cả 3 Stages của mô hình MobileViT.

---

## 4. MÃ NGUỒN RTL SYSTEMVERILOG CHUẨN HÓA

```systemverilog
`timescale 1ns / 1ps
// Module Name: scale_unit (v2.0)
module scale_unit #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        in_valid,
    input  logic signed [IN_WIDTH-1:0]  in_score,
    input  logic signed [15:0]          scale_m,
    input  logic [4:0]                  scale_e,
    input  logic [3:0]                  shift_val,
    output logic                        out_valid,
    output logic signed [OUT_WIDTH-1:0] scaled_score
);

    logic signed [47:0]          prod_stage;
    logic signed [47:0]          shifted_score;
    logic signed [OUT_WIDTH-1:0] clamped_score;
    logic [1:0]                  valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_stage    <= '0;
            valid_pipe[0] <= 1'b0;
        end else begin
            valid_pipe[0] <= in_valid;
            if (scale_m != 16'sd0)
                prod_stage <= 48'($signed(in_score)) * 48'($signed(scale_m));
            else
                prod_stage <= 48'($signed(in_score));
        end
    end

    always_comb begin
        if (scale_m != 16'sd0)
            shifted_score = prod_stage >>> scale_e;
        else
            shifted_score = prod_stage >>> shift_val;

        if (shifted_score > 48'sd32767)
            clamped_score = 16'sd32767;
        else if (shifted_score < -48'sd32768)
            clamped_score = -16'sd32768;
        else
            clamped_score = 16'(shifted_score);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_score  <= '0;
            valid_pipe[1] <= 1'b0;
        end else begin
            scaled_score  <= clamped_score;
            valid_pipe[1] <= valid_pipe[0];
        end
    end

    assign out_valid = valid_pipe[1];

endmodule
```
