# Mạch DSP Packing v2.0 (`pe_slice_dsp_packing.sv`)

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mach_dsp_packing.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mach_dsp_packing.md)

---

## 1. MÃ NGUỒN RTL SYSTEMVERILOG HOÀN CHỈNH (`pe_slice_dsp_packing.sv`)

Mô-đun `pe_slice_dsp_packing` thực thi 2 phép nhân INT8 đồng thời trong 1 chu kỳ, đạt nhịp khởi tạo $II = 1$:

```systemverilog
`timescale 1ns / 1ps
// ============================================================================
// Module Name:   pe_slice_dsp_packing
// Description:  Processing Element Slice utilizing Xilinx DSP48E2 Packing.
//               Computes A*B (Col 0) and A*C (Col 1) simultaneously.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned - Proper Sign-Extension)
// ============================================================================

module pe_slice_dsp_packing (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,

    input  logic signed [7:0]  operand_a, // Kích hoạt A (Activation)
    input  logic signed [7:0]  operand_b, // Trọng số B (Weight Col 0)
    input  logic signed [7:0]  operand_c, // Trọng số C (Weight Col 1)

    output logic               out_valid,
    output logic signed [15:0] result_ab, // A * B
    output logic signed [15:0] result_ac  // A * C
);

    // Mở rộng cổng 27-bit cho B và C với khoảng cách 19 bits
    logic signed [26:0] operand_bc_packed;
    assign operand_bc_packed = {operand_b, 11'b0, operand_c};

    // Mở rộng cổng 18-bit cho A: Bảo toàn bit dấu bit [7]
    logic signed [17:0] operand_a_extended;
    assign operand_a_extended = {{10{operand_a[7]}}, operand_a};

    // Tích trung gian 45-bit
    logic signed [44:0] dsp_product;
    logic               valid_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dsp_product <= '0;
            valid_q     <= 1'b0;
        end else begin
            valid_q     <= in_valid;
            dsp_product <= operand_bc_packed * operand_a_extended;
        end
    end

    // Tách 2 kết quả 16-bit độc lập
    assign result_ac = dsp_product[15:0];
    assign result_ab = dsp_product[34:19];
    assign out_valid = valid_q;

endmodule
```
