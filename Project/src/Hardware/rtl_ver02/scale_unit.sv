`timescale 1ns / 1ps
// ============================================================================
// Module Name:   scale_unit
// Description:  Dyadic Rescaler Unit converting INT32 GEMM scores into INT16
//               inputs for Softmax using (in_score * M) >>> E with saturation.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned - Dyadic Scaling)
// ============================================================================

module scale_unit #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        in_valid,
    input  logic signed [IN_WIDTH-1:0]  in_score,

    // Dyadic scaling parameters (from AXI-Lite CSR)
    input  logic signed [15:0]          scale_m,    // 16-bit signed mantissa
    input  logic [4:0]                  scale_e,    // 5-bit shift exponent
    input  logic [3:0]                  shift_val,  // Legacy ASR fallback

    output logic                        out_valid,
    output logic signed [OUT_WIDTH-1:0] scaled_score
);

    // 48-bit intermediate product to prevent overflow
    logic signed [47:0]          prod_stage;
    logic signed [47:0]          shifted_score;
    logic signed [OUT_WIDTH-1:0] clamped_score;
    logic [1:0]                  valid_pipe;

    // Pipeline Stage 1: Multiplication (DSP or Logic)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_stage    <= '0;
            valid_pipe[0] <= 1'b0;
        end else begin
            valid_pipe[0] <= in_valid;
            if (scale_m != 16'sd0) begin
                prod_stage <= 48'($signed(in_score)) * 48'($signed(scale_m));
            end else begin
                prod_stage <= 48'($signed(in_score)); // Fallback
            end
        end
    end

    // Pipeline Stage 2: Arithmetic Right Shift and INT16 Clamping
    always_comb begin
        if (scale_m != 16'sd0) begin
            shifted_score = prod_stage >>> scale_e;
        end else begin
            shifted_score = prod_stage >>> shift_val;
        end

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
