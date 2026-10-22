`timescale 1ns / 1ps
// ============================================================================
// Module Name:   requant_residual_add
// Description:  Dyadic Rescaling and Saturating Residual Adder for Shortcut.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned)
// ============================================================================

module requant_residual_add (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,

    input  logic signed [7:0]  i_f_data,
    input  logic signed [7:0]  i_x_data,

    input  logic signed [15:0] m_f,
    input  logic        [4:0]  e_f,
    input  logic signed [15:0] m_x,
    input  logic        [4:0]  e_x,

    output logic        out_valid,
    output logic signed [7:0]  res_out
);

    logic signed [23:0] prod_f, prod_x;
    logic signed [24:0] sum_full;
    logic signed [7:0]  clamped_out;
    logic [2:0]         valid_pipe;

    // Pipeline Stage 1: Multiplication
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_f <= '0;
            prod_x <= '0;
        end else begin
            prod_f <= 24'(i_f_data) * 24'(m_f);
            prod_x <= 24'(i_x_data) * 24'(m_x);
        end
    end

    // Pipeline Stage 2: Dyadic shift and addition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_full <= '0;
        end else begin
            sum_full <= 25'(prod_f >>> e_f) + 25'(prod_x >>> e_x);
        end
    end

    // Pipeline Stage 3: Saturated clamping and registered output
    always_comb begin
        if (sum_full > 25'sd127)
            clamped_out = 8'sd127;
        else if (sum_full < -25'sd128)
            clamped_out = -8'sd128;
        else
            clamped_out = 8'(sum_full);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= '0;
            res_out    <= '0;
        end else begin
            valid_pipe <= {valid_pipe[1:0], in_valid};
            res_out    <= clamped_out;
        end
    end

    assign out_valid = valid_pipe[2];

endmodule
