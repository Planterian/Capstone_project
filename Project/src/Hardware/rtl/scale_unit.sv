`timescale 1ns / 1ps
// ============================================================================
// Module Name:   scale_unit
// Description:  Arithmetic Right Shift (ASR) Scaler Unit converting INT32
//               GEMM scores into INT16 inputs for Softmax with 0 DSPs.
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module scale_unit #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,
    input  logic signed [IN_WIDTH-1:0] in_score,
    input  logic        [3:0]       shift_val,

    output logic                    out_valid,
    output logic signed [OUT_WIDTH-1:0] scaled_score
);

    logic signed [IN_WIDTH-1:0]  shifted_score;
    logic signed [OUT_WIDTH-1:0] clamped_score;

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
