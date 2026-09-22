`timescale 1ns / 1ps
// ============================================================================
// Module Name:   pipelined_adder_tree
// Description:  Pipelined Binary Reduction Adder Tree for Systolic Array row accumulation.
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module pipelined_adder_tree #(
    parameter int N          = 4,
    parameter int IN_WIDTH   = 8,
    parameter int OUT_WIDTH  = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     in_valid,
    input  logic signed [IN_WIDTH-1:0] a [0:N-1],
    input  logic signed [IN_WIDTH-1:0] b [0:N-1],

    output logic                     out_valid,
    output logic signed [OUT_WIDTH-1:0] dot_product
);

    logic signed [2*IN_WIDTH-1:0] stage0_prod [0:N-1];
    logic                         stage0_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage0_valid <= 1'b0;
            for (int i = 0; i < N; i++) stage0_prod[i] <= '0;
        end else begin
            stage0_valid <= in_valid;
            for (int i = 0; i < N; i++) begin
                stage0_prod[i] <= a[i] * b[i];
            end
        end
    end

    logic signed [2*IN_WIDTH:0] stage1_sum [0:(N/2)-1];
    logic                       stage1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 1'b0;
            for (int i = 0; i < N/2; i++) stage1_sum[i] <= '0;
        end else begin
            stage1_valid <= stage0_valid;
            for (int i = 0; i < N/2; i++) begin
                stage1_sum[i] <= $signed(stage0_prod[2*i]) + $signed(stage0_prod[2*i+1]);
            end
        end
    end

    logic signed [OUT_WIDTH-1:0] stage2_root_sum;
    logic                        stage2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_valid    <= 1'b0;
            stage2_root_sum <= '0;
        end else begin
            stage2_valid    <= stage1_valid;
            stage2_root_sum <= $signed(stage1_sum[0]) + $signed(stage1_sum[1]);
        end
    end

    assign dot_product = stage2_root_sum;
    assign out_valid   = stage2_valid;

endmodule
