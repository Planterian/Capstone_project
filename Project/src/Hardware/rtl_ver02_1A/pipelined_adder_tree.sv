`timescale 1ns / 1ps
// ============================================================================
// Module Name:   pipelined_adder_tree
// Description:  Pipelined Reduction Tree supporting 16-element INT8xINT8 signed
//               multiplications and 4-stage binary adder reduction into INT32.
//               Architecture: 100% FPGA Logic Fabric (0 DSP48E2 used).
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v1.0 (Variant 1A Pure Fabric Multipliers)
// ============================================================================

module pipelined_adder_tree #(
    parameter int N         = 16,
    parameter int IN_WIDTH  = 8,
    parameter int OUT_WIDTH = 32,
    parameter bit HAS_MULT  = 1'b1
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                in_valid,
    input  logic signed [IN_WIDTH-1:0]          a [0:N-1],
    input  logic signed [IN_WIDTH-1:0]          b [0:N-1],

    output logic                                out_valid,
    output logic signed [OUT_WIDTH-1:0]         dot_product
);

    // ------------------------------------------------------------------------
    // Stage 0: 16 Parallel Multipliers (INT8 x INT8 -> INT16 / INT32)
    // ------------------------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] mult_stage [0:N-1];
    logic                        mult_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_valid <= 1'b0;
            for (int i = 0; i < N; i++) mult_stage[i] <= '0;
        end else begin
            mult_valid <= in_valid;
            for (int i = 0; i < N; i++) begin
                if (HAS_MULT) begin
                    mult_stage[i] <= 32'($signed(a[i])) * 32'($signed(b[i]));
                end else begin
                    mult_stage[i] <= 32'($signed(a[i]));
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // Stage 1: 8 Pairwise Adders (16 -> 8)
    // ------------------------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] stage1_sum [0:(N/2)-1];
    logic                        stage1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 1'b0;
            for (int i = 0; i < N/2; i++) stage1_sum[i] <= '0;
        end else begin
            stage1_valid <= mult_valid;
            for (int i = 0; i < N/2; i++) begin
                stage1_sum[i] <= mult_stage[2*i] + mult_stage[2*i+1];
            end
        end
    end

    // ------------------------------------------------------------------------
    // Stage 2: 4 Pairwise Adders (8 -> 4)
    // ------------------------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] stage2_sum [0:(N/4)-1];
    logic                        stage2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_valid <= 1'b0;
            for (int i = 0; i < N/4; i++) stage2_sum[i] <= '0;
        end else begin
            stage2_valid <= stage1_valid;
            for (int i = 0; i < N/4; i++) begin
                stage2_sum[i] <= stage1_sum[2*i] + stage1_sum[2*i+1];
            end
        end
    end

    // ------------------------------------------------------------------------
    // Stage 3: 2 Pairwise Adders (4 -> 2)
    // ------------------------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] stage3_sum [0:(N/8)-1];
    logic                        stage3_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage3_valid <= 1'b0;
            for (int i = 0; i < N/8; i++) stage3_sum[i] <= '0;
        end else begin
            stage3_valid <= stage2_valid;
            for (int i = 0; i < N/8; i++) begin
                stage3_sum[i] <= stage2_sum[2*i] + stage2_sum[2*i+1];
            end
        end
    end

    // ------------------------------------------------------------------------
    // Stage 4: 1 Final Root Adder (2 -> 1)
    // ------------------------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] stage4_root_sum;
    logic                        stage4_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage4_valid    <= 1'b0;
            stage4_root_sum <= '0;
        end else begin
            stage4_valid    <= stage3_valid;
            stage4_root_sum <= stage3_sum[0] + stage3_sum[1];
        end
    end

    assign dot_product = stage4_root_sum;
    assign out_valid   = stage4_valid;

endmodule
