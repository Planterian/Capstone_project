`timescale 1ns / 1ps
// ============================================================================
// Module Name:   pipelined_adder_tree
// Description:  Pipelined Reduction Adder Tree supporting both pure addition
//               (HAS_MULT = 0) and dot-product reduction (HAS_MULT = 1) across N inputs.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned)
// ============================================================================

module pipelined_adder_tree #(
    parameter int N         = 4,
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 32,
    parameter bit HAS_MULT  = 1'b0
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        in_valid,
    input  logic signed [IN_WIDTH-1:0]  a [0:N-1],
    input  logic signed [IN_WIDTH-1:0]  b [0:N-1],

    output logic                        out_valid,
    output logic signed [OUT_WIDTH-1:0] dot_product
);

    // Pipeline registers for Dot-Product mode (HAS_MULT = 1)
    logic signed [2*IN_WIDTH-1:0] mult_stage [0:N-1];
    logic                         mult_valid;

    // Pipeline stage 1: Pairwise additions
    logic signed [OUT_WIDTH-1:0]  stage1_sum [0:(N/2)-1];
    logic                         stage1_valid;

    // Pipeline stage 2: Final root sum
    logic signed [OUT_WIDTH-1:0]  stage2_root_sum;
    logic                         stage2_valid;

    generate
        if (HAS_MULT) begin : gen_mult_tree
            // Multiplier stage
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mult_valid <= 1'b0;
                    for (int i = 0; i < N; i++) mult_stage[i] <= '0;
                end else begin
                    mult_valid <= in_valid;
                    for (int i = 0; i < N; i++) begin
                        mult_stage[i] <= a[i] * b[i];
                    end
                end
            end

            // Stage 1 addition
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    stage1_valid <= 1'b0;
                    for (int i = 0; i < N/2; i++) stage1_sum[i] <= '0;
                end else begin
                    stage1_valid <= mult_valid;
                    for (int i = 0; i < N/2; i++) begin
                        stage1_sum[i] <= $signed(mult_stage[2*i]) + $signed(mult_stage[2*i+1]);
                    end
                end
            end
        end else begin : gen_add_only_tree
            // Direct adder tree without multipliers (pure addition of vector a)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    stage1_valid <= 1'b0;
                    for (int i = 0; i < N/2; i++) stage1_sum[i] <= '0;
                end else begin
                    stage1_valid <= in_valid;
                    for (int i = 0; i < N/2; i++) begin
                        stage1_sum[i] <= $signed(a[2*i]) + $signed(a[2*i+1]);
                    end
                end
            end
        end
    endgenerate

    // Final stage: Root summation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_valid    <= 1'b0;
            stage2_root_sum <= '0;
        end else begin
            stage2_valid    <= stage1_valid;
            if (N == 4) begin
                stage2_root_sum <= $signed(stage1_sum[0]) + $signed(stage1_sum[1]);
            end else if (N == 2) begin
                stage2_root_sum <= $signed(stage1_sum[0]);
            end else begin
                stage2_root_sum <= $signed(stage1_sum[0]) + $signed(stage1_sum[1]);
            end
        end
    end

    assign dot_product = stage2_root_sum;
    assign out_valid   = stage2_valid;

endmodule
