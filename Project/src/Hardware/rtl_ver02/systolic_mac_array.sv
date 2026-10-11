`timescale 1ns / 1ps
// ============================================================================
// Module Name:   systolic_mac_array
// Description:  2D Output-Stationary Systolic MAC Array Core integrating
//               pe_slice_dsp_packing nodes and pipelined_adder_tree reduction.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned - 32-bit Accumulator Preserved)
// ============================================================================

module systolic_mac_array #(
    parameter int ARRAY_SIZE = 4,
    parameter int IN_WIDTH   = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        gemm_start,
    input  logic                        clr_acc,
    input  logic                        in_valid,
    input  logic                        mode_sel, // 0: Stage 1 (QK^T), 1: Stage 2 (SV)

    input  logic signed [IN_WIDTH-1:0]  act_row [0:ARRAY_SIZE-1],
    input  logic signed [IN_WIDTH-1:0]  wt_col0 [0:ARRAY_SIZE-1],
    input  logic signed [IN_WIDTH-1:0]  wt_col1 [0:ARRAY_SIZE-1],

    output logic                        out_valid,
    output logic signed [ACC_WIDTH-1:0] dot_product_out [0:ARRAY_SIZE-1]
);

    // Systolic Skew Pipelines (Clean procedural registers)
    logic signed [IN_WIDTH-1:0]  act_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt0_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt1_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Wavefront valid propagation
    localparam int WAVE_LEN = 2 * ARRAY_SIZE + 4;
    logic [WAVE_LEN-1:0]         valid_wave;
    logic signed [ACC_WIDTH-1:0] pe_acc_matrix [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_wave <= '0;
        end else begin
            valid_wave <= {valid_wave[WAVE_LEN-2:0], in_valid};
        end
    end

    // 2D Processing Element Grid
    genvar r, c;
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_pe_row
            for (c = 0; c < ARRAY_SIZE; c++) begin : gen_pe_col

                logic signed [15:0] pe_prod_ab, pe_prod_ac;
                logic               pe_out_valid;
                logic               current_pe_valid;

                // Wavefront delay is (r + c)
                assign current_pe_valid = valid_wave[r + c];

                // Propagate activation across row (from left to right)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        act_pipe[r][c] <= '0;
                    end else begin
                        if (c == 0)
                            act_pipe[r][0] <= act_row[r];
                        else
                            act_pipe[r][c] <= act_pipe[r][c-1];
                    end
                end

                // Propagate weights down column (from top to bottom)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        wt0_pipe[r][c] <= '0;
                        wt1_pipe[r][c] <= '0;
                    end else begin
                        if (r == 0) begin
                            wt0_pipe[0][c] <= wt_col0[c];
                            wt1_pipe[0][c] <= wt_col1[c];
                        end else begin
                            wt0_pipe[r][c] <= wt0_pipe[r-1][c];
                            wt1_pipe[r][c] <= wt1_pipe[r-1][c];
                        end
                    end
                end

                pe_slice_dsp_packing u_pe_slice (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .in_valid   (current_pe_valid),
                    .operand_a  (act_pipe[r][c]),
                    .operand_b  (wt0_pipe[r][c]),
                    .operand_c  (wt1_pipe[r][c]),
                    .out_valid  (pe_out_valid),
                    .result_ab  (pe_prod_ab),
                    .result_ac  (pe_prod_ac)
                );

                // 32-bit Accumulator per PE
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        pe_acc_matrix[r][c] <= '0;
                    end else if (clr_acc) begin
                        pe_acc_matrix[r][c] <= '0;
                    end else if (pe_out_valid) begin
                        pe_acc_matrix[r][c] <= pe_acc_matrix[r][c] + 32'(pe_prod_ab) + 32'(pe_prod_ac);
                    end
                end

            end
        end
    endgenerate

    // Pipelined Row Reduction Tree - Full 32-Bit Precision Preserved
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_adder_tree_row
            logic signed [ACC_WIDTH-1:0] tree_in_a [0:ARRAY_SIZE-1];
            logic signed [ACC_WIDTH-1:0] tree_dummy_b [0:ARRAY_SIZE-1];

            for (c = 0; c < ARRAY_SIZE; c++) begin : gen_tree_assign
                // Bit Correction: Keep full 32-bit accumulator without any truncation
                assign tree_in_a[c]    = pe_acc_matrix[r][c];
                assign tree_dummy_b[c] = '0;
            end

            pipelined_adder_tree #(
                .N          (ARRAY_SIZE),
                .IN_WIDTH   (ACC_WIDTH),
                .OUT_WIDTH  (ACC_WIDTH),
                .HAS_MULT   (1'b0)
            ) u_adder_tree (
                .clk         (clk),
                .rst_n       (rst_n),
                .in_valid    (valid_wave[ARRAY_SIZE + 2]),
                .a           (tree_in_a),
                .b           (tree_dummy_b),
                .out_valid   (),
                .dot_product (dot_product_out[r])
            );
        end
    endgenerate

    assign out_valid = valid_wave[WAVE_LEN-1];

endmodule
