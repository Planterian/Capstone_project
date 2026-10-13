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

    input  logic                        drain_start,
    output logic                        drain_done,
    output logic                        out_valid,
    output logic signed [ACC_WIDTH-1:0] dot_product_stream,
    output logic signed [ACC_WIDTH-1:0] dot_product_out [0:ARRAY_SIZE-1]
);

    // Systolic Skew Pipelines (Clean procedural registers)
    logic signed [IN_WIDTH-1:0]  act_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt0_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt1_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Wavefront valid propagation
    localparam int WAVE_LEN = 2 * ARRAY_SIZE + 4;
    logic [WAVE_LEN-1:0]         valid_wave;

    // FL-03 FIX: Separate 32-bit accumulators for Column 0 (even) and Column 1 (odd)
    logic signed [ACC_WIDTH-1:0] pe_acc_col0 [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [ACC_WIDTH-1:0] pe_acc_col1 [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

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

                // FL-03 FIX: Independent accumulation per PE - No cross-column crosstalk
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        pe_acc_col0[r][c] <= '0;
                        pe_acc_col1[r][c] <= '0;
                    end else if (clr_acc) begin
                        pe_acc_col0[r][c] <= '0;
                        pe_acc_col1[r][c] <= '0;
                    end else if (pe_out_valid) begin
                        pe_acc_col0[r][c] <= pe_acc_col0[r][c] + 32'($signed(pe_prod_ab)); // Col 2c
                        pe_acc_col1[r][c] <= pe_acc_col1[r][c] + 32'($signed(pe_prod_ac)); // Col 2c + 1
                    end
                end

            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // FL-04 FIX: Drain / Unload Shift-Out Logic
    // Replaces horizontal pipelined_adder_tree with true sequential tile drain.
    // Drains 2 * ARRAY_SIZE * ARRAY_SIZE dot product items to scale_unit.
    // ------------------------------------------------------------------------
    localparam int TOTAL_DRAIN_ITEMS = 2 * ARRAY_SIZE * ARRAY_SIZE;

    typedef enum logic [1:0] {
        DRAIN_IDLE = 2'b00,
        DRAIN_RUN  = 2'b01,
        DRAIN_LAST = 2'b10
    } drain_state_t;

    drain_state_t drain_state;
    logic [5:0]   drain_item_cnt;
    logic [3:0]   drain_r, drain_c;
    logic         drain_subcol; // 0: col0, 1: col1

    assign drain_r      = drain_item_cnt / (2 * ARRAY_SIZE);
    assign drain_c      = (drain_item_cnt % (2 * ARRAY_SIZE)) / 2;
    assign drain_subcol = drain_item_cnt % 2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain_state        <= DRAIN_IDLE;
            drain_item_cnt     <= '0;
            out_valid          <= 1'b0;
            drain_done         <= 1'b0;
            dot_product_stream <= '0;
        end else begin
            drain_done <= 1'b0;

            case (drain_state)
                DRAIN_IDLE: begin
                    out_valid      <= 1'b0;
                    drain_item_cnt <= '0;
                    if (drain_start) begin
                        drain_state <= DRAIN_RUN;
                    end
                end

                DRAIN_RUN: begin
                    out_valid <= 1'b1;
                    if (!drain_subcol) begin
                        dot_product_stream <= pe_acc_col0[drain_r][drain_c];
                    end else begin
                        dot_product_stream <= pe_acc_col1[drain_r][drain_c];
                    end

                    if (drain_item_cnt == TOTAL_DRAIN_ITEMS - 1) begin
                        drain_item_cnt <= '0;
                        drain_done     <= 1'b1;
                        drain_state    <= DRAIN_IDLE;
                    end else begin
                        drain_item_cnt <= drain_item_cnt + 1'b1;
                    end
                end

                default: drain_state <= DRAIN_IDLE;
            endcase
        end
    end

    // Parallel row export for backwards compatibility
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_row_out
            assign dot_product_out[r] = (r == 0) ? dot_product_stream : pe_acc_col0[r][0];
        end
    endgenerate

endmodule
