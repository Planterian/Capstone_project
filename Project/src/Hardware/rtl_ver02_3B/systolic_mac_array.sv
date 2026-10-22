`timescale 1ns / 1ps
// ============================================================================
// Module Name:   systolic_mac_array
// Description:  2D Output-Stationary Systolic MAC Array Core integrating
//               pe_slice_dsp_packing nodes and automatic input wavefront skewing.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Architecture: Variant 3B (2D Output-Stationary Systolic MAC Array with DSP Packing)
// Version:      v3.0 (Auto-Skew Wavefront & Drain Aligned)
// ============================================================================

module systolic_mac_array #(
    parameter int ARRAY_SIZE = 4,
    parameter int IN_WIDTH   = 8,
    parameter int ACC_WIDTH  = 32,
    parameter bit AUTO_SKEW  = 1'b1
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

    // ------------------------------------------------------------------------
    // 1. AUTOMATIC WAVEFRONT SKEW REGISTERS
    //    Row r delayed by r clock cycles; Column c delayed by c clock cycles.
    // ------------------------------------------------------------------------
    logic signed [IN_WIDTH-1:0] act_skewed [0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0] wt0_skewed [0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0] wt1_skewed [0:ARRAY_SIZE-1];

    generate
        if (AUTO_SKEW) begin : gen_auto_skew
            // Row Skew Registers
            logic signed [IN_WIDTH-1:0] act_s1 [1:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] act_s2 [2:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] act_s3 [3:ARRAY_SIZE-1];

            // Col Skew Registers
            logic signed [IN_WIDTH-1:0] wt0_s1 [1:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] wt0_s2 [2:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] wt0_s3 [3:ARRAY_SIZE-1];

            logic signed [IN_WIDTH-1:0] wt1_s1 [1:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] wt1_s2 [2:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] wt1_s3 [3:ARRAY_SIZE-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 1; i < ARRAY_SIZE; i++) begin
                        act_s1[i] <= '0;
                        wt0_s1[i] <= '0;
                        wt1_s1[i] <= '0;
                    end
                    for (int i = 2; i < ARRAY_SIZE; i++) begin
                        act_s2[i] <= '0;
                        wt0_s2[i] <= '0;
                        wt1_s2[i] <= '0;
                    end
                    for (int i = 3; i < ARRAY_SIZE; i++) begin
                        act_s3[i] <= '0;
                        wt0_s3[i] <= '0;
                        wt1_s3[i] <= '0;
                    end
                end else begin
                    // Stage 1
                    for (int i = 1; i < ARRAY_SIZE; i++) begin
                        act_s1[i] <= act_row[i];
                        wt0_s1[i] <= wt_col0[i];
                        wt1_s1[i] <= wt_col1[i];
                    end
                    // Stage 2
                    for (int i = 2; i < ARRAY_SIZE; i++) begin
                        act_s2[i] <= act_s1[i];
                        wt0_s2[i] <= wt0_s1[i];
                        wt1_s2[i] <= wt1_s1[i];
                    end
                    // Stage 3
                    for (int i = 3; i < ARRAY_SIZE; i++) begin
                        act_s3[i] <= act_s2[i];
                        wt0_s3[i] <= wt0_s2[i];
                        wt1_s3[i] <= wt1_s2[i];
                    end
                end
            end

            assign act_skewed[0] = act_row[0];
            assign act_skewed[1] = act_s1[1];
            assign act_skewed[2] = act_s2[2];
            assign act_skewed[3] = act_s3[3];

            assign wt0_skewed[0] = wt_col0[0];
            assign wt0_skewed[1] = wt0_s1[1];
            assign wt0_skewed[2] = wt0_s2[2];
            assign wt0_skewed[3] = wt0_s3[3];

            assign wt1_skewed[0] = wt_col1[0];
            assign wt1_skewed[1] = wt1_s1[1];
            assign wt1_skewed[2] = wt1_s2[2];
            assign wt1_skewed[3] = wt1_s3[3];

        end else begin : gen_no_skew
            assign act_skewed = act_row;
            assign wt0_skewed = wt_col0;
            assign wt1_skewed = wt_col1;
        end
    endgenerate

    // ------------------------------------------------------------------------
    // 2. SYSTOLIC INTERCONNECT & WAVEFRONT PROPAGATION
    // ------------------------------------------------------------------------
    logic signed [IN_WIDTH-1:0]  act_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt0_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt1_pipe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    localparam int WAVE_LEN = 2 * ARRAY_SIZE + 4;
    logic [WAVE_LEN-1:0]         valid_wave;

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

                assign current_pe_valid = valid_wave[r + c];

                // Propagate activation across row (from left to right)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        act_pipe[r][c] <= '0;
                    end else begin
                        if (c == 0)
                            act_pipe[r][0] <= act_skewed[r];
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
                            wt0_pipe[0][c] <= wt0_skewed[c];
                            wt1_pipe[0][c] <= wt1_skewed[c];
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

                // Independent 32-bit accumulators per subcolumn (Col 2c and Col 2c+1)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        pe_acc_col0[r][c] <= '0;
                        pe_acc_col1[r][c] <= '0;
                    end else if (clr_acc) begin
                        pe_acc_col0[r][c] <= '0;
                        pe_acc_col1[r][c] <= '0;
                    end else if (pe_out_valid) begin
                        pe_acc_col0[r][c] <= pe_acc_col0[r][c] + 32'($signed(pe_prod_ab));
                        pe_acc_col1[r][c] <= pe_acc_col1[r][c] + 32'($signed(pe_prod_ac));
                    end
                end

            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // 3. TRUE SEQUENTIAL TILE DRAIN LOGIC (32 dot product items per tile)
    //    Items 0..7:   Row 0, Cols 0..7
    //    Items 8..15:  Row 1, Cols 0..7
    //    Items 16..23: Row 2, Cols 0..7
    //    Items 24..31: Row 3, Cols 0..7
    // ------------------------------------------------------------------------
    localparam int TOTAL_DRAIN_ITEMS = 2 * ARRAY_SIZE * ARRAY_SIZE; // 32 items

    typedef enum logic [1:0] {
        DRAIN_IDLE = 2'b00,
        DRAIN_RUN  = 2'b01,
        DRAIN_LAST = 2'b10
    } drain_state_t;

    drain_state_t drain_state;
    logic [5:0]   drain_item_cnt;
    logic [1:0]   drain_r, drain_c;
    logic         drain_subcol;

    assign drain_r      = drain_item_cnt[4:3]; // drain_item_cnt / 8 (0 to 3)
    assign drain_c      = drain_item_cnt[2:1]; // (drain_item_cnt % 8) / 2 (0 to 3)
    assign drain_subcol = drain_item_cnt[0];   // drain_item_cnt % 2 (0 or 1)

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

    // Parallel row export
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_row_out
            assign dot_product_out[r] = (r == 0) ? dot_product_stream : pe_acc_col0[r][0];
        end
    endgenerate

endmodule
