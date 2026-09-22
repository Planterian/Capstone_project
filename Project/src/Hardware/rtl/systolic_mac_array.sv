`timescale 1ns / 1ps
// ============================================================================
// Module Name:   systolic_mac_array
// Description:  2D Output-Stationary Systolic MAC Array Core integrating
//               pe_slice_dsp_packing nodes and pipelined_adder_tree reduction.
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module systolic_mac_array #(
    parameter int ARRAY_SIZE = 4,
    parameter int IN_WIDTH   = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     gemm_start,
    input  logic                     clr_acc,
    input  logic                     in_valid,
    input  logic                     mode_sel,

    input  logic signed [IN_WIDTH-1:0] act_row [0:ARRAY_SIZE-1],
    input  logic signed [IN_WIDTH-1:0] wt_col0 [0:ARRAY_SIZE-1],
    input  logic signed [IN_WIDTH-1:0] wt_col1 [0:ARRAY_SIZE-1],

    output logic                     out_valid,
    output logic signed [ACC_WIDTH-1:0] dot_product_out [0:ARRAY_SIZE-1]
);

    logic signed [IN_WIDTH-1:0] act_shift [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    logic signed [IN_WIDTH-1:0] wt0_shift [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0] wt1_shift [0:ARRAY_SIZE][0:ARRAY_SIZE-1];

    logic pe_valid_net [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [ACC_WIDTH-1:0] pe_acc_matrix [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic [ARRAY_SIZE+2:0] valid_pipeline;

    genvar r, c;
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_row_in
            assign act_shift[r][0] = act_row[r];
        end

        for (c = 0; c < ARRAY_SIZE; c++) begin : gen_col_in
            assign wt0_shift[0][c] = wt_col0[c];
            assign wt1_shift[0][c] = wt_col1[c];
        end
    endgenerate

    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_pe_row
            for (c = 0; c < ARRAY_SIZE; c++) begin : gen_pe_col

                logic signed [15:0] pe_prod_ab, pe_prod_ac;
                logic pe_out_valid;
                logic current_in_valid;

                assign current_in_valid = (r == 0 && c == 0) ? in_valid : pe_valid_net[r][c];

                pe_slice_dsp_packing u_pe_slice (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .in_valid   (current_in_valid),
                    .operand_a  (act_shift[r][c]),
                    .operand_b  (wt0_shift[r][c]),
                    .operand_c  (wt1_shift[r][c]),
                    .out_valid  (pe_out_valid),
                    .result_ab  (pe_prod_ab),
                    .result_ac  (pe_prod_ac)
                );

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        act_shift[r][c+1] <= '0;
                        wt0_shift[r+1][c] <= '0;
                        wt1_shift[r+1][c] <= '0;
                        pe_valid_net[r][c] <= 1'b0;
                    end else begin
                        act_shift[r][c+1] <= act_shift[r][c];
                        wt0_shift[r+1][c] <= wt0_shift[r][c];
                        wt1_shift[r+1][c] <= wt1_shift[r][c];
                        if (c < ARRAY_SIZE - 1) pe_valid_net[r][c+1] <= current_in_valid;
                        if (r < ARRAY_SIZE - 1) pe_valid_net[r+1][c] <= current_in_valid;
                    end
                end

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

    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gen_adder_tree_row
            logic signed [IN_WIDTH-1:0] tree_in_a [0:ARRAY_SIZE-1];
            logic signed [IN_WIDTH-1:0] tree_in_b [0:ARRAY_SIZE-1];

            for (c = 0; c < ARRAY_SIZE; c++) begin : gen_tree_assign
                assign tree_in_a[c] = pe_acc_matrix[r][c][7:0];
                assign tree_in_b[c] = 8'sd1;
            end

            pipelined_adder_tree #(
                .N          (ARRAY_SIZE),
                .IN_WIDTH   (IN_WIDTH),
                .OUT_WIDTH  (ACC_WIDTH)
            ) u_adder_tree (
                .clk         (clk),
                .rst_n       (rst_n),
                .in_valid    (valid_pipeline[ARRAY_SIZE]),
                .a           (tree_in_a),
                .b           (tree_in_b),
                .out_valid   (),
                .dot_product (dot_product_out[r])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipeline <= '0;
        end else begin
            valid_pipeline <= {valid_pipeline[ARRAY_SIZE+1:0], in_valid};
        end
    end

    assign out_valid = valid_pipeline[ARRAY_SIZE+2];

endmodule
