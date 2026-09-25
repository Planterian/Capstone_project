`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_pipelined_adder_tree
// Description:  Automated Self-Checking Testbench for Pipelined Adder Tree
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_pipelined_adder_tree;

    localparam int N         = 4;
    localparam int IN_WIDTH  = 8;
    localparam int OUT_WIDTH = 32;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                     clk;
    logic                     rst_n;
    logic                     in_valid;
    logic signed [IN_WIDTH-1:0] a [0:N-1];
    logic signed [IN_WIDTH-1:0] b [0:N-1];

    logic                     out_valid;
    logic signed [OUT_WIDTH-1:0] dot_product;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    pipelined_adder_tree #(
        .N(N),
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .a(a),
        .b(b),
        .out_valid(out_valid),
        .dot_product(dot_product)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: pipelined_adder_tree (Level 2)");
        $display("=========================================================");

        rst_n    = 1'b0;
        in_valid = 1'b0;
        for (int i = 0; i < N; i++) begin
            a[i] = '0;
            b[i] = '0;
        end

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Test Case 1: Positive Dot Product ([1,2,3,4] . [5,6,7,8] = 5+12+21+32 = 70)
        apply_test_vectors(
            '{8'sd1, 8'sd2, 8'sd3, 8'sd4},
            '{8'sd5, 8'sd6, 8'sd7, 8'sd8},
            32'sd70,
            "Positive Dot Product ([1,2,3,4] . [5,6,7,8])"
        );

        // Test Case 2: Mixed Signed Dot Product ([-10,20,-30,40] . [2,-3,4,-5] = -20-60-120-200 = -400)
        apply_test_vectors(
            '{-8'sd10, 8'sd20, -8'sd30, 8'sd40},
            '{8'sd2, -8'sd3, 8'sd4, -8'sd5},
            -32'sd400,
            "Mixed Signed Dot Product ([-10,20,-30,40] . [2,-3,4,-5])"
        );

        #(CLK_PERIOD * 5);
        $display("=========================================================");
        if (error_count == 0)
            $display("   *** TEST PASSED SUCCESSFULLY (%0d/%0d Cases Matched) ***", pass_count, pass_count);
        else
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

    task apply_test_vectors(
        input logic signed [IN_WIDTH-1:0] vec_a [0:N-1],
        input logic signed [IN_WIDTH-1:0] vec_b [0:N-1],
        input logic signed [OUT_WIDTH-1:0] exp_dot,
        input string                       test_name
    );
        @(posedge clk);
        in_valid <= 1'b1;
        a        <= vec_a;
        b        <= vec_b;

        @(posedge clk);
        in_valid <= 1'b0;

        wait (out_valid == 1'b1);
        $write("[TEST]: %s -> DUT = %0d | Expected = %0d ---> ", test_name, dot_product, exp_dot);

        if (dot_product == exp_dot) begin
            $display("[ PASS ]");
            pass_count++;
        end else begin
            $display("[ FAIL ]");
            error_count++;
        end
    endtask

endmodule
