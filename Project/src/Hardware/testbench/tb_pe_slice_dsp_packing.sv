`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_pe_slice_dsp_packing
// Description:  Automated Self-Checking Testbench for PE Slice (DSP Packing 2x INT8)
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_pe_slice_dsp_packing;

    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic        clk;
    logic        rst_n;
    logic        in_valid;
    logic signed [7:0] operand_a;
    logic signed [7:0] operand_b;
    logic signed [7:0] operand_c;

    logic        out_valid;
    logic signed [15:0] result_ab;
    logic signed [15:0] result_ac;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    pe_slice_dsp_packing u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .operand_c(operand_c),
        .out_valid(out_valid),
        .result_ab(result_ab),
        .result_ac(result_ac)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: pe_slice_dsp_packing (Level 2)");
        $display("=========================================================");

        rst_n     = 1'b0;
        in_valid  = 1'b0;
        operand_a = '0;
        operand_b = '0;
        operand_c = '0;

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Test Case 1: Positive operands (A=10, B=20, C=30 -> AB=200, AC=300)
        apply_test(8'sd10, 8'sd20, 8'sd30, 16'sd200, 16'sd300, "Positive Operands (10*20, 10*30)");

        // Test Case 2: Negative A (A=-5, B=12, C=8 -> AB=-60, AC=-40)
        apply_test(-8'sd5, 8'sd12, 8'sd8, -16'sd60, -16'sd40, "Negative A (-5*12, -5*8)");

        // Test Case 3: Negative B and C (A=7, B=-10, C=-15 -> AB=-70, AC=-105)
        apply_test(8'sd7, -8'sd10, -8'sd15, -16'sd70, -16'sd105, "Negative B and C (7*-10, 7*-15)");

        // Test Case 4: Mixed signs (A=-8, B=-9, C=11 -> AB=72, AC=-88)
        apply_test(-8'sd8, -8'sd9, 8'sd11, 16'sd72, -16'sd88, "Mixed Signs (-8*-9, -8*11)");

        #(CLK_PERIOD * 5);
        $display("=========================================================");
        if (error_count == 0)
            $display("   *** TEST PASSED SUCCESSFULLY (%0d/%0d Cases Matched) ***", pass_count, pass_count);
        else
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

    task apply_test(
        input logic signed [7:0]  a_val,
        input logic signed [7:0]  b_val,
        input logic signed [7:0]  c_val,
        input logic signed [15:0] exp_ab,
        input logic signed [15:0] exp_ac,
        input string              test_name
    );
        @(posedge clk);
        in_valid  <= 1'b1;
        operand_a <= a_val;
        operand_b <= b_val;
        operand_c <= c_val;

        @(posedge clk);
        in_valid  <= 1'b0;

        wait (out_valid == 1'b1);
        $write("[TEST]: %s | A=%0d, B=%0d, C=%0d -> AB=%0d, AC=%0d | Exp=(%0d, %0d) ---> ",
               test_name, a_val, b_val, c_val, result_ab, result_ac, exp_ab, exp_ac);

        if (result_ab == exp_ab && result_ac == exp_ac) begin
            $display("[ PASS ]");
            pass_count++;
        end else begin
            $display("[ FAIL ]");
            error_count++;
        end
    endtask

endmodule
