`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_requant_residual_add
// Description:  Automated Testbench for Dyadic Rescaling & Saturating Residual Adder
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_requant_residual_add;

    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic        clk;
    logic        rst_n;
    logic        in_valid;

    logic signed [7:0]  i_f_data;
    logic signed [7:0]  i_x_data;

    logic signed [15:0] m_f;
    logic        [4:0]  e_f;
    logic signed [15:0] m_x;
    logic        [4:0]  e_x;

    logic        out_valid;
    logic signed [7:0]  res_out;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    requant_residual_add u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .i_f_data(i_f_data),
        .i_x_data(i_x_data),
        .m_f(m_f),
        .e_f(e_f),
        .m_x(m_x),
        .e_x(e_x),
        .out_valid(out_valid),
        .res_out(res_out)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: requant_residual_add (Level 3)");
        $display("=========================================================");

        rst_n    = 1'b0;
        in_valid = 1'b0;
        i_f_data = '0;
        i_x_data = '0;
        m_f      = 16'd256; e_f = 5'd8; // Scale F = 1.0
        m_x      = 16'd256; e_x = 5'd8; // Scale X = 1.0

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Test Case 1: Identity Addition (20 + 30 = 50)
        apply_test(8'sd20, 8'sd30, 16'd256, 5'd8, 16'd256, 5'd8, 8'sd50, "Identity Addition (20 + 30 = 50)");

        // Test Case 2: Positive Saturation Clamp (100 + 100 = 200 > 127 -> 127)
        apply_test(8'sd100, 8'sd100, 16'd256, 5'd8, 16'd256, 5'd8, 8'sd127, "Positive Saturation Clamp");

        // Test Case 3: Negative Saturation Clamp (-100 + -100 = -200 < -128 -> -128)
        apply_test(-8'sd100, -8'sd100, 16'd256, 5'd8, 16'd256, 5'd8, -8'sd128, "Negative Saturation Clamp");

        // Test Case 4: Dyadic Scaling (F*0.5 + X*2.0 = 40*0.5 + 10*2.0 = 20 + 20 = 40)
        apply_test(8'sd40, 8'sd10, 16'd128, 5'd8, 16'd512, 5'd8, 8'sd40, "Dyadic Rescaling Addition");

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
        input logic signed [7:0]  f_in,
        input logic signed [7:0]  x_in,
        input logic signed [15:0] mf_val,
        input logic        [4:0]  ef_val,
        input logic signed [15:0] mx_val,
        input logic        [4:0]  ex_val,
        input logic signed [7:0]  exp_out,
        input string              test_name
    );
        @(posedge clk);
        in_valid <= 1'b1;
        i_f_data <= f_in;
        i_x_data <= x_in;
        m_f      <= mf_val;
        e_f      <= ef_val;
        m_x      <= mx_val;
        e_x      <= ex_val;

        @(posedge clk);
        in_valid <= 1'b0;

        wait (out_valid == 1'b1);
        $write("[TEST]: %s | F = %0d, X = %0d -> DUT = %0d | Expected = %0d ---> ",
               test_name, f_in, x_in, res_out, exp_out);

        if (res_out == exp_out) begin
            $display("[ PASS ]");
            pass_count++;
        end else begin
            $display("[ FAIL ]");
            error_count++;
        end
    endtask

endmodule
