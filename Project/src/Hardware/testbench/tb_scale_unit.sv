`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_scale_unit
// Description:  Automated Self-Checking Testbench for Scaler Unit (ASR & Clamp)
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_scale_unit;

    localparam int IN_WIDTH  = 32;
    localparam int OUT_WIDTH = 16;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                    clk;
    logic                    rst_n;
    logic                    in_valid;
    logic signed [IN_WIDTH-1:0] in_score;
    logic        [3:0]       shift_val;

    logic                    out_valid;
    logic signed [OUT_WIDTH-1:0] scaled_score;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    scale_unit #(
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_score(in_score),
        .shift_val(shift_val),
        .out_valid(out_valid),
        .scaled_score(scaled_score)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: scale_unit (Level 3)");
        $display("=========================================================");

        rst_n     = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;
        shift_val = 4'd2; // Shift right by 2 bits (/4)

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Test Case 1: Normal positive scaling (100 >>> 2 = 25)
        apply_test(32'sd100, 4'd2, 16'sd25, "Normal Positive Scaling (100 >> 2 = 25)");

        // Test Case 2: Normal negative scaling (-100 >>> 2 = -25)
        apply_test(-32'sd100, 4'd2, -16'sd25, "Normal Negative Scaling (-100 >> 2 = -25)");

        // Test Case 3: Positive Saturation Clamp (200,000 >>> 2 = 50,000 > 32767 -> Clamp 32767)
        apply_test(32'sd200000, 4'd2, 16'sd32767, "Positive Saturation Clamp");

        // Test Case 4: Negative Saturation Clamp (-200,000 >>> 2 = -50,000 < -32768 -> Clamp -32768)
        apply_test(-32'sd200000, 4'd2, -16'sd32768, "Negative Saturation Clamp");

        // Test Case 5: Zero shift (300 >>> 0 = 300)
        apply_test(32'sd300, 4'd0, 16'sd300, "Zero Shift");

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
        input logic signed [IN_WIDTH-1:0]  score_in,
        input logic        [3:0]           s_val,
        input logic signed [OUT_WIDTH-1:0] exp_out,
        input string                       test_name
    );
        @(posedge clk);
        in_valid  <= 1'b1;
        in_score  <= score_in;
        shift_val <= s_val;

        @(posedge clk);
        in_valid  <= 1'b0;

        // Sample on out_valid
        wait (out_valid == 1'b1);
        $write("[TEST]: %s | Input = %0d, Shift = %0d -> DUT = %0d | Expected = %0d ---> ",
               test_name, score_in, s_val, scaled_score, exp_out);

        if (scaled_score == exp_out) begin
            $display("[ PASS ]");
            pass_count++;
        end else begin
            $display("[ FAIL ]");
            error_count++;
        end
    endtask

endmodule
