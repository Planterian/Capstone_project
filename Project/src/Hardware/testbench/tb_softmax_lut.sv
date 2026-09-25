`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_softmax_lut
// Description:  Automated Self-Checking Testbench for Hardware Softmax Engine
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_softmax_lut;

    localparam int N_TOKENS  = 16;  // Fast simulation token count
    localparam int IN_WIDTH  = 16;
    localparam int OUT_WIDTH = 8;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                   clk;
    logic                   rst_n;
    logic                   start_row;
    logic                   in_valid;
    logic signed [IN_WIDTH-1:0] in_score;

    logic                   out_valid;
    logic [OUT_WIDTH-1:0]   out_prob;
    logic                   busy;

    logic signed [IN_WIDTH-1:0] test_scores [0:N_TOKENS-1];
    int total_prob_sum = 0;
    int received_count = 0;
    int error_count    = 0;

    // DUT Instantiation
    softmax_lut #(
        .N_TOKENS(N_TOKENS),
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_row(start_row),
        .in_valid(in_valid),
        .in_score(in_score),
        .out_valid(out_valid),
        .out_prob(out_prob),
        .busy(busy)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: softmax_lut (Level 3)");
        $display("=========================================================");

        rst_n     = 1'b0;
        start_row = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;

        // Initialize test scores (one high peak at index 5)
        for (int i = 0; i < N_TOKENS; i++) begin
            if (i == 5)
                test_scores[i] = 16'sd500;  // High score peak
            else
                test_scores[i] = 16'sd100;  // Lower baseline scores
        end

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Trigger row processing
        @(posedge clk);
        start_row <= 1'b1;
        @(posedge clk);
        start_row <= 1'b0;

        // Stream scores in FIND_MAX phase
        $display("[TB INFO]: Streaming %0d score tokens into Softmax Engine...", N_TOKENS);
        for (int i = 0; i < N_TOKENS; i++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_score <= test_scores[i];
        end
        @(posedge clk);
        in_valid <= 1'b0;

        // Wait for normalized probabilities output
        $display("[TB INFO]: Waiting for Softmax output stream...");

        while (received_count < N_TOKENS) begin
            @(posedge clk);
            if (out_valid) begin
                $display("  [PROB OUT] Token %0d | Score = %0d | Output Prob = %0d / 255",
                         received_count, test_scores[received_count], out_prob);
                total_prob_sum += out_prob;

                // Index 5 (peak score) should receive highest probability
                if (received_count == 5 && out_prob < 100) begin
                    $display("  [ERROR]: Token 5 expected high probability, got %0d", out_prob);
                    error_count++;
                end

                received_count++;
            end
        end

        $display("---------------------------------------------------------");
        $display("[TB INFO]: Total Probability Sum = %0d / 255", total_prob_sum);

        // Verify probability distribution sum (should be ~200-260 due to fixed-point precision)
        if (total_prob_sum >= 200 && total_prob_sum <= 260) begin
            $display("[PASS]: Total probability sum is well-normalized.");
        end else begin
            $display("[FAIL]: Probability sum out of expected range!");
            error_count++;
        end

        $display("=========================================================");
        if (error_count == 0)
            $display("   *** TEST PASSED SUCCESSFULLY ***");
        else
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

endmodule
