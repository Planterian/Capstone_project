`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_golden_scale_unit
// Description:  Bit-Exact Verification Testbench for scale_unit (rtl_ver02)
//               against PyTorch I-ViT Golden Vectors:
//               Input:  golden_gemm_qk.hex (65,536 INT32 elements)
//               Golden: golden_scaled_score.hex (65,536 INT16 elements)
// ============================================================================

`include "hw_scale_params.svh"

module tb_golden_scale_unit;

    localparam int TOTAL_ELEMENTS = 65536; // 256 x 256 matrix
    localparam real CLK_PERIOD    = 5.0;   // 200 MHz

    logic                      clk;
    logic                      rst_n;
    logic                      in_valid;
    logic signed [31:0]        in_score;
    logic signed [15:0]        scale_m;
    logic [4:0]                scale_e;
    logic [3:0]                shift_val;

    logic                      out_valid;
    logic signed [15:0]        scaled_score;

    // Golden Vector Arrays
    logic signed [31:0]        gold_in_gemm   [0:TOTAL_ELEMENTS-1];
    logic signed [15:0]        gold_exp_score [0:TOTAL_ELEMENTS-1];

    int pass_count  = 0;
    int error_count = 0;
    int egress_idx  = 0;

    // DUT Instantiation
    scale_unit #(
        .IN_WIDTH(32),
        .OUT_WIDTH(16)
    ) u_dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (in_valid),
        .in_score     (in_score),
        .scale_m      (scale_m),
        .scale_e      (scale_e),
        .shift_val    (shift_val),
        .out_valid    (out_valid),
        .scaled_score (scaled_score)
    );

    // 200 MHz Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Monitor Output and Compare with Golden Reference
    always @(posedge clk) begin
        if (out_valid) begin
            logic signed [15:0] exp_val;
            exp_val = gold_exp_score[egress_idx];

            if (scaled_score == exp_val) begin
                pass_count++;
            end else begin
                error_count++;
                if (error_count <= 10) begin
                    $display("[MISMATCH @%0d]: In = %0d | DUT = %0d | Expected = %0d", 
                             egress_idx, gold_in_gemm[egress_idx], scaled_score, exp_val);
                end
            end

            egress_idx++;
        end
    end

    // Test Sequence
    initial begin
        $display("=======================================================================");
        $display("   TESTBENCH: scale_unit vs PyTorch I-ViT Golden Model (Stage 2)");
        $display("=======================================================================");

        rst_n     = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;
        scale_m   = `STAGE2_SCALE_M; // 23480
        scale_e   = `STAGE2_SCALE_E; // 26
        shift_val = 4'd2;

        // Load Golden Files
        $display("[TB INFO]: Loading golden_gemm_qk.hex (%0d samples)...", TOTAL_ELEMENTS);
        $readmemh("golden_gemm_qk.hex", gold_in_gemm);
        $display("[TB INFO]: Loading golden_scaled_score.hex (%0d samples)...", TOTAL_ELEMENTS);
        $readmemh("golden_scaled_score.hex", gold_exp_score);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset Released. Streaming inputs through scale_unit...");

        // Stream all 65,536 elements
        for (int i = 0; i < TOTAL_ELEMENTS; i++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_score <= gold_in_gemm[i];
        end

        @(posedge clk);
        in_valid <= 1'b0;

        // Wait for pipeline drain (2 cycles)
        #(CLK_PERIOD * 10);

        $display("=======================================================================");
        $display("   FINAL VERIFICATION REPORT: SCALE_UNIT vs GOLDEN MODEL");
        $display("=======================================================================");
        $display("   Total Elements Tested : %0d", egress_idx);
        $display("   Exact Matches (0 LSB) : %0d", pass_count);
        $display("   Mismatches Detected   : %0d", error_count);
        if (error_count == 0 && pass_count == TOTAL_ELEMENTS) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================");
        $finish;
    end

endmodule
