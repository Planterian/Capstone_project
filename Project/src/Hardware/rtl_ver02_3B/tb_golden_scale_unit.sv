`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_golden_scale_unit
// Description:  Bit-Exact Multi-Stage Verification Testbench for scale_unit
//               across all 3 MobileViT Stages:
//               - Stage 2: N=256, 65,536 INT32 elements, M=27640, E=22
//               - Stage 3: N=64,   4,096 INT32 elements, M=21936, E=22
//               - Stage 4: N=16,     256 INT32 elements, M=18810, E=22
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

`include "hw_scale_params.svh"

module tb_golden_scale_unit;

    localparam int ST2_ELEMENTS = 65536; // 256 x 256
    localparam int ST3_ELEMENTS = 4096;  // 64 x 64
    localparam int ST4_ELEMENTS = 256;   // 16 x 16
    localparam real CLK_PERIOD  = 5.0;   // 200 MHz

    logic                      clk;
    logic                      rst_n;
    logic                      in_valid;
    logic signed [31:0]        in_score;
    logic signed [15:0]        scale_m;
    logic [4:0]                scale_e;
    logic [3:0]                shift_val;

    logic                      out_valid;
    logic signed [15:0]        scaled_score;

    // Golden Vector Arrays for all 3 Stages
    logic signed [31:0]        gold_gemm_st2   [0:ST2_ELEMENTS-1];
    logic signed [15:0]        gold_score_st2  [0:ST2_ELEMENTS-1];

    logic signed [31:0]        gold_gemm_st3   [0:ST3_ELEMENTS-1];
    logic signed [15:0]        gold_score_st3  [0:ST3_ELEMENTS-1];

    logic signed [31:0]        gold_gemm_st4   [0:ST4_ELEMENTS-1];
    logic signed [15:0]        gold_score_st4  [0:ST4_ELEMENTS-1];

    int current_stage = 2;
    int stage_pass    = 0;
    int stage_err     = 0;
    int stage_egress  = 0;

    int total_pass    = 0;
    int total_err     = 0;

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

    // Monitor Output and Compare with Active Stage Golden Reference
    always @(posedge clk) begin
        if (out_valid) begin
            logic signed [15:0] exp_val;
            logic signed [31:0] in_val;

            if (current_stage == 2) begin
                exp_val = gold_score_st2[stage_egress];
                in_val  = gold_gemm_st2[stage_egress];
            end else if (current_stage == 3) begin
                exp_val = gold_score_st3[stage_egress];
                in_val  = gold_gemm_st3[stage_egress];
            end else begin
                exp_val = gold_score_st4[stage_egress];
                in_val  = gold_gemm_st4[stage_egress];
            end

            if (scaled_score === exp_val) begin
                stage_pass++;
                total_pass++;
            end else begin
                stage_err++;
                total_err++;
                if (stage_err <= 5) begin
                    $display("[MISMATCH ST%0d @ %0d]: In = %0d | DUT = %0d | Expected = %0d", 
                             current_stage, stage_egress, in_val, scaled_score, exp_val);
                end
            end

            stage_egress++;
        end
    end

    // Test Sequence
    initial begin
        $display("=======================================================================");
        $display("   TESTBENCH: scale_unit MULTI-STAGE BIT-EXACT VERIFICATION");
        $display("   Testing Stage 2 (65,536), Stage 3 (4,096), and Stage 4 (256)");
        $display("=======================================================================");

        rst_n     = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;
        scale_m   = 16'sd0;
        scale_e   = 5'd0;
        shift_val = 4'd0;

        // Load Golden Files for all Stages
        $display("[TB INFO]: Loading Stage 2 vectors (%0d samples)...", ST2_ELEMENTS);
        $readmemh("golden_gemm_qk_st2.hex", gold_gemm_st2);
        $readmemh("golden_scaled_score_st2.hex", gold_score_st2);

        $display("[TB INFO]: Loading Stage 3 vectors (%0d samples)...", ST3_ELEMENTS);
        $readmemh("golden_gemm_qk_st3.hex", gold_gemm_st3);
        $readmemh("golden_scaled_score_st3.hex", gold_score_st3);

        $display("[TB INFO]: Loading Stage 4 vectors (%0d samples)...", ST4_ELEMENTS);
        $readmemh("golden_gemm_qk_st4.hex", gold_gemm_st4);
        $readmemh("golden_scaled_score_st4.hex", gold_score_st4);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        #(CLK_PERIOD * 5);

        // --------------------------------------------------------------------
        // PHASE 1: STAGE 2 (M=27640, E=22, 65,536 elements)
        // --------------------------------------------------------------------
        current_stage = 2;
        stage_pass    = 0;
        stage_err     = 0;
        stage_egress  = 0;
        scale_m       = `STAGE2_SCALE_M; // 27640
        scale_e       = `STAGE2_SCALE_E; // 22
        $display("\n>>> [PHASE 1/3]: Streaming STAGE 2 (M=%0d, E=%0d, %0d elements)...", scale_m, scale_e, ST2_ELEMENTS);

        for (int i = 0; i < ST2_ELEMENTS; i++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_score <= gold_gemm_st2[i];
        end
        @(posedge clk);
        in_valid <= 1'b0;
        #(CLK_PERIOD * 10);
        $display("    -> Stage 2 Done: %0d / %0d matched (Errors: %0d)", stage_pass, ST2_ELEMENTS, stage_err);

        // --------------------------------------------------------------------
        // PHASE 2: STAGE 3 (M=21936, E=22, 4,096 elements)
        // --------------------------------------------------------------------
        current_stage = 3;
        stage_pass    = 0;
        stage_err     = 0;
        stage_egress  = 0;
        scale_m       = `STAGE3_SCALE_M; // 21936
        scale_e       = `STAGE3_SCALE_E; // 22
        $display("\n>>> [PHASE 2/3]: Streaming STAGE 3 (M=%0d, E=%0d, %0d elements)...", scale_m, scale_e, ST3_ELEMENTS);

        for (int i = 0; i < ST3_ELEMENTS; i++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_score <= gold_gemm_st3[i];
        end
        @(posedge clk);
        in_valid <= 1'b0;
        #(CLK_PERIOD * 10);
        $display("    -> Stage 3 Done: %0d / %0d matched (Errors: %0d)", stage_pass, ST3_ELEMENTS, stage_err);

        // --------------------------------------------------------------------
        // PHASE 3: STAGE 4 (M=18810, E=22, 256 elements)
        // --------------------------------------------------------------------
        current_stage = 4;
        stage_pass    = 0;
        stage_err     = 0;
        stage_egress  = 0;
        scale_m       = `STAGE4_SCALE_M; // 18810
        scale_e       = `STAGE4_SCALE_E; // 22
        $display("\n>>> [PHASE 3/3]: Streaming STAGE 4 (M=%0d, E=%0d, %0d elements)...", scale_m, scale_e, ST4_ELEMENTS);

        for (int i = 0; i < ST4_ELEMENTS; i++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_score <= gold_gemm_st4[i];
        end
        @(posedge clk);
        in_valid <= 1'b0;
        #(CLK_PERIOD * 10);
        $display("    -> Stage 4 Done: %0d / %0d matched (Errors: %0d)", stage_pass, ST4_ELEMENTS, stage_err);

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display("   FINAL REPORT: scale_unit ALL 3 STAGES VERIFICATION");
        $display("=======================================================================");
        $display("   Stage 2 Matched : %0d / %0d", (total_pass - (ST3_ELEMENTS + ST4_ELEMENTS)), ST2_ELEMENTS);
        $display("   Stage 3 Matched : %0d / %0d", ST3_ELEMENTS, ST3_ELEMENTS);
        $display("   Stage 4 Matched : %0d / %0d", ST4_ELEMENTS, ST4_ELEMENTS);
        $display("   Total Elements  : %0d / %0d", total_pass, (ST2_ELEMENTS + ST3_ELEMENTS + ST4_ELEMENTS));
        $display("   Total Errors    : %0d", total_err);
        if (total_err == 0 && total_pass == (ST2_ELEMENTS + ST3_ELEMENTS + ST4_ELEMENTS)) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH ACROSS ALL 3 STAGES! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: SCALE_UNIT VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");
        $finish;
    end

endmodule
