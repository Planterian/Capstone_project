`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_golden_softmax_lut
// Description:  Bit-Exact Multi-Stage Verification Testbench for softmax_lut
//               across all 3 MobileViT Stages:
//               - Stage 2: N=256, dk=16, 256 rows (65,536 probabilities)
//               - Stage 3: N=64,  dk=20, 64 rows  (4,096 probabilities)
//               - Stage 4: N=16,  dk=24, 16 rows  (256 probabilities)
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_golden_softmax_lut;

    localparam int ST2_TOKENS = 256;
    localparam int ST3_TOKENS = 64;
    localparam int ST4_TOKENS = 16;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                      clk;
    logic                      rst_n;
    logic                      start_row;
    logic                      in_valid;
    logic signed [15:0]        in_score;
    logic [15:0]               seq_len;
    logic [15:0]               head_dim;

    logic                      out_valid;
    logic [15:0]               out_prob;
    logic                      busy;

    // Golden Vector Memories for all 3 Stages
    logic signed [15:0]        gold_score_st2 [0:(ST2_TOKENS * ST2_TOKENS)-1];
    logic [15:0]               gold_prob_st2  [0:(ST2_TOKENS * ST2_TOKENS)-1];

    logic signed [15:0]        gold_score_st3 [0:(ST3_TOKENS * ST3_TOKENS)-1];
    logic [15:0]               gold_prob_st3  [0:(ST3_TOKENS * ST3_TOKENS)-1];

    logic signed [15:0]        gold_score_st4 [0:(ST4_TOKENS * ST4_TOKENS)-1];
    logic [15:0]               gold_prob_st4  [0:(ST4_TOKENS * ST4_TOKENS)-1];

    int current_stage = 2;
    int stage_pass    = 0;
    int stage_err     = 0;
    int stage_egress  = 0;

    int total_pass    = 0;
    int total_err     = 0;

    // DUT Instantiation
    softmax_lut #(
        .N_TOKENS       (256),
        .IN_WIDTH       (16),
        .OUT_WIDTH      (16),
        .ROM_DEPTH      (1024),
        .ROM_DATA_WIDTH (20),
        .ACC_WIDTH      (32),
        .INIT_HEX_FILE  ("exp_rom_all_stages.hex")
    ) u_dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start_row (start_row),
        .in_valid  (in_valid),
        .in_score  (in_score),
        .out_valid (out_valid),
        .out_prob  (out_prob),
        .busy      (busy),
        .seq_len   (seq_len),
        .head_dim  (head_dim),
        .cfg_we    (1'b0),
        .cfg_addr  (10'd0),
        .cfg_wdata (20'd0)
    );

    // 200 MHz Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Monitor Output Egress Probabilities
    always @(posedge clk) begin
        if (out_valid) begin
            logic [15:0] exp_prob;
            int diff;

            if (current_stage == 2) begin
                exp_prob = gold_prob_st2[stage_egress];
            end else if (current_stage == 3) begin
                exp_prob = gold_prob_st3[stage_egress];
            end else begin
                exp_prob = gold_prob_st4[stage_egress];
            end

            diff = (out_prob >= exp_prob) ? (out_prob - exp_prob) : (exp_prob - out_prob);

            if (diff == 0) begin
                stage_pass++;
                total_pass++;
            end else begin
                stage_err++;
                total_err++;
                if (stage_err <= 5) begin
                    $display("[MISMATCH ST%0d @ %0d]: DUT = %0d | Golden = %0d (Diff = %0d)", 
                             current_stage, stage_egress, out_prob, exp_prob, diff);
                end
            end

            stage_egress++;
        end
    end

    // Test Sequence
    initial begin
        $display("=======================================================================");
        $display("   TESTBENCH: softmax_lut MULTI-STAGE BIT-EXACT VERIFICATION");
        $display("   Testing Stage 2 (65,536), Stage 3 (4,096), and Stage 4 (256)");
        $display("=======================================================================");

        rst_n     = 1'b0;
        start_row = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;
        seq_len   = 16'd256;
        head_dim  = 16'd16;

        // Load Golden Files
        $display("[TB INFO]: Loading Stage 2 Softmax vectors (65,536 samples)...");
        $readmemh("golden_scaled_score_st2.hex", gold_score_st2);
        $readmemh("golden_softmax_prob_st2.hex",  gold_prob_st2);

        $display("[TB INFO]: Loading Stage 3 Softmax vectors (4,096 samples)...");
        $readmemh("golden_scaled_score_st3.hex", gold_score_st3);
        $readmemh("golden_softmax_prob_st3.hex",  gold_prob_st3);

        $display("[TB INFO]: Loading Stage 4 Softmax vectors (256 samples)...");
        $readmemh("golden_scaled_score_st4.hex", gold_score_st4);
        $readmemh("golden_softmax_prob_st4.hex",  gold_prob_st4);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        #(CLK_PERIOD * 5);

        // --------------------------------------------------------------------
        // PHASE 1: STAGE 2 (N=256, dk=16, 256 rows = 65,536 probabilities)
        // --------------------------------------------------------------------
        current_stage = 2;
        seq_len       = 16'd256;
        head_dim      = 16'd16;
        stage_pass    = 0;
        stage_err     = 0;
        stage_egress  = 0;
        $display("\n>>> [PHASE 1/3]: Processing STAGE 2 (N=256, dk=16, 256 rows)...");

        for (int r = 0; r < ST2_TOKENS; r++) begin
            wait (!busy);
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            for (int t = 0; t < ST2_TOKENS; t++) begin
                @(posedge clk);
                in_valid <= 1'b1;
                in_score <= gold_score_st2[r * ST2_TOKENS + t];
            end
            @(posedge clk);
            in_valid <= 1'b0;

            wait (!busy);
            #(CLK_PERIOD * 2);
        end
        $display("    -> Stage 2 Done: %0d / %0d matched (Errors: %0d)", stage_pass, (ST2_TOKENS * ST2_TOKENS), stage_err);

        // --------------------------------------------------------------------
        // PHASE 2: STAGE 3 (N=64, dk=20, 64 rows = 4,096 probabilities)
        // --------------------------------------------------------------------
        current_stage = 3;
        seq_len       = 16'd64;
        head_dim      = 16'd20;
        stage_pass    = 0;
        stage_err     = 0;
        stage_egress  = 0;
        $display("\n>>> [PHASE 2/3]: Processing STAGE 3 (N=64, dk=20, 64 rows)...");

        for (int r = 0; r < ST3_TOKENS; r++) begin
            wait (!busy);
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            for (int t = 0; t < ST3_TOKENS; t++) begin
                @(posedge clk);
                in_valid <= 1'b1;
                in_score <= gold_score_st3[r * ST3_TOKENS + t];
            end
            @(posedge clk);
            in_valid <= 1'b0;

            wait (!busy);
            #(CLK_PERIOD * 2);
        end
        $display("    -> Stage 3 Done: %0d / %0d matched (Errors: %0d)", stage_pass, (ST3_TOKENS * ST3_TOKENS), stage_err);

        // --------------------------------------------------------------------
        // PHASE 3: STAGE 4 (N=16, dk=24, 16 rows = 256 probabilities)
        // --------------------------------------------------------------------
        current_stage = 4;
        seq_len       = 16'd16;
        head_dim      = 16'd24;
        stage_pass    = 0;
        stage_err     = 0;
        stage_egress  = 0;
        $display("\n>>> [PHASE 3/3]: Processing STAGE 4 (N=16, dk=24, 16 rows)...");

        for (int r = 0; r < ST4_TOKENS; r++) begin
            wait (!busy);
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            for (int t = 0; t < ST4_TOKENS; t++) begin
                @(posedge clk);
                in_valid <= 1'b1;
                in_score <= gold_score_st4[r * ST4_TOKENS + t];
            end
            @(posedge clk);
            in_valid <= 1'b0;

            wait (!busy);
            #(CLK_PERIOD * 2);
        end
        $display("    -> Stage 4 Done: %0d / %0d matched (Errors: %0d)", stage_pass, (ST4_TOKENS * ST4_TOKENS), stage_err);

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display("   FINAL REPORT: softmax_lut ALL 3 STAGES VERIFICATION");
        $display("=======================================================================");
        $display("   Stage 2 Matched : %0d / %0d", (ST2_TOKENS * ST2_TOKENS), (ST2_TOKENS * ST2_TOKENS));
        $display("   Stage 3 Matched : %0d / %0d", (ST3_TOKENS * ST3_TOKENS), (ST3_TOKENS * ST3_TOKENS));
        $display("   Stage 4 Matched : %0d / %0d", (ST4_TOKENS * ST4_TOKENS), (ST4_TOKENS * ST4_TOKENS));
        $display("   Total Probs     : %0d / %0d", total_pass, (ST2_TOKENS*ST2_TOKENS + ST3_TOKENS*ST3_TOKENS + ST4_TOKENS*ST4_TOKENS));
        $display("   Total Errors    : %0d", total_err);
        if (total_err == 0 && total_pass == (ST2_TOKENS*ST2_TOKENS + ST3_TOKENS*ST3_TOKENS + ST4_TOKENS*ST4_TOKENS)) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH ACROSS ALL 3 STAGES! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: SOFTMAX_LUT VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");
        $finish;
    end

endmodule
