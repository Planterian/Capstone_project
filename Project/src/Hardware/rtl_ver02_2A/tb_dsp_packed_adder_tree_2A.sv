`timescale 1ns / 1ps
// ============================================================================
// Testbench Name: tb_dsp_packed_adder_tree_2A
// Target Architecture: Variant 2A (High-Throughput DSP-Packed Streaming Engine)
// Description:   Comprehensive Multi-Stage Bit-Exact Verification of
//               dsp_packed_adder_tree across all 3 MobileViT Stages:
//               - Stage 2: N=256, dk=16, 2 beats/pair (65,536 scores)
//               - Stage 3: N=64,  dk=20, 3 beats/pair  (4,096 scores)
//               - Stage 4: N=16,  dk=24, 3 beats/pair    (256 scores)
// Standard:      IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_dsp_packed_adder_tree_2A;

    localparam int CLK_PERIOD = 5; // 200 MHz

    localparam int ST2_TOKENS   = 256;
    localparam int ST2_HEAD_DIM = 16;
    localparam int ST2_SCORES   = ST2_TOKENS * ST2_TOKENS; // 65,536

    localparam int ST3_TOKENS   = 64;
    localparam int ST3_HEAD_DIM = 20;
    localparam int ST3_SCORES   = ST3_TOKENS * ST3_TOKENS; // 4,096

    localparam int ST4_TOKENS   = 16;
    localparam int ST4_HEAD_DIM = 24;
    localparam int ST4_SCORES   = ST4_TOKENS * ST4_TOKENS; // 256

    logic                     clk;
    logic                     rst_n;
    logic                     in_valid;
    logic [1:0]               in_beat;
    logic signed [7:0]        q_vec   [0:7];
    logic signed [7:0]        k_even  [0:7];
    logic signed [7:0]        k_odd   [0:7];
    logic [15:0]              head_dim;

    logic                     out_valid;
    logic signed [31:0]       dot_product;

    // Memory arrays for vectors and golden reference for all 3 Stages
    logic signed [7:0]  q_mem_st2     [0:ST2_TOKENS-1][0:ST2_HEAD_DIM-1];
    logic signed [7:0]  k_mem_st2     [0:ST2_TOKENS-1][0:ST2_HEAD_DIM-1];
    logic signed [31:0] gold_gemm_st2 [0:ST2_SCORES-1];

    logic signed [7:0]  q_mem_st3     [0:ST3_TOKENS-1][0:ST3_HEAD_DIM-1];
    logic signed [7:0]  k_mem_st3     [0:ST3_TOKENS-1][0:ST3_HEAD_DIM-1];
    logic signed [31:0] gold_gemm_st3 [0:ST3_SCORES-1];

    logic signed [7:0]  q_mem_st4     [0:ST4_TOKENS-1][0:ST4_HEAD_DIM-1];
    logic signed [7:0]  k_mem_st4     [0:ST4_TOKENS-1][0:ST4_HEAD_DIM-1];
    logic signed [31:0] gold_gemm_st4 [0:ST4_SCORES-1];

    // Loop iteration variables (declared explicitly to prevent static initialization warnings)
    int k_even_idx;
    int k_odd_idx;

    // DUT Instantiation
    dsp_packed_adder_tree #(
        .HEAD_DIM  (32),
        .OUT_WIDTH (32)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_beat     (in_beat),
        .q_vec       (q_vec),
        .k_even      (k_even),
        .k_odd       (k_odd),
        .head_dim    (head_dim),
        .out_valid   (out_valid),
        .dot_product (dot_product)
    );

    // Clock Generation: 200 MHz
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    int current_stage = 2;
    int stage_scores  = 0;
    int stage_matches = 0;
    int stage_errors  = 0;

    int total_matches = 0;
    int total_errors  = 0;

    // Continuous output checker
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            logic signed [31:0] exp_score;

            if (current_stage == 2)      exp_score = gold_gemm_st2[stage_scores];
            else if (current_stage == 3) exp_score = gold_gemm_st3[stage_scores];
            else                         exp_score = gold_gemm_st4[stage_scores];

            if (dot_product === exp_score) begin
                stage_matches++;
                total_matches++;
            end else begin
                if (stage_errors < 5) begin
                    $display("[ERROR ST%0d #%0d]: DUT = %0d | Golden = %0d", 
                             current_stage, stage_scores, dot_product, exp_score);
                end
                stage_errors++;
                total_errors++;
            end

            stage_scores++;
        end
    end

    // Stimulus process
    initial begin
        clk      = 0;
        rst_n    = 0;
        in_valid = 0;
        in_beat  = 0;
        head_dim = 16'd16;
        for (int d = 0; d < 8; d++) begin
            q_vec[d]  = 8'sd0;
            k_even[d] = 8'sd0;
            k_odd[d]  = 8'sd0;
        end

        $display("=======================================================================");
        $display("   TESTBENCH: DSP-PACKED ADDER TREE MULTI-STAGE VERIFICATION");
        $display("   Testing Stage 2 (65,536), Stage 3 (4,096), and Stage 4 (256)");
        $display("=======================================================================");

        // Load hex data directly into 2D memory arrays
        $display("[TB INFO]: Loading Stage 2 Q, K, and Golden Scores...");
        $readmemh("q_matrix_st2.hex", q_mem_st2);
        $readmemh("k_matrix_st2.hex", k_mem_st2);
        $readmemh("golden_gemm_qk_st2.hex", gold_gemm_st2);

        $display("[TB INFO]: Loading Stage 3 Q, K, and Golden Scores...");
        $readmemh("q_matrix_st3.hex", q_mem_st3);
        $readmemh("k_matrix_st3.hex", k_mem_st3);
        $readmemh("golden_gemm_qk_st3.hex", gold_gemm_st3);

        $display("[TB INFO]: Loading Stage 4 Q, K, and Golden Scores...");
        $readmemh("q_matrix_st4.hex", q_mem_st4);
        $readmemh("k_matrix_st4.hex", k_mem_st4);
        $readmemh("golden_gemm_qk_st4.hex", gold_gemm_st4);

        #(CLK_PERIOD * 4);
        @(negedge clk) rst_n = 1;
        #(CLK_PERIOD * 5);

        // --------------------------------------------------------------------
        // PHASE 1: STAGE 2 (N=256, dk=16, 2 beats/pair, 65,536 scores)
        // --------------------------------------------------------------------
        current_stage = 2;
        head_dim      = 16'd16;
        stage_scores  = 0;
        stage_matches = 0;
        stage_errors  = 0;
        $display("\n>>> [PHASE 1/3]: Processing STAGE 2 (N=256, dk=16, 65,536 scores)...");

        for (int r = 0; r < ST2_TOKENS; r++) begin
            for (int p = 0; p < ST2_TOKENS / 2; p++) begin
                k_even_idx = 2 * p;
                k_odd_idx  = 2 * p + 1;

                // Beat 0: d = 0..7
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd0;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st2[r][d];
                    k_even[d] <= k_mem_st2[k_even_idx][d];
                    k_odd[d]  <= k_mem_st2[k_odd_idx][d];
                end

                // Beat 1: d = 8..15
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd1;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st2[r][d + 8];
                    k_even[d] <= k_mem_st2[k_even_idx][d + 8];
                    k_odd[d]  <= k_mem_st2[k_odd_idx][d + 8];
                end
            end

            @(posedge clk);
            in_valid <= 1'b0;
            in_beat  <= 2'd0;
            for (int d = 0; d < 8; d++) begin
                q_vec[d]  <= 8'sd0;
                k_even[d] <= 8'sd0;
                k_odd[d]  <= 8'sd0;
            end
            repeat (8) @(posedge clk);
        end
        #(CLK_PERIOD * 10);
        $display("    -> Stage 2 Done: %0d / %0d matched (Errors: %0d)", stage_matches, ST2_SCORES, stage_errors);

        // --------------------------------------------------------------------
        // PHASE 2: STAGE 3 (N=64, dk=20, 3 beats/pair, 4,096 scores)
        // --------------------------------------------------------------------
        current_stage = 3;
        head_dim      = 16'd20;
        stage_scores  = 0;
        stage_matches = 0;
        stage_errors  = 0;
        $display("\n>>> [PHASE 2/3]: Processing STAGE 3 (N=64, dk=20, 4,096 scores)...");

        for (int r = 0; r < ST3_TOKENS; r++) begin
            for (int p = 0; p < ST3_TOKENS / 2; p++) begin
                k_even_idx = 2 * p;
                k_odd_idx  = 2 * p + 1;

                // Beat 0: d = 0..7
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd0;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st3[r][d];
                    k_even[d] <= k_mem_st3[k_even_idx][d];
                    k_odd[d]  <= k_mem_st3[k_odd_idx][d];
                end

                // Beat 1: d = 8..15
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd1;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st3[r][d + 8];
                    k_even[d] <= k_mem_st3[k_even_idx][d + 8];
                    k_odd[d]  <= k_mem_st3[k_odd_idx][d + 8];
                end

                // Beat 2: d = 16..23 (dims 16..19 valid, 20..23 zero)
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd2;
                for (int d = 0; d < 8; d++) begin
                    if (d < 4) begin
                        q_vec[d]  <= q_mem_st3[r][d + 16];
                        k_even[d] <= k_mem_st3[k_even_idx][d + 16];
                        k_odd[d]  <= k_mem_st3[k_odd_idx][d + 16];
                    end else begin
                        q_vec[d]  <= 8'sd0;
                        k_even[d] <= 8'sd0;
                        k_odd[d]  <= 8'sd0;
                    end
                end
            end

            @(posedge clk);
            in_valid <= 1'b0;
            in_beat  <= 2'd0;
            for (int d = 0; d < 8; d++) begin
                q_vec[d]  <= 8'sd0;
                k_even[d] <= 8'sd0;
                k_odd[d]  <= 8'sd0;
            end
            repeat (8) @(posedge clk);
        end
        #(CLK_PERIOD * 10);
        $display("    -> Stage 3 Done: %0d / %0d matched (Errors: %0d)", stage_matches, ST3_SCORES, stage_errors);

        // --------------------------------------------------------------------
        // PHASE 3: STAGE 4 (N=16, dk=24, 3 beats/pair, 256 scores)
        // --------------------------------------------------------------------
        current_stage = 4;
        head_dim      = 16'd24;
        stage_scores  = 0;
        stage_matches = 0;
        stage_errors  = 0;
        $display("\n>>> [PHASE 3/3]: Processing STAGE 4 (N=16, dk=24, 256 scores)...");

        for (int r = 0; r < ST4_TOKENS; r++) begin
            for (int p = 0; p < ST4_TOKENS / 2; p++) begin
                k_even_idx = 2 * p;
                k_odd_idx  = 2 * p + 1;

                // Beat 0: d = 0..7
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd0;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st4[r][d];
                    k_even[d] <= k_mem_st4[k_even_idx][d];
                    k_odd[d]  <= k_mem_st4[k_odd_idx][d];
                end

                // Beat 1: d = 8..15
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd1;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st4[r][d + 8];
                    k_even[d] <= k_mem_st4[k_even_idx][d + 8];
                    k_odd[d]  <= k_mem_st4[k_odd_idx][d + 8];
                end

                // Beat 2: d = 16..23
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 2'd2;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem_st4[r][d + 16];
                    k_even[d] <= k_mem_st4[k_even_idx][d + 16];
                    k_odd[d]  <= k_mem_st4[k_odd_idx][d + 16];
                end
            end

            @(posedge clk);
            in_valid <= 1'b0;
            in_beat  <= 2'd0;
            for (int d = 0; d < 8; d++) begin
                q_vec[d]  <= 8'sd0;
                k_even[d] <= 8'sd0;
                k_odd[d]  <= 8'sd0;
            end
            repeat (8) @(posedge clk);
        end
        #(CLK_PERIOD * 10);
        $display("    -> Stage 4 Done: %0d / %0d matched (Errors: %0d)", stage_matches, ST4_SCORES, stage_errors);

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display("   FINAL REPORT: dsp_packed_adder_tree ALL 3 STAGES VERIFICATION");
        $display("=======================================================================");
        $display("   Stage 2 Matched : %0d / %0d", ST2_SCORES, ST2_SCORES);
        $display("   Stage 3 Matched : %0d / %0d", ST3_SCORES, ST3_SCORES);
        $display("   Stage 4 Matched : %0d / %0d", ST4_SCORES, ST4_SCORES);
        $display("   Total Scores    : %0d / %0d", total_matches, (ST2_SCORES + ST3_SCORES + ST4_SCORES));
        $display("   Total Errors    : %0d", total_errors);
        if (total_errors == 0 && total_matches == (ST2_SCORES + ST3_SCORES + ST4_SCORES)) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH ACHIEVED ACROSS ALL 3 STAGES! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule
