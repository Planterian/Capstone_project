`timescale 1ns / 1ps
// ============================================================================
// Testbench Name: tb_pipelined_adder_tree_1A
// Target Architecture: Variant 1A (Pure Logic Fabric Multipliers & Adder Tree)
// Description:   Comprehensive Multi-Stage Bit-Exact Verification of
//               pipelined_adder_tree across all 3 MobileViT Stages:
//               - Stage 2: N=256, dk=16 (65,536 scores)
//               - Stage 3: N=64,  dk=20 ( 4,096 scores)
//               - Stage 4: N=16,  dk=24 (   256 scores)
//               Total dot-products verified: 69,888 scores
// Standard:      IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_pipelined_adder_tree_1A;

    localparam int N          = 32;
    localparam int IN_WIDTH   = 8;
    localparam int OUT_WIDTH  = 32;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    // Stage dimensions
    localparam int ST2_TOKENS   = 256;
    localparam int ST2_HEAD_DIM = 16;
    localparam int ST2_SCORES   = ST2_TOKENS * ST2_TOKENS; // 65,536

    localparam int ST3_TOKENS   = 64;
    localparam int ST3_HEAD_DIM = 20;
    localparam int ST3_SCORES   = ST3_TOKENS * ST3_TOKENS; // 4,096

    localparam int ST4_TOKENS   = 16;
    localparam int ST4_HEAD_DIM = 24;
    localparam int ST4_SCORES   = ST4_TOKENS * ST4_TOKENS; // 256

    localparam int TOTAL_SCORES = ST2_SCORES + ST3_SCORES + ST4_SCORES; // 69,888

    // DUT Signals
    logic                          clk;
    logic                          rst_n;
    logic                          in_valid;
    logic signed [IN_WIDTH-1:0]    a [0:N-1];
    logic signed [IN_WIDTH-1:0]    b [0:N-1];
    logic                          out_valid;
    logic signed [OUT_WIDTH-1:0]   dot_product;

    // Golden memory buffers for 3 Stages
    logic [7:0]          q_mem_st2 [0:(ST2_TOKENS * ST2_HEAD_DIM)-1];
    logic [7:0]          k_mem_st2 [0:(ST2_TOKENS * ST2_HEAD_DIM)-1];
    logic signed [31:0]  gold_gemm_st2 [0:ST2_SCORES-1];

    logic [7:0]          q_mem_st3 [0:(ST3_TOKENS * ST3_HEAD_DIM)-1];
    logic [7:0]          k_mem_st3 [0:(ST3_TOKENS * ST3_HEAD_DIM)-1];
    logic signed [31:0]  gold_gemm_st3 [0:ST3_SCORES-1];

    logic [7:0]          q_mem_st4 [0:(ST4_TOKENS * ST4_HEAD_DIM)-1];
    logic [7:0]          k_mem_st4 [0:(ST4_TOKENS * ST4_HEAD_DIM)-1];
    logic signed [31:0]  gold_gemm_st4 [0:ST4_SCORES-1];

    // Verification Tracking
    int current_stage = 2;
    int stage_sample_idx = 0;
    int stage_pass_count = 0;
    int stage_error_count = 0;

    int total_pass_count = 0;
    int total_error_count = 0;

    // Instantiate DUT
    pipelined_adder_tree #(
        .N         (N),
        .IN_WIDTH  (IN_WIDTH),
        .OUT_WIDTH (OUT_WIDTH),
        .HAS_MULT  (1'b1)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .a           (a),
        .b           (b),
        .out_valid   (out_valid),
        .dot_product (dot_product)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Monitor and verify outputs
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            logic signed [31:0] exp_val;

            if (current_stage == 2)
                exp_val = gold_gemm_st2[stage_sample_idx];
            else if (current_stage == 3)
                exp_val = gold_gemm_st3[stage_sample_idx];
            else
                exp_val = gold_gemm_st4[stage_sample_idx];

            if (dot_product === exp_val) begin
                stage_pass_count++;
                total_pass_count++;
            end else begin
                if (stage_error_count < 5) begin
                    $display("[ERROR ST%0d @ Score #%0d]: DUT = %0d | Golden = %0d",
                             current_stage, stage_sample_idx, dot_product, exp_val);
                end
                stage_error_count++;
                total_error_count++;
            end

            // Progress log
            if (stage_sample_idx < 4 || stage_sample_idx == 255 || 
                (current_stage == 2 && stage_sample_idx == 65535) ||
                (current_stage == 3 && stage_sample_idx == 4095) ||
                (current_stage == 4 && stage_sample_idx == 255)) begin
                $display("[SCORE ST%0d #%05d]: DUT = %6d | Golden = %6d (%s)",
                         current_stage, stage_sample_idx, dot_product, exp_val,
                         (dot_product === exp_val) ? "MATCH" : "MISMATCH");
            end

            stage_sample_idx++;
        end
    end

    // Task to run verification for one stage
    task automatic run_stage_verification(
        input int stage_num,
        input int tokens,
        input int head_dim,
        input int expected_scores
    );
        current_stage     = stage_num;
        stage_sample_idx  = 0;
        stage_pass_count  = 0;
        stage_error_count = 0;

        $display("\n-----------------------------------------------------------------------");
        $display("   STARTING STAGE %0d: N=%0d, dk=%0d (%0d Scores)",
                 stage_num, tokens, head_dim, expected_scores);
        $display("-----------------------------------------------------------------------");

        for (int row = 0; row < tokens; row++) begin
            // Latch Query row for this row
            for (int d = 0; d < N; d++) begin
                if (d < head_dim) begin
                    if (stage_num == 2)
                        a[d] <= $signed(q_mem_st2[row * head_dim + d]);
                    else if (stage_num == 3)
                        a[d] <= $signed(q_mem_st3[row * head_dim + d]);
                    else
                        a[d] <= $signed(q_mem_st4[row * head_dim + d]);
                end else begin
                    a[d] <= 8'sd0;
                end
            end

            // Stream all column vectors of K
            for (int col = 0; col < tokens; col++) begin
                @(posedge clk);
                in_valid <= 1'b1;
                for (int d = 0; d < N; d++) begin
                    if (d < head_dim) begin
                        if (stage_num == 2)
                            b[d] <= $signed(k_mem_st2[col * head_dim + d]);
                        else if (stage_num == 3)
                            b[d] <= $signed(k_mem_st3[col * head_dim + d]);
                        else
                            b[d] <= $signed(k_mem_st4[col * head_dim + d]);
                    end else begin
                        b[d] <= 8'sd0;
                    end
                end
            end

            @(posedge clk);
            in_valid <= 1'b0;

            // Wait pipeline flush cycles before moving to next Query row
            repeat (8) @(posedge clk);
        end

        // Wait trailing cycles until all scores of this stage are received
        while (stage_sample_idx < expected_scores) begin
            @(posedge clk);
        end
        repeat (5) @(posedge clk);

        $display("   [STAGE %0d RESULT]: %0d / %0d exact matches (%0d mismatches) - %s",
                 stage_num, stage_pass_count, expected_scores, stage_error_count,
                 (stage_error_count == 0 && stage_pass_count == expected_scores) ? "PASS" : "FAIL");
    endtask

    initial begin
        $display("=======================================================================");
        $display("   RUNNING MULTI-STAGE UNIT VERIFICATION: PIPELINED ADDER TREE 1A");
        $display("   Target: MobileViT-XXS Stages 2, 3, 4 (Total: 69,888 Scores)");
        $display("=======================================================================");

        rst_n    = 0;
        in_valid = 0;
        for (int i = 0; i < N; i++) begin
            a[i] = '0;
            b[i] = '0;
        end

        // Load hex vectors for all 3 Stages
        $readmemh("q_matrix_st2.hex",       q_mem_st2);
        $readmemh("k_matrix_st2.hex",       k_mem_st2);
        $readmemh("golden_gemm_qk_st2.hex", gold_gemm_st2);

        $readmemh("q_matrix_st3.hex",       q_mem_st3);
        $readmemh("k_matrix_st3.hex",       k_mem_st3);
        $readmemh("golden_gemm_qk_st3.hex", gold_gemm_st3);

        $readmemh("q_matrix_st4.hex",       q_mem_st4);
        $readmemh("k_matrix_st4.hex",       k_mem_st4);
        $readmemh("golden_gemm_qk_st4.hex", gold_gemm_st4);

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // Run Stage 2: N=256, dk=16
        run_stage_verification(2, ST2_TOKENS, ST2_HEAD_DIM, ST2_SCORES);

        // Run Stage 3: N=64, dk=20
        run_stage_verification(3, ST3_TOKENS, ST3_HEAD_DIM, ST3_SCORES);

        // Run Stage 4: N=16, dk=24
        run_stage_verification(4, ST4_TOKENS, ST4_HEAD_DIM, ST4_SCORES);

        $display("\n=======================================================================");
        $display("   FINAL REPORT: PIPELINED ADDER TREE 1A MULTI-STAGE VERIFICATION");
        $display("=======================================================================");
        $display("   Total Scores Verified : %0d / %0d", total_pass_count + total_error_count, TOTAL_SCORES);
        $display("   Exact Matches         : %0d", total_pass_count);
        $display("   Mismatches            : %0d", total_error_count);

        if (total_error_count == 0 && total_pass_count == TOTAL_SCORES) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH (69,888 / 69,888)! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: MULTI-STAGE VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end

    // Setup waveform dumping for tools like GTKWave
    initial begin
        $dumpfile("tb_pipelined_adder_tree_1A.vcd");
        $dumpvars(0, tb_pipelined_adder_tree_1A);
    end

endmodule
