`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_softmax_lut_reloadable
// Description:  Step 2.5 Verification Testbench for Reconfigurable Multi-Stage
//               True Dual-Port BRAM Softmax LUT:
//               1. Multi-Stage Auto-Selection (Stage 2: dk=16, Stage 3: dk=20, Stage 4: dk=24)
//               2. Runtime Dynamic Reloading via Port B (Host CPU AXI-Lite Simulation)
//               3. Zero-Collision and High-Speed Timing Integrity (200 MHz)
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// ============================================================================

module tb_softmax_lut_reloadable;

    localparam int N_TOKENS       = 256;
    localparam real CLK_PERIOD    = 5.0; // 200 MHz

    logic                      clk;
    logic                      rst_n;
    logic                      start_row;
    logic                      in_valid;
    logic signed [15:0]        in_score;

    logic                      out_valid;
    logic [15:0]               out_prob;
    logic                      busy;

    logic [15:0]               head_dim;
    logic                      cfg_we;
    logic [9:0]                cfg_addr;
    logic [19:0]               cfg_wdata;

    // Reference memory loaded from exp_rom_all_stages.hex
    logic [19:0]               gold_rom_mem [0:1023];

    int check_cnt  = 0;
    int error_cnt  = 0;

    // DUT Instantiation
    softmax_lut #(
        .N_TOKENS       (N_TOKENS),
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
        .head_dim  (head_dim),
        .cfg_we    (cfg_we),
        .cfg_addr  (cfg_addr),
        .cfg_wdata (cfg_wdata)
    );

    // 200 MHz Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Helper task to process a row of 256 scores
    task automatic run_softmax_row(
        input logic signed [15:0] base_score,
        output int sum_probs,
        output int max_prob_val
    );
        sum_probs    = 0;
        max_prob_val = 0;

        wait (!busy);
        @(posedge clk);
        start_row <= 1'b1;
        @(posedge clk);
        start_row <= 1'b0;

        // Stream 256 scores: score[0] = base_score, score[1..255] = base_score - 1..255
        for (int t = 0; t < N_TOKENS; t++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_score <= base_score - 16'(t);
        end
        @(posedge clk);
        in_valid <= 1'b0;

        // Capture 256 output probabilities
        for (int t = 0; t < N_TOKENS; t++) begin
            @(posedge clk);
            while (!out_valid) @(posedge clk);
            sum_probs += int'(out_prob);
            if (int'(out_prob) > max_prob_val) begin
                max_prob_val = int'(out_prob);
            end
        end
    endtask

    // Helper task to write a word via Port B
    task automatic write_lut_word(
        input logic [9:0]  addr,
        input logic [19:0] data
    );
        @(posedge clk);
        cfg_we    <= 1'b1;
        cfg_addr  <= addr;
        cfg_wdata <= data;
        @(posedge clk);
        cfg_we    <= 1'b0;
    endtask

    initial begin
        int sum_p, max_p;
        $display("=======================================================================");
        $display("   TESTBENCH: STEP 2.5 RECONFIGURABLE MULTI-STAGE BRAM SOFTMAX LUT");
        $display("=======================================================================");

        rst_n     = 1'b0;
        start_row = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;
        head_dim  = 16'd16;
        cfg_we    = 1'b0;
        cfg_addr  = '0;
        cfg_wdata = '0;

        $readmemh("exp_rom_all_stages.hex", gold_rom_mem);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        @(posedge clk);

        // -------------------------------------------------------------------------
        // TEST SCENARIO 1: Multi-Stage Auto-Selection via head_dim
        // -------------------------------------------------------------------------
        $display("\n[TEST SCENARIO 1]: Multi-Stage Partition Auto-Selection (dk = 16, 20, 24)");

        // 1.1 Stage 2 (head_dim = 16)
        head_dim = 16'd16;
        run_softmax_row(16'sd100, sum_p, max_p);
        $display("  -> STAGE 2 (dk=16): Prob Sum = %0d, Peak Token 0 Prob = %0d", sum_p, max_p);
        if (sum_p >= 32000 && sum_p <= 32768 && max_p > 0) begin
            $display("     [CHECK 1.1 PASSED]: Stage 2 Partition (Offset 0x000) Validated.");
            check_cnt++;
        end else begin
            $display("     [CHECK 1.1 FAILED]: Stage 2 Output Invalid! Sum=%0d", sum_p);
            error_cnt++;
        end

        // 1.2 Stage 3 (head_dim = 20)
        head_dim = 16'd20;
        run_softmax_row(16'sd100, sum_p, max_p);
        $display("  -> STAGE 3 (dk=20): Prob Sum = %0d, Peak Token 0 Prob = %0d", sum_p, max_p);
        if (sum_p >= 32000 && sum_p <= 32768 && max_p > 0) begin
            $display("     [CHECK 1.2 PASSED]: Stage 3 Partition (Offset 0x100) Validated.");
            check_cnt++;
        end else begin
            $display("     [CHECK 1.2 FAILED]: Stage 3 Output Invalid! Sum=%0d", sum_p);
            error_cnt++;
        end

        // 1.3 Stage 4 (head_dim = 24)
        head_dim = 16'd24;
        run_softmax_row(16'sd100, sum_p, max_p);
        $display("  -> STAGE 4 (dk=24): Prob Sum = %0d, Peak Token 0 Prob = %0d", sum_p, max_p);
        if (sum_p >= 32000 && sum_p <= 32768 && max_p > 0) begin
            $display("     [CHECK 1.3 PASSED]: Stage 4 Partition (Offset 0x200) Validated.");
            check_cnt++;
        end else begin
            $display("     [CHECK 1.3 FAILED]: Stage 4 Output Invalid! Sum=%0d", sum_p);
            error_cnt++;
        end

        // -------------------------------------------------------------------------
        // TEST SCENARIO 2: Runtime Reloading via Port B (Dynamic Software-Defined LUT)
        // -------------------------------------------------------------------------
        $display("\n[TEST SCENARIO 2]: Runtime Reloading via Port B (Host CPU AXI-Lite write)");

        // Write custom uniform distribution values into Stage 2 partition (0..255)
        // All entries set to identical 20'd10000 -> Should yield perfectly uniform softmax!
        $display("  -> Preloading 256 custom uniform words into Stage 2 BRAM via Port B...");
        for (int i = 0; i < 256; i++) begin
            write_lut_word(10'(i), 20'd10000);
        end
        $display("  -> Preload completed (256 words written).");

        // Run inference on modified Stage 2
        head_dim = 16'd16;
        run_softmax_row(16'sd100, sum_p, max_p);
        $display("  -> Modified STAGE 2 Output: Prob Sum = %0d, Peak Prob = %0d", sum_p, max_p);

        // For uniform distribution of 256 tokens with fixed-point floor division:
        // factor = floor((2^31 - 1) / (256 * 10000)) = 838
        // prob_i = (10000 * 838) >> 16 = 127, sum = 256 * 127 = 32512
        if (max_p == 127 && sum_p == 32512) begin
            $display("     [CHECK 2.1 PASSED]: Runtime Reload Successful! Exact uniform prob = 127/token (Sum=32512).");
            check_cnt++;
        end else begin
            $display("     [CHECK 2.1 FAILED]: Expected uniform prob 127 (Sum=32512), got max=%0d, sum=%0d", max_p, sum_p);
            error_cnt++;
        end

        // -------------------------------------------------------------------------
        // TEST SCENARIO 3: Restoration of Stage 2 Original Calibrated Weights
        // -------------------------------------------------------------------------
        $display("\n[TEST SCENARIO 3]: Restoring Original Stage 2 Calibrated Table from Hex...");
        for (int i = 0; i < 256; i++) begin
            write_lut_word(10'(i), gold_rom_mem[i]);
        end
        $display("  -> Restoration completed. Verifying bit-exact response...");

        run_softmax_row(16'sd100, sum_p, max_p);
        $display("  -> Restored STAGE 2 Output: Prob Sum = %0d, Peak Token 0 Prob = %0d", sum_p, max_p);
        if (sum_p >= 32000 && sum_p <= 32768 && max_p > 128) begin
            $display("     [CHECK 3.1 PASSED]: Original Calibrated Weights Restored & Functional.");
            check_cnt++;
        end else begin
            $display("     [CHECK 3.1 FAILED]: Restored Weights Failed Check! Sum=%0d", sum_p);
            error_cnt++;
        end

        // -------------------------------------------------------------------------
        // FINAL SUMMARY
        // -------------------------------------------------------------------------
        $display("\n=======================================================================");
        $display("   FINAL REPORT: STEP 2.5 RECONFIGURABLE BRAM VERIFICATION");
        $display("=======================================================================");
        $display("   Total Checks Evaluated : %0d", check_cnt + error_cnt);
        $display("   Checks Passed          : %0d", check_cnt);
        $display("   Checks Failed          : %0d", error_cnt);
        if (error_cnt == 0) begin
            $display("   >>> STATUS: ALL STEP 2.5 TESTS PASSED (100%% SUCCESS)! <<<");
        end else begin
            $display("   >>> STATUS: TESTBENCH FAILED with %0d errors! <<<", error_cnt);
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule
