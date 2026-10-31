`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_golden_softmax_lut
// Description:  Bit-Exact Verification Testbench for softmax_lut (rtl_ver02)
//               against PyTorch I-ViT Golden Vectors:
//               Input:  golden_scaled_score.hex (256 rows x 256 INT16 elements)
//               Golden: golden_softmax_prob.hex (256 rows x 256 UINT16 elements)
// ============================================================================

module tb_golden_softmax_lut;

    localparam int N_TOKENS       = 256;
    localparam int NUM_TEST_ROWS  = 4;   // Test first 4 rows (1024 tokens)
    localparam real CLK_PERIOD    = 5.0; // 200 MHz

    logic                      clk;
    logic                      rst_n;
    logic                      start_row;
    logic                      in_valid;
    logic signed [15:0]        in_score;

    logic                      out_valid;
    logic [15:0]               out_prob;
    logic                      busy;

    // Golden Vector Memories
    logic signed [15:0]        gold_scaled_mem [0:(NUM_TEST_ROWS * N_TOKENS)-1];
    logic [15:0]               gold_prob_mem   [0:(NUM_TEST_ROWS * N_TOKENS)-1];

    int pass_count  = 0;
    int error_count = 0;
    int egress_idx  = 0;

    // DUT Instantiation (Step 2.5 Reconfigurable Multi-Stage BRAM)
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
        .head_dim  (16'd16),    // Stage 2 (dk=16) partition offset 0x000
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
            exp_prob = gold_prob_mem[egress_idx];
            diff = (out_prob >= exp_prob) ? (out_prob - exp_prob) : (exp_prob - out_prob);

            if (diff == 0) begin
                pass_count++;
            end else begin
                error_count++;
                if (error_count <= 10) begin
                    $display("[SOFTMAX MISMATCH @%0d]: DUT = %0d | Golden = %0d (Diff = %0d)", 
                             egress_idx, out_prob, exp_prob, diff);
                end
            end

            if (egress_idx < 16 || egress_idx % 256 == 0) begin
                $display("[ROW %0d TOKEN %03d]: DUT Prob = %0d/32768 | Golden = %0d/32768 (Diff = %0d)", 
                         egress_idx / 256, egress_idx % 256, out_prob, exp_prob, diff);
            end

            egress_idx++;
        end
    end

    // Test Sequence
    initial begin
        $display("=======================================================================");
        $display("   TESTBENCH: softmax_lut vs PyTorch I-ViT Golden Model (Stage 2)");
        $display("=======================================================================");

        rst_n     = 1'b0;
        start_row = 1'b0;
        in_valid  = 1'b0;
        in_score  = '0;

        // Load Golden Files
        $display("[TB INFO]: Loading golden_scaled_score.hex (%0d samples)...", NUM_TEST_ROWS * N_TOKENS);
        $readmemh("golden_scaled_score.hex", gold_scaled_mem);
        $display("[TB INFO]: Loading golden_softmax_prob.hex (%0d samples)...", NUM_TEST_ROWS * N_TOKENS);
        $readmemh("golden_softmax_prob.hex", gold_prob_mem);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset Released. Starting row-by-row Softmax processing...");

        for (int r = 0; r < NUM_TEST_ROWS; r++) begin
            $display("[TB INFO]: Starting Row %0d of 256 tokens...", r);
            
            // Wait for DUT to be ready
            wait (!busy);
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            // Stream 256 scores
            for (int t = 0; t < N_TOKENS; t++) begin
                @(posedge clk);
                in_valid <= 1'b1;
                in_score <= gold_scaled_mem[r * N_TOKENS + t];
            end

            @(posedge clk);
            in_valid <= 1'b0;

            // Wait for row normalization and output completion
            wait (!busy);
            #(CLK_PERIOD * 10);
        end

        #(CLK_PERIOD * 50);

        $display("=======================================================================");
        $display("   FINAL VERIFICATION REPORT: SOFTMAX_LUT vs GOLDEN MODEL");
        $display("=======================================================================");
        $display("   Total Softmax Tokens Tested : %0d", egress_idx);
        $display("   Matched Tokens (0 LSB Error) : %0d", pass_count);
        $display("   Mismatched Tokens (> 0 LSB) : %0d", error_count);
        if (error_count == 0 && egress_idx == NUM_TEST_ROWS * N_TOKENS) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH (Zero Error)! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: SOFTMAX VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================");
        $finish;
    end


    // Setup waveform dumping for tools like GTKWave
    initial begin
        $dumpfile("tb_golden_softmax_lut.vcd");
        $dumpvars(0, tb_golden_softmax_lut);
    end

endmodule
