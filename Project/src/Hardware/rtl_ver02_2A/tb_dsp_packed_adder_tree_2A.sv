`timescale 1ns / 1ps
// ============================================================================
// Testbench Name: tb_dsp_packed_adder_tree_2A
// Target Architecture: Variant 2A (High-Throughput DSP-Packed Streaming Engine)
// Description:   Comprehensive Bit-Exact Verification of dsp_packed_adder_tree
//               (8 DSP48E2 PE slices + 3-stage adder tree + serializer)
//               against golden_gemm_qk.hex (65,536 matrix elements).
// Standard:      IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_dsp_packed_adder_tree_2A;

    localparam int CLK_PERIOD = 10;
    localparam int NUM_TOKENS = 256;
    localparam int HEAD_DIM   = 16;
    localparam int TOTAL_SCORES = NUM_TOKENS * NUM_TOKENS; // 65,536

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

    // Memory arrays for vectors and golden reference
    logic signed [7:0]  q_mem      [0:NUM_TOKENS-1][0:HEAD_DIM-1];
    logic signed [7:0]  k_mem      [0:NUM_TOKENS-1][0:HEAD_DIM-1];
    logic signed [31:0] golden_qk  [0:TOTAL_SCORES-1];

    // DUT Instantiation
    dsp_packed_adder_tree #(
        .HEAD_DIM  (16),
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

    // Clock Generation
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // Golden verification counters
    int score_idx   = 0;
    int match_count = 0;
    int error_count = 0;

    // Continuous output checker
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            if (score_idx < TOTAL_SCORES) begin
                if (dot_product === golden_qk[score_idx]) begin
                    match_count++;
                end else begin
                    error_count++;
                    if (error_count <= 10) begin
                        $display("[ERROR]: Score #%0d Mismatch! DUT = %0d, Golden = %0d",
                                 score_idx, dot_product, golden_qk[score_idx]);
                    end
                end
                score_idx++;
            end
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
        $display("   TESTBENCH: DSP-PACKED ADDER TREE VERIFICATION (Variant 2A)");
        $display("   8 DSP48E2 Hard PEs + 3-Stage Adder Tree + Zero-Bubble Serializer");
        $display("=======================================================================");

        // Load hex data
        $readmemh("q_matrix.hex", q_mem);
        $readmemh("k_matrix.hex", k_mem);
        $readmemh("golden_gemm_qk.hex", golden_qk);
        $display("[TB INFO]: Loaded %0d Q tokens, %0d K tokens.", NUM_TOKENS, NUM_TOKENS);
        $display("[TB INFO]: Loaded %0d expected scores from golden_gemm_qk.hex.", TOTAL_SCORES);

        // Reset DUT
        #(CLK_PERIOD * 4);
        @(negedge clk) rst_n = 1;
        $display("[TB INFO]: Reset released. Starting stream through 8 DSP Packing PEs...");

        // Stream all 256 Query rows
        for (int r = 0; r < NUM_TOKENS; r++) begin
            // For each Query row, stream 128 Key pairs (256 keys total)
            for (int p = 0; p < NUM_TOKENS / 2; p++) begin
                int k_even_idx;
                int k_odd_idx;
                k_even_idx = 2 * p;
                k_odd_idx  = 2 * p + 1;

                // Sub-cycle / Beat 0: dimensions 0 to 7
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 1'b0;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem[r][d];
                    k_even[d] <= k_mem[k_even_idx][d];
                    k_odd[d]  <= k_mem[k_odd_idx][d];
                end

                // Sub-cycle / Beat 1: dimensions 8 to 15
                @(posedge clk);
                in_valid <= 1'b1;
                in_beat  <= 1'b1;
                for (int d = 0; d < 8; d++) begin
                    q_vec[d]  <= q_mem[r][d + 8];
                    k_even[d] <= k_mem[k_even_idx][d + 8];
                    k_odd[d]  <= k_mem[k_odd_idx][d + 8];
                end
            end

            // End of row stream: deassert in_valid and allow pipeline flush
            @(posedge clk);
            in_valid <= 1'b0;
            in_beat  <= 1'b0;
            for (int d = 0; d < 8; d++) begin
                q_vec[d]  <= 8'sd0;
                k_even[d] <= 8'sd0;
                k_odd[d]  <= 8'sd0;
            end

            // Wait 8 cycles for pipeline flushing before row update
            repeat (8) @(posedge clk);

            if ((r + 1) % 64 == 0 || r == NUM_TOKENS - 1) begin
                $display("[TB INFO]: Row %0d / 256 completed. Total Checked Scores: %0d", r + 1, score_idx);
            end
        end

        // Wait extra cycles for final egress drainage
        #(CLK_PERIOD * 20);

        // Verification Report
        $display("\n=======================================================================");
        $display("   DSP-PACKED ADDER TREE VERIFICATION SUMMARY (Variant 2A)");
        $display("=======================================================================");
        $display("   Total Tested Scores : %0d / %0d", score_idx, TOTAL_SCORES);
        $display("   Exact Matched Scores: %0d", match_count);
        $display("   Mismatches          : %0d", error_count);
        if (error_count == 0 && score_idx == TOTAL_SCORES) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH ACHIEVED! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: VERIFICATION FAILED! [ERROR] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule
