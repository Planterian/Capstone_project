`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_pipelined_adder_tree_1A
// Description:  Unit verification for 16-element Pipelined Adder Tree against
//               PyTorch golden QK^T matrix products (golden_gemm_qk.hex).
//               Tests all 256 x 256 = 65,536 dot products.
// ============================================================================

module tb_pipelined_adder_tree_1A;

    localparam int N          = 16;
    localparam int IN_WIDTH   = 8;
    localparam int OUT_WIDTH  = 32;
    localparam int NUM_TOKENS = 256;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                          clk;
    logic                          rst_n;
    logic                          in_valid;
    logic signed [IN_WIDTH-1:0]    a [0:N-1];
    logic signed [IN_WIDTH-1:0]    b [0:N-1];
    logic                          out_valid;
    logic signed [OUT_WIDTH-1:0]   dot_product;

    // Golden memory buffers
    logic [7:0]          q_mem [0:(NUM_TOKENS * N)-1];
    logic [7:0]          k_mem [0:(NUM_TOKENS * N)-1];
    logic signed [31:0]  gold_gemm_mem [0:(NUM_TOKENS * NUM_TOKENS)-1];

    int pass_count  = 0;
    int error_count = 0;
    int sample_idx  = 0;

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

    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Monitor and verify outputs
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            logic signed [31:0] exp_val;
            exp_val = gold_gemm_mem[sample_idx];

            if (dot_product === exp_val) begin
                pass_count++;
            end else begin
                if (error_count < 10) begin
                    $display("[ERROR @ Score #%0d]: DUT = %0d | Golden = %0d", 
                             sample_idx, dot_product, exp_val);
                end
                error_count++;
            end

            if (sample_idx < 8 || sample_idx == 255 || sample_idx == 65535) begin
                $display("[SCORE CHECK #%05d]: DUT = %6d | Golden = %6d (%s)",
                         sample_idx, dot_product, exp_val, 
                         (dot_product === exp_val) ? "MATCH" : "MISMATCH");
            end

            sample_idx++;
        end
    end

    initial begin
        $display("=======================================================================");
        $display("   RUNNING UNIT VERIFICATION: PIPELINED ADDER TREE 1A (65,536 SCORES)");
        $display("=======================================================================");

        rst_n    = 0;
        in_valid = 0;
        for (int i = 0; i < N; i++) begin
            a[i] = '0;
            b[i] = '0;
        end

        // Load hex vectors
        $readmemh("q_matrix.hex",       q_mem);
        $readmemh("k_matrix.hex",       k_mem);
        $readmemh("golden_gemm_qk.hex", gold_gemm_mem);

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // Stream 256 rows x 256 columns
        for (int row = 0; row < NUM_TOKENS; row++) begin
            // Latch Query row for this row
            for (int d = 0; d < N; d++) begin
                a[d] <= $signed(q_mem[row * N + d]);
            end

            for (int col = 0; col < NUM_TOKENS; col++) begin
                @(posedge clk);
                in_valid <= 1'b1;
                for (int d = 0; d < N; d++) begin
                    b[d] <= $signed(k_mem[col * N + d]);
                end
            end

            @(posedge clk);
            in_valid <= 1'b0;

            // Wait 5 pipeline flush cycles before changing Query row `a`
            repeat (5) @(posedge clk);
        end

        // Wait a few trailing cycles for final score
        repeat (10) @(posedge clk);

        $display("\n=======================================================================");
        $display("   FINAL REPORT: PIPELINED ADDER TREE 1A VERIFICATION");
        $display("=======================================================================");
        $display("   Total Scores Verified : %0d / 65536", sample_idx);
        $display("   Exact Matches         : %0d", pass_count);
        $display("   Mismatches            : %0d", error_count);

        if (error_count == 0 && sample_idx == 65536 && pass_count == 65536) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH (65,536 / 65,536)! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule
