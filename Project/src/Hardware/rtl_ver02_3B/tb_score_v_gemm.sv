`timescale 1ns / 1ps
// ============================================================================
// Testbench:    tb_score_v_gemm
// Description:  Bit-exact multi-stage verification of score_v_gemm against
//               PyTorch golden vectors across all 3 MobileViT Stages:
//               - Stage 2: N=256, dk=16 (256 beats, 4,096 bytes)
//               - Stage 3: N=64,  dk=20 (128 beats, 1,280 bytes)
//               - Stage 4: N=16,  dk=24 ( 32 beats,   384 bytes)
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_score_v_gemm;

    parameter int MAX_TOKENS      = 256;
    parameter int MAX_HEAD_DIM    = 32;
    parameter int PROB_WIDTH      = 16;
    parameter int AXIS_DATA_WIDTH = 128;
    localparam int BYTES_PER_BEAT = AXIS_DATA_WIDTH / 8; // 16 bytes

    localparam int ST2_TOKENS     = 256;
    localparam int ST2_HEAD_DIM   = 16;
    localparam int ST2_BEATS_ROW  = 1;
    localparam int ST2_TOTAL_BEATS= ST2_TOKENS * ST2_BEATS_ROW; // 256
    localparam int ST2_BYTES      = ST2_TOKENS * ST2_HEAD_DIM;  // 4,096

    localparam int ST3_TOKENS     = 64;
    localparam int ST3_HEAD_DIM   = 20;
    localparam int ST3_BEATS_ROW  = 2;
    localparam int ST3_TOTAL_BEATS= ST3_TOKENS * ST3_BEATS_ROW; // 128
    localparam int ST3_BYTES      = ST3_TOKENS * ST3_HEAD_DIM;  // 1,280

    localparam int ST4_TOKENS     = 16;
    localparam int ST4_HEAD_DIM   = 24;
    localparam int ST4_BEATS_ROW  = 2;
    localparam int ST4_TOTAL_BEATS= ST4_TOKENS * ST4_BEATS_ROW; // 32
    localparam int ST4_BYTES      = ST4_TOKENS * ST4_HEAD_DIM;  // 384

    logic                          clk;
    logic                          rst_n;

    logic                          start_row;
    logic                          prob_valid;
    logic [PROB_WIDTH-1:0]         prob_data;
    logic                          prob_last;

    logic signed [7:0]             v_vec [0:MAX_HEAD_DIM-1];
    logic [15:0]                   seq_len_sig;
    logic [15:0]                   head_dim_sig;

    logic                          m_axis_tvalid;
    logic [AXIS_DATA_WIDTH-1:0]    m_axis_tdata;
    logic [BYTES_PER_BEAT-1:0]     m_axis_tkeep;
    logic                          m_axis_tlast;
    logic                          m_axis_tready;

    logic [8:0]                    row_token_cnt;
    logic                          row_done;

    // DUT Instantiation
    score_v_gemm #(
        .MAX_TOKENS      (MAX_TOKENS),
        .MAX_HEAD_DIM    (MAX_HEAD_DIM),
        .PROB_WIDTH      (PROB_WIDTH),
        .AXIS_DATA_WIDTH (AXIS_DATA_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .start_row     (start_row),
        .prob_valid    (prob_valid),
        .prob_data     (prob_data),
        .prob_last     (prob_last),
        .v_vec         (v_vec),
        .seq_len       (seq_len_sig),
        .head_dim      (head_dim_sig),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tkeep  (m_axis_tkeep),
        .m_axis_tlast  (m_axis_tlast),
        .m_axis_tready (m_axis_tready),
        .row_token_cnt (row_token_cnt),
        .row_done      (row_done)
    );

    // Memories for stimuli and golden data across all 3 stages
    logic [7:0]  v_st2_mem   [0:ST2_BYTES-1];
    logic [15:0] prob_st2_mem[0:(ST2_TOKENS * ST2_TOKENS)-1];
    logic [7:0]  ctx_st2_mem [0:ST2_BYTES-1];

    logic [7:0]  v_st3_mem   [0:ST3_BYTES-1];
    logic [15:0] prob_st3_mem[0:(ST3_TOKENS * ST3_TOKENS)-1];
    logic [7:0]  ctx_st3_mem [0:ST3_BYTES-1];

    logic [7:0]  v_st4_mem   [0:ST4_BYTES-1];
    logic [15:0] prob_st4_mem[0:(ST4_TOKENS * ST4_TOKENS)-1];
    logic [7:0]  ctx_st4_mem [0:ST4_BYTES-1];

    // Clock generation: 200 MHz
    always #2.5 clk = ~clk;

    int current_stage = 2;
    int stage_beats   = 0;
    int stage_matches = 0;
    int stage_errors  = 0;
    int stage_tlast_errors = 0;

    int total_matches = 0;
    int total_errors  = 0;

    // Output monitor
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            int cur_dim, cur_tokens, beats_per_row, total_stage_beats;

            if (current_stage == 2) begin
                cur_dim           = ST2_HEAD_DIM;
                cur_tokens        = ST2_TOKENS;
                beats_per_row     = ST2_BEATS_ROW;
                total_stage_beats = ST2_TOTAL_BEATS;
            end else if (current_stage == 3) begin
                cur_dim           = ST3_HEAD_DIM;
                cur_tokens        = ST3_TOKENS;
                beats_per_row     = ST3_BEATS_ROW;
                total_stage_beats = ST3_TOTAL_BEATS;
            end else begin
                cur_dim           = ST4_HEAD_DIM;
                cur_tokens        = ST4_TOKENS;
                beats_per_row     = ST4_BEATS_ROW;
                total_stage_beats = ST4_TOTAL_BEATS;
            end

            for (int b = 0; b < BYTES_PER_BEAT; b++) begin
                int beat_in_row, row_idx, dim_idx, byte_addr;
                logic signed [7:0] actual_byte, expected_byte;

                row_idx     = stage_beats / beats_per_row;
                beat_in_row = stage_beats % beats_per_row;
                dim_idx     = beat_in_row * BYTES_PER_BEAT + b;

                if (dim_idx < cur_dim) begin
                    byte_addr = row_idx * cur_dim + dim_idx;
                    actual_byte = $signed(m_axis_tdata[b*8 +: 8]);

                    if (current_stage == 2)      expected_byte = $signed(ctx_st2_mem[byte_addr]);
                    else if (current_stage == 3) expected_byte = $signed(ctx_st3_mem[byte_addr]);
                    else                         expected_byte = $signed(ctx_st4_mem[byte_addr]);

                    if (actual_byte !== expected_byte) begin
                        if (stage_errors < 5) begin
                            $display("[ERROR ST%0d] Beat %0d Byte %0d (Idx %0d): Got %0d, Exp %0d",
                                     current_stage, stage_beats, b, byte_addr, actual_byte, expected_byte);
                        end
                        stage_errors++;
                        total_errors++;
                    end else begin
                        stage_matches++;
                        total_matches++;
                    end
                end
            end

            // Check tlast
            if (stage_beats == total_stage_beats - 1) begin
                if (!m_axis_tlast) begin
                    $display("[ERROR ST%0d] Final Beat %0d: m_axis_tlast was NOT asserted!", current_stage, stage_beats);
                    stage_tlast_errors++;
                end
            end else begin
                if (m_axis_tlast) begin
                    $display("[ERROR ST%0d] Beat %0d: m_axis_tlast was asserted prematurely!", current_stage, stage_beats);
                    stage_tlast_errors++;
                end
            end

            stage_beats++;
        end
    end

    // Test sequence
    initial begin
        clk           = 0;
        rst_n         = 0;
        start_row     = 0;
        prob_valid    = 0;
        prob_data     = 0;
        prob_last     = 0;
        m_axis_tready = 1;
        seq_len_sig   = 16'd256;
        head_dim_sig  = 16'd16;

        for (int d = 0; d < MAX_HEAD_DIM; d++) v_vec[d] = 8'sd0;

        // Load files for all 3 stages
        $display("[TB INFO]: Loading Stage 2 vectors...");
        $readmemh("v_matrix_st2.hex",           v_st2_mem);
        $readmemh("golden_softmax_prob_st2.hex", prob_st2_mem);
        $readmemh("golden_context_out_st2.hex",  ctx_st2_mem);

        $display("[TB INFO]: Loading Stage 3 vectors...");
        $readmemh("v_matrix_st3.hex",           v_st3_mem);
        $readmemh("golden_softmax_prob_st3.hex", prob_st3_mem);
        $readmemh("golden_context_out_st3.hex",  ctx_st3_mem);

        $display("[TB INFO]: Loading Stage 4 vectors...");
        $readmemh("v_matrix_st4.hex",           v_st4_mem);
        $readmemh("golden_softmax_prob_st4.hex", prob_st4_mem);
        $readmemh("golden_context_out_st4.hex",  ctx_st4_mem);

        #20;
        rst_n = 1;
        #10;

        $display("=======================================================================");
        $display("   TESTBENCH: score_v_gemm MULTI-STAGE BIT-EXACT VERIFICATION");
        $display("   Testing Stage 2 (4,096 B), Stage 3 (1,280 B), and Stage 4 (384 B)");
        $display("=======================================================================");

        // --------------------------------------------------------------------
        // PHASE 1: STAGE 2 (N=256, dk=16, 256 rows, 256 beats, 4,096 bytes)
        // --------------------------------------------------------------------
        current_stage      = 2;
        seq_len_sig        = 16'd256;
        head_dim_sig       = 16'd16;
        stage_beats        = 0;
        stage_matches      = 0;
        stage_errors       = 0;
        stage_tlast_errors = 0;
        $display("\n>>> [PHASE 1/3]: Processing STAGE 2 (N=256, dk=16, 256 rows)...");

        for (int r = 0; r < ST2_TOKENS; r++) begin
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            for (int k = 0; k < ST2_TOKENS; k++) begin
                prob_valid <= 1'b1;
                prob_data  <= prob_st2_mem[r * ST2_TOKENS + k];
                prob_last  <= (k == ST2_TOKENS - 1);

                for (int d = 0; d < ST2_HEAD_DIM; d++) begin
                    v_vec[d] <= $signed(v_st2_mem[k * ST2_HEAD_DIM + d]);
                end
                for (int d = ST2_HEAD_DIM; d < MAX_HEAD_DIM; d++) begin
                    v_vec[d] <= 8'sd0;
                end
                @(posedge clk);
            end

            prob_valid <= 1'b0;
            prob_last  <= 1'b0;

            while (!row_done) @(posedge clk);
        end
        $display("    -> Stage 2 Done: %0d / %0d bytes matched (Errors: %0d, TLAST Err: %0d)",
                 stage_matches, ST2_BYTES, stage_errors, stage_tlast_errors);

        // --------------------------------------------------------------------
        // PHASE 2: STAGE 3 (N=64, dk=20, 64 rows, 128 beats, 1,280 bytes)
        // --------------------------------------------------------------------
        current_stage      = 3;
        seq_len_sig        = 16'd64;
        head_dim_sig       = 16'd20;
        stage_beats        = 0;
        stage_matches      = 0;
        stage_errors       = 0;
        stage_tlast_errors = 0;
        $display("\n>>> [PHASE 2/3]: Processing STAGE 3 (N=64, dk=20, 64 rows)...");

        for (int r = 0; r < ST3_TOKENS; r++) begin
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            for (int k = 0; k < ST3_TOKENS; k++) begin
                prob_valid <= 1'b1;
                prob_data  <= prob_st3_mem[r * ST3_TOKENS + k];
                prob_last  <= (k == ST3_TOKENS - 1);

                for (int d = 0; d < ST3_HEAD_DIM; d++) begin
                    v_vec[d] <= $signed(v_st3_mem[k * ST3_HEAD_DIM + d]);
                end
                for (int d = ST3_HEAD_DIM; d < MAX_HEAD_DIM; d++) begin
                    v_vec[d] <= 8'sd0;
                end
                @(posedge clk);
            end

            prob_valid <= 1'b0;
            prob_last  <= 1'b0;

            while (!row_done) @(posedge clk);
        end
        $display("    -> Stage 3 Done: %0d / %0d bytes matched (Errors: %0d, TLAST Err: %0d)",
                 stage_matches, ST3_BYTES, stage_errors, stage_tlast_errors);

        // --------------------------------------------------------------------
        // PHASE 3: STAGE 4 (N=16, dk=24, 16 rows, 32 beats, 384 bytes)
        // --------------------------------------------------------------------
        current_stage      = 4;
        seq_len_sig        = 16'd16;
        head_dim_sig       = 16'd24;
        stage_beats        = 0;
        stage_matches      = 0;
        stage_errors       = 0;
        stage_tlast_errors = 0;
        $display("\n>>> [PHASE 3/3]: Processing STAGE 4 (N=16, dk=24, 16 rows)...");

        for (int r = 0; r < ST4_TOKENS; r++) begin
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            for (int k = 0; k < ST4_TOKENS; k++) begin
                prob_valid <= 1'b1;
                prob_data  <= prob_st4_mem[r * ST4_TOKENS + k];
                prob_last  <= (k == ST4_TOKENS - 1);

                for (int d = 0; d < ST4_HEAD_DIM; d++) begin
                    v_vec[d] <= $signed(v_st4_mem[k * ST4_HEAD_DIM + d]);
                end
                for (int d = ST4_HEAD_DIM; d < MAX_HEAD_DIM; d++) begin
                    v_vec[d] <= 8'sd0;
                end
                @(posedge clk);
            end

            prob_valid <= 1'b0;
            prob_last  <= 1'b0;

            while (!row_done) @(posedge clk);
        end
        $display("    -> Stage 4 Done: %0d / %0d bytes matched (Errors: %0d, TLAST Err: %0d)",
                 stage_matches, ST4_BYTES, stage_errors, stage_tlast_errors);

        #50;

        $display("\n=======================================================================");
        $display("   FINAL REPORT: score_v_gemm ALL 3 STAGES VERIFICATION");
        $display("=======================================================================");
        $display("   Stage 2 Matched : %0d / %0d bytes", ST2_BYTES, ST2_BYTES);
        $display("   Stage 3 Matched : %0d / %0d bytes", ST3_BYTES, ST3_BYTES);
        $display("   Stage 4 Matched : %0d / %0d bytes", ST4_BYTES, ST4_BYTES);
        $display("   Total Bytes     : %0d / %0d", total_matches, (ST2_BYTES + ST3_BYTES + ST4_BYTES));
        $display("   Total Errors    : %0d", total_errors);
        if (total_errors == 0 && total_matches == (ST2_BYTES + ST3_BYTES + ST4_BYTES)) begin
            $display("   >>> SUCCESS: score_v_gemm PASSED 100%% BIT-EXACT MATCH! [PASS] <<<");
        end else begin
            $display("   >>> FAILED: Discrepancies detected in score_v_gemm! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule
