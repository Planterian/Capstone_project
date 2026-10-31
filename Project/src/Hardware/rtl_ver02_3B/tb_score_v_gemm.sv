`timescale 1ns / 1ps
// ============================================================================
// Testbench:    tb_score_v_gemm
// Description:  Bit-exact verification of score_v_gemm against
//               PyTorch golden vectors (golden_context_out.hex).
//               Validates all 256 rows (512 beats, 4,096 bytes) and tlast on beat 511.
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_score_v_gemm;

    parameter int MAX_TOKENS      = 256;
    parameter int MAX_HEAD_DIM    = 32;
    parameter int HEAD_DIM        = 16;
    parameter int PROB_WIDTH      = 16;
    parameter int AXIS_DATA_WIDTH = 128;

    localparam int BYTES_PER_BEAT = AXIS_DATA_WIDTH / 8;
    localparam int BEATS_PER_ROW  = (AXIS_DATA_WIDTH == 128) ? ((HEAD_DIM <= 16) ? 1 : 2) : ((HEAD_DIM + 7) / 8);
    localparam int TOTAL_BEATS    = MAX_TOKENS * BEATS_PER_ROW;

    logic clk;
    logic rst_n;

    logic                          start_row;
    logic                          prob_valid;
    logic [PROB_WIDTH-1:0]         prob_data;
    logic                          prob_last;

    logic signed [7:0]             v_vec [0:MAX_HEAD_DIM-1];
    logic [15:0]                   head_dim_sig;

    logic                          m_axis_tvalid;
    logic [AXIS_DATA_WIDTH-1:0]    m_axis_tdata;
    logic                          m_axis_tlast;
    logic                          m_axis_tready;

    logic [8:0]                    row_token_cnt;
    logic                          row_done;

    // DUT Instantiation
    score_v_gemm #(
        .MAX_TOKENS(MAX_TOKENS),
        .MAX_HEAD_DIM(MAX_HEAD_DIM),
        .PROB_WIDTH(PROB_WIDTH),
        .AXIS_DATA_WIDTH(AXIS_DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_row(start_row),
        .prob_valid(prob_valid),
        .prob_data(prob_data),
        .prob_last(prob_last),
        .v_vec(v_vec),
        .head_dim(head_dim_sig),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tready(m_axis_tready),
        .row_token_cnt(row_token_cnt),
        .row_done(row_done)
    );

    // Memories for stimuli and golden data
    logic [7:0]  v_raw_mem     [0:4095];
    logic [15:0] prob_raw_mem  [0:65535];
    logic [7:0]  golden_ctx_mem[0:4095];

    // Clock generation: 200 MHz (5.0 ns period)
    always #2.5 clk = ~clk;

    int beat_count;
    int error_count;
    int match_count;
    int tlast_errors;

    // Output monitor
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            int current_row;
            int beat_in_row;
            current_row = beat_count / BEATS_PER_ROW;
            beat_in_row = beat_count % BEATS_PER_ROW;

            for (int b = 0; b < BYTES_PER_BEAT; b++) begin
                int dim_idx;
                dim_idx = beat_in_row * BYTES_PER_BEAT + b;
                if (dim_idx < HEAD_DIM) begin
                    int byte_idx;
                    logic signed [7:0] actual_byte;
                    logic signed [7:0] expected_byte;

                    byte_idx = current_row * HEAD_DIM + dim_idx;
                    actual_byte = $signed(m_axis_tdata[b*8 +: 8]);
                    expected_byte = $signed(golden_ctx_mem[byte_idx]);

                    if (actual_byte !== expected_byte) begin
                        if (error_count < 10) begin
                            $display("[ERROR] Beat %0d Byte %0d (Idx %0d): Got 0x%02h (%0d), Expected 0x%02h (%0d)",
                                     beat_count, b, byte_idx, actual_byte, actual_byte, expected_byte, expected_byte);
                        end
                        error_count++;
                    end else begin
                        match_count++;
                    end
                end
            end

            // Check tlast
            if (beat_count == TOTAL_BEATS - 1) begin
                if (!m_axis_tlast) begin
                    $display("[ERROR] Beat %0d: m_axis_tlast is 0, expected 1!", beat_count);
                    tlast_errors++;
                end
            end else begin
                if (m_axis_tlast) begin
                    $display("[ERROR] Beat %0d: m_axis_tlast is 1 unexpectedly!", beat_count);
                    tlast_errors++;
                end
            end

            beat_count++;
        end
    end

    initial begin
        clk           = 0;
        rst_n         = 0;
        start_row     = 0;
        prob_valid    = 0;
        prob_data     = 0;
        prob_last     = 0;
        head_dim_sig  = HEAD_DIM;
        m_axis_tready = 1;
        beat_count    = 0;
        error_count   = 0;
        match_count   = 0;
        tlast_errors  = 0;

        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
            v_vec[d] = 8'sd0;
        end

        // Load files
        $readmemh("v_matrix.hex", v_raw_mem);
        $readmemh("golden_softmax_prob.hex", prob_raw_mem);
        $readmemh("golden_context_out.hex", golden_ctx_mem);

        #20;
        rst_n = 1;
        #10;

        $display("=== STARTING SCORE_V_GEMM VERIFICATION (%0d ROWS, %0d BEATS, %0d-BIT BUS) ===",
                 MAX_TOKENS, TOTAL_BEATS, AXIS_DATA_WIDTH);

        // Iterate through all 256 rows
        for (int r = 0; r < MAX_TOKENS; r++) begin
            @(posedge clk);
            start_row <= 1'b1;
            @(posedge clk);
            start_row <= 1'b0;

            // Stream 256 probability tokens
            for (int k = 0; k < MAX_TOKENS; k++) begin
                prob_valid <= 1'b1;
                prob_data  <= prob_raw_mem[r * MAX_TOKENS + k];
                prob_last  <= (k == MAX_TOKENS - 1);

                for (int d = 0; d < HEAD_DIM; d++) begin
                    v_vec[d] <= $signed(v_raw_mem[k * HEAD_DIM + d]);
                end

                @(posedge clk);
            end

            prob_valid <= 1'b0;
            prob_last  <= 1'b0;

            // Wait for row egress to complete
            while (!row_done) begin
                @(posedge clk);
            end
        end

        // Wait a few cycles
        #50;

        $display("=== SCORE_V_GEMM VERIFICATION REPORT ===");
        $display("Total Beats Processed:  %0d (Expected: %0d)", beat_count, TOTAL_BEATS);
        $display("Total Bytes Matched:    %0d / 4096", match_count);
        $display("Total Byte Errors:      %0d", error_count);
        $display("Total TLAST Errors:     %0d", tlast_errors);

        if (error_count == 0 && tlast_errors == 0 && beat_count == TOTAL_BEATS && match_count == 4096) begin
            $display(">>> SUCCESS: score_v_gemm PASSED 100%%%% BIT-EXACT MATCH! <<<");
        end else begin
            $display(">>> FAILED: Discrepancies detected in score_v_gemm! <<<");
        end

        $finish;
    end


    // Setup waveform dumping for tools like GTKWave
    initial begin
        $dumpfile("tb_score_v_gemm.vcd");
        $dumpvars(0, tb_score_v_gemm);
    end

endmodule
