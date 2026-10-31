`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_stage4_isolated_3B
// Description:  Dedicated Isolated Verification Testbench for Stage 4 (dk=24).
//               Variant 3B: 2D Systolic Array with DSP Packing (16 DSP48E2).
//               128-Bit Native AXI4-Stream Ingress/Egress, 2 Beats per Token.
//               Loads PyTorch Golden Test Vectors (_st4.hex):
//                 - N = 256 tokens, dk = 24
//                 - Ingress: 1,536 beats total (512 beats Q, 512 K, 512 V)
//                 - Egress:  512 beats total (2 beats per row, tlast @ beat 511)
//                 - Total verified bytes: 6,144 bytes
//               Target: AMD Kria KV260 / Icarus Verilog Simulation
// ============================================================================

`include "hw_scale_params.svh"

module tb_stage4_isolated_3B;

    localparam int C_S_AXI_DATA_WIDTH = 32;
    localparam int C_S_AXI_ADDR_WIDTH = 8;
    localparam int C_AXIS_DATA_WIDTH  = 128;
    localparam int ARRAY_SIZE         = 4;
    localparam int TEST_TOKENS        = 256;
    localparam int TEST_HEAD_DIM      = 24;  // STAGE 4: dk = 24
    localparam real CLK_PERIOD        = 5.0; // 200 MHz

    logic                          clk;
    logic                          rst_n;

    // AXI4-Lite
    logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr;
    logic                          s_axi_awvalid;
    logic                          s_axi_awready;
    logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata;
    logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb;
    logic                          s_axi_wvalid;
    logic                          s_axi_wready;
    logic [1:0]                    s_axi_bresp;
    logic                          s_axi_bvalid;
    logic                          s_axi_bready;
    logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr;
    logic                          s_axi_arvalid;
    logic                          s_axi_arready;
    logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata;
    logic [1:0]                    s_axi_rresp;
    logic                          s_axi_rvalid;
    logic                          s_axi_rready;

    // AXI4-Stream Ingress RX (128-bit)
    logic [C_AXIS_DATA_WIDTH-1:0]     s_axis_tdata;
    logic [(C_AXIS_DATA_WIDTH/8)-1:0] s_axis_tkeep;
    logic                             s_axis_tlast;
    logic                             s_axis_tvalid;
    logic                             s_axis_tready;

    // AXI4-Stream Egress TX (128-bit)
    logic [C_AXIS_DATA_WIDTH-1:0]     m_axis_tdata;
    logic [(C_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep;
    logic                             m_axis_tlast;
    logic                             m_axis_tvalid;
    logic                             m_axis_tready;

    // Golden Vector Memories (256 * 24 = 6144 bytes)
    logic [7:0] q_mem            [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];
    logic [7:0] k_mem            [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];
    logic [7:0] v_mem            [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];
    logic [7:0] gold_context_mem [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];

    int pass_count    = 0;
    int error_count   = 0;
    int egress_beats  = 0;
    int egress_bytes  = 0;
    int tlast_errors  = 0;
    int tkeep_errors  = 0;

    // Monitor Output Egress Stream
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            int is_second_beat;
            int row_idx;
            int num_bytes_in_beat;

            is_second_beat = (egress_beats & 1);
            row_idx        = egress_beats >> 1;

            if (!is_second_beat) begin
                num_bytes_in_beat = 16;
                if (m_axis_tkeep !== 16'hFFFF) begin
                    $display("[ERROR @ Beat %0d]: TKEEP mismatch! Expected 16'hFFFF, got 16'h%04h", egress_beats, m_axis_tkeep);
                    tkeep_errors++;
                end
            end else begin
                num_bytes_in_beat = 8;
                if (m_axis_tkeep !== 16'h00FF) begin
                    $display("[ERROR @ Beat %0d]: TKEEP mismatch! Expected 16'h00FF, got 16'h%04h", egress_beats, m_axis_tkeep);
                    tkeep_errors++;
                end
            end

            for (int b = 0; b < num_bytes_in_beat; b++) begin
                logic signed [7:0] dut_ctx;
                logic signed [7:0] exp_ctx;
                int byte_addr;

                byte_addr = row_idx * TEST_HEAD_DIM + (is_second_beat ? (16 + b) : b);
                dut_ctx   = $signed(m_axis_tdata[b*8 +: 8]);

                if (byte_addr < (TEST_TOKENS * TEST_HEAD_DIM)) begin
                    exp_ctx = $signed(gold_context_mem[byte_addr]);

                    if (dut_ctx === exp_ctx) begin
                        pass_count++;
                    end else begin
                        if (error_count < 10) begin
                            $display("[ERROR @ Byte %0d (Row %0d, Dim %0d)]: DUT = %0d | Golden = %0d",
                                     byte_addr, row_idx, (is_second_beat ? (16 + b) : b), dut_ctx, exp_ctx);
                        end
                        error_count++;
                    end

                    if (byte_addr < 24 || byte_addr == 3071 || byte_addr == 6143) begin
                        $display("[STAGE 4 3B CONTEXT #%04d]: DUT = %0d | Golden = %0d (%s)", 
                                 byte_addr, dut_ctx, exp_ctx, (dut_ctx === exp_ctx) ? "MATCH" : "MISMATCH");
                    end

                    egress_bytes++;
                end
            end

            // Check TLAST at the very last beat (beat 511)
            if (egress_beats == (TEST_TOKENS * 2 - 1)) begin
                if (!m_axis_tlast) begin
                    $display("[ERROR]: m_axis_tlast was NOT asserted on final beat %0d!", egress_beats);
                    tlast_errors++;
                end else begin
                    $display("[TB INFO]: m_axis_tlast correctly asserted on beat %0d (Row 255, Beat 1)!", egress_beats);
                end
            end else if (m_axis_tlast) begin
                $display("[ERROR]: m_axis_tlast asserted PREMATURELY on beat %0d!", egress_beats);
                tlast_errors++;
            end

            egress_beats++;
        end
    end

    // Instantiate Variant 3B Top Module
    attention_core_top_3B #(
        .C_S_AXI_DATA_WIDTH (C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH (C_S_AXI_ADDR_WIDTH),
        .C_AXIS_DATA_WIDTH  (C_AXIS_DATA_WIDTH),
        .ARRAY_SIZE         (ARRAY_SIZE),
        .MAX_TOKENS         (TEST_TOKENS),
        .MAX_HEAD_DIM       (32)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axi_awaddr  (s_axi_awaddr),
        .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata   (s_axi_wdata),
        .s_axi_wstrb   (s_axi_wstrb),
        .s_axi_wvalid  (s_axi_wvalid),
        .s_axi_wready  (s_axi_wready),
        .s_axi_bresp   (s_axi_bresp),
        .s_axi_bvalid  (s_axi_bvalid),
        .s_axi_bready  (s_axi_bready),
        .s_axi_araddr  (s_axi_araddr),
        .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),
        .s_axi_rdata   (s_axi_rdata),
        .s_axi_rresp   (s_axi_rresp),
        .s_axi_rvalid  (s_axi_rvalid),
        .s_axi_rready  (s_axi_rready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tkeep  (s_axis_tkeep),
        .s_axis_tlast  (s_axis_tlast),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tkeep  (m_axis_tkeep),
        .m_axis_tlast  (m_axis_tlast),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready)
    );

    // 200 MHz Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Stream 128-bit (2 beats per token for dk=24): 512 beats total per matrix
    task stream_q;
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {q_mem[i*TEST_HEAD_DIM + 15],
                              q_mem[i*TEST_HEAD_DIM + 14],
                              q_mem[i*TEST_HEAD_DIM + 13],
                              q_mem[i*TEST_HEAD_DIM + 12],
                              q_mem[i*TEST_HEAD_DIM + 11],
                              q_mem[i*TEST_HEAD_DIM + 10],
                              q_mem[i*TEST_HEAD_DIM + 9],
                              q_mem[i*TEST_HEAD_DIM + 8],
                              q_mem[i*TEST_HEAD_DIM + 7],
                              q_mem[i*TEST_HEAD_DIM + 6],
                              q_mem[i*TEST_HEAD_DIM + 5],
                              q_mem[i*TEST_HEAD_DIM + 4],
                              q_mem[i*TEST_HEAD_DIM + 3],
                              q_mem[i*TEST_HEAD_DIM + 2],
                              q_mem[i*TEST_HEAD_DIM + 1],
                              q_mem[i*TEST_HEAD_DIM + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h00FF;
            s_axis_tdata  <= {64'd0,
                              q_mem[i*TEST_HEAD_DIM + 23],
                              q_mem[i*TEST_HEAD_DIM + 22],
                              q_mem[i*TEST_HEAD_DIM + 21],
                              q_mem[i*TEST_HEAD_DIM + 20],
                              q_mem[i*TEST_HEAD_DIM + 19],
                              q_mem[i*TEST_HEAD_DIM + 18],
                              q_mem[i*TEST_HEAD_DIM + 17],
                              q_mem[i*TEST_HEAD_DIM + 16]};
            s_axis_tlast  <= 1'b0;
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    task stream_k;
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {k_mem[i*TEST_HEAD_DIM + 15],
                              k_mem[i*TEST_HEAD_DIM + 14],
                              k_mem[i*TEST_HEAD_DIM + 13],
                              k_mem[i*TEST_HEAD_DIM + 12],
                              k_mem[i*TEST_HEAD_DIM + 11],
                              k_mem[i*TEST_HEAD_DIM + 10],
                              k_mem[i*TEST_HEAD_DIM + 9],
                              k_mem[i*TEST_HEAD_DIM + 8],
                              k_mem[i*TEST_HEAD_DIM + 7],
                              k_mem[i*TEST_HEAD_DIM + 6],
                              k_mem[i*TEST_HEAD_DIM + 5],
                              k_mem[i*TEST_HEAD_DIM + 4],
                              k_mem[i*TEST_HEAD_DIM + 3],
                              k_mem[i*TEST_HEAD_DIM + 2],
                              k_mem[i*TEST_HEAD_DIM + 1],
                              k_mem[i*TEST_HEAD_DIM + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h00FF;
            s_axis_tdata  <= {64'd0,
                              k_mem[i*TEST_HEAD_DIM + 23],
                              k_mem[i*TEST_HEAD_DIM + 22],
                              k_mem[i*TEST_HEAD_DIM + 21],
                              k_mem[i*TEST_HEAD_DIM + 20],
                              k_mem[i*TEST_HEAD_DIM + 19],
                              k_mem[i*TEST_HEAD_DIM + 18],
                              k_mem[i*TEST_HEAD_DIM + 17],
                              k_mem[i*TEST_HEAD_DIM + 16]};
            s_axis_tlast  <= 1'b0;
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    task stream_v;
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {v_mem[i*TEST_HEAD_DIM + 15],
                              v_mem[i*TEST_HEAD_DIM + 14],
                              v_mem[i*TEST_HEAD_DIM + 13],
                              v_mem[i*TEST_HEAD_DIM + 12],
                              v_mem[i*TEST_HEAD_DIM + 11],
                              v_mem[i*TEST_HEAD_DIM + 10],
                              v_mem[i*TEST_HEAD_DIM + 9],
                              v_mem[i*TEST_HEAD_DIM + 8],
                              v_mem[i*TEST_HEAD_DIM + 7],
                              v_mem[i*TEST_HEAD_DIM + 6],
                              v_mem[i*TEST_HEAD_DIM + 5],
                              v_mem[i*TEST_HEAD_DIM + 4],
                              v_mem[i*TEST_HEAD_DIM + 3],
                              v_mem[i*TEST_HEAD_DIM + 2],
                              v_mem[i*TEST_HEAD_DIM + 1],
                              v_mem[i*TEST_HEAD_DIM + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h00FF;
            s_axis_tdata  <= {64'd0,
                              v_mem[i*TEST_HEAD_DIM + 23],
                              v_mem[i*TEST_HEAD_DIM + 22],
                              v_mem[i*TEST_HEAD_DIM + 21],
                              v_mem[i*TEST_HEAD_DIM + 20],
                              v_mem[i*TEST_HEAD_DIM + 19],
                              v_mem[i*TEST_HEAD_DIM + 18],
                              v_mem[i*TEST_HEAD_DIM + 17],
                              v_mem[i*TEST_HEAD_DIM + 16]};
            s_axis_tlast  <= (i == TEST_TOKENS - 1);
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    // Helper task to write AXI-Lite register
    task automatic axi_write_csr(input [7:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;

            @(posedge clk);
            while (!s_axi_awready || !s_axi_wready) @(posedge clk);

            @(posedge clk);
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;

            while (!s_axi_bvalid) @(posedge clk);
            @(posedge clk);
            s_axi_bready  <= 1'b0;
        end
    endtask

    int timeout_cycles;

    initial begin
        $display("=======================================================================");
        $display("   ISOLATED VERIFICATION TESTBENCH: STAGE 4 (dk=24, M=18810, E=22)");
        $display("   VARIANT 3B: 2D SYSTOLIC MAC ARRAY WITH DSP PACKING (16 DSP48E2)");
        $display("=======================================================================");

        rst_n         = 1'b0;
        s_axi_awaddr  = '0;
        s_axi_awvalid = 1'b0;
        s_axi_wdata   = '0;
        s_axi_wstrb   = 4'hF;
        s_axi_wvalid  = 1'b0;
        s_axi_bready  = 1'b0;
        s_axi_araddr  = '0;
        s_axi_arvalid = 1'b0;
        s_axi_rready  = 1'b0;

        s_axis_tdata  = '0;
        s_axis_tkeep  = 16'hFFFF;
        s_axis_tlast  = 1'b0;
        s_axis_tvalid = 1'b0;
        m_axis_tready = 1'b1;

        // Step 1: Load Golden Test Vectors for Stage 4
        $display("[TB INFO]: Step 1 - Loading Stage 4 Golden Test Vectors from .hex files...");
        $readmemh("q_matrix_st4.hex",           q_mem);
        $readmemh("k_matrix_st4.hex",           k_mem);
        $readmemh("v_matrix_st4.hex",           v_mem);
        $readmemh("golden_context_out_st4.hex", gold_context_mem);
        $display("[TB INFO]: Loaded %0d tokens (dk=%0d) for Q, K, V (Expected Context: 6,144 bytes).", 
                 TEST_TOKENS, TEST_HEAD_DIM);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset Released.");

        // Step 2: Configure CSR Registers via AXI4-Lite for Stage 4
        $display("[TB INFO]: Step 2 - Configuring CSR for Stage 4 (N=256, dk=24, H=1, M=18810, E=22)...");
        axi_write_csr(8'h10, 32'd256);                   // CSR_SEQ_LEN = 256
        axi_write_csr(8'h14, 32'd24);                    // CSR_HEAD_DIM = 24
        axi_write_csr(8'h18, 32'd1);                     // CSR_NUM_HEADS = 1
        axi_write_csr(8'h20, 32'd18810);                 // CSR_SCALE_M = 18810
        axi_write_csr(8'h24, 32'd22);                    // CSR_SCALE_E = 22

        // Step 3: Stream Token Vectors into AXI-Stream Ingress (128-bit: 512 beats/mat = 1,536 beats total)
        $display("[TB INFO]: Step 3 - Streaming Q, K, V matrices via 128-bit AXI-Stream (1,536 beats total)...");
        stream_q();
        stream_k();
        stream_v();
        $display("[TB INFO]: Finished Streaming 1,536 beats into Ingress Storage.");

        // Step 4: Trigger START pulse via AXI-Lite
        $display("[TB INFO]: Step 4 - Triggering START Pulse via AXI-Lite (0x00 <= 1)...");
        axi_write_csr(8'h00, 32'd1);

        // Step 5: Wait for Computation Pipeline Completion
        $display("[TB INFO]: Step 5 - Waiting for Core Pipeline Execution (256 rows, 512 beats)...");
        timeout_cycles = 0;
        while (egress_beats < (TEST_TOKENS * 2) && timeout_cycles < 500000) begin
            @(posedge clk);
            timeout_cycles++;
            if (timeout_cycles % 50000 == 0) begin
                $display("[TB INFO]: Running... Cycle %0d, Egress Beats: %0d / 512, Egress Bytes: %0d / 6144", 
                         timeout_cycles, egress_beats, egress_bytes);
            end
        end

        // Wait a few trailing cycles
        repeat (20) @(posedge clk);

        // Step 6: Final Verification Report
        $display("\n=======================================================================");
        $display("   FINAL VERIFICATION REPORT: STAGE 4 ISOLATED (3B: 2D SYSTOLIC MAC)");
        $display("=======================================================================");
        $display("   Total Egress Beats Processed   : %0d / 512", egress_beats);
        $display("   Total Context Bytes Verified   : %0d / 6144", egress_bytes);
        $display("   Exact Matched Samples (0-LSB)  : %0d", pass_count);
        $display("   Mismatched Samples             : %0d", error_count);
        $display("   TKEEP Formatting Errors        : %0d", tkeep_errors);
        $display("   TLAST Assertion Errors         : %0d", tlast_errors);
        $display("   Total Execution Clock Cycles   : %0d", timeout_cycles);

        if (error_count == 0 && tlast_errors == 0 && tkeep_errors == 0 && egress_bytes == (TEST_TOKENS * TEST_HEAD_DIM)) begin
            $display("   >>> STATUS: STAGE 4 ISOLATED 3B 100%% BIT-EXACT MATCH ACHIEVED! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: STAGE 4 3B VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end


    // Setup waveform dumping for tools like GTKWave
    initial begin
        $dumpfile("tb_stage4_isolated_3B.vcd");
        $dumpvars(0, tb_stage4_isolated_3B);
    end

endmodule
