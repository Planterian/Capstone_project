`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_attention_core_1A
// Description:  Full Verification Testbench for Attention Core Top (Variant 1A).
//               128-Bit Native AXI4-Stream Ingress/Egress, N=32 Reduction Tree.
//               Loads PyTorch Golden Test Vectors (.hex), configures CSR registers,
//               streams inputs (Q, K, V = 768 beats), and verifies PL outputs against
//               PyTorch golden reference data (golden_context_out.hex).
// ============================================================================

`include "hw_scale_params.svh"

module tb_attention_core_1A;

    localparam int C_S_AXI_DATA_WIDTH = 32;
    localparam int C_S_AXI_ADDR_WIDTH = 8;
    localparam int C_AXIS_DATA_WIDTH  = 128; // 16 INT8 tokens/beat
    localparam int ARRAY_SIZE         = 4;
    localparam int TEST_TOKENS        = 256;
    localparam int TEST_HEAD_DIM      = 16;
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

    // Golden Vector Memories
    logic [7:0] q_mem            [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];
    logic [7:0] k_mem            [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];
    logic [7:0] v_mem            [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];
    logic [7:0] gold_context_mem [0:(TEST_TOKENS * TEST_HEAD_DIM)-1];

    int pass_count   = 0;
    int error_count  = 0;
    int egress_cnt   = 0;
    int tlast_errors = 0;

    // Monitor Output Egress Stream (16 bytes per beat on 128-bit bus)
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            for (int b = 0; b < 16; b++) begin
                logic signed [7:0] dut_ctx;
                logic signed [7:0] exp_ctx;

                dut_ctx = $signed(m_axis_tdata[b*8 +: 8]);
                if (egress_cnt < (TEST_TOKENS * TEST_HEAD_DIM)) begin
                    exp_ctx = $signed(gold_context_mem[egress_cnt]);

                    if (dut_ctx === exp_ctx) begin
                        pass_count++;
                    end else begin
                        if (error_count < 10) begin
                            $display("[ERROR @ Egress %0d]: DUT = 0x%02h (%0d) | Golden = 0x%02h (%0d)",
                                     egress_cnt, dut_ctx, dut_ctx, exp_ctx, exp_ctx);
                        end
                        error_count++;
                    end

                    if (egress_cnt < 16 || egress_cnt == 511 || egress_cnt == 4095) begin
                        $display("[CONTEXT EGRESS #%04d]: DUT = %0d | Golden = %0d (%s)", 
                                 egress_cnt, dut_ctx, exp_ctx, (dut_ctx === exp_ctx) ? "MATCH" : "MISMATCH");
                    end

                    egress_cnt++;
                end
            end

            if (egress_cnt == (TEST_TOKENS * TEST_HEAD_DIM) && !m_axis_tlast) begin
                $display("[ERROR]: m_axis_tlast was NOT asserted on the final beat!");
                tlast_errors++;
            end else if (m_axis_tlast && egress_cnt == (TEST_TOKENS * TEST_HEAD_DIM)) begin
                $display("[TB INFO]: m_axis_tlast correctly asserted on beat %0d!", (egress_cnt/16)-1);
            end
        end
    end

    // Instantiate Variant 1A Top Module
    attention_core_top_1A #(
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

    // Stream 128-bit (16 bytes per beat)
    task stream_q;
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
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
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
    endtask

    task stream_k;
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
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
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
    endtask

    task stream_v;
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
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
        $display("   STARTING INTEGRATED ATTENTION CORE VERIFICATION (VARIANT 1A)");
        $display("   128-BIT NATIVE AXI-STREAM + N=32 PIPELINED ADDER TREE (0 DSP48E2)");
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

        // Step 1: Load Golden Test Vectors
        $display("[TB INFO]: Loading Golden Test Vectors from .hex files...");
        $readmemh("q_matrix.hex",           q_mem);
        $readmemh("k_matrix.hex",           k_mem);
        $readmemh("v_matrix.hex",           v_mem);
        $readmemh("golden_context_out.hex", gold_context_mem);
        $display("[TB INFO]: Loaded %0d Q tokens, %0d K tokens, %0d V tokens.", 
                 TEST_TOKENS, TEST_TOKENS, TEST_TOKENS);

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset Released.");

        // Step 2: Configure CSR Registers via AXI4-Lite
        $display("[TB INFO]: Step 2 - Configuring CSR (N=256, dk=16, H=1, M=27640, E=22)...");
        axi_write_csr(8'h10, 32'd256);                   // CSR_SEQ_LEN = 256
        axi_write_csr(8'h14, 32'd16);                    // CSR_HEAD_DIM = 16
        axi_write_csr(8'h18, 32'd1);                     // CSR_NUM_HEADS = 1 (Single Head Test)
        axi_write_csr(8'h20, 32'd27640);                 // CSR_SCALE_M = 27640
        axi_write_csr(8'h24, 32'd22);                    // CSR_SCALE_E = 22

        // Step 3: Stream Token Vectors into AXI-Stream Ingress (128-bit: 256 beats per matrix = 768 beats total!)
        $display("[TB INFO]: Step 3 - Streaming Q, K, V matrices via 128-bit AXI-Stream (768 beats)...");
        stream_q();
        stream_k();
        stream_v();
        $display("[TB INFO]: Finished Streaming 768 beats into Ingress BRAM (50%% Faster than 64-bit!).");

        // Step 4: Trigger START pulse via AXI-Lite
        $display("[TB INFO]: Step 4 - Triggering START Pulse via AXI-Lite (0x00 <= 1)...");
        axi_write_csr(8'h00, 32'd1);

        // Step 5: Wait for Computation Pipeline Completion
        $display("[TB INFO]: Step 5 - Waiting for Core Pipeline Execution (256 rows)...");
        timeout_cycles = 0;
        while (egress_cnt < (TEST_TOKENS * TEST_HEAD_DIM) && timeout_cycles < 400000) begin
            @(posedge clk);
            timeout_cycles++;
            if (timeout_cycles % 50000 == 0) begin
                $display("[TB INFO]: Running... Cycle %0d, Egress Samples: %0d / 4096", timeout_cycles, egress_cnt);
            end
        end

        // Wait a few trailing cycles
        repeat (20) @(posedge clk);

        // Step 6: Final Verification Report
        $display("\n=======================================================================");
        $display("   FINAL VERIFICATION REPORT: ATTENTION CORE TOP 1A (128-BIT, N=32)");
        $display("=======================================================================");
        $display("   Total Egress Samples Monitored : %0d / %0d", egress_cnt, (TEST_TOKENS * TEST_HEAD_DIM));
        $display("   Exact Matched Samples          : %0d", pass_count);
        $display("   Mismatched Samples             : %0d", error_count);
        $display("   TLAST Assertion Errors         : %0d", tlast_errors);
        $display("   Total Execution Clock Cycles   : %0d", timeout_cycles);

        if (error_count == 0 && tlast_errors == 0 && egress_cnt == (TEST_TOKENS * TEST_HEAD_DIM)) begin
            $display("   >>> STATUS: 100%% BIT-EXACT MATCH ACHIEVED! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end


    // Setup waveform dumping for tools like GTKWave
    initial begin
        $dumpfile("tb_attention_core_1A.vcd");
        $dumpvars(0, tb_attention_core_1A);
    end

endmodule
