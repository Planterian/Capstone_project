`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_multi_stage_seq_3B
// Description:  Sequential Multi-Stage Verification Testbench for MobileViT-XXS
//               Variant 3B: 2D Systolic MAC Array with DSP Packing (16 DSP48E2).
//               Executes ALL 3 Stages in Continuous Sequence without Hardware Reset:
//                 - Phase 1: Stage 2 (dk = 16, M = 27640, 4,096 context bytes)
//                 - Phase 2: Stage 3 (dk = 20, M = 21936, 5,120 context bytes)
//                 - Phase 3: Stage 4 (dk = 24, M = 18810, 6,144 context bytes)
//               Total Verified: 15,360 Bytes across 1,280 Egress beats.
// Target:        AMD Kria KV260 / Icarus Verilog Simulation
// ============================================================================

`include "hw_scale_params.svh"

module tb_multi_stage_seq_3B;

    localparam int C_S_AXI_DATA_WIDTH = 32;
    localparam int C_S_AXI_ADDR_WIDTH = 8;
    localparam int C_AXIS_DATA_WIDTH  = 128;
    localparam int ARRAY_SIZE         = 4;
    localparam int TEST_TOKENS        = 256;
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

    // Golden Vector Memories for all 3 Stages
    logic [7:0] q_mem_st2   [0:(TEST_TOKENS * 16)-1];
    logic [7:0] k_mem_st2   [0:(TEST_TOKENS * 16)-1];
    logic [7:0] v_mem_st2   [0:(TEST_TOKENS * 16)-1];
    logic [7:0] ctx_mem_st2 [0:(TEST_TOKENS * 16)-1];

    logic [7:0] q_mem_st3   [0:(TEST_TOKENS * 20)-1];
    logic [7:0] k_mem_st3   [0:(TEST_TOKENS * 20)-1];
    logic [7:0] v_mem_st3   [0:(TEST_TOKENS * 20)-1];
    logic [7:0] ctx_mem_st3 [0:(TEST_TOKENS * 20)-1];

    logic [7:0] q_mem_st4   [0:(TEST_TOKENS * 24)-1];
    logic [7:0] k_mem_st4   [0:(TEST_TOKENS * 24)-1];
    logic [7:0] v_mem_st4   [0:(TEST_TOKENS * 24)-1];
    logic [7:0] ctx_mem_st4 [0:(TEST_TOKENS * 24)-1];

    // Verification Tracking State
    int current_stage = 2; // 2, 3, or 4
    int current_head_dim = 16;
    int expected_egress_beats = 256;
    int stage_pass_count = 0;
    int stage_err_count = 0;
    int stage_egress_beats = 0;
    int stage_egress_bytes = 0;
    int stage_tlast_errors = 0;
    int stage_tkeep_errors = 0;

    int total_pass_count = 0;
    int total_err_count = 0;
    int total_egress_bytes = 0;

    // Monitor dynamically variables
    int is_second_beat;
    int row_idx;
    int num_bytes;
    logic [15:0] exp_keep;

    // Monitor Output Egress Stream Dynamically Based on Current Stage
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (current_stage == 2) begin
                // Stage 2: 1 beat per token (16 bytes/beat, tkeep=16'hFFFF)
                if (m_axis_tkeep !== 16'hFFFF) begin
                    $display("[ERROR ST2 @ Beat %0d]: TKEEP mismatch! Expected 16'hFFFF, got 16'h%04h", stage_egress_beats, m_axis_tkeep);
                    stage_tkeep_errors++;
                end

                for (int b = 0; b < 16; b++) begin
                    logic signed [7:0] dut_ctx, exp_ctx;
                    int byte_addr;
                    byte_addr = stage_egress_beats * 16 + b;
                    dut_ctx   = $signed(m_axis_tdata[b*8 +: 8]);

                    if (byte_addr < (TEST_TOKENS * 16)) begin
                        exp_ctx = $signed(ctx_mem_st2[byte_addr]);
                        if (dut_ctx === exp_ctx) begin
                            stage_pass_count++;
                        end else begin
                            if (stage_err_count < 10) begin
                                $display("[ERROR ST2 @ Byte %0d]: DUT = %0d | Golden = %0d", byte_addr, dut_ctx, exp_ctx);
                            end
                            stage_err_count++;
                        end
                        stage_egress_bytes++;
                    end
                end

                if (stage_egress_beats == (TEST_TOKENS - 1)) begin
                    if (!m_axis_tlast) begin
                        $display("[ERROR ST2]: m_axis_tlast was NOT asserted on beat 255!");
                        stage_tlast_errors++;
                    end
                end else if (m_axis_tlast) begin
                    $display("[ERROR ST2]: m_axis_tlast asserted prematurely on beat %0d!", stage_egress_beats);
                    stage_tlast_errors++;
                end

            end else if (current_stage == 3) begin
                // Stage 3: 2 beats per token (Beat 0 = 16B, Beat 1 = 4B)
                is_second_beat = (stage_egress_beats & 1);
                row_idx        = stage_egress_beats >> 1;
                num_bytes      = is_second_beat ? 4 : 16;
                exp_keep       = is_second_beat ? 16'h000F : 16'hFFFF;

                if (m_axis_tkeep !== exp_keep) begin
                    $display("[ERROR ST3 @ Beat %0d]: TKEEP mismatch! Expected 16'h%04h, got 16'h%04h", stage_egress_beats, exp_keep, m_axis_tkeep);
                    stage_tkeep_errors++;
                end

                for (int b = 0; b < num_bytes; b++) begin
                    logic signed [7:0] dut_ctx, exp_ctx;
                    int byte_addr;
                    byte_addr = row_idx * 20 + (is_second_beat ? (16 + b) : b);
                    dut_ctx   = $signed(m_axis_tdata[b*8 +: 8]);

                    if (byte_addr < (TEST_TOKENS * 20)) begin
                        exp_ctx = $signed(ctx_mem_st3[byte_addr]);
                        if (dut_ctx === exp_ctx) begin
                            stage_pass_count++;
                        end else begin
                            if (stage_err_count < 10) begin
                                $display("[ERROR ST3 @ Byte %0d]: DUT = %0d | Golden = %0d", byte_addr, dut_ctx, exp_ctx);
                            end
                            stage_err_count++;
                        end
                        stage_egress_bytes++;
                    end
                end

                if (stage_egress_beats == (TEST_TOKENS * 2 - 1)) begin
                    if (!m_axis_tlast) begin
                        $display("[ERROR ST3]: m_axis_tlast was NOT asserted on beat 511!");
                        stage_tlast_errors++;
                    end
                end else if (m_axis_tlast) begin
                    $display("[ERROR ST3]: m_axis_tlast asserted prematurely on beat %0d!", stage_egress_beats);
                    stage_tlast_errors++;
                end

            end else if (current_stage == 4) begin
                // Stage 4: 2 beats per token (Beat 0 = 16B, Beat 1 = 8B)
                is_second_beat = (stage_egress_beats & 1);
                row_idx        = stage_egress_beats >> 1;
                num_bytes      = is_second_beat ? 8 : 16;
                exp_keep       = is_second_beat ? 16'h00FF : 16'hFFFF;

                if (m_axis_tkeep !== exp_keep) begin
                    $display("[ERROR ST4 @ Beat %0d]: TKEEP mismatch! Expected 16'h%04h, got 16'h%04h", stage_egress_beats, exp_keep, m_axis_tkeep);
                    stage_tkeep_errors++;
                end

                for (int b = 0; b < num_bytes; b++) begin
                    logic signed [7:0] dut_ctx, exp_ctx;
                    int byte_addr;
                    byte_addr = row_idx * 24 + (is_second_beat ? (16 + b) : b);
                    dut_ctx   = $signed(m_axis_tdata[b*8 +: 8]);

                    if (byte_addr < (TEST_TOKENS * 24)) begin
                        exp_ctx = $signed(ctx_mem_st4[byte_addr]);
                        if (dut_ctx === exp_ctx) begin
                            stage_pass_count++;
                        end else begin
                            if (stage_err_count < 10) begin
                                $display("[ERROR ST4 @ Byte %0d]: DUT = %0d | Golden = %0d", byte_addr, dut_ctx, exp_ctx);
                            end
                            stage_err_count++;
                        end
                        stage_egress_bytes++;
                    end
                end

                if (stage_egress_beats == (TEST_TOKENS * 2 - 1)) begin
                    if (!m_axis_tlast) begin
                        $display("[ERROR ST4]: m_axis_tlast was NOT asserted on beat 511!");
                        stage_tlast_errors++;
                    end
                end else if (m_axis_tlast) begin
                    $display("[ERROR ST4]: m_axis_tlast asserted prematurely on beat %0d!", stage_egress_beats);
                    stage_tlast_errors++;
                end
            end

            stage_egress_beats++;
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

    // Streaming tasks for Stage 2 (1 beat/token)
    task stream_st2;
        // Stream Q
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {q_mem_st2[i*16 + 15], q_mem_st2[i*16 + 14], q_mem_st2[i*16 + 13], q_mem_st2[i*16 + 12],
                              q_mem_st2[i*16 + 11], q_mem_st2[i*16 + 10], q_mem_st2[i*16 + 9],  q_mem_st2[i*16 + 8],
                              q_mem_st2[i*16 + 7],  q_mem_st2[i*16 + 6],  q_mem_st2[i*16 + 5],  q_mem_st2[i*16 + 4],
                              q_mem_st2[i*16 + 3],  q_mem_st2[i*16 + 2],  q_mem_st2[i*16 + 1],  q_mem_st2[i*16 + 0]};
            s_axis_tlast  <= 1'b0;
        end
        // Stream K
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {k_mem_st2[i*16 + 15], k_mem_st2[i*16 + 14], k_mem_st2[i*16 + 13], k_mem_st2[i*16 + 12],
                              k_mem_st2[i*16 + 11], k_mem_st2[i*16 + 10], k_mem_st2[i*16 + 9],  k_mem_st2[i*16 + 8],
                              k_mem_st2[i*16 + 7],  k_mem_st2[i*16 + 6],  k_mem_st2[i*16 + 5],  k_mem_st2[i*16 + 4],
                              k_mem_st2[i*16 + 3],  k_mem_st2[i*16 + 2],  k_mem_st2[i*16 + 1],  k_mem_st2[i*16 + 0]};
            s_axis_tlast  <= 1'b0;
        end
        // Stream V
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {v_mem_st2[i*16 + 15], v_mem_st2[i*16 + 14], v_mem_st2[i*16 + 13], v_mem_st2[i*16 + 12],
                              v_mem_st2[i*16 + 11], v_mem_st2[i*16 + 10], v_mem_st2[i*16 + 9],  v_mem_st2[i*16 + 8],
                              v_mem_st2[i*16 + 7],  v_mem_st2[i*16 + 6],  v_mem_st2[i*16 + 5],  v_mem_st2[i*16 + 4],
                              v_mem_st2[i*16 + 3],  v_mem_st2[i*16 + 2],  v_mem_st2[i*16 + 1],  v_mem_st2[i*16 + 0]};
            s_axis_tlast  <= (i == TEST_TOKENS - 1);
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    // Streaming tasks for Stage 3 (2 beats/token, dk=20)
    task stream_st3;
        // Stream Q
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {q_mem_st3[i*20 + 15], q_mem_st3[i*20 + 14], q_mem_st3[i*20 + 13], q_mem_st3[i*20 + 12],
                              q_mem_st3[i*20 + 11], q_mem_st3[i*20 + 10], q_mem_st3[i*20 + 9],  q_mem_st3[i*20 + 8],
                              q_mem_st3[i*20 + 7],  q_mem_st3[i*20 + 6],  q_mem_st3[i*20 + 5],  q_mem_st3[i*20 + 4],
                              q_mem_st3[i*20 + 3],  q_mem_st3[i*20 + 2],  q_mem_st3[i*20 + 1],  q_mem_st3[i*20 + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h000F;
            s_axis_tdata  <= {96'd0, q_mem_st3[i*20 + 19], q_mem_st3[i*20 + 18], q_mem_st3[i*20 + 17], q_mem_st3[i*20 + 16]};
            s_axis_tlast  <= 1'b0;
        end
        // Stream K
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {k_mem_st3[i*20 + 15], k_mem_st3[i*20 + 14], k_mem_st3[i*20 + 13], k_mem_st3[i*20 + 12],
                              k_mem_st3[i*20 + 11], k_mem_st3[i*20 + 10], k_mem_st3[i*20 + 9],  k_mem_st3[i*20 + 8],
                              k_mem_st3[i*20 + 7],  k_mem_st3[i*20 + 6],  k_mem_st3[i*20 + 5],  k_mem_st3[i*20 + 4],
                              k_mem_st3[i*20 + 3],  k_mem_st3[i*20 + 2],  k_mem_st3[i*20 + 1],  k_mem_st3[i*20 + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h000F;
            s_axis_tdata  <= {96'd0, k_mem_st3[i*20 + 19], k_mem_st3[i*20 + 18], k_mem_st3[i*20 + 17], k_mem_st3[i*20 + 16]};
            s_axis_tlast  <= 1'b0;
        end
        // Stream V
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {v_mem_st3[i*20 + 15], v_mem_st3[i*20 + 14], v_mem_st3[i*20 + 13], v_mem_st3[i*20 + 12],
                              v_mem_st3[i*20 + 11], v_mem_st3[i*20 + 10], v_mem_st3[i*20 + 9],  v_mem_st3[i*20 + 8],
                              v_mem_st3[i*20 + 7],  v_mem_st3[i*20 + 6],  v_mem_st3[i*20 + 5],  v_mem_st3[i*20 + 4],
                              v_mem_st3[i*20 + 3],  v_mem_st3[i*20 + 2],  v_mem_st3[i*20 + 1],  v_mem_st3[i*20 + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h000F;
            s_axis_tdata  <= {96'd0, v_mem_st3[i*20 + 19], v_mem_st3[i*20 + 18], v_mem_st3[i*20 + 17], v_mem_st3[i*20 + 16]};
            s_axis_tlast  <= (i == TEST_TOKENS - 1);
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    // Streaming tasks for Stage 4 (2 beats/token, dk=24)
    task stream_st4;
        // Stream Q
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {q_mem_st4[i*24 + 15], q_mem_st4[i*24 + 14], q_mem_st4[i*24 + 13], q_mem_st4[i*24 + 12],
                              q_mem_st4[i*24 + 11], q_mem_st4[i*24 + 10], q_mem_st4[i*24 + 9],  q_mem_st4[i*24 + 8],
                              q_mem_st4[i*24 + 7],  q_mem_st4[i*24 + 6],  q_mem_st4[i*24 + 5],  q_mem_st4[i*24 + 4],
                              q_mem_st4[i*24 + 3],  q_mem_st4[i*24 + 2],  q_mem_st4[i*24 + 1],  q_mem_st4[i*24 + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h00FF;
            s_axis_tdata  <= {64'd0, q_mem_st4[i*24 + 23], q_mem_st4[i*24 + 22], q_mem_st4[i*24 + 21], q_mem_st4[i*24 + 20],
                                     q_mem_st4[i*24 + 19], q_mem_st4[i*24 + 18], q_mem_st4[i*24 + 17], q_mem_st4[i*24 + 16]};
            s_axis_tlast  <= 1'b0;
        end
        // Stream K
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {k_mem_st4[i*24 + 15], k_mem_st4[i*24 + 14], k_mem_st4[i*24 + 13], k_mem_st4[i*24 + 12],
                              k_mem_st4[i*24 + 11], k_mem_st4[i*24 + 10], k_mem_st4[i*24 + 9],  k_mem_st4[i*24 + 8],
                              k_mem_st4[i*24 + 7],  k_mem_st4[i*24 + 6],  k_mem_st4[i*24 + 5],  k_mem_st4[i*24 + 4],
                              k_mem_st4[i*24 + 3],  k_mem_st4[i*24 + 2],  k_mem_st4[i*24 + 1],  k_mem_st4[i*24 + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h00FF;
            s_axis_tdata  <= {64'd0, k_mem_st4[i*24 + 23], k_mem_st4[i*24 + 22], k_mem_st4[i*24 + 21], k_mem_st4[i*24 + 20],
                                     k_mem_st4[i*24 + 19], k_mem_st4[i*24 + 18], k_mem_st4[i*24 + 17], k_mem_st4[i*24 + 16]};
            s_axis_tlast  <= 1'b0;
        end
        // Stream V
        for (int i = 0; i < TEST_TOKENS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'hFFFF;
            s_axis_tdata  <= {v_mem_st4[i*24 + 15], v_mem_st4[i*24 + 14], v_mem_st4[i*24 + 13], v_mem_st4[i*24 + 12],
                              v_mem_st4[i*24 + 11], v_mem_st4[i*24 + 10], v_mem_st4[i*24 + 9],  v_mem_st4[i*24 + 8],
                              v_mem_st4[i*24 + 7],  v_mem_st4[i*24 + 6],  v_mem_st4[i*24 + 5],  v_mem_st4[i*24 + 4],
                              v_mem_st4[i*24 + 3],  v_mem_st4[i*24 + 2],  v_mem_st4[i*24 + 1],  v_mem_st4[i*24 + 0]};
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tkeep  <= 16'h00FF;
            s_axis_tdata  <= {64'd0, v_mem_st4[i*24 + 23], v_mem_st4[i*24 + 22], v_mem_st4[i*24 + 21], v_mem_st4[i*24 + 20],
                                     v_mem_st4[i*24 + 19], v_mem_st4[i*24 + 18], v_mem_st4[i*24 + 17], v_mem_st4[i*24 + 16]};
            s_axis_tlast  <= (i == TEST_TOKENS - 1);
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    int timeout_cycles;
    int st2_cycles, st3_cycles, st4_cycles;
    int st2_pass, st3_pass, st4_pass;
    int st2_err, st3_err, st4_err;

    initial begin
        $display("=======================================================================");
        $display("   MASTER SEQUENTIAL MULTI-STAGE DV SUITE - VARIANT 3B (2D SYSTOLIC)");
        $display("   TESTING CONTINUOUS EXECUTION ACROSS ALL 3 STAGES WITHOUT RESET");
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

        // Step 1: Load All Test Vectors
        $display("[TB INFO]: Loading Golden Test Vectors for Stage 2, 3, 4...");
        $readmemh("q_matrix_st2.hex", q_mem_st2);
        $readmemh("k_matrix_st2.hex", k_mem_st2);
        $readmemh("v_matrix_st2.hex", v_mem_st2);
        $readmemh("golden_context_out_st2.hex", ctx_mem_st2);

        $readmemh("q_matrix_st3.hex", q_mem_st3);
        $readmemh("k_matrix_st3.hex", k_mem_st3);
        $readmemh("v_matrix_st3.hex", v_mem_st3);
        $readmemh("golden_context_out_st3.hex", ctx_mem_st3);

        $readmemh("q_matrix_st4.hex", q_mem_st4);
        $readmemh("k_matrix_st4.hex", k_mem_st4);
        $readmemh("v_matrix_st4.hex", v_mem_st4);
        $readmemh("golden_context_out_st4.hex", ctx_mem_st4);
        $display("[TB INFO]: Loaded 15,360 total expected context bytes.");

        // Release hardware reset ONCE at startup
        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Hardware Initial Reset Released. Commencing Sequential Run.\n");

        // --------------------------------------------------------------------
        // PHASE 1: STAGE 2 (dk=16, 4,096 Bytes)
        // --------------------------------------------------------------------
        current_stage = 2;
        current_head_dim = 16;
        expected_egress_beats = 256;
        stage_pass_count = 0;
        stage_err_count = 0;
        stage_egress_beats = 0;
        stage_egress_bytes = 0;
        stage_tlast_errors = 0;
        stage_tkeep_errors = 0;

        $display(">>> [PHASE 1/3]: EXECUTING STAGE 2 (dk=16, M=27640, E=22) <<<");
        // Clear buffer before streaming
        axi_write_csr(8'h00, 32'd2); // Clear buffer pulse
        axi_write_csr(8'h00, 32'd0); // Deassert
        axi_write_csr(8'h10, 32'd256);
        axi_write_csr(8'h14, 32'd16);
        axi_write_csr(8'h18, 32'd1);
        axi_write_csr(8'h20, 32'd27640);
        axi_write_csr(8'h24, 32'd22);

        stream_st2();
        axi_write_csr(8'h00, 32'd1); // Trigger Start
        axi_write_csr(8'h00, 32'd0); // Deassert Start

        timeout_cycles = 0;
        while (stage_egress_beats < expected_egress_beats && timeout_cycles < 500000) begin
            @(posedge clk);
            timeout_cycles++;
        end
        repeat (50) @(posedge clk);
        st2_cycles = timeout_cycles;
        st2_pass   = stage_pass_count;
        st2_err    = stage_err_count;
        $display(">>> [PHASE 1/3 DONE]: Stage 2 Finished in %0d cycles. Matched: %0d / 4096 (Errors: %0d)\n", 
                 st2_cycles, st2_pass, st2_err);

        // --------------------------------------------------------------------
        // PHASE 2: STAGE 3 (dk=20, 5,120 Bytes) -- NO RESET!
        // --------------------------------------------------------------------
        current_stage = 3;
        current_head_dim = 20;
        expected_egress_beats = 512;
        stage_pass_count = 0;
        stage_err_count = 0;
        stage_egress_beats = 0;
        stage_egress_bytes = 0;
        stage_tlast_errors = 0;
        stage_tkeep_errors = 0;

        $display(">>> [PHASE 2/3]: RECONFIGURING CSR FOR STAGE 3 (dk=20, M=21936, E=22) WITHOUT RESET <<<");
        // Clear buffer before streaming
        axi_write_csr(8'h00, 32'd2); // Clear buffer pulse
        axi_write_csr(8'h00, 32'd0); // Deassert
        axi_write_csr(8'h10, 32'd256);
        axi_write_csr(8'h14, 32'd20);
        axi_write_csr(8'h18, 32'd1);
        axi_write_csr(8'h20, 32'd21936);
        axi_write_csr(8'h24, 32'd22);

        stream_st3();
        axi_write_csr(8'h00, 32'd1); // Trigger Start
        axi_write_csr(8'h00, 32'd0); // Deassert Start

        timeout_cycles = 0;
        while (stage_egress_beats < expected_egress_beats && timeout_cycles < 500000) begin
            @(posedge clk);
            timeout_cycles++;
        end
        repeat (50) @(posedge clk);
        st3_cycles = timeout_cycles;
        st3_pass   = stage_pass_count;
        st3_err    = stage_err_count;
        $display(">>> [PHASE 2/3 DONE]: Stage 3 Finished in %0d cycles. Matched: %0d / 5120 (Errors: %0d)\n", 
                 st3_cycles, st3_pass, st3_err);

        // --------------------------------------------------------------------
        // PHASE 3: STAGE 4 (dk=24, 6,144 Bytes) -- NO RESET!
        // --------------------------------------------------------------------
        current_stage = 4;
        current_head_dim = 24;
        expected_egress_beats = 512;
        stage_pass_count = 0;
        stage_err_count = 0;
        stage_egress_beats = 0;
        stage_egress_bytes = 0;
        stage_tlast_errors = 0;
        stage_tkeep_errors = 0;

        $display(">>> [PHASE 3/3]: RECONFIGURING CSR FOR STAGE 4 (dk=24, M=18810, E=22) WITHOUT RESET <<<");
        // Clear buffer before streaming
        axi_write_csr(8'h00, 32'd2); // Clear buffer pulse
        axi_write_csr(8'h00, 32'd0); // Deassert
        axi_write_csr(8'h10, 32'd256);
        axi_write_csr(8'h14, 32'd24);
        axi_write_csr(8'h18, 32'd1);
        axi_write_csr(8'h20, 32'd18810);
        axi_write_csr(8'h24, 32'd22);

        stream_st4();
        axi_write_csr(8'h00, 32'd1); // Trigger Start
        axi_write_csr(8'h00, 32'd0); // Deassert Start

        timeout_cycles = 0;
        while (stage_egress_beats < expected_egress_beats && timeout_cycles < 500000) begin
            @(posedge clk);
            timeout_cycles++;
        end
        repeat (50) @(posedge clk);
        st4_cycles = timeout_cycles;
        st4_pass   = stage_pass_count;
        st4_err    = stage_err_count;
        $display(">>> [PHASE 3/3 DONE]: Stage 4 Finished in %0d cycles. Matched: %0d / 6144 (Errors: %0d)\n", 
                 st4_cycles, st4_pass, st4_err);

        // --------------------------------------------------------------------
        // GRAND MASTER SEQUENTIAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        total_pass_count = st2_pass + st3_pass + st4_pass;
        total_err_count  = st2_err + st3_err + st4_err;
        total_egress_bytes = 4096 + 5120 + 6144;

        $display("=======================================================================");
        $display("   GRAND MASTER SEQUENTIAL VERIFICATION REPORT (VARIANT 3B)");
        $display("=======================================================================");
        $display("   STAGE 2 (dk=16): %0d / 4,096 bytes matched (%0d cycles)", st2_pass, st2_cycles);
        $display("   STAGE 3 (dk=20): %0d / 5,120 bytes matched (%0d cycles)", st3_pass, st3_cycles);
        $display("   STAGE 4 (dk=24): %0d / 6,144 bytes matched (%0d cycles)", st4_pass, st4_cycles);
        $display("   --------------------------------------------------------------------");
        $display("   TOTAL BYTES VERIFIED           : %0d / %0d", total_pass_count, total_egress_bytes);
        $display("   TOTAL MISMATCHED SAMPLES       : %0d", total_err_count);
        $display("   TOTAL SEQUENTIAL CLOCK CYCLES  : %0d (~%0.2f ms @ 200MHz)", 
                 (st2_cycles + st3_cycles + st4_cycles), (st2_cycles + st3_cycles + st4_cycles) * 0.000005);

        if (total_err_count == 0 && total_pass_count == total_egress_bytes) begin
            $display("   >>> STATUS: SEQUENTIAL MULTI-STAGE 3B 100%% BIT-EXACT MATCH! [PASS] <<<");
        end else begin
            $display("   >>> STATUS: SEQUENTIAL MULTI-STAGE 3B VERIFICATION FAILED! [FAIL] <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end


    // Setup waveform dumping for tools like GTKWave
    initial begin
        $dumpfile("tb_multi_stage_seq_3B.vcd");
        $dumpvars(0, tb_multi_stage_seq_3B);
    end

endmodule
