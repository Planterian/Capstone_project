`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_axis_adapter
// Description:  Automated Testbench for AXI4-Stream Skid Buffer (Backpressure)
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_axis_adapter;

    localparam int DATA_WIDTH = 64;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                  clk;
    logic                  rst_n;

    logic [DATA_WIDTH-1:0] s_axis_tdata;
    logic                  s_axis_tvalid;
    logic                  s_axis_tready;
    logic                  s_axis_tlast;

    logic [DATA_WIDTH-1:0] m_axis_tdata;
    logic                  m_axis_tvalid;
    logic                  m_axis_tready;
    logic                  m_axis_tlast;

    logic [DATA_WIDTH-1:0] sent_queue [$];
    logic [DATA_WIDTH-1:0] received_queue [$];

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    axis_adapter #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Monitor Output Data
    always @(posedge clk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            received_queue.push_back(m_axis_tdata);
            $display("[EGRESS RECV] Received Data = 0x%h", m_axis_tdata);
        end
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: axis_adapter (Level 1)");
        $display("=========================================================");

        rst_n         = 1'b0;
        s_axis_tdata  = '0;
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
        m_axis_tready = 1'b1;

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Phase 1: Normal Stream Flow (5 words)
        $display("[TB INFO]: Phase 1 - Streaming 5 words under normal ready flow...");
        for (int i = 1; i <= 5; i++) begin
            send_word(64'(i));
        end

        // Phase 2: Simulate Downstream Backpressure (m_axis_tready = 0)
        #(CLK_PERIOD * 2);
        $display("[TB INFO]: Phase 2 - Introducing Backpressure (m_axis_tready = 0)...");
        @(posedge clk);
        m_axis_tready <= 1'b0;

        // Drive new data while downstream is blocked
        @(posedge clk);
        s_axis_tvalid <= 1'b1;
        s_axis_tdata  <= 64'hAAAA_BBBB_CCCC_DDDD;
        sent_queue.push_back(64'hAAAA_BBBB_CCCC_DDDD);

        @(posedge clk);
        if (s_axis_tready) begin
            $display("[SKID CAPTURE]: Skid buffer captured backpressure beat.");
        end

        // Drive another beat (should cause s_axis_tready to drop)
        s_axis_tdata <= 64'h1111_2222_3333_4444;

        #(CLK_PERIOD * 3);
        $display("[TB INFO]: Phase 3 - Releasing Backpressure (m_axis_tready = 1)...");
        sent_queue.push_back(64'h1111_2222_3333_4444);
        @(posedge clk);
        m_axis_tready <= 1'b1;

        #(CLK_PERIOD * 10);

        // Phase 4: Verify Queues Match
        $display("---------------------------------------------------------");
        $display("[VERIFY]: Comparing Sent vs Received Data Streams...");
        if (sent_queue.size() == received_queue.size()) begin
            $display("[SIZE MATCH]: Sent %0d words, Received %0d words.", sent_queue.size(), received_queue.size());
            for (int k = 0; k < sent_queue.size(); k++) begin
                if (sent_queue[k] == received_queue[k]) begin
                    pass_count++;
                end else begin
                    $display("[MISMATCH]: Word %0d - Sent 0x%h != Recv 0x%h", k, sent_queue[k], received_queue[k]);
                    error_count++;
                end
            end
        end else begin
            $display("[FAIL]: Queue size mismatch! Sent %0d, Recv %0d", sent_queue.size(), received_queue.size());
            error_count++;
        end

        $display("=========================================================");
        if (error_count == 0)
            $display("   *** TEST PASSED SUCCESSFULLY (%0d Words Matched) ***", pass_count);
        else
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

    task send_word(input logic [DATA_WIDTH-1:0] data);
        @(posedge clk);
        s_axis_tvalid <= 1'b1;
        s_axis_tdata  <= data;
        sent_queue.push_back(data);

        wait (s_axis_tready == 1'b1);
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
    endtask

endmodule
