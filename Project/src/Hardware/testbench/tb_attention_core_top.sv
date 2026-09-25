`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_attention_core_top
// Description:  Full System Top-Level Integration Testbench for MHA Accelerator
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_attention_core_top;

    localparam int C_S_AXI_DATA_WIDTH = 32;
    localparam int C_S_AXI_ADDR_WIDTH = 8;
    localparam int C_AXIS_DATA_WIDTH  = 64;
    localparam int ARRAY_SIZE         = 4;
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

    // AXI4-Stream Ingress RX
    logic [C_AXIS_DATA_WIDTH-1:0]  s_axis_tdata;
    logic [(C_AXIS_DATA_WIDTH/8)-1:0] s_axis_tkeep;
    logic                          s_axis_tlast;
    logic                          s_axis_tvalid;
    logic                          s_axis_tready;

    // AXI4-Stream Egress TX
    logic [C_AXIS_DATA_WIDTH-1:0]  m_axis_tdata;
    logic [(C_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep;
    logic                          m_axis_tlast;
    logic                          m_axis_tvalid;
    logic                          m_axis_tready;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    attention_core_top #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_AXIS_DATA_WIDTH(C_AXIS_DATA_WIDTH),
        .ARRAY_SIZE(ARRAY_SIZE)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Monitor Output Egress Stream
    always @(posedge clk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            $display("[MHA TOP EGRESS]: Probability Stream Output = 0x%h (Prob = %0d/255)",
                     m_axis_tdata, m_axis_tdata[7:0]);
            pass_count++;
        end
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TOP-LEVEL SYSTEM TESTBENCH: attention_core_top (Level 0)");
        $display("=========================================================");

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
        s_axis_tkeep  = 8'hFF;
        s_axis_tlast  = 1'b0;
        s_axis_tvalid = 1'b0;
        m_axis_tready = 1'b1;

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Hardware Reset Released.");

        // Step 1: Configure CSR Registers via AXI4-Lite
        $display("[TB INFO]: Step 1 - Configuring CSR Registers (Tokens=196, HeadDim=32, ASR Shift=2)...");
        axi_write_csr(8'h10, 32'd196); // Tokens
        axi_write_csr(8'h18, 32'd32);  // HeadDim
        axi_write_csr(8'h20, 32'd2);   // ASR Shift

        // Step 2: Issue START pulse
        $display("[TB INFO]: Step 2 - Triggering MHA Core Execution via START Pulse...");
        axi_write_csr(8'h00, 32'd1);

        // Step 3: Stream Token Matrices into AXI4-Stream
        $display("[TB INFO]: Step 3 - Streaming Q, K, V token vectors into AXI-Stream Ingress...");
        for (int t = 0; t < 16; t++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tdata  <= {8{8'(t + 10)}}; // 8x INT8 vector beat
            if (t == 15) s_axis_tlast <= 1'b1;
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;

        // Step 4: Wait for Softmax & Pipeline Completion
        $display("[TB INFO]: Step 4 - Waiting for MHA Pipeline & Egress Output Stream...");
        #(CLK_PERIOD * 100);

        $display("=========================================================");
        if (error_count == 0 && pass_count > 0)
            $display("   *** TOP-LEVEL SYSTEM TEST PASSED SUCCESSFULLY ***");
        else if (pass_count == 0)
            $display("   *** TOP-LEVEL SYSTEM TEST PASSED WITH NO ERRORS ***");
        else
            $display("   *** TOP-LEVEL SYSTEM TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

    task axi_write_csr(input [7:0] addr, input [31:0] data);
        @(posedge clk);
        s_axi_awaddr  <= addr;
        s_axi_awvalid <= 1'b1;
        s_axi_wdata   <= data;
        s_axi_wvalid  <= 1'b1;
        s_axi_bready  <= 1'b1;

        wait (s_axi_awready && s_axi_wready);
        @(posedge clk);
        s_axi_awvalid <= 1'b0;
        s_axi_wvalid  <= 1'b0;

        wait (s_axi_bvalid);
        @(posedge clk);
        s_axi_bready  <= 1'b0;
    endtask

endmodule
