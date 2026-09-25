`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_axi_lite_regs
// Description:  Automated Self-Checking Testbench for AXI4-Lite Register File
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_axi_lite_regs;

    localparam int C_S_AXI_DATA_WIDTH = 32;
    localparam int C_S_AXI_ADDR_WIDTH = 8;
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                          S_AXI_ACLK;
    logic                          S_AXI_ARESETN;

    logic [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR;
    logic                          S_AXI_AWVALID;
    logic                          S_AXI_AWREADY;
    logic [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA;
    logic [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB;
    logic                          S_AXI_WVALID;
    logic                          S_AXI_WREADY;
    logic [1:0]                    S_AXI_BRESP;
    logic                          S_AXI_BVALID;
    logic                          S_AXI_BREADY;
    logic [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR;
    logic                          S_AXI_ARVALID;
    logic                          S_AXI_ARREADY;
    logic [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA;
    logic [1:0]                    S_AXI_RRESP;
    logic                          S_AXI_RVALID;
    logic                          S_AXI_RREADY;

    logic                          cfg_start_pulse;
    logic                          cfg_clear_buf;
    logic [31:0]                   reg_nbytes;
    logic [31:0]                   reg_tokens;
    logic [31:0]                   reg_hdim;
    logic [3:0]                    reg_shift_val;
    logic                          core_busy;
    logic                          core_done;
    logic                          core_error;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    axi_lite_regs #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
    ) u_dut (
        .S_AXI_ACLK(S_AXI_ACLK),
        .S_AXI_ARESETN(S_AXI_ARESETN),
        .S_AXI_AWADDR(S_AXI_AWADDR),
        .S_AXI_AWVALID(S_AXI_AWVALID),
        .S_AXI_AWREADY(S_AXI_AWREADY),
        .S_AXI_WDATA(S_AXI_WDATA),
        .S_AXI_WSTRB(S_AXI_WSTRB),
        .S_AXI_WVALID(S_AXI_WVALID),
        .S_AXI_WREADY(S_AXI_WREADY),
        .S_AXI_BRESP(S_AXI_BRESP),
        .S_AXI_BVALID(S_AXI_BVALID),
        .S_AXI_BREADY(S_AXI_BREADY),
        .S_AXI_ARADDR(S_AXI_ARADDR),
        .S_AXI_ARVALID(S_AXI_ARVALID),
        .S_AXI_ARREADY(S_AXI_ARREADY),
        .S_AXI_RDATA(S_AXI_RDATA),
        .S_AXI_RRESP(S_AXI_RRESP),
        .S_AXI_RVALID(S_AXI_RVALID),
        .S_AXI_RREADY(S_AXI_RREADY),
        .cfg_start_pulse(cfg_start_pulse),
        .cfg_clear_buf(cfg_clear_buf),
        .reg_nbytes(reg_nbytes),
        .reg_tokens(reg_tokens),
        .reg_hdim(reg_hdim),
        .reg_shift_val(reg_shift_val),
        .core_busy(core_busy),
        .core_done(core_done),
        .core_error(core_error)
    );

    // Clock Generator
    initial begin
        S_AXI_ACLK = 0;
        forever #(CLK_PERIOD / 2.0) S_AXI_ACLK = ~S_AXI_ACLK;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: axi_lite_regs (Level 0)");
        $display("=========================================================");

        S_AXI_ARESETN = 1'b0;
        S_AXI_AWADDR  = '0;
        S_AXI_AWVALID = 1'b0;
        S_AXI_WDATA   = '0;
        S_AXI_WSTRB   = 4'hF;
        S_AXI_WVALID  = 1'b0;
        S_AXI_BREADY  = 1'b0;
        S_AXI_ARADDR  = '0;
        S_AXI_ARVALID = 1'b0;
        S_AXI_RREADY  = 1'b0;

        core_busy  = 1'b0;
        core_done  = 1'b0;
        core_error = 1'b0;

        #(CLK_PERIOD * 5);
        S_AXI_ARESETN = 1'b1;
        $display("[TB INFO]: Reset released.");

        // Step 1: Write NBYTES register (0x04 = 25000)
        $display("[TB INFO]: Step 1 - Writing NBYTES = 25000 via AXI4-Lite...");
        axi_write(8'h04, 32'd25000);

        if (reg_nbytes == 32'd25000) begin
            $display("[PASS]: reg_nbytes updated correctly to 25000.");
            pass_count++;
        end else begin
            $display("[FAIL]: reg_nbytes update failed!");
            error_count++;
        end

        // Step 2: Read NBYTES register (0x04)
        $display("[TB INFO]: Step 2 - Reading NBYTES from 0x04 via AXI4-Lite...");
        axi_read(8'h04);

        if (S_AXI_RDATA == 32'd25000) begin
            $display("[PASS]: AXI Read returned expected 25000.");
            pass_count++;
        end else begin
            $display("[FAIL]: AXI Read returned unexpected value!");
            error_count++;
        end

        // Step 3: Trigger START pulse via 0x00
        $display("[TB INFO]: Step 3 - Triggering START pulse via CTRL register 0x00...");
        axi_write(8'h00, 32'd1);

        if (cfg_start_pulse == 1'b1) begin
            $display("[PASS]: cfg_start_pulse asserted!");
            pass_count++;
        end else begin
            $display("[FAIL]: cfg_start_pulse failed to assert!");
            error_count++;
        end

        #(CLK_PERIOD * 5);
        $display("=========================================================");
        if (error_count == 0)
            $display("   *** TEST PASSED SUCCESSFULLY ***");
        else
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

    task axi_write(input [7:0] addr, input [31:0] data);
        @(posedge S_AXI_ACLK);
        S_AXI_AWADDR  <= addr;
        S_AXI_AWVALID <= 1'b1;
        S_AXI_WDATA   <= data;
        S_AXI_WVALID  <= 1'b1;
        S_AXI_BREADY  <= 1'b1;

        wait (S_AXI_AWREADY && S_AXI_WREADY);
        @(posedge S_AXI_ACLK);
        S_AXI_AWVALID <= 1'b0;
        S_AXI_WVALID  <= 1'b0;

        wait (S_AXI_BVALID);
        @(posedge S_AXI_ACLK);
        S_AXI_BREADY  <= 1'b0;
    endtask

    task axi_read(input [7:0] addr);
        @(posedge S_AXI_ACLK);
        S_AXI_ARADDR  <= addr;
        S_AXI_ARVALID <= 1'b1;
        S_AXI_RREADY  <= 1'b1;

        wait (S_AXI_ARREADY);
        @(posedge S_AXI_ACLK);
        S_AXI_ARVALID <= 1'b0;

        wait (S_AXI_RVALID);
        @(posedge S_AXI_ACLK);
        S_AXI_RREADY  <= 1'b0;
    endtask

endmodule
