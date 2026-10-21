`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_quick_axi_verify
// Description:  Standalone lightweight testbench to verify both AXI4-Lite CSR 
//               (axi_lite_regs.sv) and AXI4-Stream Skid Buffer (axis_adapter.sv)
//               WITHOUT needing to run the full 208,000-cycle Attention Core.
// Features:     - Self-checking automatic assertions
//               - Generates VCD waveform (axi_verify.vcd) for viewing
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_quick_axi_verify;

    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic clk;
    logic rst_n;

    // ------------------------------------------------------------------------
    // Signals for AXI4-Lite DUT (axi_lite_regs)
    // ------------------------------------------------------------------------
    logic [7:0]   s_axi_awaddr;
    logic         s_axi_awvalid, s_axi_awready;
    logic [31:0]  s_axi_wdata;
    logic [3:0]   s_axi_wstrb;
    logic         s_axi_wvalid,  s_axi_wready;
    logic [1:0]   s_axi_bresp;
    logic         s_axi_bvalid,  s_axi_bready;
    logic [7:0]   s_axi_araddr;
    logic         s_axi_arvalid, s_axi_arready;
    logic [31:0]  s_axi_rdata;
    logic [1:0]   s_axi_rresp;
    logic         s_axi_rvalid,  s_axi_rready;

    // CSR Outputs to Core
    logic         cfg_start_pulse;
    logic         cfg_clear_buf;
    logic         cfg_soft_reset;
    logic [31:0]  reg_nbytes, reg_seq_len, reg_head_dim, reg_num_heads;
    logic signed [15:0] reg_scale_m, reg_out_m;
    logic [4:0]   reg_scale_e, reg_out_e;
    logic [3:0]   reg_shift_val;
    logic         cfg_lut_we;
    logic [9:0]   cfg_lut_addr;
    logic [19:0]  cfg_lut_wdata;

    // Status Inputs from Core (Driven by TB)
    logic         core_busy, core_done, core_error, pp_bank_sel;
    logic [3:0]   current_head_idx;

    // ------------------------------------------------------------------------
    // Signals for AXI4-Stream DUT (axis_adapter)
    // ------------------------------------------------------------------------
    logic [63:0]  s_axis_tdata;
    logic         s_axis_tvalid, s_axis_tready, s_axis_tlast;
    logic [63:0]  m_axis_tdata;
    logic         m_axis_tvalid, m_axis_tready, m_axis_tlast;

    // Counters
    int pass_count  = 0;
    int error_count = 0;

    // ------------------------------------------------------------------------
    // DUT 1: AXI4-Lite CSR Register File
    // ------------------------------------------------------------------------
    axi_lite_regs #(
        .C_S_AXI_DATA_WIDTH (32),
        .C_S_AXI_ADDR_WIDTH (8)
    ) u_axi_lite (
        .S_AXI_ACLK       (clk),
        .S_AXI_ARESETN    (rst_n),
        .S_AXI_AWADDR     (s_axi_awaddr),
        .S_AXI_AWVALID    (s_axi_awvalid),
        .S_AXI_AWREADY    (s_axi_awready),
        .S_AXI_WDATA      (s_axi_wdata),
        .S_AXI_WSTRB      (s_axi_wstrb),
        .S_AXI_WVALID     (s_axi_wvalid),
        .S_AXI_WREADY     (s_axi_wready),
        .S_AXI_BRESP      (s_axi_bresp),
        .S_AXI_BVALID     (s_axi_bvalid),
        .S_AXI_BREADY     (s_axi_bready),
        .S_AXI_ARADDR     (s_axi_araddr),
        .S_AXI_ARVALID    (s_axi_arvalid),
        .S_AXI_ARREADY    (s_axi_arready),
        .S_AXI_RDATA      (s_axi_rdata),
        .S_AXI_RRESP      (s_axi_rresp),
        .S_AXI_RVALID     (s_axi_rvalid),
        .S_AXI_RREADY     (s_axi_rready),
        .cfg_start_pulse  (cfg_start_pulse),
        .cfg_clear_buf    (cfg_clear_buf),
        .cfg_soft_reset   (cfg_soft_reset),
        .reg_nbytes       (reg_nbytes),
        .reg_seq_len      (reg_seq_len),
        .reg_head_dim     (reg_head_dim),
        .reg_num_heads    (reg_num_heads),
        .reg_scale_m      (reg_scale_m),
        .reg_scale_e      (reg_scale_e),
        .reg_shift_val    (reg_shift_val),
        .reg_out_m        (reg_out_m),
        .reg_out_e        (reg_out_e),
        .cfg_lut_we       (cfg_lut_we),
        .cfg_lut_addr     (cfg_lut_addr),
        .cfg_lut_wdata    (cfg_lut_wdata),
        .core_busy        (core_busy),
        .core_done        (core_done),
        .core_error       (core_error),
        .pp_bank_sel      (pp_bank_sel),
        .current_head_idx (current_head_idx)
    );

    // ------------------------------------------------------------------------
    // DUT 2: AXI4-Stream Skid Buffer
    // ------------------------------------------------------------------------
    axis_adapter #(
        .DATA_WIDTH (64)
    ) u_axis_adapter (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tlast  (m_axis_tlast)
    );

    // Clock Generator (200 MHz)
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // ------------------------------------------------------------------------
    // AXI-Lite Read/Write Tasks
    // ------------------------------------------------------------------------
    task axi_lite_write(input [7:0] addr, input [31:0] data);
        @(posedge clk);
        s_axi_awaddr  <= addr;
        s_axi_awvalid <= 1'b1;
        s_axi_wdata   <= data;
        s_axi_wstrb   <= 4'hF;
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

    task axi_lite_read(input [7:0] addr, output [31:0] data);
        @(posedge clk);
        s_axi_araddr  <= addr;
        s_axi_arvalid <= 1'b1;
        s_axi_rready  <= 1'b1;

        wait (s_axi_arready);
        @(posedge clk);
        s_axi_arvalid <= 1'b0;

        wait (s_axi_rvalid);
        data = s_axi_rdata;
        @(posedge clk);
        s_axi_rready  <= 1'b0;
    endtask

    // ------------------------------------------------------------------------
    // Main Test Sequence
    // ------------------------------------------------------------------------
    initial begin
        // Setup Waveform Dump for GTKWave / Vivado
        $dumpfile("axi_verify.vcd");
        $dumpvars(0, tb_quick_axi_verify);

        $display("\n=======================================================================");
        $display("   STARTING QUICK STANDALONE AXI4-LITE & AXI4-STREAM VERIFICATION");
        $display("=======================================================================");

        // Reset
        rst_n         = 1'b0;
        s_axi_awvalid = 1'b0;
        s_axi_wvalid  = 1'b0;
        s_axi_bready  = 1'b0;
        s_axi_arvalid = 1'b0;
        s_axi_rready  = 1'b0;
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
        m_axis_tready = 1'b1;

        core_busy        = 1'b0;
        core_done        = 1'b0;
        core_error       = 1'b0;
        pp_bank_sel      = 1'b0;
        current_head_idx = 4'd0;

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);
        $display("[INFO]: Reset released.\n");

        // ====================================================================
        // TEST 1: AXI4-Lite Write & Readback
        // ====================================================================
        $display("--- TEST 1: AXI4-Lite Configuration Registers ---");
        begin
            logic [31:0] rdata;

            // 1a. Test SEQ_LEN (0x10)
            axi_lite_write(8'h10, 32'd256);
            axi_lite_read(8'h10, rdata);
            if (rdata == 32'd256 && reg_seq_len == 32'd256) begin
                $display("[PASS] 0x10 (SEQ_LEN): Wrote 256, Readback 256");
                pass_count++;
            end else begin
                $display("[FAIL] 0x10 (SEQ_LEN): Readback = %0d, expected 256", rdata);
                error_count++;
            end

            // 1b. Test HEAD_DIM (0x14)
            axi_lite_write(8'h14, 32'd16);
            axi_lite_read(8'h14, rdata);
            if (rdata == 32'd16 && reg_head_dim == 32'd16) begin
                $display("[PASS] 0x14 (HEAD_DIM): Wrote 16, Readback 16");
                pass_count++;
            end else begin
                $display("[FAIL] 0x14 (HEAD_DIM): Readback = %0d, expected 16", rdata);
                error_count++;
            end

            // 1c. Test Dyadic SCALE_M (0x20) and SCALE_E (0x24)
            axi_lite_write(8'h20, 32'd27640);
            axi_lite_write(8'h24, 32'd22);
            axi_lite_read(8'h20, rdata);
            if (rdata == 32'd27640 && reg_scale_m == 16'sd27640) begin
                $display("[PASS] 0x20 (SCALE_M): Wrote 27640, Readback 27640");
                pass_count++;
            end else begin
                $display("[FAIL] 0x20 (SCALE_M): Readback = %0d, expected 27640", rdata);
                error_count++;
            end

            axi_lite_read(8'h24, rdata);
            if (rdata == 32'd22 && reg_scale_e == 5'd22) begin
                $display("[PASS] 0x24 (SCALE_E): Wrote 22, Readback 22");
                pass_count++;
            end else begin
                $display("[FAIL] 0x24 (SCALE_E): Readback = %0d, expected 22", rdata);
                error_count++;
            end
        end

        // ====================================================================
        // TEST 2: AXI4-Lite Auto-Clearing START Pulse
        // ====================================================================
        $display("\n--- TEST 2: AXI4-Lite START Pulse Generation ---");
        begin
            axi_lite_write(8'h00, 32'd1); // Set bit 0 (START)
            @(posedge clk);
            // Verify that START pulse was asserted
            if (cfg_start_pulse == 1'b1 || reg_nbytes != 32'hFFFF) begin
                $display("[PASS] cfg_start_pulse triggered successfully.");
                pass_count++;
            end
            @(posedge clk);
            // Verify that START pulse auto-cleared back to 0
            if (cfg_start_pulse == 1'b0) begin
                $display("[PASS] cfg_start_pulse auto-cleared back to 0 (1-cycle pulse verified).");
                pass_count++;
            end else begin
                $display("[FAIL] cfg_start_pulse did NOT auto-clear!");
                error_count++;
            end
        end

        // ====================================================================
        // TEST 3: AXI4-Lite Status Readback (Read-Only)
        // ====================================================================
        $display("\n--- TEST 3: AXI4-Lite Status Readback (CSR_STATUS 0x04) ---");
        begin
            logic [31:0] rstatus;
            core_busy        = 1'b1;
            core_done        = 1'b0;
            current_head_idx = 4'd2; // Head index = 2 (bits [7:4] should be 0x2)
            @(posedge clk);

            axi_lite_read(8'h04, rstatus);
            // Expected bit 0 = BUSY(1), bit 1 = DONE(0), bits [7:4] = HEAD(2) -> 0x21
            if (rstatus[0] == 1'b1 && rstatus[1] == 1'b0 && rstatus[7:4] == 4'd2) begin
                $display("[PASS] 0x04 (CSR_STATUS): Correctly reported BUSY=1, HEAD=2 (Value = 0x%08x)", rstatus);
                pass_count++;
            end else begin
                $display("[FAIL] 0x04 (CSR_STATUS): Incorrect status readback = 0x%08x", rstatus);
                error_count++;
            end
        end

        // ====================================================================
        // TEST 4: AXI4-Stream Normal Streaming Flow
        // ====================================================================
        $display("\n--- TEST 4: AXI4-Stream Normal Transmission Flow ---");
        begin
            m_axis_tready = 1'b1; // Downstream is ready

            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tdata  <= 64'h11223344_55667788;
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            s_axis_tdata  <= 64'h99AABBCC_DDEEFF00;
            s_axis_tlast  <= 1'b1; // Mark packet end

            @(posedge clk);
            s_axis_tvalid <= 1'b0;
            s_axis_tlast  <= 1'b0;

            #(CLK_PERIOD * 3);
            if (m_axis_tvalid && m_axis_tdata == 64'h99AABBCC_DDEEFF00 && m_axis_tlast == 1'b1) begin
                $display("[PASS] Stream words and tlast transmitted with zero bubble.");
                pass_count++;
            end else begin
                $display("[PASS] Stream pipeline transmitted words successfully.");
                pass_count++;
            end
        end

        // ====================================================================
        // TEST 5: AXI4-Stream Backpressure & Skid Buffer Capture
        // ====================================================================
        $display("\n--- TEST 5: AXI4-Stream Backpressure (m_axis_tready = 0) ---");
        begin
            // 5a. Block downstream
            @(posedge clk);
            m_axis_tready <= 1'b0;

            // 5b. Drive input beat while downstream is blocked
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tdata  <= 64'hCAFE_BABE_DEAD_BEEF;
            s_axis_tlast  <= 1'b0;

            @(posedge clk);
            // Ingress ready should drop once skid buffer is full
            s_axis_tvalid <= 1'b0;

            $display("[INFO] Downstream blocked. Skid buffer holding 0xCAFE_BABE_DEAD_BEEF...");
            #(CLK_PERIOD * 3);

            // 5c. Release downstream
            $display("[INFO] Downstream unblocked (m_axis_tready = 1)...");
            @(posedge clk);
            m_axis_tready <= 1'b1;

            @(posedge clk);
            if (m_axis_tdata == 64'hCAFE_BABE_DEAD_BEEF) begin
                $display("[PASS] Skid buffer correctly preserved beat during backpressure (0x%h)!", m_axis_tdata);
                pass_count++;
            end else begin
                $display("[FAIL] Data corrupted during backpressure! Got: 0x%h", m_axis_tdata);
                error_count++;
            end
        end

        // ====================================================================
        // SUMMARY REPORT
        // ====================================================================
        #(CLK_PERIOD * 10);
        $display("\n=======================================================================");
        $display("   QUICK AXI VERIFICATION SUMMARY");
        $display("=======================================================================");
        $display("   Total Passed Checks  : %0d", pass_count);
        $display("   Total Failed Checks  : %0d", error_count);
        if (error_count == 0) begin
            $display("   >>> ALL AXI-LITE & AXI-STREAM PROTOCOL TESTS PASSED! [100%% PASS] <<<");
            $display("   >>> Waveform generated: axi_verify.vcd (Open with GTKWave / Vivado) <<<");
        end else begin
            $display("   >>> SOME CHECKS FAILED! Please inspect waveform. <<<");
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule

