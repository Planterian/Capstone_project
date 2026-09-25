`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_ping_pong_bram_buffer
// Description:  Automated Self-Checking Testbench for Ping-Pong BRAM Buffer
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_ping_pong_bram_buffer;

    localparam int DATA_WIDTH = 64;
    localparam int RAM_WIDTH  = 32;
    localparam int RAM_DEPTH  = 16;  // Fast simulation depth
    localparam int ADDR_WIDTH = 4;   // log2(16)
    localparam real CLK_PERIOD = 5.0; // 200 MHz

    logic                    clk;
    logic                    rst_n;

    logic [DATA_WIDTH-1:0]   s_axis_tdata;
    logic                    s_axis_tvalid;
    logic                    s_axis_tready;
    logic                    s_axis_tlast;
    logic [1:0]              ram_target_sel;

    logic                    read_en;
    logic [ADDR_WIDTH-1:0]   raddr_q, raddr_k, raddr_v;
    logic [RAM_WIDTH-1:0]    rdata_q_vec, rdata_k_vec, rdata_v_vec;

    logic                    tile_read_done;
    logic                    tile_write_done;
    logic                    bank_write_free;
    logic                    bank_read_ready;
    logic                    pp_bank_sel;

    int pass_count  = 0;
    int error_count = 0;

    // DUT Instantiation
    ping_pong_bram_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .RAM_WIDTH(RAM_WIDTH),
        .RAM_DEPTH(RAM_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .ram_target_sel(ram_target_sel),
        .read_en(read_en),
        .raddr_q(raddr_q),
        .raddr_k(raddr_k),
        .raddr_v(raddr_v),
        .rdata_q_vec(rdata_q_vec),
        .rdata_k_vec(rdata_k_vec),
        .rdata_v_vec(rdata_v_vec),
        .tile_read_done(tile_read_done),
        .tile_write_done(tile_write_done),
        .bank_write_free(bank_write_free),
        .bank_read_ready(bank_read_ready),
        .pp_bank_sel(pp_bank_sel)
    );

    // Clock Generator
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("=========================================================");
        $display("   STARTING TESTBENCH: ping_pong_bram_buffer (Level 1)");
        $display("=========================================================");

        rst_n          = 1'b0;
        s_axis_tdata   = '0;
        s_axis_tvalid  = 1'b0;
        s_axis_tlast   = 1'b0;
        ram_target_sel = 2'b00; // Q_RAM

        read_en        = 1'b0;
        raddr_q        = '0;
        raddr_k        = '0;
        raddr_v        = '0;
        tile_read_done = 1'b0;

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released. Initial pp_bank_sel = %0b", pp_bank_sel);

        // Step 1: Fill Bank 0 with Q_RAM vector tile (16 entries)
        $display("[TB INFO]: Step 1 - Writing 16 Q_RAM vectors into Bank 0...");
        ram_target_sel = 2'b00; // Q_RAM
        for (int i = 0; i < RAM_DEPTH; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            s_axis_tdata  <= 64'(32'hA000_0000 + i);
            if (i == RAM_DEPTH - 1) s_axis_tlast <= 1'b1;
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;

        wait (tile_write_done == 1'b1);
        $display("[TB INFO]: Bank 0 Write Complete! tile_write_done asserted.");

        #(CLK_PERIOD * 2);
        // Step 2: Verify Bank Pointer Swap
        if (bank_read_ready) begin
            $display("[PASS]: bank_read_ready is ASSERTED for Bank 0.");
            pass_count++;
        end else begin
            $display("[FAIL]: bank_read_ready failed to assert!");
            error_count++;
        end

        // Step 3: Trigger Bank Swap and Read Back Data
        $display("[TB INFO]: Step 3 - Triggering Bank Swap and reading from Bank 0...");
        @(posedge clk);
        tile_read_done <= 1'b1;
        @(posedge clk);
        tile_read_done <= 1'b0;

        #(CLK_PERIOD * 2);
        $display("[TB INFO]: Active pp_bank_sel after swap = %0b", pp_bank_sel);

        // Read address 5
        @(posedge clk);
        read_en <= 1'b1;
        raddr_q <= 4'd5;
        @(posedge clk);
        @(posedge clk); // Pipeline read delay
        read_en <= 1'b0;

        $display("[READ VERIFY]: Read Address 5 | Q_Vec = 0x%h | Expected = 0x%h",
                 rdata_q_vec, 32'hA000_0005);

        if (rdata_q_vec == 32'hA000_0005) begin
            $display("[PASS]: Data integrity verified across Ping-Pong BRAM!");
            pass_count++;
        end else begin
            $display("[FAIL]: Read data mismatch!");
            error_count++;
        end

        $display("=========================================================");
        if (error_count == 0)
            $display("   *** TEST PASSED SUCCESSFULLY ***");
        else
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        $display("=========================================================");
        $finish;
    end

endmodule
