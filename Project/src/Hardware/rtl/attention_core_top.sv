`timescale 1ns / 1ps
// ============================================================================
// Module Name:   attention_core_top
// Description:  Top-Level Shell for Multi-Head Attention (MHA) Accelerator.
//               Integrates AXI4-Lite Control Slave, AXI4-Stream Ingress/Egress,
//               Skid Buffer, Ping-Pong BRAM Buffer, Output-Stationary Systolic
//               MAC Array, Scaler Unit, Hardware Softmax, and Re-quantization.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// ============================================================================

module attention_core_top #(
    parameter int C_S_AXI_DATA_WIDTH = 32,
    parameter int C_S_AXI_ADDR_WIDTH = 8,
    parameter int C_AXIS_DATA_WIDTH  = 64,
    parameter int ARRAY_SIZE         = 4,
    parameter int INT8_PER_WORD      = 8
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // ------------------------------------------------------------------------
    // 1. AXI4-Lite Slave Interface (Control & Status Plane)
    // ------------------------------------------------------------------------
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                          s_axi_awvalid,
    output logic                          s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  logic                          s_axi_wvalid,
    output logic                          s_axi_wready,
    output logic [1:0]                    s_axi_bresp,
    output logic                          s_axi_bvalid,
    input  logic                          s_axi_bready,
    
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                          s_axi_arvalid,
    output logic                          s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0]                    s_axi_rresp,
    output logic                          s_axi_rvalid,
    input  logic                          s_axi_rready,

    // ------------------------------------------------------------------------
    // 2. AXI4-Stream RX Ingress Interface (Read DDR4 -> PL)
    // ------------------------------------------------------------------------
    input  logic [C_AXIS_DATA_WIDTH-1:0]  s_axis_tdata,
    input  logic [(C_AXIS_DATA_WIDTH/8)-1:0] s_axis_tkeep,
    input  logic                          s_axis_tlast,
    input  logic                          s_axis_tvalid,
    output logic                          s_axis_tready,

    // ------------------------------------------------------------------------
    // 3. AXI4-Stream TX Egress Interface (PL -> Write DDR4)
    // ------------------------------------------------------------------------
    output logic [C_AXIS_DATA_WIDTH-1:0]  m_axis_tdata,
    output logic [(C_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep,
    output logic                          m_axis_tlast,
    output logic                          m_axis_tvalid,
    input  logic                          m_axis_tready
);

    // ------------------------------------------------------------------------
    // CSR REGISTER WIRES & CONTROL SIGNALS
    // ------------------------------------------------------------------------
    logic [31:0] reg_ctrl;
    logic [31:0] reg_nbytes;
    logic [31:0] reg_seq_len;
    logic [31:0] reg_head_dim;
    logic [31:0] reg_heads;
    logic [31:0] reg_shift_val;

    logic        start_pulse;
    logic        clear_buf_pulse;
    logic        core_busy;
    logic        core_done;
    logic        core_error;

    // AXI4-Lite Internal Handshake Logic
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr_reg;
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr_reg;
    logic                          axi_awready_reg, axi_wready_reg, axi_bvalid_reg;
    logic                          axi_arready_reg, axi_rvalid_reg;
    logic [C_S_AXI_DATA_WIDTH-1:0] axi_rdata_reg;

    assign s_axi_awready = axi_awready_reg;
    assign s_axi_wready  = axi_wready_reg;
    assign s_axi_bresp   = 2'b00; // OKAY
    assign s_axi_bvalid  = axi_bvalid_reg;
    assign s_axi_arready = axi_arready_reg;
    assign s_axi_rdata   = axi_rdata_reg;
    assign s_axi_rresp   = 2'b00; // OKAY
    assign s_axi_rvalid  = axi_rvalid_reg;

    // AXI4-Lite Write Address & Data Decoding
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_awready_reg <= 1'b0;
            axi_wready_reg  <= 1'b0;
            axi_bvalid_reg  <= 1'b0;
            axi_awaddr_reg  <= '0;
            reg_ctrl        <= '0;
            reg_nbytes      <= 32'd18816; // Default Q,K,V packet bytes
            reg_seq_len     <= 32'd196;   // N = 196 tokens
            reg_head_dim    <= 32'd32;    // d_k = 32
            reg_heads       <= 32'd2;     // H = 2
            reg_shift_val   <= 32'd2;     // ASR shift = 2
            start_pulse     <= 1'b0;
            clear_buf_pulse <= 1'b0;
        end else begin
            start_pulse     <= 1'b0;
            clear_buf_pulse <= 1'b0;

            // AWREADY & WREADY Handshake
            if (!axi_awready_reg && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready_reg <= 1'b1;
                axi_wready_reg  <= 1'b1;
                axi_awaddr_reg  <= s_axi_awaddr;
            end else begin
                axi_awready_reg <= 1'b0;
                axi_wready_reg  <= 1'b0;
            end

            // Register Write Decoding
            if (axi_awready_reg && s_axi_awvalid && axi_wready_reg && s_axi_wvalid) begin
                axi_bvalid_reg <= 1'b1;
                case (axi_awaddr_reg[5:0])
                    6'h00: begin
                        if (s_axi_wdata[0]) start_pulse     <= 1'b1;
                        if (s_axi_wdata[1]) clear_buf_pulse <= 1'b1;
                    end
                    6'h04: reg_nbytes    <= s_axi_wdata;
                    6'h10: reg_seq_len   <= s_axi_wdata;
                    6'h18: reg_head_dim  <= s_axi_wdata;
                    6'h1C: reg_heads     <= s_axi_wdata;
                    6'h20: reg_shift_val <= s_axi_wdata;
                    default: ;
                endcase
            end else if (s_axi_bready && axi_bvalid_reg) begin
                axi_bvalid_reg <= 1'b0;
            end
        end
    end

    // AXI4-Lite Read Address & Data Decoding
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_arready_reg <= 1'b0;
            axi_rvalid_reg  <= 1'b0;
            axi_rdata_reg   <= '0;
        end else begin
            if (!axi_arready_reg && s_axi_arvalid) begin
                axi_arready_reg <= 1'b1;
                axi_araddr_reg  <= s_axi_araddr;
            end else begin
                axi_arready_reg <= 1'b0;
            end

            if (axi_arready_reg && s_axi_arvalid && !axi_rvalid_reg) begin
                axi_rvalid_reg <= 1'b1;
                case (axi_araddr_reg[5:0])
                    6'h00: axi_rdata_reg <= {13'b0, core_error, core_done, core_busy, 16'b0};
                    6'h04: axi_rdata_reg <= reg_nbytes;
                    6'h10: axi_rdata_reg <= reg_seq_len;
                    6'h18: axi_rdata_reg <= reg_head_dim;
                    6'h1C: axi_rdata_reg <= reg_heads;
                    6'h20: axi_rdata_reg <= reg_shift_val;
                    default: axi_rdata_reg <= 32'hDEADBEEF;
                endcase
            end else if (s_axi_rready && axi_rvalid_reg) begin
                axi_rvalid_reg <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // INTERMEDIATE STREAMING & PIPELINE WIRES
    // ------------------------------------------------------------------------
    // Skid Buffer Egress Stream
    logic [C_AXIS_DATA_WIDTH-1:0] skid_tdata;
    logic                         skid_tvalid, skid_tready, skid_tlast;

    // Ping-Pong BRAM Egress to Systolic MAC Array
    logic [31:0] rdata_q_vec, rdata_k_vec, rdata_v_vec;
    logic        bram_read_en;
    logic [7:0]  raddr_q, raddr_k, raddr_v;
    logic        tile_read_done, tile_write_done, pp_bank_sel;

    // Systolic MAC Array Outputs
    logic signed [31:0] dot_product_out [0:ARRAY_SIZE-1];
    logic               systolic_out_valid;

    // Scaler Unit Outputs
    logic signed [15:0] scaled_score_out;
    logic               scaler_out_valid;

    // Softmax Engine Outputs
    logic [7:0]         softmax_prob_out;
    logic               softmax_out_valid;

    // ------------------------------------------------------------------------
    // MODULE INSTANTIATIONS
    // ------------------------------------------------------------------------

    // 1. AXI4-Stream Ingress Skid Buffer Adapter
    axis_adapter #(
        .DATA_WIDTH (C_AXIS_DATA_WIDTH)
    ) u_skid_buffer_rx (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tdata  (skid_tdata),
        .m_axis_tvalid (skid_tvalid),
        .m_axis_tready (skid_tready),
        .m_axis_tlast  (skid_tlast)
    );

    // 2. Ping-Pong BRAM Buffer Subsystem
    ping_pong_bram_buffer #(
        .DATA_WIDTH (C_AXIS_DATA_WIDTH),
        .RAM_WIDTH  (32),
        .RAM_DEPTH  (256),
        .ADDR_WIDTH (8)
    ) u_ping_pong_buffer (
        .clk             (clk),
        .rst_n           (rst_n),
        .s_axis_tdata    (skid_tdata),
        .s_axis_tvalid   (skid_tvalid),
        .s_axis_tready   (skid_tready),
        .s_axis_tlast    (skid_tlast),
        .ram_target_sel  (2'b00), // Default Q/K/V multiplexed
        .read_en         (bram_read_en),
        .raddr_q         (raddr_q),
        .raddr_k         (raddr_k),
        .raddr_v         (raddr_v),
        .rdata_q_vec     (rdata_q_vec),
        .rdata_k_vec     (rdata_k_vec),
        .rdata_v_vec     (rdata_v_vec),
        .tile_read_done  (tile_read_done),
        .tile_write_done (tile_write_done),
        .bank_write_free (),
        .bank_read_ready (),
        .pp_bank_sel     (pp_bank_sel)
    );

    // 3. Compute Engine: Systolic MAC Array
    systolic_mac_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .IN_WIDTH   (8),
        .ACC_WIDTH  (32)
    ) u_systolic_mac_array (
        .clk             (clk),
        .rst_n           (rst_n),
        .gemm_start      (start_pulse),
        .clr_acc         (clear_buf_pulse),
        .in_valid        (skid_tvalid),
        .mode_sel        (1'b0), // Stage 1 QK^T
        .act_row         ('{rdata_q_vec[7:0], rdata_q_vec[15:8], rdata_q_vec[23:16], rdata_q_vec[31:24]}),
        .wt_col0         ('{rdata_k_vec[7:0], rdata_k_vec[15:8], rdata_k_vec[23:16], rdata_k_vec[31:24]}),
        .wt_col1         ('{rdata_v_vec[7:0], rdata_v_vec[15:8], rdata_v_vec[23:16], rdata_v_vec[31:24]}),
        .out_valid       (systolic_out_valid),
        .dot_product_out (dot_product_out)
    );

    // 4. Scaler Unit (0 DSPs Barrel Shifter)
    scale_unit #(
        .IN_WIDTH  (32),
        .OUT_WIDTH (16)
    ) u_scale_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (systolic_out_valid),
        .in_score     (dot_product_out[0]),
        .shift_val    (reg_shift_val[3:0]),
        .out_valid    (scaler_out_valid),
        .scaled_score (scaled_score_out)
    );

    // 5. Hardware Softmax Engine
    softmax_lut #(
        .N_TOKENS  (196),
        .IN_WIDTH  (16),
        .OUT_WIDTH (8)
    ) u_softmax_engine (
        .clk        (clk),
        .rst_n      (rst_n),
        .start_row  (scaler_out_valid),
        .in_valid   (scaler_out_valid),
        .in_score   (scaled_score_out),
        .out_valid  (softmax_out_valid),
        .out_prob   (softmax_prob_out),
        .busy       (core_busy)
    );

    // 6. Egress Streaming Logic
    assign m_axis_tdata  = {56'b0, softmax_prob_out};
    assign m_axis_tkeep  = 8'hFF;
    assign m_axis_tlast  = softmax_out_valid;
    assign m_axis_tvalid = softmax_out_valid;

    assign core_done  = softmax_out_valid;
    assign core_error = 1'b0;

endmodule
