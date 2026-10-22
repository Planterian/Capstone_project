`timescale 1ns / 1ps
// ============================================================================
// Module Name:   axi_lite_regs
// Description:  AXI4-Lite Slave Register File (CSR) for Host CPU / PYNQ control.
//               Supports dynamic sequence length (N=256), head dim (dk=16,20,24),
//               dyadic scaling (M, E), and sequential head monitoring.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned)
// ============================================================================

module axi_lite_regs #(
    parameter int C_S_AXI_DATA_WIDTH = 32,
    parameter int C_S_AXI_ADDR_WIDTH = 8
)(
    input  logic                          S_AXI_ACLK,
    input  logic                          S_AXI_ARESETN,

    // AXI4-Lite Slave Ports
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  logic                          S_AXI_AWVALID,
    output logic                          S_AXI_AWREADY,
    input  logic [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  logic                          S_AXI_WVALID,
    output logic                          S_AXI_WREADY,
    output logic [1:0]                    S_AXI_BRESP,
    output logic                          S_AXI_BVALID,
    input  logic                          S_AXI_BREADY,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  logic                          S_AXI_ARVALID,
    output logic                          S_AXI_ARREADY,
    output logic [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output logic [1:0]                    S_AXI_RRESP,
    output logic                          S_AXI_RVALID,
    input  logic                          S_AXI_RREADY,

    // Exported Configuration Registers to Core
    output logic                          cfg_start_pulse,
    output logic                          cfg_clear_buf,
    output logic                          cfg_soft_reset,
    output logic [31:0]                   reg_nbytes,
    output logic [31:0]                   reg_seq_len,
    output logic [31:0]                   reg_head_dim,
    output logic [31:0]                   reg_num_heads,
    output logic signed [15:0]            reg_scale_m,
    output logic [4:0]                    reg_scale_e,
    output logic [3:0]                    reg_shift_val,
    output logic signed [15:0]            reg_out_m,
    output logic [4:0]                    reg_out_e,

    // Step 2.5: BRAM LUT Reload Ports (to softmax_lut Port B)
    output logic                          cfg_lut_we,
    output logic [9:0]                    cfg_lut_addr,
    output logic [19:0]                   cfg_lut_wdata,

    // Status Inputs from Core
    input  logic                          core_busy,
    input  logic                          core_done,
    input  logic                          core_error,
    input  logic                          pp_bank_sel,
    input  logic [3:0]                    current_head_idx
);

    // CSR Address Map (Word aligned, byte offsets)
    localparam [7:0] ADDR_CTRL      = 8'h00; // Bit 0: START, 1: CLEAR, 2: RESET
    localparam [7:0] ADDR_STATUS    = 8'h04; // Status readback
    localparam [7:0] ADDR_NBYTES    = 8'h08; // Total byte count
    localparam [7:0] ADDR_SEQ_LEN   = 8'h10; // Sequence length N (default 256)
    localparam [7:0] ADDR_HEAD_DIM  = 8'h14; // Head dimension dk (16, 20, 24)
    localparam [7:0] ADDR_NUM_HEADS = 8'h18; // Number of heads H (default 4)
    localparam [7:0] ADDR_SHIFT_VAL = 8'h1C; // Legacy ASR shift
    localparam [7:0] ADDR_SCALE_M   = 8'h20; // Dyadic M (16-bit signed)
    localparam [7:0] ADDR_SCALE_E   = 8'h24; // Dyadic E (5-bit unsigned)
    localparam [7:0] ADDR_OUT_M     = 8'h28; // Output M
    localparam [7:0] ADDR_OUT_E     = 8'h2C; // Output E
    localparam [7:0] ADDR_LUT_ADDR  = 8'h40; // Step 2.5: BRAM LUT Address [9:0] (0..1023)
    localparam [7:0] ADDR_LUT_DATA  = 8'h44; // Step 2.5: BRAM LUT Data [19:0] (Triggers 1-cycle write pulse)

    // Internal Register State
    logic [31:0] slv_ctrl_reg;
    logic [31:0] slv_nbytes_reg;
    logic [31:0] slv_seq_len_reg;
    logic [31:0] slv_head_dim_reg;
    logic [31:0] slv_num_heads_reg;
    logic [31:0] slv_shift_val_reg;
    logic [31:0] slv_scale_m_reg;
    logic [31:0] slv_scale_e_reg;
    logic [31:0] slv_out_m_reg;
    logic [31:0] slv_out_e_reg;
    logic [31:0] slv_lut_addr_reg;
    logic [31:0] slv_lut_data_reg;
    logic        lut_we_pulse;

    logic axi_awready_reg, axi_wready_reg, axi_bvalid_reg, axi_arready_reg, axi_rvalid_reg;
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr_reg, axi_araddr_reg;

    assign S_AXI_AWREADY = axi_awready_reg;
    assign S_AXI_WREADY  = axi_wready_reg;
    assign S_AXI_BRESP   = 2'b00;
    assign S_AXI_BVALID  = axi_bvalid_reg;
    assign S_AXI_ARREADY = axi_arready_reg;
    assign S_AXI_RRESP   = 2'b00;
    assign S_AXI_RVALID  = axi_rvalid_reg;

    // AXI4-Lite Write Channel
    always_ff @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            axi_awready_reg   <= 1'b0;
            axi_wready_reg    <= 1'b0;
            axi_bvalid_reg    <= 1'b0;
            axi_awaddr_reg    <= '0;
            slv_ctrl_reg      <= '0;
            slv_nbytes_reg    <= 32'd24576; // Default: 256 * 32 * 3
            slv_seq_len_reg   <= 32'd256;   // N = 256 tokens
            slv_head_dim_reg  <= 32'd16;    // dk = 16 (Stage 2 default)
            slv_num_heads_reg <= 32'd4;     // H = 4
            slv_shift_val_reg <= 32'd2;     // ASR shift = 2
            slv_scale_m_reg   <= 32'd27640; // Stage 2 M (Bit-Exact Calibrated)
            slv_scale_e_reg   <= 32'd22;    // Stage 2 E (Bit-Exact Calibrated)
            slv_out_m_reg     <= 32'd1;
            slv_out_e_reg     <= 32'd8;
            slv_lut_addr_reg  <= '0;
            slv_lut_data_reg  <= '0;
            lut_we_pulse      <= 1'b0;
        end else begin
            // Clear auto-pulsed bits
            slv_ctrl_reg[0] <= 1'b0; // start_pulse self-clearing
            lut_we_pulse    <= 1'b0; // cfg_lut_we self-clearing 1-cycle pulse

            if (~axi_awready_reg && S_AXI_AWVALID && S_AXI_WVALID) begin
                axi_awready_reg <= 1'b1;
                axi_wready_reg  <= 1'b1;
                axi_awaddr_reg  <= S_AXI_AWADDR;
            end else begin
                axi_awready_reg <= 1'b0;
                axi_wready_reg  <= 1'b0;
            end

            if (axi_awready_reg && S_AXI_AWVALID && axi_wready_reg && S_AXI_WVALID) begin
                axi_bvalid_reg <= 1'b1;
                case (axi_awaddr_reg)
                    ADDR_CTRL:      slv_ctrl_reg      <= S_AXI_WDATA;
                    ADDR_NBYTES:    slv_nbytes_reg    <= S_AXI_WDATA;
                    ADDR_SEQ_LEN:   slv_seq_len_reg   <= S_AXI_WDATA;
                    ADDR_HEAD_DIM:  slv_head_dim_reg  <= S_AXI_WDATA;
                    ADDR_NUM_HEADS: slv_num_heads_reg <= S_AXI_WDATA;
                    ADDR_SHIFT_VAL: slv_shift_val_reg <= S_AXI_WDATA;
                    ADDR_SCALE_M:   slv_scale_m_reg   <= S_AXI_WDATA;
                    ADDR_SCALE_E:   slv_scale_e_reg   <= S_AXI_WDATA;
                    ADDR_OUT_M:     slv_out_m_reg     <= S_AXI_WDATA;
                    ADDR_OUT_E:     slv_out_e_reg     <= S_AXI_WDATA;
                    ADDR_LUT_ADDR:  slv_lut_addr_reg  <= S_AXI_WDATA;
                    ADDR_LUT_DATA:  begin
                        slv_lut_data_reg <= S_AXI_WDATA;
                        lut_we_pulse     <= 1'b1; // Pulse WE for 1 cycle when data is written
                    end
                    default: ;
                endcase
            end else if (S_AXI_BREADY && axi_bvalid_reg) begin
                axi_bvalid_reg <= 1'b0;
            end
        end
    end

    // AXI4-Lite Read Channel
    always_ff @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            axi_arready_reg <= 1'b0;
            axi_rvalid_reg  <= 1'b0;
            S_AXI_RDATA     <= '0;
        end else begin
            if (~axi_arready_reg && S_AXI_ARVALID) begin
                axi_arready_reg <= 1'b1;
                axi_araddr_reg  <= S_AXI_ARADDR;
            end else begin
                axi_arready_reg <= 1'b0;
            end

            if (axi_arready_reg && S_AXI_ARVALID && ~axi_rvalid_reg) begin
                axi_rvalid_reg <= 1'b1;
                case (axi_araddr_reg)
                    ADDR_CTRL:      S_AXI_RDATA <= slv_ctrl_reg;
                    ADDR_STATUS:    S_AXI_RDATA <= {24'b0, current_head_idx, pp_bank_sel, core_error, core_done, core_busy};
                    ADDR_NBYTES:    S_AXI_RDATA <= slv_nbytes_reg;
                    ADDR_SEQ_LEN:   S_AXI_RDATA <= slv_seq_len_reg;
                    ADDR_HEAD_DIM:  S_AXI_RDATA <= slv_head_dim_reg;
                    ADDR_NUM_HEADS: S_AXI_RDATA <= slv_num_heads_reg;
                    ADDR_SHIFT_VAL: S_AXI_RDATA <= slv_shift_val_reg;
                    ADDR_SCALE_M:   S_AXI_RDATA <= slv_scale_m_reg;
                    ADDR_SCALE_E:   S_AXI_RDATA <= slv_scale_e_reg;
                    ADDR_OUT_M:     S_AXI_RDATA <= slv_out_m_reg;
                    ADDR_OUT_E:     S_AXI_RDATA <= slv_out_e_reg;
                    ADDR_LUT_ADDR:  S_AXI_RDATA <= slv_lut_addr_reg;
                    ADDR_LUT_DATA:  S_AXI_RDATA <= slv_lut_data_reg;
                    default:        S_AXI_RDATA <= 32'hDEADBEEF;
                endcase
            end else if (axi_rvalid_reg && S_AXI_RREADY) begin
                axi_rvalid_reg <= 1'b0;
            end
        end
    end

    // Exported Control Signals
    assign cfg_start_pulse = slv_ctrl_reg[0];
    assign cfg_clear_buf   = slv_ctrl_reg[1];
    assign cfg_soft_reset  = slv_ctrl_reg[2];
    assign reg_nbytes      = slv_nbytes_reg;
    assign reg_seq_len     = slv_seq_len_reg;
    assign reg_head_dim    = slv_head_dim_reg;
    assign reg_num_heads   = slv_num_heads_reg;
    assign reg_shift_val   = slv_shift_val_reg[3:0];
    assign reg_scale_m     = slv_scale_m_reg[15:0];
    assign reg_scale_e     = slv_scale_e_reg[4:0];
    assign reg_out_m       = slv_out_m_reg[15:0];
    assign reg_out_e       = slv_out_e_reg[4:0];
    assign cfg_lut_we      = lut_we_pulse;
    assign cfg_lut_addr    = slv_lut_addr_reg[9:0];
    assign cfg_lut_wdata   = slv_lut_data_reg[19:0];

endmodule
