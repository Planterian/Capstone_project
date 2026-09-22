`timescale 1ns / 1ps
// ============================================================================
// Module Name:   axi_lite_regs
// Description:  AXI4-Lite Slave Register File (CSR) for Host CPU control
//               and configuration parameters (0x00..0x1C).
// Standard:     IEEE 1800-2012 SystemVerilog
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

    // Exported Control/Status Signals
    output logic                          cfg_start_pulse,
    output logic                          cfg_clear_buf,
    output logic [31:0]                   reg_nbytes,
    output logic [31:0]                   reg_tokens,
    output logic [31:0]                   reg_hdim,
    output logic [3:0]                    reg_shift_val,
    input  logic                          core_busy,
    input  logic                          core_done,
    input  logic                          core_error
);

    localparam [7:0] ADDR_CTRL   = 8'h00;
    localparam [7:0] ADDR_NBYTES = 8'h04;
    localparam [7:0] ADDR_TOKENS = 8'h10;
    localparam [7:0] ADDR_HDIM   = 8'h18;
    localparam [7:0] ADDR_SHIFT  = 8'h1C;

    logic [31:0] slv_reg_ctrl, slv_reg_nbytes, slv_reg_tokens, slv_reg_hdim, slv_reg_shift;
    logic axi_awready_reg, axi_wready_reg, axi_bvalid_reg, axi_arready_reg, axi_rvalid_reg;
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr_reg, axi_araddr_reg;

    assign S_AXI_AWREADY = axi_awready_reg;
    assign S_AXI_WREADY  = axi_wready_reg;
    assign S_AXI_BRESP   = 2'b00;
    assign S_AXI_BVALID  = axi_bvalid_reg;
    assign S_AXI_ARREADY = axi_arready_reg;
    assign S_AXI_RRESP   = 2'b00;
    assign S_AXI_RVALID  = axi_rvalid_reg;

    always_ff @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            axi_awready_reg <= 1'b0;
            axi_wready_reg  <= 1'b0;
            axi_bvalid_reg  <= 1'b0;
            slv_reg_ctrl    <= '0;
            slv_reg_nbytes  <= 32'd18816;
            slv_reg_tokens  <= 32'd196;
            slv_reg_hdim    <= 32'd32;
            slv_reg_shift   <= 32'd2;
        end else begin
            slv_reg_ctrl <= 1'b0;

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
                    ADDR_CTRL:   slv_reg_ctrl   <= S_AXI_WDATA;
                    ADDR_NBYTES: slv_reg_nbytes <= S_AXI_WDATA;
                    ADDR_TOKENS: slv_reg_tokens <= S_AXI_WDATA;
                    ADDR_HDIM:   slv_reg_hdim   <= S_AXI_WDATA;
                    ADDR_SHIFT:  slv_reg_shift  <= S_AXI_WDATA;
                    default: ;
                endcase
            end else if (S_AXI_BREADY && axi_bvalid_reg) begin
                axi_bvalid_reg <= 1'b0;
            end
        end
    end

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
                    ADDR_CTRL:   S_AXI_RDATA <= {28'b0, core_error, core_done, core_busy, 1'b0};
                    ADDR_NBYTES: S_AXI_RDATA <= slv_reg_nbytes;
                    ADDR_TOKENS: S_AXI_RDATA <= slv_reg_tokens;
                    ADDR_HDIM:   S_AXI_RDATA <= slv_reg_hdim;
                    ADDR_SHIFT:  S_AXI_RDATA <= slv_reg_shift;
                    default:     S_AXI_RDATA <= '0;
                endcase
            end else if (axi_rvalid_reg && S_AXI_RREADY) begin
                axi_rvalid_reg <= 1'b0;
            end
        end
    end

    assign cfg_start_pulse = slv_reg_ctrl[0];
    assign cfg_clear_buf   = slv_reg_ctrl[1];
    assign reg_nbytes      = slv_reg_nbytes;
    assign reg_tokens      = slv_reg_tokens;
    assign reg_hdim        = slv_reg_hdim;
    assign reg_shift_val   = slv_reg_shift[3:0];

endmodule
