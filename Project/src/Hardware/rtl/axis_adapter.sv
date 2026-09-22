`timescale 1ns / 1ps
// ============================================================================
// Module Name:   axis_adapter
// Description:  AXI4-Stream Skid Buffer to absorb backpressure when downstream
//               ready signal drops, preventing pipeline bubble and data loss.
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module axis_adapter #(
    parameter int DATA_WIDTH = 64
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // Ingress AXI-Stream
    input  logic [DATA_WIDTH-1:0] s_axis_tdata,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic                  s_axis_tlast,

    // Egress Internal Stream
    output logic [DATA_WIDTH-1:0] m_axis_tdata,
    output logic                  m_axis_tvalid,
    input  logic                  m_axis_tready,
    output logic                  m_axis_tlast
);

    logic [DATA_WIDTH-1:0] reg_main_data, reg_skid_data;
    logic                  reg_main_valid, reg_skid_valid;
    logic                  reg_main_last,  reg_skid_last;

    assign s_axis_tready = !reg_skid_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_main_valid <= 1'b0;
            reg_skid_valid <= 1'b0;
            reg_main_data  <= '0;
            reg_skid_data  <= '0;
            reg_main_last  <= 1'b0;
            reg_skid_last  <= 1'b0;
        end else begin
            if (s_axis_tready && s_axis_tvalid) begin
                if (m_axis_tready || !m_axis_tvalid) begin
                    reg_main_data  <= s_axis_tdata;
                    reg_main_valid <= 1'b1;
                    reg_main_last  <= s_axis_tlast;
                end else begin
                    reg_skid_data  <= s_axis_tdata;
                    reg_skid_valid <= 1'b1;
                    reg_skid_last  <= s_axis_tlast;
                end
            end else if (m_axis_tready) begin
                if (reg_skid_valid) begin
                    reg_main_data  <= reg_skid_data;
                    reg_main_valid <= 1'b1;
                    reg_main_last  <= reg_skid_last;
                    reg_skid_valid <= 1'b0;
                end else begin
                    reg_main_valid <= 1'b0;
                end
            end
        end
    end

    assign m_axis_tdata  = reg_main_data;
    assign m_axis_tvalid = reg_main_valid;
    assign m_axis_tlast  = reg_main_last;

endmodule
