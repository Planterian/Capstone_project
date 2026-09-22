`timescale 1ns / 1ps
// ============================================================================
// Module Name:   pe_slice_dsp_packing
// Description:  Processing Element (PE) Slice performing two simultaneous
//               signed INT8 multiplies (A*B and A*C) using a single DSP48E2
//               hard slice (18x27-bit multiplier) with sign-correction logic.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// ============================================================================

module pe_slice_dsp_packing (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic signed [7:0] operand_a,
    input  logic signed [7:0] operand_b,
    input  logic signed [7:0] operand_c,

    output logic        out_valid,
    output logic signed [15:0] result_ab,
    output logic signed [15:0] result_ac
);

    logic signed [17:0] operand_a_extended;
    logic signed [26:0] operand_b_extended;
    logic signed [26:0] operand_c_extended;
    logic signed [26:0] packed_operand;

    (* use_dsp = "yes" *)
    logic signed [44:0] dsp_product;

    logic signed [16:0] high_corrected;
    logic [1:0]        valid_pipe;

    assign operand_a_extended = {{10{operand_a}}, operand_a};
    assign operand_b_extended = {{19{operand_b}}, operand_b};
    assign operand_c_extended = {{19{operand_c}}, operand_c};

    assign packed_operand = ($signed(operand_b_extended) <<< 18) + $signed(operand_c_extended);

    assign high_corrected = $signed({dsp_product[44], dsp_product[33:18]}) + $signed({16'b0, dsp_product[15]});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dsp_product <= '0;
            result_ab   <= '0;
            result_ac   <= '0;
            valid_pipe  <= '0;
        end else begin
            dsp_product <= $signed(operand_a_extended) * $signed(packed_operand);
            result_ab   <= high_corrected[15:0];
            result_ac   <= dsp_product[15:0];
            valid_pipe  <= {valid_pipe[0], in_valid};
        end
    end

    assign out_valid = valid_pipe[1];

endmodule
