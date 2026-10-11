`timescale 1ns / 1ps
// ============================================================================
// Module Name:   pe_slice_dsp_packing
// Description:  Processing Element (PE) Slice performing two simultaneous
//               signed INT8 multiplies (A*B and A*C) using a single DSP48E2
//               hard slice (18x27-bit multiplier) with sign-correction logic.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned)
// ============================================================================

module pe_slice_dsp_packing (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              in_valid,
    input  logic signed [7:0] operand_a,
    input  logic signed [7:0] operand_b,
    input  logic signed [7:0] operand_c,

    output logic              out_valid,
    output logic signed [15:0] result_ab,
    output logic signed [15:0] result_ac
);

    // 1. Bit-Accurate Sign Extension for DSP48E2 Inputs
    // Port A is 18-bit signed (replicate operand_a[7] 10 times)
    logic signed [17:0] operand_a_extended;
    // Port B is 27-bit signed (replicate operand_b[7] and operand_c[7] 19 times)
    logic signed [26:0] operand_b_extended;
    logic signed [26:0] operand_c_extended;
    logic signed [26:0] packed_operand;

    (* use_dsp = "yes" *)
    logic signed [44:0] dsp_product;
    logic [1:0]        valid_pipe;

    assign operand_a_extended = {{10{operand_a[7]}}, operand_a};
    assign operand_b_extended = {{19{operand_b[7]}}, operand_b};
    assign operand_c_extended = {{19{operand_c[7]}}, operand_c};

    // Pack B into bits [26:18] and C into bits [17:0]
    assign packed_operand = ($signed(operand_b_extended) <<< 18) + $signed(operand_c_extended);

    // 2. Synchronous Multiplier and Output Pipeline Registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dsp_product <= '0;
            result_ab   <= '0;
            result_ac   <= '0;
            valid_pipe  <= '0;
        end else begin
            // Stage 1: DSP Multiplication (A * (B*2^18 + C) = A*B*2^18 + A*C)
            dsp_product <= $signed(operand_a_extended) * $signed(packed_operand);

            // Stage 2: Post-Adder Sign Correction and Unpacking
            // If (A*C) is negative (indicated by bit 15), it borrowed 1 from the upper 18 bits.
            // Add 1 to compensate and restore bit-exact (A*B).
            result_ab   <= (dsp_product >>> 18) + (dsp_product[15] ? 16'sd1 : 16'sd0);
            result_ac   <= dsp_product[15:0];

            valid_pipe  <= {valid_pipe[0], in_valid};
        end
    end

    assign out_valid = valid_pipe[1];

endmodule
