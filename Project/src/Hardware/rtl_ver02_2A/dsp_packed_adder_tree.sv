`timescale 1ns / 1ps
// ============================================================================
// Module Name:   dsp_packed_adder_tree
// Description:  High-Throughput Vector Streaming GEMM 1 (Q * K^T) Engine
//               using 8 DSP48E2 hard slices (pe_slice_dsp_packing) and a
//               3-stage pipelined binary reduction adder tree.
//               Computes dot-products for 2 Key tokens (even and odd) in parallel
//               over 2 beats (8 INT8 multiplies per beat) and serializes them
//               into a continuous 1-score-per-cycle output stream.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Architecture: Variant 2A (High-Throughput DSP-Packed Streaming Engine)
// DSP Usage:    Exactly 8 DSP48E2
// ============================================================================

module dsp_packed_adder_tree #(
    parameter int HEAD_DIM  = 16,
    parameter int OUT_WIDTH = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,

    // Streaming Beat Interface
    input  logic                     in_valid,
    input  logic [1:0]               in_beat,       // 0: d=0..7, 1: d=8..15, 2: d=16..23, 3: d=24..31
    input  logic signed [7:0]        q_vec   [0:7], // 8 Query elements for current beat
    input  logic signed [7:0]        k_even  [0:7], // 8 Key Even elements for current beat
    input  logic signed [7:0]        k_odd   [0:7], // 8 Key Odd elements for current beat
    input  logic [15:0]              head_dim,

    // Serialized Output Stream (1 score / cycle sustained for dk=16)
    output logic                     out_valid,
    output logic signed [OUT_WIDTH-1:0] dot_product
);

    // ------------------------------------------------------------------------
    // 1. DYNAMIC HEAD_DIM MASKING FOR 8 PARALLEL DSP PACKING SLICES
    // ------------------------------------------------------------------------
    logic signed [7:0] op_a [0:7];
    logic signed [7:0] op_b [0:7];
    logic signed [7:0] op_c [0:7];
    logic              is_last_beat;

    assign is_last_beat = (head_dim <= 16'd16) ? (in_beat == 2'd1) :
                          (head_dim <= 16'd24) ? (in_beat == 2'd2) : (in_beat == 2'd3);

    always_comb begin
        for (int d = 0; d < 8; d++) begin
            int actual_dim;
            actual_dim = in_beat * 8 + d;
            if (actual_dim < head_dim) begin
                op_a[d] = q_vec[d];
                op_b[d] = k_even[d];
                op_c[d] = k_odd[d];
            end else begin
                op_a[d] = 8'sd0;
                op_b[d] = 8'sd0;
                op_c[d] = 8'sd0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // 2. STAGE 1 & 2: 8 INSTANCES OF PE_SLICE_DSP_PACKING (8 DSP48E2)
    //    Latency: 2 clock cycles (Multiplier + Unpack & Sign Correction)
    // ------------------------------------------------------------------------
    logic signed [15:0] pe_res_ab [0:7]; // A * B (Key Even partial products)
    logic signed [15:0] pe_res_ac [0:7]; // A * C (Key Odd partial products)
    logic [7:0]         pe_valid_out;

    genvar i;
    generate
        for (i = 0; i < 8; i++) begin : gen_dsp_pe
            pe_slice_dsp_packing u_dsp_pe (
                .clk        (clk),
                .rst_n      (rst_n),
                .in_valid   (in_valid),
                .operand_a  (op_a[i]),
                .operand_b  (op_b[i]),
                .operand_c  (op_c[i]),
                .out_valid  (pe_valid_out[i]),
                .result_ab  (pe_res_ab[i]),
                .result_ac  (pe_res_ac[i])
            );
        end
    endgenerate

    // Pipeline delay register for in_beat and is_last_beat tracking across DSP PE (2 cycles)
    logic [1:0] beat_dsp_pipe [0:1];
    logic [1:0] last_dsp_pipe;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat_dsp_pipe[0] <= 2'b00;
            beat_dsp_pipe[1] <= 2'b00;
            last_dsp_pipe    <= 2'b00;
        end else begin
            beat_dsp_pipe[0] <= in_beat;
            beat_dsp_pipe[1] <= beat_dsp_pipe[0];
            last_dsp_pipe    <= {last_dsp_pipe[0], is_last_beat};
        end
    end

    // ------------------------------------------------------------------------
    // 3. 3-STAGE PIPELINED BINARY ADDER TREE (8 -> 4 -> 2 -> 1)
    // ------------------------------------------------------------------------
    // Tree Stage 1 (8 to 4) - Latency +1 cycle (Total 3)
    logic signed [16:0] t1_even [0:3];
    logic signed [16:0] t1_odd  [0:3];
    logic               t1_valid;
    logic [1:0]         t1_beat;
    logic               t1_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            t1_valid <= 1'b0;
            t1_beat  <= 2'b00;
            t1_last  <= 1'b0;
            for (int k = 0; k < 4; k++) begin
                t1_even[k] <= '0;
                t1_odd[k]  <= '0;
            end
        end else begin
            t1_valid <= pe_valid_out[0];
            t1_beat  <= beat_dsp_pipe[1];
            t1_last  <= last_dsp_pipe[1];
            for (int k = 0; k < 4; k++) begin
                t1_even[k] <= $signed(pe_res_ab[2*k]) + $signed(pe_res_ab[2*k + 1]);
                t1_odd[k]  <= $signed(pe_res_ac[2*k]) + $signed(pe_res_ac[2*k + 1]);
            end
        end
    end

    // Tree Stage 2 (4 to 2) - Latency +1 cycle (Total 4)
    logic signed [17:0] t2_even [0:1];
    logic signed [17:0] t2_odd  [0:1];
    logic               t2_valid;
    logic [1:0]         t2_beat;
    logic               t2_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            t2_valid <= 1'b0;
            t2_beat  <= 2'b00;
            t2_last  <= 1'b0;
            for (int k = 0; k < 2; k++) begin
                t2_even[k] <= '0;
                t2_odd[k]  <= '0;
            end
        end else begin
            t2_valid <= t1_valid;
            t2_beat  <= t1_beat;
            t2_last  <= t1_last;
            for (int k = 0; k < 2; k++) begin
                t2_even[k] <= $signed(t1_even[2*k]) + $signed(t1_even[2*k + 1]);
                t2_odd[k]  <= $signed(t1_odd[2*k])  + $signed(t1_odd[2*k + 1]);
            end
        end
    end

    // Tree Stage 3 (2 to 1 Root Sum) - Latency +1 cycle (Total 5)
    logic signed [18:0] t3_even;
    logic signed [18:0] t3_odd;
    logic               t3_valid;
    logic [1:0]         t3_beat;
    logic               t3_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            t3_valid <= 1'b0;
            t3_beat  <= 2'b00;
            t3_last  <= 1'b0;
            t3_even  <= '0;
            t3_odd   <= '0;
        end else begin
            t3_valid <= t2_valid;
            t3_beat  <= t2_beat;
            t3_last  <= t2_last;
            t3_even  <= $signed(t2_even[0]) + $signed(t2_even[1]);
            t3_odd   <= $signed(t2_odd[0])  + $signed(t2_odd[1]);
        end
    end

    // ------------------------------------------------------------------------
    // 4. ACCUMULATION (MULTI-BEAT) & ZERO-BUBBLE SERIALIZER
    //    Latency +1 cycle (Total 6 cycles from in_valid to first out_valid)
    // ------------------------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] accum_even;
    logic signed [OUT_WIDTH-1:0] accum_odd;
    logic signed [OUT_WIDTH-1:0] odd_hold_reg;
    logic                        odd_pending;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_even   <= '0;
            accum_odd    <= '0;
            odd_hold_reg <= '0;
            odd_pending  <= 1'b0;
            out_valid    <= 1'b0;
            dot_product  <= '0;
        end else begin
            out_valid <= 1'b0;

            // Pipeline arrival at Tree Root
            if (t3_valid) begin
                if (t3_beat == 2'b00) begin
                    accum_even <= $signed({{13{t3_even[18]}}, t3_even});
                    accum_odd  <= $signed({{13{t3_odd[18]}}, t3_odd});
                end else if (!t3_last) begin
                    accum_even <= accum_even + $signed({{13{t3_even[18]}}, t3_even});
                    accum_odd  <= accum_odd  + $signed({{13{t3_odd[18]}}, t3_odd});
                end

                if (t3_last) begin
                    if (t3_beat == 2'b00) begin
                        dot_product  <= $signed({{13{t3_even[18]}}, t3_even});
                        odd_hold_reg <= $signed({{13{t3_odd[18]}}, t3_odd});
                    end else begin
                        dot_product  <= accum_even + $signed({{13{t3_even[18]}}, t3_even});
                        odd_hold_reg <= accum_odd  + $signed({{13{t3_odd[18]}}, t3_odd});
                    end
                    out_valid   <= 1'b1;
                    odd_pending <= 1'b1;
                end
            end

            // Zero-bubble serializer for Odd Token Score
            if (odd_pending && (!t3_valid || !t3_last)) begin
                dot_product <= odd_hold_reg;
                out_valid   <= 1'b1;
                odd_pending <= 1'b0;
            end
        end
    end

endmodule
