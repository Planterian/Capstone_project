`timescale 1ns / 1ps
// ============================================================================
// Module Name:   score_v_gemm
// Description:  STAGE 4: Score x V Context Aggregation GEMM Engine.
//               Computes Context = Softmax(Prob) * V, scales down (>>> 8),
//               saturates to INT8 [-128, +127], and packs into 64-bit AXI-Stream beats.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned - Matches Golden Context Out)
`ifndef SCORE_V_GEMM_SV
`define SCORE_V_GEMM_SV

module score_v_gemm #(
    parameter int MAX_TOKENS   = 256,
    parameter int MAX_HEAD_DIM = 32,
    parameter int PROB_WIDTH   = 16
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // Interface from Stage 3: Hardware Softmax
    input  logic                          start_row,
    input  logic                          prob_valid,
    input  logic [PROB_WIDTH-1:0]         prob_data,     // UINT16 [0, 32767]
    input  logic                          prob_last,     // High on token 255 of row

    // Vector V[k, :] input for token k
    input  logic signed [7:0]             v_vec [0:MAX_HEAD_DIM-1],
    input  logic [15:0]                   head_dim,      // 16, 20, 24, 32

    // Interface to AXI4-Stream TX Egress
    output logic                          m_axis_tvalid,
    output logic [63:0]                   m_axis_tdata,
    output logic                          m_axis_tlast,
    input  logic                          m_axis_tready,

    output logic [8:0]                    row_token_cnt,
    output logic                          row_done
);

    // 32-bit Accumulator per feature dimension
    logic signed [31:0] ctx_acc [0:MAX_HEAD_DIM-1];
    logic signed [7:0]  ctx_clamped [0:MAX_HEAD_DIM-1];
    logic [8:0]         row_cnt;

    initial begin
        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
            ctx_acc[d] = '0;
        end
    end

    typedef enum logic [2:0] {
        ST_ACCUMULATE = 3'b000,
        ST_OUT_BEAT0  = 3'b001,
        ST_OUT_BEAT1  = 3'b010,
        ST_OUT_BEAT2  = 3'b011,
        ST_OUT_BEAT3  = 3'b100
    } gemm2_state_t;

    gemm2_state_t state_reg;

    // INT8 Clamping after arithmetic shift right by 15 bits (matching sum of 16-bit probs ~ 32768)
    genvar j;
    generate
        for (j = 0; j < MAX_HEAD_DIM; j++) begin : gen_clamp
            logic signed [23:0] shifted_val;
            assign shifted_val = ctx_acc[j] >>> 15;

            always_comb begin
                if (shifted_val > 24'sd127)
                    ctx_clamped[j] = 8'sd127;
                else if (shifted_val < -24'sd128)
                    ctx_clamped[j] = -8'sd128;
                else
                    ctx_clamped[j] = 8'(shifted_val);
            end
        end
    endgenerate

    logic signed [7:0] ctx_safe [0:31];
    always_comb begin
        for (int k = 0; k < 32; k++) begin
            if (k < MAX_HEAD_DIM)
                ctx_safe[k] = ctx_clamped[k];
            else
                ctx_safe[k] = 8'sd0;
        end
    end

    // Combinational AXI4-Stream outputs decoded directly from state
    assign m_axis_tvalid = (state_reg == ST_OUT_BEAT0) ||
                           (state_reg == ST_OUT_BEAT1) ||
                           (state_reg == ST_OUT_BEAT2) ||
                           (state_reg == ST_OUT_BEAT3);

    always_comb begin
        case (state_reg)
            ST_OUT_BEAT0: begin
                m_axis_tdata = {ctx_safe[7], ctx_safe[6], ctx_safe[5], ctx_safe[4],
                                ctx_safe[3], ctx_safe[2], ctx_safe[1], ctx_safe[0]};
                m_axis_tlast = (head_dim <= 16'd8) && (row_cnt == MAX_TOKENS - 1);
            end
            ST_OUT_BEAT1: begin
                m_axis_tdata = {ctx_safe[15], ctx_safe[14], ctx_safe[13], ctx_safe[12],
                                ctx_safe[11], ctx_safe[10], ctx_safe[9],  ctx_safe[8]};
                m_axis_tlast = (head_dim <= 16'd16) && (row_cnt == MAX_TOKENS - 1);
            end
            ST_OUT_BEAT2: begin
                m_axis_tdata = {ctx_safe[23], ctx_safe[22], ctx_safe[21], ctx_safe[20],
                                ctx_safe[19], ctx_safe[18], ctx_safe[17], ctx_safe[16]};
                m_axis_tlast = (head_dim <= 16'd24) && (row_cnt == MAX_TOKENS - 1);
            end
            ST_OUT_BEAT3: begin
                m_axis_tdata = {ctx_safe[31], ctx_safe[30], ctx_safe[29], ctx_safe[28],
                                ctx_safe[27], ctx_safe[26], ctx_safe[25], ctx_safe[24]};
                m_axis_tlast = (row_cnt == MAX_TOKENS - 1);
            end
            default: begin
                m_axis_tdata = 64'd0;
                m_axis_tlast = 1'b0;
            end
        endcase
    end

    // Accumulation and Multi-Token Egress FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg <= ST_ACCUMULATE;
            row_cnt   <= '0;
            row_done  <= 1'b0;
            for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
        end else begin
            row_done <= 1'b0;

            case (state_reg)
                ST_ACCUMULATE: begin
                    if (start_row) begin
                        for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                    end else if (prob_valid) begin
                        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
                            ctx_acc[d] <= ctx_acc[d] + (32'($signed({1'b0, prob_data})) * 32'($signed(v_vec[d])));
                        end

                        if (prob_last) begin
                            state_reg <= ST_OUT_BEAT0;
                        end
                    end
                end

                ST_OUT_BEAT0: begin
                    if (m_axis_tready) begin
                        if (head_dim <= 16'd8) begin
                            row_done <= 1'b1;
                            if (row_cnt == MAX_TOKENS - 1) row_cnt <= '0;
                            else row_cnt <= row_cnt + 1'b1;
                            for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                            state_reg <= ST_ACCUMULATE;
                        end else begin
                            state_reg <= ST_OUT_BEAT1;
                        end
                    end
                end

                ST_OUT_BEAT1: begin
                    if (m_axis_tready) begin
                        if (head_dim <= 16'd16) begin
                            row_done <= 1'b1;
                            if (row_cnt == MAX_TOKENS - 1) row_cnt <= '0;
                            else row_cnt <= row_cnt + 1'b1;
                            for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                            state_reg <= ST_ACCUMULATE;
                        end else begin
                            state_reg <= ST_OUT_BEAT2;
                        end
                    end
                end

                ST_OUT_BEAT2: begin
                    if (m_axis_tready) begin
                        if (head_dim <= 16'd24) begin
                            row_done <= 1'b1;
                            if (row_cnt == MAX_TOKENS - 1) row_cnt <= '0;
                            else row_cnt <= row_cnt + 1'b1;
                            for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                            state_reg <= ST_ACCUMULATE;
                        end else begin
                            state_reg <= ST_OUT_BEAT3;
                        end
                    end
                end

                ST_OUT_BEAT3: begin
                    if (m_axis_tready) begin
                        row_done <= 1'b1;
                        if (row_cnt == MAX_TOKENS - 1) row_cnt <= '0;
                        else row_cnt <= row_cnt + 1'b1;
                        for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                        state_reg <= ST_ACCUMULATE;
                    end
                end

                default: state_reg <= ST_ACCUMULATE;
            endcase
        end
    end

    assign row_token_cnt = row_cnt;

endmodule

`endif // SCORE_V_GEMM_SV
