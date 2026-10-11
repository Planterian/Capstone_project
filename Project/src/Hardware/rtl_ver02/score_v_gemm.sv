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
    parameter int MAX_HEAD_DIM = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // Interface from Stage 3: Hardware Softmax
    input  logic                          start_row,
    input  logic                          prob_valid,
    input  logic [7:0]                    prob_data,     // UINT8 [0, 255]
    input  logic                          prob_last,     // High on token 255 of row

    // Vector V[k, :] input for token k
    input  logic signed [7:0]             v_vec [0:MAX_HEAD_DIM-1],
    input  logic [15:0]                   head_dim,      // 16, 20, 24, 32

    // Interface to AXI4-Stream TX Egress
    output logic                          m_axis_tvalid,
    output logic [63:0]                   m_axis_tdata,
    output logic                          m_axis_tlast,
    input  logic                          m_axis_tready
);

    // 32-bit Accumulator per feature dimension
    logic signed [31:0] ctx_acc [0:MAX_HEAD_DIM-1];
    logic signed [7:0]  ctx_clamped [0:MAX_HEAD_DIM-1];

    initial begin
        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
            ctx_acc[d] = '0;
        end
    end

    typedef enum logic [1:0] {
        ST_ACCUMULATE = 2'b00,
        ST_OUT_BEAT0  = 2'b01,
        ST_OUT_BEAT1  = 2'b10,
        ST_OUT_BEAT2  = 2'b11
    } gemm2_state_t;

    gemm2_state_t state_reg;

    // INT8 Clamping after arithmetic shift right by 8 bits
    genvar j;
    generate
        for (j = 0; j < MAX_HEAD_DIM; j++) begin : gen_clamp
            logic signed [23:0] shifted_val;
            assign shifted_val = ctx_acc[j] >>> 8;

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

    // Accumulation and Egress FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg      <= ST_ACCUMULATE;
            m_axis_tvalid  <= 1'b0;
            m_axis_tdata   <= '0;
            m_axis_tlast   <= 1'b0;
            for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
        end else begin
            case (state_reg)
                ST_ACCUMULATE: begin
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;

                    if (start_row) begin
                        for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                    end else if (prob_valid) begin
                        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
                            ctx_acc[d] <= ctx_acc[d] + (32'($signed({1'b0, prob_data})) * 32'($signed(v_vec[d])));
                        end

                        if (prob_data > 0) begin
                            $display("[SCORE_V ACCUM]: prob=%0d, v[0]=%0d, v[1]=%0d, acc[0]=%0d", 
                                     prob_data, v_vec[0], v_vec[1], ctx_acc[0]);
                        end

                        if (prob_last) begin
                            $display("[SCORE_V DONE]: ctx_acc[0]=%0d, shifted=%0d, clamped=%0d", 
                                     ctx_acc[0], ctx_acc[0] >>> 8, ctx_clamped[0]);
                            state_reg <= ST_OUT_BEAT0;
                        end
                    end
                end

                ST_OUT_BEAT0: begin
                    // Stream first 8 bytes (Dimensions 0..7)
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {ctx_safe[7], ctx_safe[6], ctx_safe[5], ctx_safe[4],
                                      ctx_safe[3], ctx_safe[2], ctx_safe[1], ctx_safe[0]};

                    if (head_dim <= 16'd8) begin
                        m_axis_tlast <= 1'b1;
                        state_reg    <= ST_ACCUMULATE;
                    end else begin
                        m_axis_tlast <= 1'b0;
                        state_reg    <= ST_OUT_BEAT1;
                    end
                end

                ST_OUT_BEAT1: begin
                    // Stream second 8 bytes (Dimensions 8..15)
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {ctx_safe[15], ctx_safe[14], ctx_safe[13], ctx_safe[12],
                                      ctx_safe[11], ctx_safe[10], ctx_safe[9],  ctx_safe[8]};

                    if (head_dim <= 16'd16) begin
                        m_axis_tlast <= 1'b1;
                        for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                        state_reg    <= ST_ACCUMULATE;
                    end else begin
                        m_axis_tlast <= 1'b0;
                        state_reg    <= ST_OUT_BEAT2;
                    end
                end

                ST_OUT_BEAT2: begin
                    // Stream third 8 bytes (Dimensions 16..23)
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {ctx_safe[23], ctx_safe[22], ctx_safe[21], ctx_safe[20],
                                      ctx_safe[19], ctx_safe[18], ctx_safe[17], ctx_safe[16]};
                    m_axis_tlast  <= 1'b1;
                    for (int d = 0; d < MAX_HEAD_DIM; d++) ctx_acc[d] <= '0;
                    state_reg     <= ST_ACCUMULATE;
                end

                default: state_reg <= ST_ACCUMULATE;
            endcase
        end
    end

endmodule

`endif // SCORE_V_GEMM_SV
