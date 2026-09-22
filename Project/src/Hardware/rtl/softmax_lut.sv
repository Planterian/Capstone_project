`timescale 1ns / 1ps
// ============================================================================
// Module Name:   softmax_lut
// Description:  Hardware Softmax Engine using Max-Subtraction, BRAM Exp ROM,
//               and Dyadic Reciprocal Division (0 Hardware Dividers).
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module softmax_lut #(
    parameter int N_TOKENS   = 196,
    parameter int IN_WIDTH   = 16,
    parameter int OUT_WIDTH  = 8,
    parameter int ROM_DEPTH  = 256,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   start_row,
    input  logic                   in_valid,
    input  logic signed [IN_WIDTH-1:0] in_score,

    output logic                   out_valid,
    output logic [OUT_WIDTH-1:0]   out_prob,
    output logic                   busy
);

    typedef enum logic [2:0] {
        ST_IDLE     = 3'b000,
        ST_FIND_MAX = 3'b001,
        ST_EXP_ACC  = 3'b010,
        ST_CALC_REC = 3'b011,
        ST_NORM_OUT = 3'b100
    } sm_state_t;

    sm_state_t state_reg, state_next;

    logic signed [IN_WIDTH-1:0] score_buffer [0:N_TOKENS-1];
    logic [7:0]                 cnt_reg, cnt_next;
    logic signed [IN_WIDTH-1:0] s_max_reg, s_max_next;
    logic [ACC_WIDTH-1:0]       sum_exp_reg, sum_exp_next;
    logic [31:0]                inv_sum_reg, inv_sum_next;
    logic [4:0]                 recip_shift_reg;

    logic [7:0]                 rom_addr;
    logic [15:0]                rom_data_exp;

    (* ram_style = "block" *) logic [15:0] exp_rom [0:ROM_DEPTH-1];

    initial begin
        for (int i = 0; i < ROM_DEPTH; i++) begin
            exp_rom[i] = 16'd65535 >> (i >> 3);
        end
    end

    always_ff @(posedge clk) begin
        rom_data_exp <= exp_rom[rom_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg   <= ST_IDLE;
            cnt_reg     <= '0;
            s_max_reg   <= -16'sd32768;
            sum_exp_reg <= '0;
            inv_sum_reg <= '0;
        end else begin
            state_reg   <= state_next;
            cnt_reg     <= cnt_next;
            s_max_reg   <= s_max_next;
            sum_exp_reg <= sum_exp_next;
            inv_sum_reg <= inv_sum_next;
        end
    end

    always_comb begin
        state_next   = state_reg;
        cnt_next     = cnt_reg;
        s_max_next   = s_max_reg;
        sum_exp_next = sum_exp_reg;
        inv_sum_next = inv_sum_reg;

        rom_addr     = '0;
        busy         = 1'b1;
        out_valid    = 1'b0;
        out_prob     = '0;

        case (state_reg)
            ST_IDLE: begin
                busy = 1'b0;
                cnt_next = '0;
                if (start_row) begin
                    s_max_next   = -16'sd32768;
                    sum_exp_next = '0;
                    state_next   = ST_FIND_MAX;
                end
            end

            ST_FIND_MAX: begin
                if (in_valid) begin
                    score_buffer[cnt_reg] <= in_score;
                    if (in_score > s_max_reg) begin
                        s_max_next = in_score;
                    end

                    if (cnt_reg == N_TOKENS - 1) begin
                        cnt_next   = '0;
                        state_next = ST_EXP_ACC;
                    end else begin
                        cnt_next   = cnt_reg + 1'b1;
                    end
                end
            end

            ST_EXP_ACC: begin
                logic signed [IN_WIDTH-1:0] delta_s;
                delta_s  = score_buffer[cnt_reg] - s_max_reg;
                rom_addr = (delta_s < -255) ? 8'd255 : 8'(-delta_s);

                sum_exp_next = sum_exp_reg + rom_data_exp;

                if (cnt_reg == N_TOKENS - 1) begin
                    cnt_next   = '0;
                    state_next = ST_CALC_REC;
                end else begin
                    cnt_next   = cnt_reg + 1'b1;
                end
            end

            ST_CALC_REC: begin
                if (sum_exp_reg != 0) begin
                    inv_sum_next = (32'h0100_0000) / sum_exp_reg;
                end else begin
                    inv_sum_next = 32'h0000_FFFF;
                end
                recip_shift_reg = 5'd16;
                cnt_next   = '0;
                state_next = ST_NORM_OUT;
            end

            ST_NORM_OUT: begin
                logic signed [IN_WIDTH-1:0] delta_s;
                logic [31:0] prod_prob;

                delta_s  = score_buffer[cnt_reg] - s_max_reg;
                rom_addr = (delta_s < -255) ? 8'd255 : 8'(-delta_s);

                prod_prob = (rom_data_exp * inv_sum_reg) >> recip_shift_reg;

                out_prob  = (prod_prob > 32'd255) ? 8'hFF : prod_prob[7:0];
                out_valid = 1'b1;

                if (cnt_reg == N_TOKENS - 1) begin
                    cnt_next   = '0;
                    state_next = ST_IDLE;
                end else begin
                    cnt_next   = cnt_reg + 1'b1;
                end
            end

            default: state_next = ST_IDLE;
        endcase
    end

endmodule
