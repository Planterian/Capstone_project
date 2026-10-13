`timescale 1ns / 1ps
// ============================================================================
// Module Name:   softmax_lut
// Description:  Bit-Exact Hardware Softmax Engine (I-ViT Shiftmax Aligned).
//               Implements Max-Subtraction, Pre-computed Exp ROM (19-bit),
//               32-cycle Restoring Divider for Reciprocal Normalization (inv_sum),
//               and 16-bit Bit-Exact Probability Normalization:
//               out_prob = (rom_data_exp * inv_sum) >> 16
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.2 (Bit-Exact I-ViT Aligned - 16-bit / 8-bit Configurable)
// ============================================================================

module softmax_lut #(
    parameter int N_TOKENS       = 256,
    parameter int IN_WIDTH       = 16,
    parameter int OUT_WIDTH      = 16,      // 16-bit for I-ViT Bit-Exact Resolution
    parameter int ROM_DEPTH      = 1024,    // 1024 words: 4 partitions of 256 words
    parameter int ROM_DATA_WIDTH = 20,      // 20-bit wide (supports Stage 2..4 up to 0xA0000 = 655360)
    parameter int ACC_WIDTH      = 32,
    parameter     INIT_HEX_FILE  = "exp_rom_all_stages.hex"
)(
    input  logic                                clk,
    input  logic                                rst_n,

    input  logic                                start_row,
    input  logic                                in_valid,
    input  logic signed [IN_WIDTH-1:0]          in_score,

    output logic                                out_valid,
    output logic [OUT_WIDTH-1:0]                out_prob,
    output logic                                busy,

    // Step 2.5: Multi-Stage and Reconfigurable LUT Interface
    input  logic [15:0]                         head_dim,     // 16 -> Stage 2 (0..255), 20 -> Stage 3 (256..511), 24 -> Stage 4 (512..767)
    input  logic                                cfg_we,       // Port B Write Enable from AXI-Lite
    input  logic [9:0]                          cfg_addr,     // Port B Word Address (0..1023)
    input  logic [ROM_DATA_WIDTH-1:0]           cfg_wdata     // Port B Write Data (20-bit)
);

    typedef enum logic [2:0] {
        ST_IDLE       = 3'b000,
        ST_FIND_MAX   = 3'b001,
        ST_EXP_ACC    = 3'b010,
        ST_DIV_START  = 3'b011,
        ST_DIV_RUN    = 3'b100,
        ST_DIV_DONE   = 3'b101,
        ST_NORM_OUT   = 3'b110
    } sm_state_t;

    sm_state_t state_reg;

    // Buffer for scores of a row (N_TOKENS = 256)
    logic signed [IN_WIDTH-1:0] score_buffer [0:N_TOKENS-1];

    logic [8:0]                 cnt_reg;
    logic signed [IN_WIDTH-1:0] s_max_reg;
    logic [ACC_WIDTH-1:0]       sum_exp_reg;
    logic [31:0]                inv_sum_reg;

    // Pipeline valid flag
    logic                       pipe_valid;

    // 32-step Restoring Divider Registers
    logic [31:0]                div_rem_reg;
    logic [31:0]                div_quot_reg;
    logic [31:0]                div_denom_reg;
    logic [5:0]                 div_step_reg;

    // Exp ROM lookup
    logic [7:0]                 rom_addr;
    logic [9:0]                 rom_addr_stage;
    logic [ROM_DATA_WIDTH-1:0]  rom_data_exp;

    (* ram_style = "block" *) logic [ROM_DATA_WIDTH-1:0] exp_rom [0:ROM_DEPTH-1];

    // Stage Partition Address Mapping (Port A)
    always_comb begin
        case (head_dim)
            16'd16:  rom_addr_stage = {2'b00, rom_addr}; // Stage 2: Offset 0x000 (0..255)
            16'd20:  rom_addr_stage = {2'b01, rom_addr}; // Stage 3: Offset 0x100 (256..511)
            16'd24:  rom_addr_stage = {2'b10, rom_addr}; // Stage 4: Offset 0x200 (512..767)
            default: rom_addr_stage = {2'b00, rom_addr}; // Default: Stage 2
        endcase
    end

    // ROM Initialization
    initial begin
        for (int idx = 0; idx < ROM_DEPTH; idx++) begin
            exp_rom[idx] = 20'd8;
        end
        $readmemh(INIT_HEX_FILE, exp_rom);
    end

    // Port A: Read with 1-cycle registered latency for inference (RAMB18/36 compatible)
    always_ff @(posedge clk) begin
        rom_data_exp <= exp_rom[rom_addr_stage];
    end

    // Port B: Runtime write from AXI-Lite
    always_ff @(posedge clk) begin
        if (cfg_we) begin
            exp_rom[cfg_addr] <= cfg_wdata;
        end
    end

    // Normalization intermediate wires
    logic [51:0] mult_full;
    logic [35:0] shifted_prob;
    assign mult_full    = {32'd0, rom_data_exp} * {20'd0, inv_sum_reg};
    assign shifted_prob = mult_full >> 16;

    // FSM and Datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg     <= ST_IDLE;
            cnt_reg       <= '0;
            s_max_reg     <= -16'sd32768;
            sum_exp_reg   <= '0;
            inv_sum_reg   <= '0;
            div_rem_reg   <= '0;
            div_quot_reg  <= '0;
            div_denom_reg <= '0;
            div_step_reg  <= '0;
            pipe_valid    <= 1'b0;
            out_valid     <= 1'b0;
            out_prob      <= '0;
        end else begin
            out_valid <= 1'b0;

            case (state_reg)
                ST_IDLE: begin
                    cnt_reg    <= '0;
                    pipe_valid <= 1'b0;
                    if (start_row) begin
                        s_max_reg   <= -16'sd32768;
                        sum_exp_reg <= '0;
                        state_reg   <= ST_FIND_MAX;
                        if (in_valid) begin
                            score_buffer[0] <= in_score;
                            s_max_reg       <= in_score;
                            cnt_reg         <= 9'd1;
                        end
                    end
                end

                ST_FIND_MAX: begin
                    if (in_valid) begin
                        score_buffer[cnt_reg] <= in_score;
                        if (in_score > s_max_reg) begin
                            s_max_reg <= in_score;
                        end

                        if (cnt_reg == N_TOKENS - 1) begin
                            cnt_reg    <= '0;
                            pipe_valid <= 1'b0;
                            state_reg  <= ST_EXP_ACC;
                        end else begin
                            cnt_reg <= cnt_reg + 1'b1;
                        end
                    end
                end

                ST_EXP_ACC: begin
                    // Stage 0 requests rom_addr, Stage 1 latches rom_data_exp and accumulates
                    if (cnt_reg < N_TOKENS) begin
                        pipe_valid <= 1'b1;
                        cnt_reg    <= cnt_reg + 1'b1;
                    end else begin
                        pipe_valid <= 1'b0;
                    end

                    if (pipe_valid) begin
                        sum_exp_reg <= sum_exp_reg + {12'd0, rom_data_exp};
                    end

                    if (cnt_reg == N_TOKENS && !pipe_valid) begin
                        cnt_reg   <= '0;
                        state_reg <= ST_DIV_START;
                    end
                end

                ST_DIV_START: begin
                    // Initialize 32-step restoring divider: Numerator = 32'h7FFF_FFFF (2^31 - 1)
                    div_denom_reg <= (sum_exp_reg == 0) ? 32'd1 : sum_exp_reg;
                    div_rem_reg   <= 32'd0;
                    div_quot_reg  <= 32'h7FFF_FFFF;
                    div_step_reg  <= 6'd32;
                    state_reg     <= ST_DIV_RUN;
                end

                ST_DIV_RUN: begin
                    // 32-bit Restoring Division Step
                    logic [32:0] rem_trial;
                    rem_trial = {div_rem_reg[30:0], div_quot_reg[31]} - {1'b0, div_denom_reg};

                    if (rem_trial[32] == 1'b0) begin
                        div_rem_reg  <= rem_trial[31:0];
                        div_quot_reg <= {div_quot_reg[30:0], 1'b1};
                    end else begin
                        div_rem_reg  <= {div_rem_reg[30:0], div_quot_reg[31]};
                        div_quot_reg <= {div_quot_reg[30:0], 1'b0};
                    end

                    if (div_step_reg == 6'd1) begin
                        state_reg <= ST_DIV_DONE;
                    end else begin
                        div_step_reg <= div_step_reg - 1'b1;
                    end
                end

                ST_DIV_DONE: begin
                    inv_sum_reg <= div_quot_reg;
                    cnt_reg     <= '0;
                    pipe_valid  <= 1'b0;
                    state_reg   <= ST_NORM_OUT;
                end

                ST_NORM_OUT: begin
                    // Stage 0 requests rom_addr, Stage 1 latches rom_data_exp and outputs out_prob
                    if (cnt_reg < N_TOKENS) begin
                        pipe_valid <= 1'b1;
                        cnt_reg    <= cnt_reg + 1'b1;
                    end else begin
                        pipe_valid <= 1'b0;
                    end

                    if (pipe_valid) begin
                        out_valid <= 1'b1;
                        if (OUT_WIDTH == 16) begin
                            out_prob <= (shifted_prob > 36'd65535) ? 16'hFFFF : shifted_prob[15:0];
                        end else begin
                            out_prob <= (shifted_prob > 36'd255) ? 8'hFF : shifted_prob[7:0];
                        end
                    end

                    if (cnt_reg == N_TOKENS && !pipe_valid) begin
                        cnt_reg   <= '0;
                        state_reg <= ST_IDLE;
                    end
                end

                default: state_reg <= ST_IDLE;
            endcase
        end
    end

    // ROM Address Generation (Comb based on cnt_reg and score_buffer)
    always_comb begin
        logic signed [IN_WIDTH-1:0] cur_score;
        logic signed [IN_WIDTH-1:0] delta_s;
        int                         idx;

        idx = (cnt_reg < N_TOKENS) ? int'(cnt_reg) : (N_TOKENS - 1);
        cur_score = score_buffer[idx];
        delta_s   = cur_score - s_max_reg;

        if (delta_s < -255) begin
            rom_addr = 8'd255;
        end else begin
            rom_addr = 8'(-delta_s);
        end
    end

    assign busy = (state_reg != ST_IDLE);

endmodule
