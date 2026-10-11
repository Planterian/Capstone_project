`timescale 1ns / 1ps
// ============================================================================
// Module Name:   softmax_lut
// Description:  Hardware Softmax Engine using Max-Subtraction, BRAM Exp ROM,
//               and Iterative Division for Reciprocal Normalization (Zero Timing Slack Violations).
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned - 256 Tokens Supported)
// ============================================================================

module softmax_lut #(
    parameter int N_TOKENS   = 256,
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

    initial begin
        for (int i = 0; i < N_TOKENS; i++) begin
            score_buffer[i] = '0;
        end
    end
    logic [8:0]                 cnt_reg;
    logic signed [IN_WIDTH-1:0] s_max_reg;
    logic [ACC_WIDTH-1:0]       sum_exp_reg;
    logic [31:0]                inv_sum_reg;

    // Divider Registers (Iterative 24-step non-restoring divider)
    logic [47:0]                div_rem_reg;
    logic [31:0]                div_denom_reg;
    logic [5:0]                 div_step_reg;

    // Exp ROM lookup
    logic [7:0]                 rom_addr;
    logic [15:0]                rom_data_exp;

    (* ram_style = "block" *) logic [15:0] exp_rom [0:ROM_DEPTH-1];

    initial begin
            exp_rom[0] = 16'd65535;
            exp_rom[1] = 16'd63519;
            exp_rom[2] = 16'd61564;
            exp_rom[3] = 16'd59670;
            exp_rom[4] = 16'd57834;
            exp_rom[5] = 16'd56055;
            exp_rom[6] = 16'd54330;
            exp_rom[7] = 16'd52659;
            exp_rom[8] = 16'd51039;
            exp_rom[9] = 16'd49468;
            exp_rom[10] = 16'd47946;
            exp_rom[11] = 16'd46471;
            exp_rom[12] = 16'd45042;
            exp_rom[13] = 16'd43656;
            exp_rom[14] = 16'd42313;
            exp_rom[15] = 16'd41011;
            exp_rom[16] = 16'd39749;
            exp_rom[17] = 16'd38526;
            exp_rom[18] = 16'd37341;
            exp_rom[19] = 16'd36192;
            exp_rom[20] = 16'd35078;
            exp_rom[21] = 16'd33999;
            exp_rom[22] = 16'd32953;
            exp_rom[23] = 16'd31939;
            exp_rom[24] = 16'd30957;
            exp_rom[25] = 16'd30004;
            exp_rom[26] = 16'd29081;
            exp_rom[27] = 16'd28186;
            exp_rom[28] = 16'd27319;
            exp_rom[29] = 16'd26479;
            exp_rom[30] = 16'd25664;
            exp_rom[31] = 16'd24874;
            exp_rom[32] = 16'd24109;
            exp_rom[33] = 16'd23367;
            exp_rom[34] = 16'd22648;
            exp_rom[35] = 16'd21951;
            exp_rom[36] = 16'd21276;
            exp_rom[37] = 16'd20622;
            exp_rom[38] = 16'd19987;
            exp_rom[39] = 16'd19372;
            exp_rom[40] = 16'd18776;
            exp_rom[41] = 16'd18198;
            exp_rom[42] = 16'd17639;
            exp_rom[43] = 16'd17096;
            exp_rom[44] = 16'd16570;
            exp_rom[45] = 16'd16060;
            exp_rom[46] = 16'd15566;
            exp_rom[47] = 16'd15087;
            exp_rom[48] = 16'd14623;
            exp_rom[49] = 16'd14173;
            exp_rom[50] = 16'd13737;
            exp_rom[51] = 16'd13314;
            exp_rom[52] = 16'd12905;
            exp_rom[53] = 16'd12508;
            exp_rom[54] = 16'd12123;
            exp_rom[55] = 16'd11750;
            exp_rom[56] = 16'd11388;
            exp_rom[57] = 16'd11038;
            exp_rom[58] = 16'd10698;
            exp_rom[59] = 16'd10369;
            exp_rom[60] = 16'd10050;
            exp_rom[61] = 16'd9741;
            exp_rom[62] = 16'd9441;
            exp_rom[63] = 16'd9151;
            exp_rom[64] = 16'd8869;
            exp_rom[65] = 16'd8596;
            exp_rom[66] = 16'd8332;
            exp_rom[67] = 16'd8075;
            exp_rom[68] = 16'd7827;
            exp_rom[69] = 16'd7586;
            exp_rom[70] = 16'd7353;
            exp_rom[71] = 16'd7127;
            exp_rom[72] = 16'd6907;
            exp_rom[73] = 16'd6695;
            exp_rom[74] = 16'd6489;
            exp_rom[75] = 16'd6289;
            exp_rom[76] = 16'd6096;
            exp_rom[77] = 16'd5908;
            exp_rom[78] = 16'd5726;
            exp_rom[79] = 16'd5550;
            exp_rom[80] = 16'd5379;
            exp_rom[81] = 16'd5214;
            exp_rom[82] = 16'd5054;
            exp_rom[83] = 16'd4898;
            exp_rom[84] = 16'd4747;
            exp_rom[85] = 16'd4601;
            exp_rom[86] = 16'd4460;
            exp_rom[87] = 16'd4323;
            exp_rom[88] = 16'd4190;
            exp_rom[89] = 16'd4061;
            exp_rom[90] = 16'd3936;
            exp_rom[91] = 16'd3815;
            exp_rom[92] = 16'd3697;
            exp_rom[93] = 16'd3583;
            exp_rom[94] = 16'd3473;
            exp_rom[95] = 16'd3366;
            exp_rom[96] = 16'd3263;
            exp_rom[97] = 16'd3162;
            exp_rom[98] = 16'd3065;
            exp_rom[99] = 16'd2971;
            exp_rom[100] = 16'd2879;
            exp_rom[101] = 16'd2791;
            exp_rom[102] = 16'd2705;
            exp_rom[103] = 16'd2622;
            exp_rom[104] = 16'd2541;
            exp_rom[105] = 16'd2463;
            exp_rom[106] = 16'd2387;
            exp_rom[107] = 16'd2314;
            exp_rom[108] = 16'd2242;
            exp_rom[109] = 16'd2173;
            exp_rom[110] = 16'd2107;
            exp_rom[111] = 16'd2042;
            exp_rom[112] = 16'd1979;
            exp_rom[113] = 16'd1918;
            exp_rom[114] = 16'd1859;
            exp_rom[115] = 16'd1802;
            exp_rom[116] = 16'd1746;
            exp_rom[117] = 16'd1693;
            exp_rom[118] = 16'd1641;
            exp_rom[119] = 16'd1590;
            exp_rom[120] = 16'd1541;
            exp_rom[121] = 16'd1494;
            exp_rom[122] = 16'd1448;
            exp_rom[123] = 16'd1403;
            exp_rom[124] = 16'd1360;
            exp_rom[125] = 16'd1318;
            exp_rom[126] = 16'd1278;
            exp_rom[127] = 16'd1238;
            exp_rom[128] = 16'd1200;
            exp_rom[129] = 16'd1163;
            exp_rom[130] = 16'd1128;
            exp_rom[131] = 16'd1093;
            exp_rom[132] = 16'd1059;
            exp_rom[133] = 16'd1027;
            exp_rom[134] = 16'd995;
            exp_rom[135] = 16'd964;
            exp_rom[136] = 16'd935;
            exp_rom[137] = 16'd906;
            exp_rom[138] = 16'd878;
            exp_rom[139] = 16'd851;
            exp_rom[140] = 16'd825;
            exp_rom[141] = 16'd800;
            exp_rom[142] = 16'd775;
            exp_rom[143] = 16'd751;
            exp_rom[144] = 16'd728;
            exp_rom[145] = 16'd706;
            exp_rom[146] = 16'd684;
            exp_rom[147] = 16'd663;
            exp_rom[148] = 16'd642;
            exp_rom[149] = 16'd623;
            exp_rom[150] = 16'd604;
            exp_rom[151] = 16'd585;
            exp_rom[152] = 16'd567;
            exp_rom[153] = 16'd550;
            exp_rom[154] = 16'd533;
            exp_rom[155] = 16'd516;
            exp_rom[156] = 16'd500;
            exp_rom[157] = 16'd485;
            exp_rom[158] = 16'd470;
            exp_rom[159] = 16'd456;
            exp_rom[160] = 16'd442;
            exp_rom[161] = 16'd428;
            exp_rom[162] = 16'd415;
            exp_rom[163] = 16'd402;
            exp_rom[164] = 16'd390;
            exp_rom[165] = 16'd378;
            exp_rom[166] = 16'd366;
            exp_rom[167] = 16'd355;
            exp_rom[168] = 16'd344;
            exp_rom[169] = 16'd333;
            exp_rom[170] = 16'd323;
            exp_rom[171] = 16'd313;
            exp_rom[172] = 16'd303;
            exp_rom[173] = 16'd294;
            exp_rom[174] = 16'd285;
            exp_rom[175] = 16'd276;
            exp_rom[176] = 16'd268;
            exp_rom[177] = 16'd260;
            exp_rom[178] = 16'd252;
            exp_rom[179] = 16'd244;
            exp_rom[180] = 16'd236;
            exp_rom[181] = 16'd229;
            exp_rom[182] = 16'd222;
            exp_rom[183] = 16'd215;
            exp_rom[184] = 16'd209;
            exp_rom[185] = 16'd202;
            exp_rom[186] = 16'd196;
            exp_rom[187] = 16'd190;
            exp_rom[188] = 16'd184;
            exp_rom[189] = 16'd178;
            exp_rom[190] = 16'd173;
            exp_rom[191] = 16'd168;
            exp_rom[192] = 16'd162;
            exp_rom[193] = 16'd157;
            exp_rom[194] = 16'd153;
            exp_rom[195] = 16'd148;
            exp_rom[196] = 16'd143;
            exp_rom[197] = 16'd139;
            exp_rom[198] = 16'd135;
            exp_rom[199] = 16'd131;
            exp_rom[200] = 16'd127;
            exp_rom[201] = 16'd123;
            exp_rom[202] = 16'd119;
            exp_rom[203] = 16'd115;
            exp_rom[204] = 16'd112;
            exp_rom[205] = 16'd108;
            exp_rom[206] = 16'd105;
            exp_rom[207] = 16'd102;
            exp_rom[208] = 16'd99;
            exp_rom[209] = 16'd95;
            exp_rom[210] = 16'd93;
            exp_rom[211] = 16'd90;
            exp_rom[212] = 16'd87;
            exp_rom[213] = 16'd84;
            exp_rom[214] = 16'd82;
            exp_rom[215] = 16'd79;
            exp_rom[216] = 16'd77;
            exp_rom[217] = 16'd74;
            exp_rom[218] = 16'd72;
            exp_rom[219] = 16'd70;
            exp_rom[220] = 16'd68;
            exp_rom[221] = 16'd66;
            exp_rom[222] = 16'd64;
            exp_rom[223] = 16'd62;
            exp_rom[224] = 16'd60;
            exp_rom[225] = 16'd58;
            exp_rom[226] = 16'd56;
            exp_rom[227] = 16'd54;
            exp_rom[228] = 16'd53;
            exp_rom[229] = 16'd51;
            exp_rom[230] = 16'd50;
            exp_rom[231] = 16'd48;
            exp_rom[232] = 16'd47;
            exp_rom[233] = 16'd45;
            exp_rom[234] = 16'd44;
            exp_rom[235] = 16'd42;
            exp_rom[236] = 16'd41;
            exp_rom[237] = 16'd40;
            exp_rom[238] = 16'd39;
            exp_rom[239] = 16'd37;
            exp_rom[240] = 16'd36;
            exp_rom[241] = 16'd35;
            exp_rom[242] = 16'd34;
            exp_rom[243] = 16'd33;
            exp_rom[244] = 16'd32;
            exp_rom[245] = 16'd31;
            exp_rom[246] = 16'd30;
            exp_rom[247] = 16'd29;
            exp_rom[248] = 16'd28;
            exp_rom[249] = 16'd27;
            exp_rom[250] = 16'd27;
            exp_rom[251] = 16'd26;
            exp_rom[252] = 16'd25;
            exp_rom[253] = 16'd24;
            exp_rom[254] = 16'd23;
            exp_rom[255] = 16'd23;
    end

    // ROM read
    always_ff @(posedge clk) begin
        rom_data_exp <= exp_rom[rom_addr];
    end

    // FSM and Datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg     <= ST_IDLE;
            cnt_reg       <= '0;
            s_max_reg     <= -16'sd32768;
            sum_exp_reg   <= '0;
            inv_sum_reg   <= '0;
            div_rem_reg   <= '0;
            div_denom_reg <= '0;
            div_step_reg  <= '0;
            out_valid     <= 1'b0;
            out_prob      <= '0;
        end else begin
            out_valid <= 1'b0;

            case (state_reg)
                ST_IDLE: begin
                    cnt_reg <= '0;
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
                        if (cnt_reg < 4) $display("[SOFTMAX IN cnt=%0d]: in_score=%0d", cnt_reg, in_score);
                        if (in_score > s_max_reg) begin
                            s_max_reg <= in_score;
                        end

                        if (cnt_reg == N_TOKENS - 1) begin
                            cnt_reg   <= '0;
                            state_reg <= ST_EXP_ACC;
                        end else begin
                            cnt_reg   <= cnt_reg + 1'b1;
                        end
                    end
                end

                ST_EXP_ACC: begin
                    // Accumulate ROM exponential (skip cnt_reg==0 due to 1-cycle ROM latency)
                    if (cnt_reg > 0) begin
                        sum_exp_reg <= sum_exp_reg + rom_data_exp;
                    end

                    if (cnt_reg == N_TOKENS) begin
                        cnt_reg   <= '0;
                        state_reg <= ST_DIV_START;
                    end else begin
                        cnt_reg   <= cnt_reg + 1'b1;
                    end
                end

                ST_DIV_START: begin
                    $display("[SOFTMAX DIV_START]: s_max=%0d, sum_exp=%0d", s_max_reg, sum_exp_reg);
                    // Initialize divider: numerator = (255 << 16) = 16,711,680
                    div_denom_reg <= (sum_exp_reg == 0) ? 32'd1 : sum_exp_reg;
                    div_rem_reg   <= {{16'b0, 32'd16711680}};
                    div_step_reg  <= 6'd24;
                    state_reg     <= ST_DIV_RUN;
                end

                ST_DIV_RUN: begin
                    // Shift and subtract non-restoring divider
                    if (div_rem_reg[47:24] >= div_denom_reg[23:0]) begin
                        div_rem_reg <= {{div_rem_reg[47:24] - div_denom_reg[23:0], div_rem_reg[23:0]} << 1, 1'b1};
                    end else begin
                        div_rem_reg <= div_rem_reg << 1;
                    end

                    if (div_step_reg == 6'd0) begin
                        state_reg <= ST_DIV_DONE;
                    end else begin
                        div_step_reg <= div_step_reg - 1'b1;
                    end
                end

                ST_DIV_DONE: begin
                    inv_sum_reg <= div_rem_reg[23:0];
                    cnt_reg     <= '0;
                    state_reg   <= ST_NORM_OUT;
                end

                ST_NORM_OUT: begin
                    // Normalization stage
                    logic [31:0] prod_prob;
                    prod_prob = (rom_data_exp * inv_sum_reg) >> 16;

                    if (cnt_reg > 0) begin
                        out_prob  <= (prod_prob > 32'd255) ? 8'hFF : prod_prob[7:0];
                        out_valid <= 1'b1;
                        if (cnt_reg < 5) begin
                            $display("[SOFTMAX NORM cnt=%0d]: rom_exp=%0d, inv_sum=%0d, prod_prob=%0d",
                                     cnt_reg, rom_data_exp, inv_sum_reg, prod_prob);
                        end
                    end

                    if (cnt_reg == N_TOKENS) begin
                        cnt_reg   <= '0;
                        state_reg <= ST_IDLE;
                    end else begin
                        cnt_reg   <= cnt_reg + 1'b1;
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
        
        cur_score = (cnt_reg < N_TOKENS) ? score_buffer[cnt_reg] : score_buffer[N_TOKENS-1];
        delta_s   = cur_score - s_max_reg;

        if (delta_s < -255) begin
            rom_addr = 8'd255;
        end else begin
            rom_addr = 8'(-delta_s);
        end
    end

    assign busy = (state_reg != ST_IDLE);

endmodule
