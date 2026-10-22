`timescale 1ns / 1ps
// ============================================================================
// Module Name:   attention_core_top_2A
// Description:  Top-Level Multi-Head Self-Attention Core Accelerator
//               Variant 2A: High-Throughput DSP-Packed Streaming Engine
//               Replaces 16 LUT multipliers with 8 DSP48E2 PE slices
//               (pe_slice_dsp_packing) paired with a 3-stage binary reduction
//               adder tree, achieving sustained II = 1 (1 score / cycle) with
//               only 8 DSPs on AMD Kria KV260.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Architecture: Variant 2A (8 DSP48E2 Hard Slices for GEMM 1)
// ============================================================================

module attention_core_top_2A #(
    parameter int C_S_AXI_DATA_WIDTH = 32,
    parameter int C_S_AXI_ADDR_WIDTH = 8,
    parameter int C_AXIS_DATA_WIDTH  = 128,  // 16 INT8 tokens/beat native AXI-Stream
    parameter int ARRAY_SIZE         = 4,
    parameter int MAX_TOKENS         = 256,
    parameter int MAX_HEAD_DIM       = 32
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // ------------------------------------------------------------------------
    // AXI4-Lite Control & Status Registers (CSR)
    // ------------------------------------------------------------------------
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]       s_axi_awaddr,
    input  logic                                s_axi_awvalid,
    output logic                                s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0]       s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0]   s_axi_wstrb,
    input  logic                                s_axi_wvalid,
    output logic                                s_axi_wready,
    output logic [1:0]                          s_axi_bresp,
    output logic                                s_axi_bvalid,
    input  logic                                s_axi_bready,

    input  logic [C_S_AXI_ADDR_WIDTH-1:0]       s_axi_araddr,
    input  logic                                s_axi_arvalid,
    output logic                                s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0]       s_axi_rdata,
    output logic [1:0]                          s_axi_rresp,
    output logic                                s_axi_rvalid,
    input  logic                                s_axi_rready,

    // ------------------------------------------------------------------------
    // AXI4-Stream Slave (Ingress: Packed Q, K, V Matrices)
    // ------------------------------------------------------------------------
    input  logic [C_AXIS_DATA_WIDTH-1:0]        s_axis_tdata,
    input  logic [(C_AXIS_DATA_WIDTH/8)-1:0]    s_axis_tkeep,
    input  logic                                s_axis_tlast,
    input  logic                                s_axis_tvalid,
    output logic                                s_axis_tready,

    // ------------------------------------------------------------------------
    // AXI4-Stream Master (Egress: Final Requantized Context Vector)
    // ------------------------------------------------------------------------
    output logic [C_AXIS_DATA_WIDTH-1:0]        m_axis_tdata,
    output logic [(C_AXIS_DATA_WIDTH/8)-1:0]    m_axis_tkeep,
    output logic                                m_axis_tlast,
    output logic                                m_axis_tvalid,
    input  logic                                m_axis_tready
);

    // Assign egress keep: all bytes valid
    assign m_axis_tkeep = '1;

    // ------------------------------------------------------------------------
    // 1. Control & Status Signals from AXI-Lite
    // ------------------------------------------------------------------------
    logic               start_pulse;
    logic               clear_buf_pulse;
    logic               soft_reset;
    logic [31:0]        reg_nbytes;
    logic [31:0]        reg_seq_len;
    logic [31:0]        reg_head_dim;
    logic [31:0]        reg_num_heads;
    logic signed [15:0] reg_scale_m;
    logic [4:0]         reg_scale_e;
    logic [3:0]         reg_shift_val;
    logic signed [15:0] reg_out_m;
    logic [4:0]         reg_out_e;

    // Step 2.5: BRAM LUT Reload Ports (from AXI-Lite CSR to softmax_lut Port B)
    logic               cfg_lut_we;
    logic [9:0]         cfg_lut_addr;
    logic [19:0]        cfg_lut_wdata;

    // Core Status Signals
    logic               core_busy;
    logic               core_done;
    logic               core_error;
    logic               pp_bank_sel;
    logic [3:0]         head_cnt;

    // ------------------------------------------------------------------------
    // 1. AXI4-Lite Register File (CSR Slave)
    // ------------------------------------------------------------------------
    axi_lite_regs #(
        .C_S_AXI_DATA_WIDTH (C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH (C_S_AXI_ADDR_WIDTH)
    ) u_axi_lite_regs (
        .S_AXI_ACLK       (clk),
        .S_AXI_ARESETN    (rst_n),
        .S_AXI_AWADDR     (s_axi_awaddr),
        .S_AXI_AWVALID    (s_axi_awvalid),
        .S_AXI_AWREADY    (s_axi_awready),
        .S_AXI_WDATA      (s_axi_wdata),
        .S_AXI_WSTRB      (s_axi_wstrb),
        .S_AXI_WVALID     (s_axi_wvalid),
        .S_AXI_WREADY     (s_axi_wready),
        .S_AXI_BRESP      (s_axi_bresp),
        .S_AXI_BVALID     (s_axi_bvalid),
        .S_AXI_BREADY     (s_axi_bready),
        .S_AXI_ARADDR     (s_axi_araddr),
        .S_AXI_ARVALID    (s_axi_arvalid),
        .S_AXI_ARREADY    (s_axi_arready),
        .S_AXI_RDATA      (s_axi_rdata),
        .S_AXI_RRESP      (s_axi_rresp),
        .S_AXI_RVALID     (s_axi_rvalid),
        .S_AXI_RREADY     (s_axi_rready),
        .cfg_start_pulse  (start_pulse),
        .cfg_clear_buf    (clear_buf_pulse),
        .cfg_soft_reset   (soft_reset),
        .reg_nbytes       (reg_nbytes),
        .reg_seq_len      (reg_seq_len),
        .reg_head_dim     (reg_head_dim),
        .reg_num_heads    (reg_num_heads),
        .reg_scale_m      (reg_scale_m),
        .reg_scale_e      (reg_scale_e),
        .reg_shift_val    (reg_shift_val),
        .reg_out_m        (reg_out_m),
        .reg_out_e        (reg_out_e),
        .cfg_lut_we       (cfg_lut_we),
        .cfg_lut_addr     (cfg_lut_addr),
        .cfg_lut_wdata    (cfg_lut_wdata),
        .core_busy        (core_busy),
        .core_done        (core_done),
        .core_error       (core_error),
        .pp_bank_sel      (pp_bank_sel),
        .current_head_idx (head_cnt)
    );

    // ------------------------------------------------------------------------
    // 2. AXI4-Stream Ingress Skid Buffer
    // ------------------------------------------------------------------------
    logic [C_AXIS_DATA_WIDTH-1:0] skid_tdata;
    logic                         skid_tvalid, skid_tready, skid_tlast;

    axis_adapter #(
        .DATA_WIDTH (C_AXIS_DATA_WIDTH)
    ) u_skid_buffer_rx (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tdata  (skid_tdata),
        .m_axis_tvalid (skid_tvalid),
        .m_axis_tready (skid_tready),
        .m_axis_tlast  (skid_tlast)
    );

    // ------------------------------------------------------------------------
    // 3. Ping-Pong BRAM Buffer Subsystem
    // ------------------------------------------------------------------------
    logic [31:0] rdata_q_vec, rdata_k_vec, rdata_v_vec;
    logic        bram_read_en;
    logic [10:0] raddr_q, raddr_k, raddr_v;
    logic        tile_read_done, tile_write_done;

    ping_pong_bram_buffer #(
        .DATA_WIDTH (C_AXIS_DATA_WIDTH),
        .RAM_WIDTH  (32),
        .RAM_DEPTH  (2048),
        .ADDR_WIDTH (11)
    ) u_ping_pong_buffer (
        .clk             (clk),
        .rst_n           (rst_n),
        .s_axis_tdata    (skid_tdata),
        .s_axis_tvalid   (skid_tvalid),
        .s_axis_tready   (skid_tready),
        .s_axis_tlast    (skid_tlast),
        .ram_target_sel  (2'b00),
        .read_en         (bram_read_en),
        .raddr_q         (raddr_q),
        .raddr_k         (raddr_k),
        .raddr_v         (raddr_v),
        .rdata_q_vec     (rdata_q_vec),
        .rdata_k_vec     (rdata_k_vec),
        .rdata_v_vec     (rdata_v_vec),
        .tile_read_done  (tile_read_done),
        .tile_write_done (tile_write_done),
        .bank_write_free (),
        .bank_read_ready (),
        .pp_bank_sel     (pp_bank_sel)
    );

    // ------------------------------------------------------------------------
    // 4. ON-CHIP Q, K, V STORAGE FOR ACTIVE HEAD (Synthesizable Block RAM)
    // ------------------------------------------------------------------------
    (* ram_style = "block" *) logic signed [7:0] q_mem [0:MAX_TOKENS-1][0:MAX_HEAD_DIM-1];
    (* ram_style = "block" *) logic signed [7:0] k_mem [0:MAX_TOKENS-1][0:MAX_HEAD_DIM-1];
    (* ram_style = "block" *) logic signed [7:0] v_mem [0:MAX_TOKENS-1][0:MAX_HEAD_DIM-1];

    typedef enum logic [1:0] {
        ING_Q = 2'b00,
        ING_K = 2'b01,
        ING_V = 2'b10,
        ING_READY = 2'b11
    } ing_load_state_t;

    ing_load_state_t ing_load_state;
    logic [15:0] ing_beat_cnt;
    logic [7:0]  ing_tok_idx;
    logic        ing_slice_idx;
    logic        head_data_ready;

    // Beats per matrix = (head_dim <= 16) ? seq_len : (seq_len * 2)
    logic [15:0] beats_per_mat;
    assign beats_per_mat = (reg_head_dim <= 16'd16) ? reg_seq_len[15:0] : (reg_seq_len[15:0] << 1);

    always_comb begin
        if (reg_head_dim <= 16'd16) begin
            ing_tok_idx   = 8'(ing_beat_cnt);
            ing_slice_idx = 1'b0;
        end else begin
            ing_tok_idx   = 8'(ing_beat_cnt >> 1);
            ing_slice_idx = 1'(ing_beat_cnt & 16'd1);
        end
    end

    // Ingress write into active head on-chip storage (16 bytes / beat from 128-bit bus)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ing_load_state  <= ING_Q;
            ing_beat_cnt    <= '0;
            head_data_ready <= 1'b0;
        end else begin
            if (clear_buf_pulse || soft_reset) begin
                ing_load_state  <= ING_Q;
                ing_beat_cnt    <= '0;
                head_data_ready <= 1'b0;
            end else if (skid_tvalid && skid_tready) begin
                for (int b = 0; b < 16; b++) begin
                    int dim_idx;
                    dim_idx = ing_slice_idx * 16 + b;
                    if (dim_idx < MAX_HEAD_DIM) begin
                        case (ing_load_state)
                            ING_Q: q_mem[ing_tok_idx][dim_idx] <= skid_tdata[b*8 +: 8];
                            ING_K: k_mem[ing_tok_idx][dim_idx] <= skid_tdata[b*8 +: 8];
                            ING_V: v_mem[ing_tok_idx][dim_idx] <= skid_tdata[b*8 +: 8];
                            default: ;
                        endcase
                    end
                end

                if (ing_beat_cnt == beats_per_mat - 1) begin
                    ing_beat_cnt <= '0;
                    case (ing_load_state)
                        ING_Q: ing_load_state <= ING_K;
                        ING_K: ing_load_state <= ING_V;
                        ING_V: begin
                            ing_load_state  <= ING_READY;
                            head_data_ready <= 1'b1;
                        end
                        default: ;
                    endcase
                end else begin
                    ing_beat_cnt <= ing_beat_cnt + 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // 5. STAGE 1: VARIANT 2A DSP-PACKED ADDER TREE (8 DSP48E2 SLICES)
    //    Initiation Interval II = 1 (1 score / cycle sustained for dk=16).
    // ------------------------------------------------------------------------
    logic signed [31:0] stage1_score;
    logic               stage1_valid;
    logic [8:0]         q_row_idx;
    logic signed [7:0]  q_curr_vec [0:MAX_HEAD_DIM-1];
    logic               row_start_pulse;

    // Streaming controls for 128 Key pairs (256 keys total, multi-beat)
    logic [6:0]         k_pair_idx;
    logic [1:0]         k_sub_beat;
    logic               k_stream_valid;
    logic [1:0]         beats_per_pair_m1;

    assign beats_per_pair_m1 = (reg_head_dim <= 16'd16) ? 2'd1 :
                               (reg_head_dim <= 16'd24) ? 2'd2 : 2'd3;

    logic signed [7:0]  dsp_tree_q      [0:7];
    logic signed [7:0]  dsp_tree_k_even [0:7];
    logic signed [7:0]  dsp_tree_k_odd  [0:7];

    always_comb begin
        int col_even, col_odd;
        col_even = {k_pair_idx, 1'b0};
        col_odd  = {k_pair_idx, 1'b1};

        for (int d = 0; d < 8; d++) begin
            int actual_d;
            actual_d = k_sub_beat * 8 + d;
            if (actual_d < MAX_HEAD_DIM) begin
                dsp_tree_q[d]      = q_curr_vec[actual_d];
                dsp_tree_k_even[d] = k_mem[col_even][actual_d];
                dsp_tree_k_odd[d]  = k_mem[col_odd][actual_d];
            end else begin
                dsp_tree_q[d]      = 8'sd0;
                dsp_tree_k_even[d] = 8'sd0;
                dsp_tree_k_odd[d]  = 8'sd0;
            end
        end
    end

    dsp_packed_adder_tree #(
        .HEAD_DIM  (MAX_HEAD_DIM),
        .OUT_WIDTH (32)
    ) u_dsp_packed_adder_tree (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (k_stream_valid),
        .in_beat     (k_sub_beat),
        .q_vec       (dsp_tree_q),
        .k_even      (dsp_tree_k_even),
        .k_odd       (dsp_tree_k_odd),
        .head_dim    (reg_head_dim[15:0]),
        .out_valid   (stage1_valid),
        .dot_product (stage1_score)
    );

    // ------------------------------------------------------------------------
    // 6. STAGE 2: DYADIC SCALER UNIT ((in_score * M) >>> E)
    // ------------------------------------------------------------------------
    logic signed [15:0] scaled_score_out;
    logic               scaler_out_valid;

    scale_unit #(
        .IN_WIDTH  (32),
        .OUT_WIDTH (16)
    ) u_scale_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (stage1_valid),
        .in_score     (stage1_score),
        .scale_m      (reg_scale_m),
        .scale_e      (reg_scale_e),
        .shift_val    (reg_shift_val),
        .out_valid    (scaler_out_valid),
        .scaled_score (scaled_score_out)
    );

    // ------------------------------------------------------------------------
    // 7. STAGE 3: RECONFIGURABLE HARDWARE SOFTMAX
    // ------------------------------------------------------------------------
    logic [15:0] softmax_prob_out;
    logic        softmax_out_valid;
    logic        softmax_busy;

    softmax_lut #(
        .N_TOKENS       (MAX_TOKENS),
        .IN_WIDTH       (16),
        .OUT_WIDTH      (16),
        .ROM_DEPTH      (1024),
        .ROM_DATA_WIDTH (20),
        .ACC_WIDTH      (32),
        .INIT_HEX_FILE  ("exp_rom_all_stages.hex")
    ) u_softmax_engine (
        .clk        (clk),
        .rst_n      (rst_n),
        .start_row  (row_start_pulse),
        .in_valid   (scaler_out_valid),
        .in_score   (scaled_score_out),
        .out_valid  (softmax_out_valid),
        .out_prob   (softmax_prob_out),
        .busy       (softmax_busy),
        .head_dim   (reg_head_dim[15:0]),
        .cfg_we     (cfg_lut_we),
        .cfg_addr   (cfg_lut_addr),
        .cfg_wdata  (cfg_lut_wdata)
    );

    // ------------------------------------------------------------------------
    // 8. STAGE 4: SCORE x V CONTEXT AGGREGATION GEMM ENGINE (128-BIT EGRESS)
    // ------------------------------------------------------------------------
    logic [C_AXIS_DATA_WIDTH-1:0] context_axis_tdata;
    logic                         context_axis_tvalid;
    logic                         context_axis_tlast;
    logic [8:0]                   score_v_token_cnt;
    logic                         score_v_row_done;
    logic signed [7:0]            v_vec_active [0:MAX_HEAD_DIM-1];
    logic [8:0]                   v_tok_rd_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_tok_rd_idx <= '0;
        end else if (row_start_pulse) begin
            v_tok_rd_idx <= '0;
        end else if (softmax_out_valid) begin
            if (v_tok_rd_idx == reg_seq_len - 1)
                v_tok_rd_idx <= '0;
            else
                v_tok_rd_idx <= v_tok_rd_idx + 1'b1;
        end
    end

    always_comb begin
        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
            v_vec_active[d] = v_mem[v_tok_rd_idx][d];
        end
    end

    score_v_gemm #(
        .MAX_TOKENS      (MAX_TOKENS),
        .MAX_HEAD_DIM    (MAX_HEAD_DIM),
        .PROB_WIDTH      (16),
        .AXIS_DATA_WIDTH (C_AXIS_DATA_WIDTH)
    ) u_score_v_gemm (
        .clk           (clk),
        .rst_n         (rst_n),
        .start_row     (row_start_pulse),
        .prob_valid    (softmax_out_valid),
        .prob_data     (softmax_prob_out),
        .prob_last     (softmax_out_valid && (v_tok_rd_idx == reg_seq_len - 1)),
        .v_vec         (v_vec_active),
        .head_dim      (reg_head_dim[15:0]),
        .m_axis_tvalid (context_axis_tvalid),
        .m_axis_tdata  (context_axis_tdata),
        .m_axis_tlast  (context_axis_tlast),
        .m_axis_tready (m_axis_tready),
        .row_token_cnt (score_v_token_cnt),
        .row_done      (score_v_row_done)
    );

    // ------------------------------------------------------------------------
    // 9. SEQUENTIAL HEADS & DATAFLOW MASTER CONTROL FSM
    // ------------------------------------------------------------------------
    typedef enum logic [3:0] {
        MHA_IDLE       = 4'b0000,
        MHA_WAIT_DATA  = 4'b0001,
        MHA_ROW_START  = 4'b0010,
        MHA_ROW_PREP   = 4'b0011,
        MHA_ROW_GEMM_K = 4'b0100,
        MHA_ROW_WAIT   = 4'b0101,
        MHA_ROW_NEXT   = 4'b0110,
        MHA_HEAD_NEXT  = 4'b0111,
        MHA_DONE       = 4'b1000
    } mha_state_t;

    mha_state_t mha_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mha_state       <= MHA_IDLE;
            head_cnt        <= '0;
            q_row_idx       <= '0;
            k_pair_idx      <= '0;
            k_sub_beat      <= 2'b00;
            k_stream_valid  <= 1'b0;
            row_start_pulse <= 1'b0;
            bram_read_en    <= 1'b0;
            tile_read_done  <= 1'b0;
            raddr_q         <= '0;
            raddr_k         <= '0;
            raddr_v         <= '0;
            for (int d = 0; d < MAX_HEAD_DIM; d++) q_curr_vec[d] <= 8'sd0;
        end else begin
            row_start_pulse <= 1'b0;
            tile_read_done  <= 1'b0;

            case (mha_state)
                MHA_IDLE: begin
                    head_cnt       <= '0;
                    q_row_idx      <= '0;
                    k_pair_idx     <= '0;
                    k_sub_beat     <= 2'b00;
                    k_stream_valid <= 1'b0;
                    bram_read_en   <= 1'b0;
                    if (start_pulse) begin
                        mha_state <= MHA_WAIT_DATA;
                    end
                end

                MHA_WAIT_DATA: begin
                    if (head_data_ready) begin
                        bram_read_en <= 1'b1;
                        q_row_idx    <= '0;
                        mha_state    <= MHA_ROW_START;
                    end
                end

                MHA_ROW_START: begin
                    for (int d = 0; d < MAX_HEAD_DIM; d++) begin
                        q_curr_vec[d] <= q_mem[q_row_idx][d];
                    end
                    mha_state <= MHA_ROW_PREP;
                end

                MHA_ROW_PREP: begin
                    row_start_pulse <= 1'b1;
                    k_pair_idx      <= '0;
                    k_sub_beat      <= 2'b00;
                    k_stream_valid  <= 1'b1;
                    mha_state       <= MHA_ROW_GEMM_K;
                end

                MHA_ROW_GEMM_K: begin
                    if (k_sub_beat < beats_per_pair_m1) begin
                        k_sub_beat <= k_sub_beat + 1'b1;
                    end else begin
                        k_sub_beat <= 2'b00;
                        if (k_pair_idx == 7'd127) begin // 128 pairs * 2 = 256 keys
                            k_pair_idx     <= '0;
                            k_stream_valid <= 1'b0;
                            mha_state      <= MHA_ROW_WAIT;
                        end else begin
                            k_pair_idx <= k_pair_idx + 1'b1;
                        end
                    end
                end

                MHA_ROW_WAIT: begin
                    // Wait for both Softmax and Score-V Context GEMM to finish current row
                    if (score_v_row_done && !softmax_busy) begin
                        mha_state <= MHA_ROW_NEXT;
                    end
                end

                MHA_ROW_NEXT: begin
                    if (q_row_idx == reg_seq_len - 1) begin
                        q_row_idx <= '0;
                        mha_state <= MHA_HEAD_NEXT;
                    end else begin
                        q_row_idx <= q_row_idx + 1'b1;
                        mha_state <= MHA_ROW_START;
                    end
                end

                MHA_HEAD_NEXT: begin
                    if (head_cnt == reg_num_heads - 1) begin
                        tile_read_done <= 1'b1;
                        mha_state      <= MHA_DONE;
                    end else begin
                        head_cnt  <= head_cnt + 1'b1;
                        mha_state <= MHA_ROW_START;
                    end
                end

                MHA_DONE: begin
                    if (!start_pulse) begin
                        mha_state <= MHA_IDLE;
                    end
                end

                default: mha_state <= MHA_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // 10. Status & Egress Wiring
    // ------------------------------------------------------------------------
    assign core_busy  = (mha_state != MHA_IDLE) && (mha_state != MHA_DONE);
    assign core_done  = (mha_state == MHA_DONE);
    assign core_error = 1'b0;

    assign m_axis_tdata  = context_axis_tdata;
    assign m_axis_tvalid = context_axis_tvalid;
    assign m_axis_tlast  = context_axis_tlast;

endmodule
