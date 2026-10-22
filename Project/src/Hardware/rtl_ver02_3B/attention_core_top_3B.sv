`timescale 1ns / 1ps
// ============================================================================
// Module Name:   attention_core_top_3B
// Description:  Top-Level Multi-Head Self-Attention Core Accelerator
//               Variant 3B: 2D Output-Stationary Systolic MAC Array (16 DSP48E2)
//               with Tile-Based GEMM 1 and Score Matrix Reordering Buffer.
//               Computes Attention Scores in 4x8 tiles using a 4x4 PE grid,
//               reassembles scores in reorder_buffer_3B, and streams row-by-row
//               into Dyadic Scale Unit, IntSoftmax LUT, and Score-V GEMM 2.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Architecture: Variant 3B (16 DSP48E2 Hard Slices for GEMM 1)
// ============================================================================

module attention_core_top_3B #(
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

    logic               cfg_lut_we;
    logic [9:0]         cfg_lut_addr;
    logic [19:0]        cfg_lut_wdata;

    logic               core_busy;
    logic               core_done;
    logic               core_error;
    logic               pp_bank_sel;
    logic [3:0]         head_cnt;

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
    // 5. STAGE 1: 2D OUTPUT-STATIONARY SYSTOLIC MAC ARRAY (16 DSP48E2)
    // ------------------------------------------------------------------------
    logic signed [7:0]  systolic_act_row [0:ARRAY_SIZE-1];
    logic signed [7:0]  systolic_wt_col0 [0:ARRAY_SIZE-1];
    logic signed [7:0]  systolic_wt_col1 [0:ARRAY_SIZE-1];
    logic               systolic_in_valid;
    logic               systolic_clr_acc;
    logic               systolic_drain_start;
    logic               systolic_drain_done;
    logic               systolic_out_valid;
    logic signed [31:0] systolic_dot_stream;

    systolic_mac_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .IN_WIDTH   (8),
        .ACC_WIDTH  (32),
        .AUTO_SKEW  (1'b1)
    ) u_systolic_mac_array (
        .clk                (clk),
        .rst_n              (rst_n),
        .gemm_start         (start_pulse),
        .clr_acc            (systolic_clr_acc),
        .in_valid           (systolic_in_valid),
        .mode_sel           (1'b0),
        .act_row            (systolic_act_row),
        .wt_col0            (systolic_wt_col0),
        .wt_col1            (systolic_wt_col1),
        .drain_start        (systolic_drain_start),
        .drain_done         (systolic_drain_done),
        .out_valid          (systolic_out_valid),
        .dot_product_stream (systolic_dot_stream),
        .dot_product_out    ()
    );

    // ------------------------------------------------------------------------
    // 6. STAGE 1.5: TILE-TO-ROW SCORE REORDERING BUFFER (4x256 Words INT32)
    // ------------------------------------------------------------------------
    logic [4:0]  reorder_tile_col;
    logic [4:0]  drain_item_cnt;
    logic        reorder_read_start;
    logic [1:0]  reorder_read_row_sel;
    logic        reorder_read_valid;
    logic signed [31:0] reorder_read_score;
    logic        reorder_read_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain_item_cnt <= '0;
        end else if (systolic_out_valid) begin
            if (drain_item_cnt == 5'd31)
                drain_item_cnt <= '0;
            else
                drain_item_cnt <= drain_item_cnt + 1'b1;
        end
    end

    reorder_buffer_3B #(
        .NUM_ROWS   (4),
        .ROW_LEN    (256),
        .DATA_WIDTH (32)
    ) u_reorder_buffer (
        .clk            (clk),
        .rst_n          (rst_n),
        .wr_en          (systolic_out_valid),
        .tile_col_idx   (reorder_tile_col),
        .drain_item_cnt (drain_item_cnt),
        .wr_data        (systolic_dot_stream),
        .read_start     (reorder_read_start),
        .read_row_sel   (reorder_read_row_sel),
        .read_valid     (reorder_read_valid),
        .read_data      (reorder_read_score),
        .read_done      (reorder_read_done)
    );

    // ------------------------------------------------------------------------
    // 7. STAGE 2: DYADIC SCALER UNIT ((in_score * M) >>> E)
    // ------------------------------------------------------------------------
    logic signed [15:0] scaled_score_out;
    logic               scaler_out_valid;

    scale_unit #(
        .IN_WIDTH  (32),
        .OUT_WIDTH (16)
    ) u_scale_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (reorder_read_valid),
        .in_score     (reorder_read_score),
        .scale_m      (reg_scale_m),
        .scale_e      (reg_scale_e),
        .shift_val    (reg_shift_val),
        .out_valid    (scaler_out_valid),
        .scaled_score (scaled_score_out)
    );

    // ------------------------------------------------------------------------
    // 8. STAGE 3: RECONFIGURABLE HARDWARE SOFTMAX
    // ------------------------------------------------------------------------
    logic [15:0] softmax_prob_out;
    logic        softmax_out_valid;
    logic        softmax_busy;
    logic        row_start_pulse;

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
    // 9. STAGE 4: SCORE x V CONTEXT AGGREGATION GEMM ENGINE (128-BIT EGRESS)
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
    // 10. TILE-BASED MASTER CONTROL FSM (Variant 3B)
    // ------------------------------------------------------------------------
    typedef enum logic [3:0] {
        MHA_IDLE       = 4'b0000,
        MHA_WAIT_DATA  = 4'b0001,
        TILE_ROW_BLOCK = 4'b0010,
        TILE_INIT      = 4'b0011,
        TILE_FEED_K    = 4'b0100,
        TILE_FLUSH     = 4'b0101,
        TILE_DRAIN     = 4'b0110,
        SM_ROW_START   = 4'b0111,
        SM_ROW_STREAM  = 4'b1000,
        SM_ROW_WAIT    = 4'b1001,
        MHA_HEAD_NEXT  = 4'b1010,
        MHA_DONE       = 4'b1011
    } tile_fsm_state_t;

    tile_fsm_state_t state;

    logic [5:0]  tile_row_blk;   // 0 to 63 (64 blocks of 4 rows = 256 rows)
    logic [4:0]  tile_col_idx;   // 0 to 31 (32 horizontal tiles of width 8)
    logic [4:0]  dim_cnt;        // 0 to 15 (sweep d dimension)
    logic [3:0]  flush_cnt;      // 0 to 11 (wavefront propagation flush)
    logic [1:0]  sm_sub_row;     // 0 to 3 (read 4 rows out of reorder buffer)

    assign reorder_tile_col     = tile_col_idx;
    assign reorder_read_row_sel = sm_sub_row;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                <= MHA_IDLE;
            head_cnt             <= '0;
            tile_row_blk         <= '0;
            tile_col_idx         <= '0;
            dim_cnt              <= '0;
            flush_cnt            <= '0;
            sm_sub_row           <= '0;
            systolic_clr_acc     <= 1'b0;
            systolic_in_valid    <= 1'b0;
            systolic_drain_start <= 1'b0;
            reorder_read_start   <= 1'b0;
            row_start_pulse      <= 1'b0;
            tile_read_done       <= 1'b0;
            bram_read_en         <= 1'b0;
            for (int r = 0; r < ARRAY_SIZE; r++) systolic_act_row[r] <= 8'sd0;
            for (int c = 0; c < ARRAY_SIZE; c++) begin
                systolic_wt_col0[c] <= 8'sd0;
                systolic_wt_col1[c] <= 8'sd0;
            end
        end else begin
            systolic_clr_acc     <= 1'b0;
            systolic_drain_start <= 1'b0;
            reorder_read_start   <= 1'b0;
            row_start_pulse      <= 1'b0;
            tile_read_done       <= 1'b0;

            case (state)
                MHA_IDLE: begin
                    head_cnt          <= '0;
                    tile_row_blk      <= '0;
                    tile_col_idx      <= '0;
                    systolic_in_valid <= 1'b0;
                    bram_read_en      <= 1'b0;
                    if (start_pulse) begin
                        state <= MHA_WAIT_DATA;
                    end
                end

                MHA_WAIT_DATA: begin
                    if (head_data_ready) begin
                        bram_read_en <= 1'b1;
                        tile_row_blk <= '0;
                        state        <= TILE_ROW_BLOCK;
                    end
                end

                TILE_ROW_BLOCK: begin
                    tile_col_idx <= '0;
                    state        <= TILE_INIT;
                end

                TILE_INIT: begin
                    // Clear 32 accumulators for the new 4x8 tile
                    systolic_clr_acc  <= 1'b1;
                    dim_cnt           <= '0;
                    systolic_in_valid <= 1'b0;
                    state             <= TILE_FEED_K;
                end

                TILE_FEED_K: begin
                    systolic_in_valid <= 1'b1;

                    // Present activations for Row 0..3 at dimension d
                    for (int r = 0; r < ARRAY_SIZE; r++) begin
                        int row_idx;
                        row_idx = {tile_row_blk, 2'b00} + r;
                        systolic_act_row[r] <= q_mem[row_idx][dim_cnt];
                    end

                    // Present weights for Col 0..7 (paired into col0 and col1) at dimension d
                    for (int c = 0; c < ARRAY_SIZE; c++) begin
                        int col0_idx, col1_idx;
                        col0_idx = {tile_col_idx, 3'b000} + 2 * c;
                        col1_idx = {tile_col_idx, 3'b000} + 2 * c + 1;
                        systolic_wt_col0[c] <= k_mem[col0_idx][dim_cnt];
                        systolic_wt_col1[c] <= k_mem[col1_idx][dim_cnt];
                    end

                    if (dim_cnt == reg_head_dim - 1) begin
                        dim_cnt           <= '0;
                        flush_cnt         <= '0;
                        state             <= TILE_FLUSH;
                    end else begin
                        dim_cnt <= dim_cnt + 1'b1;
                    end
                end

                TILE_FLUSH: begin
                    // End of K streaming: drive zero inputs and wait for wavefront to reach PE grid
                    systolic_in_valid <= 1'b0;
                    for (int r = 0; r < ARRAY_SIZE; r++) systolic_act_row[r] <= 8'sd0;
                    for (int c = 0; c < ARRAY_SIZE; c++) begin
                        systolic_wt_col0[c] <= 8'sd0;
                        systolic_wt_col1[c] <= 8'sd0;
                    end

                    // 12 cycles wavefront propagation latency
                    if (flush_cnt == 4'd11) begin
                        flush_cnt            <= '0;
                        systolic_drain_start <= 1'b1;
                        state                <= TILE_DRAIN;
                    end else begin
                        flush_cnt <= flush_cnt + 1'b1;
                    end
                end

                TILE_DRAIN: begin
                    // Wait for 32-item sequential tile drain into reorder_buffer_3B
                    if (systolic_drain_done) begin
                        if (tile_col_idx == 5'd31) begin // All 32 horizontal tiles (256 cols) finished!
                            tile_col_idx <= '0;
                            sm_sub_row   <= '0;
                            state        <= SM_ROW_START;
                        end else begin
                            tile_col_idx <= tile_col_idx + 1'b1;
                            state        <= TILE_INIT; // Next horizontal tile
                        end
                    end
                end

                SM_ROW_START: begin
                    // Trigger row-by-row streaming from reorder_buffer_3B to Scale Unit & Softmax
                    row_start_pulse    <= 1'b1;
                    reorder_read_start <= 1'b1;
                    state              <= SM_ROW_STREAM;
                end

                SM_ROW_STREAM: begin
                    // Wait for 256 elements of the row to be streamed out
                    if (reorder_read_done) begin
                        state <= SM_ROW_WAIT;
                    end
                end

                SM_ROW_WAIT: begin
                    // Wait for Softmax and Score-V Context GEMM to finish this row
                    if (score_v_row_done && !softmax_busy) begin
                        if (sm_sub_row == 2'd3) begin
                            sm_sub_row <= '0;
                            if (tile_row_blk == 6'd63) begin // All 64 row blocks (256 rows) finished!
                                tile_row_blk <= '0;
                                state        <= MHA_HEAD_NEXT;
                            end else begin
                                tile_row_blk <= tile_row_blk + 1'b1;
                                state        <= TILE_ROW_BLOCK;
                            end
                        end else begin
                            sm_sub_row <= sm_sub_row + 1'b1;
                            state      <= SM_ROW_START;
                        end
                    end
                end

                MHA_HEAD_NEXT: begin
                    if (head_cnt == reg_num_heads - 1) begin
                        tile_read_done <= 1'b1;
                        state          <= MHA_DONE;
                    end else begin
                        head_cnt <= head_cnt + 1'b1;
                        state    <= TILE_ROW_BLOCK;
                    end
                end

                MHA_DONE: begin
                    if (!start_pulse) begin
                        state <= MHA_IDLE;
                    end
                end

                default: state <= MHA_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // 11. Status & Egress Wiring
    // ------------------------------------------------------------------------
    assign core_busy  = (state != MHA_IDLE) && (state != MHA_DONE);
    assign core_done  = (state == MHA_DONE);
    assign core_error = 1'b0;

    assign m_axis_tdata  = context_axis_tdata;
    assign m_axis_tvalid = context_axis_tvalid;
    assign m_axis_tlast  = context_axis_tlast;

endmodule
