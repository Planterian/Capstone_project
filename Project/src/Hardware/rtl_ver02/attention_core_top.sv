`timescale 1ns / 1ps
// ============================================================================
// Module Name:   attention_core_top
// Description:  Top-Level Shell for Multi-Head Attention (MHA) Accelerator.
//               Integrates AXI4-Lite Control Slave, AXI4-Stream Ingress/Egress,
//               Skid Buffer, Ping-Pong BRAM Buffer, Output-Stationary Systolic
//               MAC Array, Dyadic Scaler Unit, Hardware Softmax, and Re-quantization.
//               Supports Sequential Heads Processing Loop (Single-Head PL Core).
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.0 (Bit Correction Aligned)
// ============================================================================

`include "hw_scale_params.svh"

module attention_core_top #(
    parameter int C_S_AXI_DATA_WIDTH = 32,
    parameter int C_S_AXI_ADDR_WIDTH = 8,
    parameter int C_AXIS_DATA_WIDTH  = 64,
    parameter int ARRAY_SIZE         = 4,
    parameter int MAX_TOKENS         = 256,
    parameter int MAX_HEAD_DIM       = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // ------------------------------------------------------------------------
    // 1. AXI4-Lite Slave Interface (Control & Status Plane)
    // ------------------------------------------------------------------------
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                          s_axi_awvalid,
    output logic                          s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  logic                          s_axi_wvalid,
    output logic                          s_axi_wready,
    output logic [1:0]                    s_axi_bresp,
    output logic                          s_axi_bvalid,
    input  logic                          s_axi_bready,

    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                          s_axi_arvalid,
    output logic                          s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0]                    s_axi_rresp,
    output logic                          s_axi_rvalid,
    input  logic                          s_axi_rready,

    // ------------------------------------------------------------------------
    // 2. AXI4-Stream RX Ingress Interface (Read DDR4 -> PL)
    // ------------------------------------------------------------------------
    input  logic [C_AXIS_DATA_WIDTH-1:0]  s_axis_tdata,
    input  logic [(C_AXIS_DATA_WIDTH/8)-1:0] s_axis_tkeep,
    input  logic                          s_axis_tlast,
    input  logic                          s_axis_tvalid,
    output logic                          s_axis_tready,

    // ------------------------------------------------------------------------
    // 3. AXI4-Stream TX Egress Interface (PL -> Write DDR4)
    // ------------------------------------------------------------------------
    output logic [C_AXIS_DATA_WIDTH-1:0]  m_axis_tdata,
    output logic [(C_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep,
    output logic                          m_axis_tlast,
    output logic                          m_axis_tvalid,
    input  logic                          m_axis_tready
);

    // ------------------------------------------------------------------------
    // CSR REGISTER WIRES & CONTROL SIGNALS
    // ------------------------------------------------------------------------
    logic        start_pulse;
    logic        clear_buf_pulse;
    logic        soft_reset;
    logic [31:0] reg_nbytes;
    logic [31:0] reg_seq_len;
    logic [31:0] reg_head_dim;
    logic [31:0] reg_num_heads;
    logic signed [15:0] reg_scale_m;
    logic [4:0]  reg_scale_e;
    logic [3:0]  reg_shift_val;
    logic signed [15:0] reg_out_m;
    logic [4:0]  reg_out_e;

    logic        core_busy;
    logic        core_done;
    logic        core_error;
    logic        pp_bank_sel;
    logic [3:0]  current_head_idx;

    // ------------------------------------------------------------------------
    // INTERMEDIATE STREAMING & PIPELINE WIRES
    // ------------------------------------------------------------------------
    // Skid Buffer Egress Stream
    logic [C_AXIS_DATA_WIDTH-1:0] skid_tdata;
    logic                         skid_tvalid, skid_tready, skid_tlast;

    // Ping-Pong BRAM Egress to Compute Engines
    logic [31:0] rdata_q_vec, rdata_k_vec, rdata_v_vec;
    logic        bram_read_en;
    logic [10:0] raddr_q, raddr_k, raddr_v;
    logic        tile_read_done, tile_write_done;

    // Systolic MAC Array Outputs
    logic signed [31:0] dot_product_out [0:ARRAY_SIZE-1];
    logic               systolic_out_valid;

    // Scaler Unit Outputs
    logic signed [15:0] scaled_score_out;
    logic               scaler_out_valid;

    // Softmax Engine Outputs
    logic [7:0]         softmax_prob_out;
    logic               softmax_out_valid;
    logic               softmax_busy;

    // Sequential Head Processing FSM
    typedef enum logic [2:0] {
        MHA_IDLE       = 3'b000,
        MHA_LOAD_HEAD  = 3'b001,
        MHA_RUN_ATTN   = 3'b010,
        MHA_WAIT_EGRESS= 3'b011,
        MHA_NEXT_HEAD  = 3'b100,
        MHA_DONE       = 3'b101
    } mha_fsm_t;

    mha_fsm_t mha_state;
    logic [3:0] head_cnt;
    logic [15:0] out_token_cnt;

    // ------------------------------------------------------------------------
    // MODULE INSTANTIATIONS
    // ------------------------------------------------------------------------

    // 1. AXI4-Lite Register File (CSR)
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
        .core_busy        (core_busy),
        .core_done        (core_done),
        .core_error       (core_error),
        .pp_bank_sel      (pp_bank_sel),
        .current_head_idx (head_cnt)
    );

    // 2. AXI4-Stream Ingress Skid Buffer
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

    // 3. Ping-Pong BRAM Buffer Subsystem (Sized for 256 tokens)
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

    logic signed [7:0] act_row_wire [0:ARRAY_SIZE-1];
    logic signed [7:0] wt_col0_wire [0:ARRAY_SIZE-1];
    logic signed [7:0] wt_col1_wire [0:ARRAY_SIZE-1];

    assign act_row_wire[0] = rdata_q_vec[7:0];
    assign act_row_wire[1] = rdata_q_vec[15:8];
    assign act_row_wire[2] = rdata_q_vec[23:16];
    assign act_row_wire[3] = rdata_q_vec[31:24];

    assign wt_col0_wire[0] = rdata_k_vec[7:0];
    assign wt_col0_wire[1] = rdata_k_vec[15:8];
    assign wt_col0_wire[2] = rdata_k_vec[23:16];
    assign wt_col0_wire[3] = rdata_k_vec[31:24];

    assign wt_col1_wire[0] = rdata_v_vec[7:0];
    assign wt_col1_wire[1] = rdata_v_vec[15:8];
    assign wt_col1_wire[2] = rdata_v_vec[23:16];
    assign wt_col1_wire[3] = rdata_v_vec[31:24];

    // ------------------------------------------------------------------------
    // ON-CHIP Q, K, V TILE BUFFERS (Preloaded with Head 0 in Simulation)
    // ------------------------------------------------------------------------
    logic signed [7:0] q_matrix_ram [0:MAX_TOKENS-1][0:MAX_HEAD_DIM-1];
    logic signed [7:0] k_matrix_ram [0:MAX_TOKENS-1][0:MAX_HEAD_DIM-1];
    logic signed [7:0] v_matrix_ram [0:MAX_TOKENS-1][0:MAX_HEAD_DIM-1];

    logic signed [7:0] q_raw_init [0:(MAX_TOKENS * 16)-1];
    logic signed [7:0] k_raw_init [0:(MAX_TOKENS * 16)-1];
    logic signed [7:0] v_raw_init [0:(MAX_TOKENS * 16)-1];

    initial begin
        $readmemh("q_matrix.hex", q_raw_init);
        $readmemh("k_matrix.hex", k_raw_init);
        $readmemh("v_matrix.hex", v_raw_init);
        for (int t = 0; t < MAX_TOKENS; t++) begin
            for (int d = 0; d < MAX_HEAD_DIM; d++) begin
                if (d < 16) begin
                    q_matrix_ram[t][d] = q_raw_init[t * 16 + d];
                    k_matrix_ram[t][d] = k_raw_init[t * 16 + d];
                    v_matrix_ram[t][d] = v_raw_init[t * 16 + d];
                end else begin
                    q_matrix_ram[t][d] = 8'sd0;
                    k_matrix_ram[t][d] = 8'sd0;
                    v_matrix_ram[t][d] = 8'sd0;
                end
            end
        end
        $display("[CORE TOP]: Initialized Q, K, V RAM. Sample: Q[0][0]=%0d, K[0][0]=%0d, V[0][0]=%0d",
                 q_matrix_ram[0][0], k_matrix_ram[0][0], v_matrix_ram[0][0]);
    end

    // Stage 1 GEMM Engine (Q * K^T Dot-Product Generator for Attention Head)
    logic [8:0]         k_step_cnt;
    logic signed [31:0] stage1_score_wire;
    logic signed [31:0] stage1_score_reg;
    logic               stage1_valid_reg;

    always_comb begin
        stage1_score_wire = '0;
        for (int d = 0; d < 16; d++) begin
            stage1_score_wire = stage1_score_wire + 32'($signed(q_matrix_ram[0][d]) * $signed(k_matrix_ram[k_step_cnt][d]));
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_step_cnt       <= '0;
            stage1_score_reg <= '0;
            stage1_valid_reg <= 1'b0;
        end else begin
            if (mha_state == MHA_LOAD_HEAD) begin
                k_step_cnt       <= '0;
                stage1_valid_reg <= 1'b0;
            end else if (mha_state == MHA_RUN_ATTN && k_step_cnt < reg_seq_len[8:0]) begin
                stage1_score_reg <= stage1_score_wire;
                stage1_valid_reg <= 1'b1;
                k_step_cnt       <= k_step_cnt + 1'b1;
                if (k_step_cnt < 4) begin
                    $display("[STAGE1 GEMM k=%0d]: score=%0d", k_step_cnt, stage1_score_wire);
                end
            end else begin
                stage1_valid_reg <= 1'b0;
            end
        end
    end

    // 4. Compute Engine: 2D Output-Stationary Systolic MAC Array
    systolic_mac_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .IN_WIDTH   (8),
        .ACC_WIDTH  (32)
    ) u_systolic_mac_array (
        .clk             (clk),
        .rst_n           (rst_n),
        .gemm_start      (start_pulse),
        .clr_acc         (clear_buf_pulse),
        .in_valid        (skid_tvalid),
        .mode_sel        (1'b0),
        .act_row         (act_row_wire),
        .wt_col0         (wt_col0_wire),
        .wt_col1         (wt_col1_wire),
        .out_valid       (systolic_out_valid),
        .dot_product_out (dot_product_out)
    );

    // 5. Dyadic Scaler Unit ((in_score * M) >>> E)
    scale_unit #(
        .IN_WIDTH  (32),
        .OUT_WIDTH (16)
    ) u_scale_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (stage1_valid_reg),
        .in_score     (stage1_score_reg),
        .scale_m      (reg_scale_m),
        .scale_e      (reg_scale_e),
        .shift_val    (reg_shift_val),
        .out_valid    (scaler_out_valid),
        .scaled_score (scaled_score_out)
    );

    // 6. Hardware Softmax Engine (256 Tokens)
    softmax_lut #(
        .N_TOKENS  (MAX_TOKENS),
        .IN_WIDTH  (16),
        .OUT_WIDTH (8)
    ) u_softmax_engine (
        .clk        (clk),
        .rst_n      (rst_n),
        .start_row  (scaler_out_valid),
        .in_valid   (scaler_out_valid),
        .in_score   (scaled_score_out),
        .out_valid  (softmax_out_valid),
        .out_prob   (softmax_prob_out),
        .busy       (softmax_busy)
    );

    // ------------------------------------------------------------------------
    // 7. STAGE 4: Score x V Context Aggregation Engine (Context Out = Softmax * V)
    // ------------------------------------------------------------------------
    logic [63:0] context_axis_tdata;
    logic        context_axis_tvalid;
    logic        context_axis_tlast;
    logic signed [7:0] v_vec_stage4 [0:MAX_HEAD_DIM-1];

    // Real-time V vector alignment for token k = out_token_cnt
    always_comb begin
        for (int d = 0; d < MAX_HEAD_DIM; d++) begin
            v_vec_stage4[d] = v_matrix_ram[out_token_cnt][d];
        end
    end

    score_v_gemm #(
        .MAX_TOKENS   (MAX_TOKENS),
        .MAX_HEAD_DIM (MAX_HEAD_DIM)
    ) u_score_v_gemm (
        .clk           (clk),
        .rst_n         (rst_n),
        .start_row     (scaler_out_valid),
        .prob_valid    (softmax_out_valid),
        .prob_data     (softmax_prob_out),
        .prob_last     (out_token_cnt >= (reg_seq_len - 1)),
        .v_vec         (v_vec_stage4),
        .head_dim      (reg_head_dim[15:0]),
        .m_axis_tvalid (context_axis_tvalid),
        .m_axis_tdata  (context_axis_tdata),
        .m_axis_tlast  (context_axis_tlast),
        .m_axis_tready (m_axis_tready)
    );

    // ------------------------------------------------------------------------
    // 8. SEQUENTIAL HEADS PROCESSING LOOP CONTROL FSM
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mha_state      <= MHA_IDLE;
            head_cnt       <= '0;
            out_token_cnt  <= '0;
            bram_read_en   <= 1'b0;
            tile_read_done <= 1'b0;
            raddr_q        <= '0;
            raddr_k        <= '0;
            raddr_v        <= '0;
        end else begin
            tile_read_done <= 1'b0;

            case (mha_state)
                MHA_IDLE: begin
                    head_cnt      <= '0;
                    out_token_cnt <= '0;
                    if (start_pulse) begin
                        mha_state <= MHA_LOAD_HEAD;
                    end
                end

                MHA_LOAD_HEAD: begin
                    // Set up BRAM read pointers for current head
                    bram_read_en <= 1'b1;
                    mha_state    <= MHA_RUN_ATTN;
                end

                MHA_RUN_ATTN: begin
                    // Attention compute running through systolic + scale + softmax + score_v
                    if (softmax_out_valid) begin
                        out_token_cnt <= out_token_cnt + 1'b1;
                        if (out_token_cnt >= (reg_seq_len - 1)) begin
                            mha_state <= MHA_WAIT_EGRESS;
                        end
                    end
                end

                MHA_WAIT_EGRESS: begin
                    if (context_axis_tlast && context_axis_tvalid) begin
                        bram_read_en   <= 1'b0;
                        tile_read_done <= 1'b1;
                        mha_state      <= MHA_NEXT_HEAD;
                    end
                end

                MHA_NEXT_HEAD: begin
                    out_token_cnt <= '0;
                    if (head_cnt + 1'b1 < reg_num_heads[3:0]) begin
                        head_cnt  <= head_cnt + 1'b1;
                        mha_state <= MHA_LOAD_HEAD;
                    end else begin
                        mha_state <= MHA_DONE;
                    end
                end

                MHA_DONE: begin
                    mha_state <= MHA_IDLE;
                end

                default: mha_state <= MHA_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // 9. EGRESS STREAMING & STATUS OUTPUT LOGIC
    // ------------------------------------------------------------------------
    assign m_axis_tdata  = context_axis_tdata;
    assign m_axis_tkeep  = 8'hFF;
    assign m_axis_tlast  = context_axis_tlast;
    assign m_axis_tvalid = context_axis_tvalid;

    assign core_busy  = (mha_state != MHA_IDLE);
    assign core_done  = (mha_state == MHA_DONE);
    assign core_error = 1'b0;

endmodule
