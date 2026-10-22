`timescale 1ns / 1ps
// ============================================================================
// Module Name:   reorder_buffer_3B
// Description:  Tile-to-Row Reordering Buffer for Variant 3B Systolic Attention.
//               Stores 4 rows x 256 columns of INT32 Attention Scores (1,024 words).
//               Reassembles 4x8 tiles from Systolic MAC Array drain stream into
//               continuous 256-element row vectors for downstream Softmax Engine.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Architecture: Variant 3B (Tile-Based GEMM with Score Matrix Reordering Buffer)
// ============================================================================

module reorder_buffer_3B #(
    parameter int NUM_ROWS    = 4,
    parameter int ROW_LEN     = 256,
    parameter int DATA_WIDTH  = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,

    // Write Interface (from Systolic MAC Array Drain stream)
    input  logic                     wr_en,
    input  logic [4:0]               tile_col_idx,   // 0 to 31 (32 horizontal tiles of width 8)
    input  logic [4:0]               drain_item_cnt, // 0 to 31 (32 items per 4x8 tile)
    input  logic signed [DATA_WIDTH-1:0] wr_data,

    // Read Interface (Streaming continuous row to Scale Unit / Softmax)
    input  logic                     read_start,     // Pulse to start streaming one row
    input  logic [1:0]               read_row_sel,   // Row index within active 4-row block (0 to 3)
    output logic                     read_valid,
    output logic signed [DATA_WIDTH-1:0] read_data,
    output logic                     read_done       // High on the 256th element of the row
);

    // 4 rows x 256 columns of INT32 scores (1,024 words total)
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] score_ram [0:NUM_ROWS-1][0:ROW_LEN-1];

    // Simulation-safe initialization
    initial begin
        for (int r = 0; r < NUM_ROWS; r++) begin
            for (int c = 0; c < ROW_LEN; c++) begin
                score_ram[r][c] = '0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // 1. TILE DRAIN WRITE LOGIC
    //    Each 4x8 tile is drained row-by-row (8 elements per row, across 4 rows):
    //    r_in   = drain_item_cnt / 8 (0 to 3)
    //    c_sub  = drain_item_cnt % 8 (0 to 7)
    //    c_full = tile_col_idx * 8 + c_sub (0 to 255)
    // ------------------------------------------------------------------------
    logic [1:0] wr_row;
    logic [7:0] wr_col;

    assign wr_row = drain_item_cnt[4:3]; // drain_item_cnt / 8
    assign wr_col = {tile_col_idx, drain_item_cnt[2:0]}; // tile_col_idx * 8 + (drain_item_cnt % 8)

    always_ff @(posedge clk) begin
        if (wr_en) begin
            score_ram[wr_row][wr_col] <= wr_data;
        end
    end

    // ------------------------------------------------------------------------
    // 2. SEQUENTIAL ROW READOUT LOGIC (256 elements back-to-back)
    // ------------------------------------------------------------------------
    typedef enum logic [1:0] {
        RD_IDLE = 2'b00,
        RD_RUN  = 2'b01,
        RD_DONE = 2'b10
    } rd_state_t;

    rd_state_t rd_state;
    logic [7:0] rd_col_cnt;
    logic [1:0] active_row;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state   <= RD_IDLE;
            rd_col_cnt <= '0;
            active_row <= '0;
            read_valid <= 1'b0;
            read_done  <= 1'b0;
            read_data  <= '0;
        end else begin
            read_done <= 1'b0;

            case (rd_state)
                RD_IDLE: begin
                    read_valid <= 1'b0;
                    rd_col_cnt <= '0;
                    if (read_start) begin
                        active_row <= read_row_sel;
                        rd_col_cnt <= '0;
                        rd_state   <= RD_RUN;
                    end
                end

                RD_RUN: begin
                    read_valid <= 1'b1;
                    read_data  <= score_ram[active_row][rd_col_cnt];

                    if (rd_col_cnt == ROW_LEN - 1) begin
                        read_done  <= 1'b1;
                        rd_col_cnt <= '0;
                        rd_state   <= RD_DONE;
                    end else begin
                        rd_col_cnt <= rd_col_cnt + 1'b1;
                    end
                end

                RD_DONE: begin
                    read_valid <= 1'b0;
                    rd_state   <= RD_IDLE;
                end

                default: rd_state <= RD_IDLE;
            endcase
        end
    end

endmodule
