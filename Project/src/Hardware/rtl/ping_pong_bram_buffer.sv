`timescale 1ns / 1ps
// ============================================================================
// Module Name:   ping_pong_bram_buffer
// Description:  Multi-Bank Ping-Pong BRAM Buffer Manager for MHA Accelerator.
//               Manages two memory banks (Bank 0 and Bank 1) for Q, K, V matrices.
//               Enables zero-latency overlap between AXI-Stream DDR4 writes and
//               Systolic Array reads using 1-clock bank-swapping FSM.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// ============================================================================

module ping_pong_bram_buffer #(
    parameter int DATA_WIDTH = 64,      // AXI-Stream Ingress Data Width (8x INT8)
    parameter int RAM_WIDTH  = 32,      // Systolic Array Vector Width (4x INT8)
    parameter int RAM_DEPTH  = 256,     // Depth per Q/K/V RAM (256 entries)
    parameter int ADDR_WIDTH = 8        // log2(RAM_DEPTH) = 8 bits
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // ------------------------------------------------------------------------
    // Ingress Stream Interface (from AXI-Stream Skid Buffer)
    // ------------------------------------------------------------------------
    input  logic [DATA_WIDTH-1:0]   s_axis_tdata,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,
    input  logic                    s_axis_tlast,
    input  logic [1:0]              ram_target_sel, // 2'b00: Q_RAM, 2'b01: K_RAM, 2'b10: V_RAM

    // ------------------------------------------------------------------------
    // Egress Interface (to Systolic MAC Array & Compute Engines)
    // ------------------------------------------------------------------------
    input  logic                    read_en,
    input  logic [ADDR_WIDTH-1:0]   raddr_q,
    input  logic [ADDR_WIDTH-1:0]   raddr_k,
    input  logic [ADDR_WIDTH-1:0]   raddr_v,

    output logic [RAM_WIDTH-1:0]    rdata_q_vec,    // Vector Q (4x INT8)
    output logic [RAM_WIDTH-1:0]    rdata_k_vec,    // Vector K (4x INT8)
    output logic [RAM_WIDTH-1:0]    rdata_v_vec,    // Vector V (4x INT8)

    // ------------------------------------------------------------------------
    // Control & Status Signals
    // ------------------------------------------------------------------------
    input  logic                    tile_read_done, // Pulse when Stage 1/4 completes reading tile
    output logic                    tile_write_done,// Pulse when 1 Tile is fully written to active bank
    output logic                    bank_write_free,// Active bank is ready for writing
    output logic                    bank_read_ready,// Active bank has valid tile data for reading
    output logic                    pp_bank_sel     // 0: Bank 0 Write / Bank 1 Read | 1: Bank 1 Write / Bank 0 Read
);

    // ------------------------------------------------------------------------
    // 1. DUAL-BANK BRAM ARRAY DECLARATIONS (AMD BRAM Inferred)
    // ------------------------------------------------------------------------
    // Bank 0 RAMs
    (* ram_style = "block" *) logic [RAM_WIDTH-1:0] ram_q_b0 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [RAM_WIDTH-1:0] ram_k_b0 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [RAM_WIDTH-1:0] ram_v_b0 [0:RAM_DEPTH-1];

    // Bank 1 RAMs
    (* ram_style = "block" *) logic [RAM_WIDTH-1:0] ram_q_b1 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [RAM_WIDTH-1:0] ram_k_b1 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [RAM_WIDTH-1:0] ram_v_b1 [0:RAM_DEPTH-1];

    // Internal Write Address Counters
    logic [ADDR_WIDTH-1:0] waddr_cnt;

    // Ping-Pong Status Flags
    logic bank0_full, bank1_full;
    logic pp_bank_reg;

    assign pp_bank_sel     = pp_bank_reg;
    assign bank_write_free = (pp_bank_reg == 1'b0) ? !bank0_full : !bank1_full;
    assign bank_read_ready = (pp_bank_reg == 1'b0) ?  bank1_full :  bank0_full;
    assign s_axis_tready   = bank_write_free && !tile_write_done;

    // ------------------------------------------------------------------------
    // 2. PING-PONG BANK SWAPPING FSM (1-Clock Zero Overhead)
    // ------------------------------------------------------------------------
    typedef enum logic [1:0] {
        PP_IDLE     = 2'b00,
        PP_WRITING  = 2'b01,
        PP_SWAP     = 2'b10
    } pp_state_t;

    pp_state_t pp_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pp_state        <= PP_IDLE;
            pp_bank_reg     <= 1'b0; // Start: Write Bank 0, Read Bank 1
            bank0_full      <= 1'b0;
            bank1_full      <= 1'b0;
            waddr_cnt       <= '0;
            tile_write_done <= 1'b0;
        end else begin
            tile_write_done <= 1'b0;

            case (pp_state)
                PP_IDLE: begin
                    waddr_cnt <= '0;
                    if (s_axis_tvalid && bank_write_free) begin
                        pp_state <= PP_WRITING;
                    end
                end

                PP_WRITING: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        if (waddr_cnt == RAM_DEPTH - 1 || s_axis_tlast) begin
                            waddr_cnt       <= '0;
                            tile_write_done <= 1'b1;
                            pp_state        <= PP_SWAP;

                            // Mark current write bank as FULL
                            if (pp_bank_reg == 1'b0)
                                bank0_full <= 1'b1;
                            else
                                bank1_full <= 1'b1;
                        end else begin
                            waddr_cnt <= waddr_cnt + 1'b1;
                        end
                    end
                end

                PP_SWAP: begin
                    // Swap bank pointers if read side has consumed previous tile
                    if (tile_read_done || !bank_read_ready) begin
                        pp_bank_reg <= ~pp_bank_reg;
                        
                        // Clear FULL status of newly read bank
                        if (pp_bank_reg == 1'b0)
                            bank1_full <= 1'b0;
                        else
                            bank0_full <= 1'b0;

                        pp_state <= PP_IDLE;
                    end
                end

                default: pp_state <= PP_IDLE;
            endcase

            // Independent Read Completion Clear
            if (tile_read_done) begin
                if (pp_bank_reg == 1'b0)
                    bank1_full <= 1'b0;
                else
                    bank0_full <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // 3. SYNCHRONOUS BRAM WRITE LOGIC (INGRESS ROUTING)
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (s_axis_tvalid && s_axis_tready) begin
            if (pp_bank_reg == 1'b0) begin
                // Write to BANK 0
                case (ram_target_sel)
                    2'b00: ram_q_b0[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                    2'b01: ram_k_b0[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                    2'b10: ram_v_b0[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                    default: ram_q_b0[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                endcase
            end else begin
                // Write to BANK 1
                case (ram_target_sel)
                    2'b00: ram_q_b1[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                    2'b01: ram_k_b1[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                    2'b10: ram_v_b1[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                    default: ram_q_b1[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0];
                endcase
            end
        end
    end

    // ------------------------------------------------------------------------
    // 4. SYNCHRONOUS BRAM READ LOGIC (EGRESS ROUTING TO SYSTOLIC ARRAY)
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (read_en) begin
            if (pp_bank_reg == 1'b0) begin
                // Read from BANK 1 (Active Read Bank)
                rdata_q_vec <= ram_q_b1[raddr_q];
                rdata_k_vec <= ram_k_b1[raddr_k];
                rdata_v_vec <= ram_v_b1[raddr_v];
            end else begin
                // Read from BANK 0 (Active Read Bank)
                rdata_q_vec <= ram_q_b0[raddr_q];
                rdata_k_vec <= ram_k_b0[raddr_k];
                rdata_v_vec <= ram_v_b0[raddr_v];
            end
        end
    end

    // ------------------------------------------------------------------------
    // 5. FORMAL ASSERTION SAFETY CHECKS (NON-SYNTHESIS)
    // ------------------------------------------------------------------------
    `ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            // Assert that Write and Read never target the same BRAM Bank simultaneously
            assert (!(pp_state == PP_WRITING && read_en && 
                      ((pp_bank_reg == 1'b0 && bank0_full) || (pp_bank_reg == 1'b1 && bank1_full))))
            else $error("[PING-PONG BRAM ERROR]: R/W Collision or Bank Overflow Detected!");
        end
    end
    `endif

endmodule
