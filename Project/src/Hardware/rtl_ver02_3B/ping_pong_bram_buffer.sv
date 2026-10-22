`timescale 1ns / 1ps
// ============================================================================
// Module Name:   ping_pong_bram_buffer
// Description:  Multi-Bank Ping-Pong BRAM Buffer Manager for MHA Accelerator.
//               Manages two memory banks (Bank 0 and Bank 1) for Q, K, V matrices.
//               Supports 64-bit Ingress AXI-Stream (8x INT8 per beat) without
//               bit loss, with an Automatic FSM routing Q -> K -> V (512 beats each),
//               and 32-bit Vector Egress (4x INT8) to the Systolic MAC Array.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       AMD Kria KV260 / Zynq UltraScale+ MPSoC
// Version:      v2.3 (Zero-Bubble Ingress Aligned, Auto Q->K->V FSM, Ping-Pong Protected)
// ============================================================================

module ping_pong_bram_buffer #(
    parameter int DATA_WIDTH       = 64,   // AXI-Stream Ingress Data Width (8x INT8)
    parameter int RAM_WIDTH        = 32,   // Systolic Array Vector Width (4x INT8)
    parameter int RAM_DEPTH        = 2048, // 64-bit word depth per Q/K/V RAM
    parameter int ADDR_WIDTH       = 11,   // log2(RAM_DEPTH) = 11 bits
    parameter int BEATS_PER_MATRIX = 512,  // 256 tokens * 16 bytes / 8 bytes/beat = 512 beats
    parameter bit AUTO_ROUTE       = 1'b1  // 1: Auto FSM Q->K->V, 0: Manual ram_target_sel
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
    input  logic [1:0]              ram_target_sel, // Manual override (if AUTO_ROUTE=0)

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
    input  logic                    tile_read_done, // Pulse when Core completes reading current tile
    output logic                    tile_write_done,// Pulse when 1 Tile is fully written to active bank
    output logic                    bank_write_free,// Active bank is ready for writing
    output logic                    bank_read_ready,// Active bank has valid tile data for reading
    output logic                    pp_bank_sel,    // 0: Bank 0 Write / Bank 1 Read | 1: Bank 1 Write / Bank 0 Read
    output logic [1:0]              current_target_ram // 0: Q_RAM, 1: K_RAM, 2: V_RAM
);

    // ------------------------------------------------------------------------
    // 1. DUAL-BANK BRAM ARRAY DECLARATIONS (64-bit wide storage)
    // ------------------------------------------------------------------------
    // Bank 0 RAMs (64-bit data width per entry)
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram_q_b0 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram_k_b0 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram_v_b0 [0:RAM_DEPTH-1];

    // Bank 1 RAMs (64-bit data width per entry)
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram_q_b1 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram_k_b1 [0:RAM_DEPTH-1];
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram_v_b1 [0:RAM_DEPTH-1];

    // Simulation-safe BRAM initialization
    initial begin
        for (int i = 0; i < RAM_DEPTH; i++) begin
            ram_q_b0[i] = '0;
            ram_k_b0[i] = '0;
            ram_v_b0[i] = '0;
            ram_q_b1[i] = '0;
            ram_k_b1[i] = '0;
            ram_v_b1[i] = '0;
        end
    end

    // Ping-Pong Status Flags
    logic bank0_full, bank1_full;
    logic pp_bank_reg;

    assign pp_bank_sel     = pp_bank_reg;
    assign bank_write_free = (pp_bank_reg == 1'b0) ? !bank0_full : !bank1_full;
    assign bank_read_ready = (pp_bank_reg == 1'b0) ?  bank1_full :  bank0_full;

    // ------------------------------------------------------------------------
    // 2. INGRESS SEQUENTIAL ROUTING FSM (Q -> K -> V) & PING-PONG CONTROLLER
    // ------------------------------------------------------------------------
    typedef enum logic [1:0] {
        ING_LOAD_Q   = 2'b00,
        ING_LOAD_K   = 2'b01,
        ING_LOAD_V   = 2'b10,
        ING_SWAP     = 2'b11
    } ing_state_t;

    ing_state_t ing_state;

    // Address and Beat Counters
    logic [ADDR_WIDTH-1:0] waddr_cnt;
    logic [1:0]            active_target;

    assign current_target_ram = (AUTO_ROUTE) ? active_target : ram_target_sel;
    assign s_axis_tready      = bank_write_free && (ing_state != ING_SWAP);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ing_state       <= ING_LOAD_Q;
            pp_bank_reg     <= 1'b0; // Start: Write Bank 0, Bank 1 Empty
            bank0_full      <= 1'b0;
            bank1_full      <= 1'b0;
            waddr_cnt       <= '0;
            active_target   <= 2'b00;
            tile_write_done <= 1'b0;
        end else begin
            tile_write_done <= 1'b0;

            // Handle Read Completion
            if (tile_read_done) begin
                if (pp_bank_reg == 1'b0)
                    bank1_full <= 1'b0;
                else
                    bank0_full <= 1'b0;
            end

            case (ing_state)
                ING_LOAD_Q: begin
                    active_target <= 2'b00;
                    if (s_axis_tvalid && s_axis_tready) begin
                        if (waddr_cnt == (BEATS_PER_MATRIX - 1) || s_axis_tlast) begin
                            waddr_cnt     <= '0;
                            active_target <= 2'b01; // Advance to K_RAM
                            ing_state     <= ING_LOAD_K;
                        end else begin
                            waddr_cnt <= waddr_cnt + 1'b1;
                        end
                    end
                end

                ING_LOAD_K: begin
                    active_target <= 2'b01;
                    if (s_axis_tvalid && s_axis_tready) begin
                        if (waddr_cnt == (BEATS_PER_MATRIX - 1) || s_axis_tlast) begin
                            waddr_cnt     <= '0;
                            active_target <= 2'b10; // Advance to V_RAM
                            ing_state     <= ING_LOAD_V;
                        end else begin
                            waddr_cnt <= waddr_cnt + 1'b1;
                        end
                    end
                end

                ING_LOAD_V: begin
                    active_target <= 2'b10;
                    if (s_axis_tvalid && s_axis_tready) begin
                        if (waddr_cnt == (BEATS_PER_MATRIX - 1) || s_axis_tlast) begin
                            waddr_cnt       <= '0;
                            tile_write_done <= 1'b1;
                            ing_state       <= ING_SWAP;

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

                ING_SWAP: begin
                    // Swap bank pointers if read side has completed reading previous tile
                    // or if the read side was idle/empty (!bank_read_ready)
                    if (tile_read_done || !bank_read_ready) begin
                        pp_bank_reg   <= ~pp_bank_reg;
                        active_target <= 2'b00; // Ready for Q of next tile
                        waddr_cnt     <= '0;

                        // Clear FULL status of newly read bank
                        if (pp_bank_reg == 1'b0)
                            bank1_full <= 1'b0;
                        else
                            bank0_full <= 1'b0;

                        ing_state <= ING_LOAD_Q;
                    end
                end

                default: ing_state <= ING_LOAD_Q;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // 3. SYNCHRONOUS BRAM WRITE LOGIC (FULL 64-BIT INGRESS WRITING)
    // ------------------------------------------------------------------------
    wire [1:0] write_target = (AUTO_ROUTE) ? active_target : ram_target_sel;

    always_ff @(posedge clk) begin
        if (s_axis_tvalid && s_axis_tready) begin
            if (pp_bank_reg == 1'b0) begin
                // Write to BANK 0
                case (write_target)
                    2'b00:   ram_q_b0[waddr_cnt] <= s_axis_tdata;
                    2'b01:   ram_k_b0[waddr_cnt] <= s_axis_tdata;
                    2'b10:   ram_v_b0[waddr_cnt] <= s_axis_tdata;
                    default: ram_q_b0[waddr_cnt] <= s_axis_tdata;
                endcase
            end else begin
                // Write to BANK 1
                case (write_target)
                    2'b00:   ram_q_b1[waddr_cnt] <= s_axis_tdata;
                    2'b01:   ram_k_b1[waddr_cnt] <= s_axis_tdata;
                    2'b10:   ram_v_b1[waddr_cnt] <= s_axis_tdata;
                    default: ram_q_b1[waddr_cnt] <= s_axis_tdata;
                endcase
            end
        end
    end

    // ------------------------------------------------------------------------
    // 4. SYNCHRONOUS BRAM READ LOGIC (SUBWORD MULTIPLEXED 32-BIT VECTORS)
    // ------------------------------------------------------------------------
    generate
        if (RAM_WIDTH == 32) begin : gen_subword_read
            // 32-bit vector read from 64-bit BRAM words (raddr[0] selects upper/lower 32-bit)
            always_ff @(posedge clk) begin
                if (read_en && bank_read_ready) begin
                    if (pp_bank_reg == 1'b0) begin
                        // Read from BANK 1 (Active Read Bank)
                        rdata_q_vec <= raddr_q[0] ? ram_q_b1[raddr_q >> 1][63:32] : ram_q_b1[raddr_q >> 1][31:0];
                        rdata_k_vec <= raddr_k[0] ? ram_k_b1[raddr_k >> 1][63:32] : ram_k_b1[raddr_k >> 1][31:0];
                        rdata_v_vec <= raddr_v[0] ? ram_v_b1[raddr_v >> 1][63:32] : ram_v_b1[raddr_v >> 1][31:0];
                    end else begin
                        // Read from BANK 0 (Active Read Bank)
                        rdata_q_vec <= raddr_q[0] ? ram_q_b0[raddr_q >> 1][63:32] : ram_q_b0[raddr_q >> 1][31:0];
                        rdata_k_vec <= raddr_k[0] ? ram_k_b0[raddr_k >> 1][63:32] : ram_k_b0[raddr_k >> 1][31:0];
                        rdata_v_vec <= raddr_v[0] ? ram_v_b0[raddr_v >> 1][63:32] : ram_v_b0[raddr_v >> 1][31:0];
                    end
                end else if (!read_en) begin
                    rdata_q_vec <= '0;
                    rdata_k_vec <= '0;
                    rdata_v_vec <= '0;
                end
            end
        end else begin : gen_direct_read
            // Direct 64-bit read
            always_ff @(posedge clk) begin
                if (read_en && bank_read_ready) begin
                    if (pp_bank_reg == 1'b0) begin
                        rdata_q_vec <= ram_q_b1[raddr_q];
                        rdata_k_vec <= ram_k_b1[raddr_k];
                        rdata_v_vec <= ram_v_b1[raddr_v];
                    end else begin
                        rdata_q_vec <= ram_q_b0[raddr_q];
                        rdata_k_vec <= ram_k_b0[raddr_k];
                        rdata_v_vec <= ram_v_b0[raddr_v];
                    end
                end else if (!read_en) begin
                    rdata_q_vec <= '0;
                    rdata_k_vec <= '0;
                    rdata_v_vec <= '0;
                end
            end
        end
    endgenerate

endmodule
