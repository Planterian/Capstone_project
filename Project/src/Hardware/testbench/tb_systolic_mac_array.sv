`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_systolic_mac_array
// Description:  Automated Self-Checking SystemVerilog Testbench for 2D 
//               Systolic MAC Array Engine with 2x INT8 DSP Packing.
// Standard:     IEEE 1800-2012 SystemVerilog
// Target:       Simulation Verification (ModelSim / Vivado Simulator / EDA)
// ============================================================================

module tb_systolic_mac_array;

    // ------------------------------------------------------------------------
    // 1. PARAMETERS & CLOCK DEFINITIONS
    // ------------------------------------------------------------------------
    localparam int ARRAY_SIZE   = 4;
    localparam int IN_WIDTH     = 8;
    localparam int ACC_WIDTH    = 32;
    localparam int NUM_PATTERNS = 8;
    localparam real CLK_PERIOD  = 5.0; // 200 MHz Clock (5.0 ns period)

    // Signals for DUT Interface
    logic                     clk;
    logic                     rst_n;
    logic                     gemm_start;
    logic                     clr_acc;
    logic                     in_valid;
    logic                     mode_sel;

    logic signed [IN_WIDTH-1:0] act_row [0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0] wt_col0 [0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0] wt_col1 [0:ARRAY_SIZE-1];

    logic                     out_valid;
    logic signed [ACC_WIDTH-1:0] dot_product_out [0:ARRAY_SIZE-1];

    // Test Memory Arrays for .hex Loading
    logic [ARRAY_SIZE*IN_WIDTH-1:0] act_mem [0:NUM_PATTERNS-1];
    logic [ARRAY_SIZE*IN_WIDTH-1:0] wt0_mem [0:NUM_PATTERNS-1];
    logic [ARRAY_SIZE*IN_WIDTH-1:0] wt1_mem [0:NUM_PATTERNS-1];
    logic signed [ACC_WIDTH-1:0]    exp_out_mem [0:ARRAY_SIZE-1];

    // Verification Counters
    int error_count = 0;
    int pass_count  = 0;

    // ------------------------------------------------------------------------
    // 2. DUT INSTANTIATION
    // ------------------------------------------------------------------------
    systolic_mac_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .IN_WIDTH   (IN_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .gemm_start      (gemm_start),
        .clr_acc         (clr_acc),
        .in_valid        (in_valid),
        .mode_sel        (mode_sel),
        .act_row         (act_row),
        .wt_col0         (wt_col0),
        .wt_col1         (wt_col1),
        .out_valid       (out_valid),
        .dot_product_out (dot_product_out)
    );

    // ------------------------------------------------------------------------
    // 3. CLOCK GENERATOR (200 MHz)
    // ------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2.0) clk = ~clk;
    end

    // ------------------------------------------------------------------------
    // 4. MAIN STIMULUS & VERIFICATION SEQUENCE
    // ------------------------------------------------------------------------
    initial begin
        $display("=======================================================================");
        $display("   STARTING AUTOMATED TESTBENCH FOR SYSTOLIC MAC ARRAY (2x INT8 PACKING)");
        $display("=======================================================================");

        // Step 1: Initialize Control Signals & Memory
        rst_n      = 1'b0;
        gemm_start = 1'b0;
        clr_acc    = 1'b0;
        in_valid   = 1'b0;
        mode_sel   = 1'b0; // 0: Stage 1 QK^T Mode

        for (int i = 0; i < ARRAY_SIZE; i++) begin
            act_row[i] = '0;
            wt_col0[i] = '0;
            wt_col1[i] = '0;
        end

        // Step 2: Load Sample .hex Test Vectors
        $display("[TB INFO]: Loading test vectors from .hex files...");
        $readmemh("act_vectors.hex",  act_mem);
        $readmemh("wt0_vectors.hex",  wt0_mem);
        $readmemh("wt1_vectors.hex",  wt1_mem);
        $readmemh("expected_out.hex", exp_out_mem);

        // Step 3: Apply Reset
        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        $display("[TB INFO]: Reset released at time %0t ns", $time);

        // Step 4: Clear Accumulators
        @(posedge clk);
        clr_acc = 1'b1;
        @(posedge clk);
        clr_acc = 1'b0;
        gemm_start = 1'b1;
        @(posedge clk);
        gemm_start = 1'b0;

        // Step 5: Feed Vectors into Systolic Array
        $display("[TB INFO]: Streaming %0d test vectors into Systolic Array...", NUM_PATTERNS);
        for (int t = 0; t < NUM_PATTERNS; t++) begin
            @(posedge clk);
            in_valid = 1'b1;
            
            // Unpack 32-bit row/col vectors into array elements
            for (int k = 0; k < ARRAY_SIZE; k++) begin
                act_row[k] = $signed(act_mem[t][k*8 +: 8]);
                wt_col0[k] = $signed(wt0_mem[t][k*8 +: 8]);
                wt_col1[k] = $signed(wt1_mem[t][k*8 +: 8]);
            end
        end

        // De-assert in_valid after streaming
        @(posedge clk);
        in_valid = 1'b0;
        for (int k = 0; k < ARRAY_SIZE; k++) begin
            act_row[k] = '0;
            wt_col0[k] = '0;
            wt_col1[k] = '0;
        end

        // Step 6: Wait for Computation Pipeline Completion
        $display("[TB INFO]: Waiting for execution pipeline & Adder Tree completion...");
        wait (out_valid == 1'b1);
        @(posedge clk); // Sample output

        // Step 7: Self-Checking Output Comparison
        $display("\n-----------------------------------------------------------------------");
        $display("   OUTPUT VERIFICATION RESULT (COMPARING DUT vs GOLDEN EXPECTED)");
        $display("-----------------------------------------------------------------------");

        for (int r = 0; r < ARRAY_SIZE; r++) begin
            $write("Row %0d Output: DUT = %0d | Expected = %0d ---> ", 
                   r, dot_product_out[r], exp_out_mem[r]);
            
            if (dot_product_out[r] == exp_out_mem[r]) begin
                $display("[ MATCH - PASS ]");
                pass_count++;
            end else begin
                $display("[ MISMATCH - FAIL ]");
                error_count++;
            end
        end

        // Step 8: Final Summary & Report
        $display("=======================================================================");
        if (error_count == 0) begin
            $display("   *** TEST PASSED SUCCESSFULLY (%0d/%0d Rows Matched) ***", pass_count, ARRAY_SIZE);
        end else begin
            $display("   *** TEST FAILED (%0d Errors Detected) ***", error_count);
        end
        $display("=======================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    // Timeout Monitor Safety Check
    initial begin
        #(CLK_PERIOD * 500);
        $display("\n[TB FATAL]: Simulation Timed Out! Check Valid handshake signals.");
        $finish;
    end

endmodule
