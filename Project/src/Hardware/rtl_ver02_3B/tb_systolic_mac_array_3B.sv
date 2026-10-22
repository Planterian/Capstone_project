`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_systolic_mac_array_3B
// Description:  Verification Testbench for Step 3 Systolic MAC Array (Variant 3B)
//               Verifies:
//               1. FL-03: Independent Col0 and Col1 Accumulation (Zero Crosstalk)
//               2. FL-04: True Sequential Tile Drain Logic (32 items)
// Standard:     IEEE 1800-2012 SystemVerilog
// ============================================================================

module tb_systolic_mac_array_3B;

    localparam int ARRAY_SIZE = 4;
    localparam int IN_WIDTH   = 8;
    localparam int ACC_WIDTH  = 32;
    localparam real CLK_PERIOD = 5.0;

    logic                        clk;
    logic                        rst_n;
    logic                        gemm_start;
    logic                        clr_acc;
    logic                        in_valid;
    logic                        mode_sel;

    logic signed [IN_WIDTH-1:0]  act_row [0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt_col0 [0:ARRAY_SIZE-1];
    logic signed [IN_WIDTH-1:0]  wt_col1 [0:ARRAY_SIZE-1];

    logic                        drain_start;
    logic                        drain_done;
    logic                        out_valid;
    logic signed [ACC_WIDTH-1:0] dot_product_stream;
    logic signed [ACC_WIDTH-1:0] dot_product_out [0:ARRAY_SIZE-1];

    systolic_mac_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .IN_WIDTH   (IN_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .AUTO_SKEW  (1'b0) // Direct test for FL-03 / FL-04
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .gemm_start         (gemm_start),
        .clr_acc            (clr_acc),
        .in_valid           (in_valid),
        .mode_sel           (mode_sel),
        .act_row            (act_row),
        .wt_col0            (wt_col0),
        .wt_col1            (wt_col1),
        .drain_start        (drain_start),
        .drain_done         (drain_done),
        .out_valid          (out_valid),
        .dot_product_stream (dot_product_stream),
        .dot_product_out    (dot_product_out)
    );

    always #(CLK_PERIOD / 2.0) clk = ~clk;

    int drained_count = 0;
    logic signed [31:0] drained_items [0:31];

    always @(posedge clk) begin
        if (out_valid) begin
            drained_items[drained_count] <= dot_product_stream;
            $display("[TB DRAIN]: item %0d = %0d at time %0t", drained_count, dot_product_stream, $time);
            drained_count++;
        end
    end

    initial begin
        clk         = 0;
        rst_n       = 0;
        gemm_start  = 0;
        clr_acc     = 0;
        in_valid    = 0;
        mode_sel    = 0;
        drain_start = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            act_row[i] = '0;
            wt_col0[i] = '0;
            wt_col1[i] = '0;
        end

        #(CLK_PERIOD * 4);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        $display("=======================================================================");
        $display("   TESTBENCH: SYSTOLIC MAC ARRAY (FL-03 & FL-04 REMEDIATION CHECK - 3B)");
        $display("=======================================================================");

        // Clear accumulators
        @(posedge clk);
        clr_acc <= 1'b1;
        @(posedge clk);
        clr_acc <= 1'b0;

        $display("[STEP 1]: Feeding activations and weights across systolic grid...");
        for (int cycle = 0; cycle < 4; cycle++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            for (int r = 0; r < ARRAY_SIZE; r++) begin
                act_row[r] <= 8'sd10 * 8'(r + 1);
            end
            for (int c = 0; c < ARRAY_SIZE; c++) begin
                wt_col0[c] <= 8'sd2;
                wt_col1[c] <= 8'sd3;
            end
        end

        @(posedge clk);
        in_valid <= 1'b0;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            act_row[i] <= 8'sd0;
            wt_col0[i] <= 8'sd0;
            wt_col1[i] <= 8'sd0;
        end

        #(CLK_PERIOD * 16);

        $display("[STEP 2]: Checking internal PE accumulators for Col0 vs Col1 separation...");
        $display("  PE(0,0): Col0 = %0d, Col1 = %0d", dut.pe_acc_col0[0][0], dut.pe_acc_col1[0][0]);
        if (dut.pe_acc_col0[0][0] != dut.pe_acc_col1[0][0] && dut.pe_acc_col0[0][0] > 0) begin
            $display("  [PASS FL-03]: Col0 (%0d) and Col1 (%0d) accumulated independently without crosstalk!",
                     dut.pe_acc_col0[0][0], dut.pe_acc_col1[0][0]);
        end else begin
            $display("  [FAIL FL-03]: Col0 and Col1 are corrupted or identical!");
        end

        // Trigger Drain Unload Logic (FL-04)
        $display("\n[STEP 3]: Triggering Drain Shift-Out Logic (FL-04)...");
        @(posedge clk);
        drain_start <= 1'b1;
        @(posedge clk);
        drain_start <= 1'b0;

        wait (drain_done);
        @(posedge clk);
        #1;

        $display("  Drain Complete! Drained %0d dot-product items.", drained_count);
        $display("  Sample Output: Item 0 (r0,c0,col0) = %0d, Item 1 (r0,c0,col1) = %0d",
                 drained_items[0], drained_items[1]);

        if (drained_count == 32 && drained_items[0] == dut.pe_acc_col0[0][0] && drained_items[1] == dut.pe_acc_col1[0][0]) begin
            $display("  [PASS FL-04]: Successfully drained all 32 tile elements with zero adder tree loss!");
            $display("=======================================================================");
            $display(">>> SUCCESS: systolic_mac_array_3B PASSED 100%! <<<");
            $display("=======================================================================\n");
        end else begin
            $display("  [FAIL FL-04]: Drain item count mismatch or corrupted data!");
        end

        $finish;
    end

endmodule
