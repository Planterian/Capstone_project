`timescale 1ns / 1ps
// ============================================================================
// Module Name:   tb_scale_unit
// Description:  Bit-Exact Verification for scale_unit.sv against golden vectors
// ============================================================================

module tb_scale_unit;
    logic clk;
    logic rst_n;
    logic in_valid;
    logic signed [31:0] in_score;
    logic signed [15:0] scale_m;
    logic [4:0] scale_e;
    logic [3:0] shift_val;
    logic out_valid;
    logic signed [15:0] scaled_score;

    scale_unit #(
        .IN_WIDTH(32),
        .OUT_WIDTH(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_score(in_score),
        .scale_m(scale_m),
        .scale_e(scale_e),
        .shift_val(shift_val),
        .out_valid(out_valid),
        .scaled_score(scaled_score)
    );

    always #2.5 clk = ~clk;

    logic signed [31:0] gold_gemm [0:1023];
    logic signed [15:0] gold_scaled [0:1023];
    int match_cnt = 0;
    int err_cnt = 0;
    int out_idx = 0;

    always @(posedge clk) begin
        if (out_valid) begin
            if (scaled_score == gold_scaled[out_idx]) begin
                match_cnt++;
            end else begin
                err_cnt++;
            end
            out_idx++;
        end
    end

    initial begin
        clk = 0;
        rst_n = 0;
        in_valid = 0;
        in_score = 0;
        scale_m = 16'sd27640;
        scale_e = 5'd22;
        shift_val = 4'd2;
        $readmemh("golden_gemm_qk.hex", gold_gemm);
        $readmemh("golden_scaled_score.hex", gold_scaled);
        #20;
        rst_n = 1;
        #10;
        for (int i = 0; i < 1024; i++) begin
            @(posedge clk);
            in_valid <= 1;
            in_score <= gold_gemm[i];
        end
        @(posedge clk);
        in_valid <= 0;
        #50;
        $display("SCALE_UNIT RESULTS: Matched=%0d, Errors=%0d", match_cnt, err_cnt);
        if (err_cnt == 0 && match_cnt == 1024)
            $display(">>> STATUS: SCALE_UNIT 100%% BIT-EXACT PASS! <<<");
        else
            $display(">>> STATUS: SCALE_UNIT FAILED! <<<");
        $finish;
    end
endmodule
