module uart_tx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 115200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       start,
    output reg        tx   = 1'b1,
    output reg        busy = 1'b0
);

    localparam integer BIT_CYCLES = (CLK_HZ + BAUD/2) / BAUD;

    reg [31:0] baud_count = 0;
    reg [3:0]  bit_index  = 0;
    reg [9:0]  frame      = 10'h3FF;

    always @(posedge clk) begin
        if (rst) begin
            tx         <= 1'b1;
            busy       <= 1'b0;
            baud_count <= 0;
            bit_index  <= 0;
            frame      <= 10'h3FF;
        end else if (!busy) begin
            tx <= 1'b1;

            if (start) begin
                // Start bit, 8 data bits LSB first, stop bit.
                frame      <= {1'b1, data, 1'b0};
                tx         <= 1'b0;
                busy       <= 1'b1;
                baud_count <= 0;
                bit_index  <= 0;
            end
        end else begin
            if (baud_count == BIT_CYCLES - 1) begin
                baud_count <= 0;

                if (bit_index == 9) begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                end else begin
                    bit_index <= bit_index + 1'b1;
                    frame     <= {1'b1, frame[9:1]};
                    tx        <= frame[1];
                end
            end else begin
                baud_count <= baud_count + 1'b1;
            end
        end
    end

endmodule