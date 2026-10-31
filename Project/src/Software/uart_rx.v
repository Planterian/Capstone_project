module uart_rx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 115200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data  = 0,
    output reg        valid = 0
);

    localparam integer BIT_CYCLES =
        (CLK_HZ + BAUD/2) / BAUD;
    localparam integer HALF_CYCLES = BIT_CYCLES / 2;

    localparam [2:0]
        IDLE    = 0,
        START   = 1,
        RECEIVE = 2,
        STOP    = 3,
        RECOVER = 4;

    // Synchronize the asynchronous UART input.
    (* ASYNC_REG = "TRUE" *) reg rx_meta = 1'b1;
    (* ASYNC_REG = "TRUE" *) reg rx_sync = 1'b1;

    reg [2:0]  state     = IDLE;
    reg [31:0] count     = 0;
    reg [2:0]  bit_index = 0;
    reg [7:0]  buffer    = 0;

    always @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always @(posedge clk) begin
        valid <= 1'b0;

        if (rst) begin
            state     <= IDLE;
            count     <= 0;
            bit_index <= 0;
            buffer    <= 0;
            data      <= 0;
        end else begin
            case (state)
                IDLE: begin
                    count <= 0;
                    if (!rx_sync)
                        state <= START;
                end

                START: begin
                    if (count == HALF_CYCLES - 1) begin
                        count <= 0;
                        if (!rx_sync) begin
                            bit_index <= 0;
                            state     <= RECEIVE;
                        end else begin
                            state <= IDLE;
                        end
                    end else begin
                        count <= count + 1'b1;
                    end
                end

                RECEIVE: begin
                    if (count == BIT_CYCLES - 1) begin
                        count             <= 0;
                        buffer[bit_index] <= rx_sync;

                        if (bit_index == 7)
                            state <= STOP;
                        else
                            bit_index <= bit_index + 1'b1;
                    end else begin
                        count <= count + 1'b1;
                    end
                end

                STOP: begin
                    if (count == BIT_CYCLES - 1) begin
                        count <= 0;
                        if (rx_sync) begin
                            data  <= buffer;
                            valid <= 1'b1;
                            state <= IDLE;
                        end else begin
                            // Invalid stop bit: discard this byte.
                            state <= RECOVER;
                        end
                    end else begin
                        count <= count + 1'b1;
                    end
                end

                RECOVER: begin
                    if (rx_sync)
                        state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule