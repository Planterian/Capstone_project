module vcu118_uart_top (
    input  wire clk300_p,
    input  wire clk300_n,
    input  wire uart_rx_i,
    output wire uart_tx_o
);

    wire clk100;
    wire locked;

    clk_wiz_0 u_clock (
        .clk_in1_p (clk300_p),
        .clk_in1_n (clk300_n),
        .reset     (1'b0),
        .clk_out1  (clk100),
        .locked    (locked)
    );

    // Assert reset when the clock loses lock.
    // Release reset synchronously after four clock cycles.
    (* ASYNC_REG = "TRUE" *)
    reg [3:0] reset_pipe = 4'b1111;

    always @(posedge clk100 or negedge locked) begin
        if (!locked)
            reset_pipe <= 4'b1111;
        else
            reset_pipe <= {reset_pipe[2:0], 1'b0};
    end

    wire rst = reset_pipe[3];

    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_HZ (100_000_000),
        .BAUD   (115200)
    ) u_rx (
        .clk   (clk100),
        .rst   (rst),
        .rx    (uart_rx_i),
        .data  (rx_data),
        .valid (rx_valid)
    );

    reg  [7:0] tx_data  = 0;
    reg        tx_start = 0;
    wire       tx_busy;

    uart_tx #(
        .CLK_HZ (100_000_000),
        .BAUD   (115200)
    ) u_tx (
        .clk   (clk100),
        .rst   (rst),
        .data  (tx_data),
        .start (tx_start),
        .tx    (uart_tx_o),
        .busy  (tx_busy)
    );

    localparam [1:0]
        IDLE      = 0,
        LAUNCH    = 1,
        WAIT_BUSY = 2,
        WAIT_DONE = 3;

    reg [1:0]  state       = IDLE;
    reg [1:0]  char_index  = 0;
    reg [15:0] pending     = 0;

    wire line_received = rx_valid && (rx_data == 8'h0A);

    wire reply_done = (state == WAIT_DONE)
                   && !tx_busy
                   && (char_index == 3);

    always @(posedge clk100) begin
        if (rst) begin
            pending <= 0;
        end else begin
            // Count complete lines waiting for an OK reply.
            case ({line_received, reply_done})
                2'b10: begin
                    if (pending != 16'hFFFF)
                        pending <= pending + 1'b1;
                end

                2'b01: begin
                    if (pending != 0)
                        pending <= pending - 1'b1;
                end

                default: pending <= pending;
            endcase
        end
    end

    always @(posedge clk100) begin
        tx_start <= 1'b0;

        if (rst) begin
            state      <= IDLE;
            char_index <= 0;
            tx_data    <= 0;
            tx_start   <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (pending != 0) begin
                        char_index <= 0;
                        state      <= LAUNCH;
                    end
                end

                LAUNCH: begin
                    case (char_index)
                        0: tx_data <= 8'h4F; // O
                        1: tx_data <= 8'h4B; // K
                        2: tx_data <= 8'h0D; // CR
                        3: tx_data <= 8'h0A; // LF
                    endcase

                    tx_start <= 1'b1;
                    state    <= WAIT_BUSY;
                end

                WAIT_BUSY: begin
                    if (tx_busy)
                        state <= WAIT_DONE;
                end

                WAIT_DONE: begin
                    if (!tx_busy) begin
                        if (char_index == 3) begin
                            state <= IDLE;
                        end else begin
                            char_index <= char_index + 1'b1;
                            state      <= LAUNCH;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule