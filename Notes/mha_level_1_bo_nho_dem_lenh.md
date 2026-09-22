# Bộ Nhớ Đệm Lệnh - MHA Level 1: Sub-system / Memory & Streaming Routing

Dưới đây là **LEVEL 1 — SUB-SYSTEM / MEMORY & STREAMING ROUTING** theo quy chuẩn phân rã cấu trúc Top-Down Hierarchy.
Phần này tập trung vào vi mạch quản lý luồng dữ liệu liên tục: **Mạch chống nghẽn Skid Buffer (AXI-Stream Adapter)**, **Hệ thống bộ đệm BRAM Ping-Pong đa ngân hàng (Multi-Bank Ping-Pong BRAM)** và **Máy trạng thái FSM tráo con trỏ đệm 1 chu kỳ clock**.

---

## 1. Sub-system Block Diagram (memory_streaming_subsys.sv)

Sơ đồ phác thảo chi tiết sự kết hợp giữa **Skid Buffer (chống mất gói AXI-Stream)**, **Bộ phân giải địa chỉ (Address Generator)**, và **Kiến trúc BRAM Ping-Pong Ngân hàng kép (Dual-Bank BRAM)**:

```text
========================================================================================================================
                            LEVEL 1: MEMORY & STREAMING SUB-SYSTEM (memory_streaming_subsys.sv)
========================================================================================================================

   AXI4-Stream Ingress Raw Data [63:0]                     Handshake Control Signals (s_axis_tvalid, s_axis_tready)
            │                                                                      │
            v                                                                      v
   +-------------------------------------------------------------------------------------------------------------------+
   | AXI4-STREAM SKID BUFFER ADAPTER (axis_adapter.sv)                                                                 |
   | - Holds incoming data beat when downstream BRAM writing is stalled                                                |
   | - Guarantees zero bubble cycles and compliant TVALID/TREADY protocol                                              |
   +---------------------------------------------------+---------------------------------------------------------------+
                                                       │
                                                       v Skid_Data [63:0] & Skid_Valid
   +---------------------------------------------------+---------------------------------------------------------------+
   | ADDRESS GENERATOR & BANK SWITCHING FSM (ping_pong_fsm.sv)                                                         |
   | - Auto-increments write addresses waddr_q, waddr_k, waddr_v                                                       |
   | - Toggles pp_bank_sel (Bank 0 / Bank 1) upon tile completion in 1 clock cycle                                     |
   +---------------------------------------------------+---------------------------------------------------------------+
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           │                                                       │
                           v (Bank 0 Active for Writing)                           v (Bank 1 Active for Reading)
   +---------------------------------------------------+   +---------------------------------------------------+
   | PING-PONG BRAM BANK 0                             |   | PING-PONG BRAM BANK 1                             |
   |  [ Q_RAM_0 ] [ K_RAM_0 ] [ V_RAM_0 ]              |   |  [ Q_RAM_1 ] [ K_RAM_1 ] [ V_RAM_1 ]              |
   |  (Storage Size: N x d_k bytes INT8)               |   |  (Storage Size: N x d_k bytes INT8)               |
   +---------------------------------------------------+   +---------------------------------------------------+
                                                       │
                                                       v
                                            rdata_q_vec, rdata_k_vec (32-bit Vector to Systolic Array)
```

---

### 2. Bảng Mô Tả Tín Hiệu & Routing Sub-system (LEVEL 1 Signal Table)

| Tên Tín hiệu (Signal) | Hướng | Độ rộng bit | Mô tả Chức năng & Định tuyến Routing |
| ------ | ------ | ------ | ------ |
| clk / rst_n | Input | 1 / 1 bit | Clock hệ thống 200 MHz và Reset bất đồng bộ tích cực mức thấp. |
| s_axis_tdata_raw | Input | 64 bits | Tín hiệu dữ liệu thô từ AXI DMA MM2S. |
| s_axis_tvalid_raw | Input | 1 bit | Cờ báo dữ liệu thô hợp lệ từ DMA. |
| s_axis_tready_raw | Output | 1 bit | Tín hiệu báo sẵn sàng nhận từ Skid Buffer (gửi ngược về DMA). |
| s_axis_tlast_raw | Input | 1 bit | Cờ báo beat cuối cùng của tile dữ liệu đầu vào. |
| skid_tdata | Internal | 64 bits | Dữ liệu đã qua Skid Buffer, sẵn sàng nạp vào BRAM. |
| skid_tvalid | Internal | 1 bit | Cờ Valid ổn định không bị giật nhịp khi downstream bận. |
| pp_bank_sel | Internal | 1 bit | Con trỏ chọn ngân hàng BRAM Ping-Pong (0: Bank 0 Write / Bank 1 Read; 1: Bank 1 Write / Bank 0 Read). |
| waddr_q / waddr_k | Internal | 8 bits | Địa chỉ ghi vào BRAM Q, K (0 đến 255). |
| we_q / we_k / we_v | Internal | 1 bit | Tín hiệu Enable ghi tương ứng cho từng BRAM. |
| raddr_q / raddr_k | Internal | 8 bits | Địa chỉ đọc BRAM từ mảng Systolic MAC Array. |
| rdata_q_vec | Output | 32 bits | Vector $4 \times \text{INT8}$ đọc từ BRAM Q đưa thẳng vào mảng Systolic. |
| rdata_k_vec | Output | 32 bits | Vector $4 \times \text{INT8}$ đọc từ BRAM K đưa thẳng vào mảng Systolic. |
| tile_write_done | Internal | 1 bit | Xung báo hoàn tất ghi đủ 1 Tile ($N \times d_k$) vào Bank đang Active. |
| tile_read_done | Internal | 1 bit | Xung báo Stage 1 GEMM đã đọc xong 1 Tile từ Bank Active. |

---

### 3. Sơ đồ ASMD Chart Điều Khiển Ping-Pong Buffer (ping_pong_fsm.sv)

Sơ đồ điều khiển chuyển đổi con trỏ đệm Ping-Pong đảm bảo **Zero-Latency Overhead** (không mất chu kỳ trống khi hoán đổi ngân hàng bộ nhớ):

```text
                               +-------------------+
                               |     ST_PP_RESET   |
                               | pp_bank_sel <= 0  |
                               | bank0_free  <= 1  |
                               | bank1_free  <= 1  |
                               +---------+---------+
                                         |
                                         v
                               +-------------------+
                               |     ST_PP_IDLE    | <----------------------+
                               +---------+---------+                        |
                                         |                                  |
                                (s_axis_tvalid == 1)                        |
                                 /               \                          |
                               YES                NO                        |
                               /                    \                       |
                              v                      +----------------------+
                    (bank_write_free?)
                      /          \                    
                    YES           NO
                    /               \                   
                   v                 v
         +-------------------+     +-------------------+
         |   ST_WRITE_ACTIVE |     |   ST_WAIT_COMPUTE |
         | Write to active   |     | Hold TREADY = 0   |
         | bank (0 or 1)     |     | (Backpressure)    |
         +---------+---------+     +---------+---------+
                   |                         |
                   v                         v
          (tile_write_done?)          (tile_read_done?)
           /            \               /          \        
         YES             NO           YES          NO
         /                \           /             \        
        v                  v         v               v
  +---------------+   (Keep Write) ST_WRITE_ACTIVE (Keep Wait)
  | ST_SWAP_BANKS |
  | pp_bank_sel <=|
  | ~pp_bank_sel  |
  | Trigger GEMM  |
  +-------+-------+
          |
          v
      ST_PP_IDLE
```

### 4. Mã RTL SystemVerilog Chống Nghẽn Chuẩn (axis_adapter.sv)

```verilog
module axis_adapter #(
    parameter int DATA_WIDTH = 64
)(
    input  logic                  clk,
    input  logic                  rst_n,
    // Ingress AXI-Stream
    input  logic [DATA_WIDTH-1:0] s_axis_tdata,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic                  s_axis_tlast,
    // Egress Internal Stream (Sạch)
    output logic [DATA_WIDTH-1:0] m_axis_tdata,
    output logic                  m_axis_tvalid,
    input  logic                  m_axis_tready,
    output logic                  m_axis_tlast
);

    logic [DATA_WIDTH-1:0] reg_main_data, reg_skid_data;
    logic                  reg_main_valid, reg_skid_valid;
    logic                  reg_main_last,  reg_skid_last;

    assign s_axis_tready = !reg_skid_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_main_valid <= 1'b0;
            reg_skid_valid <= 1'b0;
        end else begin
            if (s_axis_tready && s_axis_tvalid) begin
                if (m_axis_tready || !m_axis_tvalid) begin
                    reg_main_data  <= s_axis_tdata;
                    reg_main_valid <= 1'b1;
                    reg_main_last  <= s_axis_tlast;
                end else begin
                    reg_skid_data  <= s_axis_tdata;
                    reg_skid_valid <= 1'b1;
                    reg_skid_last  <= s_axis_tlast;
                end
            end else if (m_axis_tready) begin
                if (reg_skid_valid) begin
                    reg_main_data  <= reg_skid_data;
                    reg_main_valid <= 1'b1;
                    reg_main_last  <= reg_skid_last;
                    reg_skid_valid <= 1'b0;
                end else begin
                    reg_main_valid <= 1'b0;
                end
            end
        end
    end

    assign m_axis_tdata  = reg_main_data;
    assign m_axis_tvalid = reg_main_valid;
    assign m_axis_tlast  = reg_main_last;

endmodule
```
