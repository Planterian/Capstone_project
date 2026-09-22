# Kiến Trúc MHA Level 0: Top-Level System Architecture, ASMD & Bandwidth Analysis

## 📋 KẾ HOẠCH PHÁT TRIỂN HỆ THỐNG TOP-DOWN HIERARCHY

Toàn bộ hệ thống tăng tốc **Multi-Head Attention (MHA) Core** được chia làm 4 cấp độ từ ngoài vào trong:

1. **LEVEL 0**: Top-Level System Architecture, Giao diện AXI4-Lite/AXI4-Stream, Bảng Định tuyến Tín hiệu Data/Control, Sơ đồ ASMD Chart Trung tâm, và Phân tích Băng thông Bus.
2. **LEVEL 1**: Sub-system Bộ nhớ & Luồng Streaming (Ping-Pong BRAM Manager, Skid Buffer chống nghẽn Backpressure, FSM tráo con trỏ đệm).
3. **LEVEL 2**: Compute Engine — GEMM & DSP Packing (Systolic MAC Array Output-Stationary, DSP48E2 $18 \times 27$-bit 2x INT8, Pipelined Adder Tree).
4. **LEVEL 3**: Pipelines Phi tuyến & Activation (Scaler Unit dịch bit ASR 0-DSP, Hardware Softmax Engine, Re-quantization Dyadic Scaling).

---

## 🏛️ LEVEL 0: SYSTEM ARCHITECTURE, ASMD & BANDWIDTH MODEL

### 1. Sơ đồ Khối Cấp Cao (Top-Level Architecture Block Diagram - attention_core_top.sv)

```text
========================================================================================================================
                                LEVEL 0: TOP-LEVEL RTL ARCHITECTURE (attention_core_top.sv)
========================================================================================================================

       ARM Processing System (PS) / Host                                  AXI DMA Controller (Simple Mode)
  +-------------------------------------------+                        +------------------------------------+
  | AXI4-Lite Master (Memory-Mapped)          |                        | MM2S (Read DDR4)   S2MM (Write DDR4)|
  | Regs: 0x00=CTRL, 0x04=NBYTES, 0x10=N, ... |                        | Stream Master      Stream Slave    |
  +---------------------+---------------------+                        +---------+------------------+---------------+
                        |                                                        |                  ^
                        | AXI4-Lite Bus                                          | AXI4-Stream      | AXI4-Stream
                        | (32-bit Control/Status)                                | RX (64-bit)      | TX (64-bit)
                        v                                                        v                  |
  +------------------------------------------------------------------------------------------------------------------+
  | ATTENTION ACCELERATOR TOP SHELL (attention_core_top.sv)                                                         |
  |                                                                                                                  |
  |  +------------------------------------------------------------------------------------------------------------+  |
  |  | CONTROL PATH: AXI4-Lite Slave Register File & Central ASMD Controller (axi_lite_regs.sv)                |  |
  |  |                                                                                                            |  |
  |  |  Registers (CSR):                 Internal Control Signals:                                                |  |
  |  |   - 0x00: CTRL/STATUS [START/DONE] - start_pulse  ──► [Start FSM]        - reg_nbytes ──► [Packet Size] |  |
  |  |   - 0x04: NBYTES                  - clear_buf    ──► [Reset Buffers]    - reg_tokens ──► [Seq Len N]   |  |
  |  |   - 0x10: SEQ_LEN (N=196)         - core_busy    ◄── [Compute Status]   - reg_hdim   ──► [Head Dim d_k] |  |
  |  |   - 0x18: HEAD_DIM (d_k=32)       - done_pulse   ◄── [Pipeline Done]    - reg_heads  ──► [Num Heads H] |  |
  |  +-------------------------------------+----------------------------------------------------------------------+  |
  |                                        |                                                                         |
  |                                        | Control & Config Routing (start_tile, mode_sel, ping_pong_sel)          |
  |                                        v                                                                         |
  |  +------------------------------------------------------------------------------------------------------------+  |
  |  | DATA PATH: PIPELINED MULTI-HEAD ATTENTION ENGINE                                                          |  |
  |  |                                                                                                            |  |
  |  |   AXI-Stream RX Adapter                                                                                    |  |
  |  |   +-------------------+                                                                                    |  |
  |  |   | axis_adapter_rx   |====/ (64-bit Internal Data Stream / Handshake Control) ===+                         |  |
  |  |   +-------------------+   64                                                          |                        |  |
  |  |                                                                                       v                        |  |
  |  |                                                                         +----------------------------+         |  |
  |  |                                                                         | Input Ping-Pong BRAM       |         |  |
  |  |                                                                         | (ping_pong_bram_buffer.sv) |         |  |
  |  |                                                                         | [Q_RAM] [K_RAM] [V_RAM]    |         |  |
  |  |                                                                         +--------------+-------------+         |  |
  |  |                                                                                        | (INT8 Vector)         |  |
  |  |                                                                                        v                       |  |
  |  |                                                                         +----------------------------+         |  |
  |  |                                                                         | STAGE 1: QK^T GEMM Engine  |         |  |
  |  |                                                                         | (systolic_mac_array.sv)    |         |  |
  |  |                                                                         | [2x INT8 DSP Packing]      |         |  |
  |  |                                                                         +--------------+-------------+         |  |
  |  |                                                                                        | (INT32 Partial Sum)   |  |
  |  |                                                                                        v                       |  |
  |  |                                                                         +----------------------------+         |  |
  |  |                                                                         | STAGE 2: Scaler Unit       |         |  |
  |  |                                                                         | (scale_unit.sv)            |         |  |
  |  |                                                                         | [ASR Shift 1/sqrt(d_k)]    |         |  |
  |  |                                                                         +--------------+-------------+         |  |
  |  |                                                                                        | (INT16 Scaled Score)  |  |
  |  |                                                                                        v                       |  |
  |  |                                                                         +----------------------------+         |  |
  |  |                                                                         | STAGE 3: Hardware Softmax  |         |  |
  |  |                                                                         | (softmax_lut.sv)           |         |  |
  |  |                                                                         | [Max-Sub + Exp LUT + Div]  |         |  |
  |  |                                                                         +--------------+-------------+         |  |
  |  |                                                                                        | (INT8 Probability)    |  |
  |  |                                                                                        v                       |  |
  |  |                                                                         +----------------------------+         |  |
  |  |                                                                         | STAGE 4: Score x V GEMM    |         |  |
  |  |                                                                         | (systolic_mac_array.sv)    |         |  |
  |  |                                                                         +--------------+-------------+         |  |
  |  |                                                                                        | (INT8 Output Tokens)  |  |
  |  |                                                                                        v                       |  |
  |  |                                                                         +----------------------------+         |  |
  |  |                                                                         | Output Ping-Pong Buffer    |         |  |
  |  |                                                                         +--------------+-------------+         |  |
  |  |                                                                                        |                       |  |
  |  |   AXI-Stream TX Adapter                                                                |                       |  |
  |  |   +-------------------+                                                                |                       |  |
  |  |   | axis_adapter_tx   |<====/ (64-bit Internal Data / Handshake) =================-----+                       |  |
  |  |   +-------------------+   64                                                                                   |  |
  |  +----------------------------------------------------------------------------------------------------------------+  |
  +----------------------------------------------------------------------------------------------------------------------+
```

### 2. Bảng Mô Tả Tín Hiệu & Định Tuyến (Interface Signal & Routing Table)

| Nhóm Giao diện | Tên Tín hiệu (Signal) | Hướng | Độ rộng bit | Ý nghĩa Kỹ thuật & Định tuyến |
| :--- | :--- | :--- | :--- | :--- |
| **Clock & Reset** | `clk` | Input | 1 bit | Xung clock hệ thống PL (Tần số danh định 200 MHz). |
| | `rst_n` | Input | 1 bit | Reset tích cực mức thấp (Asynchronous Assert, Sync Deassert). |
| **AXI4-Lite Slave**<br />*(Control Plane)* | `s_axi_awaddr` | Input | 8 bits | Địa chỉ ghi Memory-Mapped (Offsets 0x00 - 0x1C). |
| | `s_axi_awvalid` / `s_axi_awready` | In / Out | 1 bit | Handshake kênh địa chỉ ghi (Address Write). |
| | `s_axi_wdata` | Input | 32 bits | Dữ liệu ghi cấu hình tham số từ CPU ARM. |
| | `s_axi_wstrb` | Input | 4 bits | Strobe chọn byte ghi hợp lệ (Byte Enables). |
| | `s_axi_wvalid` / `s_axi_wready` | In / Out | 1 bit | Handshake kênh dữ liệu ghi (Data Write). |
| | `s_axi_bresp` / `s_axi_bvalid` / `s_axi_bready` | Out / In | 2 / 1 / 1 | Kênh phản hồi trạng thái ghi (Write Response). |
| | `s_axi_araddr` / `s_axi_arvalid` / `s_axi_arready` | In / In / Out | 8 / 1 / 1 | Handshake kênh địa chỉ đọc (Address Read). |
| | `s_axi_rdata` / `s_axi_rresp` / `s_axi_rvalid` / `s_axi_rready` | Out / In | 32 / 2 / 1 / 1 | Kênh trả dữ liệu đọc CSR (Read Data/Response). |
| **AXI4-Stream RX**<br />*(Data Ingress)* | `s_axis_tdata` | Input | 64 bits | Payload tensor đầu vào ($8 \times \text{INT8}$ tokens $Q, K, V$). |
| | `s_axis_tkeep` | Input | 8 bits | Strobe đánh dấu các lane byte hợp lệ. |
| | `s_axis_tlast` | Input | 1 bit | Cờ đánh dấu word cuối cùng của Packet/Tile. |
| | `s_axis_tvalid` / `s_axis_tready` | In / Out | 1 bit | Bắt tay ngắt luồng (Backpressure Handshake). |
| **AXI4-Stream TX**<br />*(Data Egress)* | `m_axis_tdata` | Output | 64 bits | Payload tensor kết quả Attention ($8 \times \text{INT8}$ tokens). |
| | `m_axis_tkeep` / `m_axis_tlast` | Output | 8 / 1 bit | Strobe byte và Cờ báo kết thúc packet đầu ra. |
| | `m_axis_tvalid` / `m_axis_tready` | Out / In | 1 bit | Bắt tay đẩy dữ liệu ra AXI DMA S2MM. |
| **Nội bộ (Internal Routing)** | `cfg_start_pulse` | Internal | 1 bit | Xung kích hoạt FSM bắt đầu tính toán. |
| | `cfg_nbytes` / `cfg_tokens` / `cfg_hdim` | Internal | 32 bits | Kích thước Packet (bytes), $N=196$, $d_k=32$. |
| | `core_busy` / `core_done` / `core_error` | Internal | 1 bit | Tín hiệu trạng thái FSM gửi ngược về CSR 0x00. |

### 3. Sơ đồ ASMD Chart Điều Khiển Trung Tâm (Central ASMD Chart)

```text
                                +-------------------+
                                |     ST_RESET      |
                                |  busy      = 0    |
                                |  done_flag = 0    |
                                |  error_flag= 0    |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |      ST_IDLE      | <--------------------+
                                |  busy = 0         |                      |
                                +---------+---------+                      |
                                          |                                |
                                   (start_pulse == 1)                      |
                                    /           \                          |
                                  YES            NO                        |
                                  /                \                       |
                                 v                  +----------------------+
                       (reg_nbytes Valid?)
                         /                                 YES           NO
                       /                                     v                 v
            +-------------------+     +-------------------+
            |     ST_ACTIVE     |     |     ST_ERROR      |
            | busy         = 1  |     | error_flag = 1    |
            | remaining <= nbyte|     | busy       = 0    |
            +---------+---------+     +---------+---------+
                      |                         |
                      v                         v
            (Transfer_In Event?)          (Host Clear?)
             /            \                 /                   YES             NO             YES         NO
           /                \             /                      v                  v           v              v
    [remaining <=       (Keep Wait)  ST_IDLE        ST_ERROR
     remaining - 8]
          |
          v
    (remaining == 0 || TLAST?)
     /               YES             NO
   /                  v                  v
+-------------------+  ST_ACTIVE
|      ST_DONE      |
| done_flag  <= 1   |
| busy       <= 0   |
+---------+---------+
          |
          v
       ST_IDLE
```

```verilog
// Định nghĩa Trạng thái theo chuẩn Pong P. Chu
typedef enum logic [2:0] {
    ST_RESET  = 3'b000,
    ST_IDLE   = 3'b001,
    ST_ACTIVE = 3'b010,
    ST_DONE   = 3'b011,
    ST_ERROR  = 3'b100
} state_t;

state_t state_reg, state_next;

// FSM Next-State & Register Transfer Logic
always_comb begin
    state_next = state_reg;
    case (state_reg)
        ST_IDLE: begin
            if (start_pulse) begin
                if (reg_nbytes > 0 && reg_nbytes <= 32'h03FFFFFF)
                    state_next = ST_ACTIVE;
                else
                    state_next = ST_ERROR;
            end
        end
        ST_ACTIVE: begin
            if (input_fire && (remaining <= 8 || s_axis_tlast))
                state_next = ST_DONE;
            else if (framing_error)
                state_next = ST_ERROR;
        end
        ST_DONE:  state_next = ST_IDLE;
        ST_ERROR: if (clear_pulse) state_next = ST_IDLE;
        default:  state_next = ST_IDLE;
    endcase
end
```

### 4. Bảng Tính Toán Thời Gian & Băng Thông Thực Tế Cho MobileViT-XXS

Xét cấu hình mô hình tiêu chuẩn: Số Token $N = 196$, Chiều ẩn Head $d_k = 32$, Số Head $H = 2$, Kiểu dữ liệu $I_X \in \text{INT8}$ (1 byte/element):

| Hạng mục Tính toán | Công thức Toán học | Giá trị Số học | Số Chu kỳ Clock ($C_{transfer}$) | Thời gian Truyền ($T_{time}$) |
| ------ | ------ | ------ | ------ | ------ |
| **Kích thước Ma trận $Q, K, V$** | $3 \times (N \times d_k) \times 1 \text{ byte}$ | $3 \times 196 \times 32 = 18,816 \text{ Bytes}$ | 2,352 beats ($64\text{-bit}$) | $11.76 \, \mu\text{s}$ (ở $\eta = 1.0$) |
| **Kích thước Score Matrix ($QK^T$)** | $H \times (N \times N) \times 1 \text{ byte}$ | $2 \times 196 \times 196 = 76,832 \text{ Bytes}$ | *Lưu nội bộ On-Chip BRAM* | **$0.00 \, \mu\text{s}$ (Zero AXI Traffic)** |
| **Kích thước Output Tokens** | $(N \times (H \cdot d_k)) \times 1 \text{ byte}$ | $196 \times 64 = 12,544 \text{ Bytes}$ | 1,568 beats ($64\text{-bit}$) | $7.84 \, \mu\text{s}$ (ở $\eta = 1.0$) |
| **Tổng Dữ liệu Giao tiếp AXI** | $D_{total} = D_{in} + D_{out}$ | **$31,360 \text{ Bytes}$** | **3,920 beats** | **$19.60 \, \mu\text{s}$** |
| **Tiết kiệm Băng thông Bus nhờ On-chip Score Buffer** | $\frac{D_{score}}{D_{total} + D_{score}}$ | **Tiết kiệm $71.0\%$ băng thông AXI** | Triệt tiêu 9,604 beats nạp DRAM | Giảm $48.02 \, \mu\text{s}$ trễ bus DRAM |
