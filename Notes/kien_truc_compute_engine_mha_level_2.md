# Kiến Trúc Compute Engine - MHA Level 2: Stage 1 & Stage 4 GEMM

Dưới đây là **LEVEL 2 — COMPUTE ENGINES: STAGE 1 & STAGE 4 GEMM** theo cấu trúc Phân cấp Từ trên xuống (Top-Down Hierarchy).

Phân rã này tập trung vào kiến trúc mảng **Systolic MAC Array chế độ Output-Stationary (OS)**, vi mạch **PE Slice dùng kỹ thuật DSP Packing $18 \times 27$-bit (2x INT8)** và **Cây cộng Pipelined Adder Tree** quản lý Bit-Growth đường ống.

---

## 1. Top-Level Compute Engine Block Diagram (systolic_mac_array.sv)

Mảng tính toán GEMM vận hành theo cơ chế **Output-Stationary (OS)**: Giữ tích lũy tại từng thanh ghi PE, cho phép nạp liên tục $Q/S$ theo hàng và $K/V$ theo cột để đạt hiệu suất khai thác DSP $100\%$.

```text
========================================================================================================================
                                LEVEL 2: SYSTOLIC MAC ARRAY ARCHITECTURE (systolic_mac_array.sv)
========================================================================================================================

    Row Act Input: Q_vec[31:0] (4x INT8)                Col Weight Input: K_vec[31:0] (4x INT8)
    (Horizontal Broadcast / Shift)                      (Vertical Broadcast / Shift)
              │                                                   │
              ├───────────────────────────────────────────────────┼──────────────────────────────────┐
              │                                                   │                                  │
              v                                                   v                                  v
     +----------------------------------+                +----------------------------------+       +----------------------------------+
     | PE_SLICE                  |                | PE_SLICE                  |  ...  | PE_SLICE[M-1]                |
     | - DSP48E2 2x INT8 Packing        |──Act_Shift────►| - DSP48E2 2x INT8 Packing        |       | - DSP48E2 2x INT8 Packing        |
     | - Acc_Reg [31:0] (INT32)         |                | - Acc_Reg [31:0] (INT32)         |       | - Acc_Reg [31:0] (INT32)         |
     +----------------+-----------------+                +----------------+-----------------+       +----------------+-----------------+
                      │                                                   │                                  │
                      │ Wt_Shift                                          │ Wt_Shift                         │ Wt_Shift
                      v                                                   v                                  v
     +----------------------------------+                +----------------------------------+       +----------------------------------+
     | PE_SLICE                  |                | PE_SLICE                  |  ...  | PE_SLICE[M-1]                |
     | - DSP48E2 2x INT8 Packing        |──Act_Shift────►| - DSP48E2 2x INT8 Packing        |       | - DSP48E2 2x INT8 Packing        |
     | - Acc_Reg [31:0] (INT32)         |                | - Acc_Reg [31:0] (INT32)         |       | - Acc_Reg [31:0] (INT32)         |
     +----------------+-----------------+                +----------------+-----------------+       +----------------+-----------------+
                      │                                                   │                                  │
                      v                                                   v                                  v
     +-----------------------------------------------------------------------------------------------------------------+
     | PIPELINED ADDER TREE REDUCTION NETWORK (pipelined_adder_tree.sv)                                                |
     | Accumulates row/col partial products -> Produces 32-bit Dot Product Output per clock cycle                      |
     +--------------------------------------------------+--------------------------------------------------------------+
                                                        │
                                                        v
                                            Gemm_Out_Score [31:0] (INT32)
```

### 2. Bảng Mô Tả Tín Hiệu & Routing Component (LEVEL 2 Signal Table)

| Tên Tín hiệu (Signal) | Hướng | Độ rộng bit | Mô tả Chức năng & Định tuyến Routing |
| ------ | ------ | ------ | ------ |
| clk / rst_n | Input | 1 / 1 bit | Clock hệ thống 200 MHz và Reset bất đồng bộ mức thấp. |
| gemm_start | Input | 1 bit | Kích hoạt chu kỳ tính toán ma trận mới. |
| mode_sel | Input | 1 bit | Chọn chế độ tính toán: 0 cho $QK^T$ (Stage 1), 1 cho $Score \times V$ (Stage 4). |
| act_in | Input | 32 bits | Vector $4 \times \text{INT8}$ dữ liệu Kích hoạt ($Q$ hoặc $S$). |
| wt_in | Input | 32 bits | Vector $4 \times \text{INT8}$ dữ liệu Trọng số ($K$ hoặc $V$). |
| pe_acc_clear | Internal | 1 bit | Xóa thanh ghi tích lũy Acc_Reg về 0 trước lượt Tile mới. |
| dsp_p_raw | Internal | 45 bits | Kết quả thô đầu ra từ bộ nhân DSP48E2 $18 \times 27$-bit. |
| prod_ab / prod_ac | Internal | 16 bits | Hai tích số $\text{INT8} \times \text{INT8}$ bóc tách từ dsp_p_raw. |
| adder_tree_out | Output | 32 bits | Tổng tích vô hướng $\text{INT32}$ thu được sau $1 + \lceil \log_2 N \rceil$ chu kỳ trễ. |
| gemm_valid_out | Output | 1 bit | Cờ báo dữ liệu tích vô hướng đầu ra hợp lệ. |

### 3. Kiến Trúc Chi Tiết Vi Mạch PE Slice (pe_slice_dsp_packing.sv)

```text
========================================================================================================================
                            DSP48E2 PE SLICE WITH 2x INT8 PACKING (pe_slice_dsp_packing.sv)
========================================================================================================================

     Act_A [7:0] (INT8)          Wt_B [7:0] (INT8)               Wt_C [7:0] (INT8)
             │                           │                               │
             v                           v                               v
    +-----------------+         +-----------------+             +-----------------+
    | Sign-Extend     |         | Sign-Extend     |             | Sign-Extend     |
    | (to 18-bit)     |         | (to 27-bit)     |             | (to 27-bit)     |
    +--------┬--------+         +--------┬--------+             +--------┬--------+
             │                           │                               │
             │                           v                               │
             │                  +-----------------+                      │
             │                  | Shift <<< 18    |                      │
             │                  +--------┬--------+                      │
             │                           │                               │
             │                           +---------------+---------------+
             │                                           │
             │                                           v
             │                                +---------------------+
             │                                | Add: (B<<18) + C    | ==> Packed Operand [26:0]
             │                                +----------┬----------+
             │                                           │
             +--------------------+----------------------+
                                  │
                                  v
                       +----------------------+
                       | DSP48E2 Multiplier   |
                       |  (18-bit x 27-bit)   |
                       +----------┬-----------+
                                  │
                                  v
                       Result P [44:0] = (A x B) * 2^18 + (A x C)
                                  │
           +----------------------+----------------------+
           │                                             │
           v                                             v
     Bits [33:18] (Raw A x B)                      Bits [15:0] (Raw A x C)
           │                                             │
           v                                             v
    +----------------------+                      +----------------------+
    | Sign Correction      |                      | Direct Extract       |
    | (Check P[17:16] Guard|                      | (Product A x C)      |
    +----------┬-----------+                      +----------┬-----------+
               │                                             │
               v                                             v
        Prod_AB [15:0]                                Prod_AC [15:0]
               │                                             │
               +----------------------+----------------------+
                                      │
                                      v
                        +----------------------------+
                        | Acc_Reg <= Acc_Reg + AB+AC | (INT32 Accumulator)
                        +----------------------------+
```

```verilog
module pe_slice_dsp_packing (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clr_acc,
    input  logic signed [7:0]  act_a,   // Operand A (Shared Activation)
    input  logic signed [7:0]  wt_b,    // Operand B (Weight 0)
    input  logic signed [7:0]  wt_c,    // Operand C (Weight 1)
    output logic signed [31:0] acc_out  // Accumulated INT32 Output
);

    // Pipeline Registers
    logic signed [17:0] op_a_reg;
    logic signed [26:0] op_bc_packed;
    logic signed [44:0] dsp_p_reg;
    logic signed [15:0] prod_ab, prod_ac;
    logic signed [31:0] accumulator;

    // Stage 1: Operand Packing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_a_reg     <= '0;
            op_bc_packed <= '0;
        end else begin
            op_a_reg     <= 18'(act_a);
            op_bc_packed <= (27'(wt_b) <<< 18) + 27'(wt_c);
        end
    end

    // Stage 2: DSP Multiply
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dsp_p_reg <= '0;
        end else begin
            dsp_p_reg <= op_a_reg * op_bc_packed;
        end
    end

    // Stage 3: Unpacking & Accumulation
    assign prod_ac = dsp_p_reg[15:0];
    assign prod_ab = dsp_p_reg[33:18] + dsp_p_reg[15]; // Sign guard correction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else if (clr_acc) begin
            accumulator <= '0;
        end else begin
            accumulator <= accumulator + 32'(prod_ab) + 32'(prod_ac);
        end
    end

    assign acc_out = accumulator;

endmodule
```

### 4. Cây Cộng Đường Ống Quản Lý Bit-Growth (pipelined_adder_tree.sv)

Mạch giảm nhị phân (Binary Reduction) $N=16$ phần tử đảm bảo nhịp xử lý $II=1$ và kiểm soát tăng bit an toàn ($W_k = 16 + k$):

```text
========================================================================================================================
                            PIPELINED ADDER TREE REDUCTION LOGIC (N = 16 Inputs)
========================================================================================================================

 Input Products : 16x INT16 Products from PE Array [P0 .. P15]
                     │
                     v (Stage 0: 8 Adders / Width: 17-bit)
               +-----------+           +-----------+
               | P0 + P1   |   . . .   | P14 + P15 |
               +-----+-----+           +-----+-----+
                     │                       │
                     v (Reg Stage 1)         v (Reg Stage 1)
               +-----------+           +-----------+
               | Adder S1  |   . . .   | Adder S1  |  (4 Adders / Width: 18-bit)
               +-----+-----+           +-----+-----+
                     │                       │
                     v (Reg Stage 2)         v (Reg Stage 2)
               +-----------+           +-----------+
               | Adder S2  |   . . .   | Adder S2  |  (2 Adders / Width: 19-bit)
               +-----+-----+           +-----+-----+
                     │                       │
                     +-----------+-----------+
                                 │
                                 v (Reg Stage 3)
                           +-----------+
                           | Root Adder|              (1 Adder / Width: 20-bit -> Sign Ext to 32-bit)
                           +-----+-----+
                                 │
                                 v
                     Dot_Product_Out [31:0] (INT32)
```
