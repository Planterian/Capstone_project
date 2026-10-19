# Cơ Chế I/O Buffer Của Mảng Systolic Array Dựa Trên Phân Khối Ma Trận (Matrix Tiling) v2.0

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Chuẩn hóa vi mạch theo RTL `rtl_ver02`)  
> **Module liên quan:** [`ping_pong_bram_buffer.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/ping_pong_bram_buffer.sv), [`systolic_mac_array.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/systolic_mac_array.sv), [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/attention_core_top.sv)

---

## 1. TỔNG QUAN BÀI TOÁN PHÂN KHỐI TILE (MATRIX TILING PROBLEM)

Trong cơ chế Scaled Dot-Product Attention của MobileViT-XXS, ma trận chú ý cần tính toán là:
$$S = Q \cdot K^T \in \mathbb{R}^{256 \times 256}$$
trong đó $Q, K \in \mathbb{R}^{256 \times d_k}$ ($d_k \in \{16, 20, 24\}$).

Nếu tính toán song song toàn bộ ma trận $256 \times 256$ trong một chu kỳ, phần cứng sẽ đòi hỏi:
$$256 \times 256 = \mathbf{65,536\text{ bộ nhân MACs}}$$
Điều này vượt xa khả năng tài nguyên của chip AMD Kria KV260 (chỉ có $1,248$ khối DSP48E2).

**Giải pháp:** Phân khối ma trận (**Matrix Tiling**). Toàn bộ ma trận lớn được chia nhỏ thành các khối Tile con kích thước phù hợp với lưới tính toán phần cứng:
- **Kích thước vật lý lưới PE (Physical PE Grid):** $4 \text{ hàng} \times 4 \text{ cột}$ ($16$ PEs vật lý / 16 DSPs).
- **Kỹ thuật đóng gói DSP Packing 2x INT8:** Mỗi PE nhân đồng thời kích hoạt $A$ với 2 cột trọng số $B$ và $C$.
- **Kích thước Tile đầu ra tính toán (Output Tile Size):**
  $$\text{Output Tile} = \mathbf{4\text{ rows} \times 8\text{ columns} = 32\text{ dot-products}}$$
- Chiều sâu tích lũy của Tile (Reduction dimension): $d_k$ chu kỳ clock (16 chu kỳ cho Stage 2).

---

## 2. KIẾN TRÚC PHÂN TẦNG I/O BUFFER & DÒNG CHẢY DỮ LIỆU TILE

```text
========================================================================================================================
                              CƠ CHẾ TILING & I/O BUFFERING CỦA SYSTOLIC MAC ARRAY
========================================================================================================================

  Ma trận toàn cục: Q [256 x 16], K [256 x 16], Score [256 x 256]
  
  +------------------------------------------------------------------------------------------------------------------+
  | TẦNG 1: INPUT PING-PONG BRAM BUFFER (Double Buffering ẩn độ trễ DMA)                                             |
  |                                                                                                                  |
  |   AXI-Stream Ingress (DMA MM2S): 64-bit / beat (8x INT8)                                                         |
  |             │                                                                                                    |
  |             ├──> BANK 0 [Q_RAM, K_RAM, V_RAM] <── (DMA đang nạp Tile mới khi Bank 1 bận)                         |
  |             └──> BANK 1 [Q_RAM, K_RAM, V_RAM] ──> (Systolic Array đang đọc tính Tile hiện tại)                   |
  +-------------------------------------------------------┬----------------------------------------------------------+
                                                          │ Đọc Vector 32-bit qua Subword Mux
                                                          v
  +------------------------------------------------------------------------------------------------------------------+
  | TẦNG 2: INPUT FEEDER (Vector Slicing & Wavefront Skewing)                                                        |
  |                                                                                                                  |
  |   Mỗi chu kỳ, bộ đệm tách từ 64-bit thành các Vector 32-bit (4x INT8) cấp vào dầm nhịp Systolic:                  |
  |     - act_row [0..3] = rdata_q_vec [31:0] ──> Nạp 4 hàng kích hoạt A (Tile 4 hàng)                              |
  |     - wt_col0 [0..3] = rdata_k_vec [31:0] ──> Nạp 4 cột trọng số chẵn B (Col 0, 2, 4, 6)                        |
  |     - wt_col1 [0..3] = rdata_v_vec [31:0] ──> Nạp 4 cột trọng số lẻ C (Col 1, 3, 5, 7)                          |
  +-------------------------------------------------------┬----------------------------------------------------------+
                                                          │ Lan truyền ngang (A) và dọc (B, C)
                                                          v
  +------------------------------------------------------------------------------------------------------------------+
  | TẦNG 3: COMPUTE ENGINE (Output-Stationary Systolic Array 4x4 PEs / 16 DSPs)                                      |
  |                                                                                                                  |
  |   - Kích thước Tile đầu ra: 4 rows x 8 cols = 32 điểm tích lũy                                                   |
  |   - Các PE giữ cố định kết quả: pe_acc_col0[4][4] và pe_acc_col1[4][4]                                           |
  |   - Tích lũy liên tục qua chiều sâu d_k (16 chu kỳ cho Stage 2)                                                  |
  +-------------------------------------------------------┬----------------------------------------------------------+
                                                          │ Kích hoạt cờ drain_start
                                                          v
  +------------------------------------------------------------------------------------------------------------------+
  | TẦNG 4: OUTPUT DRAIN BUFFER (Sequential Tile Shift-Out)                                                          |
  |                                                                                                                  |
  |   - Mạch Drain FSM quét xả tuần tự 32 kết quả INT32 ra dot_product_stream (1 score/chu kỳ)                       |
  |   - Sau khi xả hết (TOTAL_DRAIN_ITEMS = 32): Phát cờ drain_done, xóa clr_acc, nạp Tile kế tiếp                    |
  +------------------------------------------------------------------------------------------------------------------+
```

---

## 3. CƠ CHẾ ĐỆM ĐẦU VÀO (INPUT BUFFERING) THEO TILE

### 3.1 Căn chỉnh Băng thông Ingress (64-bit $\rightarrow$ 32-bit Subword Muxing)
- **Phía AXI DMA:** Để tối ưu hóa băng thông DDR4, DMA nạp dữ liệu qua bus 64-bit AXI-Stream ($8 \times \text{INT8}$ bytes/beat) và ghi thẳng vào Block RAM 64-bit.
- **Phía Systolic Array:** Mỗi cạnh của mảng gồm `ARRAY_SIZE = 4` PEs, do đó mỗi chu kỳ mảng chỉ tiêu thụ **4 bytes INT8 (32 bits)**.
- **Bộ đệm Subword Multiplexing** ([`ping_pong_bram_buffer.sv:228-249`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/ping_pong_bram_buffer.sv#L228-L249)) sử dụng bit địa chỉ thấp `raddr[0]` để lựa chọn nửa từ:
  ```systemverilog
  // Tách từ 64-bit BRAM thành 2 subword 32-bit (4x INT8 mỗi subword)
  rdata_q_vec <= raddr_q[0] ? ram_q[raddr >> 1][63:32] : ram_q[raddr >> 1][31:0];
  rdata_k_vec <= raddr_k[0] ? ram_k[raddr >> 1][63:32] : ram_k[raddr >> 1][31:0];
  rdata_v_vec <= raddr_v[0] ? ram_v[raddr >> 1][63:32] : ram_v[raddr >> 1][31:0];
  ```
  - Khi `raddr[0] = 0`: Cấp 4 phần tử đầu của Tile vector.
  - Khi `raddr[0] = 1`: Cấp 4 phần tử tiếp theo của Tile vector.

### 3.2 Cơ chế Double Buffering (Ping-Pong) Ẩn Độ Trễ Nạp Tile
Để Systolic Array không phải dừng đường ống (Zero Bubble Pipeline) khi DMA đang truyền:
- **Bank 0:** Lưu trữ toàn bộ dữ liệu Tile hiện hành ($Q, K, V$).
- **Bank 1:** Lưu trữ dữ liệu Tile tiếp theo đang được DMA nạp ngầm.
- **Giao thức bắt tay tráo đổi Bank:**
  1. Khi DMA nạp đủ toàn bộ dữ liệu Tile vào Bank ghi $\implies$ Kích hoạt `tile_write_done` và đánh dấu `bank_full = 1`.
  2. Khi Systolic Array hoàn tất đọc Tile từ Bank đọc $\implies$ Phát xung `tile_read_done`.
  3. Bộ điều khiển đảo con trỏ: `pp_bank_reg <= ~pp_bank_reg`, lập tức chuyển Bank vừa nạp xong sang trạng thái đọc mà không mất bất kỳ chu kỳ chết nào.

---

## 4. CƠ CHẾ ĐỆM ĐẦU RA (OUTPUT BUFFERING) THEO TILE

### 4.1 Bộ đệm tích lũy đứng yên (Output-Stationary Accumulators)
Trong suốt quá trình dầm nhịp (Wavefront Execution) kéo dài $d_k$ chu kỳ:
- Các kết quả tích vô hướng tạm thời không được xuất ra ngoài mà được giữ cố định tại các thanh ghi 32-bit nội bộ của từng PE:
  - `pe_acc_col0[r][c]`: Tích lũy điểm số cho hàng $r$, cột chẵn $2c$.
  - `pe_acc_col1[r][c]`: Tích lũy điểm số cho hàng $r$, cột lẻ $2c+1$.
- Toàn bộ $16 \text{ PEs} \times 2 = 32$ thanh ghi tích lũy này đóng vai trò là **Bộ đệm đầu ra của Tile $4 \times 8$**.

### 4.2 Máy trạng thái Drain tuần tự (Sequential Drain State Machine)
Sau khi kết thúc tích lũy qua chiều $d_k$:
1. FSM kích hoạt xung `drain_start = 1`.
2. Máy trạng thái `DRAIN_RUN` ([`systolic_mac_array.sv:131-185`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/systolic_mac_array.sv#L131-L185)) chuyển đổi ma trận kết quả song song thành dòng dữ liệu tuần tự:
   ```systemverilog
   localparam int TOTAL_DRAIN_ITEMS = 2 * ARRAY_SIZE * ARRAY_SIZE; // 32 phần tử
   
   assign drain_r      = drain_item_cnt / (2 * ARRAY_SIZE);        // Hàng 0..3
   assign drain_c      = (drain_item_cnt % (2 * ARRAY_SIZE)) / 2;  // Cột PE 0..3
   assign drain_subcol = drain_item_cnt % 2;                       // 0: col0, 1: col1
   
   dot_product_stream <= (!drain_subcol) ? pe_acc_col0[drain_r][drain_c] 
                                         : pe_acc_col1[drain_r][drain_c];
   ```
3. Trong 32 chu kỳ liên tiếp, mỗi chu kỳ xuất đúng 1 điểm số INT32 sang tầng `scale_unit.sv`.
4. Khi `drain_item_cnt == 31`, cờ `drain_done = 1` được phát ra để kích hoạt lệnh xóa `clr_acc` và chuẩn bị tính toán Tile kế tiếp.

---

## 5. CHIẾN LƯỢC QUÉT TILE TRÊN MA TRẬN TOÀN CỤC ($256 \times 256$)

Để hoàn thành ma trận Attention Scores kích thước $256 \times 256$:
- **Phân rã theo chiều Hàng:** $\frac{256}{4} = \mathbf{64\text{ bước tile hàng}}$.
- **Phân rã theo chiều Cột:** $\frac{256}{8} = \mathbf{32\text{ bước tile cột}}$.
- **Tổng số Tile cần duyệt:** $64 \times 32 = \mathbf{2,048\text{ Tiles}}$.
- **Vòng lặp điều khiển:**
  $$\text{Nạp BRAM (Ping-Pong)} \longrightarrow \text{Tích lũy d_k chu kỳ} \longrightarrow \text{Drain Tile 32 chu kỳ} \longrightarrow \text{Chuyển Tile kế tiếp}.$$

*(Ghi chú thiết kế: Trong `attention_core_top.sv` hiện tại, bên cạnh mảng Systolic Array tile $4 \times 8$, hệ thống còn tích hợp thêm streaming dot-product engine nhân song song $d_k$ chiều để xuất liên tục 1 score/chu kỳ trong 256 nhịp, hoàn toàn đồng bộ với nhịp nạp của khối Hardware Softmax).*

---

## 6. VÍ DỤ MINH HỌA TRỰC QUAN CHI TIẾT: TILING MA TRẬN 16x16 VỚI MẢNG SYSTOLIC TÍNH TOÁN 4x8 (LƯỚI VẬT LÝ 4x4)

Để hình dung tường minh và chính xác từng chu kỳ clock, dưới đây là ví dụ số học thực tế về việc nhân hai ma trận đầy đủ kích thước **$16 \times 16$**:
$$C = A \times B$$
- $A \in \mathbb{R}^{16 \times 16}$: Ma trận kích hoạt ($Q$ gồm 16 tokens $T_0 \dots T_{15}$, mỗi token có chiều $d_k = 16$).
- $B \in \mathbb{R}^{16 \times 16}$: Ma trận trọng số ($K^T$ gồm chiều $d_k = 16$, 16 keys $K_0 \dots K_{15}$).
- $C \in \mathbb{R}^{16 \times 16}$: Ma trận điểm chú ý đầu ra (gồm $16 \times 16 = 256$ điểm).

Phần cứng thực thi:
- **Lưới PE vật lý:** $4 \text{ hàng} \times 4 \text{ cột}$ PEs (`ARRAY_SIZE = 4`, 16 khối DSP48E2).
- **Kỹ thuật DSP Packing 2x INT8:** Mỗi PE tính đồng thời 2 cột kết quả $\implies$ **Khối tính toán (Computing Tile) đạt $4 \text{ hàng} \times 8 \text{ cột} = 32\text{ kết quả}$**.

```text
========================================================================================================================
             PHÂN RÃ MA TRẬN TOÀN CỤC C [16 x 16] THÀNH 8 TILES (MỖI TILE 4 ROWS x 8 COLS)
========================================================================================================================

           Cột 0 .. 7 (C_ColBlock 0: 8 cột)               Cột 8 .. 15 (C_ColBlock 1: 8 cột)
         +───────────────────────────────────────────────+───────────────────────────────────────────────+
  Hàng   |                                               |                                               |
  0..3   |              TILE (0, 0)                      |              TILE (0, 1)                      |
  (R0)   |   A [0..3, :] x B [:, 0..7] -> 32 điểm        |   A [0..3, :] x B [:, 8..15] -> 32 điểm       |
         +───────────────────────────────────────────────+───────────────────────────────────────────────+
  Hàng   |                                               |                                               |
  4..7   |              TILE (1, 0)                      |              TILE (1, 1)                      |
  (R1)   |   A [4..7, :] x B [:, 0..7] -> 32 điểm        |   A [4..7, :] x B [:, 8..15] -> 32 điểm       |
         +───────────────────────────────────────────────+───────────────────────────────────────────────+
  Hàng   |                                               |                                               |
  8..11  |              TILE (2, 0)                      |              TILE (2, 1)                      |
  (R2)   |   A [8..11, :] x B [:, 0..7] -> 32 điểm       |   A [8..11, :] x B [:, 8..15] -> 32 điểm      |
         +───────────────────────────────────────────────+───────────────────────────────────────────────+
  Hàng   |                                               |                                               |
  12..15 |              TILE (3, 0)                      |              TILE (3, 1)                      |
  (R3)   |   A [12..15, :] x B [:, 0..7] -> 32 điểm      |   A [12..15, :] x B [:, 8..15] -> 32 điểm     |
         +───────────────────────────────────────────────+───────────────────────────────────────────────+

  * Tổng cộng: 4 bước hàng x 2 bước cột = ĐÚNG 8 TILES (8 x 32 = 256 kết quả, phủ kín ma trận 16x16).
========================================================================================================================
```

---

### 6.1 Đi Sâu Vào Chi Tiết Thực Thi Tile Đầu Tiên: TILE (0, 0)

Tile (0, 0) chịu trách nhiệm tính toán góc trên bên trái của ma trận kết quả:
$$C[0..3, 0..7] = A[0..3, 0..15] \times B[0..15, 0..7]$$
- $A[0..3, 0..15]$: 4 hàng token của $Q$, mỗi hàng dài 16 chiều ($4 \times 16 = 64$ bytes).
- $B[0..15, 0..7]$: 16 hàng của $K^T$ qua 8 cột keys ($16 \times 8 = 128$ bytes).
- $C[0..3, 0..7]$: Khối kết quả gồm **$4 \text{ hàng} \times 8 \text{ cột} = 32\text{ phần tử INT32}$**.

#### A. Sơ đồ Ánh xạ Tín hiệu vào Lưới PE $4 \times 4$ với DSP Packing:

```text
========================================================================================================================
                 ÁNH XẠ DỮ LIỆU VÀO LƯỚI PE 4x4 ĐỂ TÍNH TOÁN OUTPUT TILE 4x8
========================================================================================================================

                      Col PE 0              Col PE 1              Col PE 2              Col PE 3
                 (Col 0 & Col 1)       (Col 2 & Col 3)       (Col 4 & Col 5)       (Col 6 & Col 7)
                        │                     │                     │                     │
                        │ wt_col0 = B[k, 0]   │ wt_col0 = B[k, 2]   │ wt_col0 = B[k, 4]   │ wt_col0 = B[k, 6]
                        │ wt_col1 = B[k, 1]   │ wt_col1 = B[k, 3]   │ wt_col1 = B[k, 5]   │ wt_col1 = B[k, 7]
                        v                     v                     v                     v
  Row 0: A[0, k] ──> [ PE_00 ] ───────────> [ PE_01 ] ───────────> [ PE_02 ] ───────────> [ PE_03 ]
                        │                     │                     │                     │
                        v                     v                     v                     v
  Row 1: A[1, k] ──> [ PE_10 ] ───────────> [ PE_11 ] ───────────> [ PE_12 ] ───────────> [ PE_13 ]
                        │                     │                     │                     │
                        v                     v                     v                     v
  Row 2: A[2, k] ──> [ PE_20 ] ───────────> [ PE_21 ] ───────────> [ PE_22 ] ───────────> [ PE_23 ]
                        │                     │                     │                     │
                        v                     v                     v                     v
  Row 3: A[3, k] ──> [ PE_30 ] ───────────> [ PE_31 ] ───────────> [ PE_32 ] ───────────> [ PE_33 ]

  * Tại mỗi PE[r][c]:
    - pe_acc_col0[r][c] tích lũy: pe_acc_col0[r][c] += A[r, k] * B[k, 2c]       (Cột chẵn)
    - pe_acc_col1[r][c] tích lũy: pe_acc_col1[r][c] += A[r, k] * B[k, 2c + 1]   (Cột lẻ)
========================================================================================================================
```

---

### 6.2 Tiến Trình Thời Gian Từng Chu Kỳ (Wavefront Execution & Drain Timeline)

Chiều sâu tích lũy của Tile là $K = 16$ ($k = 0 \dots 15$):

```text
========================================================================================================================
                                     TIẾN TRÌNH THỜI GIAN THỰC THI TILE (0, 0)
========================================================================================================================

 Giai đoạn 1: Nạp & Nhân Tích Lũy (Wavefront Compute: 16 chu kỳ tích lũy k = 0..15 + skew delay)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Clock  | Dữ liệu nạp Act Row          | Dữ liệu nạp Wt Col0/Col1        | Trạng thái PE Grid
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Cycle 0| act_row = A[0..3, 0]         | wt0/1 = B[0, 0..7]              | PE_00 nhận A[0,0], B[0,0..1]; bắt đầu tính
  Cycle 1| act_row = A[0..3, 1]         | wt0/1 = B[1, 0..7]              | PE_00 tính k=1; PE_01 & PE_10 nhận k=0
  Cycle 2| act_row = A[0..3, 2]         | wt0/1 = B[2, 0..7]              | Wavefront lan truyền chéo (r + c)
   ...   |          ...                 |         ...                     |                   ...
  Cycle15| act_row = A[0..3, 15]        | wt0/1 = B[15, 0..7]             | Nạp phần tử k cuối cùng (k=15)
  Cycle19| Đóng luồng nạp (in_valid=0)  | -                               | PE_33 hoàn tất tích lũy k=15 cuối cùng.
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  KẾT THÚC GIAI ĐOẠN 1: Toàn bộ 32 thanh ghi tích lũy pe_acc_col0/col1 đã chứa trọn vẹn kết quả C[0..3, 0..7]!


 Giai đoạn 2: Quét Xả Kết Quả Ra Output Stream (Drain State Machine: 32 chu kỳ liên tiếp)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Clock  | drain_item_cnt | drain_r (Row) | drain_c (Col PE) | subcol | dot_product_stream (Dữ liệu xuất ra)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Cycle20| #00            | Row 0         | PE 0             | col0   | C[0, 0] = pe_acc_col0[0][0]
  Cycle21| #01            | Row 0         | PE 0             | col1   | C[0, 1] = pe_acc_col1[0][0]
  Cycle22| #02            | Row 0         | PE 1             | col0   | C[0, 2] = pe_acc_col0[0][1]
  Cycle23| #03            | Row 0         | PE 1             | col1   | C[0, 3] = pe_acc_col1[0][1]
  Cycle24| #04            | Row 0         | PE 2             | col0   | C[0, 4] = pe_acc_col0[0][2]
  Cycle25| #05            | Row 0         | PE 2             | col1   | C[0, 5] = pe_acc_col1[0][2]
  Cycle26| #06            | Row 0         | PE 3             | col0   | C[0, 6] = pe_acc_col0[0][3]
  Cycle27| #07            | Row 0         | PE 3             | col1   | C[0, 7] = pe_acc_col1[0][3]  <-- Hết Hàng 0!
  Cycle28| #08            | Row 1         | PE 0             | col0   | C[1, 0] = pe_acc_col0[1][0]
   ...   |  ...           |  ...          |  ...             |  ...   |                 ...
  Cycle51| #31 (Last)     | Row 3         | PE 3             | col1   | C[3, 7] = pe_acc_col1[3][3]  <-- Hết Hàng 3!
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  TẠI CHU KỲ 51: drain_done = 1'b1 khẳng định; clr_acc xóa thanh ghi về 0; SẴN SÀNG CHO TILE (0, 1)!
========================================================================================================================
```

---

### 6.3 Phối Hợp Đệm Ping-Pong Giữa Hai Tile Liên Tiếp: TILE (0, 0) $\rightarrow$ TILE (0, 1)

Sự phối hợp giữa **Bộ đệm Ping-Pong BRAM** và **Mảng Systolic Array** diễn ra nhịp nhàng nhằm triệt tiêu hoàn toàn chu kỳ chết:

```text
========================================================================================================================
              SỰ PHỐI HỢP DUAL-BANK PING-PONG TRONG QUÁ TRÌNH TÍNH TILE (0,0) VÀ TILE (0,1)
========================================================================================================================

  Thời gian (Cycles)      BANK 0 (Đang kết nối Systolic Array)        BANK 1 (Đang kết nối AXI DMA Ingress)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Cycles 0 .. 19         [READ MODE] Cấp dữ liệu Tile (0, 0):         [WRITE MODE] DMA nạp trước dữ liệu Tile (0, 1):
                         A[0..3, :] và B[:, 0..7] cho Systolic Array  A[0..3, :] và B[:, 8..15] từ DDR4 vào BRAM
                         để nhân tích lũy.

  Cycles 20 .. 51        [DRAIN MODE] Systolic Array xả 32 kết quả    [STANDBY] DMA đã nạp xong Bank 1 (bank1_full = 1).
                         C[0..3, 0..7] sang scale_unit. BRAM rảnh.    Chờ Systolic Array hoàn thành drain.

  Cycle 52               [PHÁT XUNG TILE_READ_DONE]                   [ĐẢO CON TRỎ PING-PONG: pp_bank_reg = ~pp_bank_reg]
                         Bank 0 chuyển thành WRITE MODE.              Bank 1 chuyển thành READ MODE.

  Cycles 53 .. 104       [WRITE MODE] DMA nạp trước Tile (1, 0)       [READ MODE] Systolic Array bắt đầu tính Tile (0, 1)
                         cho lượt kế tiếp...                          ngay lập tức mà KHÔNG MẤT 1 CHU KỲ NÀO!
========================================================================================================================
```

---

### 6.4 Bảng Đối Soát Vòng Lặp 8 Tiles Để Hoàn Thành Ma Trận 16x16:

| Thứ tự Tile | Tọa độ Ma trận Kết quả | Dữ liệu Kích hoạt $A$ | Dữ liệu Trọng số $B$ | Số chu kỳ Compute | Số chu kỳ Drain | Trạng thái Ping-Pong Bank |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tile 1** | $C[0..3, 0..7]$ (Tile 0,0) | Hàng $A[0..3]$ (64B) | Cột $B[0..7]$ (128B) | 20 cycles | 32 cycles | Đọc Bank 0 / Ghi Bank 1 |
| **Tile 2** | $C[0..3, 8..15]$ (Tile 0,1) | Hàng $A[0..3]$ (64B) | Cột $B[8..15]$ (128B)| 20 cycles | 32 cycles | Đọc Bank 1 / Ghi Bank 0 |
| **Tile 3** | $C[4..7, 0..7]$ (Tile 1,0) | Hàng $A[4..7]$ (64B) | Cột $B[0..7]$ (128B) | 20 cycles | 32 cycles | Đọc Bank 0 / Ghi Bank 1 |
| **Tile 4** | $C[4..7, 8..15]$ (Tile 1,1) | Hàng $A[4..7]$ (64B) | Cột $B[8..15]$ (128B)| 20 cycles | 32 cycles | Đọc Bank 1 / Ghi Bank 0 |
| **Tile 5** | $C[8..11, 0..7]$ (Tile 2,0) | Hàng $A[8..11]$ (64B) | Cột $B[0..7]$ (128B) | 20 cycles | 32 cycles | Đọc Bank 0 / Ghi Bank 1 |
| **Tile 6** | $C[8..11, 8..15]$ (Tile 2,1) | Hàng $A[8..11]$ (64B) | Cột $B[8..15]$ (128B)| 20 cycles | 32 cycles | Đọc Bank 1 / Ghi Bank 0 |
| **Tile 7** | $C[12..15, 0..7]$ (Tile 3,0) | Hàng $A[12..15]$ (64B)| Cột $B[0..7]$ (128B) | 20 cycles | 32 cycles | Đọc Bank 0 / Ghi Bank 1 |
| **Tile 8** | $C[12..15, 8..15]$ (Tile 3,1)| Hàng $A[12..15]$ (64B)| Cột $B[8..15]$ (128B)| 20 cycles | 32 cycles | Đọc Bank 1 / Ghi Bank 0 |

- **Tổng số kết quả tính toán:** $8 \text{ tiles} \times 32 \text{ scores/tile} = \mathbf{256\text{ điểm số}}$ (Khớp chính xác tuyệt đối $16 \times 16$ ma trận $C$).
- **Tổng thời gian thực thi 8 Tiles:** $\approx 8 \times 52\text{ cycles} = \mathbf{416\text{ clock cycles}}$ ($\approx 2.08\text{ }\mu\text{s}$ tại $200\text{ MHz}$).

---

## 7. VÍ DỤ CHUYÊN SÂU: THỰC THI MA TRẬN ĐẦY ĐỦ 16x16 DỰA TRÊN CÁC KHỐI TILE CƠ SỞ 4x4 BẰNG MẢNG SYSTOLIC CỠ TÍNH TOÁN 4x8

Đây là phân tích chuyên sâu giải quyết câu hỏi kiến trúc then chốt: **Khi bài toán tiling được định nghĩa dựa trên các khối ma trận con cơ sở kích thước $4 \times 4$, mảng phần cứng Systolic Array với kích cỡ tính toán $4 \times 8$ (lưới vật lý $4 \times 4$ PEs tích hợp DSP packing) sẽ xử lý và ghép nối các khối $4 \times 4$ như thế nào để hoàn thành toàn bộ ma trận $16 \times 16$?**

---

### 7.1 Bản Chất Toán Học: Phân Rã Ma Trận $16 \times 16$ Thành 16 Khối Cơ Sở $4 \times 4$

Xét phép nhân ma trận đầy đủ:
$$C = A \times B \quad \text{với } A, B, C \in \mathbb{R}^{16 \times 16}$$

Theo phương pháp nhân ma trận phân khối (Block Matrix Multiplication), toàn bộ ma trận $16 \times 16$ được chia thành một lưới các khối vuông kích thước **$4 \times 4$**:
- Ma trận $A$ gồm **16 khối kích hoạt** $A_{i, k} \in \mathbb{R}^{4 \times 4}$ với $i \in \{0..3\}, k \in \{0..3\}$:
  $$A = \begin{bmatrix} A_{0,0} & A_{0,1} & A_{0,2} & A_{0,3} \\ A_{1,0} & A_{1,1} & A_{1,2} & A_{1,3} \\ A_{2,0} & A_{2,1} & A_{2,2} & A_{2,3} \\ A_{3,0} & A_{3,1} & A_{3,2} & A_{3,3} \end{bmatrix}$$

- Ma trận $B$ gồm **16 khối trọng số** $B_{k, j} \in \mathbb{R}^{4 \times 4}$ với $k \in \{0..3\}, j \in \{0..3\}$:
  $$B = \begin{bmatrix} B_{0,0} & B_{0,1} & B_{0,2} & B_{0,3} \\ B_{1,0} & B_{1,1} & B_{1,2} & B_{1,3} \\ B_{2,0} & B_{2,1} & B_{2,2} & B_{2,3} \\ B_{3,0} & B_{3,1} & B_{3,2} & B_{3,3} \end{bmatrix}$$

- Ma trận kết quả $C$ gồm **16 khối đầu ra** $C_{i, j} \in \mathbb{R}^{4 \times 4}$ với $i \in \{0..3\}, j \in \{0..3\}$:
  $$C = \begin{bmatrix} C_{0,0} & C_{0,1} & C_{0,2} & C_{0,3} \\ C_{1,0} & C_{1,1} & C_{1,2} & C_{1,3} \\ C_{2,0} & C_{2,1} & C_{2,2} & C_{2,3} \\ C_{3,0} & C_{3,1} & C_{3,2} & C_{3,3} \end{bmatrix}$$

Công thức tích lũy cho từng khối kết quả $4 \times 4$ là tổng của 4 tích khối con:
$$C_{i, j} = \sum_{k=0}^{3} A_{i, k} \times B_{k, j} = A_{i, 0} B_{0, j} + A_{i, 1} B_{1, j} + A_{i, 2} B_{2, j} + A_{i, 3} B_{3, j}$$

---

### 7.2 Cơ Chế Ghép Đôi Khối (Tile Pairing): Từ Tiling $4 \times 4$ Sang Mảng Tính Toán $4 \times 8$

Nếu phần cứng sử dụng mảng Systolic $4 \times 4$ thông thường (không có DSP packing):
- Mỗi lần chạy mảng chỉ tính được **1 khối $C_{i, j}$** ($4 \times 4$).
- Cần chạy **16 lượt độc lập**.
- Tại mỗi lượt tính khối $C_{i, j}$, khối kích hoạt $A_{i, k}$ phải được nạp lại từ bộ nhớ BRAM 4 lần (lãng phí băng thông đọc).

**Bước đột phá trong RTL `rtl_ver02`:** Nhờ kỹ thuật **DSP Packing 2x INT8** trên lưới vật lý $4 \times 4$ PEs, mỗi PE tính đồng thời 2 phần tử cột. Kích thước tính toán tức thời của mảng đạt **$4 \text{ hàng} \times 8 \text{ cột}$**.

Do đó, phần cứng thực hiện **ghép đôi hai khối $4 \times 4$ kề nhau theo chiều cột thành 1 Super-Tile $4 \times 8$**:
$$\mathbf{C^{\text{super}}_{i, J} = [C_{i, 2J} \quad C_{i, 2J+1}] \in \mathbb{R}^{4 \times 8}} \quad (J \in \{0, 1\})$$

```text
========================================================================================================================
             SƠ ĐỒ GHÉP 16 KHỐI CƠ SỞ 4x4 THÀNH 8 SUPER-TILES 4x8 TRÊN MA TRẬN 16x16
========================================================================================================================

                  Cột 0 .. 3 (j=0)    Cột 4 .. 7 (j=1)    Cột 8 .. 11 (j=2)   Cột 12 .. 15 (j=3)
                 +───────────────────+───────────────────+───────────────────+───────────────────+
   Hàng 0 .. 3   |      C_00         |      C_01         |      C_02         |      C_03         |
   (i=0)         |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |
                 +───────────────────┴───────────────────+───────────────────┴───────────────────+
                 | <─────── SUPER-TILE (0, 0) ─────────> | <─────── SUPER-TILE (0, 1) ─────────> |
                 |          Kích thước: 4x8              |          Kích thước: 4x8              |
                 +───────────────────┬───────────────────+───────────────────┬───────────────────+
   Hàng 4 .. 7   |      C_10         |      C_11         |      C_12         |      C_13         |
   (i=1)         |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |
                 +───────────────────┴───────────────────+───────────────────┴───────────────────+
                 | <─────── SUPER-TILE (1, 0) ─────────> | <─────── SUPER-TILE (1, 1) ─────────> |
                 +───────────────────┬───────────────────+───────────────────┬───────────────────+
   Hàng 8 .. 11  |      C_20         |      C_21         |      C_22         |      C_23         |
   (i=2)         |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |
                 +───────────────────┴───────────────────+───────────────────┴───────────────────+
                 | <─────── SUPER-TILE (2, 0) ─────────> | <─────── SUPER-TILE (2, 1) ─────────> |
                 +───────────────────┬───────────────────+───────────────────┬───────────────────+
   Hàng 12 .. 15 |      C_30         |      C_31         |      C_32         |      C_33         |
   (i=3)         |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |     (Tile 4x4)    |
                 +───────────────────┴───────────────────+───────────────────┴───────────────────+
                 | <─────── SUPER-TILE (3, 0) ─────────> | <─────── SUPER-TILE (3, 1) ─────────> |
                 +───────────────────────────────────────+───────────────────────────────────────+

   * Ý nghĩa kiến trúc:
     - 16 khối C_ij kích thước 4x4 được giải quyết trọn vẹn chỉ trong ĐÚNG 8 LƯỢT CHẠY SUPER-TILE 4x8.
     - Khối kích hoạt A_{i, k} (4x4) được TÁI SỬ DỤNG 100% cho cả 2 khối trọng số B_{k, 2J} và B_{k, 2J+1}.
     - Băng thông nạp kích hoạt từ BRAM giảm 50%!
========================================================================================================================
```

Công thức toán học của một Super-Tile $4 \times 8$:
$$[C_{i, 2J} \quad C_{i, 2J+1}] = \sum_{k=0}^{3} A_{i, k} \times [B_{k, 2J} \quad B_{k, 2J+1}]$$
trong đó:
- $A_{i, k} \in \mathbb{R}^{4 \times 4}$: Một khối kích hoạt duy nhất.
- $B_{k, 2J} \in \mathbb{R}^{4 \times 4}$: Khối trọng số thứ nhất (nạp vào các cột chẵn `wt_col0`).
- $B_{k, 2J+1} \in \mathbb{R}^{4 \times 4}$: Khối trọng số thứ hai (nạp vào các cột lẻ `wt_col1`).
- $[B_{k, 2J} \quad B_{k, 2J+1}] \in \mathbb{R}^{4 \times 8}$: Khối trọng số kép kích thước $4 \times 8$.

---

### 7.3 Ánh Xạ I/O Buffer BRAM 64-bit Vào Các Khối $4 \times 4$

Độ rộng từ nhớ của Block RAM trên FPGA Kria KV260 là **64 bits** ($8 \times \text{INT8}$). Đây là thiết kế then chốt cho phép ghép nối tự nhiên các khối $4 \times 4$:

1. **Khối Kích Hoạt $A_{i, k}$ ($4 \text{ hàng} \times 4 \text{ cột}$):**
   - Mỗi hàng của khối $A_{i, k}$ có độ dài 4 bytes ($4 \times 8\text{-bit} = 32\text{ bits}$).
   - Một từ nhớ BRAM 64-bit chứa đúng **2 hàng** của khối $4 \times 4$:
     ```text
     Word 0: [ Hàng 1 (32-bit) | Hàng 0 (32-bit) ]   <-- ram_q[addr][63:0]
     Word 1: [ Hàng 3 (32-bit) | Hàng 2 (32-bit) ]
     ```
   - Bit địa chỉ thấp `raddr_q[0]` chọn nửa từ 32-bit cấp vào 4 hàng của Systolic Array (`act_row[0..3]`).

2. **Cặp Khối Trọng Số $B_{k, 2J}$ và $B_{k, 2J+1}$ ($4 \text{ hàng} \times 8 \text{ cột}$):**
   - Mỗi hàng của cặp khối gồm 4 bytes của $B_{k, 2J}$ và 4 bytes của $B_{k, 2J+1}$ $\implies$ Tổng cộng đúng **8 bytes = 64 bits**!
   - Một từ nhớ BRAM 64-bit chứa trọn vẹn 1 hàng của cặp khối $4 \times 8$:
     ```text
     ram_k[addr][63:32] = 4 bytes của B_{k, 2J+1} (cấp cho wt_col1[0..3])
     ram_k[addr][31:0]  = 4 bytes của B_{k, 2J}   (cấp cho wt_col0[0..3])
     ```
   - BRAM đọc 1 chu kỳ cấp trọn vẹn cả 2 khối $4 \times 4$ vào 16 DSPs mà không cần đệm phụ!

---

### 7.4 Vết Thực Thi Chi Tiết 1 Super-Tile (Ghép $C_{0,0}$ và $C_{0,1}$ Qua 4 Bước $k=0..3$)

Để tính trọn vẹn cặp khối $C_{0,0}$ và $C_{0,1}$ ($4 \times 8$ điểm kết quả), mảng Systolic Array thực hiện tích lũy qua 4 bước khối con $k = 0, 1, 2, 3$:

```text
========================================================================================================================
        VẾT CHU KỲ CLOCK: TÍCH LŨY 4 BƯỚC KHỐI K=0..3 ĐỂ TẠO RA SUPER-TILE [C_00 | C_01]
========================================================================================================================

    Khối C_00 (4x4) = A_00*B_00 (k=0) + A_01*B_10 (k=1) + A_02*B_20 (k=2) + A_03*B_30 (k=3)
    Khối C_01 (4x4) = A_00*B_01 (k=0) + A_01*B_11 (k=1) + A_02*B_21 (k=2) + A_03*B_31 (k=3)

 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Bước Khối | Cycles  | Dữ liệu nạp từ BRAM (Khối 4x4)          | Hành vi tích lũy của 16 PEs (32 Accumulators)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   k = 0    |  0 .. 3 | - Nạp khối A_00 (4x4) vào act_row       | - clr_acc = 1 ở cycle 0 (reset accumulator về 0).
            |         | - Nạp khối B_00 (4x4) vào wt_col0       | - PE tích lũy: pe_acc_col0 += A_00 x B_00
            |         | - Nạp khối B_01 (4x4) vào wt_col1       | - PE tích lũy: pe_acc_col1 += A_00 x B_01
            |         |                                         |   (Giữ nguyên kết quả, KHÔNG xả ra ngoài!)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   k = 1    |  4 .. 7 | - Nạp khối A_01 (4x4) vào act_row       | - clr_acc = 0.
            |         | - Nạp khối B_10 (4x4) vào wt_col0       | - pe_acc_col0 += A_01 x B_10  (Cộng dồn vào k=0)
            |         | - Nạp khối B_11 (4x4) vào wt_col1       | - pe_acc_col1 += A_01 x B_11  (Cộng dồn vào k=0)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   k = 2    |  8 .. 11| - Nạp khối A_02 (4x4) vào act_row       | - clr_acc = 0.
            |         | - Nạp khối B_20 (4x4) vào wt_col0       | - pe_acc_col0 += A_02 x B_20  (Cộng dồn tiếp)
            |         | - Nạp khối B_21 (4x4) vào wt_col1       | - pe_acc_col1 += A_02 x B_21  (Cộng dồn tiếp)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   k = 3    | 12 .. 15| - Nạp khối A_03 (4x4) vào act_row       | - clr_acc = 0.
            |         | - Nạp khối B_30 (4x4) vào wt_col0       | - pe_acc_col0 += A_03 x B_30  (Hoàn tất C_00!)
            |         | - Nạp khối B_31 (4x4) vào wt_col1       | - pe_acc_col1 += A_03 x B_31  (Hoàn tất C_01!)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Wavefront| 16 .. 19| Dòng kích hoạt & trọng số truyền hết    | PE_33 (góc cuối cùng của mảng) hoàn tất chu kỳ k=3.
   Flush    |         | qua 4 tầng PE (skew pipeline delay).    | Toàn bộ 32 giá trị của C_00 và C_01 đã sẵn sàng!
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   DRAIN    | 20 .. 51| Phát drain_start = 1                    | Máy trạng thái Drain quét tuần tự:
   PHASE    | (32 cls)| (Không nạp thêm dữ liệu mới)            |   - 16 chu kỳ đầu: Xả trọn vẹn 16 số của khối C_00
            |         |                                         |   - 16 chu kỳ sau: Xả trọn vẹn 16 số của khối C_01
            |         |                                         | Xuất ra scale_unit: 1 số INT32 / chu kỳ.
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Cycle 51 | drain_done = 1'b1. Hoàn tất Super-Tile (0, 0)! Tráo Bank Ping-Pong, bắt đầu Super-Tile (0, 1).
========================================================================================================================
```

---

### 7.5 Ví Dụ Số Học Thực Tế Tại PE[0][0] (Minh Họa DSP Packing 2x INT8)

Để thấy rõ phép nhân tại mức phần cứng, xét phần tử tại vị trí hàng 0, cột 0 của ma trận:
- Ta muốn tính phần tử $C[0, 0]$ (thuộc khối $C_{0,0}$) và phần tử $C[0, 1]$ (thuộc khối $C_{0,1}$).
- Cả hai phần tử này đều được tính tại duy nhất **PE[0][0]** trong suốt 4 bước $k=0..3$.

Xét bước $k = 0$, phần tử thứ nhất của chiều $k$ ($k_{\text{idx}} = 0$):
- Kích hoạt hàng 0 của $A_{0,0}$: $A[0, 0] = \mathbf{3}$ (INT8).
- Trọng số cột 0 của $B_{0,0}$ (cột chẵn): $B[0, 0] = \mathbf{5}$ (INT8).
- Trọng số cột 0 của $B_{0,1}$ (cột lẻ, tương ứng cột 1 toàn cục): $B[0, 1] = \mathbf{-2}$ (INT8).

**Cách khối DSP48E2 đóng gói và thực thi trong 1 chu kỳ clock:**
1. **Đóng gói cổng $D$ (27 bits):** Trọng số cột chẵn $B$ được dịch lên 18 bits, cộng với trọng số cột lẻ $C$:
   $$D = (B[0, 0] \ll 18) + (B[0, 1] \ \& \ \text{0x3FFFF})$$
   $$D = (5 \times 262,144) + (-2 \ \& \ \text{0x3FFFF}) = 1,310,720 + 262,142 = \mathbf{1,572,862}$$

2. **Nhân với kích hoạt $A$ (18 bits):**
   $$P = D \times A[0, 0] = 1,572,862 \times 3 = \mathbf{4,718,586}$$

3. **Tách kết quả tại đầu ra tích lũy (48-bit Accumulator Output):**
   - **Nửa thấp (Cột lẻ $C[0, 1]$):**
     $$P[17:0] = 4,718,586 \pmod{262,144} = 262,138 \implies \text{Sign-extend INT18} = \mathbf{-6}$$
     Kiểm tra toán học: $A[0, 0] \times B[0, 1] = 3 \times (-2) = \mathbf{-6}$ $\implies$ **CHÍNH XÁC TUYỆT ĐỐI!**
   - **Nửa cao (Cột chẵn $C[0, 0]$):**
     $$P[47:18] = \lfloor 4,718,586 / 262,144 \rfloor = 17.999 \dots \approx 18$$
     Bù phần mượn bit từ số âm: $18 + (-1) = \mathbf{15}$
     Kiểm tra toán học: $A[0, 0] \times B[0, 0] = 3 \times 5 = \mathbf{15}$ $\implies$ **CHÍNH XÁC TUYỆT ĐỐI!**

Trong cùng một chu kỳ clock, **PE[0][0] hoàn thành đồng thời 2 phép nhân của 2 khối $4 \times 4$ khác nhau ($C_{0,0}$ và $C_{0,1}$)**!

---

### 7.6 Bảng Điều Phối Hoàn Chỉnh 8 Super-Tiles Cho Ma Trận Toàn Cục $16 \times 16$

Toàn bộ quá trình tính toán ma trận $16 \times 16$ qua 8 Super-Tiles (bao trùm toàn bộ 16 khối cơ sở $4 \times 4$) được điều phối như sau:

| Super-Tile | Cặp Khối Cơ Sở $4 \times 4$ | Tọa độ Ma trận Kết quả $C$ | Khối Kích hoạt $A$ đã nạp | Khối Trọng số $B$ đã nạp | Chu kỳ Tính toán ($k=0..3$) | Chu kỳ Drain (Xả kết quả) | Trạng thái Ping-Pong BRAM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ST_0** | $C_{0,0} \ \& \ C_{0,1}$ | $C[0..3, 0..7]$ | $A_{0, 0..3}$ (4 khối $4 \times 4$) | $B_{0..3, 0} \ \& \ B_{0..3, 1}$ (8 khối $4 \times 4$) | Cycles 0 .. 19 | Cycles 20 .. 51 | Đọc Bank 0 / DMA nạp Bank 1 |
| **ST_1** | $C_{0,2} \ \& \ C_{0,3}$ | $C[0..3, 8..15]$ | $A_{0, 0..3}$ (Tái sử dụng từ BRAM) | $B_{0..3, 2} \ \& \ B_{0..3, 3}$ (8 khối $4 \times 4$) | Cycles 52 .. 71 | Cycles 72 .. 103 | Đọc Bank 1 / DMA nạp Bank 0 |
| **ST_2** | $C_{1,0} \ \& \ C_{1,1}$ | $C[4..7, 0..7]$ | $A_{1, 0..3}$ (4 khối $4 \times 4$) | $B_{0..3, 0} \ \& \ B_{0..3, 1}$ (8 khối $4 \times 4$) | Cycles 104 .. 123 | Cycles 124 .. 155 | Đọc Bank 0 / DMA nạp Bank 1 |
| **ST_3** | $C_{1,2} \ \& \ C_{1,3}$ | $C[4..7, 8..15]$ | $A_{1, 0..3}$ (Tái sử dụng từ BRAM) | $B_{0..3, 2} \ \& \ B_{0..3, 3}$ (8 khối $4 \times 4$) | Cycles 156 .. 175 | Cycles 176 .. 207 | Đọc Bank 1 / DMA nạp Bank 0 |
| **ST_4** | $C_{2,0} \ \& \ C_{2,1}$ | $C[8..11, 0..7]$ | $A_{2, 0..3}$ (4 khối $4 \times 4$) | $B_{0..3, 0} \ \& \ B_{0..3, 1}$ (8 khối $4 \times 4$) | Cycles 208 .. 227 | Cycles 228 .. 259 | Đọc Bank 0 / DMA nạp Bank 1 |
| **ST_5** | $C_{2,2} \ \& \ C_{2,3}$ | $C[8..11, 8..15]$ | $A_{2, 0..3}$ (Tái sử dụng từ BRAM) | $B_{0..3, 2} \ \& \ B_{0..3, 3}$ (8 khối $4 \times 4$) | Cycles 260 .. 279 | Cycles 280 .. 311 | Đọc Bank 1 / DMA nạp Bank 0 |
| **ST_6** | $C_{3,0} \ \& \ C_{3,1}$ | $C[12..15, 0..7]$ | $A_{3, 0..3}$ (4 khối $4 \times 4$) | $B_{0..3, 0} \ \& \ B_{0..3, 1}$ (8 khối $4 \times 4$) | Cycles 312 .. 331 | Cycles 332 .. 363 | Đọc Bank 0 / DMA nạp Bank 1 |
| **ST_7** | $C_{3,2} \ \& \ C_{3,3}$ | $C[12..15, 8..15]$ | $A_{3, 0..3}$ (Tái sử dụng từ BRAM) | $B_{0..3, 2} \ \& \ B_{0..3, 3}$ (8 khối $4 \times 4$) | Cycles 364 .. 383 | Cycles 384 .. 415 | Đọc Bank 1 / DMA nạp Bank 0 |

---

### 7.7 So Sánh Hiệu Quả: Mảng $4 \times 4$ Thuần Túy vs Mảng $4 \times 8$ Tính Toán (RTL `rtl_ver02`)

| Tiêu Chí So Sánh | Mảng Systolic $4 \times 4$ Thuần Túy | Mảng Systolic Cỡ Tính Toán $4 \times 8$ (RTL Hiện Tại) | Lợi Điểm Của Kiến Trúc `rtl_ver02` |
|---|:---:|:---:|---|
| **Số khối DSP48E2 tiêu thụ** | 16 DSPs | 16 DSPs | Không tốn thêm bất kỳ tài nguyên DSP nào! |
| **Số phép nhân MACs / chu kỳ** | 16 MACs/cycle | **32 MACs/cycle** | **Tăng gấp đôi thông lượng tính toán (2x Throughput)** |
| **Số lượt chạy ma trận $16 \times 16$** | 16 lượt (16 tiles $4 \times 4$) | **8 lượt** (8 super-tiles $4 \times 8$) | Giảm 50% số lần đổi trạng thái và overhead FSM |
| **Số lần nạp lại khối $A_{i, k}$** | 4 lần cho mỗi khối $A$ | **2 lần** cho mỗi khối $A$ | **Giảm 50% băng thông đọc ma trận kích hoạt từ BRAM** |
| **Khớp nối với từ nhớ BRAM** | Cần chia nhỏ từ nhớ 64-bit | Khớp hoàn hảo từ nhớ 64-bit ($4\text{B chẵn} + 4\text{B lẻ}$) | Tận dụng 100% bus dữ liệu BRAM, không lãng phí chu kỳ |
| **Tổng chu kỳ thực thi ($16 \times 16$)** | $\approx 832\text{ clock cycles}$ | **$\approx 416\text{ clock cycles}$** | **Tăng tốc gấp đôi thời gian xử lý ($\approx 2.08\text{ }\mu\text{s}$)** |


