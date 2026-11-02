# Kiến Trúc MHA Level 1 v2.0: Hệ Thống Bộ Nhớ Đệm Ingress, Ping-Pong BRAM & Skid Buffer

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mha_level_1_bo_nho_dem_lenh.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/m## 1. TỔNG QUAN HỆ THỐNG BỘ NHỚ ĐỆM INGRESS

Hệ thống bộ nhớ đệm Ingress chịu trách nhiệm thu nhận dòng dữ liệu AXI4-Stream 128-bit native tốc độ cao từ kênh DMA MM2S, loại bỏ hiện tượng bong bóng đường ống (Zero Bubble Ingress), và giải mã tự động nạp vào 3 phân vùng bộ nhớ nội chip của Head hiện hành ($Q, K, V$).

```text
========================================================================================================================
                              LEVEL 1: INGRESS BUFFERING & ON-CHIP MEMORY ARCHITECTURE (128-BIT NATIVE)
========================================================================================================================

    AXI DMA MM2S (DDR4 Memory)
   +---------------------------+
   | 128-bit Stream TX Master  |
   +-------------+-------------+
                 |
                 | S_AXIS_TDATA [127:0], TVALID, TREADY, TLAST (16 INT8 tokens/beat)
                 v
   +-------------------------------------------------------------------------------------------------------------------+
   | SKID BUFFER (axis_adapter.sv - DATA_WIDTH = 128)                                                                  |
   | - Lưu trữ tức thời 1 beat 128-bit dự phòng khi downstream bận (tready = 0)                                        |
   | - Triệt tiêu 100% rủi ro mất dữ liệu trên bus bắt tay AXI-Stream                                                  |
   +--------------------------------------------+----------------------------------------------------------------------+
                                                |
                                                | skid_tdata [127:0], skid_tvalid, skid_tready
                                                v
   +-------------------------------------------------------------------------------------------------------------------+
   | FSM GIẢI MÃ NẠP TỰ ĐỘNG INGRESS (ing_load_state in attention_core_top_*.sv)                                       |
   |                                                                                                                   |
   |   [ING_Q]  --> Nạp đủ beats_per_mat beats vào q_mem [256][MAX_HEAD_DIM]                                          |
   |      |                                                                                                            |
   |      v                                                                                                            |
   |   [ING_K]  --> Tự động chuyển nạp beats_per_mat beats vào k_mem [256][MAX_HEAD_DIM]                               |
   |      |                                                                                                            |
   |      v                                                                                                            |
   |   [ING_V]  --> Tự động chuyển nạp beats_per_mat beats vào v_mem [256][MAX_HEAD_DIM]                               |
   |      |                                                                                                            |
   |      v                                                                                                            |
   |   [ING_READY] -> Phát cờ head_data_ready = 1'b1 kích hoạt Attention Pipeline bắt đầu tính toán                    |
   +----------------------------------------------------+--------------------------------------------------------------+
                                                        |
                                                        v
   +-------------------------------------------------------------------------------------------------------------------+
   | ON-CHIP ACTIVE HEAD STORAGE (Block RAM tổng hợp được trên UltraScale+)                                             |
   | (* ram_style = "block" *)                                                                                         |
   |                                                                                                                   |
   |   +------------------------------------+  +------------------------------------+  +----------------------------+  |
   |   | q_mem [256][32] (INT8 Signed)      |  | k_mem [256][32] (INT8 Signed)      |  | v_mem [256][32] (INT8 S.)  |  |
   |   | Port A: Ghi từ Ingress Stream      |  | Port A: Ghi từ Ingress Stream      |  | Port A: Ghi Ingress Stream |  |
   |   | Port B: Đọc song song d_k chiều    |  | Port B: Đọc song song d_k chiều    |  | Port B: Cung cấp cho S*V   |  |
   |   +------------------------------------+  +------------------------------------+  +----------------------------+  |
   +-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. NGUYÊN LÝ GIẢI MÃ ĐỊA CHỈ TOKEN ĐA CHIỀU (MULTI-STAGE ADDRESSING TRÊN BUS 128-BIT)

Mỗi beat AXI4-Stream 128-bit chứa **16 giá trị INT8** (16 bytes/beat). Mạch giải mã trong RTL kiểm tra trực tiếp kích thước $d_k$ (`reg_head_dim`) để xác định số beats cần nạp:

```systemverilog
always_comb begin
    if (reg_head_dim <= 16'd16) begin
        beats_per_mat = 16'(reg_seq_len);         // 256 beats (1 beat/token)
        ing_tok_idx   = 8'(ing_beat_cnt);
        ing_slice_idx = 2'b00;
    end else begin
        beats_per_mat = 16'(reg_seq_len << 1);     // 512 beats (2 beats/token)
        ing_tok_idx   = 8'(ing_beat_cnt >> 1);
        ing_slice_idx = 2'(ing_beat_cnt & 16'd1);
    end
end
```

### 2.1 Stage 2 ($d_k = 16$):
- Mỗi token chiếm đúng 16 bytes $\implies$ **đúng 1 beat 128-bit duy nhất**:
  - Beat 0: Chứa trọn vẹn 16 chiều đặc trưng (`dim 0..15`).
- Giải mã địa chỉ:
  - Token index: `ing_tok_idx = ing_beat_cnt;`
  - Slice index: `ing_slice_idx = 2'b00;`
- Tổng số beats mỗi ma trận: $256 \times 1 = \mathbf{256\text{ beats}}$.
- Tổng số beats cho cả 3 ma trận $Q, K, V$: $3 \times 256 = \mathbf{768\text{ beats}}$ (tiết kiệm 50% số nhịp truyền so với bus 64-bit cũ).

### 2.2 Stage 3 ($d_k = 20$) & Stage 4 ($d_k = 24$):
- Số chiều vượt quá 16 bytes, áp dụng chuẩn **Token-Aligned Padding** (mỗi token chiếm 2 beats 128-bit = 32 bytes):
  - **Beat 0** (`ing_slice_idx == 0`): 16 bytes đầu (`dim 0..15`).
  - **Beat 1** (`ing_slice_idx == 1`):
    - Stage 3 ($d_k = 20$): 4 bytes (`dim 16..19`) + 12 bytes đệm zero.
    - Stage 4 ($d_k = 24$): 8 bytes (`dim 16..23`) + 8 bytes đệm zero.
    - Mở rộng $d_k = 32$: 16 bytes trọn vẹn (`dim 16..31`).
- Giải mã địa chỉ:
  - Token index: `ing_tok_idx = ing_beat_cnt >> 1;`
  - Slice index: `ing_slice_idx = ing_beat_cnt & 1;`
- Tổng số beats mỗi ma trận: $256 \times 2 = \mathbf{512\text{ beats}}$.
- Tổng số beats cho cả 3 ma trận $Q, K, V$: $3 \times 512 = \mathbf{1,536\text{ beats}}$.

---

## 3. CƠ CHẾ BẢO VỆ CHỐNG NGHẼN SKID BUFFER (`axis_adapter.sv`)

Mô-đun Skid Buffer giải quyết triệt để vấn đề phản hồi trễ tín hiệu `TREADY` (Backpressure Handshake):
- Khi `m_axis_tready = 1`: Dữ liệu đi thẳng qua tầng đệm với độ trễ 0 cycle.
- Khi `m_axis_tready = 0` bất ngờ: Thanh ghi nội bộ lưu giữ ngay beat hiện tại và hạ `s_axis_tready = 0` về phía DMA ở chu kỳ kế tiếp, đảm bảo không mất mát bất kỳ byte dữ liệu nào.
