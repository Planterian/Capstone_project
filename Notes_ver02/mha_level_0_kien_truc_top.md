# Kiến Trúc MHA Level 0 v2.0: Top-Level System Architecture, Control Plane & Bandwidth Analysis

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mha_level_0_kien_truc_top.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mha_level_0_kien_truc_top.md)

---

## 1. SƠ ĐỒ KHỐI CẤP CAO (TOP-LEVEL SYSTEM BLOCK DIAGRAM)

```text
========================================================================================================================
                                LEVEL 0: TOP-LEVEL RTL ARCHITECTURE (attention_core_top.sv)
========================================================================================================================

       ARM Cortex-A53 PS (PYNQ Driver / Linux Host)                     AXI Direct Memory Access (DMA Engine)
  +----------------------------------------------------+          +-----------------------------------------------+
  | AXI4-Lite Master (Memory-Mapped Bus)               |          | Simple DMA MM2S (Read DDR4) -> Stream Master  |
  | Regs: 0x00=CTRL, 0x04=STATUS, 0x10=N, 0x14=dk, ... |          | Simple DMA S2MM (Write DDR4) <- Stream Slave  |
  +-------------------------+--------------------------+          +-----------+-----------------------+-----------+
                            |                                                 |                       ^
                            | AXI4-Lite (32-bit Control / CSR)                | AXI4-Stream           | AXI4-Stream
                            | (awaddr, wdata, araddr, rdata)                  | RX (64-bit Ingress)   | TX (64-bit Egress)
                            v                                                 v                       |
  +-------------------------------------------------------------------------------------------------------------------+
  | ATTENTION CORE TOP-LEVEL SHELL (attention_core_top.sv)                                                            |
  |                                                                                                                   |
  |  +-------------------------------------------------------------------------------------------------------------+  |
  |  | CONTROL PLANE: AXI4-Lite Register File & Sequential Head Controller (axi_lite_regs.sv)                      |  |
  |  |   - 0x00: CONTROL [START, CLEAR_BUF, SOFT_RESET]  - 0x20: SCALE_M (16-bit signed mantissa: 27640)           |  |
  |  |   - 0x04: STATUS  [BUSY, DONE, ERROR, BANK_SEL]   - 0x24: SCALE_E (5-bit shift exponent: 22)               |  |
  |  |   - 0x10: SEQ_LEN (N = 256 tokens)                - 0x40: LUT_WR_ADDR (10-bit BRAM ROM write address)       |  |
  |  |   - 0x14: HEAD_DIM (d_k = 16, 20, 24)             - 0x44: LUT_WR_DATA (20-bit BRAM ROM write data)          |  |
  |  |   - 0x18: NUM_HEADS (H = 4 sequential heads)                                                                |  |
  |  +---------------------------------------------------+---------------------------------------------------------+  |
  |                                                      |                                                            |
  |                                                      | Control Wires: start_pulse, reg_head_dim, cfg_lut_*        |
  |                                                      v                                                            |
  |  +-------------------------------------------------------------------------------------------------------------+  |
  |  | DATA PLANE: STREAMING ATTENTION PIPELINE                                                                    |  |
  |  |                                                                                                             |  |
  |  |  AXI-Stream RX Skid Buffer                                                                                  |  |
  |  |  +------------------+                                                                                       |  |
  |  |  | axis_adapter_rx  |===(64-bit Stream with Backpressure TVALID/TREADY/TLAST)===+                            |  |
  |  |  +------------------+                                                           |                            |  |
  |  |                                                                                 v                            |  |
  |  |                                                             +---------------------------------------+        |  |
  |  |                                                             | Ingress Multi-Bank Buffer             |        |  |
  |  |                                                             | FSM nạp đủ Q -> K -> V (1536 beats)   |        |  |
  |  |                                                             | On-chip BRAM (q_mem, k_mem, v_mem)    |        |  |
  |  |                                                             +-------------------+-------------------+        |  |
  |  |                                                                                 |                            |  |
  |  |                                                                                 v                            |  |
  |  |                                                             +---------------------------------------+        |  |
  |  |                                                             | STAGE 1: QK^T GEMM Engine             |        |  |
  |  |                                                             | Nhân song song d_k chiều (INT32)      |        |  |
  |  |                                                             +-------------------+-------------------+        |  |
  |  |                                                                                 | (INT32 Raw Scores)         |  |
  |  |                                                                                 v                            |  |
  |  |                                                             +---------------------------------------+        |  |
  |  |                                                             | STAGE 2: Dyadic Rescaler Unit         |        |  |
  |  |                                                             | (Score x M) >>> E -> INT16 Clamped    |        |  |
  |  |                                                             +-------------------+-------------------+        |  |
  |  |                                                                                 | (INT16 Scaled Scores)      |  |
  |  |                                                                                 v                            |  |
  |  |                                                             +---------------------------------------+        |  |
  |  |                                                             | STAGE 3: Reconfigurable Softmax Engine|        |  |
  |  |                                                             | True Dual-Port 1024w BRAM Exp ROM     |        |  |
  |  |                                                             | 32-step Restoring Divider -> UINT16   |        |  |
  |  |                                                             +-------------------+-------------------+        |  |
  |  |                                                                                 | (UINT16 Probabilities)     |  |
  |  |                                                                                 v                            |  |
  |  |                                                             +---------------------------------------+        |  |
  |  |                                                             | STAGE 4: Score x V Context GEMM       |        |  |
  |  |                                                             | Tích lũy 256 tokens, >>> 15, INT8 Out |        |  |
  |  |                                                             +-------------------+-------------------+        |  |
  |  |                                                                                 |                            |  |
  |  |                                                                                 v                            |  |
  |  |  AXI-Stream TX Interface                                                        |                            |  |
  |  |  +------------------+                                                           |                            |  |
  |  |  | m_axis_tdata     |<==========================================================+                            |  |
  |  |  | 512 beats / head |                                                                                        |  |
  |  |  +------------------+                                                                                        |  |
  |  +-------------------------------------------------------------------------------------------------------------+  |
  +-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. BẢN ĐỒ THANH GHI AXI4-LITE CSR (REGISTER MAP)

| Offset Địa chỉ | Tên Thanh Ghi | Quyền | Reset Value | Mô tả Kỹ Thuật |
|:---:|:---|:---:|:---:|:---|
| `0x00` | **CONTROL** | R/W | `0x0000_0000` | Bit [0]: START (tự động xóa sau 1 nhịp)<br>Bit [1]: CLEAR_BUF (xóa bộ đệm)<br>Bit [2]: SOFT_RESET (reset mềm) |
| `0x04` | **STATUS** | RO | `0x0000_0000` | Bit [0]: BUSY (1 = Core đang tính toán)<br>Bit [1]: DONE (1 = Hoàn thành toàn bộ Heads)<br>Bit [2]: ERROR<br>Bit [3]: PP_BANK (Bank đang đọc) |
| `0x08` | **NBYTES** | R/W | `0x0000_1000` | Tổng số bytes truyền nhận cho một giao dịch DMA |
| `0x0C` | **CORE_ID** | RO | `0x5649_5432` | Mã định danh phần cứng (`"VIT2"`) |
| `0x10` | **SEQ_LEN** | R/W | `0x0000_0100` | Chiều dài chuỗi token $N = 256$ (MobileViT-XXS) |
| `0x14` | **HEAD_DIM** | R/W | `0x0000_0010` | Số chiều mỗi Head: `16` (Stage 2), `20` (Stage 3), `24` (Stage 4) |
| `0x18` | **NUM_HEADS** | R/W | `0x0000_0004` | Số lượng Heads tuần tự: $H = 4$ |
| `0x20` | **SCALE_M** | R/W | `0x0000_6BF8` | Hệ số nhân Dyadic Mantissa 16-bit (Stage 2 mặc định: $M = 27640$) |
| `0x24` | **SCALE_E** | R/W | `0x0000_0016` | Số bit dịch Dyadic Exponent 5-bit (mặc định: $E = 22$) |
| `0x40` | **LUT_WR_ADDR**| R/W | `0x0000_0000` | Địa chỉ nạp BRAM Softmax LUT Port B (0..1023) |
| `0x44` | **LUT_WR_DATA**| R/W | `0x0000_0000` | Dữ liệu 20-bit ghi vào BRAM Exp ROM tại địa chỉ `LUT_WR_ADDR` |

---

## 3. PHÂN TÍCH BĂNG THÔNG & ĐỘ TRỄ (BANDWIDTH & LATENCY MODEL)

### 3.1 Lưu lượng dữ liệu trên 1 Head ($N=256, d_k=16$):
1. **Dữ liệu nạp Ingress (PS -> PL qua DMA MM2S):**
   - Ma trận Q: $256 \times 16 \text{ bytes} = 4,096\text{ bytes} \implies 512\text{ beats (64-bit)}$.
   - Ma trận K: $256 \times 16 \text{ bytes} = 4,096\text{ bytes} \implies 512\text{ beats (64-bit)}$.
   - Ma trận V: $256 \times 16 \text{ bytes} = 4,096\text{ bytes} \implies 512\text{ beats (64-bit)}$.
   - **Tổng lưu lượng Ingress:** $1,536\text{ beats} = 12,288\text{ bytes}$.
2. **Dữ liệu xuất Egress (PL -> PS qua DMA S2MM):**
   - Ma trận Context Output: $256 \times 16 \text{ bytes} = 4,096\text{ bytes} \implies 512\text{ beats (64-bit)}$.

### 3.2 Chu kỳ thực thi (Cycle Breakdown):
Tại tần số $200\text{ MHz}$ ($T_{\text{clk}} = 5.0\text{ ns}$):
- Thời gian nạp dữ liệu Ingress: $1,536\text{ cycles}$.
- Thời gian tính toán Attention ($256\text{ hàng}$):
  - Mỗi hàng mất khoảng $800\text{ cycles}$ (gồm $256\text{ cycles}$ GEMM $QK^T$, $256\text{ cycles}$ Softmax max/acc, $32\text{ cycles}$ chia, $256\text{ cycles}$ Score*V).
  - $256 \times 800 \approx 204,800\text{ cycles}$.
- **Tổng chu kỳ đo được thực tế:** **208,384 clock cycles** ($\approx 1.04\text{ ms}$).
- Đối với toàn bộ 4 Heads: $4 \times 1.04\text{ ms} \approx 4.16\text{ ms}$ cho 1 lớp Attention.
- Băng thông AXI4-Stream trung bình: $\approx 1.6\text{ GB/s}$ (trong khả năng đáp ứng $3.2\text{ GB/s}$ của bus AXI HP trên Zynq UltraScale+).
