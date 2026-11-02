# BÁO CÁO KẾT QUẢ MÔ PHỎNG THẨM ĐỊNH RTL (DYNAMIC SEQUENCE LENGTH N) TRÊN CẢ 3 HARDWARE VARIANTS

**Ngày cập nhật:** 02/11/2026  
**Dự án:** Co-Design MobileViT-XXS + I-ViT trên nền tảng AMD Kria KV260 (`xck26-sfvc784-2LV-c`)  
**Mục tiêu:** Thẩm định độ chính xác 100% Bit-Exact (0-LSB error) và đo lường xung nhịp thực thi (clock cycles) của phần cứng MHA Core khi chuyển đổi linh hoạt kích thước chuỗi token $N \in \{256, 64, 16\}$ tương ứng với 3 Stage Transformer (Stage 2, 3, 4).

---

## 1. Bảng Thông Số Kỹ Thuật Các Stage (H = 4 Cố Định)

| Tham số Transformer | Stage 2 (2 Layers) | Stage 3 (4 Layers) | Stage 4 (3 Layers) |
| :--- | :---: | :---: | :---: |
| **Kích thước chuỗi token ($N$)** | **256** | **64** | **16** |
| **Độ dài nhúng ($d$)** | 64 | 80 | 96 |
| **Số lượng Head ($H$)** | 4 | 4 | 4 |
| **Head Dimension ($d_k = d/H$)** | **16** | **20** | **24** |
| **Kích thước ma trận $Q, K, V$ / Head** | $256 \times 16$ (4,096 B) | $64 \times 20$ (1,280 B) | $16 \times 24$ (384 B) |
| **Tổng Ingress DMA ($Q + K + V$)** | 12,288 Bytes | 3,840 Bytes | 1,152 Bytes |
| **Kích thước Score Matrix ($Q \times K^T$)** | $256 \times 256$ (65,536 items) | $64 \times 64$ (4,096 items) | $16 \times 16$ (256 items) |
| **Tổng Egress DMA (Context Output)** | 4,096 Bytes | 1,280 Bytes | 384 Bytes |
| **Dyadic Scaling ($M, E$)** | $M=27640, E=22$ | $M=21936, E=22$ | $M=18810, E=22$ |

---

## 2. Kết Quả Kiểm Thử Độc Lập Từng Stage (Isolated Stage DV)

### Variant 2A (DSP-Packed Adder Tree - Baseline đề tài)
*File script chạy:* `run_stage_isolated_dv_2A.bat`
* **Stage 2 ($N=256, d_k=16$):**
  - Egress samples verified: 4,096 / 4,096 bytes (100% Bit-Exact, 0 error).
  - TLAST & TKEEP: Hoàn toàn chính xác trên beat 255.
  - Chu kỳ xung nhịp (Clock Cycles): **210,183 cycles** (~1.05 ms @ 200 MHz).
* **Stage 3 ($N=64, d_k=20$):**
  - Egress samples verified: 1,280 / 1,280 bytes (100% Bit-Exact, 0 error).
  - TLAST: Assert chính xác tại beat 127.
  - Chu kỳ xung nhịp (Clock Cycles): **17,791 cycles** (~88.95 $\mu$s @ 200 MHz).
  - **Tăng tốc (Speedup) so với Stage 2:** **11.81x**!
* **Stage 4 ($N=16, d_k=24$):**
  - Egress samples verified: 384 / 384 bytes (100% Bit-Exact, 0 error).
  - TLAST: Assert chính xác tại beat 31.
  - Chu kỳ xung nhịp (Clock Cycles): **1,759 cycles** (~8.79 $\mu$s @ 200 MHz).
  - **Tăng tốc (Speedup) so với Stage 2:** **119.49x**!

---

### Variant 1A (Pipelined Adder Tree - 16 DSP48E2)
*File script chạy:* `run_stage_isolated_dv_1A.bat`
* **Stage 2 ($N=256, d_k=16$):**
  - Egress samples verified: 4,096 / 4,096 bytes (100% Bit-Exact, 0 error).
  - Chu kỳ xung nhịp: **207,623 cycles** (~1.04 ms @ 200 MHz).
* **Stage 3 ($N=64, d_k=20$):**
  - Egress samples verified: 1,280 / 1,280 bytes (100% Bit-Exact, 0 error).
  - Chu kỳ xung nhịp: **17,842 cycles** (~89.21 $\mu$s @ 200 MHz).
* **Stage 4 ($N=16, d_k=24$):**
  - Egress samples verified: 384 / 384 bytes (100% Bit-Exact, 0 error).
  - Chu kỳ xung nhịp: **1,733 cycles** (~8.66 $\mu$s @ 200 MHz).

---

### Variant 3B (2D Systolic MAC Array - 16 DSP48E2 Tile-based)
*File script chạy:* `run_stage_isolated_dv_3B.bat`
* **Stage 2 ($N=256, d_k=16$):**
  - Egress samples verified: 4,096 / 4,096 bytes (100% Bit-Exact, 0 error).
  - Chu kỳ xung nhịp: **337,472 cycles** (~1.69 ms @ 200 MHz).
* **Stage 3 ($N=64, d_k=20$):**
  - Egress samples verified: 1,280 / 1,280 bytes (100% Bit-Exact, 0 error).
  - Chu kỳ xung nhịp: **23,888 cycles** (~119.44 $\mu$s @ 200 MHz).
  - **Tăng tốc (Speedup) so với Stage 2:** **14.13x**!
* **Stage 4 ($N=16, d_k=24$):**
  - Egress samples verified: 384 / 384 bytes (100% Bit-Exact, 0 error).
  - Chu kỳ xung nhịp: **2,092 cycles** (~10.46 $\mu$s @ 200 MHz).
  - **Tăng tốc (Speedup) so với Stage 2:** **161.32x**!

---

## 3. Kết Quả Kiểm Thử Chuỗi Liên Tục Không Reset (Sequential Multi-Stage DV)

Mục đích: Đảm bảo phần cứng chuyển đổi cấu hình thanh ghi CSR trong lúc hoạt động bình thường giữa Stage 2 -> Stage 3 -> Stage 4 mà không bị kẹt FSM, không rò rỉ buffer, và giữ vững 100% bit-exact.

| Phần cứng Accelerator | Tổng Context Bytes So Sánh | Số Lỗi Mismatch (0-LSB) | Tổng Chu Kỳ Xung Nhịp (Cycles) | Thời gian tương đương (@ 200 MHz) | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant 1A (Pipelined Adder Tree)** | 5,760 / 5,760 | **0** | **227,198 cycles** | 1.136 ms | **PASS (100% Bit-Exact)** |
| **Variant 2A (DSP-Packed Tree)** | 5,760 / 5,760 | **0** | **229,710 cycles** | 1.148 ms | **PASS (100% Bit-Exact)** |
| **Variant 3B (2D Systolic Array)** | 5,760 / 5,760 | **0** | **363,437 cycles** | 1.817 ms | **PASS (100% Bit-Exact)** |

*Ghi chú:* Tổng số bytes kiểm tra trong chuỗi liên tục = $4,096 + 1,280 + 384 = 5,760$ Bytes.

---

## 4. Đánh Giá Hiệu Năng & Kết Luận Kỹ Thuật

1. **Hiệu quả của Dynamic Token Sizing:**
   Trước khi sửa lỗi, phần cứng bị ép chạy cứng $N=256$ cho tất cả các stage, làm lãng phí hàng trăm ngàn chu kỳ tính toán vô ích tại Stage 3 và Stage 4.  
   Sau khi hỗ trợ dynamic $N$:
   - Stage 3 ($N=64$) giảm thời gian chạy từ ~210,000 cycles xuống còn **~17,790 cycles** (giảm 91.5% thời gian).
   - Stage 4 ($N=16$) giảm từ ~210,000 cycles xuống còn **~1,759 cycles** (giảm 99.16% thời gian).
2. **Khả năng triển khai End-to-End:**
   Cả 3 kiến trúc phần cứng (1A, 2A, 3B) đã hoàn toàn đồng bộ với logic phần mềm `pynq_multi_stage_attention_hook.py` (Phase 4). Tất cả 9 khối Transformer trong MobileViT-XXS giờ đây có thể được tăng tốc trực tiếp trên chip Kria KV260.


---

## 5. Báo Cáo Tổng Hợp Phần Cứng & Sinh Bitstream (Variant 2A - AMD Kria KV260)

* **Thiết bị mục tiêu:** AMD Kria KV260 Vision AI Starter Kit (`xck26-sfvc784-2LV-c`)
* **Công cụ EDA:** AMD Vivado Design Suite v2026.1 (64-bit) Batch Mode
* **Thời gian thực thi:** 25 phút 35 giây (chạy giới hạn 2 luồng `-jobs 2` để đảm bảo nhiệt độ an toàn cho CPU laptop)
* **Kết quả Design Rule Check (DRC):** **0 Errors, 0 Critical Warnings** (Hoàn toàn sạch DRC).
* **Các file đầu ra (Hardware Artifacts):**
  - Bitstream: `attention_core_top_2A.bit` (7,797,909 bytes)
  - Hardware Handoff: `attention_core_top_2A.hwh` (337,592 bytes)
  - Hardware Platform: `attention_core_top_2A.xsa` (1,267,181 bytes)
* **Vị trí tự động đồng bộ sẵn sàng nạp lên KV260:**
  - `Project/src/Software/`
  - `Project/src/Software/03_ivit_hw_accelerated/overlay/` (Đầy đủ Overlay cho Phase 3)
  - `Project/src/Software/04_all_9_transformers_hw/overlay/` (Đầy đủ Overlay cho Phase 4 - Toàn bộ 9 Transformer Layers)
