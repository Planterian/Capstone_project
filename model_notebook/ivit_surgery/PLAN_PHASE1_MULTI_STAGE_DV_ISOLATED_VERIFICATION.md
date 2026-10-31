# Kế Hoạch Chi Tiết Giai Đoạn 1: Xác Minh Chức Năng (DV) Độc Lập Từng Stage & Đa Tầng Liên Hoàn

> **Dự án:** Bộ Tăng Tốc Phần Cứng Attention Core Đa Chiều ($d_k \in \{16, 20, 24\}$) cho MobileViT-XXS trên AMD Kria KV260  
> **Tài liệu lưu trữ:** [`model_notebook/ivit_surgery/PLAN_PHASE1_MULTI_STAGE_DV_ISOLATED_VERIFICATION.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE1_MULTI_STAGE_DV_ISOLATED_VERIFICATION.md)  
> **Phiên bản:** v1.0 — Chuẩn hóa quy trình Senior DV (Divide-and-Conquer)  
> **Mục tiêu:** Đảm bảo $100\%$ Bit-Exact (0-LSB Mismatch) trên từng Stage độc lập trước khi chạy liên hoàn, phục vụ trích xuất số liệu chuẩn xác cho luận văn tốt nghiệp và tiến đến tổng hợp Vivado.

---

## 1. TÔN CHỈ THIẾT KẾ & PHƯƠNG PHÁP LUẬN (METHODOLOGY)

### 1.1. Chiến Lược "Chia Để Trị" (Divide-and-Conquer)
Việc chuyển từ một cấu hình cố định ($d_k = 16$) sang kiến trúc đa tầng ($d_k \in \{16, 20, 24\}$) làm tăng độ phức tạp của bộ đếm nhịp (beat counter), căn chỉnh bus 128-bit (padding byte alignment), và cấu hình Dyadic Rescaler. 

Nếu chạy ngay testbench liên hoàn 3 stages khi chưa kiểm thử đơn lẻ, việc gỡ lỗi (debug) khi xuất hiện sai lệch LSB hoặc lệch nhịp `m_axis_tlast` sẽ vô cùng khó khăn. Do đó, kế hoạch thực thi tuân thủ nghiêm ngặt **3 bước kiểm thử phân tầng**:

```
+---------------------------------------------------------------------------------------------------+
| BƯỚC 1: TRÍCH XUẤT GOLDEN VECTORS PYTORCH I-ViT (3 Stages độc lập: Stage 2, Stage 3, Stage 4)    |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| BƯỚC 2: THỰC THI KIỂM THỬ ĐỘC LẬP TỪNG STAGE (Dedicated Isolated Testbenches)                   |
|   ├── Stage 2 ($d_k = 16$): 256 tokens, 1 beat/token, Ingress 768 beats, Egress 256 beats       |
|   ├── Stage 3 ($d_k = 20$): 256 tokens, 2 beats/token, Ingress 1,536 beats, Egress 512 beats     |
|   └── Stage 4 ($d_k = 24$): 256 tokens, 2 beats/token, Ingress 1,536 beats, Egress 512 beats     |
|   ==> Đánh giá chi tiết: Latency (Cycles), Throughput, Khớp 100% 0-LSB, Lập bảng so sánh         |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| BƯỚC 3: KIỂM THỬ TÍCH HỢP ĐA TẦNG LIÊN HOÀN (Unified Sequential Multi-Stage Testbench)            |
|   ├── Cấu hình động CSR qua AXI-Lite trong thời gian thực (Runtime Dynamic Reconfiguration)       |
|   └── Chuyển tầng liên tục không cần nạp lại bitstream: Stage 2 -> Stage 3 -> Stage 4            |
+---------------------------------------------------------------------------------------------------+
```

### 1.2. Tiêu Chuẩn Nghiệm Thu Khắt Khe (Pass/Fail Acceptance Criteria)
1. **Toán học Bit-Exact:** Mọi byte đầu ra Context Output tại từng Stage so với Python Golden Model phải đạt **Error = 0 LSB**.
2. **Giao thức AXI4-Stream:**
   - Tín hiệu `m_axis_tlast` phải khẳng định chính xác ở beat cuối cùng của từng Stage.
   - Cờ `m_axis_tkeep` phản ánh đúng số byte có hiệu lực của beat cuối cùng theo chiều $d_k$:
     - $d_k = 16$: `16'hFFFF` (16 bytes hợp lệ).
     - $d_k = 20$: `16'h000F` (4 bytes hợp lệ trên beat 1, 12 bytes zero padding).
     - $d_k = 24$: `16'h00FF` (8 bytes hợp lệ trên beat 1, 8 bytes zero padding).
3. **Phần cứng không Treo (Zero Hang/Deadlock):** Cờ trạng thái `done` phải khẳng định đúng lúc, tự xóa khi có lệnh khởi động mới, pipeline flush sạch sẽ giữa các lần chuyển Stage.

---

## 2. MA TRẬN THÔNG SỐ CẤU HÌNH 3 TẦNG TRANSFORMER (MOBILEVIT-XXS)

| Đặc tính | STAGE 2 (Layer 2) | STAGE 3 (Layer 3) | STAGE 4 (Layer 4) | Ghi chú phần cứng |
| :--- | :---: | :---: | :---: | :--- |
| **Chiều vector ($d_k$)** | **16** | **20** | **24** | Cấu hình qua CSR `0x14` |
| **Độ dài chuỗi ($N$)** | 256 | 256 | 256 | Cấu hình qua CSR `0x10` |
| **Số Head ($H$)** | 4 | 4 | 4 | Single-head DV (Head 0) |
| **Hệ số tỷ lệ ($S_Q$)** | 0.05797501 | 0.03653108 | 0.03324918 | Trích xuất từ I-ViT QAT |
| **Hệ số tỷ lệ ($S_{score}$)** | 0.12751131 | 0.05705866 | 0.05031879 | Trích xuất từ I-ViT QAT |
| **Hệ số tỷ lệ ($S_{eff}$)** | 0.00658981 | 0.00522984 | 0.00448463 | $S_Q^2 / (S_{score} \cdot \sqrt{d_k})$ |
| **Hệ số Dyadic ($M$)** | **27,640** | **21,936** | **18,810** | Cấu hình qua CSR `0x20` |
| **Số dịch bit ($E$)** | **22** | **22** | **22** | Cấu hình qua CSR `0x24` |
| **Bảng Exp LUT Slot** | Slot 0 (`0x000..0x0FF`) | Slot 1 (`0x100..0x1FF`) | Slot 2 (`0x200..0x2FF`) | Exp ROM 1024 words |
| **Số beat Ingress / ma trận** | 256 beats | 512 beats | 512 beats | 1 beat/token ($d_k \le 16$); 2 beats/token ($d_k > 16$) |
| **Tổng beat Ingress ($Q, K, V$)** | **768 beats** | **1,536 beats** | **1,536 beats** | Nạp qua `s_axis_tdata[127:0]` |
| **Số beat Egress Context** | **256 beats** | **512 beats** | **512 beats** | Xuất qua `m_axis_tdata[127:0]` |
| **Tổng số byte xác minh** | **4,096 Bytes** | **5,120 Bytes** | **6,144 Bytes** | So sánh 100% với Golden Model |

---

## 3. LỘ TRÌNH THỰC HIỆN CHI TIẾT (STEP-BY-STEP ACTION PLAN)

### 📌 BƯỚC 1.1: Mở Rộng Python Golden Generator (`export_golden_vectors.py`)
- **Tệp nguồn:** [`model_notebook/ivit_surgery/export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py)
- **Nhiệm vụ:**
  1. Thêm forward hook cho cả 3 tầng Transformer trong MobileViT-XXS:
     - `STAGE2`: `model.mobilevit.encoder.layer[2].transformer.layer[0].attn`
     - `STAGE3`: `model.mobilevit.encoder.layer[3].transformer.layer[0].attn`
     - `STAGE4`: `model.mobilevit.encoder.layer[4].transformer.layer[0].attn`
  2. Xuất bộ vector riêng biệt có gắn hậu tố định danh rõ ràng:
     - Stage 2: `q_matrix_st2.hex`, `k_matrix_st2.hex`, `v_matrix_st2.hex`, `golden_context_out_st2.hex` (vẫn duy trì bản không hậu tố để tương thích testbench cũ).
     - Stage 3: `q_matrix_st3.hex`, `k_matrix_st3.hex`, `v_matrix_st3.hex`, `golden_context_out_st3.hex`, `golden_scaled_score_st3.hex`, `golden_softmax_prob_st3.hex`.
     - Stage 4: `q_matrix_st4.hex`, `k_matrix_st4.hex`, `v_matrix_st4.hex`, `golden_context_out_st4.hex`, `golden_scaled_score_st4.hex`, `golden_softmax_prob_st4.hex`.
  3. Cập nhật và xác thực lại bảng `exp_rom_all_stages.hex` (1,024 từ $\times$ 20-bit).
  4. Tự động đồng bộ các tệp `.hex` vừa sinh sang cả 3 thư mục:
     - `Project/src/Hardware/rtl_ver02_1A/`
     - `Project/src/Hardware/rtl_ver02_2A/`
     - `Project/src/Hardware/rtl_ver02_3B/`

---

### 📌 BƯỚC 1.2: Triển Khai Bộ Testbench Độc Lập Cho Từng Stage (Isolated DV)
Để phục vụ việc debug trực quan và lập bảng số liệu cho từng tầng riêng biệt, tạo các testbench chuyên biệt cho từng Stage:

#### 1. Testbench Độc Lập Cho STAGE 2 ($d_k = 16$)
- **Tệp kiểm thử:** `tb_stage2_isolated.sv` (hoặc tái sử dụng `tb_attention_core_*.sv` hiện có).
- **Đặc trưng kiểm tra:**
  - $d_k = 16 \implies 1\text{ beat/token}$.
  - Nạp: 768 beats Ingress liên tục.
  - Kiểm tra: 256 beats Egress (4,096 bytes).
  - Khẳng định `m_axis_tlast` tại beat thứ 255.

#### 2. Testbench Độc Lập Cho STAGE 3 ($d_k = 20$)
- **Tệp kiểm thử:** `tb_stage3_isolated.sv` (Xây dựng cho 1A, 2A, 3B).
- **Đặc trưng kiểm tra:**
  - $d_k = 20 \implies 2\text{ beats/token}$.
  - Beat 0: 16 bytes đầu của token $i$.
  - Beat 1: 4 bytes cuối của token $i$ + 12 zero padding bytes (`128'h0000_0000_0000_0000_0000_0000_XXXX_XXXX`).
  - Nạp: $256 \times 2 = 512$ beats mỗi ma trận $\implies$ Tổng Ingress = 1,536 beats.
  - Cấu hình CSR: `HEAD_DIM = 20`, `SCALE_M = 21936`, `SCALE_E = 22`.
  - Kiểm tra Egress: 512 beats Egress. Mỗi token nhận 2 beats:
    - Beat 0: 16 bytes ngữ cảnh (`tkeep = 16'hFFFF`).
    - Beat 1: 4 bytes ngữ cảnh cao (`tkeep = 16'h000F`).
  - Tổng số byte xác minh: $256 \times 20 = 5,120$ bytes đạt **0-LSB Error**.
  - Khẳng định `m_axis_tlast` tại beat thứ 511.

#### 3. Testbench Độc Lập Cho STAGE 4 ($d_k = 24$)
- **Tệp kiểm thử:** `tb_stage4_isolated.sv` (Xây dựng cho 1A, 2A, 3B).
- **Đặc trưng kiểm tra:**
  - $d_k = 24 \implies 2\text{ beats/token}$.
  - Beat 0: 16 bytes đầu của token $i$.
  - Beat 1: 8 bytes cuối của token $i$ + 8 zero padding bytes.
  - Nạp: $256 \times 2 = 512$ beats mỗi ma trận $\implies$ Tổng Ingress = 1,536 beats.
  - Cấu hình CSR: `HEAD_DIM = 24`, `SCALE_M = 18810`, `SCALE_E = 22`.
  - Kiểm tra Egress: 512 beats Egress. Mỗi token nhận 2 beats:
    - Beat 0: 16 bytes ngữ cảnh (`tkeep = 16'hFFFF`).
    - Beat 1: 8 bytes ngữ cảnh cao (`tkeep = 16'h00FF`).
  - Tổng số byte xác minh: $256 \times 24 = 6,144$ bytes đạt **0-LSB Error**.
  - Khẳng định `m_axis_tlast` tại beat thứ 511.

---

### 📌 BƯỚC 1.3: Thu Thập Số Liệu & Lập Bảng Đánh Giá Từng Stage (Isolated Metrics)
Sau khi hoàn tất kiểm thử độc lập, trích xuất bộ dữ liệu đo đạc chu kỳ xung nhịp và độ chính xác để điền vào bảng báo cáo:

| Biến thể kiến trúc | Stage | Chiều $d_k$ | Ingress Latency (cyc) | Compute Latency (cyc) | Egress Latency (cyc) | Tổng Chu Kỳ (cyc) | Tỷ lệ Match Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant 1A** (Pure Fabric) | Stage 2 | 16 | 768 | 208,639 | 256 | **209,663** | **100% (4,096/4,096 B - 0 LSB)** |
| | Stage 3 | 20 | 1,536 | 207,871 | 512 | **209,919** | **100% (5,120/5,120 B - 0 LSB)** |
| | Stage 4 | 24 | 1,536 | 207,871 | 512 | **209,919** | **100% (6,144/6,144 B - 0 LSB)** |
| **Variant 2A** (DSP Packing) | Stage 2 | 16 | 768 | 208,895 | 256 | **209,919** | **100% (4,096/4,096 B - 0 LSB)** |
| | Stage 3 | 20 | 1,536 | 240,895 | 512 | **242,943** | **100% (5,120/5,120 B - 0 LSB)** |
| | Stage 4 | 24 | 1,536 | 240,895 | 512 | **242,943** | **100% (6,144/6,144 B - 0 LSB)** |
| **Variant 3B** (Systolic 2D) | Stage 2 | 16 | 768 | 336,192 | 256 | **337,216** | **100% (4,096/4,096 B - 0 LSB)** |
| | Stage 3 | 20 | 1,536 | 343,616 | 512 | **345,664** | **100% (5,120/5,120 B - 0 LSB)** |
| | Stage 4 | 24 | 1,536 | 351,808 | 512 | **353,856** | **100% (6,144/6,144 B - 0 LSB)** |

---

### 📌 BƯỚC 1.4: Tích Hợp Testbench Đa Tầng Liên Hoàn (`tb_multi_stage_seq_*.sv`) — ĐÃ HOÀN TẤT 100% BIT-EXACT

Toàn bộ 3 biến thể kiến trúc phần cứng đã được xác minh thành công trên kịch bản chuỗi liên tục thực tế của mô hình MobileViT (Stage 2 $\to$ CSR Reconfig $\to$ Stage 3 $\to$ CSR Reconfig $\to$ Stage 4) **mà KHÔNG kích hoạt lại tín hiệu Hardware Reset**:

#### Kết Quả Xác Minh Liên Hoàn Thực Nghiệm (Sequential Execution Matrix):

| Biến Thể Kiến Trúc | Stage 2 ($d_k=16$) | Stage 3 ($d_k=20$) | Stage 4 ($d_k=24$) | Tổng Chu Kỳ (Cycles) | Thời Gian @ 200MHz | Kết Quả Bit-Exact (0-LSB Error) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Biến thể 1A** (Pure Fabric LUT Tree) | 209,658 cyc | 209,914 cyc | 209,914 cyc | **629,486 cycles** | **~3.15 ms** | **PASS 100% (15,360 / 15,360 B)** |
| **Biến thể 2A** (8 DSP Packing Adder Tree) | 209,914 cyc | 242,938 cyc | 242,938 cyc | **695,790 cycles** | **~3.48 ms** | **PASS 100% (15,360 / 15,360 B)** |
| **Biến thể 3B** (16 DSP 2D Systolic MAC Array) | 337,211 cyc | 345,659 cyc | 353,851 cyc | **1,036,721 cycles** | **~5.18 ms** | **PASS 100% (15,360 / 15,360 B)** |

#### Các Cơ Chế Phần Cứng Được Xác Minh Tuyệt Đối:
1. **Dynamic CSR Reconfiguration**: Phần cứng chuyển đổi mượt mà cấu hình chiều đặc trưng $d_k$ (16, 20, 24), hệ số tỷ lệ dyadic multiplier $M$ (27640, 21936, 18810) và exponent $e=22$.
2. **Buffer Flush & Ingress Resync**: Giao thức Clear Pulse (`CSR 0x00 <= 2` rồi `0x00 <= 0`) giải phóng hoàn toàn và đồng bộ lại FSM nạp BRAM đệm, loại bỏ 100% nguy cơ ô nhiễm dữ liệu giữa các Stage.
3. **Variable AXI4-Stream Beat & TKEEP Masking**: Giao tiếp AXI4-Stream Egress 128-bit tự động áp dụng mặt nạ byte `16'hFFFF`, `16'h000F` (Stage 3 beat lẻ) và `16'h00FF` (Stage 4 beat lẻ) chuẩn xác đến từng bit.
4. **Tổng số byte xác minh Bit-Exact không sai số**:
   - Bộ Isolated DV Suite (9 bài test): 46,080 bytes.
   - Bộ Sequential DV Suite (3 bài test liên hoàn): 46,080 bytes.
   - **TỔNG CỘNG ĐÃ VERIFY**: **92,160 bytes (0-LSB Error)**.

---

## 4. TỔ CHỨC CÁC TỆP TIN & KỊCH BẢN THỰC THI (FILES & AUTOMATION)

### 4.1. Danh Sách Tệp Đã Tạo & Xác Minh

```
Capstone_Project/
├── model_notebook/ivit_surgery/
│   ├── PLAN_PHASE1_MULTI_STAGE_DV_ISOLATED_VERIFICATION.md  <-- [Tài liệu theo dõi tiến độ]
│   ├── export_golden_vectors.py                             <-- [Xuất vector Stage 2, 3, 4]
│   └── golden_vectors/                                      <-- [Chứa .hex Stage 2, 3, 4]
│
├── Project/src/Hardware/
│   ├── rtl_ver02_1A/ (tương tự cho 2A và 3B)
│   │   ├── q_matrix_st[2,3,4].hex, k_matrix_st[2,3,4].hex, v_matrix_st[2,3,4].hex
│   │   ├── golden_context_out_st[2,3,4].hex
│   │   ├── tb_stage2_isolated_1A.sv, tb_stage3_isolated_1A.sv, tb_stage4_isolated_1A.sv
│   │   ├── tb_multi_stage_seq_1A.sv
│   │   ├── run_stage_isolated_dv_1A.bat
│   │   └── run_multi_stage_seq_1A.bat
│   │
│   └── synthesis/                                           <-- [Thư mục chuẩn bị cho Phase 2]
│       ├── timing_kv260.xdc                                 <-- [Ràng buộc xung 200 MHz cho KV260]
│       ├── synth_variant_1A.tcl                             <-- [Script tổng hợp OOC cho 1A]
│       ├── synth_variant_2A.tcl                             <-- [Script tổng hợp OOC cho 2A]
│       ├── synth_variant_3B.tcl                             <-- [Script tổng hợp OOC cho 3B]
│       └── run_synth_kv260.bat                              <-- [Runner tự động hóa Vivado 2026.1]
```

---

## 5. KẾT THÚC GIAI ĐOẠN 1 & TIẾN TRÌNH GIAI ĐOẠN 2 (VIVADO SYNTHESIS & PnR)

- [x] **Bước 1.1:** Xuất Golden Vectors đa tầng hoàn tất và đồng bộ sang tất cả các thư mục RTL.
- [x] **Bước 1.2:** Chạy và pass 100% 9/9 bài test Isolated DV đơn lẻ trên cả 3 biến thể.
- [x] **Bước 1.3:** Ghi nhận và phân tích chi tiết chu kỳ thực thi và hiệu năng vi kiến trúc.
- [x] **Bước 1.4:** Chạy và pass 100% 3/3 bài test Sequential Multi-Stage liên hoàn không reset phần cứng.
- [ ] **Giai đoạn 2 (Vivado OOC Synthesis & PnR trên AMD Kria KV260):**
  - [x] **Variant 1A (Pure Fabric Tree):**
    - Hoàn tất Synthesis & Implementation (Route status: Fully Routed, 0 Errors, 0 Violations).
    - Tài nguyên: **12,641 LUTs** (10.79%), **8,168 FFs** (3.49%), **3,552 LUTRAMs**, **1 BRAM36** (0.69%), **36 DSPs** (2.88%, 0 DSP ở Stage 1).
    - Timing: Target 200 MHz, WNS = -0.803 ns, WHS = +0.007 ns (Hold MET), **Fmax = 172.33 MHz**.
    - Công suất: **0.466 W** (Dynamic: 0.177 W, Static: 0.289 W).
    - Báo cáo chi tiết: [`reports_1A/post_route_utilization.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_1A/post_route_utilization.rpt), [`post_route_timing.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_1A/post_route_timing.rpt), [`post_route_power.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_1A/post_route_power.rpt).
  - [x] **Variant 2A (8 DSP Packing SIMD Tree):**
    - Hoàn tất Synthesis & Implementation (Route status: Fully Routed, 0 Errors, 0 Violations).
    - Tài nguyên: **10,579 LUTs** (9.03%), **7,337 FFs** (3.13%), **4,736 LUTRAMs**, **1 BRAM36** (0.69%), **52 DSPs** (4.17%, 16 DSP ở Stage 1).
    - Tối ưu tài nguyên: **Giảm 3,250 Logic LUTs (tiết kiệm 35.8%)** so với 1A!
    - Timing: Target 200 MHz, WNS = -1.095 ns, WHS = +0.009 ns (Hold MET), **Fmax = 164.07 MHz** (Vận hành an toàn 160 MHz).
    - Công suất: **0.438 W** (Dynamic: **0.149 W - tiết kiệm 15.8% công suất động** so với 1A!).
    - Báo cáo chi tiết: [`reports_2A/post_route_utilization.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_2A/post_route_utilization.rpt), [`post_route_timing.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_2A/post_route_timing.rpt), [`post_route_power.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_2A/post_route_power.rpt).
  - [x] **Variant 3B (16 DSP 2D Systolic Array):**
    - Hoàn tất Synthesis & Implementation (Route status: Fully Routed, 0 Errors, 0 Violations).
    - Tài nguyên: **39,135 LUTs** (33.42%), **42,320 FFs** (18.07%), **15,392 LUTRAMs**, **1 BRAM36** (0.69%), **52 DSPs** (4.17%, 16 DSP ở Stage 1).
    - Phân bổ: 16 DSPs trong Mảng Systolic 2D $4 \times 4$, Bộ đệm Reorder Buffer 32Kb ($4 \times 256$ words INT32).
    - Timing: Target 200 MHz, WNS = -0.924 ns, WHS = +0.011 ns (Hold MET), **Fmax = 168.80 MHz** (Vận hành an toàn 165 MHz).
    - Công suất: **0.781 W** (Dynamic: 0.490 W, Static: 0.291 W).
    - Báo cáo chi tiết: [`reports_3B/post_route_utilization.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_3B/post_route_utilization.rpt), [`post_route_timing.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_3B/post_route_timing.rpt), [`post_route_power.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_3B/post_route_power.rpt).
  - [x] **Tổng hợp bảng đối sánh MASTER toàn diện 3 biến thể:** Đã hoàn tất và lưu tại Mục 9.7 trong [`report_MASTER_comparison.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_MASTER_comparison.md).

---

## 6. NHẬT KÝ CẬP NHẬT THIẾT KẾ: TỐI ƯU TIMING SOFTMAX & HOÀN TẤT TÁI THẨM ĐỊNH BIT-EXACT (2026-10-27)

### 6.1. Mục Tiêu & Sửa Đổi Vi Kiến Trúc
- **Vấn đề phát hiện:** Toàn bộ vi phạm Setup Slack tại 200 MHz ở giai đoạn PnR ($WNS \approx -0.8\text{ ns}$ đến $-1.1\text{ ns}$) xuất phát từ đường truyền tổ hợp dài $5.8\text{ ns}$ trong `softmax_lut.sv` (BRAM Read $\to$ DSP48E2 Mult $\to$ Shift/Clamp $\to$ Output Reg).
- **Hành động kỹ thuật:**
  - Bổ sung tầng thanh ghi pipeline `mult_full_reg [51:0]` và cờ điều khiển `pipe_valid_stage2`.
  - Phân tách đường trễ thành 2 chu kỳ: Cycle 1 ($\sim 3.4\text{ ns}$) và Cycle 2 ($\sim 1.4\text{ ns}$), cho phép Vivado suy luận thanh ghi nội bộ `MREG / PREG` trong khối DSP48E2.
  - Cập nhật đồng bộ trên cả 3 biến thể: [`rtl_ver02_1A`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/softmax_lut.sv), [`rtl_ver02_2A`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/softmax_lut.sv), [`rtl_ver02_3B`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/softmax_lut.sv).

### 6.2. Kết Quả Tái Thẩm Định Bit-Exact (Verification Pass Report)
- [x] **Kiểm thử độc lập Softmax (`tb_golden_softmax_lut.sv`):**
  - Variant 1A: 1,024 / 1,024 tokens (100% Bit-Exact Match, 0 LSB Error).
  - Variant 2A: 1,024 / 1,024 tokens (100% Bit-Exact Match, 0 LSB Error).
  - Variant 3B: 1,024 / 1,024 tokens (100% Bit-Exact Match, 0 LSB Error).
- [x] **Kiểm thử Isolated Multi-Stage Suite (`run_stage_isolated_dv_*.bat`):**
  - Variant 1A: 15,360 / 15,360 bytes (Stage 2: 4,096 B, Stage 3: 5,120 B, Stage 4: 6,144 B) $\to$ **100% PASS**.
  - Variant 2A: 15,360 / 15,360 bytes (Stage 2: 4,096 B, Stage 3: 5,120 B, Stage 4: 6,144 B) $\to$ **100% PASS**.
  - Variant 3B: 15,360 / 15,360 bytes (Stage 2: 4,096 B, Stage 3: 5,120 B, Stage 4: 6,144 B) $\to$ **100% PASS**.
- [x] **Kiểm thử Sequential Multi-Stage Suite không reset (`run_multi_stage_seq_*.bat`):**
  - Variant 1A: 630,254 cycles, 15,360 / 15,360 bytes $\to$ **100% PASS**.
  - Variant 2A: 696,558 cycles, 15,360 / 15,360 bytes $\to$ **100% PASS**.
  - Variant 3B: 1,037,489 cycles, 15,360 / 15,360 bytes $\to$ **100% PASS**.
- [x] **Cập nhật tài liệu thiết kế:**
  - [`Notes_ver02/report_MASTER_comparison.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_MASTER_comparison.md) (Mục 10).
  - [`Notes_ver02/mach_softmax_phan_cung.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/mach_softmax_phan_cung.md) (Mục 4).

### 6.3. Tái Chạy Vivado Synthesis & PnR Cho Variant 1A Sau Khi Pipeline (2026-10-27)
- **Trạng thái PnR:** Hoàn tất thành công (Fully Routed, 0 lỗi, 0 DRC violations, 0 critical warnings).
- **Tài nguyên sử dụng:**
  - CLB LUTs: **12,725** (10.86%) (Logic: 9,173; LUTRAM: 3,552).
  - CLB FFs: **8,183** (3.49%).
  - BRAM (RAMB36E2): **1** (0.69%).
  - DSP48E2: **36** (2.88%, 0 DSP ở Stage 1).
- **Cải thiện Timing ngoạn mục tại 200.0 MHz ($T_{clk} = 5.0\text{ ns}$):**
  - **WNS:** Giảm từ **-0.803 ns** xuống chỉ còn **-0.095 ns** (triệt tiêu 88.2% độ vi phạm).
  - **WHS:** **+0.007 ns** (MET 100%).
  - **Fmax:** Đạt **196.27 MHz** (tăng thêm +23.94 MHz, vận hành an toàn ở **195.0 MHz**).
- **Công suất:** **0.463 W** (Dynamic: 0.174 W, Static: 0.289 W).
- Báo cáo cập nhật: [`reports_1A/post_route_timing.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_1A/post_route_timing.rpt), [`reports_1A/post_route_utilization.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_1A/post_route_utilization.rpt), [`reports_1A/post_route_power.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_1A/post_route_power.rpt).

### 6.4. Tái Chạy Vivado Synthesis & PnR Cho Variant 2A Sau Khi Pipeline (2026-10-27)
- **Trạng thái PnR:** Hoàn tất thành công xuất sắc (Fully Routed, 0 lỗi, 0 DRC violations, 0 critical warnings).
- **Timing Closure 100% tại 200.0 MHz ($T_{clk} = 5.0\text{ ns}$):**
  - **WNS:** **+0.028 ns (MET 100%, DƯƠNG!)** — Đóng timing hoàn toàn!
  - **TNS:** **0.000 ns** (0 failing endpoint trên toàn bộ 51,544 endpoints).
  - **WHS:** **+0.007 ns (MET 100%)**.
  - **Fmax:** Đạt **201.13 MHz** (vận hành an toàn tuyệt đối ở 200.0 MHz!).
- **Tài nguyên sử dụng:**
  - CLB LUTs: **10,553** (9.01%) (Logic: **5,813** [4.96%], LUTRAM: **4,736** [8.22%]).
  - CLB FFs: **7,340** (3.13%).
  - BRAM (RAMB36E2): **1** (0.69%).
  - DSP48E2: **52** (4.17%, 16 DSP ở Stage 1).
- **Công suất tiêu thụ:** **0.438 W** (Dynamic: **0.149 W - tiết kiệm 14.4% công suất động so với 1A!**, Static: **0.289 W**).
- Báo cáo cập nhật: [`reports_2A/post_route_timing.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_2A/post_route_timing.rpt), [`reports_2A/post_route_utilization.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_2A/post_route_utilization.rpt), [`reports_2A/post_route_power.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_2A/post_route_power.rpt).

### 6.5. Tái Chạy Vivado Synthesis & PnR Cho Variant 3B Sau Khi Pipeline (2026-10-27)
- **Trạng thái PnR:** Hoàn tất thành công (Fully Routed, 0 lỗi, 0 DRC violations, 0 critical warnings).
- **Tài nguyên sử dụng:**
  - CLB LUTs: **39,123** (33.40%) (Logic: 23,707; LUTRAM: 15,392).
  - CLB FFs: **42,308** (18.06%).
  - BRAM (RAMB36E2): **1** (0.69%).
  - DSP48E2: **52** (4.17%, 16 DSP ở mảng 2D Systolic $4 \times 4$).
- **Cải thiện Timing ngoạn mục tại 200.0 MHz ($T_{clk} = 5.0\text{ ns}$):**
  - **WNS:** Giảm từ **-0.924 ns** xuống **-0.431 ns** (cắt giảm hơn 53% mức vi phạm, giải tỏa hoàn toàn điểm nghẽn bộ nhân DSP).
  - **WHS:** **+0.016 ns** (MET 100%).
  - **Fmax:** Đạt **184.13 MHz** (tăng thêm +15.33 MHz, vận hành an toàn ở **180.0 MHz**).
- **Công suất:** **0.826 W** (Dynamic: 0.535 W, Static: 0.291 W).
- Báo cáo cập nhật: [`reports_3B/post_route_timing.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_3B/post_route_timing.rpt), [`reports_3B/post_route_utilization.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_3B/post_route_utilization.rpt), [`reports_3B/post_route_power.rpt`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis/reports_3B/post_route_power.rpt).





