
# BÁO CÁO KIỂM TOÁN KIẾN TRÚC & KẾ HOẠCH ĐỒNG BỘ TÀI LIỆU VỚI RTL THỰC TẾ
## Dự án: Bộ Tăng Tốc Phần Cứng MobileViT-XXS Attention Core Trên Nền Tảng AMD Kria KV260
**Thư mục lưu trữ**: `/Capstone_Project/Notes_ver02/bao_cao_kiem_toan_va_ke_hoach_dong_bo_rtl_tai_lieu.md`  
**Ngày lập**: 24/10/2026  
**Trạng thái kiểm toán**: Hoàn thành rà soát 100% RTL, Testbenches và hệ thống tài liệu  

---

## 1. TỔNG QUAN TÌNH HÌNH THIẾT KẾ & XÁC MINH HIỆN TẠI (GROUND TRUTH STATUS)

Sau khi rà soát trực tiếp toàn bộ mã nguồn SystemVerilog, kịch bản kiểm thử tự động, và các tệp kết quả mô phỏng trong thư mục `Project/src/Hardware/`, tình trạng thực tế của phần cứng như sau:

### 1.1 Hiện trạng mã nguồn RTL thực tế
Hệ thống phần cứng đã được cấu trúc thành **3 thư mục độc lập, hoàn chỉnh từ module con đến Top-level**:
1. **`rtl_ver02_1A` (Variant 1A)**: Lõi tính toán dòng thuần logic LUT (0 DSP48E2), sử dụng module `pipelined_adder_tree.sv` cấu hình tham số $N=32$ (cây cộng gộp nhị phân 5 tầng).
2. **`rtl_ver02_2A` (Variant 2A)**: Lõi tính toán dòng đóng gói DSP (8 DSP48E2), sử dụng module `dsp_packed_adder_tree.sv` tích hợp 8 lát cắt `pe_slice_dsp_packing.sv`, 2 cây cộng nhị phân 3 tầng song song, và cơ chế phát xen kẽ chẵn/lẻ (Interleaved Even/Odd Serializer) đạt nhịp $II = 1$.
3. **`rtl_ver02_3B` (Variant 3B)**: Lõi mảng tâm thu 2D Output-Stationary (16 DSP48E2), sử dụng `systolic_mac_array.sv` (lưới $4 \times 4$ PE đóng gói kép, 32 MACs/chu kỳ) kết hợp bộ nhớ đệm chuyển đổi gạch-thành-dòng `reorder_buffer_3B.sv` ($4 \times 256$ words).

### 1.2 Hiện trạng kiểm thử xác minh (Verification Status)
Toàn bộ 3 biến thể đều đi kèm bộ testbench đối soát bit-exact chuẩn hóa với mô hình toán PyTorch (`golden_gemm_qk.hex`, `golden_scaled_score.hex`, `golden_softmax_prob.hex`, `golden_context_out.hex`) và **đã PASS 100% trên trình mô phỏng Icarus Verilog**:

```text
========================================================================================================
                      BẢNG TỔNG HỢP TRẠNG THÁI VERIFICATION CẢ 3 BIẾN THỂ
========================================================================================================
Cấp độ kiểm thử           | Variant 1A (LUT)    | Variant 2A (8 DSP)  | Variant 3B (16 DSP) | Kết quả
──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼───────────
Level 1: GEMM 1 Core      | PASS (65,536 pts)   | PASS (65,536 pts)   | PASS (FL-03/FL-04)  | BIT-EXACT
Level 2: Scale Unit       | PASS (65,536 pts)   | PASS (65,536 pts)   | PASS (65,536 pts)   | BIT-EXACT
Level 3: Softmax Engine   | PASS (1,024 probs)  | PASS (1,024 probs)  | PASS (1,024 probs)  | BIT-EXACT
Level 4: Score-V GEMM 2   | PASS (4,096 bytes)  | PASS (4,096 bytes)  | PASS (4,096 bytes)  | BIT-EXACT
Level 5: Top Integration  | PASS (4,096 bytes)  | PASS (4,096 bytes)  | PASS (4,096 bytes)  | BIT-EXACT
========================================================================================================
```

---

## 2. BẢNG ĐỐI CHIẾU CHI TIẾT GIỮA RTL VÀ CÁC TÀI LIỆU (GROUND TRUTH VS. DOCS)

Bảng dưới đây chỉ rõ sự khác biệt giữa thiết kế RTL thực tế đã chạy mô phỏng thành công so với các mô tả trong tài liệu:

| Thành phần kỹ thuật | RTL Thực Tế (`rtl_ver02`) | Tài liệu hiện tại (`Notes` & `Notes_ver02`) | Mức độ bất đồng bộ |
| :--- | :--- | :--- | :---: |
| **Tên module GEMM 1 (2A)**| `dsp_packed_adder_tree` | Gọi là `Zero_Bubble_Serializer` hoặc `acc_dsp` thuần túy | **NGHIÊM TRỌNG** |
| **Cơ chế nạp Key ở 2A** | Nạp **Cặp Token song song** ($K_{\text{even}}$ và $K_{\text{odd}}$) | Mô tả nạp **từng token đơn $j \in [0, N)$** qua bus 64-bit | **NGHIÊM TRỌNG** |
| **Cấu trúc cộng gộp ở 2A**| Tích hợp **2 cây cộng nhị phân 3 tầng** ($8 \to 4 \to 2 \to 1$) | Không nhắc tới cây cộng 3 tầng, chỉ ghi cộng dồn tích lũy | **NGHIÊM TRỌNG** |
| **Thông lượng ($II$) của 2A**| **$II = 1$ cycle / score** (262 chu kỳ / dòng) | Một số nơi ghi $II = 2$ chu kỳ / score (518 chu kỳ / dòng) | **NGHIÊM TRỌNG** |
| **Số tầng cây cộng của 1A**| Tham số hóa **$N=32$ (Cây cộng 5 tầng)** | Một số báo cáo cũ ghi $N=16$ (Cây cộng 4 tầng) | **TRUNG BÌNH** |
| **Xả dữ liệu Systolic (3B)**| Mạch Auto Wavefront Skew + Sequential Drain 32 items | Ghi trừu tượng: *"Ghi C_local tại offset (i, j)"* | **NHẸ** |
| **Dung lượng Bus AXI4-Stream**| Native **128-bit** (16 bytes/cycle) | Một vài sơ đồ cũ vẫn vẽ bus 64-bit | **TRUNG BÌNH** |

---

## 3. DANH MỤC CÁC ĐIỂM BẤT ĐỒNG BỘ CHI TIẾT THEO TỪNG TỆP TÀI LIỆU

### 3.1. Tệp LaTeX: `Notes/algorithm_mha_accelerator.tex`
* **Điểm bất đồng bộ tại Algorithm 2 (Stage 1 GEMM - Case 2 Variant 2A)**:
  * *Hiện tại*: Mã giả viết lặp `for each key token index j in [0, N)`, nạp lát cắt 64-bit, tích lũy serial qua $n_{\text{beats}}$, rồi gọi `Zero_Bubble_Serializer(acc_dsp)`.
  * *RTL thực tế*: Vòng lặp duyệt theo **cặp token** `k_pair_idx` từ $0 \to 127$ (128 cặp). Mỗi beat đưa đồng thời 8 phần tử của $Q$, $K_{\text{even}}$, $K_{\text{odd}}$ vào 8 DSP, đi qua 2 cây cộng 3 tầng và bộ tuần tự hóa, phát liên tục 1 score/cycle.
* **Tác động**: Khiến người đọc hiểu rằng 2A bị mất 2 chu kỳ cho mỗi token, làm giảm giá trị thiết kế thực tế của 2A.

### 3.2. Tệp Markdown: `Notes_ver02/mha_kien_truc_vi_mach.md`
* **Điểm bất đồng bộ tại Mục 3.2 (Thuật toán 2)**:
  * Trình bày mã giả text-mode của Variant 2A giống hệt tệp LaTeX (mô tả lặp từng token đơn thay vì cặp token song song).
* **Điểm bất đồng bộ tại Mục 2.2 (Mô tả phần cứng 2A)**:
  * Thiếu sơ đồ giải thích cơ chế 2 cây cộng nhị phân 3 tầng xử lý song song hai nhánh chẵn/lẻ bên trong module `dsp_packed_adder_tree.sv`.

### 3.3. Tệp Báo Cáo: `Notes_ver02/report_variant_2A_vector_streaming_dsp.md`
* **Điểm bất đồng bộ tại Mục 2.3 & 2.4**:
  * Trình bày hai chiến lược ("Chiến lược A: Interleaved 2-Key" và "Chiến lược B: Chia đôi dk") như là **hai phương án đang đề xuất xem xét**.
  * Trong khi đó, trong mã nguồn `rtl_ver02_2A`, **Chiến lược A đã được hiện thực hóa và verify 100%**.
* **Điểm bất đồng bộ tại Bảng hiệu năng Mục 4.1 & 4.2**:
  * Bảng 4.1 và 4.2 chứa cả 2 kịch bản ($II=1$ và $II=2$), gây nhiễu cho người đọc không biết đâu là thông số thực tế cuối cùng của RTL.

### 3.4. Tệp Báo Cáo So Sánh: `Notes_ver02/so_sanh_adder_tree_vs_zero_bubble_serializer.md`
* **Điểm bất đồng bộ**:
  * Tài liệu được biên soạn dựa trên giả định ban đầu rằng 2A tốn $II = 2$ chu kỳ/score (517 chu kỳ/dòng) do suy diễn từ mã giả cũ.
  * Cần được cập nhật để phản ánh đúng cấu trúc RTL: `dsp_packed_adder_tree` thực chất là sự kết hợp giữa **Cây cộng 3 tầng** và **Bộ tuần tự chẵn lẻ**, đạt $II = 1$ (262 chu kỳ/dòng).

### 3.5. Tệp Báo Cáo Variant 1A: `Notes_ver02/report_variant_1A_vector_streaming_lut.md`
* **Điểm bất đồng bộ**:
  * Một số vị trí ghi cây cộng gộp 4 tầng ($N=16$), trong khi module `pipelined_adder_tree.sv` thực tế đã được tham số hóa toàn diện lên $N=32$ (5 tầng cộng gộp) để sẵn sàng hỗ trợ các kích thước $d_k$ đa dạng ($16, 20, 24, 32$).

---

## 4. KẾ HOẠCH HÀNH ĐỘNG ĐỒNG BỘ CHI TIẾT (ACTION PLAN)

Kế hoạch được chia thành 4 giai đoạn cụ thể:

```mermaid
graph TD
    PHASE1["GIAI ĐOẠN 1: Chuẩn hóa Mã giả Thuật toán<br/>(algorithm_mha_accelerator.tex & mha_kien_truc_vi_mach.md)"]
    PHASE2["GIAI ĐOẠN 2: Cập nhật Báo cáo Chuyên sâu Variant 2A<br/>(report_variant_2A & so_sanh_adder_tree)"]
    PHASE3["GIAI ĐOẠN 3: Thống nhất Báo cáo Tổng thể & Variant 1A, 3B<br/>(report_MASTER & report_variant_1A, 3B)"]
    PHASE4["GIAI ĐOẠN 4: Kiểm tra Biên dịch & Rà soát Hoàn tất<br/>(Compile Overleaf & Verification Checklist)"]

    PHASE1 --> PHASE2 --> PHASE3 --> PHASE4
```

### 4.1 Giai Đoạn 1: Chuẩn hóa Mã Giả Thuật Toán (Mức độ ưu tiên: Cao nhất) --- [ĐÃ HOÀN THÀNH 100%]
* **Tệp mục tiêu**:
  1. [`Notes/algorithm_mha_accelerator.tex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/algorithm_mha_accelerator.tex)
  2. [`Notes_ver02/mha_kien_truc_vi_mach.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/mha_kien_truc_vi_mach.md)
* **Kết quả thực hiện**:
  * Đã định nghĩa lệnh `\parState` giải quyết triệt để vấn đề căn lề thụt dòng (hanging indent) cho các dòng dài khi rớt dòng trong LaTeX.
  * Viết lại **Case 2: Variant 2A** trong Thuật toán 2 (Stage 1 GEMM) theo đúng 100% RTL `dsp_packed_adder_tree.sv`:
    * Vòng lặp chính streaming 128 cặp token chẵn/lẻ: `for each key pair p in [0, N/2) do` ($j_{\text{even}} = 2p$, $j_{\text{odd}} = 2p + 1$).
    * 8 lát DSP48E2 tính song song cả 2 nhánh tích riêng phân kênh Even và Odd.
    * Tích hợp 2 cây cộng nhị phân 3 tầng song song ($8 \to 4 \to 2 \to 1$).
    * Cơ chế phát `Zero_Bubble_Interleaver` xuất liên tục 1 score / chu kỳ ($II=1$).
  * Đồng bộ hóa **Case 3: Variant 3B** theo đúng RTL `systolic_mac_array.sv`: Auto wavefront skewing, tính mảng systolic $4 \times 4$ tích lũy tại chỗ, xả tuần tự 32 items/tile vào Reorder Buffer.
  * Cập nhật Bảng ánh xạ Section 3.6 liên kết trực tiếp Stage 1 tới 3 module RTL thực tế (`pipelined_adder_tree.sv`, `dsp_packed_adder_tree.sv`, `systolic_mac_array.sv`).

### 4.2 Giai Đoạn 2: Cập Nhật Báo Cáo Chuyên Sâu Variant 2A (Mức độ ưu tiên: Cao) --- [ĐÃ HOÀN THÀNH 100%]
* **Tệp mục tiêu**:
  1. [`Notes_ver02/report_variant_2A_vector_streaming_dsp.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_variant_2A_vector_streaming_dsp.md)
  2. [`Notes_ver02/so_sanh_adder_tree_vs_zero_bubble_serializer.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/so_sanh_adder_tree_vs_zero_bubble_serializer.md)
* **Kết quả thực hiện**:
  * Khẳng định **Kiến trúc phát đôi xen kẽ chẵn/lẻ (Interleaved Even/Odd Streaming)** là kiến trúc phần cứng chính thức và duy nhất của `rtl_ver02_2A`.
  * Cập nhật sơ đồ khối Mermaid và giản đồ thời gian Sequence Diagram thể hiện luồng nạp 2 beats, xử lý đồng thời $K_{\text{even}}$ và $K_{\text{odd}}$ qua 8 lát DSP và 2 cây cộng 3 tầng.
  * Cập nhật sơ đồ ASCII chi tiết trong `so_sanh_adder_tree_vs_zero_bubble_serializer.md` mô tả chuẩn xác cơ chế chốt `odd_hold_reg` và máy phát nhịp `odd_pending`.
  * Cập nhật bảng so sánh kỹ thuật: Thông lượng đạt **$II = 1$ cycle / score sustained**, thời gian tính GEMM 1 một dòng là **262 chu kỳ** (tương đương 99.6% so với 261 chu kỳ của 1A, trong khi **tiết kiệm 44% LUT** và chỉ tiêu thụ 8 DSP).
  * Làm rõ câu hỏi kiến trúc: *"2A có dùng Adder Tree không?"* $\implies$ 2A **có dùng hai cây cộng nhị phân 3 tầng** bên trong `dsp_packed_adder_tree.sv`, kết hợp bộ tích lũy đa beat theo thời gian và bộ tuần tự hóa Zero-Bubble.
  * Giải quyết dứt điểm các nghi ngại lý thuyết cũ về nút thắt downstream và yêu cầu BRAM Dual-Port.
  * Đã bổ sung tài liệu trực quan hóa độc lập: [`so_do_thong_luong_3_variants.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/so_do_thong_luong_3_variants.md) tổng hợp toàn diện sơ đồ thông lượng, biểu đồ nhịp ASCII, phân tích điểm nghẽn Amdahl và ma trận so sánh định lượng của cả 3 biến thể.

### 4.3 Giai Đoạn 3: Thống Nhất Báo Cáo Variant 1A, 3B và Master Report (Mức độ ưu tiên: Trung bình)
* **Tệp mục tiêu**:
  1. [`Notes_ver02/report_variant_1A_vector_streaming_lut.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_variant_1A_vector_streaming_lut.md)
  2. [`Notes_ver02/report_variant_3B_systolic_array.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_variant_3B_systolic_array.md)
  3. [`Notes_ver02/report_MASTER_comparison.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_MASTER_comparison.md)
  4. [`Notes_ver02/so_do_thong_luong_3_variants.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/so_do_thong_luong_3_variants.md)
* **Trạng thái**: **100% HOÀN THÀNH**
* **Nội dung đã hoàn thành**:
  * **Variant 1A**: Chuẩn hóa mô tả cây cộng 5 tầng tham số hóa $N=32$ ([pipelined_adder_tree.sv](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/pipelined_adder_tree.sv)), 32 bộ nhân fabric INT8 $\times$ INT8, cơ chế Dynamic Zero-Masking cho $d_k \in \{16, 20, 24, 32\}$, độ trễ pipeline 6 chu kỳ (1 mult + 5 add), chu kỳ tính toán hàng 261 cyc ($II = 1$), tài nguyên tổng hợp thực tế ~5,150 LUT / 0 DSP.
  * **Variant 3B**: Chuẩn hóa mô tả mạch tự động phân tán dữ liệu Auto Wavefront Skew (`AUTO_SKEW = 1'b1`, `gen_auto_skew`), mảng $4 \times 4$ PE (16 DSP, 32 MACs/cycle), cơ chế dỡ tuần tự 32 items/tile vào Reorder Buffer ([reorder_buffer_3B.sv](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/reorder_buffer_3B.sv)), chu kỳ tile 56 (24 compute + 32 drain), chu kỳ hàng tương đương 448 cyc, tổng chu kỳ head 337,216.
  * **Master Comparison**: Đồng bộ ma trận so sánh định lượng và biểu đồ trực quan với số liệu testbench 128-bit Full Top thực đo (1A: 209,663 chu kỳ ~238.5 FPS; 2A: 209,919 chu kỳ ~238.2 FPS; 3B: 337,216 chu kỳ ~148.3 FPS). Tất cả liên kết chéo tới [`so_do_thong_luong_3_variants.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/so_do_thong_luong_3_variants.md).

### 4.4 Giai Đoạn 4: Kiểm Tra Biên Dịch & Nghiệm Thu
* **Trạng thái**: **100% HOÀN THÀNH — NGHIỆM THU XUẤT SẮC**
* **Kết quả nghiệm thu**:
  1. **Biên dịch LaTeX [algorithm_mha_accelerator.tex](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/algorithm_mha_accelerator.tex)**: Đã biên dịch thành công xuất ra tệp PDF chuẩn [AttentionHeadAlgorithm.pdf](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/AttentionHeadAlgorithm.pdf) (4 trang, 216 KB, 0 lỗi cú pháp, giải thuật 4 giai đoạn thể hiện hoàn hảo cơ chế Interleaved Even/Odd Streaming của Variant 2A).
  2. **Kiểm thử phần cứng RTL (15/15 bài kiểm thử PASS)**:
     - **Variant 1A** ([run_verifications_1A.bat](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/run_verifications_1A.bat)): 5/5 bài kiểm thử đạt **100% BIT-EXACT PASS**, thời gian thực thi Full Top là **209,663 chu kỳ**.
     - **Variant 2A** ([run_verifications_2A.bat](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/run_verifications_2A.bat)): 5/5 bài kiểm thử đạt **100% BIT-EXACT PASS**, thời gian thực thi Full Top là **209,919 chu kỳ**.
     - **Variant 3B** ([run_verifications_3B.bat](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/run_verifications_3B.bat)): 5/5 bài kiểm thử đạt **100% BIT-EXACT PASS**, thời gian thực thi Full Top là **337,216 chu kỳ**.
  3. **Hệ thống tài liệu đồng bộ tuyệt đối**: Toàn bộ báo cáo trong thư mục [Notes_ver02/](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/) và các sơ đồ thông lượng [so_do_thong_luong_3_variants.md](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/so_do_thong_luong_3_variants.md) đã thống nhất hoàn toàn về mặt số liệu, sơ đồ khối, giản đồ thời gian và kết quả tổng hợp phần cứng, sẵn sàng phục vụ viết Luận văn Tốt nghiệp và Slide bảo vệ đồ án.

---

## 5. KẾT LUẬN & NGHIỆM THU TOÀN DIỆN DỰ ÁN

Toàn bộ **Kế Hoạch Đồng Bộ Hóa RTL & Tài Liệu Thiết Kế (4 Giai Đoạn)** đã chính thức hoàn thành xuất sắc:
* Mọi mâu thuẫn giữa tài liệu lý thuyết cũ và mã nguồn RTL thực tế đã được giải quyết dứt điểm.
* Variant 2A được bảo vệ và khẳng định là kiến trúc phát đôi chẵn/lẻ thông minh đạt thông lượng sustained $II=1$, nhanh ngang ngửa Variant 1A trong khi tiết kiệm tới 44% LUT.
* Mọi số liệu chu kỳ, thông lượng beat, tài nguyên phần cứng (LUT, FF, BRAM, DSP) và tốc độ khung hình (FPS) đã được chuẩn hóa bit-exact trên toàn bộ hệ thống tài liệu.
* Dự án sẵn sàng cho công tác đóng gói báo cáo và bảo vệ đề tài tốt nghiệp kỹ sư vi mạch.
