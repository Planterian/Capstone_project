# Kiến Trúc MHA Level 2 v2.0: Compute Engine — GEMM, DSP Packing & High-Throughput Dot-Product

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mha_level_2_kien_truc_compute_engine.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mha_level_2_kien_truc_compute_engine.md)

---

## 1. NGUYÊN LÝ KỸ THUẬT DSP PACKING TRÊN XILINX ULTRASCALE+ (DSP48E2)

Khối nhân DSP48E2 trên chip Xilinx Zynq UltraScale+ (Kria KV260) hỗ trợ toán hạng $27 \times 18$-bit. Bằng cách chèn khoảng trống bit phù hợp, 1 khối DSP có thể tính đồng thời 2 phép nhân $8\text{-bit} \times 8\text{-bit}$:

```text
========================================================================================================================
                                     DSP48E2 2x INT8 PACKING PRINCIPLE (pe_slice_dsp_packing.sv)
========================================================================================================================

    Toán hạng A: Activation (INT8 signed)       [-128, +127]
    Toán hạng B: Weight Col 0 (INT8 signed)    [-128, +127]
    Toán hạng C: Weight Col 1 (INT8 signed)    [-128, +127]

    1. Ghép toán hạng B và C vào cổng D-A (27-bit):
       Operand_D_A = {B[7:0], 11'b0, C[7:0]}  (B cách C 19 bits)

    2. Cổng B nhận toán hạng A (18-bit):
       Operand_B_in = {{10{A[7]}}, A[7:0]}   (ĐÃ SỬA: Sign-extend chính xác bit A[7])

    3. Phép nhân 27-bit x 18-bit:
       P = Operand_D_A * Operand_B_in
         = (B * 2^19 + C) * A
         = (A * B) * 2^19 + (A * C)

    4. Tách kết quả (Không bị chồng lấn bit nhờ khoảng cách 19 bits):
       Result_AC = P[15:0]    = A * C (INT16)
       Result_AB = P[34:19]   = A * B (INT16)
========================================================================================================================
```

---

## 2. KIẾN TRÚC MẢNG SYSTOLIC ARRAY 2D (`systolic_mac_array.sv`)

Mảng Systolic Array vận hành theo nguyên lý **Output-Stationary (OS)**: kết quả tích lũy đứng yên tại từng PE, trong khi kích hoạt lan truyền theo hàng ngang và trọng số lan truyền theo cột dọc:

```text
========================================================================================================================
                                2D OUTPUT-STATIONARY SYSTOLIC MAC ARRAY (4x4 GRID)
========================================================================================================================

          Weight Col 0/1 [0]   Weight Col 0/1 [1]   Weight Col 0/1 [2]   Weight Col 0/1 [3]
                 |                    |                    |                    |
                 v                    v                    v                    v
  Act Row [0] -> [ PE_00 ] ---------> [ PE_01 ] ---------> [ PE_02 ] ---------> [ PE_03 ]
                 |                    |                    |                    |
                 v                    v                    v                    v
  Act Row [1] -> [ PE_10 ] ---------> [ PE_11 ] ---------> [ PE_12 ] ---------> [ PE_13 ]
                 |                    |                    |                    |
                 v                    v                    v                    v
  Act Row [2] -> [ PE_20 ] ---------> [ PE_21 ] ---------> [ PE_22 ] ---------> [ PE_23 ]
                 |                    |                    |                    |
                 v                    v                    v                    v
  Act Row [3] -> [ PE_30 ] ---------> [ PE_31 ] ---------> [ PE_32 ] ---------> [ PE_33 ]

  * SỬA ĐỔI CỐT LÕI (FL-03 & FL-04):
    1. Mỗi PE sở hữu 2 bộ tích lũy 32-bit độc lập: pe_acc_col0 (cột chẵn) và pe_acc_col1 (cột lẻ).
    2. Loại bỏ cây cộng ngang pipelined_adder_tree sai quy chuẩn toán học.
    3. Tích hợp máy trạng thái Drain tuần tự (DRAIN_RUN), quét lần lượt 32 giá trị tích lũy ra đầu ra.
========================================================================================================================
```

---

## 3. CÁC KIẾN TRÚC STREAMING COMPUTE ENGINE TRONG RTL VER02

Để khắc phục hoàn toàn nút thắt trễ đường truyền tổ hợp (critical timing path) từ phiên bản tiền nhiệm, RTL `rtl_ver02` triển khai 2 kiến trúc Streaming Dot-Product Engine đường ống hóa hoàn toàn, cùng đạt thông lượng tối đa **$II = 1$ (1 kết quả tích vô hướng / chu kỳ)**:

### 3.1 Cây Cộng Pipeline Thuần Logic Fabric (Variant 1A — `pipelined_adder_tree.sv`)
- **Tài nguyên:** 0 khối DSP48E2 (100% sử dụng FPGA Logic Fabric LUTs/FFs).
- **Cấu trúc:** 32 bộ nhân INT8 song song kết hợp cây cộng nhị phân 5 tầng thanh ghi (5-stage binary reduction tree):
  - **Tầng 0:** 32 tích số nhân $Q[d] \times K[d]$ (định dạng INT16). Kênh $d \ge d_k$ tự động bị zero-masking.
  - **Tầng 1 (reg_stage1):** 16 bộ cộng 2 nhánh $\to$ 16 tổng INT17 có thanh ghi chốt.
  - **Tầng 2 (reg_stage2):** 8 bộ cộng 2 nhánh $\to$ 8 tổng INT18 có thanh ghi chốt.
  - **Tầng 3 (reg_stage3):** 4 bộ cộng 2 nhánh $\to$ 4 tổng INT19 có thanh ghi chốt.
  - **Tầng 4 (reg_stage4):** 2 bộ cộng 2 nhánh $\to$ 2 tổng INT20 có thanh ghi chốt.
  - **Tầng 5 (reg_stage5):** 1 bộ cộng cuối $\to$ 1 tổng tích lũy INT32 hoàn chỉnh.
- **Đặc tính thời gian:** Độ trễ cố định **Latency = 6 chu kỳ xung nhịp**, thông lượng **$II = 1$**, đạt tần số vận hành $F_{\max} \ge 250\text{ MHz}$ trên Kria KV260.

### 3.2 Dòng Phát Xen Kẽ & Zero-Bubble Serializer (Variant 2A — `dsp_packed_adder_tree.sv`)
- **Tài nguyên:** Đúng 8 khối DSP48E2 (tận dụng kỹ thuật đóng gói 2x INT8 Packing ở Mục 1 để thực hiện 16 phép nhân song song).
- **Cơ chế Streaming Chẵn/Lẻ (Even/Odd Interleaved Streaming):**
  - Nhịp $2k$: Nạp và nhân 8 kênh đặc trưng chẵn ($d \in \{0, 2, 4, 6, 8, 10, 12, 14\}$).
  - Nhịp $2k+1$: Nạp và nhân 8 kênh đặc trưng lẻ ($d \in \{1, 3, 5, 7, 9, 11, 13, 15\}$).
- **Thu dồn qua Zero-Bubble Serializer:**
  - 8 tích số chẵn và 8 tích số lẻ lần lượt đi qua cây cộng gom 8 nhánh (3 tầng pipeline).
  - Bộ tuần tự hóa Zero-Bubble Serializer ghép nối chính xác tổng nửa chẵn và tổng nửa lẻ của cùng một cặp token $(Q_i, K_j)$ mà không tạo ra bất kỳ khoảng trống bong bóng (0-bubble) nào trong đường ống.
- **Đặc tính thời gian:** Độ trễ cố định **Latency = 9 chu kỳ xung nhịp**, thông lượng đầu ra **$II = 1$** (1 raw score/chu kỳ), bảo toàn 100% độ chính xác bit-exact so với Variant 1A.

