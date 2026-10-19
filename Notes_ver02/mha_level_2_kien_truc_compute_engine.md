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

## 3. ENGINE TÍCH VÔ HƯỚNG TRỰC TIẾP (HIGH-THROUGHPUT DOT-PRODUCT STREAMING)

Bên cạnh mảng Systolic Array dạng tile, Top-Level tích hợp **High-Throughput Streaming Engine** để cấp dữ liệu trực tiếp 1 điểm/chu kỳ cho Softmax:
```systemverilog
always_comb begin
    stage1_score = '0;
    for (int d = 0; d < MAX_HEAD_DIM; d++) begin
        if (d < reg_head_dim) begin
            stage1_score = stage1_score + (32'($signed(q_curr_vec[d])) * 32'($signed(k_mem[k_col_idx][d])));
        end
    end
end
```
- Phép nhân song song $d_k$ chiều giúp tính toán $Q[i, :] \cdot K[j, :]^T$ tức thời trong 1 chu kỳ.
- Vòng lặp `k_col_idx` từ $0 \dots 255$ cung cấp liên tục 256 scores cho khối Hardware Softmax mà không cần bộ đệm trung gian.
