# Mảng Systolic MAC v2.0 (Output-Stationary 2D Grid)

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mang_systolic_mac.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mang_systolic_mac.md)

---

## 1. NGUYÊN LÝ VẬN HÀNH OUTPUT-STATIONARY (OS)

Trong kiến trúc **Output-Stationary (OS)**:
- Mỗi phần tử tích lũy (Accumulator) nằm cố định tại một phần tử xử lý PE $(r, c)$ tương ứng với một vị trí trong ma trận kết quả $C[r, c]$.
- Dòng kích hoạt $A$ (Activation) truyền ngang qua các hàng từ trái sang phải: $A_{r, c} \leftarrow A_{r, c-1}$.
- Dòng trọng số $W$ (Weight) truyền dọc qua các cột từ trên xuống dưới: $W_{r, c} \leftarrow W_{r-1, c}$.
- Tại mỗi chu kỳ, PE thực hiện:
  $$\text{ACC}[r, c] \leftarrow \text{ACC}[r, c] + A_{r, c} \times W_{r, c}$$

---

## 2. KHẮC PHỤC CÁC LỖI THIẾT KẾ CỐT LÕI (FL-03 & FL-04)

Trong phiên bản cũ (`Notes/mang_systolic_mac.md`), thiết kế gặp 2 lỗi nghiêm trọng:
1. **Lỗi FL-03 (Trộn cột):** 
   - Mã cũ: `pe_acc_matrix[r][c] <= pe_acc_matrix[r][c] + pe_prod_ab + pe_prod_ac;`
   - Phân tích: Vì $B$ và $C$ là 2 trọng số của 2 cột kết quả khác nhau (cột chẵn và cột lẻ), việc cộng gộp làm triệt tiêu tính độc lập của 2 cột.
   - **Đã sửa trong v2.0:** Tách thành 2 bộ tích lũy 32-bit riêng biệt:
     ```systemverilog
     pe_acc_col0[r][c] <= pe_acc_col0[r][c] + 32'($signed(pe_prod_ab)); // Cột 2c
     pe_acc_col1[r][c] <= pe_acc_col1[r][c] + 32'($signed(pe_prod_ac)); // Cột 2c + 1
     ```
2. **Lỗi FL-04 (Cây cộng Adder Tree sai quy chuẩn):**
   - Mã cũ: Đưa toàn bộ các cột qua cây cộng ngang `pipelined_adder_tree`, biến ma trận $4 \times 8$ thành một tổng vô hướng duy nhất.
   - **Đã sửa trong v2.0:** Thay thế bằng máy trạng thái quét xả tuần tự (Drain State Machine):
     ```systemverilog
     // Quét tuần tự 2 * ARRAY_SIZE * ARRAY_SIZE = 32 kết quả
     dot_product_stream <= (!drain_subcol) ? pe_acc_col0[drain_r][drain_c] : pe_acc_col1[drain_r][drain_c];
     ```

---

## 3. MÃ NGUỒN RTL SYSTEMVERILOG CHUẨN HÓA (`systolic_mac_array.sv`)

Mã nguồn đầy đủ đã kiểm tra biên dịch thành công với Icarus Verilog:
- `Project/src/Hardware/rtl_ver02/systolic_mac_array.sv`
- Testbench độc lập kiểm tra: `Project/src/Hardware/rtl_ver02/tb_systolic_mac_array.sv` (100% PASS).
