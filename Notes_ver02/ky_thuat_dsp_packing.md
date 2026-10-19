# Kỹ Thuật DSP Packing 2x INT8 v2.0 (Xilinx UltraScale+ DSP48E2)

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/ky_thuat_dsp_packing.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/ky_thuat_dsp_packing.md)

---

## 1. NGUYÊN LÝ GHÉP 2 PHÉP NHÂN INT8 VÀO 1 DSP48E2

Chip AMD Kria KV260 (Zynq UltraScale+ `xck26`) tích hợp các lát DSP48E2 với bộ nhân $27 \times 18$-bit có dấu. Bằng cách định dạng dữ liệu đầu vào:
- **Cổng A (hoặc D-A):** Độ rộng 27 bits, nhận 2 toán hạng trọng số $B$ và $C$ (INT8 signed):
  $$\text{Operand}_{27} = B \cdot 2^{19} + C$$
- **Cổng B:** Độ rộng 18 bits, nhận 1 toán hạng kích hoạt $A$ (INT8 signed):
  $$\text{Operand}_{18} = A$$
- **Phép nhân:**
  $$P = \text{Operand}_{27} \times \text{Operand}_{18} = (A \cdot B) \cdot 2^{19} + (A \cdot C)$$

Khoảng cách 19 bits giữa $B$ và $C$ đảm bảo kết quả phép nhân 16-bit $A \cdot C$ không bao giờ tràn lấn vào vùng bit của $A \cdot B$:
- $A \cdot C \in [-16384, +16256]$ (chiếm tối đa 16 bits).
- $2^{19}$ tạo ra khoảng đệm an toàn 3 bits chống tràn số có dấu.

---

## 2. KHẮC PHỤC LỖI MỞ RỘNG DẤU (SIGN-EXTENSION BUG FIX)

Trong phiên bản cũ (`rtl/`):
```verilog
assign operand_a_extended = {{10{operand_a}}, operand_a}; // SAI LẦM: Lặp toàn bộ 8-bit vector 10 lần
```
Điều này phá hủy bit dấu của toán hạng $A$. Khi $A = -1$ (`0xFF`), vector bị mở rộng thành `0xFF_FFFF`, biến thành một số dương khổng lồ.

**Đã sửa trong `rtl_ver02`:**
```systemverilog
assign operand_a_extended = {{10{operand_a[7]}}, operand_a}; // CHÍNH XÁC: Chỉ mở rộng bit dấu bit [7]
```
Toán hạng được sign-extend chuẩn xác lên 18-bit có dấu, bảo toàn 100% độ chính xác cho mọi số âm.
