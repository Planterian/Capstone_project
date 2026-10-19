# Mạch Softmax Phần Cứng v2.0 (I-ViT Shiftmax Aligned)

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mach_softmax_phan_cung.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mach_softmax_phan_cung.md)

---

## 1. NGUYÊN LÝ TOÁN HỌC & ĐỘ PHÂN GIẢI BIT-EXACT

Thuật toán Softmax số nguyên trong **I-ViT (Shiftmax)** được xây dựng dựa trên phương pháp xấp xỉ hàm mũ cơ số 2 và chuẩn hóa số dư:
1. **Trừ giá trị lớn nhất (Max-Subtraction):**
   $$\Delta_i = S_{\text{max}} - S_i \ge 0, \quad \text{với } S_{\text{max}} = \max_{j} (S_j)$$
2. **Tra cứu hàm mũ rời rạc (Exp Lookup Table):**
   Thay vì tính $e^{-\Delta_i}$, I-ViT sử dụng bảng tra `int_exp_shift`:
   $$\text{Exp\_val}_i = \text{LUT}[\Delta_i]$$
   Các giá trị này được tiền tính toán và lưu trong Block RAM nội chip.
3. **Nghịch đảo tổng mẫu số (Reciprocal Divider):**
   $$\text{Sum\_Exp} = \sum_{i=0}^{N-1} \text{Exp\_val}_i$$
   Hệ số chuẩn hóa được tính bằng phép chia nguyên:
   $$\text{inv\_sum} = \left\lfloor \frac{2^{31} - 1}{\text{Sum\_Exp}} \right\rfloor$$
4. **Chuẩn hóa xác suất 16-bit (UINT16 Probability):**
   $$\text{out\_prob}_i = (\text{Exp\_val}_i \times \text{inv\_sum}) \ggg 16$$
   Kết quả nằm trong dải $[0, 32767]$, với tổng $\sum_{i=0}^{255} \text{out\_prob}_i \approx 32,768 = 2^{15}$, **khớp chính xác 100% từng bit với PyTorch**.

---

## 2. KIẾN TRÚC VI MẠCH PHẦN CỨNG (`softmax_lut.sv`)

```text
========================================================================================================================
                                     HARDWARE SOFTMAX ENGINE PIPELINE (softmax_lut.sv)
========================================================================================================================

    in_score [15:0] (INT16, 256 tokens per row)
          │
          +-------------------------------------------------------------+
          |                                                             |
          v                                                             v
    +-------------------------------------------+             +-----------------------------------+
    | Pass 1: Max-Finder Comparator & Buffer    |             | 1024-Word True Dual-Port BRAM     |
    | - score_buffer [0..255]                   |             | (* ram_style = "block" *)         |
    | - S_max = max(Score_0 .. Score_255)       |             | Port A: rom_addr_stage (Doc LUT)  |
    +---------------------+---------------------+             | Port B: cfg_addr, cfg_wdata (CSR) |
                          │                                   +-----------------+-----------------+
                          v  Delta = S_max - Score_i                            │
    +---------------------------------------------------------------------------+
    | Pass 2: Exp ROM Lookup & Accumulator                                      │
    | rom_addr = Delta[7:0]                                                     │
    | rom_data_exp <= exp_rom [Stage_offset + rom_addr] <-----------------------+
    | sum_exp_reg <= sum_exp_reg + rom_data_exp (ACC_WIDTH = 32)
    +---------------------+
                          │
                          v  sum_exp_reg [31:0]
    +---------------------------------------------------------------------------+
    | Pass 3: 32-step Restoring Integer Divider                                 |
    | - Numerator: 32'h7FFF_FFFF (2^31 - 1)                                     |
    | - Denominator: sum_exp_reg                                                |
    | - Quotient: inv_sum_reg = (2^31 - 1) / sum_exp_reg (32 clock cycles)      |
    +---------------------+
                          │
                          v  inv_sum_reg [31:0]
    +---------------------------------------------------------------------------+
    | Pass 4: Normalization Multiplier & Shift-Right                            |
    | prod_prob = (rom_data_exp * inv_sum_reg)                                  |
    | out_prob  = prod_prob [31:16] (UINT16 [0..32767])                         |
    +---------------------------------------------------------------------------+
```

---

## 3. PHÂN VÙNG BẢNG TRA BRAM 1024 TỪ CHO 3 STAGES

Khối BRAM 1024 từ $\times$ 20 bits được nạp sẵn từ tệp `exp_rom_all_stages.hex` và tự động ánh xạ theo tham số `head_dim`:
- **Stage 2 ($d_k = 16$):** Địa chỉ `0x000` .. `0x0FF` (0 .. 255)
- **Stage 3 ($d_k = 20$):** Địa chỉ `0x100` .. `0x1FF` (256 .. 511)
- **Stage 4 ($d_k = 24$):** Địa chỉ `0x200` .. `0x2FF` (512 .. 767)
- **Phân vùng dự phòng:** Địa chỉ `0x300` .. `0x3FF` (768 .. 1023)

Cổng Port B cho phép CPU ARM ghi đè bảng tra tại thời gian chạy (Runtime Reconfiguration) qua các thanh ghi AXI-Lite `0x40` và `0x44`.
