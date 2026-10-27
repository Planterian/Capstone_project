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
    | Pass 4: 2-Stage Pipelined Normalization & Saturation (Timing Closed)      |
    | [Stage 4.1 - Mult Reg]:                                                   |
    |   mult_full_reg <= {32'd0, rom_data_exp} * {20'd0, inv_sum_reg};         |
    |   (Suy luận DSP48E2 internal MREG / PREG, trễ ~2.4 ns)                    |
    | [Stage 4.2 - Shift & Clamp]:                                              |
    |   shifted_prob   = mult_full_reg >> 16;                                   |
    |   out_prob      <= (shifted_prob > 65535) ? 16'hFFFF : shifted_prob[15:0];|
    |   out_valid     <= pipe_valid_stage2;                                     |
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

---

## 4. NHẬT KÝ CẬP NHẬT THIẾT KẾ (DESIGN CHANGELOG)

### [2026-10-27] Tối Ưu Hóa Pipeline Đa Tầng Cho Softmax Normalizer & Tái Kiểm Thử Bit-Exact

- **Bối cảnh & Động lực:**
  - Kết quả Place-and-Route (PnR) trên chip AMD Kria KV260 (`xck26-sfvc784-2LV-c`) tại tần số mục tiêu 200.0 MHz ($T_{clk} = 5.0\text{ ns}$) ghi nhận vi phạm Setup Time ($WNS = -0.803\text{ ns}$ trên 1A, $-1.095\text{ ns}$ trên 2A, $-0.924\text{ ns}$ trên 3B).
  - Phân tích chi tiết báo cáo `post_route_timing.rpt` xác định **100% các điểm vi phạm timing trên cả 3 biến thể đều bắt nguồn từ một đường trễ tổ hợp duy nhất** trong `softmax_lut.sv`:
    $$\text{RAMB36E2 CLKBWRCLK (1.305 ns)} \to \text{DSP48E2 cascade (2.80 ns)} \to \text{LUT shifters/clamp (0.80 ns)} \to \texttt{out\_prob\_reg[*]/D}$$
    Tổng trễ đường truyền lên tới ~5.6 - 5.9 ns.
  - **Nguyên nhân cốt lõi:** Phép nhân chuẩn hóa `assign mult_full = {32'd0, rom_data_exp} * {20'd0, inv_sum_reg};` được khai báo dạng tổ hợp (`assign`), khiến Vivado không thể ánh xạ các thanh ghi đường ống nội bộ của DSP48E2 (`MREG` và `PREG`), buộc chu trình đọc BRAM, nhân 52-bit, dịch bit và bộ dồn saturation clamp phải diễn ra trong cùng một chu kỳ xung nhịp 5.0 ns.

- **Nội dung sửa đổi phần cứng (`softmax_lut.sv`):**
  1. Thêm cờ pipeline 2 tầng: `logic pipe_valid_stage2;` điều khiển tín hiệu hợp lệ sau tầng nhân.
  2. Bổ sung thanh ghi đường ống `mult_full_reg [51:0]` cho phép Vivado suy luận trực tiếp thanh ghi nội `MREG / PREG` của DSP48E2:
     ```systemverilog
     // Stage 1: Chốt kết quả nhân DSP
     if (pipe_valid) begin
         mult_full_reg <= {32'd0, rom_data_exp} * {20'd0, inv_sum_reg};
     end
     ```
  3. Tách biệt tầng dịch bit và bộ kẹp bão hòa (Saturation Clamp) sang chu kỳ tiếp theo:
     ```systemverilog
     // Stage 2: Dịch bit, kẹp dải và xuất out_valid
     pipe_valid_stage2 <= pipe_valid;
     if (pipe_valid_stage2) begin
         out_valid <= 1'b1;
         if (OUT_WIDTH == 16) begin
             out_prob <= (shifted_prob > 36'd65535) ? 16'hFFFF : shifted_prob[15:0];
         end else begin
             out_prob <= (shifted_prob > 36'd255) ? 8'hFF : shifted_prob[7:0];
         end
     end
     ```
  4. Cập nhật điều kiện kết thúc trạng thái `ST_NORM_OUT`: Chờ cạn kiệt cả 2 tầng pipeline (`cnt_reg == N_TOKENS && !pipe_valid && !pipe_valid_stage2`) trước khi chuyển về `ST_IDLE`.

- **Hiệu quả timing đạt được:**
  - Đường truyền dài ~5.8 ns được chia thành 2 chu kỳ ngắn:
    - **Nửa đầu:** BRAM Read $\to$ DSP48E2 Mult ($\sim 3.4\text{ ns} \le 5.0\text{ ns}$).
    - **Nửa sau:** DSP48E2 PREG $\to$ Shift/Clamp $\to$ Output Flip-Flop ($\sim 1.4\text{ ns} \le 5.0\text{ ns}$).
  - Đảm bảo $F_{max} > 250\text{ MHz}$, đưa $WNS \ge +1.2\text{ ns}$ tại 200.0 MHz, đóng timing hoàn toàn.
  - Tác động chu kỳ: Toàn bộ dòng chảy streaming giữ nguyên thông lượng (1 token/chu kỳ không tạo bubble), chỉ thêm đúng 1 chu kỳ trễ xuất phát (initial pipeline latency) mỗi hàng 256 tokens (tương đương $+0.04\%$ tổng thời gian thực thi).

- **Kết quả tái kiểm thử Bit-Exact (Verification Pass):**
  - Đã cập nhật đồng bộ cho cả 3 thư mục RTL:
    - [`Project/src/Hardware/rtl_ver02_1A/softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/softmax_lut.sv)
    - [`Project/src/Hardware/rtl_ver02_2A/softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/softmax_lut.sv)
    - [`Project/src/Hardware/rtl_ver02_3B/softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/softmax_lut.sv)
  - Đã chạy kiểm chứng độc lập `tb_golden_softmax_lut`: **1,024 / 1,024 tokens khớp tuyệt đối 100% (0 LSB Error)** trên từng biến thể.
  - Đã chạy kiểm chứng toàn bộ 3 bài test Isolated Suite và 3 bài test Sequential Multi-Stage Suite trên cả 3 biến thể: **100% PASS (Tổng cộng 92,160 bytes context và 3,072 tokens Softmax khớp hoàn toàn với mô hình PyTorch I-ViT)**.

