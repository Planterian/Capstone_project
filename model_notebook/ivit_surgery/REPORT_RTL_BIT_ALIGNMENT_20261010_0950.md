# BÁO CÁO PHÂN TÍCH THIẾT KẾ RTL VÀ ĐỒNG BỘ BIT-EXACT (BIT CORRECTION ALIGNMENT) GIỮA PYTORCH I-VIT VÀ KRIA KV260 ACCELERATOR

**Mã tài liệu:** `REPORT_RTL_BIT_ALIGNMENT_20261010_0950`  
**Thời gian khởi tạo:** 10/10/2026 — 09:50:00 (GMT+7)  
**Tác giả:** Senior AI & SW-HW Co-Design Engineer (Antigravity Pairing Session)  
**Mục tiêu:** Phân tích mã nguồn RTL hiện tại (`Project/src/Hardware/rtl/`), đối soát từng bit/công thức số học với mô hình PyTorch I-ViT (MobileViT-XXS) để xác lập quy chuẩn "Bit Correction", phục vụ trích xuất Golden Model bit-exact cho kiểm thử RTL và triển khai tăng tốc trên AMD Kria KV260.  
**Lưu ý quan trọng từ Nhóm nghiên cứu:** Toàn bộ mã nguồn RTL hiện tại trong thư mục `Project/src/Hardware/rtl/` **chỉ đóng vai trò là mẫu tham khảo (reference template)**, chưa có giá trị thực thi chính thức do chưa trải qua quá trình kiểm thử (testbench verification) với Golden Model (vì trước đó chưa xây dựng được Golden Model chuẩn).

---

## BẢNG LỊCH SỬ THAY ĐỔI (CHANGELOG)

| Phiên bản | Thời gian | Người thực hiện | Nội dung cập nhật |
| :--- | :--- | :--- | :--- |
| **v1.0.0** | 2026-10-10 09:50:00 | Senior AI & RTL Engineer | Khởi tạo báo cáo phân tích mã nguồn RTL, xác định các điểm bất tương thích bit giữa RTL mẫu và PyTorch I-ViT, và đề xuất kiến trúc Bit Correction chuẩn hóa. |

---

## 1. TỔNG QUAN PHÂN BỔ KIẾN TRÚC SW-HW TRÊN KRIA KV260

Hệ thống tăng tốc MobileViT-XXS trên SoC AMD Kria KV260 chia thành 2 phần rõ rệt:

```
===================================================================================================================
                                  SOC AMD KRIA KV260 SW-HW CO-DESIGN PARTITIONING
===================================================================================================================

     ARM CORTEX-A53 PROCESSING SYSTEM (PS)                       PROGRAMMABLE LOGIC (FPGA PL)
  +-------------------------------------------+               +-------------------------------------------------+
  | - Image Ingress (Camera / DDR4)           |               |  ATTENTION HARDWARE ACCELERATOR (PL Core)       |
  | - Image Preprocessing (BGR, [0, 1])       |               |                                                 |
  | - CNN Stem (Conv 3x3, Inverted Residuals) |   AXI-Stream  |  +--------------------+   +-------------------+ |
  | - Patch Partitioning (Unfold -> 256 tokens| ------------> |  | Ping-Pong BRAM     |-->| Systolic GEMM QK^T| |
  | - IntLayerNorm (I-LayerNorm)              |   (Ingress)   |  | (Q, K, V Buffers)  |   | (2x INT8 DSP Pack)| |
  | - Projection Linear (QuantLinear)         |               |  +--------------------+   +---------+---------+ |
  | - MLP (ShiftGELU, QuantLinear)            |               |                                     |           |
  | - Classifier Head (Linear 1000 classes)   |               |                                     v           |
  | - PYNQ Driver & AXI-Lite CSR Management   |               |  +--------------------+   +---------+---------+ |
  |                                           | <------------ |  | Systolic GEMM S*V  |<--| Hardware Shiftmax | |
  |                                           |   AXI-Stream  |  | (Output Context)   |   | (MaxSub, Exp, Rec)| |
  |                                           |   (Egress)    |  +--------------------+   +-------------------+ |
  +-------------------------------------------+               +-------------------------------------------------+
```

* **Trọng tâm tăng tốc phần cứng (FPGA PL):** Toàn bộ lõi tính toán **Scaled Dot-Product Attention**:
  $$\text{Context} = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$
* **Các tầng còn lại chạy trên ARM PS (hoặc tăng tốc tuần tự sau):** CNN Stem, LayerNorm, Linear Projections, MLP và Head.

---

## 2. KIỂM TOÁN CHI TIẾT CÁC TỆP NGUỒN RTL MẪU (`Hardware/rtl/`)

Dưới đây là kết quả rà soát chi tiết 10 tệp mã nguồn SystemVerilog trong `Project/src/Hardware/rtl/`:

### 2.1 Tệp [`axi_lite_regs.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/axi_lite_regs.sv) & [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/attention_core_top.sv)
* **Chức năng:** Thanh ghi điều khiển (Control & Status Registers - CSR) qua bus AXI4-Lite 32-bit từ CPU ARM.
* **Các thông số cấu hình mặc định trong RTL:**
  - `ADDR_CTRL (0x00)`: bit 0 = `start_pulse`, bit 1 = `clear_buf`.
  - `ADDR_NBYTES (0x04)`: mặc định 18,816 bytes.
  - `ADDR_TOKENS (0x10)`: mặc định **196 tokens** (`N = 196`).
  - `ADDR_HDIM (0x18)`: mặc định **32** (`d_k = 32`).
  - `ADDR_SHIFT (0x1C)`: mặc định **2** (`reg_shift_val = 4'd2`).
* **Điểm bất tương thích nghiêm trọng với MobileViT-XXS:**
  1. **Số Token $N$:** ViT-Base chuẩn dùng ảnh $224 \times 224$, patch 16 $\rightarrow N = 14 \times 14 = 196$. Nhưng **MobileViT-XXS dùng patch 2 unfolding từ feature map $16 \times 16 \rightarrow N = 256$ tokens!**
     - RTL mẫu gán cứng `N_TOKENS = 196`. Khi nhận dòng dữ liệu 256 token từ MobileViT, phần cứng sẽ ngắt sau 196 token, bỏ rơi 60 token còn lại!
  2. **Số chiều Head $d_k$:** MobileViT-XXS có $d_k$ biến thiên qua 3 Stage:
     - Stage 2: `hidden_size = 64`, 4 heads $\rightarrow \mathbf{d_k = 16}$.
     - Stage 3: `hidden_size = 80`, 4 heads $\rightarrow \mathbf{d_k = 20}$.
     - Stage 4: `hidden_size = 96`, 4 heads $\rightarrow \mathbf{d_k = 24}$.
     - RTL mẫu đặt cố định `d_k = 32`, không khớp với bất kỳ stage nào của MobileViT-XXS!

---

### 2.2 Tệp [`pe_slice_dsp_packing.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/pe_slice_dsp_packing.sv)
* **Chức năng:** Đóng gói 2 phép nhân có dấu 8-bit ($A \times B$ và $A \times C$) vào 1 bộ nhân DSP48E2 ($18 \times 27$ bit) trên FPGA Xilinx UltraScale+.
* **Lỗi cú pháp mở rộng bit (Sign-Extension Bug):**
  - Dòng 35-37:
    ```verilog
    assign operand_a_extended = {{10{operand_a}}, operand_a}; // LỖI TRẦM TRỌNG!
    assign operand_b_extended = {{19{operand_b}}, operand_b};
    assign operand_c_extended = {{19{operand_c}}, operand_c};
    ```
  - **Phân tích lỗi:** `{10{operand_a}}` trong SystemVerilog nhân bản **cả vector 8-bit** 10 lần thành 80 bit, chứ không phải nhân bản bit dấu `operand_a[7]`. Khi nối với `operand_a` (8 bit) tạo thành chuỗi 88 bit, sau đó bị cắt lấy `[17:0]`, làm sai lệch hoàn toàn giá trị số học có dấu!
  - **Sửa đúng chuẩn IEEE 1800:**
    ```verilog
    assign operand_a_extended = {{10{operand_a[7]}}, operand_a};
    assign operand_b_extended = {{19{operand_b[7]}}, operand_b};
    assign operand_c_extended = {{19{operand_c[7]}}, operand_c};
    ```
* **Lỗi lệch nhịp xung (Pipelining Skew Bug):**
  - Dòng 50-52:
    `dsp_product` được gán non-blocking `<=`, nhưng `result_ab` và `result_ac` ở cùng clock edge lại lấy ngay giá trị `high_corrected` tính từ `dsp_product` của chu kỳ trước! Cần tách thanh ghi trễ 1 clock đồng bộ cho `dsp_product`.

---

### 2.3 Tệp [`systolic_mac_array.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/systolic_mac_array.sv)
* **Chức năng:** Mảng Systolic 2D kích thước $4 \times 4$ tính toán tích ma trận Output-Stationary.
* **Lỗi cắt cụt độ rộng bit (Severe Truncation Bug):**
  - Dòng 86-93 tích lũy kết quả 32-bit: `pe_acc_matrix[r][c] <= pe_acc_matrix[r][c] + ...` (INT32).
  - Nhưng dòng 106-107 khi đưa vào Adder Tree lại viết:
    ```verilog
    assign tree_in_a[c] = pe_acc_matrix[r][c][7:0]; // CẮT MẤT 24 BIT CAO!
    assign tree_in_b[c] = 8'sd1;
    ```
  - **Phân tích:** Việc vứt bỏ các bit từ `[31:8]` và chỉ giữ lại 8 bit thấp `[7:0]` khiến bộ tích lũy tràn số ngay khi kết quả nhân vượt quá 127. Đây là nguyên nhân khiến mô phỏng RTL mẫu hiện tại ra kết quả ngẫu nhiên.
  - **Sửa đúng:** Đầu ra của bộ tích lũy trong Systolic Array phải giữ nguyên vẹn độ chính xác `signed [31:0]`.

---

### 2.4 Tệp [`scale_unit.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/scale_unit.sv)
* **Chức năng:** Bộ chuẩn hóa tỷ lệ trước Softmax.
* **Hiện trạng RTL mẫu:**
  ```verilog
  assign shifted_score = in_score >>> shift_val; // Chỉ là dịch phải số học (ASR)
  ```
* **Bất tương thích toán học với I-ViT:**
  1. Phép nhân số nguyên trong Systolic Array là:
     $$\text{Raw\_Score}_{\text{int}} = Q_{\text{int}} \cdot K_{\text{int}}^T \quad (\text{INT32})$$
  2. Giá trị thực tế cần chia cho $\sqrt{d_k}$ và đưa vào Softmax:
     $$\text{Attn\_Score} = \text{Raw\_Score}_{\text{int}} \cdot \frac{S_Q \cdot S_K}{\sqrt{d_k}}$$
  3. Sau đó Softmax trong I-ViT cần lượng tử hóa đầu vào này với scale $S_{\text{score}}$:
     $$\text{Input\_to\_Softmax}_{\text{int}} = \text{round}\left( \text{Raw\_Score}_{\text{int}} \cdot \frac{S_Q \cdot S_K}{S_{\text{score}} \cdot \sqrt{d_k}} \right)$$
  4. Hệ số tỷ lệ thực sự là một số thập phân:
     $$S_{\text{effective}} = \frac{S_Q \cdot S_K}{S_{\text{score}} \cdot \sqrt{d_k}} \approx M_{\text{scale}} \cdot 2^{-E_{\text{scale}}}$$
  5. Nếu RTL chỉ dịch bit `in_score >>> shift_val` (tương đương chia cho $2^{\text{shift\_val}}$):
     - Chỉ đúng duy nhất ở Stage 2 khi $\sqrt{16} = 4 = 2^2$ VÀ NẾU $\frac{S_Q \cdot S_K}{S_{\text{score}}} = 1$.
     - Hoàn toàn sai ở Stage 3 ($d_k = 20 \rightarrow \sqrt{20} \approx 4.472$) và Stage 4 ($d_k = 24 \rightarrow \sqrt{24} \approx 4.899$).
  - **Yêu cầu Bit Correction:** Khối `scale_unit.sv` **bắt buộc phải nâng cấp thành Dyadic Multiplier Unit**:
    $$\text{scaled\_score} = \text{clamp}_{\text{INT16}}\left( (\text{in\_score} \times M_{\text{scale}}) \ggg E_{\text{scale}} \right)$$
    với $M_{\text{scale}}$ (16-bit signed) và $E_{\text{scale}}$ (5-bit unsigned) được cấu hình linh hoạt qua thanh ghi AXI-Lite từ PS.

---

### 2.5 Tệp [`softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/softmax_lut.sv)
* **Chức năng:** Khối tính Softmax phần cứng gồm: Trừ Max $\rightarrow$ Tra bảng Exp ROM $\rightarrow$ Nghịch đảo tổng $\rightarrow$ Chuẩn hóa xác suất.
* **Bất tương thích toán học với `IntSoftmax` của I-ViT:**
  1. **Bảng Exp ROM hiện tại trong RTL:**
     ```verilog
     for (int i = 0; i < ROM_DEPTH; i++) begin
         exp_rom[i] = 16'd65535 >> (i >> 3); // Bậc thang cứ mỗi 8 đơn vị thì chia đôi!
     end
     ```
     Đây là phép xấp xỉ bậc thang rất thô, làm sai lệch phân phối xác suất Attention.
  2. **Toán tử `IntSoftmax` (Shiftmax) chuẩn trong I-ViT:**
     Dùng xấp xỉ đa thức dịch bit chính xác:
     $$x_{\text{int}}' = x_{\text{int}} + \lfloor x_{\text{int}}/2 \rfloor - \lfloor x_{\text{int}}/16 \rfloor$$
     Sau đó phân rã phần nguyên $q$ và phần dư $r$ với $x_0 = \lfloor -1.0 / S_{\text{attn}} \rfloor$:
     $$\exp_{\text{int}} = (r/2 - x_0) \cdot 2^{n - q}$$
  3. **Bộ chia phần cứng tổ hợp (Hardware Divider):**
     Dòng 132: `inv_sum_next = (32'h0100_0000) / sum_exp_reg;`
     Phép chia tổ hợp 32-bit trong một chu kỳ clock là điều cấm kỵ trong thiết kế vi mạch FPGA tốc độ cao (gây vi phạm timing `slack < 0` ở tần số > 100MHz).
  - **Yêu cầu Bit Correction:** Cần đồng bộ bảng LUT hoặc đa thức dịch bit giữa Python và RTL sao cho cả 2 bên cùng tính ra đúng **từng bit một** trên 256 phần tử xác suất.

---

### 2.6 Tệp [`requant_residual_add.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl/requant_residual_add.sv)
* **Đánh giá:** Đây là module **duy nhất trong RTL mẫu được thiết kế đúng chuẩn Dyadic Scale ($M \cdot 2^{-E}$)**:
  ```verilog
  scaled_f <= (i_f_data * m_f) >>> e_f;
  scaled_x <= (i_x_data * m_x) >>> e_x;
  sum_full <= scaled_f + scaled_x;
  clamped_out <= clamp_int8(sum_full);
  ```
  Module này tương ứng 100% với hàm `fixedpoint_mul` khi có `identity` trong I-ViT:
  $$Z = \text{clamp}_{\text{INT8}}\left( \text{round}(X \cdot M_X \cdot 2^{-E_X}) + \text{round}(Y \cdot M_Y \cdot 2^{-E_Y}) \right)$$
  Kiến trúc này sẽ được giữ lại làm chuẩn mẫu để chuẩn hóa ngược lại cho `scale_unit.sv`.

---

## 3. MA TRẬN ĐỐI SOÁT BIT-EXACT (RTL VS PYTORCH I-VIT)

| Khối chức năng | RTL Mẫu Hiện Tại (`Hardware/rtl/`) | PyTorch I-ViT Chuẩn (`ivit_surgery/`) | Trạng thái đồng bộ | Hành động Bit Correction cần làm |
| :--- | :--- | :--- | :---: | :--- |
| **Token Length ($N$)** | $N = 196$ (hardcoded) | $N = 256$ (MobileViT-XXS) | ❌ Lệch 60 token | Cập nhật tham số RTL $N = 256$ và hỗ trợ cấu hình qua AXI-Lite. |
| **Head Dim ($d_k$)** | $d_k = 32$ | $d_k \in \{16, 20, 24\}$ | ❌ Lệch hoàn toàn | Hỗ trợ $d_k$ động: Stage 2 (16), Stage 3 (20), Stage 4 (24). |
| **DSP Packing PE** | Replication vector 8-bit | $A \times B$ và $A \times C$ (INT8) | ⚠️ Lỗi sign bit | Sửa replication bit dấu `operand[7]` trong `pe_slice_dsp_packing.sv`. |
| **Systolic Accumulator** | Cắt lấy `[7:0]` ở lối ra | Tích lũy INT32 đầy đủ | ❌ Tràn số trầm trọng | Giữ nguyên 32-bit accumulator ra khối Scaler. |
| **Scaler ($1/\sqrt{d_k}$)** | ASR Shift `>>> shift_val` | Dyadic Multiplier $(M, E)$ | ❌ Không khớp bit | Đổi `scale_unit` thành Dyadic Rescaler: $(X \times M) \ggg E$. |
| **Softmax Engine** | Exp ROM `65535 >> (i >> 3)` | Shiftmax $x + x/2 - x/16$ | ❌ Khác thuật toán | Chọn 1 trong 2 chuẩn để đồng bộ 100% toán tử. |
| **Output Activation** | 8-bit Unsigned $[0, 255]$ | 8-bit Signed $[-128, 127]$ | ⚠️ Lệch kiểu dữ liệu | Thống nhất dữ liệu ma trận Attention Prob là Unsigned UINT8 hay Signed INT8. |
| **Residual Requant** | Dyadic $(M, E)$ 16-bit | `fixedpoint_mul` 31-bit | 🟡 Tương đồng kiến trúc | Cắt ngắn mantissa $M$ từ 31-bit trong PyTorch về 16-bit để vừa DSP48E2. |

---

## 4. CHI TIẾT ĐỒNG BỘ BIT CORRECTION (ALIGNMENT SPECIFICATION)

Để đảm bảo trích xuất được Golden Model có thể pass testbench SystemVerilog $100\%$ không lệch 1 LSB, chúng ta định nghĩa thông số Bit Correction chi tiết cho từng khối như sau:

```
[PyTorch Quantized FP32/INT8] ---> [Trích xuất Golden Model INT8] ---> [RTL SystemVerilog Testbench]
                                                  |
                                                  v
                                     So sánh từng chu kỳ xung clock
                                     Bit-exact Match: MSE = 0, Error = 0 LSB!
```

### 4.1 Đồng bộ khối Scaler Unit ($QK^T \rightarrow$ Softmax Input)
1. **Toán học Bit Correction:**
   Cho $Q_{\text{int}} \in [-128, 127]$ và $K_{\text{int}} \in [-128, 127]$.
   Tích vô hướng một hàng độ dài $d_k$:
   $$P_{\text{acc}} = \sum_{i=1}^{d_k} Q_{\text{int}}[i] \cdot K_{\text{int}}[i] \quad \in [-128^2 \cdot d_k, 127^2 \cdot d_k]$$
   Với $d_k = 24$, dải giá trị nằm trong $[-393,216, +387,096]$, vừa vặn trong **20-bit có dấu**, được lưu trong thanh ghi **32-bit có dấu** (`signed [31:0]`).
2. **Hệ số Dyadic Scaler:**
   Hệ số chuyển đổi từ tích số nguyên sang đầu vào INT16 của Softmax:
   $$S_{\text{scaler}} = \frac{S_Q \cdot S_K}{S_{\text{attn\_in}} \cdot \sqrt{d_k}}$$
   Trong PyTorch, ta phân rã với độ rộng mantissa 16-bit:
   $$M_{\text{scaler}}, E_{\text{scaler}} = \text{frexp}_{16}(S_{\text{scaler}})$$
   sao cho:
   $$\text{Score}_{\text{scaled}} = \text{clamp}_{\text{INT16}}\left( (P_{\text{acc}} \cdot M_{\text{scaler}}) \ggg E_{\text{scaler}} \right)$$
3. **Thanh ghi cấu hình AXI-Lite tương ứng:**
   - `0x20`: `REG_SCALE_M` (16-bit signed mantissa).
   - `0x24`: `REG_SCALE_E` (5-bit shift exponent).

---

### 4.2 Đồng bộ khối Hardware Softmax (Shiftmax)
Để phần cứng đạt hiệu năng cao nhất trên FPGA Kria KV260 mà không cần bộ chia lớn:
1. **Triển khai bảng Exp LUT chính xác (256 entries $\times$ 16-bit):**
   Thay vì công thức dịch bit thô `65535 >> (i >> 3)`, bảng ROM sẽ chứa giá trị số mũ thực:
   $$\text{ROM}[i] = \text{round}\left( 65535 \cdot 2^{-i / 16} \right) \quad \text{với } i = 0 \dots 255$$
   Bảng này cho độ chính xác cực cao, hoàn toàn vừa vặn trong **1 khối BRAM 18K** của KV260.
2. **Nghịch đảo tổng số mũ bằng thuật toán Newton-Raphson hoặc Dyadic Shift:**
   Thay vì chia tổ hợp, dùng 2 chu kỳ DSP để tính:
   $$\text{Inv\_Sum} = \text{LUT\_Reciprocal}(\text{Sum}[31:16]) \cdot 2^{-E}$$
3. **Mô phỏng ngược vào PyTorch Golden Model:**
   Tạo module `BitExactSoftmax` trong PyTorch sử dụng đúng bảng LUT trên để khi chạy suy luận trên Python, từng giá trị đầu ra xác suất $\text{prob}[0 \dots 255]$ giống hệt từng bit với RTL!

---

### 4.3 Đồng bộ kích thước BRAM Ping-Pong Buffer
* **Kích thước Token:** $N = 256$ tokens.
* **Kích thước Head tối đa:** $d_k = 32$ (dự phòng cho cả $d_k=16, 20, 24$).
* **Dung lượng BRAM cần thiết cho mỗi ma trận (Q, K, V):**
  $$256 \text{ tokens} \times 32 \text{ bytes} = 8,192 \text{ bytes} = 8 \text{ KB}$$
* Mỗi Bank (Bank 0, Bank 1) cần $3 \times 8 \text{ KB} = 24 \text{ KB}$ BRAM.
* Tổng cộng Ping-Pong Buffer 2 Bank cần **48 KB BRAM**, tương đương **24 khối RAMB36K** trên Kria KV260 (chip Zynq UltraScale+ xck26 có tới 144 khối RAMB36K $\rightarrow$ chỉ chiếm **$16.6\%$ tài nguyên BRAM**, cực kỳ an toàn).

---

## 5. BẢNG THANH GHI AXI4-LITE CSR ĐỀ XUẤT CHO KRIA KV260 (ARM PS $\leftrightarrow$ FPGA PL)

Bảng phân bổ địa chỉ thanh ghi chuẩn hóa để nạp cấu hình từ ứng dụng Python PYNQ xuống lõi RTL:

| Offset | Tên thanh ghi | Quyền | Mô tả chức năng | Giá trị mặc định (MobileViT) |
| :---: | :--- | :---: | :--- | :--- |
| `0x00` | `CSR_CTRL` | R/W | Bit 0: `START` (xung kích hoạt GEMM), Bit 1: `CLEAR_BUF`, Bit 2: `SOFT_RESET` | `0x00000000` |
| `0x04` | `CSR_STATUS` | R | Bit 0: `BUSY`, Bit 1: `DONE`, Bit 2: `ERROR`, Bit 3: `BANK_PP_SEL` | `0x00000000` |
| `0x08` | `CSR_NBYTES` | R/W | Tổng số byte dữ liệu DMA truyền vào (Q+K+V) | `24576` ($256 \times 32 \times 3$) |
| `0x10` | `CSR_SEQ_LEN` | R/W | Chiều dài chuỗi token $N$ | `256` |
| `0x14` | `CSR_HEAD_DIM`| R/W | Chiều dài vector đặc trưng mỗi head ($d_k$) | `16` (Stage 2) / `20` / `24` |
| `0x18` | `CSR_NUM_HEAD`| R/W | Số lượng head chú ý ($H$) | `4` |
| `0x20` | `CSR_SCALE_M` | R/W | Mantissa $M$ của khối Scaler Unit ($1/\sqrt{d_k}$) (16-bit có dấu) | Xuất từ PyTorch QAT |
| `0x24` | `CSR_SCALE_E` | R/W | Bit-shift exponent $E$ của khối Scaler Unit (5-bit) | Xuất từ PyTorch QAT |
| `0x28` | `CSR_OUT_M`   | R/W | Mantissa $M$ tái lượng tử hóa đầu ra sau $S \times V$ | Xuất từ PyTorch QAT |
| `0x2C` | `CSR_OUT_E`   | R/W | Bit-shift exponent $E$ tái lượng tử hóa đầu ra sau $S \times V$ | Xuất từ PyTorch QAT |

---

## 6. QUY TRÌNH TRÍCH XUẤT GOLDEN MODEL TỪ PYTORCH SANG VERILOG TESTBENCH

Để xác minh thiết kế RTL với độ tin cậy tuyệt đối:

```
[PyTorch QAT Model Đã Tinh Chỉnh Xong]
                 │
                 ├── 1. forward_hook trích xuất tensor đầu vào:
                 │      - q_matrix.hex (256x16 INT8)
                 │      - k_matrix.hex (256x16 INT8)
                 │      - v_matrix.hex (256x16 INT8)
                 │
                 ├── 2. forward_hook trích xuất tensor trung gian & đầu ra (Golden Data):
                 │      - golden_gemm_qk.hex (256x256 INT32)
                 │      - golden_scaled_score.hex (256x256 INT16)
                 │      - golden_softmax_prob.hex (256x256 UINT8)
                 │      - golden_context_out.hex (256x16 INT8)
                 │
                 └── 3. Kịch bản SystemVerilog Testbench (`tb_attention_core.sv`):
                        - $readmemh("q_matrix.hex", u_top.u_bram.ram_q);
                        - Kích hoạt xung START qua AXI-Lite
                        - Chờ tín hiệu DONE
                        - So sánh từng bit đầu ra với golden_context_out.hex
                        - In báo cáo: Total Errors = 0, BIT-EXACT PASS! ✓
```

---

## 7. KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG TIẾP THEO

1. **RTL hiện tại chưa thể chạy được ngay:** Các lỗi về mở rộng bit dấu (`pe_slice`), cắt bỏ 24-bit (`systolic_mac_array`), và gán cứng $N=196, d_k=32$ khiến phần cứng không thể tương thích với MobileViT-XXS.
2. **Quy trình chuẩn kỹ thuật:**
   - **Bước 1:** Khắc phục triệt để các lỗi dữ liệu và huấn luyện QAT trong PyTorch (theo báo cáo [`REPORT_SURGERY_QAT_FLAWS_20261010_0939.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_SURGERY_QAT_FLAWS_20261010_0939.md)) để phục hồi Accuracy đạt **65% - 68%**.
   - **Bước 2:** Cố định các giá trị dyadic scale $(M, E)$ từ mô hình PyTorch đã hội tụ thành công.
   - **Bước 3:** Trích xuất bộ dữ liệu kiểm thử vàng (Golden Test Vectors).
   - **Bước 4:** Áp dụng các chỉnh sửa Bit Correction trong báo cáo này vào các tệp RTL SystemVerilog.
   - **Bước 5:** Chạy mô phỏng Vivado Testbench đối soát với Golden Model trước khi tổng hợp bitstream lên Kria KV260.

---
*Báo cáo được lưu trữ vĩnh viễn tại `model_notebook/ivit_surgery/REPORT_RTL_BIT_ALIGNMENT_20261010_0950.md` phục vụ công tác đối soát SW-HW codesign.*
