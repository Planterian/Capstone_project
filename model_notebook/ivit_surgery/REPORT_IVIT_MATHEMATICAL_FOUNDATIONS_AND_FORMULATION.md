# BÁO CÁO TOÀN DIỆN MÔ HÌNH TOÁN HỌC VÀ NGUYÊN LÝ LƯỢNG TỬ HÓA SỐ NGUYÊN THUẦN TÚY I-ViT (INTEGER-ONLY VISION TRANSFORMER)

**Mã tài liệu:** `REPORT_IVIT_MATHEMATICAL_FOUNDATIONS_AND_FORMULATION.md`  
**Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
**Tài liệu tham chiếu gốc:** Thư mục mã nguồn chính thức [`c:/Users/tuan2/Desktop/I-ViT`](file:///c:/Users/tuan2/Desktop/I-ViT) và bài báo khoa học ICCV 2023: *["I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference"](https://arxiv.org/abs/2207.01405)* (Zhikai Li, Qingyi Gu).  
**Thời gian lập:** 19/10/2026  
**Vị trí lưu trữ:** [`model_notebook/ivit_surgery/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/) và [`Notes_ver02/toan_hoc_ivit_transformer.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/toan_hoc_ivit_transformer.md)

---

## MỤC LỤC
1. [TỔNG QUAN & ĐỘNG LỰC TOÁN HỌC CỦA I-ViT (EXECUTIVE MOTIVATION)](#1-tổng-quan--động-lực-toán-học-của-i-vit)
2. [CƠ SỞ LƯỢNG TỬ HÓA ĐỐI XỨNG & ĐẠO HÀM STE (UNIFORM QUANTIZATION & STE)](#2-cơ-sở-lượng-tử-hóa-đối-xứng--đạo-hàm-ste)
3. [SỐ HỌC DYADIC & CƠ CHẾ QUY ĐỔI TỶ LỆ CỐ ĐỊNH (DYADIC SCALING ENGINE)](#3-số-học-dyadic--cơ-chế-quy-đổi-tỷ-lệ-cố-định)
4. [SHIFTMAX: MÔ HÌNH TOÁN HỌC SOFTMAX SỐ NGUYÊN THUẦN TÚY (INTEGER-ONLY SOFTMAX)](#4-shiftmax-mô-hình-toán-học-softmax-số-nguyên-thuần-túy)
   - 4.1 [Xấp xỉ hàm mũ qua cơ số 2 và hằng số dịch bit 23/16](#41-xấp-xỉ-hàm-mũ-qua-cơ-số-2-và-hằng-số-dịch-bit-2316)
   - 4.2 [Phân rã phần nguyên - phần phân số & Nội suy tuyến tính](#42-phân-rã-phần-nguyên---phần-phân-số--nội-suy-tuyến-tính)
   - 4.3 [Phép chia khôi phục số dư và chuẩn hóa xác suất 16-bit](#43-phép-chia-khôi-phục-số-dư-và-chuẩn-hóa-xác-suất-16-bit)
5. [I-LAYERNORM: CHUẨN HÓA LỚP SỐ NGUYÊN VỚI NEWTON-RAPHSON (INTEGER-ONLY LAYERNORM)](#5-i-layernorm-chuẩn-hóa-lớp-số-nguyên-với-newton-raphson)
   - 5.1 [Tính kỳ vọng và phương sai trên miền số nguyên](#51-tính-kỳ-vọng-và-phương-sai-trên-miền-số-nguyên)
   - 5.2 [Thuật toán Newton-Raphson 10 bước lặp căn bậc hai nguyên](#52-thuật-toán-newton-raphson-10-bước-lặp-căn-bậc-hai-nguyên)
   - 5.3 [Phép biến đổi Affine lượng tử hóa](#53-phép-biến-đổi-affine-lượng-tử-hóa)
6. [SHIFTGELU: HÀM KÍCH HOẠT PHI TUYẾN SỐ NGUYÊN (INTEGER-ONLY GELU)](#6-shiftgelu-hàm-kích-hoạt-phi-tuyến-số-nguyên)
7. [PHÉP CỘNG TẮT RESIDUAL SỐ NGUYÊN VÀ QUY ĐỔI TỶ LỆ (FIXEDPOINT_MUL)](#7-phép-cộng-tắt-residual-số-nguyên-và-quy-đổi-tỷ-lệ)
8. [BẢN ĐỒ ÁNH XẠ TOÁN HỌC SANG PHẦN CỨNG FPGA (KV260 HARDWARE MAPPING)](#8-bản-đồ-ánh-xạ-toán-học-sang-phần-cứng-fpga)

---

## 1. TỔNG QUAN & ĐỘNG LỰC TOÁN HỌC CỦA I-ViT

### 1.1 Nghịch lý lượng tử hóa trong Vision Transformer (ViT)
Trong các mạng tích chập truyền thống (CNN), phần lớn các phép tính là nhân tích chập tuyến tính ($\text{Conv2d}$) và hàm kích hoạt đơn giản ($\text{ReLU}$). Do đó, các kỹ thuật lượng tử hóa INT8 truyền thống (như TensorRT, TFLite) chỉ cần lượng tử hóa trọng số và kích hoạt thành số nguyên, sau đó chèn các phép nhân vô hướng dấu phẩy động (Floating-point Requantization Scale).

Tuy nhiên, trong **Vision Transformer (ViT / MobileViT)**:
1. **Các hàm phi tuyến tính phức tạp:** Softmax, LayerNorm, GELU chiếm tới **$30\% - 45\%$ thời gian tính toán**.
2. **Hiện tượng "Đảo dấu phẩy động" (Floating-Point Islands):** Nếu chỉ lượng tử hóa các phép nhân ma trận (GEMM) thành INT8 nhưng vẫn giữ Softmax và LayerNorm ở dạng FP32, dữ liệu liên tục bị ép kiểu qua lại:
   $$\text{INT8 (GEMM)} \longrightarrow \text{Dequant (FP32)} \longrightarrow \text{Softmax/LayerNorm (FP32)} \longrightarrow \text{Quant (INT8)}$$
   Sự chuyển đổi này làm tăng vọt độ trễ, triệt tiêu lợi thế của bộ xử lý phần cứng (FPGA/ASIC) và đòi hỏi các khối FPU (Floating Point Unit) cồng kềnh.
3. **Mục tiêu của I-ViT:** Loại bỏ **100% các phép tính dấu phẩy động** trên toàn bộ pipeline của Transformer Block, đưa toàn bộ mạng về **Số học Số nguyên Thuần túy (Integer-Only Arithmetic)** sử dụng duy nhất:
   - Phép nhân số nguyên: $\text{INT8} \times \text{INT8} \rightarrow \text{INT32}$.
   - Phép cộng/trừ số nguyên: $\text{INT32} \pm \text{INT32}$.
   - Phép dịch bit số học (Arithmetic Bit-Shift): $\ggg E$.

---

## 2. CƠ SỞ LƯỢNG TỬ HÓA ĐỐI XỨNG & ĐẠO HÀM STE

### 2.1 Lượng tử hóa đều đối xứng (Uniform Symmetric Linear Quantization)
Cho một tensor dấu phẩy động $X \in \mathbb{R}$, miền biểu diễn $b$-bit số nguyên có dấu nằm trong khoảng $[-2^{b-1}, 2^{b-1}-1]$ (đối với INT8: $[-128, +127]$, I-ViT chọn đối xứng $[-127, +127]$ với $n = 2^{b-1}-1 = 127$).

Hệ số tỷ lệ lượng tử hóa $S$ (Scaling Factor) được xác định bằng giá trị cực đại tuyệt đối của dải động:
$$S = \frac{\max(|X_{\min}|, |X_{\max}|)}{2^{b-1} - 1}$$

Toán tử lượng tử hóa ánh xạ số thực $x \in X$ sang số nguyên $q_x \in \mathbb{Z}$:
$$q_x = \text{quant}(x, S) = \text{clamp}\left( \left\lfloor \frac{x}{S} \right\rceil, -2^{b-1}, 2^{b-1}-1 \right)$$
trong đó $\lfloor \cdot \rceil$ là phép làm tròn tới số nguyên gần nhất (Round to Nearest).

Toán tử giải lượng tử (Dequantization) phục vụ quá trình huấn luyện Fake Quantization:
$$\hat{x} = \text{dequant}(q_x, S) = q_x \times S$$

```text
==================================================================================================
                 SƠ ĐỒ ÁNH XẠ MIỀN SỐ THỰC VÀ MIỀN SỐ NGUYÊN (INT8)
==================================================================================================
  Miền Số Thực (FP32):   -Max_Val                  0                  +Max_Val
                           [───────────────────────┼───────────────────────]
                                      │            │            │
                                      ▼ ( / S )    ▼ ( / S )    ▼ ( / S )
  Miền Số Nguyên (INT8):     -127                  0                  +127
                           [───────────────────────┼───────────────────────]
==================================================================================================
```

### 2.2 Đạo hàm xấp xỉ truyền thẳng STE (Straight-Through Estimator)
Vì hàm làm tròn $\lfloor \cdot \rceil$ và hàm sàn $\lfloor \cdot \rfloor$ có đạo hàm bằng 0 tại hầu hết mọi điểm (zero gradient almost everywhere) và không khả vi tại các điểm nguyên, quá trình tối ưu hóa Gradient Descent thông thường sẽ bị tê liệt.

I-ViT sử dụng kỹ thuật **Straight-Through Estimator (STE)**:
- Trong lượt truyền xuôi (Forward Pass): Thực thi phép làm tròn rời rạc chính xác.
- Trong lượt truyền ngược (Backward Pass): Bỏ qua hàm làm tròn và truyền nguyên vẹn gradient:
  $$\frac{\partial \lfloor x \rceil}{\partial x} \approx 1, \quad \frac{\partial \lfloor x \rfloor}{\partial x} \approx 1$$

Biểu diễn toán học của đạo hàm STE trong [`quant_utils.py`](file:///c:/Users/tuan2/Desktop/I-ViT/models/quantization_utils/quant_utils.py#L98-L120):
$$\frac{\partial \mathcal{L}}{\partial x} = \begin{cases} \frac{1}{S} \frac{\partial \mathcal{L}}{\partial q_x} & \text{nếu } q_x \in [-2^{b-1}, 2^{b-1}-1] \\ 0 & \text{nếu tràn ngưỡng (clamped)} \end{cases}$$

---

## 3. SỐ HỌC DYADIC & CƠ CHẾ QUY ĐỔI TỶ LỆ CỐ ĐỊNH

### 3.1 Định lý Phân rã Số học Dyadic (Dyadic Scale Decomposition)
Trong quá trình truyền dữ liệu giữa hai tầng nơ-ron liên tiếp, kích hoạt đầu ra có hệ số tỷ lệ $S_{\text{in}}$, nhưng tầng tiếp theo đòi hỏi dữ liệu đầu vào phải tuân theo hệ số tỷ lệ $S_{\text{out}}$.  
Phép biến đổi tỷ lệ thực tế cần nhân với hệ số thực:
$$\alpha = \frac{S_{\text{in}}}{S_{\text{out}}} \in \mathbb{R}^+$$

Để triệt tiêu hoàn toàn phép nhân số thực $\alpha$, I-ViT phân rã số thực $\alpha$ thành cặp số nguyên **Dyadic Rational Number** $(M, E)$:
$$\alpha \approx M \times 2^{-E}$$
trong đó:
- $M \in \mathbb{Z}^+$ là phần định trị nguyên (Integer Mantissa), biểu diễn dưới dạng 16-bit hoặc 31-bit ($M \in [0, 2^{31}-1]$).
- $E \in \mathbb{N}$ là số mũ nguyên (Integer Exponent Shift Amount).

### 3.2 Thuật toán phân rã Dyadic trong `batch_frexp`
Cho số thực $\alpha > 0$, hàm `frexp` phân tích:
$$\alpha = \text{mantissa} \times 2^{\text{exponent}}, \quad \text{với } \text{mantissa} \in [0.5, 1.0)$$
Với độ dài bit định trị mong muốn $\text{max\_bit}$ (ví dụ 15 bits hoặc 31 bits):
$$M = \left\lfloor \text{mantissa} \times 2^{\text{max\_bit}} \right\rceil$$
$$E = \text{max\_bit} - \text{exponent}$$

**Quy tắc thực thi phần cứng:**
Phép nhân với số thực $\alpha$ được chuyển thành **1 phép nhân nguyên và 1 phép dịch phải đại số (ASR)**:
$$q_{\text{out}} = \left\lfloor \frac{q_{\text{in}} \times M}{2^E} \right\rceil \equiv (q_{\text{in}} \times M) \ggg E$$

```text
==================================================================================================
                      NGUYÊN LÝ THỰC THI NHÂN DYADIC TRÊN PHẦN CỨNG
==================================================================================================
   Dữ liệu nguyên q_in (INT8/INT32) ───┐
                                       ├───> [ Bộ nhân số nguyên (DSP) ] ──> Tích nguyên P
   Tham số Mantissa M (INT16/INT32) ───┘                                          │
                                                                                  v
   Tham số dịch Exponent E (5-bit)  ───────> [ Bộ dịch phải đại số ASR (>>>) ] ───┘
                                                          │
                                                          v
                                              Kết quả nguyên q_out (INT8/INT16)
==================================================================================================
```

---

## 4. SHIFTMAX: MÔ HÌNH TOÁN HỌC SOFTMAX SỐ NGUYÊN THUẦN TÚY

Toán tử Softmax nguyên bản trên số thực:
$$\text{Softmax}(x_i) = \frac{e^{x_i - x_{\max}}}{\sum_{j=1}^N e^{x_j - x_{\max}}}$$
trong đó việc trừ $x_{\max} = \max_j (x_j)$ nhằm đảm bảo $x_i - x_{\max} \le 0$, tránh tràn số khi tính hàm mũ $e^x$.

Trong I-ViT, Softmax được thay thế bằng toán tử **Shiftmax**, loại bỏ hoàn toàn tính toán hàm mũ siêu việt $e^x$ và phép chia số thực thông qua 3 bước biến đổi toán học:

### 4.1 Xấp xỉ hàm mũ qua cơ số 2 và hằng số dịch bit 23/16
Giả sử $x_i$ được lượng tử hóa thành số nguyên $q_i$ với hệ số tỷ lệ $S$:
$$x_i - x_{\max} = (q_i - q_{\max}) \times S$$
Đặt $z = x_i - x_{\max} \le 0$ và số nguyên sai lệch tương ứng $q_z = q_i - q_{\max} \le 0$.

Chuyển đổi hàm mũ tự nhiên cơ số $e$ sang cơ số 2:
$$e^z = 2^{z \cdot \log_2 e}$$
Ta có hằng số:
$$\log_2 e \approx 1.4426950408889634$$

**Khám phá toán học then chốt của I-ViT:**  
Số $1.442695$ được xấp xỉ cực kỳ chính xác bằng tổng các lũy thừa của 2:
$$\log_2 e \approx 1 + \frac{1}{2} - \frac{1}{16} = 1 + 0.5 - 0.0625 = \mathbf{1.4375} = \frac{\mathbf{23}}{\mathbf{16}}$$
Sai số của phép xấp xỉ này chỉ là:
$$\epsilon = \frac{|1.442695 - 1.4375|}{1.442695} \approx \mathbf{0.36\%}$$

Nhờ đó, số nguyên trong hàm mũ được tính toán hoàn toàn bằng phép dịch bit:
$$q_z' = q_z \times \frac{23}{16} = q_z + (q_z \ggg 1) - (q_z \ggg 4)$$
*(Đoạn mã tương ứng trong [`quant_modules.py:470`](file:///c:/Users/tuan2/Desktop/I-ViT/models/quantization_utils/quant_modules.py#L470): `x_int = x_int + floor_ste(x_int / 2) - floor_ste(x_int / 16)`)*.

### 4.2 Phân rã phần nguyên - phần phân số & Nội suy tuyến tính
Sau khi nhân với hệ số tỷ lệ $S$, số mũ thực là:
$$u = q_z' \times S \le 0$$
Đặt bước chia nguyên chuẩn hóa $x_{0,\text{int}} = \left\lfloor -\frac{1}{S} \right\rfloor < 0$.  
Số mũ $q_z'$ được chia nguyên thành thương số $q$ và số dư $r$:
$$q = \left\lfloor \frac{q_z'}{x_{0,\text{int}}} \right\rfloor \ge 0, \quad r = q_z' - x_{0,\text{int}} \cdot q \quad (r \in [x_{0,\text{int}}, 0])$$

Hàm mũ cơ số 2 được phân rã thành:
$$2^u = 2^{-q + \text{fraction}} = 2^{-q} \cdot 2^{\text{fraction}}$$

Phần phân số $2^{\text{fraction}}$ được xấp xỉ bằng đa thức bậc nhất:
$$2^{\text{fraction}} \approx \frac{r}{2} - x_{0,\text{int}}$$
Kết hợp với việc dịch bit $2^{n-q}$ với $n = 15$:
$$\text{exp\_int}(q_z) = \left\lfloor \left( \frac{r}{2} - x_{0,\text{int}} \right) \times 2^{n - q} \right\rfloor$$
Giá trị $\text{exp\_int}$ được kẹp dưới $\ge 0$ và là một số nguyên dương nằm trong khoảng $[0, 2^{19}]$.

### 4.3 Phép chia khôi phục số dư và chuẩn hóa xác suất 16-bit
Tính tổng mẫu số nguyên:
$$\text{Sum\_Exp} = \sum_{j=1}^N \text{exp\_int}(q_{z, j})$$

Để không phải thực hiện phép chia cho từng phần tử, I-ViT tính toán hệ số nghịch đảo chuẩn hóa duy nhất $\text{factor}$ bằng cách chia số nguyên lớn $2^{31} - 1$:
$$\text{factor} = \left\lfloor \frac{2^{31} - 1}{\text{Sum\_Exp}} \right\rfloor$$

Sau đó, xác suất của từng token được chuẩn hóa bằng phép nhân và dịch phải:
$$\text{prob\_int}_i = \left\lfloor \frac{\text{exp\_int}_i \times \text{factor}}{2^{31 - b_{\text{out}} + 1}} \right\rfloor$$
Với độ phân giải $b_{\text{out}} = 16\text{-bit}$ (chuẩn đối soát phần cứng của đồ án):
$$31 - 16 + 1 = 16 \implies \mathbf{\text{prob\_int}_i = (\text{exp\_int}_i \times \text{factor}) \ggg 16}$$
- Giá trị $\text{prob\_int}_i \in [0, 32767]$ (số nguyên dương 16-bit).
- Tổng xác suất trên toàn hàng: $\sum_{i=1}^N \text{prob\_int}_i \approx 32,768 = 2^{15}$.

```text
==================================================================================================
                     SƠ ĐỒ KHỐI TOÁN HỌC SHIFTMAX TRONG I-VIT
==================================================================================================
   Điểm số chú ý Score q_i (INT16) ───> [ Tìm Max: q_max = max(q) ]
                                                        │
                                                        v
   Sai lệch số âm: q_z = q_i - q_max <──────────────────┘
            │
            v
   Xấp xỉ cơ số 2: q_z' = q_z + (q_z >>> 1) - (q_z >>> 4)   (Nhân 23/16 ~= log2(e))
            │
            v
   Nội suy nguyên: exp_int = (r/2 - x0) * 2^(n - q)
            │
            ├───> [ Tính tổng hàng: Sum_Exp = sum(exp_int) ]
            │                         │
            │                         v
            │         [ Bộ chia nguyên: factor = (2^31 - 1) / Sum_Exp ]
            │                         │
            v                         v
   [ Nhân chuẩn hóa: prob_int = (exp_int * factor) >>> 16 ] ──> Xác suất UINT16 [0..32767]
==================================================================================================
```

---

## 5. I-LAYERNORM: CHUẨN HÓA LỚP SỐ NGUYÊN VỚI NEWTON-RAPHSON

Trong kiến trúc Transformer, **Layer Normalization (LayerNorm)** chuẩn hóa các đặc trưng trên từng token độc lập:
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
trong đó $\mu$ là kỳ vọng, $\sigma^2$ là phương sai, $\gamma$ và $\beta$ là trọng số học affine.

I-ViT loại bỏ hoàn toàn các phép toán căn bậc hai thực ($\sqrt{\cdot}$) và chia thực thông qua **I-LayerNorm**:

### 5.1 Tính kỳ vọng và phương sai trên miền số nguyên
Với vector đầu vào $X = [x_1, x_2, \dots, x_d]$ có hệ số lượng tử hóa $S_x$, các giá trị số nguyên $q_i = x_i / S_x$:
1. **Kỳ vọng số nguyên (Integer Mean):**
   $$\mu_{\text{int}} = \left\lfloor \frac{1}{d} \sum_{i=1}^d q_i \right\rceil$$
2. **Khuynh độ lệch tâm (Centered Values):**
   $$y_i = q_i - \mu_{\text{int}}$$
3. **Tổng bình phương độ lệch (Variance Integer Sum):**
   $$\text{var}_{\text{int}} = \sum_{i=1}^d y_i^2$$

### 5.2 Thuật toán Newton-Raphson 10 bước lặp căn bậc hai nguyên
Để tính độ lệch chuẩn $\sigma_{\text{int}} = \sqrt{\text{var}_{\text{int}}}$ mà không cần bộ xử lý căn bậc hai phần cứng, I-ViT sử dụng phương pháp giải phương trình $f(k) = k^2 - \text{var}_{\text{int}} = 0$ bằng thuật toán lặp tiếp tuyến **Newton-Raphson** trên miền số nguyên:
$$k_{t+1} = \left\lfloor \frac{k_t + \left\lfloor \frac{\text{var}_{\text{int}}}{k_t} \right\rfloor}{2} \right\rfloor$$
- Khởi tạo ban đầu: $k_0 = 2^{16} = 65,536$.
- Số vòng lặp: Đúng **10 chu kỳ** lặp là đủ để $k_t$ hội tụ chính xác tới giá trị căn nguyên:
  $$\text{std}_{\text{int}} = k_{10} \approx \sqrt{\text{var}_{\text{int}}}$$

### 5.3 Phép biến đổi Affine lượng tử hóa
Sau khi có $\text{std}_{\text{int}}$:
1. **Nghịch đảo độ lệch chuẩn:**
   $$\text{factor} = \left\lfloor \frac{2^{31} - 1}{\text{std}_{\text{int}}} \right\rfloor$$
2. **Chuẩn hóa giá trị:**
   $$y_{\text{norm}, i} = \left\lfloor \frac{y_i \times \text{factor}}{2} \right\rfloor$$
3. **Tích hợp trọng số Affine và dịch bias:**
   Hệ số tỷ lệ đầu ra mới:
   $$S_{\text{norm}} = \frac{\sqrt{d}}{2^{30}} \times \gamma$$
   Bias $\beta$ được chiếu thành số nguyên tương ứng:
   $$\text{bias}_{\text{int}} = \left\lfloor \frac{\beta / \gamma}{S_{\text{norm}}} \right\rfloor$$
   Giá trị đầu ra số nguyên cuối cùng:
   $$q_{\text{out}, i} = y_{\text{norm}, i} + \text{bias}_{\text{int}}$$

---

## 6. SHIFTGELU: HÀM KÍCH HOẠT PHI TUYẾN SỐ NGUYÊN

Hàm kích hoạt Gaussian Error Linear Unit (GELU) nguyên bản:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
Hàm này thường được xấp xỉ qua hàm Sigmoid $\sigma(\cdot)$:
$$\text{GELU}(x) \approx x \cdot \sigma(1.702 \cdot x) = \frac{x}{1 + e^{-1.702 \cdot x}}$$

I-ViT kế thừa cơ chế tính hàm mũ rời rạc của Shiftmax để xây dựng **ShiftGELU**:
1. Điều chỉnh hệ số tỷ lệ Sigmoid:
   $$S_{\text{sig}} = S_x \times 1.702$$
2. Sử dụng cùng hàm `int_exp_shift` để tính hàm mũ:
   $$e_1 = \text{int\_exp\_shift}(q_x - q_{\max}, S_{\text{sig}})$$
   $$e_0 = \text{int\_exp\_shift}(-q_{\max}, S_{\text{sig}})$$
   $$\text{exp\_sum} = e_1 + e_0$$
3. Tính Sigmoid số nguyên:
   $$\text{factor} = \left\lfloor \frac{2^{31} - 1}{\text{exp\_sum}} \right\rfloor$$
   $$\text{sigmoid\_int} = \left\lfloor \frac{e_1 \times \text{factor}}{2^{31 - b_{\text{out}} + 1}} \right\rfloor \in [0, 2^{b_{\text{out}}-1}]$$
4. Nhân với giá trị đầu vào $q_x$:
   $$q_{\text{gelu}} = q_x \times \text{sigmoid\_int}$$

---

## 7. PHÉP CỘNG TẮT RESIDUAL SỐ NGUYÊN VÀ QUY ĐỔI TỶ LỆ (FIXEDPOINT_MUL)

Trong khối Transformer Encoder, tồn tại 2 phép cộng tắt Residual Add:
1. $Z_1 = X + \text{Attention}(X)$
2. $Z_2 = Z_1 + \text{MLP}(Z_1)$

Vì hai nhánh có dải động khác nhau, hệ số tỷ lệ lượng tử hóa của nhánh chính $S_x$ và nhánh phụ $S_{\text{identity}}$ là hoàn toàn khác nhau.  
Để cộng trực tiếp trên miền số nguyên, hàm `fixedpoint_mul` quy đổi cả hai về hệ số tỷ lệ đích $S_z$ bằng hai phép biến đổi Dyadic độc lập:

$$M_x, E_x = \text{frexp}\left(\frac{S_x}{S_z}\right)$$
$$M_{\text{id}}, E_{\text{id}} = \text{frexp}\left(\frac{S_{\text{identity}}}{S_z}\right)$$

Phương trình cộng tắt số nguyên:
$$q_z = \text{clamp}\left( \left\lfloor \frac{q_x \times M_x}{2^{E_x}} \right\rceil + \left\lfloor \frac{q_{\text{identity}} \times M_{\text{id}}}{2^{E_{\text{id}}}} \right\rceil, -128, 127 \right)$$

```text
==================================================================================================
                 MÔ HÌNH TOÁN HỌC PHÉP CỘNG RESIDUAL INT8 (FIXEDPOINT_MUL)
==================================================================================================
   Nhánh chính q_x (INT8, scale S_x) ────> [ Nhân Dyadic (x M_x) >>> E_x ] ──┐
                                                                              ├──> [ + ] ──> q_z
   Nhánh tắt q_id (INT8, scale S_id) ────> [ Nhân Dyadic (x M_id) >>> E_id ] ─┘      (Scale S_z)
==================================================================================================
```

---

## 8. BẢN ĐỒ ÁNH XẠ TOÁN HỌC SANG PHẦN CỨNG FPGA (KV260 HARDWARE MAPPING)

Bảng đối chiếu trực tiếp giữa công thức giải tích I-ViT và khối vi mạch RTL tương ứng trong thư mục [`Project/src/Hardware/rtl_ver02/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/):

| Toán tử Toán học I-ViT | Phương trình Giải tích | Module Phần cứng RTL | Tài nguyên FPGA AMD Kria KV260 |
|---|---|---|---|
| **Projection GEMM** | $Q, K, V = X \cdot W_{\text{QKV}} + B$ | Thực thi trên ARM PS (`QuantLinear`) | ARM Cortex-A53 CPU NEON SIMD |
| **Attention Scores**| $\text{Raw} = Q \cdot K^T$ (INT32) | `systolic_mac_array.sv` / Stream Engine | DSP48E2 Slices (2x INT8 Packing) |
| **Dyadic Scaling** | $\text{Score} = (\text{Raw} \times M) \ggg E$ | `scale_unit.sv` | 1 bộ nhân 48-bit DSP + Barrel Shifter |
| **Shiftmax (Base-2)**| $q_z' = q_z + (q_z \ggg 1) - (q_z \ggg 4)$ | `softmax_lut.sv` (Tiền tính toán BRAM) | 1 BRAM Block RAMB36K (1024 words) |
| **Reciprocal Div** | $\text{factor} = \lfloor (2^{31}-1) / \sum \text{exp} \rfloor$ | `softmax_lut.sv` (32-step Divider) | LUTs Logic (32 chu kỳ Restoring Divider) |
| **Softmax Norm** | $\text{Prob} = (\text{exp\_int} \times \text{factor}) \ggg 16$ | `softmax_lut.sv` | 1 bộ nhân DSP 32-bit $\times$ 16-bit |
| **Context GEMM** | $\text{Context} = (\text{Prob} \cdot V) \ggg 15$ | `score_v_gemm.sv` | 32 bộ tích lũy Accumulator 32-bit |
| **Residual Addition**| $q_z = (q_x M_x \ggg E_x) + (q_{\text{id}} M_{\text{id}} \ggg E_{\text{id}})$ | `requant_residual_add.sv` (hoặc ARM PS) | 2 bộ nhân DSP + 1 bộ cộng bão hòa |

---

## 9. KẾT LUẬN

Mô hình toán học của **I-ViT** là một bước đột phá toàn diện trong kỹ thuật lượng tử hóa mạng nơ-ron:
1. **Tính khép kín số học (Algebraic Completeness):** Đưa toàn bộ các hàm phi tuyến siêu việt ($\text{Softmax}$, $\text{LayerNorm}$, $\text{GELU}$) về các chuỗi toán tử số nguyên đơn giản: cộng, nhân và dịch bit.
2. **Khớp chính xác phần cứng (Hardware-Friendly):** Loại bỏ hoàn toàn sự phụ thuộc vào các khối dấu phẩy động FPU đắt đỏ, giúp thiết kế RTL [`rtl_ver02`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/) đạt **100% Bit-Exact** với mô hình PyTorch, hoạt động tối ưu tại xung nhịp $200\text{ MHz}$ trên AMD Kria KV260.
