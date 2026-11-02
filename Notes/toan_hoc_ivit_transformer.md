# Mô Hình Toán Học I-ViT Transformer v2.0 (Integer-Only Formulation)

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo mã nguồn I-ViT ICCV 2023)  
> **Tài liệu báo cáo chi tiết:** [`model_notebook/ivit_surgery/REPORT_IVIT_MATHEMATICAL_FOUNDATIONS_AND_FORMULATION.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_IVIT_MATHEMATICAL_FOUNDATIONS_AND_FORMULATION.md)  
> **Mã nguồn tham chiếu:** [`c:/Users/tuan2/Desktop/I-ViT`](file:///c:/Users/tuan2/Desktop/I-ViT)

---

## 1. TỔNG HỢP CÁC CÔNG THỨC TOÁN HỌC CỐT LÕI

### 1.1 Lượng tử hóa đều đối xứng & Đạo hàm STE
- Hệ số tỷ lệ lượng tử hóa:
  $$S = \frac{\max(|X_{\min}|, |X_{\max}|)}{2^{b-1} - 1}$$
- Lượng tử hóa số nguyên:
  $$q_x = \text{clamp}\left( \left\lfloor \frac{x}{S} \right\rceil, -2^{b-1}, 2^{b-1}-1 \right)$$
- Đạo hàm STE (Straight-Through Estimator):
  $$\frac{\partial \lfloor x \rceil}{\partial x} \approx 1, \quad \frac{\partial \mathcal{L}}{\partial x} \approx \frac{1}{S} \frac{\partial \mathcal{L}}{\partial q_x}$$

### 1.2 Phân rã Số học Dyadic (Dyadic Scaling)
- Phân rã tỷ lệ chuyển đổi:
  $$\alpha = \frac{S_{\text{in}}}{S_{\text{out}}} \approx M \times 2^{-E}$$
- Thực thi nhân dịch bit trên phần cứng:
  $$q_{\text{out}} = (q_{\text{in}} \times M) \ggg E$$

---

## 2. SHIFTMAX: TOÁN HỌC SOFTMAX SỐ NGUYÊN

1. **Trừ giá trị cực đại để chống tràn:**
   $$q_z = q_i - q_{\max} \le 0$$
2. **Xấp xỉ hàm mũ qua cơ số 2 với hằng số dịch bit 23/16:**
   $$\log_2 e \approx 1 + \frac{1}{2} - \frac{1}{16} = 1.4375 \quad (\text{sai số } 0.36\%)$$
   $$q_z' = q_z + (q_z \ggg 1) - (q_z \ggg 4)$$
3. **Phân rã phần nguyên $q$ và phần dư $r$:**
   $$x_{0} = \left\lfloor -\frac{1}{S} \right\rfloor, \quad q = \left\lfloor \frac{q_z'}{x_0} \right\rfloor, \quad r = q_z' - x_0 \cdot q$$
   $$\text{exp\_int}(q_z) = \left\lfloor \left( \frac{r}{2} - x_0 \right) \times 2^{15 - q} \right\rfloor$$
4. **Chuẩn hóa xác suất 16-bit:**
   $$\text{factor} = \left\lfloor \frac{2^{31} - 1}{\sum_{j} \text{exp\_int}_j} \right\rfloor$$
   $$\mathbf{\text{prob\_int}_i = (\text{exp\_int}_i \times \text{factor}) \ggg 16} \in [0, 32767]$$

---

## 3. I-LAYERNORM: TOÁN HỌC CHUẨN HÓA LỚP VỚI NEWTON-RAPHSON

1. **Kỳ vọng & Phương sai nguyên:**
   $$\mu_{\text{int}} = \left\lfloor \frac{1}{d} \sum_{i=1}^d q_i \right\rceil, \quad y_i = q_i - \mu_{\text{int}}, \quad \text{var}_{\text{int}} = \sum_{i=1}^d y_i^2$$
2. **Căn bậc hai nguyên bằng Newton-Raphson 10 bước lặp:**
   $$k_{t+1} = \left\lfloor \frac{k_t + \lfloor \text{var}_{\text{int}} / k_t \rfloor}{2} \right\rfloor, \quad k_0 = 65,536 \implies \text{std}_{\text{int}} = k_{10} \approx \sqrt{\text{var}_{\text{int}}}$$
3. **Chuẩn hóa:**
   $$\text{factor} = \left\lfloor \frac{2^{31} - 1}{\text{std}_{\text{int}}} \right\rfloor, \quad y_{\text{norm}, i} = \left\lfloor \frac{y_i \times \text{factor}}{2} \right\rfloor$$

---

## 4. SHIFTGELU & CỘNG RESIDUAL

- **ShiftGELU:**
  $$\text{GELU}(x) \approx x \cdot \sigma(1.702 x) \implies q_{\text{gelu}} = q_x \times \text{sigmoid\_int}$$
- **Phép cộng tắt Residual (Fixed-point Mul):**
  $$q_z = \text{clamp}\left( (q_x M_x \ggg E_x) + (q_{\text{identity}} M_{\text{id}} \ggg E_{\text{id}}), -128, 127 \right)$$
