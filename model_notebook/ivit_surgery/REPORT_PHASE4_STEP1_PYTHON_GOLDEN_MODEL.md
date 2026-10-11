# BÁO CÁO KỸ THUẬT BƯỚC 1: CHUẨN HÓA PYTHON GOLDEN MODEL & ĐỒNG BỘ BIT-EXACT
## Khắc phục Khuyết tật Dữ liệu Kiểm thử cho Bộ Lõi Tăng tốc Attention Core (AMD Kria KV260)

**Mã tài liệu:** `REPORT_PHASE4_STEP1_PYTHON_GOLDEN_MODEL`  
**Thời gian hoàn thành:** 11/10/2026 — 17:35:00 (GMT+7)  
**Tác giả:** Senior AI & SW-HW Co-Design Engineer  
**Căn cứ kỹ thuật:** Section 7 — Remediation Roadmap trong [`REPORT_PHASE4_RTL_COMPREHENSIVE_AUDIT_FLAWS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_RTL_COMPREHENSIVE_AUDIT_FLAWS.md)  
**Trạng thái thực thi:** **HOÀN THÀNH 100% (STEP 1 COMPLETE ✓)**

---

## TỔNG QUAN ĐIỀU HÀNH BƯỚC 1

Trong Bước 1 của lộ trình khắc phục, nhóm kỹ thuật đã xử lý triệt để nguyên nhân gốc rễ gây ra sự sai lệch dữ liệu giữa mô hình PyTorch I-ViT (MobileViT-XXS QAT checkpoint 67.28%) và bộ dữ liệu kiểm thử vàng (Golden Test Vectors).

### Bảng tóm tắt kết quả trước và sau Bước 1:

| Chỉ số / Đặc tính | Hiện trạng cũ trước Bước 1 | Trạng thái chuẩn hóa mới Bước 1 | Đánh giá kỹ thuật |
| :--- | :--- | :--- | :---: |
| **Hệ số Dyadic Scaler Stage 2** | $M=23480, E=26$ ($S_{\text{eff}}=0.000350$) | **$M=18308, E=22$ ($S_{\text{eff}}=0.004365$)** | **Chính xác $100\%$ ($12.5\times$ scale)** |
| **Hệ số Dyadic Scaler Stage 3** | $M=27560, E=26$ ($S_{\text{eff}}=0.000411$) | **$M=22559, E=22$ ($S_{\text{eff}}=0.005378$)** | **Chính xác $100\%$ ($13.1\times$ scale)** |
| **Hệ số Dyadic Scaler Stage 4** | $M=32759, E=27$ ($S_{\text{eff}}=0.000244$) | **$M=19947, E=22$ ($S_{\text{eff}}=0.004756$)** | **Chính xác $100\%$ ($19.5\times$ scale)** |
| **Phân phối Softmax Probabilities**| Suy thoái: **65,536/65,536 mẫu đều là `1`** | **Phân phối động 16-bit:** $[0, 1110]$, Mean=$127.0$, Std=$89.86$ | **Khôi phục hoàn toàn ($581$ mức xác suất)** |
| **Thuật toán Softmax Golden** | Tự chia $/32.0$ thô và dùng float32 exp | **Dùng chính xác toán tử `IntSoftmax` (Shiftmax 16-bit)** của I-ViT | **Bit-Exact chuẩn mực** |
| **Đối soát Context INT32 vs PyTorch**| Không khớp do Softmax là ma trận số 1 | **Max Diff = 0.0 LSB (Khớp chính xác 100%)** | **0-LSB Error Match ✓** |
| **Context Out INT8 Egress** | Đơn thuần là tổng các hàng của $V$ | **Ma trận Attention có trọng số thực sự:** $[-13, 16]$, Std=$6.63$ | **Khôi phục trọn vẹn đặc trưng chú ý** |

---

## 1. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ & CÁC ĐIỀU CHỈNH ĐÃ THỰC HIỆN

### 1.1 Sửa lỗi truy vấn khóa sai cấu trúc mạng (Key Lookup Bug)
* **Nguyên nhân:** Trong script cũ `export_golden_vectors.py`, đường dẫn trích xuất hệ số tỉ lệ được viết là:
  ```python
  qact1_key = f"{layer_pfx}.qact1"  # SAI: Trỏ nhầm vào QuantAct ngoài transformer block
  ```
  Trong khi đó, tầng Self-Attention thực tế nằm trong cấu trúc lồng `layer_pfx.attn.qact1`.
* **Khắc phục:** 
  Cập nhật truy cập trực tiếp vào đối tượng submodule của PyTorch:
  ```python
  attn_mod = model.mobilevit.encoder.layer[2].transformer.layer[0].attn
  sq = attn_mod.qact1.act_scaling_factor.item()       # 0.039402 (chính xác)
  s_score = attn_mod.qact_attn1.act_scaling_factor.item() # 0.088917 (chính xác)
  ```
* **Kết quả:** Hệ số tỷ lệ hiệu dụng $S_{\text{effective}}$ của Stage 2 được khôi phục từ giá trị sai $0.000350$ về giá trị đúng **$0.004365$** (tăng $12.5\times$, phản ánh đúng dải động học của ma trận sau phép nhân vô hướng $Q \cdot K^T$).

---

### 1.2 Phân rã chuẩn xác cặp Dyadic Scale $(M, E)$ cho cả 3 Stage
Phép biến đổi dyadic tuân thủ chặt chẽ nguyên lý:
$$S_{\text{effective}} = \frac{S_Q^2}{S_{\text{score}} \cdot \sqrt{d_k}} \approx M_{\text{scale}} \cdot 2^{-E_{\text{scale}}}$$

Áp dụng hàm `decompose_dyadic_16bit` (với mantissa 15-bit số nguyên có dấu $[0, 32767]$):

| Transformer Stage | Chiều Head ($d_k$) | $S_Q$ (QKV Projection) | $S_{\text{score}}$ (Attn Score) | $S_{\text{effective}}$ Lý thuyết | $M_{\text{scale}}$ | $E_{\text{scale}}$ | Sai số xấp xỉ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 2** (Layer 2) | **16** | `0.039402` | `0.088917` | `0.004365` | **18,308** | **22** | $4 \times 10^{-8}$ |
| **Stage 3** (Layer 3) | **20** | `0.031975` | `0.042506` | `0.005378` | **22,559** | **22** | $5 \times 10^{-8}$ |
| **Stage 4** (Layer 4) | **24** | `0.025509` | `0.027931` | `0.004756` | **19,947** | **22** | $6 \times 10^{-8}$ |

Cả 3 bộ tham số đã được xuất đồng bộ vào:
1. [`model_notebook/ivit_surgery/golden_vectors/hw_scale_params.svh`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/golden_vectors/hw_scale_params.svh)
2. [`Project/src/Hardware/rtl_ver02/hw_scale_params.svh`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/hw_scale_params.svh)
3. [`dyadic_scale_config.json`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/dyadic_scale_config.json)

---

### 1.3 Khôi phục thuật toán Softmax từ PyTorch I-ViT `IntSoftmax(16)`
* **Nguyên nhân suy thoái trước đây:** Script cũ ép kiểu xác suất Softmax float về UINT8 $[0, 255]$ bằng công thức `round(prob_float * 255.0)`. Với $N=256$, xác suất trung bình là $1/256 \approx 0.0039$. Nhân với $255$ cho kết quả $0.996 \approx 1$. Toàn bộ 65,536 ô đều bị làm tròn thành số 1.
* **Giải pháp chuẩn hóa:**
  1. Trích xuất trực tiếp tensor từ toán tử `module.int_softmax` trong tầng Transformer của PyTorch.
  2. Theo thiết kế chuẩn của I-ViT, `IntSoftmax` tích lũy với độ phân giải **16-bit** (`output_bit = 16`), với scale $S_{\text{sm}} = 1/2^{15} = 1/32768$.
  3. Giá trị nguyên của xác suất được biểu diễn dưới dạng số nguyên 16-bit $[0, 32767]$:
     $$\text{Softmax}_{\text{int16}}[i, j] = \text{round}\left( \frac{\text{attn\_sm}[i, j]}{S_{\text{sm}}} \right)$$
  4. Tổng xác suất của mỗi hàng token $i$ đạt xấp xỉ $\sum_{j=0}^{255} \text{Softmax}_{\text{int16}}[i, j] \approx 32,457 \dots 32,768$ (rất sát với $2^{15} = 32,768$).
  5. Xuất thêm tệp `golden_softmax_prob_u8.hex` với phân phối xác suất đã chuẩn hóa tỷ lệ cho các tùy chọn kiểm thử phần cứng 8-bit.

---

### 1.4 Xác thực 0-LSB Bit-Exact cho khối Context Output ($S \times V$)
* Trong PyTorch, tầng Self-Attention thực thi:
  $$\text{out\_ctx} = \text{matmul\_2}(\text{attn\_sm}, S_{\text{sm}}, V, S_V)$$
* Trong mô hình số nguyên:
  $$\text{context\_int32} = \sum_{k=0}^{255} \text{Softmax}_{\text{int16}}[i, k] \times V_{\text{int8}}[k, d]$$
* **Kết quả đối soát giữa Python Numpy và PyTorch Native Tensor:**
  $$\max | \text{context\_int32} - \text{round}(\text{out\_ctx} / (S_{\text{sm}} \cdot S_V)) | = \mathbf{0.0\text{ LSB}}$$
  **Tỷ lệ khớp bit-exact đạt 100.00% (Zero LSB Error).**
* Khi đưa ra dòng Egress 8-bit AXI-Stream, giá trị được chuẩn hóa bằng phép dịch phải số học 15-bit (tương ứng với tổng trọng số $2^{15}$ của Softmax):
  $$\text{context\_out\_int8} = \text{clamp}_{\text{INT8}}(\text{context\_int32} \ggg 15)$$

---

## 2. BẢNG THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU KIỂM THỬ VÀNG (GOLDEN VECTORS) MỚI

Toàn bộ 8 tệp vector kiểm thử vàng đã được tạo mới hoàn toàn và ghi đúp đồng thời vào 2 thư mục:
- `model_notebook/ivit_surgery/golden_vectors/`
- `Project/src/Hardware/rtl_ver02/`

| Tên tệp vector (.hex) | Kích thước mảng | Kiểu dữ liệu | Min | Max | Mean | Độ lệch chuẩn (Std) | Ý nghĩa chức năng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| [`q_matrix.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/q_matrix.hex) | $256 \times 16$ | INT8 (signed) | `-128` | `127` | `-0.21` | `43.15` | Ma trận truy vấn $Q$ (Head 0) |
| [`k_matrix.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/k_matrix.hex) | $256 \times 16$ | INT8 (signed) | `-128` | `127` | `0.45` | `42.88` | Ma trận khóa $K$ (Head 0) |
| [`v_matrix.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/v_matrix.hex) | $256 \times 16$ | INT8 (signed) | `-128` | `127` | `-0.12` | `43.02` | Ma trận giá trị $V$ (Head 0) |
| [`golden_gemm_qk.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_gemm_qk.hex) | $256 \times 256$ | INT32 (signed) | `-3,492` | `+20,056`| `3,115` | `2,777.07` | Tích ma trận thô $Q \cdot K^T$ |
| [`golden_scaled_score.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_scaled_score.hex) | $256 \times 256$ | INT16 (signed) | `-16` | `+87` | `+13.6` | `12.12` | Điểm số sau Dyadic Scaler |
| [`golden_softmax_prob.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_softmax_prob.hex) | $256 \times 256$ | UINT16 | `0` | `1,110` | `127.0` | `89.86` | Xác suất Shiftmax 16-bit I-ViT |
| [`golden_softmax_prob_u8.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_softmax_prob_u8.hex)| $256 \times 256$ | UINT8 | `0` | `18` | `1.0` | `1.45` | Bản xác suất 8-bit chuẩn hóa |
| [`golden_context_out.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_context_out.hex) | $256 \times 16$ | INT8 (signed) | `-13` | `+16` | `0.18` | `6.63` | Vector Attention Context đầu ra |

### Biểu đồ kiểm tra dải động học hàng đầu tiên (Row 0):
* `golden_scaled_score.hex` (10 phần tử đầu): `[-14, 17, 8, 11, 16, 13, 16, 10, 25, 10]`
* `golden_softmax_prob.hex` (10 phần tử đầu): `[2, 37, 16, 23, 32, 25, 34, 22, 69, 22]`
* `golden_context_out.hex` (8 phần tử đầu): `[8, 4, 0, -4, -10, 6, -8, 0]`

---

## 3. CÁC TỆP ĐÃ ĐƯỢC CẬP NHẬT TRONG BƯỚC 1

1. **[`export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py):** Viết lại hoàn chỉnh toàn bộ quy trình trích xuất hook mô hình, phân rã dyadic chính xác, tạo vector 16-bit và 8-bit với đối soát sai số tự động.
2. **[`hw_scale_params.svh`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/hw_scale_params.svh):** Cập nhật macro phần cứng mới ($M=18308, E=22$ cho Stage 2; $M=22559, E=22$ cho Stage 3; $M=19947, E=22$ cho Stage 4).
3. **[`dyadic_scale_config.json`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/dyadic_scale_config.json):** Lưu trữ cấu hình JSON chuẩn phục vụ driver PYNQ trên ARM CPU.
4. **Bộ 8 tệp `.hex`:** Đồng bộ mới hoàn toàn trong cả hai thư mục `golden_vectors/` và `rtl_ver02/`.

---

## 4. KẾT LUẬN & BÀN GIAO SẴN SÀNG CHO BƯỚC 2

> [!NOTE]
> **ĐÁNH GIÁ NGHIỆM THU BƯỚC 1:**
> * Tất cả các lỗi về dữ liệu kiểm thử vàng (suy thoái thành số 1, sai lệch dyadic scale 12x, mất mát dải động học Attention) **đã được giải quyết triệt để 100%**.
> * Bộ vector kiểm thử vàng hiện tại là **chuẩn mực toán học tối cao (Ground Truth)**, hoàn toàn bit-exact với mô hình PyTorch MobileViT-XXS I-ViT.
> * Cơ sở dữ liệu kiểm thử vàng đã sẵn sàng để làm mốc đối soát chính xác cho quá trình sửa chữa mã nguồn RTL phần cứng trong Bước 2.

### Kế hoạch hành động tiếp theo — BƯỚC 2:
1. **Sửa lỗi số học khối Softmax ([`softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/softmax_lut.sv)):**
   - Thiết kế lại bộ chia Fixed-point: hỗ trợ định dạng số nguyên 16-bit khớp với `golden_softmax_prob.hex`.
   - Mở rộng tử số bộ chia để thương số không bao giờ bị cắt cụt về 0.
2. **Sửa lỗi mất mát dữ liệu Ping-Pong BRAM Buffer ([`ping_pong_bram_buffer.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/ping_pong_bram_buffer.sv)):**
   - Đóng gói đầy đủ 64-bit từ bus AXI4-Stream vào BRAM (không vứt bỏ 32 bit cao).
   - Thiết kế FSM tự động chuyển đổi mục tiêu lưu trữ $Q \rightarrow K \rightarrow V$.
3. Chạy kiểm thử độc lập cho từng module bằng simulator trước khi tích hợp vào Top-level.

---
*Báo cáo được lưu trữ chính thức tại `model_notebook/ivit_surgery/REPORT_PHASE4_STEP1_PYTHON_GOLDEN_MODEL.md` phục vụ nghiệm thu tiến độ đồ án.*
