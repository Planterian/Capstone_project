# GIẢI TRÌNH KỸ THUẬT: TẠI SAO PHẢI THAY THẾ BẢNG LOOKUP TABLE TRONG SOFTMAX ENGINE?
## Phân tích Chuyên sâu về Cơ chế Đồng thiết kế Thuật toán & Phần cứng (SW-HW Co-Design) cho MobileViT-XXS trên AMD Kria KV260

**Mã tài liệu:** `EXPLANATION_SOFTMAX_LUT_UPGRADE`  
**Ngày khởi tạo:** 13/10/2026 (Cập nhật bổ sung giải trình nguồn gốc dữ liệu)  
**Vị trí lưu trữ:** [`model_notebook/ivit_surgery/EXPLANATION_SOFTMAX_LUT_UPGRADE.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/EXPLANATION_SOFTMAX_LUT_UPGRADE.md)  
**Tài liệu tham chiếu:**
1. Báo cáo nghiệm thu Bước 2: [`REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md)
2. Kế hoạch thực thi Phase 4: [`PLAN_PHASE4_REMAINING_STEPS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md)
3. Mã nguồn RTL Softmax: [`Project/src/Hardware/rtl_ver02/softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/softmax_lut.sv)
4. Kịch bản trích xuất vector: [`model_notebook/ivit_surgery/export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py)

---

## TỔNG QUAN VẤN ĐỀ

Trong mã nguồn RTL ban đầu (`rtl_ver02` phiên bản cũ), khối `softmax_lut.sv` sử dụng một bảng tra cứu tĩnh 16-bit gồm 256 phần tử (`exp_rom[0:255]`) với giá trị đỉnh là `65535`. 

Tuy nhiên, trong Bước 2, nhóm kỹ thuật đã **thay thế hoàn toàn** bảng này bằng bảng **19-bit `exp_rom` mới** (giá trị đỉnh là `262144`) kết hợp cùng bộ chia lặp 32-step Restoring Divider với tử số $2^{31} - 1$.

```
+---------------------------------------------------------------------------------------------------------+
|                                    BẢN CHẤT CỦA SỰ THAY THẾ                                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [BẢNG CŨ (HEURISTIC)]                                                                                 |
|   ROM_old[d] = round( exp(-d / 32.0) * 65535 )                                                          |
|       - Tự gán tỉ lệ chia 32.0, không có cơ sở toán học từ mạng AI.                                    |
|       - Hệ số suy giảm lệch 4.08 lần so với thực tế -> Phân phối chú ý bị san phẳng.                   |
|       - Đầu ra chia sai dải động học làm xác suất triệt tiêu về 0 tuyệt đối.                           |
|                                                                                                         |
|                                         ▼ THAY THẾ HOÀN TOÀN THÀNH ▼                                    |
|                                                                                                         |
|   [BẢNG MỚI (I-ViT SHIFTMAX CHUẨN XÁC)]                                                                 |
|   ROM_new[d] = int_exp_shift(-d, S_score = 0.1275113, n = 15)                                            |
|       - Khớp từng bit với toán tử IntSoftmax trong PyTorch (MobileViT-XXS QAT 67.28%).                  |
|       - Bám sát thang đo hiệu chuẩn S_score của Stage 2.                                                 |
|       - Dải động 19-bit (đỉnh 262,144) tương thích hoàn hảo với bộ chia (2^31 - 1).                     |
|       - Đạt 100% BIT-EXACT (0-LSB Error) trên toàn bộ 65,536 tokens của ma trận.                        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 1. NGUỒN GỐC DỮ LIỆU: TỪNG CON SỐ TRONG `exp_rom` ĐƯỢC TẠO RA NHƯ THẾ NÀO?

Một thắc mắc tự nhiên rất phổ biến khi đọc code RTL là:  
> *"256 con số trong `exp_rom` từ đâu chui ra? Ai tính ra chúng? Và tại sao lại để sẵn cố định trong phần cứng?"*

### 1.1 Tại sao có thể tính sẵn ngoại tuyến (Offline Pre-computation)?
Trong cơ chế Attention Softmax, với mỗi token $i$, ta tìm giá trị điểm lớn nhất trong hàng:
$$s_{\max} = \max_{j=0}^{255} (\text{score}[j])$$

Sau đó trừ đi $s_{\max}$:
$$\Delta s = \text{score}[i] - s_{\max}$$

* Do đã trừ đi $s_{\max}$, hiệu số $\Delta s$ **luôn luôn là số nguyên âm hoặc bằng 0** ($\Delta s \le 0$).
* Khi chuyển thành độ lệch dương $d = -\Delta s$:
  $$d \in [0, 1, 2, 3, \dots, 255]$$

> [!NOTE]
> **Nhận xét then chốt:** Biến đầu vào $d$ chỉ có đúng **256 giá trị nguyên có thể xảy ra**!  
> Thay vì thiết kế một mạch tính hàm mũ phức tạp, tốn kém hàng trăm DSP và chu kỳ xung nhịp trên chip FPGA, ta **tính toán trước toàn bộ 256 giá trị này bằng Python trên máy tính**, sau đó nạp sẵn vào 1 khối BRAM nhỏ (chỉ tốn nửa lát RAMB18K trên AMD Kria KV260).  
> Khi FPGA chạy, với bất kỳ độ lệch $d$ nào, phần cứng chỉ cần dùng $d$ làm địa chỉ đọc BRAM để lấy ra kết quả ngay lập tức trong **đúng 1 chu kỳ xung nhịp ($5\text{ ns}$)**!

---

### 1.2 Thuật toán 5 bước sinh ra từng giá trị trong bảng (I-ViT Shiftmax)
Từng con số trong bảng được sinh ra từ chính toán tử `IntSoftmax` của thư viện lượng tử hóa I-ViT trong PyTorch:

* **Bước 1: Lấy hệ số thang đo hiệu chuẩn của Stage 2:**
  $$S_{\text{score}} = 0.12751130759716034$$
* **Bước 2: Xác định điểm mốc cơ sở $x_0$:**
  $$x_0 = \left\lfloor -\frac{1}{S_{\text{score}}} \right\rfloor = \left\lfloor -\frac{1}{0.1275113} \right\rfloor = \mathbf{-8}$$
* **Bước 3: Hiệu chỉnh xấp xỉ số nguyên đa thức (I-ViT Polynomial):**  
  Với mỗi độ lệch $d$ ($x = -d$):
  $$x_{\text{int}} = x + \left\lfloor \frac{x}{2} \right\rfloor - \left\lfloor \frac{x}{16} \right\rfloor$$
* **Bước 4: Phân rã thành thương số $q$ và phần dư $r$:**
  $$q = \left\lfloor \frac{x_{\text{int}}}{x_0} \right\rfloor, \quad r = x_{\text{int}} - x_0 \times q$$
* **Bước 5: Tính số mũ dạng dịch bit nhị phân (Độ phân giải $n = 15$ bit):**
  $$\text{exp\_int} = \left( \frac{r}{2} - x_0 \right) \times 2^{15 - q}$$
  *(Nếu $\text{exp\_int} < 8$, kẹp cố định tại giá trị sàn là $8$ theo chuẩn I-ViT).*

---

### 1.3 Ví dụ "tính tay" cụ thể cho từng trường hợp điển hình

#### • Trường hợp 1: $d = 0$ (Phần tử đạt giá trị Max, $\Delta s = 0$)
* $x = 0 \implies x_{\text{int}} = 0$
* $q = \lfloor 0 / (-8) \rfloor = 0$
* $r = 0 - (-8) \times 0 = 0$
* $\text{exp} = \left(\frac{0}{2} - (-8)\right) \times 2^{15 - 0} = 8 \times 32,768 = \mathbf{262,144} = 2^{18}$
* 👉 **`exp_rom[0] = 19'd262144;`** (Đỉnh cực đại của bảng!)

#### • Trường hợp 2: $d = 1$ (Điểm số thấp hơn Max 1 đơn vị, $\Delta s = -1$)
* $x = -1 \implies x_{\text{int}} = -1 + \lfloor -0.5 \rfloor - \lfloor -0.0625 \rfloor = -1 + (-1) - (-1) = -1$
* $q = \lfloor -1 / (-8) \rfloor = 0$
* $r = -1 - (-8) \times 0 = -1$
* $\text{exp} = \left(\frac{-1}{2} - (-8)\right) \times 2^{15} = 7.5 \times 32,768 = \mathbf{245,760}$
* 👉 **`exp_rom[1] = 19'd245760;`**

#### • Trường hợp 3: $d = 11$ (Điểm số thấp hơn Max 11 đơn vị)
* Chạy qua các bước tương tự $\implies \text{exp} = \mathbf{65,536} = 2^{16}$
* 👉 **`exp_rom[11] = 19'd65536;`**

#### • Trường hợp 4: $d \ge 85$ (Điểm số thấp hơn Max từ 85 đơn vị trở lên)
* Điểm số quá nhỏ so với Max, xác suất giảm sát về 0. Thuật toán I-ViT quy định giá trị đáy tối thiểu (floor clamping) là $8$.
* 👉 **`exp_rom[85 ... 255] = 19'd8;`**

---

### 1.4 Đoạn mã Python tự động sinh ra bảng cho SystemVerilog

Trong tệp [`model_notebook/ivit_surgery/export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py#L240-L263), đoạn mã sau đã tự động tính toán và in ra toàn bộ mã RTL cho `softmax_lut.sv`:

```python
import numpy as np

def int_exp_shift(x_int, sf, n=15):
    """Thuật toán xấp xỉ số mũ nhị phân I-ViT (Kim et al., 2021)"""
    x_int = x_int + np.floor(x_int / 2.0) - np.floor(x_int / 16.0)
    x0_int = np.floor(-1.0 / sf)                     # Với sf = 0.1275113 -> x0_int = -8
    x_int = np.maximum(x_int, n * x0_int)
    q = np.floor(x_int / x0_int)                     # Thương số phân rã
    r = x_int - x0_int * q                           # Phần dư
    exp_int = r / 2.0 - x0_int
    exp_int = np.maximum(np.floor(exp_int * (2.0 ** (n - q))), 0) # Dịch bit nhị phân
    return int(exp_int)

# Sinh bảng 256 phần tử cho phần cứng:
s_score = 0.12751130759716034
exp_rom_table = [int_exp_shift(-d, s_score, n=15) for d in range(256)]

# In cú pháp SystemVerilog nạp vào softmax_lut.sv:
print("    initial begin")
for d in range(85):
    print(f"        exp_rom[{d:3d}] = 19'd{exp_rom_table[d]};")
print("        for (int idx = 85; idx < 256; idx++) begin")
print("            exp_rom[idx] = 19'd8;")
print("        end")
print("    end")
```

---

## 2. TẠI SAO BẢNG CŨ LẠI SAI?

### 2.1 Bảng cũ là công thức tự chế (Heuristic), không có căn cứ từ mô hình
Khi kiểm tra lại công thức sinh bảng trong commit ban đầu của dự án:
$$\text{ROM\_old}[d] = \text{round}\left(e^{-\frac{d}{32.0}} \times 65535\right)$$

Các giá trị cụ thể trong bảng cũ:
* $d = 0 \implies \text{round}(e^{0} \times 65535) = \mathbf{65535}$
* $d = 1 \implies \text{round}(e^{-1/32} \times 65535) = \mathbf{63519}$
* $\dots$
* $d = 255 \implies \text{round}(e^{-255/32} \times 65535) = \mathbf{23}$

> [!WARNING]
> **Con số chia $32.0$ này là một con số "ước lượng hú họa" (Heuristic).**  
> Người thiết kế ban đầu đã giả định bừa rằng điểm Attention Score sau khi trừ Max sẽ nằm trong khoảng $[0, 32]$, nên chia cho $32.0$ để đưa về dải $[0, 1]$ trước khi lấy hàm mũ float. Con số này **hoàn toàn không liên quan** đến mô hình MobileViT-XXS đã huấn luyện.

---

### 2.2 Lệch tỷ lệ suy giảm thang đo Attention tới 4.08 lần ($S_{\text{score}}$)

Trong cơ chế Self-Attention lượng tử hóa:
$$\Delta x = \Delta s \times S_{\text{score}}$$

* **Mô hình MobileViT-XXS Stage 2 thực tế:**
  $$S_{\text{score}} = \mathbf{0.1275113} \implies \Delta x = 1 \times 0.1275113 = \mathbf{0.1275}$$
* **Bảng LUT cũ giả định:**
  $$\text{Scale}_{\text{old}} = \frac{1}{32.0} = \mathbf{0.03125}$$

$$\text{Tỷ số lệch} = \frac{0.1275113}{0.03125} \approx \mathbf{4.08\text{ lần!}}$$

**Hậu quả:**  
* Bảng cũ suy giảm quá chậm ($e^{-0.03125} \approx 0.9692$ vs $e^{-0.1275} \approx 0.8803$).
* Nó khiến tất cả các token đều nhận được xác suất gần bằng nhau. Phân phối Softmax bị **"san phẳng" (flattened distribution)**, phá hủy hoàn toàn khả năng tập trung chú ý vào các vùng đặc trưng quan trọng của hình ảnh.

---

## 3. TẠI SAO BẮT BUỘC PHẢI DÙNG 19-BIT VÀ BỘ CHIA $2^{31}-1$?

### 3.1 Dải động học 19-bit bảo toàn 3 bit MSB
* Như đã tính ở Mục 1.3: tại đỉnh $d = 0$, giá trị mũ là:
  $$\text{exp\_int} = 8 \times 2^{15} = \mathbf{262,144} = 2^{18}$$
* Để biểu diễn được $262,144$, bắt buộc phải dùng **19 bit** ($2^{18} \le 262,144 < 2^{19}$).
* Bảng cũ chỉ dùng 16 bit (tối đa $65,535$), bị mất hẳn **3 bit trọng số cao nhất (MSB)**, làm méo mó dải động học ngay từ đầu vào.

### 3.2 Tương thích hoàn hảo với Bộ chia 32-step Restoring Divider
Để thực hiện phép chia chuẩn hóa mà không làm mất độ chính xác dấu phẩy tĩnh, I-ViT quy định:
1. **Tử số chuẩn hóa:**
   $$\text{Numerator} = 2^{31} - 1 = 32'\text{h7FFF\_FFFF} = 2,147,483,647$$
2. **Mẫu số tích lũy:**
   $$\text{sum\_exp} = \sum_{i=0}^{255} \text{exp\_rom}[i] \approx 5,000,000 \dots 16,000,000$$
3. **Thương số 32-bit:**
   $$\text{inv\_sum} = \left\lfloor \frac{2^{31} - 1}{\text{sum\_exp}} \right\rfloor \approx 130 \dots 430$$
   *(Khác hoàn toàn với bộ chia cũ bị ép thương số về bằng 1 do $(255 \ll 16) / \text{sum\_exp} = 1$).*
4. **Chuẩn hóa xác suất 16-bit:**
   $$\text{out\_prob} = \frac{\text{rom\_data\_exp} \times \text{inv\_sum}}{2^{16}}$$

Khi nhân giá trị mũ $[8, 262144]$ với $\text{inv\_sum} \approx 130 \dots 430$ rồi dịch phải 16 bit, kết quả thu được là một số nguyên 16-bit nằm trong dải $[0, 32767]$ với tổng cả hàng xấp xỉ $32,768$ (tương ứng với $100\%$ xác suất).

---

## 4. BẢNG SO SÁNH TỔNG HỢP TOÀN DIỆN

| Tiêu chí kỹ thuật | Bảng LUT ban đầu (`exp_rom_old`) | Bảng LUT nâng cấp mới (`exp_rom_new` v2.2) |
| :--- | :--- | :--- |
| **Công thức toán học** | $e^{-d / 32.0} \times 65535$ (Heuristic cảm tính) | `int_exp_shift(-d, S_score, n=15)` (I-ViT chuẩn) |
| **Hệ số thang đo áp dụng** | Gán cứng $1/32 = 0.03125$ (sai lệch $4.08\times$) | **$S_{\text{score}} = 0.1275113$ (Chính xác từ Checkpoint)** |
| **Độ rộng từ BRAM** | 16-bit (Đỉnh $65,535$) | **19-bit (Đỉnh $262,144 = 2^{18}$)** |
| **Đặc tính suy giảm số mũ** | Suy giảm quá chậm $\implies$ Xác suất bị san phẳng | Suy giảm dốc chuẩn mực $\implies$ Giữ trọn tính chú ý |
| **Tương thích bộ chia** | Gây nghẽn thương số $= 1$, đầu ra triệt tiêu về 0 | Khớp chuẩn với Restoring Divider $(2^{31}-1)$ |
| **Độ phân giải đầu ra** | 8-bit (sai lệch hoàn toàn dải I-ViT) | **16-bit (Khớp hoàn hảo với `golden_softmax_prob.hex`)** |
| **Sai số đối soát vs PyTorch**| Không thể khớp ($> 100$ LSB) | **Max Error = 0 LSB (100% Bit-Exact trên 65,536 tokens)** |

---

## KẾT LUẬN

Việc thay thế bảng Lookup Table trong `softmax_lut.sv` không đơn thuần là một chỉnh sửa mã nguồn cục bộ, mà là một bước **chuẩn hóa toán học bắt buộc** nhằm:
1. Đưa kiến trúc phần cứng bám sát thuật toán định lượng **I-ViT Shiftmax** của mô hình MobileViT-XXS.
2. Xóa bỏ triệt để lỗi số học làm đầu ra Softmax bằng 0.
3. Đảm bảo mô hình khi nạp lên bo mạch AMD Kria KV260 hoạt động ổn định và giữ nguyên độ chính xác nhận dạng **67.28%** mà không bị suy hao chất lượng.

---
*Tài liệu giải trình kỹ thuật được lưu trữ chính thức tại `model_notebook/ivit_surgery/EXPLANATION_SOFTMAX_LUT_UPGRADE.md` phục vụ hội đồng nghiệm thu đồ án Capstone.*
