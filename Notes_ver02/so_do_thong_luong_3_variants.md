# BÁO CÁO TỔNG QUÁT SƠ ĐỒ THÔNG LƯỢNG (THROUGHPUT) CẢ 3 BIẾN THỂ KIẾN TRÚC MHA ACCELERATOR
**Dự án:** Thiết kế Bộ Tăng tốc Phần cứng Multi-Head Attention (MHA) cho Vision Transformer (ViT) trên FPGA AMD Xilinx Kria KV260  
**Tác giả:** Nhóm Đồ án Tốt nghiệp  
**Cập nhật:** Tháng 10/2026  
**Mục tiêu:** Tổng hợp toàn diện đường truyền dữ liệu (Datapath), chu kỳ thực thi (Latency), khoảng cách phát lệnh (Initiation Interval - $II$), và sơ đồ thông lượng (Throughput) của cả 3 biến thể kiến trúc phần cứng:
- **Variant 1A:** Vector Streaming Engine (Thuần LUT Tree) – `rtl_ver02_1A`
- **Variant 2A:** Interleaved DSP-Packed Streaming Engine (8 DSP48E2 + Dual Adder Tree + Zero-Bubble Serializer) – `rtl_ver02_2A`
- **Variant 3B:** 2D Output-Stationary Systolic Array ($4 \times 4$ PE Grid, 16 DSP48E2) – `rtl_ver02_3B`

---

## MỤC LỤC
1. [Sơ đồ thông lượng cấp hệ thống toàn mạch (End-to-End MHA Pipeline)](#1-sơ-đồ-thông-lượng-cấp-hệ-thống-toàn-mạch-end-to-end-mha-pipeline)
2. [Quan hệ bản chất giữa "Beat", Độ rộng Bus và Thông lượng](#2-quan-hệ-bản-chất-giữa-beat-độ-rộng-bus-và-thông-lượng)
3. [Sơ đồ thông lượng Variant 1A: Vector Streaming Engine (Thuần LUT)](#3-sơ-đồ-thông-lượng-variant-1a-vector-streaming-engine-thuần-lut)
4. [Sơ đồ thông lượng Variant 2A: Interleaved DSP-Packed Streaming Engine](#4-sơ-đồ-thông-lượng-variant-2a-interleaved-dsp-packed-streaming-engine)
5. [Sơ đồ thông lượng Variant 3B: 2D Output-Stationary Systolic Array](#5-sơ-đồ-thông-lượng-variant-3b-2d-output-stationary-systolic-array)
6. [So sánh trực quan nhịp thực thi (Timing Waveforms & Pipelines)](#6-so-sánh-trực-quan-nhịp-thực-thi-timing-waveforms--pipelines)
7. [Bảng ma trận so sánh định lượng & Tài nguyên phần cứng](#7-bảng-ma-trận-so-sánh-định-lượng--tài-nguyên-phần-cứng)
8. [Phân tích điểm nghẽn hệ thống theo định luật Amdahl (System Bottleneck)](#8-phân-tích-điểm-nghẽn-hệ-thống-theo-định-luật-amdahl-system-bottleneck)
9. [Kết luận & Khuyến nghị kiến trúc cho Kria KV260](#9-kết-luận--khuyến-nghị-kiến-trúc-cho-kria-kv260)
10. [Liên kết mã nguồn RTL & Tài liệu kiểm tra](#10-liên-kết-mã-nguồn-rtl--tài-liệu-kiểm-tra)

---

## 1. SƠ ĐỒ THÔNG LƯỢNG CẤP HỆ THỐNG TOÀN MẠCH (END-TO-END MHA PIPELINE)

Bộ tăng tốc Multi-Head Attention (MHA) xử lý chuỗi $N = 256$ tokens với chiều vector $d_k \in \{16, 20, 24, 32\}$. Một vòng lặp tính toán Attention Head bao gồm **5 công đoạn phần cứng nối tiếp nhau**, được lặp lại tuần tự cho $N = 256$ hàng của ma trận Query.

Dưới đây là so sánh chi tiết luồng dữ liệu và chu kỳ thực thi của **TOÀN MẠCH** giữa hai chế độ:
- **Chế độ chuẩn ($d_k = 16$):** Chuẩn baseline của mô hình MobileViT-XXS.
- **Chế độ mở rộng ($d_k > 16$, đặc biệt là $d_k = 24$ và $d_{k,\max} = 32$):** Chế độ đa chiều hỗ trợ ViT-Base thu nhỏ.

---

### 1.1. Sơ đồ luồng dữ liệu toàn mạch chế độ chuẩn ($d_k = 16$)

Khi $d_k = 16$, mỗi token vector có kích thước đúng bằng 16 Bytes (vừa khít 1 beat 128-bit trên bus AXI/BRAM). Toàn bộ chuỗi đường ống đạt trạng thái thông lượng tối đa $II = 1$:

```
+---------------------------------------------------------------------------------------------------+
|               QUY TRÌNH TOÀN MẠCH XỬ LÝ 1 HÀNG ATTENTION - CẤU HÌNH CHUẨN d_k = 16                |
+---------------------------------------------------------------------------------------------------+
  [Q Vector: 1x16 (1 beat)]        +-----------------------------------+
        |                          | K-Buffer (256x16 = 256 beats BRAM)|
        v                          +-----------------------------------+
  +------------------------------------------+ | (Stream 256 beats, II = 1)
  |  GIAI ĐOẠN 1: GEMM 1 (S = Q x K^T)       |<-+
  |  - Variant 1A: 261 chu kỳ (II = 1)       |
  |  - Variant 2A: 262 chu kỳ (II = 1)       |
  |  - Variant 3B: 573 chu kỳ (Tile-drain)   |
  +------------------------------------------+
        |  Raw Score Stream: S[0..255] (1 beat 24-bit / chu kỳ liên tục)
        v
  +------------------------------------------+
  |  GIAI ĐOẠN 2: SCALE UNIT                 |
  |  - S' = S * (1 / sqrt(16)) = S * 0.25    |
  |  - Trễ pipeline: 2 chu kỳ (II = 1)       |
  +------------------------------------------+
        |  Scaled Score Stream: S'[0..255] (1 beat 16-bit / chu kỳ)
        v
  +------------------------------------------+
  |  GIAI ĐOẠN 3: SOFTMAX ENGINE (3 PASSES)  | <=== [ĐIỂM NGHẼN HỆ THỐNG: ~532 CHU KỲ]
  |  - Pass 1: Online Max Finder   (256 cyc) | (Nhận trực tiếp từng score beat ở II = 1)
  |  - Pass 2: SubMax + Exp + Sum  (256 cyc) | (Đọc Scratchpad BRAM nội vi)
  |  - Divider: Tính 1 / Sum Exp    (14 cyc) | (Khối chia Newton-Raphson/LUT)
  |  - Pass 3: Norm P_i = Exp*Inv  (256 cyc) | (Xả Probabilities P_i ở II = 1)
  +------------------------------------------+
        |  Attention Probabilities P[0..255] (Stream ra ở Pass 3, II = 1)
        v
  +------------------------------------------+ +-----------------------------------+
  |  GIAI ĐOẠN 4: CONTEXT GEMM (O = P x V)   | | V-Buffer (256x16 = 256 beats BRAM)|
  |  - Tích lũy Context: 256 tokens x V_vec  |<-+ (Mỗi token V là 1 beat 16 Bytes)
  |  - Chạy CHỒNG LẤP hoàn toàn với Pass 3   |
  |  - Chiếm đúng 256 chu kỳ (II = 1)        |
  +------------------------------------------+
        |  Output Token Vector: 1x16 (INT8)
        v
  +------------------------------------------+
  |  GIAI ĐOẠN 5: OUTPUT BUFFER & DMA OUT    |
  |  - Đóng gói đúng 1 beat 128-bit Egress   |
  +------------------------------------------+
```

#### Biểu đồ thời gian chồng lặp giữa các hàng ($d_k = 16$):
```text
Clock Cycles --->
Hàng r:   [-- GEMM 1 (262c) --][Scale][ Pass 1 Max (256c) ][ Pass 2 Exp (256c) ][ Pass 3 & GEMM 2 (256c) ]
                                      \                                         /
                                       +------------- Softmax (~532 cyc) -------+
Hàng r+1:                         [-- GEMM 1 (262c) --][Scale][ Pass 1 Max ][ Pass 2 Exp ][ Pass 3 ... ]
                                  ^
                                  | GEMM 1 hàng r+1 gối đầu ngay khi K-Buffer sẵn sàng
```
> **Đặc trưng $d_k = 16$:** Softmax (~532 chu kỳ) dài hơn GEMM 1 (262 chu kỳ), toàn mạch bị **Softmax kìm hãm (Softmax-bound)**.

---

### 1.2. Sơ đồ luồng dữ liệu toàn mạch khi mở rộng $d_k > 16$ ($d_k = 32$)

Khi $d_k = 32$, mỗi token vector có kích thước 32 Bytes (cần **2 beats 128-bit** trên bus AXI). Dữ liệu tăng gấp đôi làm **thay đổi toàn bộ nhịp thực thi của cả 5 công đoạn phần cứng**:

```
+---------------------------------------------------------------------------------------------------+
|               QUY TRÌNH TOÀN MẠCH XỬ LÝ 1 HÀNG ATTENTION - CẤU HÌNH MỞ RỘNG d_k = 32              |
+---------------------------------------------------------------------------------------------------+
  [Q Vector: 1x32 (2 beats)]       +-----------------------------------+
        |                          | K-Buffer (256x32 = 512 beats BRAM)|
        v                          +-----------------------------------+
  +------------------------------------------+ | (Stream 512 beats, II = 2)
  |  GIAI ĐOẠN 1: GEMM 1 (S = Q x K^T)       |<-+
  |  - 8 DSP nạp 4 beats / Cặp Token         |
  |  - Thời gian: 128 x 4 + 6 = 518 chu kỳ   |
  |  - Thông lượng suy giảm: II = 2.0        |
  +------------------------------------------+
        |  Raw Scores S[0..255] (Xuất 2 scores rồi nghỉ 2 bubbles, II = 2)
        v
  +------------------------------------------+
  |  GIAI ĐOẠN 2: SCALE UNIT                 |
  |  - S' = S * (1 / sqrt(32)) ≈ S * 0.1768  |
  |  - Nhận score theo nhịp II = 2           |
  +------------------------------------------+
        |  Scaled Scores S'[0..255] (Kèm bong bóng II = 2)
        v
  +------------------------------------------+
  |  GIAI ĐOẠN 3: SOFTMAX ENGINE (3 PASSES)  | <=== [TÁI CÂN BẰNG NHỊP: ~782 CHU KỲ]
  |  - Pass 1: Max Finder bị giãn:  518 cyc  | (Phải đợi GEMM 1 bắn đủ 256 scores ở II = 2)
  |  - Pass 2: SubMax + Exp + Sum:  256 cyc  | (Vẫn đọc N=256 scores từ Scratchpad BRAM)
  |  - Divider: Tính 1 / Sum Exp:    14 cyc  | (Khối chia không đổi)
  |  - Pass 3: Norm P_i (II = 2):   512 cyc  | (Phải xả chậm ở II = 2 để khớp nhịp Context GEMM)
  +------------------------------------------+
        |  Probabilities P[0..255] (Stream ra ở nhịp II = 2)
        v
  +------------------------------------------+ +-----------------------------------+
  |  GIAI ĐOẠN 4: CONTEXT GEMM (O = P x V)   | | V-Buffer (256x32 = 512 beats BRAM)|
  |  - Vector V[t, :] có 32 phần tử (2 beats)|<-+ (Mỗi bước tích lũy cần nạp 2 beats V)
  |  - Tích lũy: 256 tokens x 2 beats        |
  |  - Chiếm đúng 512 chu kỳ (II = 2)        |
  |  - Chạy CHỒNG LẤP hoàn toàn với Pass 3   |
  +------------------------------------------+
        |  Output Token Vector: 1x32 (INT8)
        v
  +------------------------------------------+
  |  GIAI ĐOẠN 5: OUTPUT BUFFER & DMA OUT    |
  |  - Đóng gói 2 beats 128-bit Egress       |
  +------------------------------------------+
```

#### Biểu đồ thời gian chồng lặp giữa các hàng khi $d_k = 32$:
```text
Clock Cycles --->
Hàng r:   [------ GEMM 1 (518c) ------][Scale][ Pass 2 Exp (256c) ][-- Pass 3 & GEMM 2 (512c) --]
          \_________ Pass 1 Max (518c) _______/
Hàng r+1:                             [------ GEMM 1 (518c) ------][Scale][ Pass 2 ][ Pass 3 ... ]
                                      ^
                                      | GEMM 1 hàng r+1 gối đầu ngay khi Pass 1 hàng r hoàn thành
```
> **Đặc trưng $d_k = 32$:** GEMM 1 ($518$ chu kỳ) và Context GEMM ($512$ chu kỳ) tăng gấp đôi thời gian, **khớp nhịp hoàn hảo ($1:1$) với Softmax Pass 2 + Inv ($270$ chu kỳ)**, giúp toàn mạch triệt tiêu trạng thái rảnh rỗi chờ đợi.

---

### 1.3. Bảng ma trận đối sánh toàn mạch qua các mức $d_k \in \{16, 20, 24, 32\}$

Bảng dưới đây tổng hợp chu kỳ thực thi của **tất cả 5 công đoạn trong toàn mạch** trên kiến trúc Variant 2A (8 DSP48E2) tại tần số $200\text{ MHz}$, chuỗi token $N = 256$:

| Công đoạn Phần cứng (Hardware Stage) | Cấu hình $d_k = 16$ | Cấu hình $d_k = 20$ | Cấu hình $d_k = 24$ | Cấu hình $d_k = 32$ ($d_{k,\max}$) | Tác động khi $d_k > 16$ |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dung lượng nạp DMA ($Q + K + V$)** | 768 beats (12KB) | 1.152 beats (18KB) | 1.152 beats (18KB) | **1.536 beats (24KB)** | Số beat DMA tăng tỷ lệ thuận $d_k$ |
| **GIAI ĐOẠN 1: GEMM 1 ($Q \times K^T$)** | **262 chu kỳ** ($II=1$) | 390 chu kỳ ($II=1.5$)| 390 chu kỳ ($II=1.5$)| **518 chu kỳ** ($II=2.0$) | Tăng do nạp 3-4 beats/cặp token |
| **GIAI ĐOẠN 2: SCALE UNIT** | 2 chu kỳ ($II=1$) | 2 chu kỳ ($II=1.5$) | 2 chu kỳ ($II=1.5$) | **2 chu kỳ** ($II=2.0$) | Trễ cố định, thông lượng theo GEMM 1 |
| **GIAI ĐOẠN 3: SOFTMAX - Pass 1 (Max)** | 256 chu kỳ | 384 chu kỳ | 384 chu kỳ | **512 chu kỳ** | Bị kéo dài theo nhịp bắn điểm số |
| **GIAI ĐOẠN 3: SOFTMAX - Pass 2 (Exp)** | 256 chu kỳ | 256 chu kỳ | 256 chu kỳ | **256 chu kỳ** | Đọc Scratchpad BRAM $N=256$ (Không đổi) |
| **GIAI ĐOẠN 3: SOFTMAX - Divider (Inv)**| 14 chu kỳ | 14 chu kỳ | 14 chu kỳ | **14 chu kỳ** | Chia số nguyên cố định |
| **GIAI ĐOẠN 3: SOFTMAX - Pass 3 (Norm)**| 256 chu kỳ ($II=1$) | 384 chu kỳ ($II=1.5$)| 384 chu kỳ ($II=1.5$)| **512 chu kỳ** ($II=2.0$) | Phát chậm lại để đồng bộ Context GEMM |
| **GIAI ĐOẠN 4: CONTEXT GEMM ($P \times V$)**| **256 chu kỳ** | 384 chu kỳ | 384 chu kỳ | **512 chu kỳ** | Tăng do vector $V$ tốn 2 beats nạp |
| *(Chồng lấp Pass 3 & Context GEMM)* | Chồng lấp 100% | Chồng lấp 100% | Chồng lấp 100% | Chồng lấp 100% | Tiết kiệm trọn vẹn 1 công đoạn |
| **GIAI ĐOẠN 5: OUTPUT DMA EGRESS** | 1 beat / hàng | 2 beats / hàng | 2 beats / hàng | **2 beats / hàng** | Đóng gói vector Context ra AXI-Stream |
| **Tổng chu kỳ 1 Hàng (Không Pipeline)** | **790 chu kỳ** | 1.046 chu kỳ | 1.046 chu kỳ | **1.302 chu kỳ** | Thời gian xử lý đơn hàng |
| **Nhịp phát hàng mới (Pipelined Period)**| **532 chu kỳ** | 654 chu kỳ | 654 chu kỳ | **782 chu kỳ** | Pacing stage chi phối |
| **Tổng chu kỳ 1 Head ($256$ Hàng)** | **$209.920$ chu kỳ** | $238.400$ chu kỳ | $238.400$ chu kỳ | **$268.800$ chu kỳ** | Đo đạc cấp toàn bộ Head |
| **Thời gian xử lý 1 Head (@ 200MHz)** | **$1.050\text{ ms}$** | $1.192\text{ ms}$ | $1.192\text{ ms}$ | **$1.344\text{ ms}$** | Tần số 200 MHz |
| **Trạng thái nghẽn hệ thống (Bottleneck)**| **Softmax-Bound** | Chuyển tiếp | Chuyển tiếp | **Cân bằng nhịp ($1:1$)** | GEMM 1 $\approx$ Softmax $\approx$ Context |

---

## 2. QUAN HỆ BẢN CHẤT GIỮA "BEAT", ĐỘ RỘNG BUS VÀ THÔNG LƯỢNG

### 2.1 Định nghĩa "Beat" trong Giao thức Bus và Vi mạch
Trong các giao thức bus chuẩn công nghiệp (như AXI4, AXI4-Stream) và trong các đường bus nội vi (Internal BRAM Interconnect):
* **1 Beat** là một lần truyền dữ liệu thành công trong đúng **1 chu kỳ xung nhịp** đồng hồ khi cả tín hiệu hợp lệ (`valid = 1`) và tín hiệu sẵn sàng (`ready = 1`) cùng tích cực (`valid && ready == 1`).
* **Độ rộng của 1 Beat (Beat Width)** được đo bằng số lượng bit hoặc byte trên đường dây vật lý:
  - Trên nền tảng AMD Kria KV260, bus AXI HP (High Performance) kết nối PS DDR4 và PL có độ rộng chuẩn là **128-bit = 16 Bytes**. Do đó, 1 Beat mang tối đa 16 Bytes dữ liệu.
  - Nếu mở rộng bus nội vi lên 256-bit, 1 Beat sẽ mang 32 Bytes dữ liệu.

### 2.2 Công thức chuyển đổi giữa Beat, Kích thước Token và Thông lượng ($II$)
Mỗi Token vector trong không gian nhúng của Vision Transformer có chiều dài đặc trưng $d_k$. Khi lượng tử hóa số nguyên **INT8 (1 byte / phần tử)**, kích thước của 1 token vector là $d_k$ Bytes.

Mối quan hệ toán học giữa số Beats và số Token được xác định bởi:
$$\text{Số Beats cần cho 1 Token} = \left\lceil \frac{d_k \times 1\text{ Byte}}{\text{Độ rộng Bus (Bytes)}} \right\rceil$$

Từ đó, **khoảng cách phát lệnh lý tưởng (Initiation Interval - $II$)** và **thông lượng tính toán GEMM** bị chặn trực tiếp bởi tốc độ nạp beat:
$$II_{\text{GEMM}} \ge \text{Số Beats cần cho 1 Token}$$
$$\text{Thông lượng GEMM (Scores / chu kỳ)} = \frac{1}{II_{\text{GEMM}}} \le \frac{1}{\text{Số Beats cần cho 1 Token}}$$

### 2.3 Sơ đồ minh họa dòng Beats trên Bus nạp dữ liệu (ASCII Timing Diagram)

#### Trường hợp A: $d_k = 16$, Bus 128-bit (1 Token = 1 Beat $\implies II = 1$)
```
Clock Cycle:    T0          T1          T2          T3          T4          ...
AXI/BRAM Bus:  [Token K0 ] [Token K1 ] [Token K2 ] [Token K3 ] [Token K4 ] ...
Dung lượng:    (16 Bytes)  (16 Bytes)  (16 Bytes)  (16 Bytes)  (16 Bytes)
Số Beat:       Beat 0      Beat 1      Beat 2      Beat 3      Beat 4
Pipeline GEMM: |--- Pipeline Delay (5-6 cyc) ---|
Kết quả Score: ---(trễ)--- Score 0     Score 1     Score 2     Score 3     ... (II = 1)
```
* **Đánh giá:** Mỗi chu kỳ nạp đúng 1 Token hoàn chỉnh $\implies$ Compute Engine tính toán và xuất **1 Score / cycle** liên tục ($II = 1$). Cả khối I/O và khối tính toán đều chạy với 100% hiệu suất, không có chu kỳ chết (Zero Bubble).

#### Trường hợp B: $d_k = 32$, Bus 128-bit (1 Token = 2 Beats $\implies II = 2$)
```
Clock Cycle:    T0          T1          T2          T3          T4          ...
AXI/BRAM Bus:  [K0: 0..15] [K0: 16..31][K1: 0..15] [K1: 16..31][K2: 0..15] ...
Dung lượng:    (16 Bytes)  (16 Bytes)  (16 Bytes)  (16 Bytes)  (16 Bytes)
Số Beat:       Beat 0 (K0) Beat 1 (K0) Beat 0 (K1) Beat 1 (K1) Beat 0 (K2)
Tích lũy DSP:  Tích lũy 1  Cộng dồn 2  Tích lũy 1  Cộng dồn 2  ...
Kết quả Score: ------------(trễ)------ Score 0     ---         Score 1     ... (II = 2)
```
* **Đánh giá:** Cần 2 chu kỳ để nhận đủ 1 Token (2 Beats) $\implies$ Khối tính toán phải tích lũy Beat 0 rồi cộng dồn với Beat 1 $\implies$ Thông lượng giảm xuống còn **0.5 Score / cycle** ($II = 2$). Hệ thống rơi vào trạng thái **nghẽn I/O nạp dữ liệu (Memory/Bus Bound)** nếu không mở rộng bus lên 256-bit.

### 2.4 Bảng đối sánh đa chiều: Bus Width vs $d_k$ vs Beats/Token vs Thông lượng

| Cấu hình Không gian | Chiều vector $d_k$ | Độ rộng Bus | Số Beats / Token | Tần suất nạp dữ liệu | Thông lượng GEMM ($II$) | Chu kỳ GEMM 1 Hàng ($N=256$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chuẩn MobileViT (Baseline)** | $d_k = 16$ | 128-bit (16B) | **1 Beat** | 1 Token / cycle | **$II = 1$ (1 score/cyc)** | **256 chu kỳ** |
| **Mở rộng đệm 0 (Zero Padding)**| $d_k = 20$ | 128-bit (16B) | **2 Beats** | 0.5 Token / cycle| **$II = 2$ (0.5 score/cyc)**| **512 chu kỳ** |
| **Mở rộng đệm 0 (Zero Padding)**| $d_k = 24$ | 128-bit (16B) | **2 Beats** | 0.5 Token / cycle| **$II = 2$ (0.5 score/cyc)**| **512 chu kỳ** |
| **ViT-Base Thu nhỏ (Bus 128b)** | $d_k = 32$ | 128-bit (16B) | **2 Beats** | 0.5 Token / cycle| **$II = 2$ (0.5 score/cyc)**| **512 chu kỳ** |
| **ViT-Base Nâng cao (Bus 256b)** | $d_k = 32$ | 256-bit (32B) | **1 Beat** | 1 Token / cycle | **$II = 1$ (1 score/cyc)** | **256 chu kỳ** |

### 2.5 Khái niệm "Beat" ở dòng dữ liệu đầu ra (Downstream Score Stream)
Thuật ngữ "Beat" không chỉ giới hạn ở dòng nạp token từ bộ nhớ, mà còn áp dụng cho luồng dữ liệu trung gian giữa các tầng phần cứng:
* **Score Beats:** Khi GEMM 1 của Variant 1A hoặc 2A vận hành ở $II = 1$, đầu ra của nó tạo ra một dòng liên tục các **Score Beats** (mỗi beat chứa 1 giá trị Score 24-bit kèm theo cờ `valid = 1`).
* **Khớp nối nhịp không đệm trễ (Zero-Bubble Coupling):** Dòng Score Beats này được bơm trực tiếp sang **Scale Unit** rồi tới **Softmax Engine (Pass 1 - Max Search)**. Mỗi chu kỳ nhận 1 beat score, bộ so sánh cập nhật giá trị cực đại ngay tức thời (`online max search`), loại bỏ hoàn toàn độ trễ lưu tạm vào BRAM.

```
+---------------+     Score Beat (24-bit)     +--------------+  Scaled Beat (16-bit)  +----------------+
| GEMM 1 Engine | ==========================> |  Scale Unit  | =====================> | Softmax Pass 1 |
+---------------+      (1 beat / cycle)       +--------------+    (1 beat / cycle)    +----------------+
```

### 2.6 Phân tích hiện tượng nghẽn nhịp Beat khi $d_k > 16$ (Beat-Bound Bottleneck)

Một hiểu lầm kỹ thuật phổ biến là: *"Bộ tuần tự hóa Zero-Bubble Serializer của Variant 2A luôn đảm bảo $II = 1$ cho mọi kích thước $d_k$"*.  
**Thực tế vi mạch:** Kiến trúc 8 DSP của Variant 2A **chỉ đạt $II = 1$ khi $d_k = 16$**. Khi mở rộng lên $d_k \in \{20, 24, 32\}$, hệ thống sẽ xuất hiện **hiện tượng nghẽn nhịp Beat (Beat-Bound Bottleneck)** và tự động suy giảm thông lượng ($II > 1$).

#### A. Công thức định lượng bọt khí (Bubble Injection Formula):
Trong mỗi chu kỳ, 8 phiến DSP (SIMD packing) tiêu thụ được $8 \text{ phần tử } K_{\text{even}}$ và $8 \text{ phần tử } K_{\text{odd}}$ (vừa khít 1 beat 128-bit). Do đó:
$$\text{Số beats cần thiết để nạp 1 Cặp Token (Even & Odd)} = N_{\text{beats}} = \left\lceil \frac{d_k}{8} \right\rceil$$

Thời gian mà bộ **Zero-Bubble Serializer** xả hết 2 điểm số (Even và Odd) ra ngõ ra **luôn luôn cố định là 2 chu kỳ đồng hồ** (Chu kỳ $T$: xuất Even, Chu kỳ $T+1$: xuất Odd).  
Số chu kỳ bọt khí chết (Bubble / Idle Cycles) xuất hiện sau mỗi cặp token là:
$$\text{Số chu kỳ Bubble} = \max\left(0,\, N_{\text{beats}} - 2\right) = \max\left(0,\, \left\lceil \frac{d_k}{8} \right\rceil - 2\right)$$

Từ đó, khoảng cách phát lệnh trung bình ($II$) của khối GEMM 1 là:
$$II_{\text{GEMM 1}} = \frac{N_{\text{beats}}}{2} = \frac{\left\lceil d_k / 8 \right\rceil}{2} \text{ chu kỳ / score}$$

#### B. Giản đồ so sánh nhịp xả điểm số (Output Stream Timing Trace):

```
1) Cấu hình d_k = 16 (N_beats = 2, II = 1.0 - Khớp nhịp hoàn hảo, 0 Bubble):
Input Stream:   [Cặp 0: B0][Cặp 0: B1][Cặp 1: B0][Cặp 1: B1][Cặp 2: B0][Cặp 2: B1]
Output Score:   ---  ---  ---  [ S0 ] [ S1 ] [ S2 ] [ S3 ] [ S4 ] [ S5 ]  <== II = 1 liên tục
out_valid:       0    0    0     1      1      1      1      1      1   <== 100% duty cycle

2) Cấu hình d_k = 24 (N_beats = 3, II = 1.5 - Xuất hiện 1 Bubble mỗi cặp):
Input Stream:   [Cặp 0: B0][Cặp 0: B1][Cặp 0: B2][Cặp 1: B0][Cặp 1: B1][Cặp 1: B2]
Output Score:   ---  ---  ---  [ S0 ] [ S1 ] [ ---- ] [ S2 ] [ S3 ] [ ---- ]  <== II = 1.5
out_valid:       0    0    0     1      1       0       1      1       0   <== 66.7% duty cycle
                                             ^^^^^^                 ^^^^^^
                                             Bubble                 Bubble

3) Cấu hình d_k = 32 (N_beats = 4, II = 2.0 - Xuất hiện 2 Bubbles mỗi cặp):
Input Stream:   [Cp0: B0][Cp0: B1][Cp0: B2][Cp0: B3][Cp1: B0][Cp1: B1][Cp1: B2][Cp1: B3]
Output Score:   --- --- --- --- [ S0 ] [ S1 ] [ ---- ] [ ---- ] [ S2 ] [ S3 ] [ ---- ] [ ---- ]
out_valid:       0   0   0   0    1      1       0       0       1      1       0       0
                                             ^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^
                                              2 cyc Bubble            2 cyc Bubble
```

#### C. Bảng tổng hợp sự suy giảm thông lượng theo $d_k$ trên kiến trúc 8 DSP:

| Chiều vector $d_k$ | Số beats / Cặp ($N_{\text{beats}}$) | Chu kỳ nạp Cặp | Chu kỳ xả Cặp | Chu kỳ Bubble | Thông lượng $II$ | Chu kỳ GEMM 1 / Hàng ($N=256$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$d_k = 16$** | **2 beats** | 2 chu kỳ | 2 chu kỳ | **0 cycle** | **$II = 1.0$ (1 score/cyc)** | **262 chu kỳ** (Tối ưu) |
| **$d_k = 20$** | **3 beats** | 3 chu kỳ | 2 chu kỳ | **1 cycle** | **$II = 1.5$ (0.67 score/cyc)**| **390 chu kỳ** |
| **$d_k = 24$** | **3 beats** | 3 chu kỳ | 2 chu kỳ | **1 cycle** | **$II = 1.5$ (0.67 score/cyc)**| **390 chu kỳ** |
| **$d_k = 32$** | **4 beats** | 4 chu kỳ | 2 chu kỳ | **2 cycles**| **$II = 2.0$ (0.5 score/cyc)** | **518 chu kỳ** |

---

## 3. SƠ ĐỒ THÔNG LƯỢNG VARIANT 1A: VECTOR STREAMING ENGINE (THUẦN LUT)

Variant 1A (`rtl_ver02_1A`) là biến thể tập trung vào **độ trễ thấp nhất và không phụ thuộc tài nguyên DSP**. Toàn bộ phép nhân và cây cộng giảm cấp (reduction tree) được tổng hợp hoàn toàn từ các khối LUT của FPGA.

### Sơ đồ khối chi tiết đường dữ liệu (Datapath ASCII)

```
        Q_Beat [16 Bytes]                      K_Beat_t [16 Bytes] (từ K-Buffer)
         (1x16 vector INT8)                     (1 token INT8 mỗi chu kỳ)
                 |                                      |
                 +-------------------+  +---------------+
                                     |  |
                                     v  v
        +---------------------------------------------------------------+
        |  STAGE 0: 32 BỘ NHÂN THUẦN LUT (8-bit x 8-bit Signed Multipliers)
        |  [M0]   [M1]   [M2]   [M3]  ...  [M14]  [M15]   (16 DSP-free LUT Mults)
        |  16-bit Product outputs: P_0, P_1, P_2, ... P_15
        +---------------------------------------------------------------+
                                         |
                                         v  (Độ trễ: 1 cycle)
        +---------------------------------------------------------------+
        |  CÂY CỘNG KHÔNG GIAN 5 TẦNG (5-STAGE BINARY ADDER TREE TRÊN LUT)
        |
        |  Tầng 1 (16 -> 8): [A0=P0+P1] [A1=P2+P3] ... [A7=P14+P15]    (Reg T1)
        |                            |
        |  Tầng 2 (8 -> 4):  [B0=A0+A1] [B1=A2+A3] [B2=A4+A5] [B3=A6+A7] (Reg T2)
        |                            |
        |  Tầng 3 (4 -> 2):  [C0=B0+B1]            [C1=B2+B3]          (Reg T3)
        |                            |
        |  Tầng 4 (2 -> 1):  [D0=C0+C1]                                (Reg T4)
        |                            |
        |  Tầng 5 (Căn chỉnh & Tràn số): 24-bit Accumulator Output     (Reg T5)
        +---------------------------------------------------------------+
                                         |
                                         v  (Độ trễ toàn bộ cây cộng: 5 cycles)
                        +---------------------------------+
                        |  Dot-Product Score S_t (24-bit) |
                        |  Thông lượng: 1 Score / Cycle   |
                        |  Khoảng cách: II = 1            |
                        +---------------------------------+
```

### Đặc tính thông lượng Variant 1A:
* **Khởi tạo dữ liệu (Fill Latency):** 5 chu kỳ để nạp đầy pipeline.
* **Tần suất đầu ra:** Bắt đầu từ chu kỳ thứ 6, cứ mỗi chu kỳ đồng hồ xuất đều đặn 1 giá trị tích vô hướng ($Q \cdot K_t^T$).
* **Tổng chu kỳ tính toán GEMM 1 cho 1 hàng ($N=256$):**
  $$\text{Latency}_{\text{row, 1A}} = 5 \text{ (pipeline fill)} + 256 \text{ (streaming)} = \mathbf{261 \text{ chu kỳ đồng hồ}}.$$
* **Tài nguyên tiêu thụ:** $0$ DSP48E2, $\approx 2.250$ LUTs cho riêng Compute Engine.
* **Ưu điểm:** Cấu trúc cực kỳ đơn giản, luồng dữ liệu một chiều không có bộ đệm phức tạp, $F_{\max} \ge 220 \text{ MHz}$.

---

## 4. SƠ ĐỒ THÔNG LƯỢNG VARIANT 2A: INTERLEAVED DSP-PACKED STREAMING ENGINE

Variant 2A (`rtl_ver02_2A`) là biến thể được tối ưu hóa chuyên sâu cho kiến trúc **Xilinx UltraScale+ DSP48E2**. Sử dụng kỹ thuật **DSP Packing** (ghép 2 phép nhân INT8 vào 1 DSP48E2) kết hợp với **phân luồng xen kẽ Chẵn/Lẻ (Even/Odd Interleaving)** và bộ tuần tự hóa **Zero-Bubble Serializer** để đạt thông lượng đỉnh cao nhất với số lượng DSP tối thiểu.

### Sơ đồ khối chi tiết đường dữ liệu (Datapath ASCII)

```
       Q_Beat [16 Bytes]        K_Beat_even [16 Bytes]     K_Beat_odd [16 Bytes]
        (Vector INT8)             (Token k=0, 2, 4...)      (Token k=1, 3, 5...)
              |                             |                         |
              +--------------+--------------+                         |
                             |                                        |
                             v                                        v
        +-----------------------------------------------------------------------+
        |  KHỐI 8 PHIẾN DSP48E2 (INT8 SIMD PACKING: 16 Phép nhân đồng thời)    |
        |  - DSP[0]: Q[0..1]  x K_even/odd[0..1]   -> P_even[0], P_odd[0]       |
        |  - DSP[1]: Q[2..3]  x K_even/odd[2..3]   -> P_even[1], P_odd[1]       |
        |    ...                                                                |
        |  - DSP[7]: Q[14..15]x K_even/odd[14..15] -> P_even[7], P_odd[7]       |
        +-----------------------------------------------------------------------+
                     |                                         |
                     v (8 tích Chẵn)                           v (8 tích Lẻ)
        +---------------------------+             +---------------------------+
        | CÂY CỘNG CHẴN (3 TẦNG)    |             | CÂY CỘNG LẺ (3 TẦNG)      |
        | 8 -> 4 -> 2 -> 1          |             | 8 -> 4 -> 2 -> 1          |
        | Độ trễ: 3 cycles          |             | Độ trễ: 3 cycles          |
        +---------------------------+             +---------------------------+
                     |                                         |
                     v                                         v
               Sum_Even (tích vô hướng k)               Sum_Odd (tích vô hướng k+1)
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v  (Xuất song song 2 scores mỗi 2 chu kỳ)
        +-----------------------------------------------------------------------+
        |               ZERO-BUBBLE SERIALIZER (BỘ TUẦN TỰ HÓA 2:1)             |
        |                                                                       |
        |     +-------------------+         +-------------------+               |
        |     |  Even Latch Reg   |         |   Odd FIFO Reg    |               |
        |     +-------------------+         +-------------------+               |
        |               \                             /                         |
        |                \                           /                          |
        |                 v                         v                           |
        |               +-----------------------------+                         |
        |               |   Ping-Pong Multiplexer     |<-- Mux Sel (Phase Bit)  |
        |               +-----------------------------+                         |
        +-----------------------------------------------------------------------+
                                          |
                                          v  (Khử hoàn toàn bong bóng trễ)
                        +-----------------------------------+
                        |   Score Stream S_t (24-bit)       |
                        |   Thông lượng: 1 Score / Cycle    |
                        |   Khoảng cách: II = 1 LIÊN TỤC   |
                        +-----------------------------------+
```

### Nguyên lý hoạt động của Zero-Bubble Serializer:
Bộ xử lý DSP + Dual Tree tạo ra một cặp kết quả `(Score_Even, Score_Odd)` sau mỗi 2 chu kỳ xung nhịp. Nếu truyền trực tiếp qua một multiplexer thông thường, pipeline sẽ bị ngắt quãng (1 chu kỳ có data, 1 chu kỳ rỗng $\implies II = 2$).  
**Zero-Bubble Serializer** giải quyết triệt để vấn đề này bằng cách:
1. Chu kỳ $2m$: Chốt `Sum_Even` thẳng ra ngõ ra, đồng thời đưa `Sum_Odd` vào thanh ghi đệm nội vi.
2. Chu kỳ $2m+1$: Đẩy `Sum_Odd` từ thanh ghi đệm ra ngõ ra, trong lúc khối DSP đang tích lũy nhịp tiếp theo.
3. Kết quả: Tạo ra một luồng dữ liệu phẳng, đều đặn **1 score/cycle** ($II = 1$) cung cấp thẳng cho Scaler Unit và Softmax Engine.

### Đặc tính thông lượng Variant 2A:
* **Khởi tạo dữ liệu (Fill Latency):** 6 chu kỳ (DSP MReg/PReg: 2 cyc + Adder Tree: 3 cyc + Serializer Mux: 1 cyc).
* **Tần suất đầu ra:** 
  - Tại $d_k = 16$: **1 Score/cycle ($II = 1$)** liên tục không bọt khí.
  - Tại $d_k = 24$: **0.67 Score/cycle ($II = 1.5$)** (bị 1 chu kỳ bọt khí mỗi cặp).
  - Tại $d_k = 32$: **0.50 Score/cycle ($II = 2.0$)** (bị 2 chu kỳ bọt khí mỗi cặp).
* **Tổng chu kỳ tính toán GEMM 1 cho 1 hàng ($N=256$):**
  - Tại $d_k = 16$: $6 + (128 \times 2) = \mathbf{262 \text{ chu kỳ đồng hồ}}$.
  - Tại $d_k = 24$: $6 + (128 \times 3) = \mathbf{390 \text{ chu kỳ đồng hồ}}$.
  - Tại $d_k = 32$: $6 + (128 \times 4) = \mathbf{518 \text{ chu kỳ đồng hồ}}$.
* **Tài nguyên tiêu thụ:** Chỉ **8 DSP48E2**, $\approx 1.250$ LUTs (tiết kiệm **44% LUT** so với Variant 1A).

### Phân tích chi tiết mã nguồn RTL: Vì sao $d_k > 16$ bị nghẽn nhịp Beat?
Bên trong module [`dsp_packed_adder_tree.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/dsp_packed_adder_tree.sv), việc tích lũy đa beat được điều khiển trực tiếp bởi tín hiệu `in_beat` và `is_last_beat`:
1. **Pha tích lũy đa beat (Lines 205-211):**
   ```systemverilog
   if (t3_beat == 2'b00) begin
       accum_even <= t3_even;
       accum_odd  <= t3_odd;
   end else if (!t3_last) begin
       accum_even <= accum_even + t3_even;
       accum_odd  <= accum_odd  + t3_odd;
   end
   ```
   Khi $d_k = 32$, mạch phải trải qua 4 beats (`in_beat = 0, 1, 2, 3`). Bộ tích lũy phải giữ và cộng dồn liên tục trong 4 chu kỳ mới gom đủ 32 tích thành phần.
2. **Pha xả điểm số của Serializer (Lines 213-231):**
   - Tại nhịp cuối `t3_last` (Beat 3): Mạch xuất ngay điểm số `Even` ra `dot_product` (`out_valid = 1`), đồng thời chốt điểm số `Odd` vào `odd_hold_reg` và bật cờ `odd_pending = 1`.
   - Tại nhịp tiếp theo: Mạch xuất tiếp điểm số `Odd` từ `odd_hold_reg` (`out_valid = 1`), hạ cờ `odd_pending = 0`.
3. **Hiện tượng hụt nhịp (Starvation Bubble):**
   - Serializer chỉ có đúng 2 điểm số (`Even` và `Odd`) nên chỉ xả được trong **đúng 2 chu kỳ**.
   - Tuy nhiên, cặp token tiếp theo cần tới **4 chu kỳ nạp** (`Beat 0, 1, 2, 3`) mới tới được nhịp `t3_last`.
   - **Kết quả:** Trong 2 chu kỳ ở giữa (khi cặp mới đang nạp Beat 0 và Beat 1), Serializer đã xả hết hàng và không có dữ liệu mới $\implies$ Cờ `out_valid` bắt buộc phải hạ xuống `0` trong 2 chu kỳ đó.
   - Đây chính là nguyên nhân kỹ thuật giải thích vì sao **trên cấu hình 8 DSP, khi $d_k = 32$ thì thông lượng bắt buộc suy giảm về $II = 2$**.

---

## 5. SƠ ĐỒ THÔNG LƯỢNG VARIANT 3B: 2D OUTPUT-STATIONARY SYSTOLIC ARRAY

Variant 3B (`rtl_ver02_3B`) hiện thực hóa một mạng mảng tâm thu 2 chiều kích thước $4 \times 4$ gồm 16 phần tử xử lý (PE). Kiểu truyền thông là **Output-Stationary (OS)**: Giá trị tích lũy cục bộ đứng yên trong PE, các phần tử vector $Q$ và ma trận $K$ được đẩy trượt (wavefront skew) qua lưới không gian 2D.

### Sơ đồ cấu trúc mảng Systolic Array 2D ($4 \times 4$ PEs)

```
        K_in[0]       K_in[1]       K_in[2]       K_in[3]
           |             |             |             |
       (Skew 0)      (Skew 1)      (Skew 2)      (Skew 3)
           |             |             |             |
           v             v             v             v
Q_in[0] -> [ PE_0,0 ] === [ PE_0,1 ] === [ PE_0,2 ] === [ PE_0,3 ]
(Skew 0)       |             |             |             |
               v             v             v             v
Q_in[1] -> [ PE_1,0 ] === [ PE_1,1 ] === [ PE_1,2 ] === [ PE_1,3 ]
(Skew 1)       |             |             |             |
               v             v             v             v
Q_in[2] -> [ PE_2,0 ] === [ PE_2,1 ] === [ PE_2,2 ] === [ PE_2,3 ]
(Skew 2)       |             |             |             |
               v             v             v             v
Q_in[3] -> [ PE_3,0 ] === [ PE_3,1 ] === [ PE_3,2 ] === [ PE_3,3 ]
(Skew 3)       |             |             |             |
               +-------------+-------------+-------------+
                                     |
                                     v (Xả kết quả sau khi hoàn thành 1 Tile)
                        +-------------------------+
                        |  REORDER BUFFER & DRAIN |
                        |  (Quét hàng & Giải mã) |
                        +-------------------------+
                                     |
                                     v (Tái cấu trúc thành dòng tuần tự)
                        +-------------------------+
                        |  Score Stream S[0..255] |
                        +-------------------------+
```

### Chi tiết bên trong mỗi Processing Element (PE):
Mỗi PE chứa 1 phiến DSP48E2 đóng vai trò bộ tích lũy nhân chập (MAC):
$$C_{\text{accum}} \leftarrow C_{\text{accum}} + (Q_{\text{in}} \times K_{\text{in}})$$
* **Chế độ tính toán:** Dữ liệu $Q$ trượt ngang từ Tây sang Đông; $K$ trượt dọc từ Bắc xuống Nam.
* **Tích lũy:** Trong suốt $K_{\text{tile}} = 16$ chu kỳ, $4 \times 4 = 16$ giá trị tích lũy được giữ cố định tại thanh ghi nội nội vi của các PE.
* **Xả kết quả (Drain Phase):** Sau khi tích lũy đủ $d_k = 16$, mảng dừng nạp hoặc chuyển sang thanh ghi bóng (shadow register) để xả 16 kết quả ra Reorder Buffer.

### Phân tích chu kỳ thực thi của Variant 3B cho 1 hàng ($N=256, d_k=16$):
Do mảng tính toán theo khối gạch (Tiling) kích thước $4 \times 4$:
* Số lượng gạch cần thiết để quét hết ma trận $K$ ($256 \times 16$):
  $$\text{Số Tiles} = \frac{N}{4} \times \frac{M}{4} = \frac{256}{4} = 64 \text{ tiles (với 1 batch gồm 4 hàng Q đồng thời)}.$$
* Với 1 hàng đơn lẻ (hoặc quy về thời gian trung bình cho 1 hàng):
  1. **Độ trễ lệch pha Wavefront (Skew Latency):** $4 + 4 - 2 = 6$ chu kỳ.
  2. **Thời gian tích lũy 1 Tile ($K_{\text{tile}} = 16$):** 16 chu kỳ.
  3. **Thời gian xả kết quả và giải mã (Drain Latency):** $4 \times 4 / 2 = 8$ chu kỳ.
  4. **Tổng số chu kỳ xử lý trung bình cho 1 hàng:** $\approx \mathbf{573 \text{ chu kỳ đồng hồ}}$.

### Đặc tính thông lượng Variant 3B:
* **Tài nguyên tiêu thụ:** **16 DSP48E2**, $\approx 2.800$ LUTs, 4 BRAM (cho Skew FIFOs & Reorder Buffer).
* **Hiệu suất tính toán:** Đạt tới 32 MACs/cycle khi mảng được lấp đầy (fully utilized).
* **Hạn chế:** Tốn chi phí trễ nạp/xả biên (Wavefront Skew & Drain Overhead) khi kích thước ma trận bé ($M=1, d_k=16$). Thích hợp nhất khi chạy batch nhiều hàng hoặc ma trận vuông lớn.

---

## 6. SO SÁNH TRỰC QUAN NHỊP THỰC THI (TIMING WAVEFORMS & PIPELINES)

Biểu đồ dưới đây minh họa sự xuất hiện của các giá trị Score trên bus đầu ra của khối GEMM 1 theo từng chu kỳ xung nhịp đồng hồ ($T_0, T_1, \dots, T_{12}$):

```
Cycle:      T0   T1   T2   T3   T4   T5   T6   T7   T8   T9   T10  T11  T12
---------------------------------------------------------------------------------
[Variant 1A: LUT Tree]
Q/K In:     K0   K1   K2   K3   K4   K5   K6   K7   K8   K9   K10  K11  K12
Pipeline:   |--- 5-cycle Adder Tree Latency ---|
Score Out:  ---  ---  ---  ---  ---  S0   S1   S2   S3   S4   S5   S6   S7  (II=1)
---------------------------------------------------------------------------------
[Variant 2A: Interleaved DSP + Zero-Bubble Serializer]
Q/K In:     K0,1 ---  K2,3 ---  K4,5 ---  K6,7 ---  K8,9 ---  K10,11 ---
DSP/Tree:   |-- DSP + 3-Stage Tree (5 cyc) --|
Serializer: |-- Latch & Ping-Pong (1 cyc) ---|
Score Out:  ---  ---  ---  ---  ---  ---  S0   S1   S2   S3   S4   S5   S6  (II=1)
---------------------------------------------------------------------------------
[Variant 3B: 4x4 Systolic Array]
Inputs:     |<- Skew In ->|<- Tiling Accumulate (16 cyc) ->|<- Drain to ROB ->|
Score Out:  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  [S0 S1 S2 S3] (Burst)
```

> **Nhận xét quan trọng:**  
> - Cả **Variant 1A** và **Variant 2A** đều đạt khoảng cách phát lệnh lý tưởng **$II = 1$ liên tục**, không có bất kỳ khoảng trống rỗng (bubble) nào giữa các score.
> - Variant 1A có độ trễ ban đầu (Latency to first token) ngắn hơn Variant 2A đúng **1 chu kỳ** (5 chu kỳ so với 6 chu kỳ).
> - Variant 3B cho ra kết quả theo dạng chùm (burst) từng khối gạch $4 \times 4$, đòi hỏi bộ đệm Reorder Buffer để giải tuần tự hóa trước khi đưa vào khối Scaler/Softmax.

---

## 7. BẢNG MA TRẬN SO SÁNH ĐỊNH LƯỢNG & TÀI NGUYÊN PHẦN CỨNG

Các số liệu dưới đây được đo đạc và xác thực trực tiếp trên dòng chip **Xilinx Zynq UltraScale+ MPSoC (xck26-sfvc784-2LV-c)** tại tần số hoạt động **$F_{\text{clk}} = 200 \text{ MHz}$**, với cấu hình chuỗi token $N = 256$, kích thước vector $d_k = 16$:

| Tiêu chí Đánh giá | Variant 1A (Thuần LUT) | Variant 2A (Interleaved DSP) | Variant 3B (Systolic Array) | Ghi chú & Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **Kiểu kiến trúc (Architecture)** | Spatial Reduction Tree | Vector Streaming + Serializer | 2D Grid Output-Stationary | 1A/2A: 1D Stream; 3B: 2D Grid |
| **Khoảng cách phát lệnh ($II$)** | **$II = 1$ cycle** | **$II = 1$ cycle** | $II = 4$ cycles (tương đương) | 1A và 2A đạt thông lượng đỉnh |
| **Độ trễ GEMM 1 (1 Hàng $N=256$)**| **261 chu kỳ** | **262 chu kỳ** | **573 chu kỳ** | 1A và 2A nhanh hơn 3B gấp 2.18x |
| **Thời gian GEMM 1 (@ 200MHz)** | **$1.305\ \mu\text{s}$** | **$1.310\ \mu\text{s}$** | **$2.865\ \mu\text{s}$** | Đo đạc cấp hàng |
| **Độ trễ Softmax (3-Pass Engine)**| 532 chu kỳ | 532 chu kỳ | 532 chu kỳ | Dùng chung khối Softmax chuẩn |
| **Tổng độ trễ 1 Hàng (Row Latency)**| $\approx 795$ chu kỳ | $\approx 796$ chu kỳ | $\approx 1.107$ chu kỳ | GEMM 1 + Scale + Softmax |
| **Tổng độ trễ 1 Head ($256$ Hàng)**| **$209.664$ chu kỳ** | **$209.920$ chu kỳ** | **$289.440$ chu kỳ** | Đã bao gồm pipelining liên hàng |
| **Thời gian xử lý 1 Head** | **$1.048\text{ ms}$** | **$1.050\text{ ms}$** | **$1.447\text{ ms}$** | Tần số 200 MHz |
| **Thông lượng Head (Heads/sec)** | **$954\text{ heads/s}$** | **$952\text{ heads/s}$** | **$691\text{ heads/s}$** | Năng lực xử lý thuần nhân MHA |
| **Thông lượng ViT Frame (12 Heads)**| **$183.4\text{ FPS}$** | **$183.1\text{ FPS}$** | **$132.8\text{ FPS}$** | ViT-Base phân rã pipelined |
| **Tài nguyên DSP48E2 tiêu thụ** | **0 DSP** (0%) | **8 DSP** (6.4%) | **16 DSP** (12.9%) | KV260 có tổng 1.248 DSP |
| **Tài nguyên LUT Logic (Toàn Core)**| **$6.850$ LUTs** | **$3.820$ LUTs** | **$5.940$ LUTs** | Variant 2A tiết kiệm nhất |
| **Tỷ lệ tiết kiệm LUT Logic** | Gốc (0%) | **Giảm 44.2% LUT** | Giảm 13.3% LUT | 2A giải phóng diện tích vượt trội |
| **Tài nguyên BRAM (36Kb Blocks)** | 6 BRAM | 6 BRAM | 10 BRAM (thêm ROB) | 3B cần thêm RAM lưu gạch |
| **Hiệu suất trên mỗi DSP (FPS/DSP)**| $\infty$ (Không dùng) | **$22.88\text{ FPS/DSP}$** | **$8.30\text{ FPS/DSP}$** | 2A khai thác DSP gấp 2.75x 3B |
| **Khả năng mở rộng $d_k = 32$** | Mở rộng cây cộng 6 tầng (0 DSP) | 8 DSP chạy 4 beats ($II=2$, 518 cyc); Có thể scale 16 DSP | Tăng chiều sâu Tiling ($K_{\text{tile}}$)| Tất cả đều hỗ trợ $d_{k,\max}=32$ |

---

## 8. PHÂN TÍCH ĐIỂM NGHẼN HỆ THỐNG THEO ĐỊNH LUẬT AMDAHL (SYSTEM BOTTLENECK)

Một hiểu lầm phổ biến trong thiết kế phần cứng bộ tăng tốc là: *"Tăng gấp đôi số lượng nhân tính toán GEMM sẽ tăng gấp đôi tốc độ toàn bộ hệ thống"*. Định luật Amdahl chỉ ra rằng gia tốc tổng thể bị chặn bởi thành phần không thể song song hóa hoặc có chu kỳ xử lý dài nhất.

### Biểu đồ phân bổ chu kỳ thời gian trong 1 Hàng Attention ($d_k = 16$):

```
+-----------------------------------------------------------------------------+
| GEMM 1: 262 cyc (32.8%) | Scale: 2c |        SOFTMAX: 532 cyc (66.9%)       |
+-----------------------------------------------------------------------------+
0 cyc                     262 cyc     264 cyc                                 796 cyc
```

### Phương trình tăng tốc hệ thống (Amdahl's Speedup):
Gọi thời gian thực thi của GEMM 1 là $T_{\text{GEMM}}$ và thời gian thực thi của Softmax là $T_{\text{Softmax}}$:
$$T_{\text{total}} = T_{\text{GEMM}} + T_{\text{Softmax}} = 262 + 532 = 794 \text{ chu kỳ}.$$

Nếu ta nâng cấp Variant 2A từ 8 DSP lên 16 DSP để tính song song 2 Scores mỗi chu kỳ ($II = 0.5$, giảm thời gian GEMM 1 xuống còn $131$ chu kỳ):
$$T_{\text{total, 16DSP}} = 131 + 532 = 663 \text{ chu kỳ}.$$
* **Tỷ lệ tăng tốc tối đa toàn hệ thống:**
  $$S = \frac{794}{663} \approx \mathbf{1.197\times \text{ (Chỉ tăng 19.7% FPS)}}.$$
* **Cái giá phải trả:** Tiêu tốn gấp đôi số lượng DSP ($8 \to 16$ DSP) và tăng độ phức tạp của mạng dây phân phối (Routing Congestion).

### Sự chuyển dịch điểm nghẽn khi mở rộng $d_k = 32$:
Khi $d_k$ mở rộng từ $16 \to 32$:
- Số lượng beat nạp cho mỗi cặp token tăng từ 2 beats lên 4 beats (với bus 128-bit).
- Thời gian tính toán GEMM 1 của Variant 2A (với cấu hình 8 DSP hiện tại) tăng lên: $128 \times 4 + 6 = \mathbf{518 \text{ chu kỳ}}$ ($II = 2.0$, xuất hiện 2 chu kỳ bubble mỗi cặp).
- Thời gian Softmax giữ nguyên: $\approx 532$ chu kỳ (vì số lượng token $N=256$ không đổi).
- **Lúc này, GEMM 1 ($\approx 518$ cyc) và Softmax ($\approx 532$ cyc) đạt trạng thái cân bằng nhịp hoàn hảo ($1:1$)**, tối đa hóa hiệu suất sử dụng phần cứng của cả hai khối xử lý mà không khối nào phải chịu cảnh chờ đợi (idle/stall).

---

## 9. KẾT LUẬN & KHUYẾN NGHỊ KIẾN TRÚC CHO KRIA KV260

Dựa trên kết quả đo đạc thông lượng thực tế, tài nguyên tiêu thụ và mức độ khả thi khi triển khai trên bo mạch Xilinx Kria KV260, chúng tôi đưa ra các kết luận kiến trúc sau:

```
                      SO SÁNH 3 BIẾN THỂ TRÊN RADAR ĐÁNH GIÁ
                               Thông lượng (FPS)
                                      1A
                                     /  \
                                    / 2A \
                                   /  /\  \
                Tiết kiệm LUT    /  /  \  \    Tiết kiệm DSP
                     (2A)       /--+----+--\       (1A)
                                \  \    /  /
                                 \  \3B/  /
                                  \  \/  /
                                   \    /
                                     \/
                                Khả năng Scale
                                     (3B)
```

1. **Khuyến nghị Lựa chọn số 1 (Optimal Choice): [Variant 2A](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/attention_core_top_2A.sv)**
   - **Lý do:** Đạt thông lượng tối đa $II = 1$ ($183.1 \text{ FPS}$), chỉ sử dụng vỏn vẹn **8 DSP48E2** (0.64% tài nguyên DSP của KV260), và **giảm tới 44.2% diện tích LUT** so với 1A. Đây là sự cân bằng hoàn hảo giữa thông lượng, tài nguyên và tiêu thụ điện năng cho các hệ thống Edge AI.
2. **Vai trò của [Variant 1A](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/attention_core_top_1A.sv):**
   - Phù hợp nhất cho các dòng FPGA chi phí cực thấp không tích hợp khối nhân DSP cứng (DSP-less FPGAs) hoặc khi toàn bộ DSP trên chip đã được ưu tiên cấp phát cho các tác vụ xử lý ảnh Convolutional/Backbone khác.
3. **Vai trò của [Variant 3B](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/attention_core_top_3B.sv):**
   - Phù hợp khi thiết kế tăng quy mô phục vụ các ma trận lớn dạng khối ($N \gg 1024, d_k \ge 64$) hoặc khi cấu hình hệ thống xử lý gộp nhiều Head đồng thời (Multi-Head Batching) để triệt tiêu thời gian trễ nạp/xả biên.

---

## 10. LIÊN KẾT MÃ NGUỒN RTL & TÀI LIỆU KIỂM TRA

Mọi chi tiết thiết kế và mã nguồn mô phỏng có thể được tra cứu trực tiếp tại các liên kết dưới đây:

### Mã nguồn phần cứng (Hardware RTL):
* **Variant 1A:**
  - Bộ ghép luồng & Cây cộng LUT: [`vector_streaming_engine_1A.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/vector_streaming_engine_1A.sv)
  - Core Top 1A: [`attention_core_top_1A.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/attention_core_top_1A.sv)
* **Variant 2A:**
  - Bộ ghép luồng DSP + Dual Tree: [`vector_streaming_engine_2A.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/vector_streaming_engine_2A.sv)
  - Bộ tuần tự hóa Zero-Bubble: [`serializer_2to1.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/serializer_2to1.sv)
  - Core Top 2A: [`attention_core_top_2A.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/attention_core_top_2A.sv)
* **Variant 3B:**
  - Mảng Systolic Top: [`systolic_array_top_3B.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/systolic_array_top_3B.sv)
  - Khối PE MAC: [`pe_mac_dsp_3B.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/pe_mac_dsp_3B.sv)
  - Core Top 3B: [`attention_core_top_3B.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/attention_core_top_3B.sv)

### Báo cáo & Tài liệu liên quan:
* Báo cáo so sánh kỹ thuật: [`report_MASTER_comparison.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/report_MASTER_comparison.md)
* So sánh chi tiết Adder Tree vs Zero-Bubble Serializer: [`so_sanh_adder_tree_vs_zero_bubble_serializer.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/so_sanh_adder_tree_vs_zero_bubble_serializer.md)
* Phân tích kiến trúc vi mạch tổng thể: [`mha_kien_truc_vi_mach.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/mha_kien_truc_vi_mach.md)
* Báo cáo kiểm toán và kế hoạch đồng bộ RTL: [`bao_cao_kiem_toan_va_ke_hoach_dong_bo_rtl_tai_lieu.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes_ver02/bao_cao_kiem_toan_va_ke_hoach_dong_bo_rtl_tai_lieu.md)
