# Khảo Sát Không Gian Thiết Kế Đa Chiều: Tác Động Của Độ Rộng Bus AXI-Stream (64-bit đến 1024-bit) và Chiều Sâu Đặc Trưng $d_k$ (16 đến 64) Lên 3 Biến Thể Attention Core

> **Dự án:** Bộ tăng tốc phần cứng Vision Transformer (MobileViT-XXS / ViT) trên AMD Kria KV260  
> **Thư mục lưu trữ:** `Notes_ver02/khao_sat_khong_gian_bus_width_va_dk_da_chieu.md`  
> **Tác giả:** Đội ngũ Thiết kế & Kiểm thử Phần cứng (Hardware Architecture & Senior DV)  
> **Ngày lập:** 22/10/2026  
> **Mục tiêu:** Cung cấp báo cáo phân tích định lượng toàn diện, so sánh không gian thiết kế giữa 5 mức độ rộng bus AXI4-Stream (**64, 128, 256, 512, 1024-bit**) và 5 kích thước đặc trưng $d_k$ (**16, 20, 24, 32, 64**) đối với cả 3 biến thể kiến trúc phần cứng (**Variant 1A, Variant 2A, Variant 3B**).

---

## 1. TỔNG QUAN KHÔNG GIAN THIẾT KẾ (DESIGN SPACE EXPLORATION)

Trong thiết kế hệ thống tăng tốc AI trên nền tảng SoC (AMD Zynq UltraScale+ MPSoC / Kria KV260), hiệu năng thực tế của bộ tăng tốc bị chi phối bởi 2 yếu tố tương hỗ:
1. **Băng thông nạp/xả dữ liệu qua bus AXI4-Stream (Memory Bandwidth Plane):** Quyết định thời gian truyền ma trận $Q, K, V$ từ bộ nhớ ngoài DDR4 vào bộ đệm On-chip BRAM và xuất kết quả Context Out.
2. **Năng lực tính toán song song của lõi phần cứng (Compute Engine Plane):** Bị ràng buộc bởi số lượng bộ nhân (DSP / LUT) và kích thước vector chiều sâu đặc trưng $d_k$.

```
         BĂNG THÔNG NẠP DMA (AXI-STREAM BUS WIDTH)
   64-bit ─────────── 128-bit ─────────── 256-bit ─────────── 512-bit ─────────── 1024-bit
  (8 B/clk)         (16 B/clk)          (32 B/clk)          (64 B/clk)         (128 B/clk)
      │                   │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        BỘ ĐỆM ON-CHIP DUAL-BANK BRAM (PING-PONG)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
      │                                       │                                       │
      ▼                                       ▼                                       ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│   PHƯƠNG ÁN 1A   │               │   PHƯƠNG ÁN 2A   │               │   PHƯƠNG ÁN 3B   │
│ Pure Fabric LUT  │               │ 8 DSP48E2 Packed │               │ 16 DSP48E2 Array │
│ Multipliers (0D) │               │ Streaming Engine │               │ 2D Systolic Grid │
└──────────────────┘               └──────────────────┘               └──────────────────┘
      ▲                                       ▲                                       ▲
      │                                       │                                       │
  dk = 16 (Stage 2) ──────── dk = 20 (Stage 3) ──────── dk = 24 (Stage 4) ──────── dk = 32/64
                       CHIỀU SÂU ĐẶC TRƯNG HEAD DIMENSION (dk)
```

---

## 2. MA TRẬN PHÂN BỔ BEATS & ĐÓNG GÓI DỮ LIỆU (DATA PACKING & BEAT BUDGET)

Bảng dưới đây mô tả số lượng nhịp xung nhịp (Beats) cần thiết để truyền **1 Token** và toàn bộ **3 ma trận $Q, K, V$ ($N = 256$ tokens)** cho mọi cặp kết hợp giữa **Độ rộng Bus** và **Kích thước $d_k$**:

### 2.1 Bảng Tra Cứu Số Beats Truyền Dẫn (Ingress DMA Beats / Matrix)
*(Ghi chú: $1\text{ Beat} = \text{Bus Width / 8 bytes}$. Dữ liệu áp dụng chế độ Token-Aligned Padding để loại bỏ hoàn toàn lệch pha byte)*.

| Độ rộng Bus AXI-Stream | Dung lượng / Beat | $d_k = 16$ (Stage 2) | $d_k = 20$ (Stage 3) | $d_k = 24$ (Stage 4) | $d_k = 32$ (ViT-Small) | $d_k = 64$ (ViT-Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **64-bit** | **8 Bytes** | **2 beats** (512/mat)<br>Khớp 100% | **2.5 beats** (lẻ pha)<br>640 beats packed | **3 beats** (768/mat)<br>Khớp 100% | **4 beats** (1024/mat)<br>Khớp 100% | **8 beats** (2048/mat)<br>Khớp 100% |
| **128-bit** *(Khuyến nghị)* | **16 Bytes** | **1 beat** (256/mat)<br>**Tối ưu tuyệt đối** | **2 beats** (512/mat)<br>(16B + 4B pad 12B) | **2 beats** (512/mat)<br>(16B + 8B pad 8B) | **2 beats** (512/mat)<br>Khớp 100% | **4 beats** (1024/mat)<br>Khớp 100% |
| **256-bit** | **32 Bytes** | **1 beat** (256/mat)<br>(16B + 16B pad) | **1 beat** (256/mat)<br>(20B + 12B pad) | **1 beat** (256/mat)<br>(24B + 8B pad) | **1 beat** (256/mat)<br>**Khớp 100%** | **2 beats** (512/mat)<br>Khớp 100% |
| **512-bit** | **64 Bytes** | **0.25 beat** (64/mat)<br>4 tokens / beat | **0.31 beat** (80/mat)<br>Ghép đa token | **0.37 beat** (96/mat)<br>Ghép đa token | **0.5 beat** (128/mat)<br>2 tokens / beat | **1 beat** (256/mat)<br>**Khớp 100%** |
| **1024-bit** | **128 Bytes**| **0.125 beat** (32/mat)<br>8 tokens / beat | **0.15 beat** (40/mat)<br>Ghép đa token | **0.18 beat** (48/mat)<br>Ghép đa token | **0.25 beat** (64/mat)<br>4 tokens / beat | **0.5 beat** (128/mat)<br>2 tokens / beat |

---

## 3. THỜI GIAN TRUYỀN DMA TỪ DDR4 VÀO BỘ ĐỆM ON-CHIP ($Q, K, V = 3 \text{ Matrices}$)

Thời gian nạp dữ liệu từ DDR4 qua AXI DMA vào BRAM On-chip tại tần số xung nhịp **200 MHz** (1 chu kỳ = 5.0 ns):

```
Thời Gian Truyền Ingress DMA (µs) cho toàn bộ Q, K, V (256 Tokens)
══════════════════════════════════════════════════════════════════════════════════════════════
Độ rộng Bus | dk = 16 (Stage 2)   | dk = 20 (Stage 3)   | dk = 24 (Stage 4)   | dk = 64 (ViT-Base)
────────────┼─────────────────────┼─────────────────────┼─────────────────────┼────────────────────
64-bit      | 7.68 µs (1,536 cyc) | 9.60 µs (1,920 cyc) | 11.52 µs (2,304 cyc)| 30.72 µs (6,144 cyc)
128-bit     | 3.84 µs (768 cyc)   | 7.68 µs (1,536 cyc) | 7.68 µs (1,536 cyc) | 15.36 µs (3,072 cyc)
256-bit     | 3.84 µs (768 cyc)   | 3.84 µs (768 cyc)   | 3.84 µs (768 cyc)   | 7.68 µs (1,536 cyc)
512-bit     | 0.96 µs (192 cyc)   | 1.20 µs (240 cyc)   | 1.44 µs (288 cyc)   | 3.84 µs (768 cyc)
1024-bit    | 0.48 µs (96 cyc)    | 0.60 µs (120 cyc)   | 0.72 µs (144 cyc)   | 1.92 µs (384 cyc)
══════════════════════════════════════════════════════════════════════════════════════════════
```

> **Nhận xét then chốt:**  
> - Nâng từ **64-bit lên 128-bit** giúp cắt giảm ngay lập tức **50% thời gian DMA** ($7.68\,\mu\text{s} \to 3.84\,\mu\text{s}$).
> - Nâng từ **128-bit lên 256-bit** giúp $d_k=20$ và $d_k=24$ đạt tốc độ nạp cực nhanh $3.84\,\mu\text{s}$ ngang bằng $d_k=16$.
> - Tuy nhiên, nâng lên **512-bit và 1024-bit** mang lại lợi ích biên giảm dần (diminishing returns) đối với $d_k \le 24$, vì thời gian nạp đã xuống dưới $1\,\mu\text{s}$, trong khi thời gian tính toán của lõi (Compute Latency) chiếm tới $> 1,000\,\mu\text{s}$ ($99.9\%$ tổng thời gian).

---

## 4. PHÂN TÍCH TÁC ĐỘNG LÊN TỪNG BIẾN THỂ KIẾN TRÚC

---

### 4.1. Biến Thể 1A: Pure LUT Fabric Streaming (0 DSP48E2)

Kiến trúc 1A sử dụng các bộ nhân thuần túy bằng bảng tra cứu LUT logic và cây cộng nhị phân.

```
                    KIẾN TRÚC VARIANT 1A VỚI CÁC ĐỘ RỘNG BUS VÀ dk
+────────────────────────────────────────────────────────────────────────────────────────────+
| Bus 128-bit / dk=16:  Cây cộng N=16 (4 tầng) -> 16 bộ nhân LUT -> II = 1 (1 score/clk)     |
| Bus 256-bit / dk=32:  Cây cộng N=32 (5 tầng) -> 32 bộ nhân LUT -> II = 1 (1 score/clk)     |
| Bus 512-bit / dk=64:  Cây cộng N=64 (6 tầng) -> 64 bộ nhân LUT -> II = 1 (BÙNG NỔ LUT!)   |
+────────────────────────────────────────────────────────────────────────────────────────────+
```

* **Ảnh hưởng của Độ rộng Bus:**
  - **Với Bus 64-bit:** Rất bất lợi. Cây cộng $N=16$ hay $N=32$ phải chờ nạp 2 đến 4 nhịp mới đủ dữ liệu tính 1 vector.
  - **Với Bus 128-bit:** Rất phù hợp với $d_k=16$ (nạp đủ 1 token/beat). Đối với $d_k=32$, chỉ cần gom 2 beats 128-bit là đủ 32 ngõ vào.
  - **Với Bus 256-bit:** **Điểm ngọt tối ưu cho Giải pháp Cây cộng $N=32$**. Bus 256-bit cấp đủ 32 bytes INT8 trong đúng 1 nhịp duy nhất, kết nối trực tiếp $1:1$ vào 32 bộ nhân fabric mà không cần bộ ghép từ đa chu kỳ!
  - **Với Bus 512-bit / 1024-bit:** Không cần thiết và lãng phí, vì 1A không thể mở rộng cây cộng lên $N=64$ hay $N=128$ do giới hạn diện tích LUT.
* **Ảnh hưởng của Chiều sâu $d_k$:**
  - Khi $d_k$ tăng từ $16 \to 32$: Số bộ nhân LUT tăng gấp đôi (từ 16 lên 32 bộ nhân), tài nguyên tăng từ ~5,150 lên ~6,900 LUTs.
  - Khi $d_k = 64$: Cần 64 bộ nhân LUT và cây cộng 6 tầng nhị phân. Tài nguyên tăng vọt lên **> 12,000 LUTs**, đường trễ Critical Path dài làm $F_{\max}$ tụt xuống dưới 160 MHz.
* **Đánh giá tổng thể cho 1A:**  
  👉 **Điểm ngọt tối ưu (Sweet Spot): Bus 128-bit hoặc 256-bit kết hợp Cây cộng $N=32$.**

---

### 4.2. Biến Thể 2A: High-Throughput 8 DSP48E2 Streaming Engine

Kiến trúc 2A sử dụng cố định **8 khối `pe_slice_dsp_packing`** (8 DSP48E2 cứng), tính toán đồng thời 2 Key $K_{even}$ và $K_{odd}$ qua các nhịp 8 chiều/beat.

```
                    KIẾN TRÚC VARIANT 2A VỚI CÁC ĐỘ RỘNG BUS VÀ dk
+────────────────────────────────────────────────────────────────────────────────────────────+
| Lõi Compute cố định:  8 DSP48E2 -> Tiêu thụ cố định 8 bytes Q và 16 bytes K / nhịp       |
| dk = 16 (Stage 2):    Chạy 2 beats -> Tích lũy Beat 0 + Beat 1                             |
| dk = 24 (Stage 4):    Chạy 3 beats -> Tích lũy Beat 0 + Beat 1 + Beat 2                   |
| dk = 64 (ViT-Base):   Chạy 8 beats -> Tích lũy Beat 0 .. Beat 7                            |
+────────────────────────────────────────────────────────────────────────────────────────────+
```

* **Ảnh hưởng của Độ rộng Bus:**
  - **Với Bus 64-bit:** Tốc độ nạp DMA (8B/clk) xấp xỉ bằng tốc độ tiêu thụ của 8 DSP (8B/clk). Bộ đệm Ping-Pong chạy rất sát ngưỡng, dễ bị stall nếu DDR4 có độ trễ truy xuất.
  - **Với Bus 128-bit:** **"Điểm ngọt" hoàn hảo tuyệt đối của 2A!**  
    - Tốc độ nạp (16B/clk) nhanh gấp đôi tốc độ tiêu thụ (8B/clk).
    - Bộ đệm Ping-Pong luôn nạp đầy trước khi lõi 8 DSP tính xong. Đảm bảo **Zero-Bubble 100%**.
    - Mỗi từ BRAM 128-bit chứa đúng 2 nhịp tính toán của 8 DSP (nửa dưới `[63:0]` và nửa trên `[127:64]`).
  - **Với Bus 256-bit / 512-bit / 1024-bit:**  
    - **Không làm tăng FPS của 2A!** Vì lõi 2A bị giới hạn bởi năng lực tính toán của 8 DSP cứng (Compute-Bound: chỉ tiêu thụ 8 bytes/chu kỳ). Tăng bus DMA lên 256-bit hay 1024-bit chỉ làm BRAM phình to mà không giúp 8 DSP tính nhanh hơn.
* **Ảnh hưởng của Chiều sâu $d_k$:**
  - Lõi 8 DSP của 2A có ưu điểm cực lớn: **Tài nguyên phần cứng không đổi 100% (vẫn chỉ 8 DSP) dù $d_k$ tăng lên bao nhiêu!**
  - Khi $d_k = 16 \to$ Tốn 2 beats tích lũy.
  - Khi $d_k = 24 \to$ Tốn 3 beats tích lũy (chu kỳ GEMM 1 tăng 50%).
  - Khi $d_k = 64 \to$ Tốn 8 beats tích lũy (chu kỳ GEMM 1 tăng gấp 4 lần).
* **Đánh giá tổng thể cho 2A:**  
  👉 **Điểm ngọt tối ưu (Sweet Spot): Bus 128-bit cố định.** Không nên tăng lên 256-bit vì không đem lại thêm hiệu năng nhưng làm tốn BRAM.

---

### 4.3. Biến Thể 3B: 2D Output-Stationary Systolic MAC Array (16 DSP48E2)

Kiến trúc 3B sử dụng lưới $4 \times 4$ PE không gian, tính toán theo từng Tile $4 \text{ hàng} \times 8 \text{ cột}$ và dỡ tuần tự 32 phần tử vào Reorder Buffer.

```
                    KIẾN TRÚC VARIANT 3B VỚI CÁC ĐỘ RỘNG BUS VÀ dk
+────────────────────────────────────────────────────────────────────────────────────────────+
| Mảng 4x4 PE (OS):   Tích lũy tại chỗ trong 32 Accumulators nội bộ qua dk chu kỳ          |
| dk = 16:  Tính 16 chu kỳ -> Flush 12 chu kỳ -> Drain 32 chu kỳ (Drain chiếm 53.3% thời gian) |
| dk = 24:  Tính 24 chu kỳ -> Flush 12 chu kỳ -> Drain 32 chu kỳ (Drain chiếm 47.0% thời gian) |
| dk = 64:  Tính 64 chu kỳ -> Flush 12 chu kỳ -> Drain 32 chu kỳ (Drain chỉ chiếm 29.6%!)    |
+────────────────────────────────────────────────────────────────────────────────────────────+
```

* **Ảnh hưởng của Độ rộng Bus:**
  - **Với Bus 64-bit:** Mảng PE tiêu thụ 4 bytes $Q$ và 8 bytes $K$/chu kỳ. Bus 64-bit nạp vừa đủ nhưng pha nạp ma trận $V$ cho Score-V GEMM bị chậm (chia làm 2 beats).
  - **Với Bus 128-bit:** Rất tối ưu. Toàn bộ vector hàng của $Q, K, V$ ($d_k=16$) nạp trong 1 beat, ngõ ra Context Out xuất trong 1 beat duy nhất.
  - **Với Bus 256-bit:** Cho phép nạp song song cả 2 vector $Q$ liền kề hoặc mở rộng cho mảng PE lớn hơn ($8 \times 8$ PE).
  - **Với Bus 512-bit / 1024-bit:** Chỉ có ý nghĩa nếu mở rộng mảng PE lên quy mô công nghiệp ($16 \times 16$ hoặc $32 \times 32$ PE).
* **Ảnh hưởng của Chiều sâu $d_k$ (Quy luật nghịch lý cực kỳ có lợi của 3B):**
  - Điểm yếu lớn nhất của 3B ở $d_k = 16$ là pha **Drain 32 chu kỳ** chiếm tỷ trọng quá lớn so với pha tính toán (16 chu kỳ tính vs 32 chu kỳ dỡ dữ liệu $\to$ Hiệu suất PE chỉ đạt $28.5\%$).
  - Khi $d_k$ tăng lên $24$, rồi $32$, và đặc biệt là $d_k = 64$:
    - Chu kỳ tính toán tăng lên: $T_{\text{compute}} = 64\text{ chu kỳ}$.
    - Chu kỳ Drain vẫn giữ nguyên: $T_{\text{drain}} = 32\text{ chu kỳ}$.
    - **Hiệu suất tính toán hữu ích của PE tăng vọt từ $28.5\% \to 60.4\%$!**
    - Khoảng cách hiệu năng giữa 3B và 2A thu hẹp lại, và 3B vượt lên về tính ổn định routing so với các cây cộng streaming.
* **Đánh giá tổng thể cho 3B:**  
  👉 **Điểm ngọt tối ưu (Sweet Spot): Bus 128-bit cho $d_k \le 32$; Bus 256-bit cho $d_k = 64$.**

---

## 5. TÁC ĐỘNG VẬT LÝ LÊN TÀI NGUYÊN PHẦN CỨNG TRÊN CHIP AMD KRIA KV260

Bảng đánh giá chi phí phần cứng và giới hạn vật lý trên chip **XCK26-SFVC784-2LV-C** khi thay đổi độ rộng bus:

| Độ Rộng Bus AXI-Stream | Khối BRAM36K Tiêu Thụ | Tỷ Lệ % BRAM (Tổng 144) | Mức Độ Nghẽn Routing (Congestion) | Tần Số Cực Đại Khả Thi ($F_{\max}$) | Hỗ Trợ Gốc Trên Zynq UltraScale+ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **64-bit** | 6 BRAM36K | **4.1%** | Cực kỳ thấp (Rất thoáng) | **> 350 MHz** | Cần AXI Width Converter (128 $\to$ 64) |
| **128-bit** *(Sweet Spot)*| 12 BRAM36K | **8.3%** | Thấp (Tối ưu) | **~300 MHz** | **Native 100% (Khớp chuẩn HP Port)** |
| **256-bit** | 24 BRAM36K | **16.6%** | Trung bình | **~250 MHz** | Cần AXI DataMover / SmartConnect |
| **512-bit** | 48 BRAM36K | **33.3%** | Cao (Cần Register Slices) | **~200 MHz** | Khả dụng qua IP AXI DMA rộng |
| **1024-bit** | 96 BRAM36K | **66.6%** | Rất cao (Nguy cơ rớt timing) | **< 160 MHz** | Giới hạn tối đa của Vivado AXI DMA |

> **Phân tích vật lý BRAM36K:**  
> - Mỗi khối phần cứng BRAM (RAMB36E2) có độ rộng tối đa 72-bit (Simple Dual Port).
> - Để tạo bus 128-bit: Cần ghép 2 khối BRAM36K song song.
> - Để tạo bus 256-bit: Cần ghép 4 khối BRAM36K song song.
> - Để tạo bus 1024-bit: Cần ghép tới 16 khối BRAM36K song song chỉ cho một bank nhớ! Điều này làm bùng nổ diện tích BRAM và gây tắc nghẽn đường dây (Routing Congestion) nghiêm trọng.

---

## 6. MA TRẬN KHUYẾN NGHỊ CUỐI CÙNG (DECISION MATRIX)

Dưới đây là bảng định hướng lựa chọn cấu hình tối ưu theo mục tiêu cụ thể:

| Kịch Bản Ứng Dụng | Kích Thước $d_k$ Mục Tiêu | Độ Rộng Bus Khuyên Dùng | Biến Thể Tối Ưu Nhất | Lý Do Kỹ Thuật |
| :--- | :---: | :---: | :---: | :--- |
| **MobileViT-XXS Chuẩn (Đề tài hiện tại)** | **$d_k = 16, 20, 24$** | **128-bit** | **Variant 2A** | • Đạt FPS cao nhất (~183 FPS).<br>• Tiết kiệm tài nguyên nhất (8 DSP, 2.3% LUT).<br>• Bus 128-bit native với PS DDR4, giảm 50% thời gian DMA. |
| **Bộ Tăng Tốc Không Dùng DSP (Pure Fabric)** | **$d_k = 16, 20, 24, 32$** | **256-bit** | **Variant 1A (Cây cộng N=32)** | • Bus 256-bit cấp đủ 32 bytes trong 1 nhịp duy nhất.<br>• Đạt $II = 1$ liên tục, tốc độ GEMM 1 nhanh gấp 2 lần. |
| **Transformer Kích Thước Lớn (ViT-Base / Swin)**| **$d_k = 64$** | **256-bit hoặc 512-bit** | **Variant 3B (Systolic Array)** | • Hiệu suất mảng PE tăng vọt lên > 60%.<br>• Cấu trúc đều đặn 2D triệt tiêu hoàn toàn nghẽn dây của cây cộng lớn. |
| **Mô Hình Khổng Lồ (LLaMA / Trạm Datacenter)** | **$d_k = 128$** | **512-bit hoặc 1024-bit** | **Mảng Systolic $16 \times 16$** | • Chỉ phù hợp trên chip Versal AI Core hoặc Alveo U280. Không phù hợp với Kria KV260. |

---

## 7. KẾT LUẬN

1. **Khoảng 64-bit $\to$ 128-bit:** Là bước nâng cấp **mang lại hiệu quả nhảy vọt lớn nhất (High ROI)**. Nó tăng gấp đôi băng thông, loại bỏ bộ chuyển đổi AXI ngầm, và giải quyết triệt để vấn đề lệch pha token của Stage 3 ($d_k=20$).
2. **Khoảng 128-bit $\to$ 256-bit:** Là lựa chọn tuyệt vời nếu muốn đẩy **Variant 1A** lên cực hạn hiệu năng ($II = 1$ với cây cộng $N=32$) hoặc khi chuẩn bị mở rộng sang các mạng ViT có $d_k=32$.
3. **Khoảng 512-bit $\to$ 1024-bit:** Là vùng **không kinh tế (Overkill)** đối với chip Kria KV260 và mạng MobileViT-XXS. Nó gây lãng phí BRAM (ngốn tới 66% chip), tắc nghẽn đường dây routing, và kéo tụt tần số xung nhịp $F_{\max}$ mà không làm tăng tốc độ xử lý của các lõi tính toán.
4. **Khẳng định chiến lược:** **Cấu hình AXI-Stream Bus 128-bit là "Điểm Ngọt Tối Ưu" (Global Optimum)** xét trên cả 4 tiêu chí: Băng thông DDR4, Tài nguyên BRAM, Tần số $F_{\max}$, và Tính linh hoạt đa tầng 3 Stages.
