# So Sánh Chi Tiết: Pipelined Adder Tree 5-Stage (1A) vs. Zero Bubble Serializer (2A)

Tài liệu này phân tích chi tiết vi kiến trúc, biểu đồ ASCII luồng dữ liệu (Dataflow), giản đồ thời gian (Timing Diagram) và cơ chế triệt tiêu bọt khí (Zero Bubble) giữa hai biến thể tính tích vô hướng $QK^T$ trong Attention Core trên AMD Xilinx Kria KV260:
1. **Variant 1A (`rtl_ver02_1A`)**: 32 bộ nhân Fabric LUT + **Cây cộng gộp Pipelined 5 tầng (Spatial Reduction Tree)**.
2. **Variant 2A (`rtl_ver02_2A`)**: 8 lát DSP48E2 Packed + **Cây cộng kép 3 tầng + Bộ tuần tự hóa triệt tiêu bọt khí (Zero Bubble Serializer)** ($II=1$).

---

## 1. Biểu Đồ ASCII Vi Kiến Trúc

### 1.1. Variant 1A: 5-Stage Pipelined Reduction Tree (Không Gian - Spatial)

```text
===================================================================================================================
                                      VARIANT 1A: 5-STAGE PIPELINED ADDER TREE
                                  (1 Beat/Token, II = 1 cycle, 0 DSP, 100% LUT Logic)
===================================================================================================================

 [Stationary Q Vector]              [Streaming K Token j] (1 beat từ BRAM 128-bit/256-bit)
 q_i[0:31] (32 x INT8)              K[j, 0:31] (32 x INT8 zero-padded nếu d_k < 32)
       │                                   │
       └─────────────────┬─────────────────┘
                         ▼
        ┌────────────────────────────────────────────────────────┐
        │       32 BỘ NHÂN INT8xINT8 THUẦN TÚY BẰNG LUT LOGIC     │  Latency: 1 cycle
        │ prod[0]  prod[1]  prod[2]  prod[3] ... prod[30] prod[31]│
        └────────────────────────┬───────────────────────────────┘
                                 │ 32 x INT16 (Cycle 0)
                                 ▼
        ══════════════════════════════════════════════════════════ Pipeline Register 0
        ┌────────────────────────────────────────────────────────┐
        │      TẦNG 1: 16 BỘ CỘNG INT32 (32 phần tử -> 16 sums)   │  [Adder Stage 1]
        │   (+)      (+)      (+)      (+)   ...   (+)      (+)  │
        └────────────────────────┬───────────────────────────────┘
                                 │ 16 x INT32 (Cycle 1)
                                 ▼
        ══════════════════════════════════════════════════════════ Pipeline Register 1
        ┌────────────────────────────────────────────────────────┐
        │      TẦNG 2: 8 BỘ CỘNG INT32  (16 phần tử -> 8 sums)   │  [Adder Stage 2]
        │        (+)              (+)        ...        (+)      │
        └────────────────────────┬───────────────────────────────┘
                                 │ 8 x INT32 (Cycle 2)
                                 ▼
        ══════════════════════════════════════════════════════════ Pipeline Register 2
        ┌────────────────────────────────────────────────────────┐
        │      TẦNG 3: 4 BỘ CỘNG INT32  (8 phần tử -> 4 sums)    │  [Adder Stage 3]
        │              (+)                   ...        (+)      │
        └────────────────────────┬───────────────────────────────┘
                                 │ 4 x INT32 (Cycle 3)
                                 ▼
        ══════════════════════════════════════════════════════════ Pipeline Register 3
        ┌────────────────────────────────────────────────────────┐
        │      TẦNG 4: 2 BỘ CỘNG INT32  (4 phần tử -> 2 sums)    │  [Adder Stage 4]
        │                     (+)                        (+)     │
        └────────────────────────┬───────────────────────────────┘
                                 │ 2 x INT32 (Cycle 4)
                                 ▼
        ══════════════════════════════════════════════════════════ Pipeline Register 4
        ┌────────────────────────────────────────────────────────┐
        │      TẦNG 5: 1 BỘ CỘNG INT32  (2 phần tử -> 1 scalar)  │  [Adder Stage 5]
        │                              (+)                       │
        └────────────────────────┬───────────────────────────────┘
                                 │ 1 x INT32 (Cycle 5)
                                 ▼
        ══════════════════════════════════════════════════════════ Final Output Register
                                 │
                                 ▼
                   s_raw[i, j] = q_i . K[j, :] (INT32)
              [Xuất liên tục 1 score / cycle tới Stage 2 Scaler]
```

---

### 1.2. Variant 2A: Zero Bubble Serializer (Interleaved 2-Key Streaming + Dual 3-Stage Adder Trees)

```text
===================================================================================================================
                                      VARIANT 2A: ZERO BUBBLE SERIALIZER
            (Interleaved 2-Key Streaming, 8 DSP48E2 Packed, Dual 3-Stage Trees, II = 1 cycle sustained)
===================================================================================================================

 [Stationary Q Vector]              [Streaming Key Even & Odd Token Slices]
 Latch q_i[8b : 8b+7] (8 INT8)      k_even[8b : 8b+7] & k_odd[8b : 8b+7] (2 x 8 INT8 / beat, b = 0 .. n_beats-1)
       │                                   │                 │
       └─────────────────┬─────────────────┴─────────────────┘
                         ▼
        ┌────────────────────────────────────────────────────────┐
        │     8 x DSP48E2 SLICES (pe_slice_dsp_packing)          │  Latency: 2 cycles
        │   PE#0      PE#1      PE#2      PE#3   ...   PE#7      │  (Multiplier + Unpack & Sign Correction)
        │ [DSP48E2] [DSP48E2] [DSP48E2] [DSP48E2]   [DSP48E2]    │  Mỗi DSP tính đồng thời: A*B và A*C
        └──────────────┬───────────────────────────┬─────────────┘
                       │ 8 x INT16 (Even Branch)   │ 8 x INT16 (Odd Branch)
                       ▼                           ▼
        ┌──────────────────────────────┐ ┌──────────────────────────────┐
        │  CÂY CỘNG NHỊ PHÂN EVEN      │ │  CÂY CỘNG NHỊ PHÂN ODD       │  Latency: 3 cycles
        │  3 Tầng (8 -> 4 -> 2 -> 1)   │ │  3 Tầng (8 -> 4 -> 2 -> 1)   │  (Pipeline Regs t1, t2, t3)
        └──────────────┬───────────────┘ └──────────────┬───────────────┘
                       │ 1 x INT19 (t3_even)            │ 1 x INT19 (t3_odd)
                       ▼                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                        ZERO BUBBLE SERIALIZER ENGINE                            │
  │                                                                                 │
  │        ┌────────────────────────────┐      ┌────────────────────────────┐       │
  │        │   BỘ TÍCH LŨY BEAT EVEN    │      │    BỘ TÍCH LŨY BEAT ODD    │       │
  │        │   accum_even <= accum_even │      │    accum_odd <= accum_odd  │       │
  │        │              + t3_even     │      │              + t3_odd      │       │
  │        └─────────────┬──────────────┘      └─────────────┬──────────────┘       │
  │                      │                                   │                      │
  │                      │ (Khi t3_last)                     │ (Khi t3_last)        │
  │                      ▼                                   ▼                      │
  │        ┌────────────────────────────┐      ┌────────────────────────────┐       │
  │        │   Xuất ngay điểm số Even   │      │  Chốt vào odd_hold_reg     │       │
  │        │   dot_product <= Even_sum  │      │  odd_pending <= 1'b1       │       │
  │        │   out_valid <= 1'b1        │      │                            │       │
  │        │   (Chu kỳ T)               │      │  (Chu kỳ T)                │       │
  │        └─────────────┬──────────────┘      └─────────────┬──────────────┘       │
  │                      │                                   │                      │
  │                      │                                   ▼ (Chu kỳ T+1)         │
  │                      │                     ┌────────────────────────────┐       │
  │                      │                     │  Xuất tiếp điểm số Odd     │       │
  │                      │                     │  dot_product <= hold_reg   │       │
  │                      │                     │  out_valid <= 1'b1         │       │
  │                      │                     └─────────────┬──────────────┘       │
  │                      └─────────────────┬─────────────────┘                      │
  └────────────────────────────────────────┼────────────────────────────────────────┘
                                           │
                                           ▼
                      s_raw[i, j] = q_i . K[j, :] (INT32)
       [DÒNG ĐIỂM SỐ XUẤT LIÊN TỤC 1 SCORE / CYCLE SUỐT N=256 TOKENS (II = 1)]
```

---

## 2. Giản Đồ Thời Gian (Timing Waveforms)

### 2.1. Giản đồ Thời gian Variant 1A: Tuyến tính & Liên tục ($II = 1$)
* Giả sử nạp lần lượt các token $K_0, K_1, K_2, K_3, \dots$ mỗi chu kỳ 1 token.
* Đường ống làm đầy sau 5 chu kỳ. Sau đó, **mỗi chu kỳ xuất đều đặn 1 kết quả chấm điểm**:

```text
Clock Cycle:   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ 10│
───────────────┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Input K Token: │K0 │K1 │K2 │K3 │K4 │K5 │K6 │K7 │K8 │K9 │K10│
LUT Mult (32)  │K0 │K1 │K2 │K3 │K4 │K5 │K6 │K7 │K8 │K9 │K10│
Stage 1 (16)   │ - │K0 │K1 │K2 │K3 │K4 │K5 │K6 │K7 │K8 │K9 │
Stage 2 (8)    │ - │ - │K0 │K1 │K2 │K3 │K4 │K5 │K6 │K7 │K8 │
Stage 3 (4)    │ - │ - │ - │K0 │K1 │K2 │K3 │K4 │K5 │K6 │K7 │
Stage 4 (2)    │ - │ - │ - │ - │K0 │K1 │K2 │K3 │K4 │K5 │K6 │
Stage 5 (1)    │ - │ - │ - │ - │ - │K0 │K1 │K2 │K3 │K4 │K5 │
───────────────┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Output Score:  │ - │ - │ - │ - │ - │S0 │S1 │S2 │S3 │S4 │S5 │  <== II = 1 (1 score/cycle!)
Output Valid:  │ 0 │ 0 │ 0 │ 0 │ 0 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │
```

---

### 2.2. Giản đồ Thời gian Variant 2A: Dòng Chảy Liên Tục Tuyệt Đối ($II = 1$) Nhờ Interleaving

Trong kiến trúc thực tế [`dsp_packed_adder_tree.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/dsp_packed_adder_tree.sv), ta nạp đồng thời **cặp token Even & Odd** qua các sub-beat ($d_k = 16 \implies n_{\text{beats}} = 2$):
* **Chu kỳ $2p$**: Nạp Beat 0 của cặp token $p$ (chứa $K_{2p}[0:7]$ và $K_{2p+1}[0:7]$).
* **Chu kỳ $2p+1$**: Nạp Beat 1 của cặp token $p$ (chứa $K_{2p}[8:15]$ và $K_{2p+1}[8:15]$).

Sau độ trễ lấp đầy pipeline (Fill Latency = 6 chu kỳ: 2 DSP + 3 Adder Tree + 1 Serializer):
* Tại chu kỳ hoàn tất Beat 1: Điểm số token Even ($S_{2p}$) được xuất ngay lập tức ra `dot_product`.
* Tại chu kỳ kế tiếp: Điểm số token Odd ($S_{2p+1}$) được xuất ngay lập tức từ `odd_hold_reg`.
* Do cặp token tiếp theo ($p+1$) được nạp gối đầu liên tục ngay sau đó, ngõ ra `out_valid` **luôn luôn duy trì mức cao liên tục 100% không hề có bọt khí**:

```text
Clock Cycle:      │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ 10│ 11│
──────────────────┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Input K Stream:   │P0a│P0b│P1a│P1b│P2a│P2b│P3a│P3b│P4a│P4b│P5a│P5b│ (P_x = Cặp Token 2x & 2x+1)
8x DSP48E2 PEs    │P0a│P0b│P1a│P1b│P2a│P2b│P3a│P3b│P4a│P4b│P5a│P5b│ (Mỗi DSP tính 1 Even + 1 Odd)
Dual Trees (3-stg)│ - │ - │P0a│P0b│P1a│P1b│P2a│P2b│P3a│P3b│P4a│P4b│ (Cộng gộp nhị phân 8->4->2->1)
Beat Accumulator  │ - │ - │ - │ - │P0 ├───┤P1 ├───┤P2 ├───┤P3 ├───┤ (Tích lũy trọn vẹn qua 2 beats)
Serializer Engine │ - │ - │ - │ - │ - │ - │S0 │S1 │S2 │S3 │S4 │S5 │ (S0 từ Even, S1 từ odd_hold)
──────────────────┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Output Score:     │ - │ - │ - │ - │ - │ - │S0 │S1 │S2 │S3 │S4 │S5 │ <== II = 1 (1 score / cycle liên tục!)
Output Valid:     │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │     100% HIỆU SUẤT KHÔNG STALL!
```

---

## 3. Bảng So Sánh Kỹ Thuật Chi Tiết

| Thuộc tính | Variant 1A: 5-Stage Adder Tree | Variant 2A: Zero Bubble Serializer | Nhận Xét & Lợi Thế |
| :--- | :--- | :--- | :--- |
| **Bản chất kiến trúc** | **Spatial Reduction** (Cộng dồn không gian thuần túy) | **Interleaved Spatial-Temporal** (2 cây cộng 3 tầng + Tích lũy đa beat xen kẽ) | 2A kết hợp tối ưu cả không gian và thời gian |
| **Không gian phần cứng** | 32 bộ nhân LUT + 31 bộ cộng INT32 trong Fabric | 8 DSP48E2 slices + Dual 3-Stage Binary Trees (Fabric) | 2A khai thác tính năng DSP Packing $18 \times 27$ bit |
| **DSP48E2 Count** | **0 DSP** (Tiết kiệm 100% DSP) | **8 DSP** (Chỉ 0.64% tổng tài nguyên KV260) | 2A tiêu tốn cực ít DSP |
| **LUT 6-input Usage** | ~4,500 – 5,200 LUTs (Khá lớn) | ~2,400 – 2,900 LUTs (**Tiết kiệm ~44%**) | **2A giải phóng ~2,000 LUTs cho các khối khác** |
| **Độ rộng Bus BRAM** | 128-bit Native (1 token / beat cho $d_k=16$) | 128-bit Native (2 tokens / 2 beats cho $d_k=16$) | Cả hai cùng khai thác tối đa bus 128-bit KV260 |
| **Initiation Interval ($II$)**| **$II = 1$ cycle / score** | **$II = 1$ cycle / score sustained** | **Ngang ngửa tuyệt đối về thông lượng!** |
| **Fill Latency** | 5 chu kỳ clock (5 tầng cộng nhị phân) | 6 chu kỳ clock (2 DSP + 3 Trees + 1 Serializer) | Chênh lệch vỏn vẹn 1 chu kỳ |
| **Thời gian quét $N=256$**| $256 + 5 = \mathbf{261\text{ cycles}}$ | $128 \times 2 + 6 = \mathbf{262\text{ cycles}}$ (với $d_k=16$) | **Tốc độ thực thi tương đương 99.6%!** |
| **Fmax ước tính trên KV260**| ~200 – 230 MHz (Nhiều tầng carry-chain) | ~250 – 300 MHz (DSP hard blocks + cây cộng nhỏ) | 2A đạt tần số xung nhịp cao hơn |
| **Khả năng ghép nối Softmax**| Đẩy 1 score/cycle liên tục; khớp trực tiếp với Softmax. | Đẩy 1 score/cycle liên tục; khớp trực tiếp với Softmax. | Cả hai đều giao tiếp 1 cổng đơn cực kỳ mượt mà |

---

## 4. Phân Tích Kiến Trúc: Sự Kết Hợp Adder Tree & Serializer Trong 2A

Một hiểu lầm thường gặp khi đọc tài liệu cũ là: *"Variant 1A dùng Adder Tree, còn Variant 2A thì không dùng Adder Tree mà chỉ dùng Accumulator"*. 

Trên thực tế kiểm toán mã nguồn RTL:
1. **Variant 2A THỰC CHẤT CÓ DÙNG ADDER TREE:**
   - Bên trong [`dsp_packed_adder_tree.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/dsp_packed_adder_tree.sv), có tới **hai cây cộng nhị phân 3 tầng** ($8 \to 4 \to 2 \to 1$): một cây cho nhánh chẵn (`t1_even`, `t2_even`, `t3_even`) và một cây cho nhánh lẻ (`t1_odd`, `t2_odd`, `t3_odd`).
   - Hai cây cộng này gộp 8 tích riêng INT16 của 8 lát DSP thành 1 tổng INT19 ngay trong cùng một chu kỳ xung nhịp (Spatial Reduction).
2. **Vai trò của Temporal Accumulator & Serializer:**
   - Do kích thước vector $d_k = 16$ (hoặc $20, 24$) lớn hơn số lượng DSP vật lý trong một nhánh (8 DSPs), việc tính toán bắt buộc phải trải qua nhiều beat thời gian ($n_{\text{beats}} = 2$ cho $d_k=16$; $n_{\text{beats}} = 3$ cho $d_k=20, 24$).
   - Mạch `accum_even` và `accum_odd` thực hiện cộng dồn giữa các beat theo thời gian.
   - Khi beat cuối cùng hoàn tất, bộ tuần tự hóa `Zero_Bubble_Serializer` phân bổ điểm số Even ra trước và điểm số Odd ra ngay sau đó, biến dòng dữ liệu đa beat thành dòng xuất điểm số đơn đều đặn mỗi chu kỳ một giá trị ($II=1$).
3. **Về đề xuất nâng cấp 16 DSPs (Full Spatial DSP Tree):**
   - Nếu nâng số lượng lên 16 DSPs, ta có thể xây dựng một cây cộng nhị phân 16 $\to$ 1 (hoặc hai cây 8 $\to$ 1 không cần tích lũy thời gian cho $d_k=16$).
   - Tuy nhiên, kiến trúc 8 DSPs Interleaved hiện tại đã đạt được mục tiêu tối thượng: **$II=1$ và quét 256 tokens chỉ trong 262 chu kỳ**, mà lại chỉ tiêu thụ đúng **8 DSP (0.64% chip KV260)**. Đây là tỷ lệ đánh đổi hiệu năng trên tài nguyên hoàn hảo nhất.

---

## 5. Tóm Tắt & Kết Luận Chọn Lựa

1. **Chọn Variant 1A (5-Stage Pure LUT Tree)** khi:
   * FPGA đang cạn kiệt hoàn toàn DSP do các tác vụ khác (Convolution, Pointwise Conv, Depthwise, MLP) đã chiếm hết 100% DSP trên chip.
   * FPGA còn dư dả tài nguyên logic LUT (> 80,000 LUTs rảnh rỗi).

2. **Chọn Variant 2A (Interleaved DSP-Packed + Zero Bubble Serializer)** khi:
   * **Cần giải pháp tối ưu toàn diện nhất**: Đạt tốc độ ngang ngửa 1A (262 chu kỳ vs 261 chu kỳ, $II=1$), nhưng **tiết kiệm tới 44% diện tích LUT** (~2,000 LUTs) và chỉ sử dụng đúng **8 DSPs (0.64%)**.
   * Cần đạt tần số xung nhịp cao ($F_{\max} \ge 250\text{ MHz}$) và routing sạch trên FPGA fabric.
   * Khẳng định tính ưu việt của thiết kế vi mạch trong luận văn: khai thác triệt để tính năng DSP Packing $18 \times 27$ bit chuyên sâu của dòng chip AMD UltraScale+.
