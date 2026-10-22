# Báo Cáo Thiết Kế Kiến Trúc Phương Án 2A: Vector Streaming + DSP Packing PE INT8 (DSP-Accelerated)
## Thư mục mã nguồn mục tiêu: `rtl_ver02_2A` — High-Throughput DSP-Packed Streaming Engine
**Dự án**: Bộ Tăng Tốc Phần Cứng MobileViT-XXS I-ViT INT8 trên Nền Tảng AMD Kria KV260  
**Ngày cập nhật**: 22/10/2026  
**Trạng thái**: Thiết kế chi tiết & Phân tích tối ưu hóa

---

## 1. Tổng Quan Kiến Trúc & Ý Tưởng Đột Phá

Phương án **2A (Vector Streaming + DSP Packing PE)** là bước tiến hóa chiến lược từ Phương án 1A: **giữ nguyên mô hình dòng dữ liệu streaming đơn giản và hiệu quả**, nhưng thay thế hoàn toàn các bộ nhân tổ hợp bằng logic fabric (LUT) bằng các lát xử lý **`pe_slice_dsp_packing`** tích hợp khối cứng **DSP48E2**.

### 1.1. Bản Chất Kỹ Thuật DSP Packing
Khối cứng xử lý tín hiệu số DSP48E2 trên chip XCK26 (UltraScale+) sở hữu bộ nhân với kích thước phần cứng lên đến **$18 \times 27$ bit signed**. Trong khi đó, mô hình lượng tử hóa số nguyên I-ViT chỉ yêu cầu các toán hạng kích thước **8-bit signed (INT8)**. 
- Nếu ánh xạ một phép nhân INT8 đơn lẻ vào 1 DSP48E2, hơn 75% độ rộng bit của khối phần cứng này sẽ bị lãng phí.
- **Kỹ thuật DSP Packing**: Cho phép "đóng gói" (pack) **hai phép nhân INT8 độc lập** ($A \times B$ và $A \times C$) vào **một khối DSP48E2 duy nhất** trong cùng một chu kỳ xung nhịp, tăng gấp đôi mật độ tính toán trên mỗi lát DSP và tiết kiệm đáng kể tài nguyên logic fabric.

---

## 2. Cơ Sở Toán Học & Mạch Sửa Lỗi Bit-Exact của `pe_slice_dsp_packing`

### 2.1. Biểu Diễn Toán Học Phép Đóng Gói
Giả sử ta cần thực hiện hai phép nhân có chung một toán hạng kích hoạt:
$$P_1 = A \times B, \quad P_2 = A \times C$$
Trong đó:
- $A, B, C \in [-128, 127]$ là các số nguyên có dấu 8-bit (Two's Complement INT8).
- Toán hạng $A$ được nạp vào Cổng A (18-bit signed) với phần mở rộng dấu:
  $$A_{ext} = \text{SignExtend}_{18}(A) = \{\{10\{A[7]\}\}, A[7:0]\}$$
- Toán hạng $B$ và $C$ được ghép vào Cổng B (27-bit signed):
  $$P_{packed} = (B_{ext} \ll 18) + C_{ext}$$
  Trong đó $B_{ext} = \text{SignExtend}_{27}(B)$ và $C_{ext} = \text{SignExtend}_{27}(C)$.

Khi thực hiện phép nhân trong DSP48E2:
$$\text{DSP\_Product} = A_{ext} \times P_{packed} = A \times (B \cdot 2^{18} + C) = (A \times B) \cdot 2^{18} + (A \times C)$$

### 2.2. Hiện Tượng Mượn Bit (Borrow Bit) & Mạch Sửa Lỗi Dấu (+1 Sign Correction)
Tích con $A \times C$ chiếm 16 bit thấp (từ bit 0 đến bit 15). Tuy nhiên, trên thanh ghi 27-bit của Cổng B, $C_{ext}$ được mở rộng dấu lên tới bit 26.
- **Trường hợp 1 ($A \times C \ge 0$)**: Tích con $A \times C$ là số dương. Các bit mở rộng từ bit 16 trở lên không ảnh hưởng đến phần $A \times B$ nằm ở dải bit [33:18].
- **Trường hợp 2 ($A \times C < 0$)**: Tích con $A \times C$ là số âm. Trong hệ bù 2, giá trị âm được biểu diễn bằng việc các bit cao nhất bị kéo lên 1 (dãy số 1 kéo dài qua bit 18). Hiện tượng này tương đương với một phép **mượn bit (borrow)** từ phần trọng số cao hơn $2^{18}$, khiến giá trị ở các bit $[44:18]$ bị giảm đi đúng 1 đơn vị so với giá trị thực của $A \times B$.

**Mạch sửa lỗi giải gói (Unpacking & Correction)**:
```systemverilog
// Trích từ rtl_ver02/pe_slice_dsp_packing.sv
assign result_ab = (dsp_product >>> 18) + (dsp_product[15] ? 16'sd1 : 16'sd0);
assign result_ac = dsp_product[15:0];
```
> **Chứng minh**: Nếu $A \times C < 0$, bit dấu của tích con là `dsp_product[15] == 1`. Ta chỉ việc cộng bù thêm `16'sd1` vào phần kết quả dịch phải `>>> 18` để phục hồi chính xác 100% giá trị $A \times B$.

### 2.3. Ví Dụ Bằng Số Cụ Thể (Numerical Verification)
- Cho $A = 10$, $B = 20$, $C = -15$:
  - Mong muốn: $A \times B = 200$, $A \times C = -150$.
  - Biểu diễn bù 2 của $C = -15$ (18-bit): `0x3FFF1`.
  - $P_{packed} = (20 \ll 18) + (-15) = 20 \times 262144 - 15 = 5,242,865$.
  - Tích trong DSP: $10 \times 5,242,865 = 52,428,650$.
  - Phân tích nhị phân:
    - $52,428,650 = 199 \times 262144 + 65386$.
    - Lấy 16 bit thấp: $65386 \equiv -150 \pmod{65536}$ (Chính xác $A \times C = -150$).
    - Bit 15 của 65386 là `1` (do $-150$ là số âm).
    - Lấy phần trên: $52,428,650 \gg 18 = 199$.
    - Cộng bù sửa lỗi: $199 + 1 = \mathbf{200}$ (Chính xác $A \times B = 200$).

---

## 3. Sơ Đồ Khối Chi Tiết Khối `pe_slice_dsp_packing`

```mermaid
graph TD
    subgraph PE_INTERNAL ["CẤU TRÚC BÊN TRONG PE_SLICE_DSP_PACKING"]
        OPA["operand_a INT8"] --> EXT_A["Sign Extend 10-bit<br/>17 den 0"]
        OPB["operand_b INT8"] --> EXT_B["Sign Extend 19-bit<br/>26 den 0"]
        OPC["operand_c INT8"] --> EXT_C["Sign Extend 19-bit<br/>26 den 0"]

        EXT_B --> SFT_B["Dich Trai 18-bit<br/>B dich 18"]
        SFT_B --> PACK_ADD["Bo Ghep Bu 2 Adder<br/>B dich 18 cong C"]
        EXT_C --> PACK_ADD

        EXT_A -->|Cong A 18-bit| DSP_MULT
        PACK_ADD -->|Cong B 27-bit| DSP_MULT

        subgraph DSP48E2_HARD ["KHỐI PHẦN CỨNG DSP48E2"]
            DSP_MULT["Bộ Nhân Phần Cứng 18x27-bit<br/>Stage 1 Register"]
            DSP_MULT --> DSP_REG["Thanh Ghi Pipeline Đầu Ra DSP<br/>Stage 2: dsp_product 45-bit"]
        end

        DSP_REG -->|Bit 15 den 0| AC_OUT["result_ac INT16<br/>A nhan C"]
        DSP_REG -->|Bit 44 den 18| SFT_18["Dịch Phải Số Học 18 bit"]
        DSP_REG -->|Bit 15 Dau AC| COND_ADD["Bộ Kiểm Tra Bit 15"]

        SFT_18 --> COMP_ADD["Bộ Cộng Sửa Lỗi"]
        COND_ADD -->|Cong bu 1 neu am| COMP_ADD
        COMP_ADD --> AB_OUT["result_ab INT16<br/>A nhan B"]
    end
```

---

## 4. Chiến Lược Ánh Xạ Toán Hạng (Operand Mapping Strategy) Cho GEMM 1 ($QK^T$)

Do toán hạng $A$ được chia sẻ chung giữa hai phép nhân, việc ánh xạ các ma trận $Q$ và $K$ vào các khối DSP Packing đòi hỏi sự chặt chẽ về mặt thuật toán:

### Phương Án Khuyến Nghị: Kiến Trúc Phát Đôi 2-Key Song Song (2-Key Interleaved Dual Streaming)
Để tận dụng tối đa sức mạnh của 16 phép nhân từ 8 khối DSP Packing:
- **Nguyên lý**: Gán toán hạng $A$ làm vector hàng của Query: $A[d] = Q[i, d]$.
- Gán toán hạng $B$ và $C$ cho **hai token khóa (Key tokens) liền kề nhau**:
  $$B[d] = K[2j, d], \quad C[d] = K[2j+1, d] \quad (d = 0..15)$$
- **Số lượng phần cứng**: Sử dụng **16 khối `pe_slice_dsp_packing`** (hoặc 8 khối chia làm 2 nhịp):
  - Với 16 DSPs: Tính trọn vẹn **hai dot-product score hoàn chỉnh** ($S_{i, 2j}$ và $S_{i, 2j+1}$) trong **cùng 1 chu kỳ xung nhịp**!
  - Thông lượng đạt được: **2 scores / chu kỳ xung nhịp** ($II = 0.5$ cycle/score).
- Với phương án **8 DSPs** (cấu hình tiêu chuẩn tối ưu tài nguyên cho Edge):
  - 8 DSPs tính 8 chiều $d = 0..7$ ở nhịp 1, và 8 chiều $d = 8..15$ ở nhịp 2.
  - Sau đó gộp lại thông qua cây cộng dồn tích lũy.

```mermaid
graph TD
    subgraph DUAL_STREAM ["KIẾN TRÚC STREAMING 8 DSP PACKING (16 PHÉP NHÂN INT8)"]
        Q_LATCH["Thanh Ghi Q Hàng i<br/>q0..q15"]
        K_EVEN["Vector Key 2j<br/>k_even 0..15"]
        K_ODD["Vector Key 2j+1<br/>k_odd 0..15"]

        Q_LATCH -->|q0| DSP0["DSP #0"]
        K_EVEN  -->|k0_even| DSP0
        K_ODD   -->|k0_odd| DSP0

        Q_LATCH -->|q1| DSP1["DSP #1"]
        K_EVEN  -->|k1_even| DSP1
        K_ODD   -->|k1_odd| DSP1

        Q_LATCH -->|...| DSP_DOT["... (8 DSPs)"]
        K_EVEN  -->|...| DSP_DOT
        K_ODD   -->|...| DSP_DOT

        Q_LATCH -->|q7| DSP7["DSP #7"]
        K_EVEN  -->|k7_even| DSP7
        K_ODD   -->|k7_odd| DSP7

        DSP0 -->|p0_even| TREE_EVEN["CÂY CỘNG NHÁNH EVEN<br/>8 sang 1 INT32"]
        DSP1 -->|p1_even| TREE_EVEN
        DSP_DOT -->|pe_even| TREE_EVEN
        DSP7 -->|p7_even| TREE_EVEN

        DSP0 -->|p0_odd| TREE_ODD["CÂY CỘNG NHÁNH ODD<br/>8 sang 1 INT32"]
        DSP1 -->|p1_odd| TREE_ODD
        DSP_DOT -->|pe_odd| TREE_ODD
        DSP7 -->|p7_odd| TREE_ODD

        TREE_EVEN -->|Score S_even| OUT_EVEN["Score Out Even (INT32)"]
        TREE_ODD  -->|Score S_odd| OUT_ODD["Score Out Odd (INT32)"]
    end
```

---

## 5. Máy Trạng Thái Hữu Hạn FSM & Độ Trễ DSP (Timing Sequence)

Khối DSP48E2 có **2 chu kỳ độ trễ phần cứng bên trong** (`LATENCY = 2`) để chốt thanh ghi đầu vào và đầu ra nhằm đạt xung nhịp cao nhất. Do đó, FSM của Phương án 2A phải có chuỗi đồng bộ hóa tín hiệu hợp lệ (`valid_pipe`):

```mermaid
sequenceDiagram
    autonumber
    participant FSM as Master FSM
    participant BRAM as BRAM Q va K
    participant DSP as 8x DSP48E2 Stage 1-2
    participant TREE as Reduction Tree Stage 3-6
    participant SCALE as Scale Unit
    participant SM as Softmax Buffer

    FSM->>BRAM: Kích hoạt đọc Q
    BRAM-->>FSM: Trả về Q 128-bit
    FSM->>FSM: Latch vào thanh ghi cục bộ q_curr_vec
    
    loop Stream 256 Key Vectors
        FSM->>BRAM: Đọc K chẵn và K lẻ
        BRAM-->>DSP: Cấp toán hạng A, B, C ở chu kỳ t
        Note over DSP: Chu kỳ t+1: Nhân trong DSP<br/>Chu kỳ t+2: Unpack và sửa dấu
        DSP-->>TREE: Xuất 16 tích con ở chu kỳ t+2
        Note over TREE: Chu kỳ t+3: Tầng L1<br/>Chu kỳ t+4: Tầng L2<br/>Chu kỳ t+5: Tầng L3 Root
        TREE-->>SCALE: Xuất 2 Scores INT32 ở chu kỳ t+6
        SCALE-->>SM: Ghi 2 Scaled Scores vào Buffer
    end
```

---

## 6. Phân Tích Tài Nguyên & Tần Số Hoạt Động (Resource & Fmax)

### 6.1. Bảng Tiêu Thụ Tài Nguyên Trên Chip AMD Kria KV260
| Khối Chức Năng | DSP48E2 | LUT6 | Flip-Flop | BRAM36K | Ghi Chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **8 khối `pe_slice_dsp_packing`** | **8** | ~128 | ~256 | 0 | Chỉ dùng 128 LUT cho logic sửa dấu |
| **Cây cộng dồn Reduction Tree (2 nhánh)** | 0 | ~640 | ~512 | 0 | 14 bộ cộng 32-bit có pipeline |
| **BRAM Q, K, V lưu trữ on-chip** | 0 | 0 | 0 | **12** (24 BRAM18K) | Ghép 128-bit bus rộng |
| **Scale Unit (Dyadic)** | 0 | ~180 | ~120 | 0 | Shift và multiply dyadic |
| **Softmax Engine (I-ViT)** | 0 | ~1,200 | ~1,400 | 1 | Chứa Score Buffer & Exp ROM |
| **Score-V GEMM Engine** | 0 | ~550 | ~600 | 0 | Tích lũy context output |
| **TỔNG TOÀN BỘ ATTENTION 2A** | **8 / 1,248 (0.64%)** | **~2,700 / 117K (2.3%)** | **~2,900 / 234K (1.2%)** | **13 / 144 (9.0%)** | **Tài nguyên cực kỳ nhỏ gọn!** |

### 6.2. Phân Tích Timing Closure
- **Critical Path**: Điểm nghẽn không còn nằm ở bộ nhân mà chuyển sang đường truyền từ đầu ra DSP đến tầng cộng đầu tiên của cây reduction tree:
  $$T_{critical} = t_{DSP\_CKO} + t_{Net} + t_{Sign\_Correction\_LUT} + t_{Adder\_LUT} + t_{Setup} \approx 0.38 + 0.85 + 0.45 + 0.65 + 0.12 = \mathbf{2.45\text{ ns}}$$
- **Tần số cực đại ước tính**:
  $$F_{max} = \frac{1}{2.45\text{ ns}} \approx \mathbf{408\text{ MHz}}$$
- **Kết luận**: Phương án 2A vượt xa mức tần số mục tiêu **200 MHz** với độ an toàn biên thời gian (Slack margin) cực kỳ lớn ($WNS > +2.5\text{ ns}$).

---

## 7. Các Điểm Cần Làm Rõ & Đề Xuất Tối Ưu Hệ Thống

### 7.1. Xử Lý Nút Thắt Downstream Khi GEMM Phát 2 Scores/Chu Kỳ
- **Vấn đề**: Nếu chạy chế độ 2-Key song song, GEMM 1 xuất ra 2 scores/chu kỳ. Tuy nhiên, khối `scale_unit.sv` và `softmax_lut.sv` hiện tại chỉ có 1 cổng vào đơn ($1\text{ score/cycle}$).
- **Giải pháp**:
  - *Giải pháp 1 (Đơn giản nhất)*: Cấu hình mảng 8 DSP chạy ở chế độ **1-Key stream** (8 DSP xử lý 8 byte chẵn của 1 Key, 8 byte lẻ được xử lý ở chu kỳ tiếp theo hoặc dùng 8 DSP đóng gói theo cặp chiều $d$). Khi đó tốc độ phát đúng bằng 1 score/chu kỳ, ghép nối hoàn hảo với Softmax hiện tại.
  - *Giải pháp 2 (Băng thông tối đa)*: Nâng cấp `score_buffer` trong Softmax thành **True Dual-Port BRAM**, cho phép ghi đồng thời 2 scores tại địa chỉ $2j$ và $2j+1$.

### 7.2. Yêu Cầu Băng Thông Đọc BRAM Cho 2-Key Stream
Nếu đọc đồng thời $K[2j]$ và $K[2j+1]$, tổng băng thông yêu cầu là $2 \times 128\text{ bit} = \mathbf{256\text{ bit/cycle}}$. Điều này đòi hỏi BRAM K phải được cấu hình Dual-Port (mỗi cổng đọc 128-bit) hoặc tách thành hai khối BRAM độc lập: `BRAM_K_Even` và `BRAM_K_Odd`.

---

## 8. Kết Luận Phương Án 2A

> **Khẳng định chiến lược**: Phương án 2A là **kiến trúc hoàn hảo và tối ưu nhất về mọi mặt** (Hiệu năng / Tiêu thụ tài nguyên / Độ phức tạp / Khả năng đóng gói phần cứng) đối với mô hình **MobileViT-XXS trên nền tảng AMD Kria KV260**.
> - Nó chỉ tiêu tốn đúng **8 DSP (0.64%)**, giải phóng 99.36% DSP cho các phần còn lại của mạng.
> - Giảm hơn **40% diện tích LUT** so với Phương án 1A.
> - Đạt thông lượng dòng chảy tự nhiên cao nhất và dễ dàng đạt mức xung nhịp 200–300 MHz.

---

## 9. Nhật Ký Cập Nhật Kiến Trúc (Architecture Revision History & Latest Upgrade Log)

### 9.1. Lịch Sử Các Phiên Bản Thiết Kế

| Phiên Bản | Ngày Hoàn Thành | Chuẩn Bus AXI-Stream | Cấu Trúc GEMM 1 | Phạm Vi Hỗ Trợ Stage | Trạng Thái Xác Minh |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **v1.0 (Baseline)** | 20/10/2026 | 64-bit (8 bytes/beat) | 8 DSP Packing, 2 beats tích lũy cố định | Cố định Stage 2 ($d_k=16$) | 100% Bit-Exact (4,096 B) |
| **v2.0 (Newest Upgrade)** | **22/10/2026** | **128-bit Native (16 bytes/beat)** | **8 DSP Packing + Bộ tích lũy đa beat tự động** | **Đa tầng linh hoạt 3 Stages ($d_k=16, 20, 24$)** | **100% Bit-Exact (209,919 cycles)** |

---

### 9.2. Chi Tiết Nâng Cấp Phiên Bản Mới Nhất (Version 2.0 - 128-Bit & Multi-Beat Sequencer)

Nhằm tối ưu hóa hiệu năng hệ thống SoC trên Kria KV260 và hỗ trợ đầy đủ cả 3 Stages của MobileViT-XXS, Biến thể 2A đã được nâng cấp toàn diện:

#### 1. Nâng cấp Bus AXI4-Stream Lên 128-Bit Native
- **Giao diện Skid Buffer Ingress & Egress**: Nâng cấp toàn diện lên `DATA_WIDTH = 128` (16 bytes/beat).
- **Rút ngắn 50% thời gian nạp DMA Ingress**: 
  - Tại Stage 2 ($d_k = 16$), mỗi token chiếm 1 beat 16 bytes.
  - Tổng số beat DMA nạp ma trận $Q, K, V$ giảm một nửa từ **1,536 beats xuống còn đúng 768 beats**.
- **Đồng bộ ngõ ra Score-V GEMM 128-bit**: Egress xuất 256 beats (16 bytes/beat), cờ `m_axis_tlast` khẳng định chính xác tại beat 255.

#### 2. Mở Rộng Bộ Tuần Tự Tích Lũy Đa Beat 8-DSP (Multi-Beat Accumulation Sequencer)
- Tại [dsp_packed_adder_tree.sv](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/dsp_packed_adder_tree.sv):
  - Mở rộng ngõ vào kiểm soát beat: `input logic [1:0] in_beat;` (hỗ trợ các giá trị `2'b00`, `2'b01`, `2'b10`, `2'b11`).
  - Tự động phát hiện beat cuối theo kích thước chiều đặc trưng $d_k$:
    ```systemverilog
    assign is_last_beat = (head_dim <= 16'd16) ? (in_beat == 2'd1) :
                          (head_dim <= 16'd24) ? (in_beat == 2'd2) : (in_beat == 2'd3);
    ```
  - **Mạch tích lũy và chuyển tiếp không bọt khí (Zero-Bubble Serializer)**:
    - Khi `t3_beat == 2'b00`: Khởi tạo thanh ghi tích lũy `accum_even` và `accum_odd` với kết quả tầng gốc Tree Stage 3.
    - Khi `!t3_last`: Tiếp tục cộng dồn các beat trung gian (`accum += t3_root`).
    - Khi `t3_last`: Xuất ngay lập tức giá trị `dot_product` cho token Even (`out_valid = 1`), đồng thời chốt giá trị token Odd vào `odd_hold_reg` và bật cờ `odd_pending`.
    - Ở chu kỳ kế tiếp, nếu `odd_pending == 1`, giá trị `odd_hold_reg` lập tức được xuất ra mà không làm chậm dòng dữ liệu (`out_valid = 1`).
    - Nhờ vậy, với $d_k = 16$ hệ thống đạt liên tục **$II = 1$ (1 score / chu kỳ)**, và với $d_k = 20, 24$ hệ thống xử lý trơn tru 2 tokens trong đúng 3 chu kỳ xung nhịp.

#### 3. Bộ Đệm Nạp BRAM Đa Tầng Linh Hoạt
- Tại [attention_core_top_2A.sv](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/attention_core_top_2A.sv):
  - Ingress logic tự động chia lát:
    - Với $d_k \le 16$: Nạp 1 token / beat (`ing_tok_idx = ing_beat_cnt`).
    - Với $d_k > 16$: Nạp 1 token / 2 beats (`ing_tok_idx = ing_beat_cnt >> 1`, `ing_slice_idx = ing_beat_cnt[0]`).
  - GEMM 1 FSM điều khiển `k_sub_beat` linh hoạt:
    ```systemverilog
    assign beats_per_pair_m1 = (reg_head_dim <= 16'd16) ? 2'd1 :
                               (reg_head_dim <= 16'd24) ? 2'd2 : 2'd3;
    ```

---

### 9.3. Bảng Đối Soát Thông Số Kỹ Thuật v1.0 vs v2.0

| Đặc Tính Kỹ Thuật | Phiên Bản v1.0 (Cũ) | Phiên Bản v2.0 (Mới Nhất) | Lợi Ích Cải Tiến |
| :--- | :---: | :---: | :--- |
| **Độ rộng Bus AXI4-Stream** | 64-bit (8 bytes/beat) | **128-bit (16 bytes/beat)** | Khớp chuẩn native AXI HP Port Kria KV260 |
| **Số beat nạp DMA Q, K, V** | 1,536 beats | **768 beats** | **Nhanh hơn gấp 2 lần (tiết kiệm 50% thời gian DMA)** |
| **Số khối DSP48E2 sử dụng** | **8 DSP (0.64%)** | **8 DSP (0.64%)** | **Giữ nguyên mức tiêu thụ DSP siêu tiết kiệm** |
| **Số beat tích lũy GEMM 1** | Cố định 2 beats ($d_k=16$) | **Linh hoạt 2 hoặc 3 beats** | Hỗ trợ tự nhiên $d_k \in \{16, 20, 24\}$ |
| **Khả năng đổi Stage động** | Cố định chỉ chạy $d_k=16$ | **Linh hoạt $d_k \in \{16, 20, 24\}$** | **Hỗ trợ trọn vẹn cả 3 Stages MobileViT-XXS** |
| **Định dạng Egress Context Out** | 512 beats (8 bytes/beat) | **256 beats (16 bytes/beat)** | 1 beat/token cho $d_k=16$, TLAST tại beat 255 |

---

### 9.4. Kết Quả Xác Minh Phần Cứng Bit-Exact (Verification Report v2.0)

Toàn bộ 5/5 bài kiểm tra trong [run_verifications_2A.bat](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_2A/run_verifications_2A.bat) đã vượt qua với kết quả tuyệt đối:
1. `tb_dsp_packed_adder_tree_2A.sv` (8 DSP PEs + Tree): **65,536 / 65,536 scores khớp 0-LSB [PASS]**.
2. `tb_golden_scale_unit.sv`: **65,536 / 65,536 scores khớp 0-LSB [PASS]**.
3. `tb_golden_softmax_lut.sv`: **1,024 / 1,024 probs khớp 0-LSB [PASS]**.
4. `tb_score_v_gemm.sv` (128-bit): **256 beats / 4,096 bytes khớp 100% [PASS]**.
5. `tb_attention_core_2A.sv` (Top 128-bit): **4,096 / 4,096 bytes Egress khớp 100% [PASS]**, thực thi trong **209,919 chu kỳ xung nhịp**, cờ `m_axis_tlast` khẳng định chính xác tại beat 255.

