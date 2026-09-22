# TẤT CẢ GHI CHÚ NGHIÊN CỨU & THIẾT KẾ CỦA DỰ ÁN (9/14/2026)
## FPGA-Based Edge AI Acceleration for Vision Transformers (ViT)

---

## MỤC LỤC
1. [Lý thuyết & Khung Kiến trúc Vision Transformer (ViT)](#1-lý-thuyết--khung-kiến-trúc-vision-transformer-vit)
2. [Định lượng Số nguyên INT8 & Công thức Dyadic Rescaling](#2-định-lượng-số-nguyên-int8--công-thức-dyadic-rescaling)
3. [Xấp xỉ Phi tuyến & Chuẩn hóa Thuần Số nguyên (ShiftGELU & I-LayerNorm)](#3-xấp-xỉ-phi-tuyến--chuẩn-hóa-thuần-số-nguyên-shiftgelu--i-layernorm)
4. [Tối ưu hóa Phần cứng RTL: DSP Packing & Pipelined Adder Tree](#4-tối-ưu-hóa-phần-cứng-rtl-dsp-packing--pipelined-adder-tree)
5. [Cấu trúc Mảng Systolic, QKV Attention & Residual Re-quantization](#5-cấu-trúc-mảng-systolic-qkv-attention--residual-re-quantization)
6. [Quản lý Bộ nhớ Phân cấp: SD Card, DDR4, AXI DMA & Ping-Pong BRAM](#6-quản-lý-bộ-nhớ-phân-cấp-sd-card-ddr4-axi-dma--ping-pong-bram)
7. [Luồng Thực thi Real-Time (Webcam / Laptop -> ARM PS -> FPGA PL)](#7-luồng-thực-thi-real-time-webcam--laptop---arm-ps---fpga-pl)
8. [Phương pháp luận Kiểm thử: Verilator, Cocotb, Golden Vector & PYNQ](#8-phương-pháp-luận-kiểm-thử-verilator-cocotb-golden-vector--pynq)

---

## 1. LÝ THUYẾT & KHUNG KIẾN TRÚC VISION TRANSFORMER (ViT)

### 1.1. Phân tách Patch & Embedding
Cho ảnh đầu vào $x \in \mathbb{R}^{H \times W \times C}$ (với $C=3$ kênh RGB):
- **Cắt Patch**: Ảnh được chia thành $N$ patches kích thước $(P, P)$:
  $$N = \frac{H \cdot W}{P^2}$$
  Ví dụ: Ảnh $224 \times 224$, patch $16 \times 16 \implies N = 196$ tokens.
- **Linear Projection & [CLS] Token**:
  Mỗi patch $x_p^i$ được chiếu tuyến tính qua ma trận trọng số $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ sang không gian ẩn $D$. Thêm token phân loại $x_{class} \in \mathbb{R}^{1 \times D}$ vào chỉ số $0$.
- **Positional Encoding**: Cộng ma trận $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ để giữ thông tin hình học không gian 2D.

### 1.2. Multi-Head Self-Attention (MSA)
- **Tạo $Q, K, V$**:
  $$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$
- **Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **Multi-Head Aggregation**:
  $$\text{MSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

---

## 2. ĐỊNH LƯỢNG SỐ NGUYÊN INT8 & CÔNG THỨC DYADIC RESCALING

### 2.1. Định lượng Uniform Symmetric INT8
Chuyển đổi số thực FP32 sang số nguyên có dấu INT8 $[-128, 127]$:
$$X_{float} = S_X \cdot I_X$$
Trong đó $S_X$ là Scale Factor (số thực FP32) và $I_X$ là giá trị số nguyên INT8.

### 2.2. Dyadic Rescaling ($M \cdot 2^{-e}$)
Khi nhân hai ma trận định lượng $I_A$ và $I_B$:
$$Y_{float} = S_A S_B (I_A I_B) = S_Y I_Y \implies I_Y = \left( \frac{S_A S_B}{S_Y} \right) (I_A I_B)$$
Tỉ số scale được xấp xỉ dạng Dyadic Number:
$$\frac{S_A S_B}{S_Y} \approx M \cdot 2^{-e}$$
Phép thu phóng trên phần cứng FPGA thực hiện thuần túy bằng số nguyên:
$$I_Y = \text{Clamp}_{\text{INT8}} \left( ( (I_A \times I_B) \cdot M ) \gg e \right)$$
- $M$: Hằng số nhân số nguyên INT32.
- $e$: Số bit dịch phải đại số (`>>>`).

---

## 3. XẤP XỈ PHI TUYẾN & CHUẨN HÓA THUẦN SỐ NGUYÊN

### 3.1. Mạch ShiftGELU (I-ViT)
Thay thế phép nhân hệ số $1.702$ của GELU bằng chuỗi dịch bit đại số và cộng số nguyên (0 DSPs):
$$I_p = 1.702 \cdot I_x \approx I_x + (I_x \gg 1) + (I_x \gg 3) + (I_x \gg 4)$$

#### Sơ đồ đường ống Pipelined Datapath:
```
[ I_in (INT8) ] -> [ Stage 0: 1.702x Shift-Add ] -> [ Stage 1: I_delta = Ip - I_max ]
                 -> [ Stage 2: ShiftExp / LUT Sigmoid ] -> [ Stage 3: IntDiv ]
                 -> [ Stage 4: Rescale & Clamp ] -> [ I_out (INT8) ]
```

### 3.2. Mạch I-LayerNorm
Chuẩn hóa LayerNorm không cần FPU hay phép căn số thực. Dùng thuật toán lặp Newton-Raphson số nguyên 10 vòng cố định tính $\sqrt{\text{Var}}$:
$$I_{i+1} = \left( I_i + \left\lfloor \frac{\text{Var}(I_x)}{I_i} \right\rfloor \right) \gg 1$$
Kết quả chuẩn hóa được nhân Dyadic Scale với $\gamma$ và cộng bias $\beta$ nguyên.

---

## 4. TỐI ƯU HÓA PHẦN CỨNG RTL: DSP PACKING & PIPELINED ADDER TREE

### 4.1. DSP Packing trên DSP48E2 (2x INT8 Multiplications)
Tận dụng bộ nhân $18 \times 27$-bit của khối DSP48E2 để nhân đồng thời 1 kích hoạt $A$ với 2 trọng số $B$ và $C$:
- Cổng 18-bit: Nạp $A$ (INT8)
- Cổng 27-bit: Nạp $(B \ll 18) + C$
- Tích ra 48-bit:
  $$\text{Result} = A \times ((B \ll 18) + C) = (A \times B) \ll 18 + (A \times C)$$
- Phân vùng bit: Tích $A \times C$ ở bit `[15:0]`, tích $A \times B$ ở bit `[33:18]`. Khoảng trống 18 bit làm Guard Bits chống tràn.
- **Tác động**: Tăng gấp 2 lần mật độ tính toán MAC (gấp 2.16x FPS/DSP).

### 4.2. Cây Cộng Pipelined (Pipelined Adder Tree)
Triệt tiêu đường Critical Path dài trong phép tính tích vô hướng $\sum_{i=0}^{N-1} A_i B_i$:
- Chèn thanh ghi Flip-Flop giữa từng tầng của cây cộng nhị phân $\log_2(N)$.
- Tự động quản lý Bit Growth ($\text{PROD\_WIDTH} \to \text{OUT\_WIDTH}$).
- Đạt tần số $F_{max} \ge 250 - 300\text{ MHz}$ với $II = 1$.

---

## 5. CẤU TRÚC MẢNG SYSTOLIC, QKV ATTENTION & RESIDUAL RE-QUANTIZATION

### 5.1. Tái sử dụng Mảng Systolic MAC Array (Unified GEMM Engine)
Mảng Systolic MAC Array $P_{sys} \times P_{sys}$ (Output-Stationary) linh hoạt chuyển đổi giữa 3 chế độ:
1. **LP Mode**: Linear Projections ($X \cdot W_Q, X \cdot W_K, X \cdot W_V$).
2. **MSA Mode**: 
   - Phase 1: Tính $Q \cdot K^T \to \text{Shift} \to \text{Softmax}$
   - Phase 2: Tính $\text{Score} \times V \to \text{Output}$
3. **MLP Mode**: Hidden Layer ($X \cdot W_1$) và Output Layer ($M \cdot W_2$).

### 5.2. Re-quantized Residual Addition
Căn chỉnh lệch Scale Factor ($S_F \neq S_X$) khi cộng đường tắt Shortcut:
$$I_{out} = \text{Clamp}_{\text{INT8}} \left( \left( (M_F \cdot I_F) \gg e_F \right) + \left( (M_X \cdot I_X) \gg e_X \right) \right)$$

---

## 6. QUẢN LÝ BỘ NHỚ PHÂN CẤP: SD CARD, DDR4, AXI DMA & PING-PONG BRAM

### 6.1. Kiến trúc Bộ nhớ 4 Tầng
```
[ SD Card ] ------------> [ DDR4 Memory ] ------------> [ AXI DMA Engine ] ------------> [ BRAM Ping-Pong Buffers ]
 (Lưu .pth,                (CMA Contiguous               (Burst Streaming                (Đệm đệm Tile nhỏ
  Weights & .hex)           Physical Buffers)             AXI4-Stream 64-bit)             cho Systolic Array)
```
- **Tiling Strategy**: Cắt ma trận lớn thành các Tile (ví dụ $64 \times 64$).
- **Ping-Pong Double Buffering**: Trong khi mảng PE đang tính toán trên đệm PING, AXI DMA nạp Tile tiếp theo từ DDR4 vào đệm PONG, ẩn hoàn toàn độ trễ đọc DRAM.

---

## 7. LUỒNG THỰC THI REAL-TIME (WEBCAM / LAPTOP -> ARM PS -> FPGA PL)

### 7.1. HW/SW Co-Design Partitioning
- **ARM PS (Processing System / Laptop Python)**:
  - Đọc luồng camera qua OpenCV (`cv2.VideoCapture`).
  - Preprocess: Resize $224 \times 224 \times 3$, Patch Embedding ($14 \times 14 \to N=196$ tokens), Định lượng QKV INT8.
  - Post-process: LayerNorm, MLP Head, Softmax Class, Render UI overlay (`cv2.imshow`).
- **FPGA PL (Programmable Logic / Attention Core)**:
  - Tăng tốc $O(N^2)$ Matrix Multiplications ($Q K^T$, $\text{Score} \times V$).
  - Scaling Shift ($1/\sqrt{d_k}$) và Hardware Softmax (LUT/PWL).

---

## 8. PHƯƠNG PHÁP LUẬN KIỂM THỬ: VERILATOR, COCOTB, GOLDEN VECTOR & PYNQ

### 8.1. Quy trình Đồng kiểm thử (HW/SW Co-Verification)
1. **Python / PyTorch (Colab)**: Huấn luyện mô hình, định lượng INT8, bắt tensor qua Forward Hooks, xuất file `q_tensor.hex`, `k_tensor.hex`, `golden_output.hex`.
2. **VS Code + Verilator / Icarus Verilog + Cocotb**:
   - Chạy mô phỏng RTL trực tiếp từ Jupyter Notebook trong VS Code bằng Python Cocotb testbench.
   - So sánh bit-exactness giữa RTL Output và Golden Output.
3. **Vivado IPI & STA**:
   - Đóng gói AXI IP Block, kiểm tra WNS/WHS timing closure ở $200\text{ MHz}$.
   - Biên dịch Bitstream (`.bit`) và Hardware Handoff (`.hwh`).
4. **Hardware-in-the-Loop (Kria KV260)**:
   - Nạp Overlay qua thư viện PYNQ Python.
   - Đo đạc FPS, Latency, Công suất (W) và Độ chính xác Top-1.
