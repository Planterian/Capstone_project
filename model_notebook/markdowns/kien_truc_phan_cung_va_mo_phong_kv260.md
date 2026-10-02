# Đặc Tả Kiến Trúc Phần Cứng Bo Mạch AMD Kria KV260 & Mô Phỏng Bằng Phần Mềm (Software Emulation)

---

## 1. Giới Thiệu Về Nền Tảng AMD Kria KV260 Vision AI Starter Kit

Bo mạch **AMD Kria KV260** là nền tảng Edge AI chuyên dụng cho xử lý thị giác máy tính với kiến trúc vi xử lý đa nhân không đồng nhất (Heterogeneous Computing) dựa trên chip **Zynq UltraScale+ MPSoC (ZCU102/XCK26)**:

```text
=========================================================================================================
                                KIẾN TRÚC PHẦN CỨNG KRIA KV260 (HW/SW CO-DESIGN)
=========================================================================================================

+-------------------------------------------------------------------------------------------------------+
|  ARM PROCESSING SYSTEM (PS) - Lõi Xử Lý Phần Mềm (Quad-core Cortex-A53 @ 1.33 GHz)                     |
|                                                                                                       |
|  - Hệ điều hành: Ubuntu 22.04 LTS / PYNQ Linux 3.0                                                    |
|  - Thư viện ứng dụng: Python 3.10+, OpenCV (UVC Camera Capture), NumPy, PyTorch (Host Runtime)         |
|  - Chức năng:                                                                                         |
|      1. Thu thập luồng camera USB 720p @ 30 FPS qua giao thức V4L2 (/dev/video0)                      |
|      2. Tiền xử lý (Preprocessing): Center-crop, Resize 32x32, RGB, CIFAR-10 Normalization           |
|      3. Patch Embedding & QKV Linear Projections: Chuẩn bị ma trận INT8 (Q, K, V)                     |
|      4. Điều khiển và cấu hình phần cứng qua thanh ghi AXI-Lite (Memory-Mapped I/O)                   |
|      5. Hậu xử lý (Postprocessing): Tính toán LayerNorm, MLP Head, Softmax dự đoán lớp                |
|      6. Render giao diện HUD đồ họa thời gian thực lên màn hình                                        |
+-------------------------------------------------------------------------------------------------------+
                                  │                                  ▲
        AXI-Lite Control Register │                                  │ AXI-Lite Status Register
        (START=1, N=65, d_k=32)   ▼                                  │ (DONE=1, IDLE=0)
+───────────────────────────────────────────────────────────────────────────────────────────────────────+
|  GIAO TIẾP DỮ LIỆU BỘ NHỚ: AXI DMA (Direct Memory Access Engine) + CMA (Contiguous Memory Allocation) |
|  - Băng thông bus: AXI4-Stream 64-bit @ 200 MHz = 1.6 GB/s                                            |
|  - Kênh MM2S (Memory to Stream): Đẩy Q, K, V từ RAM PS xuống FPGA PL                                  |
|  - Kênh S2MM (Stream to Memory): Nhận ma trận Attended Context từ FPGA PL trả về RAM PS               |
+───────────────────────────────────────────────────────────────────────────────────────────────────────+
                                  │                                  ▲
                    M_AXIS_MM2S   │                                  │ S_AXIS_S2MM
                    (Q, K, V INT8)▼                                  │ (Attended Context INT8)
+-------------------------------------------------------------------------------------------------------+
|  FPGA PROGRAMMABLE LOGIC (PL) - Khối Tăng Tốc Phần Cứng Attention Core (Bitstream: attention_core.bit)|
|                                                                                                       |
|  - Khối nhớ đệm vào: Dual-port Ping-Pong BRAM (Lưu trữ Q, K, V với độ trễ truy xuất 1 chu kỳ)          |
|  - Systolic Array 1: Khối nhân ma trận Q * K^T (INT8 x INT8 -> INT32 Tích luỹ)                       |
|  - Scaler Unit: Dịch bit phải số học đại số ASR 1/sqrt(d_k) -> INT16                                  |
|  - Hardware Softmax Unit: Xấp xỉ từng đoạn PWL / Bảng tra LUT 256 phần tử -> INT8 Xác suất            |
|  - Systolic Array 2: Khối nhân Score * V (INT8 x INT8 -> INT8 Kết quả ngữ cảnh)                       |
|  - Khối đệm ra: Output Ping-Pong BRAM -> Đẩy ra bus AXI4-Stream Master                                |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Cơ Chế Mô Phỏng Bằng Phần Mềm (Software-Only Emulation Engine)

Vì đang chạy trên môi trường giả lập (không cắm trực tiếp cáp vật lý vào KV260), mô-đun phần mềm của chúng ta tái hiện trung thực 100% các ràng buộc và thông số kỹ thuật của KV260:

### 2.1. Phân chia tác vụ Phần Mềm / Phần Cứng (HW/SW Partitioning)
| Thành phần tác vụ | Thực thi vật lý trên KV260 | Mô phỏng trong phần mềm |
| :--- | :--- | :--- |
| **Đọc Video USB 720p** | ARM PS (`cv2.VideoCapture`) | `ThreadedCameraStream` (OpenCV Background Thread) |
| **Tiền xử lý ảnh (Crop, Resize, Norm)** | ARM PS (Cortex-A53) | `PSPreprocessor` (NumPy + PyTorch CPU) |
| **Bộ nhớ chia sẻ (CMA Allocation)** | Nhân Linux PYNQ (`pynq.allocate`) | `CMABufferPool` (Mô phỏng mảng byte liên tục) |
| **Truyền nhận AXI DMA** | IP Core AXI DMA (PL) | `AXIDMASimulator` (Tính toán độ trễ theo băng thông 1.6 GB/s) |
| **Khối Attention Core** | Vi mạch Systolic RTL (PL) | `PLAttentionEngine` (Chạy ma trận lượng tử INT8 + đo chu kỳ) |
| **MLP Head & Phân loại** | ARM PS (Cortex-A53) | `MLPHeadRunner` (Tầng Fully-Connected INT8) |
| **Hiển thị HUD Dashboard** | ARM PS (HDMI DisplayPort) | `KV260HUDVisualizer` (OpenCV GUI + Telemetry Monitor) |

### 2.2. Mô Hình Toán Học Tính Toán Độ Trễ (Latency Model)

Mô phỏng đo đạc độ trễ toàn chu trình $T_{\text{total}}$ dựa trên tổng thời gian của các khâu:

$$T_{\text{total}} = T_{\text{read}} + T_{\text{preprocess}} + T_{\text{dma\_tx}} + T_{\text{pl\_attention}} + T_{\text{dma\_rx}} + T_{\text{postprocess}} + T_{\text{render}}$$

Trong đó:
1. **Độ trễ truyền AXI DMA ($T_{\text{dma}}$)**:
   - Kích thước dữ liệu một khối Attention: $D_{\text{bytes}} = 3 \times N \times d_{\text{model}} = 3 \times 65 \times 256 = 49,920 \text{ bytes} \approx 48.75 \text{ KB}$.
   - Với bus 64-bit @ 200 MHz, băng thông lý thuyết:
     $$\text{Bandwidth} = 8 \text{ bytes} \times 200 \times 10^6 \text{ Hz} = 1.6 \text{ GB/s}$$
   - Thời gian DMA ước tính lý thuyết:
     $$T_{\text{dma\_ideal}} = \frac{49,920}{1.6 \times 10^9} \approx 31.2 \, \mu\text{s}$$
   - Trong thực tế kèm phụ phí khởi tạo DMA (DMA setup overhead $\approx 15 \mu\text{s}$), $T_{\text{dma}} \approx 0.05 \sim 0.1 \text{ ms}$.

2. **Chu kỳ tính toán phần cứng FPGA PL ($T_{\text{pl\_attention}}$)**:
   - Với $N = 65$ tokens, $d_k = 32$, $H = 8$ heads:
     * Phép nhân $Q \times K^T$: Mỗi head cần $N \times N \times d_k = 65 \times 65 \times 32 = 135,200$ phép MAC.
     * 8 heads: $8 \times 135,200 = 1,081,600$ phép tính MAC.
     * Mảng Systolic MAC $16 \times 16$ chạy ở 200 MHz có thông lượng 256 MACs/chu kỳ $\rightarrow$ Cần xấp xỉ $4,225$ chu kỳ $\approx 21.1 \, \mu\text{s}$ mỗi lớp Attention.
     * Qua 10 tầng Transformer ($depth = 10$): Thời gian tính toán trên FPGA PL xấp xỉ $\approx 0.5 \sim 1.5 \text{ ms}$.

---

## 3. Cơ Chế Lượng Tử Hóa Nhận Biết (QAT INT8)

Mô hình `vit_qat_int8.pth` áp dụng kỹ thuật **Quantization-Aware Training (QAT)** theo chuẩn PyTorch Quantization Toolchain:
* **Weight Quantization**: Định lượng đối xứng theo từng kênh (`per_channel_symmetric`, kiểu `qint8`).
* **Activation Quantization**: Định lượng affine theo từng tensor (`per_tensor_affine`, kiểu `quint8`).
* **Ưu thế trên phần cứng Kria KV260**:
  - Tiết kiệm bộ nhớ từ $21.26 \text{ MB}$ (FP32) xuống còn $5.92 \text{ MB}$ (giảm gần **73%** dung lượng).
  - Tối ưu hóa triệt để tài nguyên DSP48E2 trên Zynq UltraScale+ FPGA (1 block DSP có thể tính đồng thời 2 phép nhân INT8 thay vì chỉ 1 phép nhân FP32).

---
*Tài liệu kỹ thuật nội bộ dành cho dự án Capstone KV260 ViT Edge AI.*
