# Thiết kế ViT trên FPGA

Việc có thêm **thẻ nhớ SD Card** và **bộ nhớ RAM ngoài (DDR4)** mở ra khả năng thiết kế hệ thống **phân cấp bộ nhớ (Memory Hierarchy)** hoàn chỉnh. Nhờ đó, FPGA không cần phải cố nạp toàn bộ hàng chục triệu tham số của mô hình Vision Transformer (ViT) vào Block RAM (BRAM) on-chip, mà có thể **stream dữ liệu trọng số theo từng khối (Tiling/Streaming Strategy)**.

Dưới đây là chiến lược tổng thể mang tính **thích ứng (adaptive)** và **mở rộng (scalable)** từ mô hình thử nghiệm nhỏ trên Colab (CIFAR-10) cho đến các mô hình ViT thực tế trên FPGA.

---

##### 1. Chiến lược Quản lý Bộ nhớ Phân cấp (Memory Hierarchy Strategy)

Để xử lý các mô hình ViT vượt quá dung lượng BRAM của FPGA nhúng (như Kria KV260 hay Zynq UltraScale+), hệ thống cần áp dụng luồng truyền dữ liệu 4 tầng:

```text
[ SD Card ] ------------> [ DDR4 Memory ] ------------> [ AXI DMA Engine ] ------------> [ BRAM Ping-Pong Buffers ]
 (Lưu .pth,                (CMA Contiguous               (Burst Streaming                (Đệm đệm Tile nhỏ
  Weights & .hex)           Physical Buffers)             AXI4-Stream 64-bit)             cho Systolic Array)
```

1. **SD Card (Dung lượng lớn - Chậm)**: Lưu trữ hệ điều hành (Ubuntu/Linux), các tập tin trọng số mô hình đã định lượng (.bin/.pth), ảnh đầu vào từ dataset, và trình điều khiển PYNQ Python.
2. **Bộ nhớ RAM ngoài DDR4 (Cấp phát liên tục - CMA)**: Khi khởi tạo ứng dụng, ARM CPU nạp trọng số từ SD Card vào vùng nhớ vật lý liên tục (Contiguous Memory Allocation - CMA) trên DDR4 để chuẩn bị cho DMA.
3. **AXI Direct Memory Access (AXI DMA IP)**: Đảm nhận truyền nổ (Burst Transfer) các ma trận trọng số và kích hoạt giữa DDR4 và FPGA PL qua chuẩn **AXI4-Stream** với băng thông cao.
4. **On-Chip BRAM / LUTRAM (Double Buffering / Ping-Pong Scheme)**:
    * Áp dụng kỹ thuật **Tiling (Phân khối ma trận)**: Cắt ma trận lớn thành các Tile nhỏ (ví dụ: $T_q \times d_k$ hoặc $64 \times 64$) vừa trọn với BRAM.
    * **Cơ chế Ping-Pong**: Trong khi mảng PE/Systolic Array đang tính toán trên đệm "Ping", AXI DMA tranh thủ nạp Tile trọng số tiếp theo từ DDR4 vào đệm "Pong". Kỹ thuật này ẩn hoàn toàn độ trễ đọc DRAM.

---

##### 2. Kiến trúc Phần cứng Thích ứng & Mở rộng (Adaptive & Scalable Architecture)

Thử thách lớn nhất của ViT là sự biến động về kích thước Tensor (ví dụ: chiều ẩn $D$, số lượng token $N$, số head $H$). Để hệ thống chạy được cả mô hình Colab lẫn các mô hình lớn hơn mà **không cần tổng hợp lại Bitstream RTL**:

* **Mảng tính toán GEMM/Systolic Array Dùng chung (Unified GEMM Engine)**:
    * Thay vì thiết kế các mạch cứng riêng biệt cho từng tầng, hãy xây dựng một mảng **Systolic MAC Array cố định** (ví dụ: kích thước $16 \times 16$ hoặc $32 \times 32$).
    * Cả hai lớp Fully Connected (FC) và Convolution (CONV stem) đều được quy về phép nhân ma trận chuẩn (GEMM) để dùng chung một mảng tính toán.
* **Quản lý Execution Granularity theo Tile (Tile Scheduling)**:
    * Đặt kích thước Tile cố định ở mức phần cứng (ví dụ: $T_N = 64, T_M = 64, T_K = 1024$).
    * Với mô hình Colab nhỏ (CIFAR-10: $N=64..196, D=128$), Scheduler điều khiển chạy 1–2 lượt Tile. Với mô hình lớn hơn (ViT-Base hay MobileViT), Scheduler chỉ cần lặp lại nhiều vòng Tile hơn trên cùng một Bitstream phần cứng.
* **HW/SW Co-Design (Phân chia công việc giữa PS và PL)**:
    * **ARM Processing System (PS)**: Thực hiện giải mã ảnh, cắt patch, tính toán LayerNorm, và quản lý luồng điều khiển.
    * **Programmable Logic (PL)**: Tập trung tối đa tài nguyên DSP48E2 cho các phép toán ma trận nặng $O(N^2)$ (như $QK^T$, $Score \times V$, MLP) và khối Softmax/ShiftGELU xấp xỉ.

---

##### 3. Quy trình Kiểm thử & Xây dựng Golden Model 4 Bước

Đánh giá tính đúng đắn của mạch RTL từ mô hình Colab đến bo mạch phần cứng thực tế qua 4 bước:

```text
[ Bước 1: PyTorch Colab ] ---> [ Bước 2: RTL & Testbench ] ---> [ Bước 3: Vivado IPI ] ---> [ Bước 4: Hardware Benchmark ]
 - Train FP32 Model             - Design Parameterized RTL       - Package Custom IP        - Load Weights via SD
 - Fixed-Point INT8 Quant       - Self-checking Testbench        - Connect AXI DMA/MPSoC    - PYNQ Driver Test
 - Export .hex Vectors           via $readmemh in xsim           - Generate Bitstream        - Measure FPS, Latency & Acc
```

###### Bước 1: Mô hình hóa & Trích xuất Test Vectors (Python/Colab)
1. Huấn luyện mô hình ViT nhỏ trên Colab (CIFAR-10).
2. Áp dụng định lượng số nguyên **INT8 (Q4.4 hoặc Symmetric INT8)**.
3. Dùng PyTorch Forward Hooks để bắt và lưu các ma trận trung gian ($Q, K, V$, kết quả Softmax, và Output chuẩn).
4. Xuất các ma trận này ra định dạng Hexadecimal (`q_tensor.hex`, `k_tensor.hex`, `golden_output.hex`).

###### Bước 2: Lập trình SystemVerilog & Kiểm thử Đơn vị (xsim)
1. Viết các mô-đun RTL bằng SystemVerilog (`mac_array.sv`, `bram_ping_pong.sv`, `softmax_lut.sv`, `axis_adapter.sv`).
2. Xây dựng **Self-Checking Testbench** sử dụng hàm `$readmemh` để nạp tự động các tệp `.hex` từ Bước 1.
3. Đối chiếu kết quả đầu ra của RTL với Golden Output từng chu kỳ xung clock để đảm bảo chính xác tuyệt đối.

###### Bước 3: Tích hợp Hệ thống SoC trên Vivado (Block Design)
1. Đóng gói khối Attention Core thành Custom IP.
2. Kết nối với **Zynq UltraScale+ MPSoC IP**, **AXI DMA IP** (chế độ Simple DMA, luồng 64-bit), và **AXI SmartConnect**.
3. Tổng hợp và biên dịch ra tập tin **Bitstream (`.bit`)** và **Hardware Handoff (`.hwh`)**.

###### Bước 4: Thử nghiệm Thực tế trên Bo mạch (Hardware-in-the-Loop trên KV260)
1. Nạp hệ điều hành Ubuntu/PYNQ và lưu tập tin trọng số mô hình vào **SD Card**.
2. Chạy kịch bản Python PYNQ trên ARM PS:
    * Cấp phát bộ nhớ liên tục bằng `pynq.allocate()`.
    * Nạp trọng số từ SD Card vào DDR4.
    * Kích hoạt AXI DMA truyền dữ liệu vào FPGA PL và nhận kết quả.
3. Đánh giá các chỉ số: Khung hình/giây (FPS), độ trễ (Latency), tài nguyên tiêu thụ (LUT, DSP, BRAM), và độ chính xác phân loại.

---

##### 4. Bảng Tóm tắt Cấu hình Thử nghiệm Mở rộng

| Thông số mô hình | Thử nghiệm Colab (PoC) | Mô hình Mở rộng (MobileViT / DeiT-T) | Tác động lên Mạch RTL FPGA |
| ------ | ------ | ------ | ------ |
| **Tập dữ liệu (Dataset)** | CIFAR-10 (28x28 / 32x32) | ImageNet-1K (224x224) | Đẩy qua bộ nhớ SD Card / DDR4. |
| **Số Token ($N$)** | $N = 16 \dots 64$ | $N = 196$ (Grid 14x14) | Tăng số lượt lặp Tile qua BRAM. |
| **Chiều Ẩn ($d_k$)** | $d_k = 16 \text{ hoặc } 32$ | $d_k = 32 \text{ hoặc } 64$ | Tự động thích ứng qua mảng MAC. |
| **Kiểu Dữ liệu** | INT8 (Q4.4) | INT8 (Q4.4 / Symmetric) | Tích lũy INT32 trên DSP48E2. |
| **Cơ chế Nạp Trọng số** | AXI DMA Stream từ DDR4 | AXI DMA Stream từ DDR4 | **Dùng chung 1 Bitstream phần cứng**. |
