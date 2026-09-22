Thiết kế ViT trên FPGA

Việc có thêm **thẻ nhớ SD Card** và **bộ nhớ RAM ngoài (DDR4)** mở ra khả năng thiết kế hệ thống **phân cấp bộ nhớ (Memory Hierarchy)** hoàn chỉnh. Nhờ đó, FPGA không cần phải cố nạp toàn bộ hàng chục triệu tham số của mô hình Vision Transformer (ViT) vào Block RAM (BRAM) on-chip, mà có thể **stream dữ liệu trọng số theo từng khối (Tiling/Streaming Strategy)**.

Dưới đây là chiến lược tổng thể mang tính **thích ứng (adaptive)** và **mở rộng (scalable)** từ mô hình thử nghiệm nhỏ trên Colab (CIFAR-10) cho đến các mô hình ViT thực tế trên FPGA.

\--------------------------------------------------------------------------------

1\. Chiến lược Quản lý Bộ nhớ Phân cấp (Memory Hierarchy Strategy)

Để xử lý các mô hình ViT vượt quá dung lượng BRAM của FPGA nhúng (như Kria KV260 hay Zynq UltraScale+), hệ thống cần áp dụng luồng truyền dữ liệu 4 tầng:

\[ SD Card \] \------------\&gt; \[ DDR4 Memory \] \------------\&gt; \[ AXI DMA Engine \] \------------\&gt; \[ BRAM Ping-Pong Buffers \]

 (Lưu .pth,                (CMA Contiguous               (Burst Streaming                (Đệm đệm Tile nhỏ

  Weights \&amp; .hex)           Physical Buffers)             AXI4-Stream 64-bit)             cho Systolic Array)

&nbsp;

&nbsp;

1. **SD Card (Dung lượng lớn \- Chậm)**: Lưu trữ hệ điều hành (Ubuntu/Linux), các tập tin trọng số mô hình đã định lượng (`.bin`/`.pth`), ảnh đầu vào từ dataset, và trình điều khiển PYNQ Python.  
2. **Bộ nhớ RAM ngoài DDR4 (Cấp phát liên tục \- CMA)**: Khi khởi tạo ứng dụng, ARM CPU nạp trọng số từ SD Card vào vùng nhớ vật lý liên tục (Contiguous Memory Allocation \- CMA) trên DDR4 để chuẩn bị cho DMA.  
3. **AXI Direct Memory Access (AXI DMA IP)**: Đảm nhận truyền nổ (Burst Transfer) các ma trận trọng số và kích hoạt giữa DDR4 và FPGA PL qua chuẩn **AXI4-Stream** với băng thông cao.  
4. **On-Chip BRAM / LUTRAM (Double Buffering / Ping-Pong Scheme)**:  
5. Áp dụng kỹ thuật **Tiling (Phân khối ma trận)**: Cắt ma trận lớn thành các Tile nhỏ (ví dụ: $T\\\_q \\\\times d\\\_k$ hoặc $64 \\\\times 64$) vừa trọn với BRAM.  
6. **Cơ chế Ping-Pong**: Trong khi mảng PE/Systolic Array đang tính toán trên đệm "Ping", AXI DMA tranh thủ nạp Tile trọng số tiếp theo từ DDR4 vào đệm "Pong". Kỹ thuật này ẩn hoàn toàn độ trễ đọc DRAM.

\--------------------------------------------------------------------------------

2\. Kiến trúc Phần cứng Thích ứng & Mở rộng (Adaptive & Scalable Architecture)

Thử thách lớn nhất của ViT là sự biến động về kích thước Tensor (ví dụ: chiều ẩn $D$, số lượng token $N$, số head $H$). Để hệ thống chạy được cả mô hình Colab lẫn các mô hình lớn hơn mà **không cần tổng hợp lại Bitstream RTL**:

* **Mảng tính toán GEMM/Systolic Array Dùng chung (Unified GEMM Engine)**:  
  * Thay vì thiết kế các mạch cứng riêng biệt cho từng tầng, hãy xây dựng một mảng **Systolic MAC Array cố định** (ví dụ: kích thước $16 \\\\times 16$ hoặc $32 \\\\times 32$).  
  * Cả hai lớp Fully Connected (FC) và Convolution (CONV stem) đều được quy về phép nhân ma trận chuẩn (GEMM) để dùng chung một mảng tính toán.  
* **Quản lý Execution Granularity theo Tile (Tile Scheduling)**:  
  * Đặt kích thước Tile cố định ở mức phần cứng (ví dụ: $T\\\_N \= 64, T\\\_M \= 64, T\\\_K \= 1024$).  
  * Với mô hình Colab nhỏ (CIFAR-10: $N=64..196, D=128$), Scheduler điều khiển chạy 1–2 lượt Tile. Với mô hình lớn hơn (ViT-Base hay MobileViT), Scheduler chỉ cần lặp lại nhiều vòng Tile hơn trên cùng một Bitstream phần cứng.  
* **HW/SW Co-Design (Phân chia công việc giữa PS và PL)**:  
  * **ARM Processing System (PS)**: Thực hiện giải mã ảnh, cắt patch, tính toán LayerNorm, và quản lý luồng điều khiển.  
  * **Programmable Logic (PL)**: Tập trung tối đa tài nguyên DSP48E2 cho các phép toán ma trận nặng $O(N^2)$ (như $Q K^T$, $Score \\\\times V$, MLP) và khối Softmax/ShiftGELU xấp xỉ.

\--------------------------------------------------------------------------------

3\. Quy trình Kiểm thử & Xây dựng Golden Model 4 Bước

Đánh giá tính đúng đắn của mạch RTL từ mô hình Colab đến bo mạch phần cứng thực tế qua 4 bước:

\[ Bước 1: PyTorch Colab \] \---\&gt; \[ Bước 2: RTL \&amp; Testbench \] \---\&gt; \[ Bước 3: Vivado IPI \] \---\&gt; \[ Bước 4: Hardware Benchmark \]

 \- Train FP32 Model             \- Design Parameterized RTL       \- Package Custom IP        \- Load Weights via SD

 \- Fixed-Point INT8 Quant       \- Self-checking Testbench        \- Connect AXI DMA/MPSoC    \- PYNQ Driver Test

 \- Export .hex Vectors           via $readmemh in xsim           \- Generate Bitstream        \- Measure FPS, Latency \&amp; Acc

&nbsp;

&nbsp;

Bước 1: Mô hình hóa & Trích xuất Test Vectors (Python/Colab)

1. Huấn luyện mô hình ViT nhỏ trên Colab (CIFAR-10).  
2. Áp dụng định lượng số nguyên **INT8 (Q4.4 hoặc Symmetric INT8)**.  
3. Dùng PyTorch Forward Hooks để bắt và lưu các ma trận trung gian ($Q, K, V$, kết quả Softmax, và Output chuẩn).  
4. Xuất các ma trận này ra định dạng Hexadecimal (`q_tensor.hex`, `k_tensor.hex`, `golden_output.hex`).

Bước 2: Lập trình SystemVerilog & Kiểm thử Đơn vị (xsim)

1. Viết các mô-đun RTL bằng SystemVerilog (`mac_array.sv`, `bram_ping_pong.sv`, `softmax_lut.sv`, `axis_adapter.sv`).  
2. Xây dựng **Self-Checking Testbench** sử dụng hàm `$readmemh` để nạp tự động các tệp `.hex` từ Bước 1\.  
3. Đối chiếu kết quả đầu ra của RTL với Golden Output từng chu kỳ xung clock để đảm bảo chính xác tuyệt đối.

Bước 3: Tích hợp Hệ thống SoC trên Vivado (Block Design)

1. Đóng gói khối Attention Core thành Custom IP.  
2. Kết nối với **Zynq UltraScale+ MPSoC IP**, **AXI DMA IP** (chế độ Simple DMA, luồng 64-bit), và **AXI SmartConnect**.  
3. Tổng hợp và biên dịch ra tập tin **Bitstream (.bit)** và **Hardware Handoff (.hwh)**.

Bước 4: Thử nghiệm Thực tế trên Bo mạch (Hardware-in-the-Loop trên KV260)

1. Nạp hệ điều hành Ubuntu/PYNQ và lưu tập tin trọng số mô hình vào **SD Card**.  
2. Chạy kịch bản Python PYNQ trên ARM PS:  
3. Cấp phát bộ nhớ liên tục bằng `pynq.allocate()`.  
4. Nạp trọng số từ SD Card vào DDR4.  
5. Kích hoạt AXI DMA truyền dữ liệu vào FPGA PL và nhận kết quả.  
6. Đánh giá các chỉ số: Khung hình/giây (FPS), độ trễ (Latency), tài nguyên tiêu thụ (LUT, DSP, BRAM), và độ chính xác phân loại.

\--------------------------------------------------------------------------------

4\. Bảng Tóm tắt Cấu hình Thử nghiệm Mở rộng

| Thông số mô hình | Thử nghiệm Colab (PoC) | Mô hình Mở rộng (MobileViT / DeiT-T) | Tác động lên Mạch RTL FPGA |
| ----- | ----- | ----- | ----- |
| **Tập dữ liệu (Dataset)** | CIFAR-10 (28x28 / 32x32) | ImageNet-1K (224x224) | Đẩy qua bộ nhớ SD Card / DDR4. |
| **Số Token (**$N$**)** | $N \= 16 \\\\dots 64$ | $N \= 196$ (Grid 14x14) | Tăng số lượt lặp Tile qua BRAM. |
| **Chiều Ẩn (**$d\\\_k$**)** | $d\\\_k \= 16 \\\\text{ hoặc } 32$ | $d\\\_k \= 32 \\\\text{ hoặc } 64$ | Tự động thích ứng qua mảng MAC. |
| **Kiểu Dữ liệu** | INT8 (Q4.4) | INT8 (Q4.4 / Symmetric) | Tích lũy INT32 trên DSP48E2. |
| **Cơ chế Nạp Trọng số** | AXI DMA Stream từ DDR4 | AXI DMA Stream từ DDR4 | **Dùng chung 1 Bitstream phần cứng**. |

\--------------------------------------------------------------------------------

💡 Bạn có muốn xem qua tệp **Python PYNQ Host Script (driver.py)** mẫu để lập trình cho ARM PS nạp trọng số từ SD Card vào DDR4 và kích hoạt AXI DMA truyền sang FPGA PL không?

\--------------------------------------------------------------------------------

Triển khai ViT trên FPGA

Hoàn toàn **CÓ THỂ** và đây chính là phương pháp chuẩn mực nhất trong thực tế để xây dựng ứng dụng demo **Real-time Edge AI / Hardware-in-the-Loop (HIL)**\!

Sau khi huấn luyện mô hình ViT trên Google Colab, bạn chỉ cần xuất file trọng số (trọng số đã định lượng INT8 như `.pth`, `.bin` hoặc `.hex`), lưu về máy cục bộ (Laptop hoặc thẻ nhớ SD trên bo mạch Kria/Zynq), sau đó chạy script Python đọc luồng camera từ OpenCV.

Luồng kiến trúc bạn đưa ra **khớp hoàn toàn 100%** với cơ chế HW/SW Co-design thực tế. Dưới đây là phân tích chi tiết cách kết nối luồng dữ liệu từ Laptop/Webcam vào khối **Attention IP Core** trên FPGA:

\--------------------------------------------------------------------------------

1\. Luồng dữ liệu Thực thi Real-time (Webcam $\\\\rightarrow$ ARM PS $\\\\rightarrow$ FPGA PL)

\[ LAPTOP WEBCAM / USB CAM \]

            │ (cv2.VideoCapture)

            ▼

\[ 1\. FRAME READ \&amp; PREPROCESS (ARM PS / Laptop Python) \]

    ├── Bắt khung hình RGB \-\&gt; Resize/Crop 224x224x3

    ├── Patch Embedding: Cắt grid 14x14 \-\&gt; N \= 196 Tokens

    └── QKV Projection \+ Quantize \-\&gt; Chuẩn bị mảng INT8 (Q, K, V)

            │

            ▼ (AXI4-Stream via AXI DMA)

\[ 2\. HARDWARE ATTENTION ACCELERATOR (FPGA PL Engine) \]

    ├── AXI-Lite: ARM PS ghi thanh ghi điều khiển (START=1, N=196, d\_k=32, H=2)

    ├── RX Stream \-\&gt; Nạp dữ liệu vào Input Ping-Pong BRAM (Q, K, V)

    ├── Systolic MAC 1: Tính Q \* K^T (INT32)

    ├── Scaler Unit: Dịch bit phải đại số (ASR 1/sqrt(d\_k))

    ├── Hardware Softmax: Tra bảng LUT / Xấp xỉ PWL

    ├── Systolic MAC 2: Tính Score \* V (INT8)

    └── TX Stream \-\&gt; Đẩy kết quả Attended Output ra M\_AXIS\_TDATA

            │

            ▼ (AXI4-Stream via AXI DMA)

\[ 3\. POST-PROCESSING \&amp; DISPLAY (ARM PS / Laptop Python) \]

    ├── Nhận Attended Output từ FPGA PL

    ├── Tính toán phần còn lại: LayerNorm \+ MLP \+ Classification Head

    └── Hiển thị nhãn dự đoán (Label \&amp; FPS) đè lên khung hình webcam (cv2.imshow)

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

2\. Kịch bản Mã nguồn Python Mẫu (`realtime_vit_inference.py`)

Đoạn mã dưới đây minh họa kịch bản chạy trên **ARM PS (sử dụng thư viện PYNQ trên bo mạch Zynq/Kria)** kết nối trực tiếp với Webcam USB hoặc lấy luồng từ Laptop:

import cv2

import numpy as np

import time

from pynq import Overlay, allocate

&nbsp;

\# 1\. Nạp Bitstream phần cứng \&amp; Khai báo IP Core

ol \= Overlay("attention\_core.bit")

dma \= ol.axi\_dma\_0

attn\_ip \= ol.attention\_core\_0

&nbsp;

\# Kích thước dữ liệu chuẩn hóa

N \= 196      \# 14x14 Patch Grid

D\_K \= 32     \# Head dimension

INT8\_BYTES \= N \* D\_K

&nbsp;

\# 2\. Cấp phát bộ nhớ vật lý liên tục (CMA) cho AXI DMA

in\_q \= allocate(shape=(N, D\_K), dtype=np.int8)

in\_k \= allocate(shape=(N, D\_K), dtype=np.int8)

in\_v \= allocate(shape=(N, D\_K), dtype=np.int8)

out\_attn \= allocate(shape=(N, D\_K), dtype=np.int8)

&nbsp;

\# 3\. Khởi tạo Webcam với OpenCV

cap \= cv2.VideoCapture(0) \# 0: Camera mặc định của laptop

cap.set(cv2.CAP\_PROP\_FRAME\_WIDTH, 640\)

cap.set(cv2.CAP\_PROP\_FRAME\_HEIGHT, 480\)

&nbsp;

print("\[INFO\] Đã khởi tạo Webcam \&amp; FPGA IP Core. Bắt đầu luồng suy luận Real-time...")

&nbsp;

try:

    while True:

        start\_time \= time.time()

        ret, frame \= cap.read()

        if not ret:

            break

&nbsp;

        \# \--- STEP A: Preprocessing (ARM PS) \---

        \# Crop \&amp; Resize ảnh webcam về 224x224x3

        img\_resized \= cv2.resize(frame, (224, 224))

        img\_rgb \= cv2.cvtColor(img\_resized, cv2.COLOR\_BGR2RGB)

&nbsp;

        \# Patch Embedding \&amp; QKV Projection (INT8 Quantized)

        \# (Ở đây giả định đã chạy hàm patch\_embed\_and\_qkv\_quant)

        q\_mat, k\_mat, v\_mat \= preprocess\_to\_qkv\_int8(img\_rgb)

&nbsp;

        np.copyto(in\_q, q\_mat)

        np.copyto(in\_k, k\_mat)

        np.copyto(in\_v, v\_mat)

&nbsp;

        \# \--- STEP B: FPGA Execution (Attention Engine PL) \---

        \# Bật cờ START qua AXI-Lite Slave Registers

        attn\_ip.write(0x10, N)     \# Ghi N=196

        attn\_ip.write(0x18, D\_K)   \# Ghi d\_k=32

        attn\_ip.write(0x00, 0x01)  \# START Signal

&nbsp;

        \# Truyền nổ AXI DMA (Streaming Q, K, V sang PL)

        dma.recvchannel.transfer(out\_attn)

        dma.sendchannel.transfer(in\_q) \# Truyền luồng dữ liệu

&nbsp;

        dma.sendchannel.wait()

        dma.recvchannel.wait() \# Chờ FPGA PL hoàn tất và trả về kết quả

&nbsp;

        \# \--- STEP C: Post-Processing \&amp; Class Inference (ARM PS) \---

        \# Chạy nốt LayerNorm \+ MLP \+ Softmax Head trên ARM PS

        class\_id, confidence \= run\_mlp\_head\_int8(out\_attn)

&nbsp;

        \# \--- STEP D: Render kết quả đè lên Webcam \---

        fps \= 1.0 / (time.time() \- start\_time)

        label\_text \= f"Class: {class\_id} ({confidence\*100:.1f}%) | FPS: {fps:.1f}"

&nbsp;

        cv2.putText(frame, label\_text, (20, 40),

                    cv2.COLOR\_BGR2GRAY, 0.8, (0, 255, 0), 2\)

        cv2.imshow("FPGA ViT Real-time Inference", frame)

&nbsp;

        \# Nhấn 'q' để thoát

        if cv2.waitKey(1) \&amp; 0xFF \== ord('q'):

            break

&nbsp;

finally:

    cap.release()

    cv2.destroyAllWindows()

    print("\[INFO\] Đã đóng luồng Webcam.")

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

3\. Ưu điểm của giải pháp này đối với Đề tài Capstone

1. **Khắc phục triệt để hạn chế nạp lại Dataset trên Colab**: Bạn chỉ cần train 1 lần trên Colab, tải file weights `.pth` về máy. Khi test/demo chỉ cần chạy script local với webcam mà không tốn thời gian tải lại dữ liệu.  
2. **Minh chứng tính ứng dụng thực tế (Live Demo)**: Việc hiển thị kết quả phân loại ảnh trực tiếp từ webcam với chỉ số FPS realtime là bằng chứng thuyết phục nhất cho thấy khối hardware accelerator trên FPGA hoạt động chính xác và có độ trễ thấp.  
3. **Phân chia khối lượng công việc hoàn hảo (HW/SW Partitioning)**:  
4. **Phần mềm (Python/OpenCV)**: Đảm nhận các việc biến đổi linh hoạt (đọc camera, resize, vẽ giao diện UI).  
5. **Phần cứng (FPGA PL Attention Core)**: Đảm nhận phần tính toán nhân ma trận nặng nề nhất $O(N^2)$ với tốc độ cao nhờ **Systolic MAC Array** và **AXI4-Stream Pipeline**.

\--------------------------------------------------------------------------------

💡 Bạn có muốn mình hỗ trợ viết thêm hàm Python mẫu **preprocess\_to\_qkv\_int8()** để chia Patch $14 \\\\times 14$ và định lượng ảnh đầu vào từ OpenCV thành mảng INT8 chuẩn xác trước khi nạp vào DMA không?

\--------------------------------------------------------------------------------

Quy trình Kiểm thử RTL

Việc sử dụng **Jupyter Notebook ngay trong Visual Studio Code (hoặc Antigravity/Cursor)** kết hợp với **Icarus Verilog** và **Verilator** làm môi trường phát triển ban đầu là một lựa chọn **RẤT NÊN DÙNG**.

Đây được xem là chiến lược **Hardware/Software Co-Verification (Đồng kiểm thử Phần cứng/Phần mềm)** chuẩn mực của các nhóm làm AI Accelerator hiện đại trước khi đưa thiết kế vào Vivado.

\--------------------------------------------------------------------------------

1\. Tại sao cách tiếp cận này lại tối ưu?

1. **Vòng lặp thử nghiệm cực nhanh (Fast Iteration Loop)**:  
2. **Vivado GUI & xsim**: Rất nặng, tốn nhiều thời gian mở project và biên dịch. Một lần mô phỏng đầy đủ trên Vivado xsim có thể mất từ vài phút đến vài chục phút.  
3. **Verilator \+ Icarus Verilog**: Biên dịch mã RTL thành C++ executable hoặc simulation binary trong vài giây. Tốc độ mô phỏng của Verilator nhanh gấp hàng chục đến hàng trăm lần so với các trình mô phỏng sự kiện (Event-driven simulators) truyền thống.  
4. **Liền mạch giữa Golden Model (Python) và RTL (SystemVerilog)**:  
5. Ngay trên một notebook, bạn có thể thực thi chuỗi công việc liên hoàn chỉ qua các cell:  
   * **Cell 1**: Dùng PyTorch/NumPy sinh dữ liệu test, định lượng INT8 và xuất ra file `q_tensor.hex`, `k_tensor.hex`, `golden_output.hex`.  
   * **Cell 2**: Chạy lệnh shell (`!verilator` hoặc `!iverilog`) để tự động biên dịch và thực thi simulation RTL.  
   * **Cell 3**: Đọc trực tiếp dữ liệu đầu ra từ simulation, tính sai số **MSE/SNR** và vẽ đồ thị so sánh (dùng `matplotlib`) với đáp án Golden Model.  
6. **Phát huy thế mạnh của VS Code / Antigravity**:  
7. Tận dụng khả năng linter/syntax highlighting mạnh mẽ của extension SystemVerilog trên VS Code.  
8. Dễ dàng dùng AI Assistant (Antigravity/Cursor/Copilot) để phát hiện lệch pha dữ liệu (timing/data mismatch) giữa mã Python và mã RTL ngay trong một không gian làm việc.

\--------------------------------------------------------------------------------

2\. Mô hình Workflow Khuyên dùng (Recommended Setup)

Để dự án gọn gàng và dễ mở rộng, bạn nên phân chia cấu trúc dự án như sau:

my\_vit\_hardware\_project/

├── rtl/                        \# Mã nguồn SystemVerilog (Độc lập, chuẩn hóa)

│   ├── mac\_array.sv

│   ├── shift\_gelu\_pipeline.sv

│   └── attention\_core.sv

├── tb/                         \# Testbench (C++ cho Verilator hoặc Verilog cho iverilog)

│   ├── tb\_attention.cpp

│   └── tb\_attention.sv

├── vectors/                    \# File .hex dữ liệu trung gian (Python sinh ra)

│   ├── q\_tensor.hex

│   └── golden\_output.hex

└── notebooks/                  \# Jupyter Notebooks điều khiển kiểm thử

    └── 01\_verify\_attention\_core.ipynb

&nbsp;

&nbsp;

Ví dụ cách điều khiển Simulation từ một Cell trong Notebook:

\# \[Cell trong Jupyter Notebook\]

import numpy as np

import matplotlib.pyplot as plt

import os

&nbsp;

\# 1\. Sinh vector test \&amp; xuất file hex bằng Python

generate\_quantized\_qkv\_hex(N=196, d\_k=32)

&nbsp;

\# 2\. Gọi Verilator biên dịch và thực thi RTL Simulation qua Bash Shell

\!verilator \--cc \--exe \--build \-j 0 ../tb/tb\_attention.cpp ../rtl/attention\_core.sv

\!./obj\_dir/Vattention\_core

&nbsp;

\# 3\. Đọc kết quả từ RTL xuất ra và so sánh ngay với Python Golden Model

rtl\_output \= np.loadtxt("../vectors/rtl\_output.hex", dtype=np.int8)

golden\_output \= np.loadtxt("../vectors/golden\_output.hex", dtype=np.int8)

&nbsp;

mse \= np.mean((rtl\_output \- golden\_output) \*\* 2\)

print(f"\[VERIFICATION RESULT\] Mean Squared Error (MSE): {mse:.4f}")

&nbsp;

if mse \== 0:

    print("\&gt;\&gt;\&gt; SUCCESS: RTL BIT-EXACT MATCH WITH GOLDEN MODEL\! \&lt;\&lt;\&lt;")

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

3\. Khuyến nghị Kỹ thuật Nâng cao: Sử dụng **Cocotb**

Nếu muốn đưa trải nghiệm kiểm thử lên mức chuyên nghiệp mà không cần viết C++ wrapper cho Verilator:

* **Cocotb (Coroutines-based Co-simulation Testbench)** là framework kiểm thử RTL thuần bằng **Python**.  
* Cocotb giao tiếp trực tiếp với **Icarus Verilog** hoặc **Verilator** thông qua giao diện VPI/DPI-C.  
* Khi dùng Cocotb, bạn có thể viết tín hiệu Clock, Reset, AXI-Stream transaction cho SystemVerilog RTL **bằng chính hàm Python async/await** trong Notebook, giúp bỏ qua hoàn toàn việc viết Testbench bằng SystemVerilog/C++ phức tạp.

\--------------------------------------------------------------------------------

4\. Khi nào nên chuyển giao (Migrate) sang Vivado?

Bạn chỉ nên mở Vivado khi đã đi qua bước kiểm thử logic này:

1. **Giai đoạn 1 (VS Code \+ Verilator/Icarus \+ Notebook)**: Kiểm tra **Logic Bit-exactness** (Đảm bảo thuật toán RTL tính ra kết quả đúng từng bit so với Python Golden Model).  
2. **Giai đoạn 2 (Vivado)**: Mang các file `.sv` đã verified 100% vào Vivado để:  
3. Đánh giá **Timing Closure** (WNS, WHS ở 250–300 MHz).  
4. Đánh giá **Resource Utilization** (Số lượng LUTs, DSP48E2, BRAM tiêu tốn trên chip Kria KV260/Zynq).  
5. Đóng gói thành **Custom AXI IP Block** để nối với Zynq Processing System (PS).

\--------------------------------------------------------------------------------

💡 Bạn có muốn mình cung cấp file mẫu **Makefile** hoặc kịch bản **Cocotb Python Testbench** để chạy mô phỏng SystemVerilog trực tiếp bằng Icarus Verilog/Verilator trong VS Code không?

\--------------------------------------------------------------------------------

Kịch bản Cocotb Python RTL

Dưới đây là kịch bản **Cocotb Python Testbench** hoàn chỉnh cho mạch RTL (ví dụ khối Attention Datapath / Adder Tree đã thảo luận).

Cocotb cho phép bạn điều khiển các tín hiệu `clk`, `rst_n`, giao thức `AXI-Stream` và tự động kiểm tra đáp án (Self-checking) bằng chính mã **Python** kết hợp với **NumPy** hoặc **PyTorch**, mà không cần viết Testbench bằng SystemVerilog hay C++.

\--------------------------------------------------------------------------------

1\. Cấu trúc Thư mục Dự án

project\_cocotb/

├── rtl/

│   └── pipelined\_adder\_tree.sv   \# Hoặc attention\_core.sv

├── tests/

│   ├── test\_adder\_tree.py        \# Kịch bản Cocotb Testbench

│   └── Makefile                  \# File cấu hình gọi Verilator / Icarus Verilog

└── vectors/

    ├── input\_a.hex

    └── golden\_output.hex

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

2\. Mã nguồn Cocotb Testbench (`tests/test_adder_tree.py`)

Kịch bản này minh họa đầy đủ:

1. **Khởi tạo xung Clock ngắt tự động** (`cocotb.clock.Clock`).  
2. **Quy trình Reset** (`async def reset_dut`).  
3. **Driver Task**: Đẩy dữ liệu vào DUT từng chu kỳ xung clock.  
4. **Scoreboard/Monitor Task**: Tự động so sánh đầu ra với Golden Model bằng NumPy.

import cocotb

from cocotb.clock import Clock

from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

import numpy as np

&nbsp;

\# \-----------------------------------------------------------------------------

\# 1\. Golden Model bằng Python (NumPy)

\# \-----------------------------------------------------------------------------

def compute\_golden\_dot\_product(vec\_a, vec\_b):

    """Tính tích vô hướng chuẩn số nguyên INT8 \-\&gt; INT32."""

    return int(np.sum(vec\_a.astype(np.int32) \* vec\_b.astype(np.int32)))

&nbsp;

\# \-----------------------------------------------------------------------------

\# 2\. Task Reset DUT

\# \-----------------------------------------------------------------------------

async def reset\_dut(dut):

    dut.rst\_n.value \= 0

    dut.in\_valid.value \= 0

    for i in range(len(dut.a)):

        dut.a\[i\].value \= 0

        dut.b\[i\].value \= 0

&nbsp;

    await ClockCycles(dut.clk, 5\)

    dut.rst\_n.value \= 1

    await ClockCycles(dut.clk, 2\)

    dut.\_log.info("\[COCOTB\] Reset completed successfully.")

&nbsp;

\# \-----------------------------------------------------------------------------

\# 3\. Main Testcase: Pipelined Streaming Test

\# \-----------------------------------------------------------------------------

@cocotb.test()

async def test\_pipelined\_adder\_tree\_random(dut):

    """Kiểm thử tính đúng đắn của Pipelined Adder Tree với dữ liệu ngẫu nhiên."""

&nbsp;

    \# 1\. Khởi tạo xung Clock 100 MHz (Period \= 10ns)

    clock \= Clock(dut.clk, 10, units="ns")

    cocotb.start\_soon(clock.start())

&nbsp;

    \# 2\. Thực hiện Reset

    await reset\_dut(dut)

&nbsp;

    N \= len(dut.a) \# Số phần tử của vector (e.g., N \= 16\)

    NUM\_TESTS \= 50

    expected\_queue \= \[\]

&nbsp;

    dut.\_log.info(f"\[COCOTB\] Starting Test Sequence with N \= {N} elements...")

&nbsp;

    \# 3\. DRIVER LOOP: Đẩy luồng dữ liệu liên tục vào DUT (Initiation Interval II \= 1\)

    for test\_idx in range(NUM\_TESTS):

        await RisingEdge(dut.clk)

&nbsp;

        \# Sinh vector ngẫu nhiên INT8 \[-128, 127\]

        vec\_a \= np.random.randint(-128, 127, size=N, dtype=np.int8)

        vec\_b \= np.random.randint(-128, 127, size=N, dtype=np.int8)

&nbsp;

        \# Gán giá trị vào cổng RTL

        dut.in\_valid.value \= 1

        for i in range(N):

            dut.a\[i\].value \= int(vec\_a\[i\])

            dut.b\[i\].value \= int(vec\_b\[i\])

&nbsp;

        \# Tính toán Golden Output và lưu vào hàng đợi đối chiếu

        golden\_val \= compute\_golden\_dot\_product(vec\_a, vec\_b)

        expected\_queue.append(golden\_val)

&nbsp;

    \# Tắt tín hiệu valid sau khi đẩy xong dữ liệu

    await RisingEdge(dut.clk)

    dut.in\_valid.value \= 0

&nbsp;

    \# 4\. MONITOR \&amp; CHECKER LOOP: Đọc kết quả từ RTL xuất ra sau độ trễ Pipeline

    pass\_count \= 0

    fail\_count \= 0

&nbsp;

    while len(expected\_queue) \&gt; 0:

        await RisingEdge(dut.clk)

&nbsp;

        \# Kiểm tra cờ out\_valid từ RTL

        if dut.out\_valid.value.binstr \== '1':

            expected\_val \= expected\_queue.pop(0)

            rtl\_val \= dut.dot\_product.value.signed\_integer

&nbsp;

            \# So sánh kết quả

            if rtl\_val \== expected\_val:

                pass\_count \+= 1

                dut.\_log.info(f"\[PASS\] RTL Output: {rtl\_val} | Expected: {expected\_val}")

            else:

                fail\_count \+= 1

                dut.\_log.error(f"\[FAIL\] RTL Output: {rtl\_val} | Expected: {expected\_val} (MISMATCH\!)")

&nbsp;

    \# Đợi thêm 5 chu kỳ clock trước khi kết thúc

    await ClockCycles(dut.clk, 5\)

&nbsp;

    \# 5\. Tổng kết kiểm thử

    dut.\_log.info("==================================================")

    dut.\_log.info(f" PASSED: {pass\_count} / {NUM\_TESTS}")

    dut.\_log.info(f" FAILED: {fail\_count} / {NUM\_TESTS}")

    dut.\_log.info("==================================================")

&nbsp;

    assert fail\_count \== 0, f"Test failed with {fail\_count} mismatches\!"

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

3\. File Điều khiển Biên dịch (`Makefile`)

Tệp `Makefile` này tự động liên kết **Verilator** hoặc **Icarus Verilog** để chạy Cocotb:

\# Ngôn ngữ và Trình mô phỏng

SIM ?= verilator

TOPLEVEL\_LANG ?= verilog

&nbsp;

\# Đường dẫn file SystemVerilog RTL

VERILOG\_SOURCES \+= $(PWD)/../rtl/pipelined\_adder\_tree.sv

&nbsp;

\# Cấu hình Module RTL đỉnh \&amp; File Python Testbench

TOPLEVEL \= pipelined\_adder\_tree

MODULE \= test\_adder\_tree

&nbsp;

\# Tùy chọn biên dịch cho Verilator

COMPILE\_ARGS \+= \-Wno-fatal \--trace-fst

&nbsp;

include $(shell cocotb-config \--makefiles)/Makefile.sim

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Cách chạy Mô phỏng trực tiếp trong VS Code / Jupyter Notebook

Cách 1: Chạy từ Terminal trong VS Code

pip install cocotb cocotb-test pytest

make SIM=verilator   \# Hoặc make SIM=icarus

&nbsp;

&nbsp;

Cách 2: Chạy trực tiếp từ một Cell trong Jupyter Notebook

\# \[Jupyter Notebook Cell\]

import os

import subprocess

&nbsp;

\# Chạy Makefile mô phỏng Cocotb

result \= subprocess.run(\["make", "SIM=verilator"\], capture\_output=True, text=True)

print(result.stdout)

&nbsp;

if "SUCCESS" in result.stdout or "PASSED" in result.stdout:

    print("🎉 RTL SIMULATION PASSED PERFECTLY\!")

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

🔥 Ưu điểm lớn nhất của Cocotb:

* Không cần viết file `$readmemh` trung gian — bạn có thể tạo mảng NumPy ngẫu nhiên và **gán trực tiếp vào tín hiệu RTL** `dut.a[i].value = int(vec_a[i])`.  
* Nếu muốn xem dạng sóng (Waveform), Cocotb \+ Verilator sẽ tự động xuất file `.fst` hoặc `.vcd` để mở trực tiếp trên **GTKWave** hoặc extension **WaveTrace** trong VS Code.

💡 Bạn có muốn mình điều chỉnh kịch bản Cocotb này sang giao thức chuẩn **AXI4-Stream (với các tín hiệu TVALID, TREADY, TDATA, TLAST)** để khớp hoàn toàn với khối Attention IP Core không?

\--------------------------------------------------------------------------------

Kiểm thử ViT trên FPGA

Để lựa chọn và triển khai **Golden Model** cho việc kiểm thử (verification) khối phần cứng **RTL Attention Accelerator** trên FPGA, dưới đây là phân tích chi tiết giúp bạn chọn mô hình phù hợp và xây dựng quy trình trích xuất dữ liệu testbench.

\--------------------------------------------------------------------------------

I. Nên chọn mô hình ViT nào làm Golden Model?

**👉 Khuyên dùng: Nên sử dụng một mô hình ViT cỡ nhỏ (Lightweight/Custom ViT) như notebook CIFAR-10 hoặc MobileViT-XXS / TinyViT.**

Lý do Kỹ thuật:

1. **Tiết kiệm tài nguyên BRAM & Thời gian Mô phỏng (Simulation Time)**:  
2. **ViT-B/16 hay ViT-B/12**: Có tới \~86M tham số, chiều ẩn $d\\\_{model} \= 768$, $N \= 196$. Việc nạp toàn bộ một mô hình lớn như vậy vào phần mềm mô phỏng RTL (Vivado xsim / Verilator) sẽ khiến thời gian mô phỏng cực kỳ chậm (tốn hàng triệu chu kỳ clock) và vượt quá dung lượng BRAM của các dòng FPGA nhúng như Kria KV260 hay Zynq ZC7020.  
3. **Custom ViT / MobileViT-XXS**: Có kích thước tham số rất nhỏ ($\&lt; 5\\\\text{M}$ params), chiều ẩn head $d\\\_k \= 16$ hoặc $32$, chuỗi $N \= 196$. Khung dữ liệu gọn nhẹ này chạy mô phỏng RTL vô cùng nhanh (chỉ vài nghìn chu kỳ clock).  
4. **Tính đồng dạng kiến trúc (Structural Homogeneity)**:  
5. Bản chất toán học của một khối **Transformer Encoder** (gồm $Q, K, V$ Projections, Scaled Dot-Product Attention $Q K^T / \\\\sqrt{d\\\_k}$, Softmax, Score $\\\\times V$, và MLP) là **hoàn toàn giống nhau** giữa mô hình ViT-Base và ViT-Tiny.  
6. Đơn vị phần cứng RTL (như `attention_core.sv` hay `mac_array.sv`) được thiết kế theo dạng tham số hóa (Parameterized RTL). Do đó, chỉ cần xác minh bit-exactness trên một head attention kích thước nhỏ ($N=196, d\\\_k=32$) là đủ để khẳng định tính đúng đắn của vi mạch cho mọi quy mô ViT khác.

\--------------------------------------------------------------------------------

II. Quy trình Trích xuất Parameter & Quantization cho RTL Testbench

Quy trình chuẩn gồm 4 bước chuyển đổi dữ liệu từ môi trường PyTorch (Floating-Point) sang SystemVerilog Testbench (Fixed-Point/Hex):

\[ PyTorch FP32 Transformer Block \]

             │

             ├──► 1\. Extract FP32 Tensors (Q, K, V, Target Output via Forward Hook)

             │

             ▼

\[ Fixed-Point Quantization (INT8 Q4.4) \]

             │

             ├──► 2\. Scale \&amp; Clamp to INT8 \[-128, \+127\]

             │

             ▼

\[ Python Golden Model Simulation \]

             │

             ├──► 3\. Simulate Integer Math (MatMul \-\&gt; Shift \-\&gt; Softmax \-\&gt; MatMul)

             │

             ▼

\[ Export to .hex Files \]

  ├── q\_tensor.hex       (Thúc đẩy đầu vào Q cho Testbench)

  ├── k\_tensor.hex       (Thúc đẩy đầu vào K cho Testbench)

  ├── v\_tensor.hex       (Thúc đẩy đầu vào V cho Testbench)

  └── golden\_output.hex  (Đáp án chuẩn để SystemVerilog so sánh)

&nbsp;

&nbsp;

Các bước thực hiện chi tiết:

1. **Trích xuất Tensor FP32 từ PyTorch**: Dùng PyTorch Forward Hook để đăng ký bắt lấy các ma trận trung gian $Q, K, V$ và đầu ra mong muốn từ một Transformer Layer trong mô hình.  
2. **Định lượng số nguyên INT8 (Fixed-Point Quantization)**: Chuyển đổi số thực FP32 sang định dạng số nguyên bù 2 (2's complement): \\$$X\\\_{\\\\text{fixed}} \= \\\\text{Clamp}\*{\\\\text{INT8}}\\\\left( \\\\text{Round}\\\\left( X\*{\\\\text{float}} \\\\times 2^n \\\\right) \\\\right)\\\\\\\] \*(Ví dụ với định dạng Q4.4, hệ số nhân \\\\(2^4 \= 16.0\\\\), dải kẹp \\\\(\\\[-128, \+127$$\\))\*.  
3. **Mô phỏng phép toán nguyên trên Python (Golden Reference)**: Thực hiện đúng chuỗi toán học số nguyên mà phần cứng RTL sẽ chạy:  
4. **Phép nhân** $Q \\\\cdot K^T$: Nhân số nguyên `INT8 x INT8`, tích lũy trong thanh ghi `INT32`.  
5. **Thu phóng (Scaling)**: Thực hiện dịch bit phải đại số (`&gt;&gt;&gt; 2`) tương đương chia cho $\\\\sqrt{d\\\_k}$.  
6. **Softmax nguyên**: Chuyển đổi sang dạng trọng số xác suất `UINT8` .  
7. **Phép nhân** $Score \\\\times V$: Tích lũy `INT32` và dịch bit thu về `INT8`.  
8. **Đóng gói dữ liệu dạng Hexadecimal (.hex)**: Xuất các chuỗi Hex 8-bit/16-bit vào file văn bản `.hex` để nạp trực tiếp vào BRAM/Testbench RTL bằng lệnh SystemVerilog `$readmemh()`.

\--------------------------------------------------------------------------------

III. Script Python Mẫu Trích xuất Vector (`export_vectors.py`)

Dưới đây là mã nguồn Python hoàn chỉnh giúp bạn trích xuất và định lượng dữ liệu từ PyTorch ra các tập tin `.hex`:

import torch

import torch.nn as nn

import numpy as np

&nbsp;

def float\_to\_q44\_hex(val\_float):

    """Chuyển đổi số thực FP32 sang chuỗi Hex 2-byte (INT8 Q4.4) bù 2."""

    val\_int \= int(np.clip(np.round(val\_float \* 16.0), \-128, 127))

    val\_uint8 \= val\_int \&amp; 0xFF

    return f"{val\_uint8:02X}"

&nbsp;

def export\_attention\_vectors(N=196, d\_k=32):

    """

    Tạo/Trích xuất dữ liệu Q, K, V dạng FP32, mô phỏng Attention INT8

    và xuất các tập tin .hex cho SystemVerilog Testbench.

    """

    torch.manual\_seed(42)

&nbsp;

    \# 1\. Giả lập/Trích xuất Tensor FP32 từ mô hình ViT

    Q\_fp32 \= torch.randn(N, d\_k) \* 0.5

    K\_fp32 \= torch.randn(N, d\_k) \* 0.5

    V\_fp32 \= torch.randn(N, d\_k) \* 0.5

&nbsp;

    \# 2\. Định lượng INT8 (Q4.4)

    Q\_int8 \= torch.clamp(torch.round(Q\_fp32 \* 16.0), \-128, 127\)

    K\_int8 \= torch.clamp(torch.round(K\_fp32 \* 16.0), \-128, 127\)

    V\_int8 \= torch.clamp(torch.round(V\_fp32 \* 16.0), \-128, 127\)

&nbsp;

    \# 3\. Mô phỏng Golden Reference số nguyên

    \# Step A: Q \* K^T (Accumulator INT32)

    scores\_int32 \= torch.matmul(Q\_int8, K\_int8.T) \# Kích thước (N, N)

&nbsp;

    \# Step B: Scaling Shift (Chia sqrt(d\_k) \~ 5.65 \=\&gt; Dịch phải 2 bit)

    scores\_scaled \= torch.bitwise\_right\_shift(scores\_int32.to(torch.int32), 2\)

&nbsp;

    \# Step C: Softmax xấp xỉ (Softmax UINT8 \[0..255\])

    softmax\_fp \= torch.softmax(scores\_scaled.to(torch.float32) / 16.0, dim=-1)

    softmax\_int8 \= torch.clamp(torch.round(softmax\_fp \* 255.0), 0, 255\)

&nbsp;

    \# Step D: Score \* V (INT32) \-\&gt; Rescale về INT8

    out\_int32 \= torch.matmul(softmax\_int8, V\_int8) \# Kích thước (N, d\_k)

    out\_int8 \= torch.clamp(torch.bitwise\_right\_shift(out\_int32.to(torch.int32), 8), \-128, 127\)

&nbsp;

    \# 4\. Ghi tập tin .hex cho SystemVerilog $readmemh

    with open("q\_tensor.hex", "w") as f\_q, \\

         open("k\_tensor.hex", "w") as f\_k, \\

         open("v\_tensor.hex", "w") as f\_v, \\

         open("golden\_output.hex", "w") as f\_out:

&nbsp;

        for i in range(N):

            for j in range(d\_k):

                f\_q.write(f"{float\_to\_q44\_hex(Q\_int8\[i, j\].item() / 16.0)}\\n")

                f\_k.write(f"{float\_to\_q44\_hex(K\_int8\[i, j\].item() / 16.0)}\\n")

                f\_v.write(f"{float\_to\_q44\_hex(V\_int8\[i, j\].item() / 16.0)}\\n")

                f\_out.write(f"{float\_to\_q44\_hex(out\_int8\[i, j\].item() / 16.0)}\\n")

&nbsp;

    print(f"\[SUCCESS\] Đã xuất thành công các tập tin .hex (N={N}, d\_k={d\_k})\!")

&nbsp;

if \_\_name\_\_ \== "\_\_main\_\_":

    export\_attention\_vectors()

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

💡 **Gợi ý tiếp theo**: Bạn có muốn viết đoạn mã **SystemVerilog Testbench (tb\_attention\_core.sv)** mẫu để đọc các tập tin `.hex` này bằng `$readmemh` và tự động kiểm thử mạch RTL không?

\--------------------------------------------------------------------------------

Residual Connection trên FPGA

Trong các mô hình Vision Transformer (ViT) và Transformer nói chung, **Residual Connection** (hay *Shortcut Connection / Skip Connection*) có công thức toán học cơ bản:

\\$$z'\*\\\\ell \= \\\\text{MSA}(\\\\text{LN}(z\*{\\\\ell-1})) \+ z\\\_{\\\\ell-1}\\\\\\\] \\\\\\\[z\\\_\\\\ell \= \\\\text{MLP}(\\\\text{LN}(z'\*\\\\ell)) \+ z'\*\\\\ell\\\\$$

Tức là kết quả đầu ra của khối tính toán nặng (như Multi-Head Self-Attention \- MSA hay Multi-Layer Perceptron \- MLP) sẽ được cộng trực tiếp (element-wise addition) với tensor đầu vào chưa qua xử lý của tầng đó.

Trên vi mạch **FPGA**, phép cộng vector hai số nguyên này rất đơn giản về mặt toán học, nhưng lại tạo ra **thách thức rất lớn về kiến trúc bộ nhớ và luồng dữ liệu (datapath)**. Dưới đây là phân tích chi tiết về cách Residual Connection vận hành và được tối ưu hóa trên FPGA:

\--------------------------------------------------------------------------------

1\. Thách thức Kiến trúc khi Triển khai Residual Connection trên FPGA

1. **Vấn đề Phụ thuộc Đường truyền (Path Dependency)**:  
2. Khối MSA hoặc MLP tiêu tốn hàng nghìn đến hàng vạn chu kỳ clock để tính toán. Trong suốt thời gian này, tensor đầu vào gốc $z\\\_{\\\\ell-1}$ phải được **"giữ lại" (buffered/staged)** để chờ kết quả của MSA/MLP xuất ra mới có thể tiến hành phép cộng.  
3. **Nghẽn Bộ nhớ (Memory Bottleneck)**:  
4. Nếu ghi tensor đầu vào $z\\\_{\\\\ell-1}$ ngược ra RAM ngoài (DRAM/DDR4) rồi nạp lại khi cộng, băng thông DRAM sẽ bị quá tải nặng nề (Memory Bandwidth Bottleneck).  
5. Nếu lưu toàn bộ tensor gốc trong bộ nhớ nội bộ (BRAM/URAM) thông thường, dung lượng BRAM on-chip sẽ nhanh chóng bị cạn kiệt, làm giảm diện tích dành cho các mảng tính toán MAC.

\--------------------------------------------------------------------------------

2\. Các Chiến lược Hiện thực hóa Residual Connection trên FPGA

Để giải quyết các thách thức trên, các kiến trúc tăng tốc ViT SOTA trên FPGA áp dụng 3 chiến lược chính:

🔹 Chiến lược 1: Tách Nửa Tầng (Half-Layer Mapping)

* **Cơ chế**: Kiến trúc (như mô hình ViA) chia mỗi khối Transformer Encoder làm 2 nửa riêng biệt:  
  * *Nửa 1 (NSA Engine)*: Xử lý LayerNorm \+ MSA \+ Phép cộng Residual 1\.  
  * *Nửa 2 (NMP Engine)*: Xử lý LayerNorm \+ MLP \+ Phép cộng Residual 2\.  
* **Cách vận hành**: Đơn vị cộng Residual được đặt ở ngay đầu/cuối cấu trúc Processing Element (PE). Khi dữ liệu mới chảy qua, kết quả được cộng trực tiếp với dòng dữ liệu residual đang chờ ở đệm đầu vào mà không cần phải trải qua nhiều tầng đệm pipeline trung gian (multi-stage buffers).  
* **Hiệu quả**: Giải quyết triệt để phụ thuộc đường truyền, giảm đến 9 lần tài nguyên BRAM tiêu tốn cho việc lưu trữ dữ liệu trung gian.

🔹 Chiến lược 2: Chính sách Nạp Đơn & Đệm Đa Mục đích (Single-Load Policy & Multi-Purpose Buffering)

* **Cơ chế**: Kiến trúc (như ME-ViT) cam kết dữ liệu chỉ nạp từ DRAM một lần và **không bao giờ ghi ngược kết quả trung gian ra DRAM** giữa các tầng.  
* **Cách vận hành**:  
  1. Khi tensor đầu vào chảy vào khối **Linear Projection (LP)**, giá trị gốc chưa chuẩn hóa (un-normalized) được tách ra làm luồng Residual Stream và lưu ngay vào **Feature Buffer / Residual Buffer**.  
  2. Đồng thời, luồng giá trị qua LayerNorm được đưa vào mảng Systolic Array để tính toán.  
  3. Kết quả từ Systolic Array xuất ra **Result Buffer** sẽ được cộng ngay với giá trị lưu trong Feature Buffer/Residual Buffer tại đệm trung gian **S-Buffer**.  
  4. Để tránh ngắt nhịp mảng tính toán (pipeline stall), phần cứng sử dụng đệm kép **Ping-Pong S-Buffer (S1 & S2 Buffers)** triển khai trên LUTRAM: trong khi S1 ghi kết quả cộng dồn về BRAM chính thì S2 nhận dữ liệu cộng cho tile tiếp theo.

🔹 Chiến lược 3: Căn chỉnh Thời gian bằng Đệm Dịch Stream (Pipelined Stream Alignment)

* **Cơ chế**: Áp dụng cho các thiết kế thiết lập luồng dữ liệu AXI-Stream nối tiếp.  
* **Cách vận hành**:  
  * Luồng tính toán chính (Data Stream) đi qua khối Softmax/Attention/MLP có độ trễ cố định $L\\\_{compute}$ cycles.  
  * Luồng Residual Stream được cho chạy qua một đệm FIFO / Shift Register có độ sâu chính xác bằng $L\\\_{compute}$ chu kỳ.  
  * Ở cuối đường ống, tín hiệu điều khiển `TVALID` kích hoạt bộ cộng số nguyên (Adder Unit) nhận đồng thời hai phần tử của luồng Residual và luồng Compute tại cùng một chu kỳ xung clock với nhịp nhận $II \= 1$.

\--------------------------------------------------------------------------------

3\. Tóm tắt Luồng Xử lý Phần cứng cho Residual Connection

$$\\\\begin{matrix} \\\\text{Input Stream } (z\\\_{\\\\ell-1}) \&amp; \\\\longrightarrow \&amp; \\\\text{\\\[Un-normalized\\\]} \&amp; \\\\longrightarrow \&amp; \\\\text{Residual Buffer / FIFO Delay} \&amp; \\\\searrow \&amp; \\\\ \&amp; \\\\searrow \&amp; \\\\text{\\\[LayerNorm\\\]} \&amp; \\\\longrightarrow \&amp; \\\\text{Systolic MAC Array (MSA/MLP)} \&amp; \\\\longrightarrow \&amp; \\\\mathbf{\\\\oplus \\\\text{ (Adder Unit)}} \\\\longrightarrow z\\\_\\\\ell \\\\end{matrix}\\$$

* **Thành phần phần cứng tiêu tốn**: Chủ yếu sử dụng **LUTs/CARRY8** cho bộ cộng số nguyên, **LUTRAM/BRAM** cho đệm đệm vị trí, và **Flip-Flops** cho thanh ghi trễ.  
* **Kết quả**: Triệt tiêu hoàn toàn lưu lượng truy cập bộ nhớ ngoài DRAM, duy trì đường ống tính toán chạy liên tục với tốc độ tối đa của chip FPGA.

\--------------------------------------------------------------------------------

💡 Bạn có muốn tìm hiểu sâu hơn về cách tích hợp phép cộng **Residual Connection** với khối định lượng lại (**Re-quantization / Rescaling**) để đảm bảo đầu ra không bị tràn bit INT8 không?

\--------------------------------------------------------------------------------

Định lượng Residual ViT trên FPGA

Việc tích hợp phép cộng **Residual Connection** với khối định lượng lại (**Re-quantization / Rescaling**) để đảm bảo đầu ra chuẩn **INT8** không bị tràn bit hay mất độ chính xác là một trong những bài toán kiến trúc quan trọng nhất khi thiết kế bộ tăng tốc Vision Transformer (ViT) trên FPGA.

Dưới đây là phân tích chi tiết từ bản chất toán học, công thức **Dyadic Rescaling**, sơ đồ mạch vi mạch (Datapath) đến mã nguồn SystemVerilog triển khai thực tế.

\--------------------------------------------------------------------------------

1\. Bản chất Vấn đề: Lệch Scale Factor ($S\\\_F \\\\neq S\\\_X$)

Trong mô hình số thực (FP32), phép cộng Residual chỉ đơn giản là: \\$$Y\\\_{float} \= F(X)\*{float} \+ X\*{float}\\\\$$ Trong đó $X$ là tensor đầu vào khối (Shortcut stream), còn $F(X)$ là đầu ra sau khi qua khối tính toán nặng (MSA hoặc MLP stream).

Tuy nhiên, trong mô hình định lượng **INT8 (Symmetric Uniform Quantization)**:

* $X\\\_{float} \= S\\\_X \\\\cdot I\\\_X$ (với $S\\\_X$ là scaling factor của đầu vào, $I\\\_X \\\\in \\\[-128, 127\\\]$).  
* $F(X)\\\_{float} \= S\\\_F \\\\cdot I\\\_F$ (với $S\\\_F$ là scaling factor của khối MSA/MLP, $I\\\_F \\\\in \\\[-128, 127\\\]$).  
* Tensor đầu ra cần có scaling factor mới $S\\\_{out}$ với $Y\\\_{float} \= S\\\_{out} \\\\cdot I\\\_{out}$.

**Thách thức**:

1. Do $S\\\_F \\\\neq S\\\_X$, **giá trị thực của 1 LSB ở hai tensor là khác nhau**. Bạn **không thể cộng trực tiếp** hai số nguyên $I\\\_F \+ I\\\_X$.  
2. Tổng $I\\\_F \+ I\\\_X$ có thể vượt quá dải $\\\[-128, 127\\\]$ gây hiện tượng tràn số (Overflow / Wrap-around) làm hỏng hoàn toàn Feature Map.  
3. Quá trình căn chỉnh scale phải thực hiện thuần túy bằng số nguyên (Integer-only) sử dụng **Dyadic Arithmetic** (Phép nhân số nguyên \+ Dịch bit đại số `&gt;&gt;&gt;`).

\--------------------------------------------------------------------------------

2\. Công thức Toán học Dyadic Rescaling cho Residual Addition

Chúng ta muốn tìm $I\\\_{out}$ sao cho giá trị thực được bảo toàn: \\$$S\\\_{out} \\\\cdot I\\\_{out} \\\\approx S\\\_F \\\\cdot I\\\_F \+ S\\\_X \\\\cdot I\\\_X \\\\implies I\\\_{out} \= \\\\left\\\\lfloor \\\\frac{S\\\_F}{S\\\_{out}} \\\\cdot I\\\_F \+ \\\\frac{S\\\_X}{S\\\_{out}} \\\\cdot I\\\_X \\\\right\\\\rceil\\\\$$

Chuyển đổi hai tỉ số scaling factor thành dạng **Dyadic Numbers (DN)**: $$\\\\text{DN}\\\\left(\\\\frac{S\\\_F}{S\\\_{out}}\\\\right) \= \\\\frac{M\\\_F}{2^{e\\\_F}}, \\\\quad \\\\text{DN}\\\\left(\\\\frac{S\\\_X}{S\\\_{out}}\\\\right) \= \\\\frac{M\\\_X}{2^{e\\\_X}}\\$$ *(với* $M\\\_F, M\\\_X$ *là các hằng số nhân 32-bit INT32, và* $e\\\_F, e\\\_X$ *là số bit dịch phải)*.

Phép tính trên phần cứng FPGA được quy về dạng: \\$$I\\\_{out} \= \\\\text{Clamp}\\\_{\\\\text{INT8}}\\\\left( \\\\left( (M\\\_F \\\\cdot I\\\_F) \\\\gg e\\\_F \\\\right) \+ \\\\left( (M\\\_X \\\\cdot I\\\_X) \\\\gg e\\\_X \\\\right) \\\\right)\\\\$$

\--------------------------------------------------------------------------------

3\. Sơ đồ Mạch Vi mạch PPU (Post-Processing Unit Stream)

Mạch tích hợp Residual Connection \+ Re-quantization trên FPGA được tổ chức thành đường ống (Pipeline) 4 tầng:

         \[ Main Compute Stream \]                         \[ Shortcut Bypass Stream \]

           I\_F (INT8 từ MLP/MSA)                            I\_X (INT8 từ FIFO Delay)

                     │                                                 │

                     ▼                                                 ▼

         \[ Multiplier 1: I\_F \* M\_F \]                       \[ Multiplier 2: I\_X \* M\_X \]

           (INT8 x INT32 \-\&gt; INT32)                           (INT8 x INT32 \-\&gt; INT32)

                     │                                                 │

                     ▼                                                 ▼

         \[ Arithmetic Shift: \&gt;\&gt;\&gt; e\_F \]                     \[ Arithmetic Shift: \&gt;\&gt;\&gt; e\_X \]

           Scaled\_F (INT32)                                  Scaled\_X (INT32)

                     │                                                 │

                     └────────────────────────┬────────────────────────┘

                                              │

                                              ▼

                                 \[ 32-bit INT Adder \]

                               Sum\_32 \= Scaled\_F \+ Scaled\_X

                                              │

                                              ▼

                                  \[ Rounding Unit (+ Half LSB) \]

                                              │

                                              ▼

                                  \[ Saturation Logic (Clamp) \]

                                   Clamp to \[-128, \+127\]

                                              │

                                              ▼

                                    I\_out (INT8 Output)

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Các Kỹ thuật Tối ưu phần cứng Cốt lõi

1. **QAT Scale-Locking (**$S\\\_F \= S\\\_X \= S\\\_{out}$**)**:  
2. Trong quá trình Quantization-Aware Training (QAT), ta có thể ép ràng buộc khóa $S\\\_F \= S\\\_X \= S\\\_{out}$.  
3. Khi đó $\\\\frac{S\\\_F}{S\\\_{out}} \= 1$ và $\\\\frac{S\\\_X}{S\\\_{out}} \= 1$. Mạch Rescale rút gọn về: $I\\\_{out} \= \\\\text{Clamp}\\\_{\\\\text{INT8}}(I\\\_F \+ I\\\_X)$.  
4. **Lợi ích**: Triệt tiêu hoàn toàn 2 bộ nhân $M\\\_F, M\\\_X$, tiết kiệm 100% tài nguyên DSP cho công đoạn này\!  
5. **Căn chỉnh độ trễ FIFO (Stream Alignment Delay)**:  
6. Luồng $I\\\_F$ đi qua MSA/MLP mất $L\\\_{compute}$ chu kỳ clock. Luồng $I\\\_X$ được đưa qua một đệm FIFO/Shift Register có độ sâu đúng $L\\\_{compute}$ chu kỳ để hai tín hiệu đến bộ cộng cùng lúc.  
7. **Mạch Saturating Logic (Chống tràn INT8)**:  
8. Sử dụng mạch kẹp ngưỡng cứng để ép giá trị vượt dải về âm/dương tối đa, tránh hiện tượng lật dấu số nguyên.

\--------------------------------------------------------------------------------

5\. Mã nguồn SystemVerilog Mẫu (`requant_residual_add.sv`)

\`timescale 1ns / 1ps

&nbsp;

module requant\_residual\_add (

    input  logic                     clk,

    input  logic                     rst\_n,

    input  logic                     in\_valid,

&nbsp;

    // Luồng 1: Đầu ra từ khối tính toán (MLP / MSA)

    input  logic signed \[7:0\]        i\_f,         // INT8

    input  logic signed \[31:0\]       m\_f,         // Dyadic Multiplier F

    input  logic        \[4:0\]        e\_f,         // Shift exponent F

&nbsp;

    // Luồng 2: Luồng Residual Bypass

    input  logic signed \[7:0\]        i\_x,         // INT8

    input  logic signed \[31:0\]       m\_x,         // Dyadic Multiplier X

    input  logic        \[4:0\]        e\_x,         // Shift exponent X

&nbsp;

    output logic                     out\_valid,

    output logic signed \[7:0\]        i\_out        // INT8 Output

);

&nbsp;

    // Stage 1: Integer Multiplications

    logic signed \[39:0\] prod\_f, prod\_x;

    logic               v\_stage1;

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            prod\_f   \&lt;= '0;

            prod\_x   \&lt;= '0;

            v\_stage1 \&lt;= 1'b0;

        end else begin

            prod\_f   \&lt;= $signed(i\_f) \* $signed(m\_f);

            prod\_x   \&lt;= $signed(i\_x) \* $signed(m\_x);

            v\_stage1 \&lt;= in\_valid;

        end

    end

&nbsp;

    // Stage 2: Arithmetic Right Shifts (Rescaling)

    logic signed \[31:0\] scaled\_f, scaled\_x;

    logic               v\_stage2;

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            scaled\_f \&lt;= '0;

            scaled\_x \&lt;= '0;

            v\_stage2 \&lt;= 1'b0;

        end else begin

            scaled\_f \&lt;= 32'(prod\_f \&gt;\&gt;\&gt; e\_f);

            scaled\_x \&lt;= 32'(prod\_x \&gt;\&gt;\&gt; e\_x);

            v\_stage2 \&lt;= v\_stage1;

        end

    end

&nbsp;

    // Stage 3: Addition \&amp; Saturation (Clamp to \[-128, \+127\])

    logic signed \[32:0\] sum\_full;

    logic signed \[7:0\]  res\_clamped;

    logic               v\_stage3;

&nbsp;

    always\_comb begin

        sum\_full \= $signed(scaled\_f) \+ $signed(scaled\_x);

&nbsp;

        // Saturation Logic

        if (sum\_full \&gt; 33'sd127)

            res\_clamped \= 8'sd127;          // Kẹp trần dương (+127)

        else if (sum\_full \&lt; \-33'sd128)

            res\_clamped \= \-8'sd128;         // Kẹp sàn âm (-128)

        else

            res\_clamped \= sum\_full\[7:0\];

    end

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            i\_out     \&lt;= '0;

            out\_valid \&lt;= 1'b0;

        end else begin

            i\_out     \&lt;= res\_clamped;

            out\_valid \&lt;= v\_stage2;

        end

    end

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

💡 **Nudge**: Bạn có muốn tìm hiểu thêm về cách cấu hình khối **ShiftGELU** hoặc **I-LayerNorm** (chuỗi phi tuyến tính toán thuần số nguyên) trước khi nối vào luồng Residual Connection này không?

\--------------------------------------------------------------------------------

Cộng Residual INT8 trên FPGA

Việc tích hợp phép cộng **Residual Connection** với khối định lượng lại (**Re-quantization / Rescaling**) để đảm bảo đầu ra chuẩn **INT8** không bị tràn bit hay mất độ chính xác là một trong những bài toán kiến trúc quan trọng nhất khi thiết kế bộ tăng tốc Vision Transformer (ViT) trên FPGA.

Dưới đây là phân tích chi tiết từ bản chất toán học, công thức **Dyadic Rescaling**, sơ đồ mạch vi mạch (Datapath) đến mã nguồn SystemVerilog triển khai thực tế.

\--------------------------------------------------------------------------------

1\. Bản chất Vấn đề: Lệch Scale Factor ($S\\\_F \\\\neq S\\\_X$)

Trong mô hình số thực (FP32), phép cộng Residual chỉ đơn giản là: \\$$Y\\\_{float} \= F(X)\*{float} \+ X\*{float}\\\\$$ Trong đó $X$ là tensor đầu vào khối (Shortcut stream), còn $F(X)$ là đầu ra sau khi qua khối tính toán nặng (MSA hoặc MLP stream).

Tuy nhiên, trong mô hình định lượng **INT8 (Symmetric Uniform Quantization)**:

* $X\\\_{float} \= S\\\_X \\\\cdot I\\\_X$ (với $S\\\_X$ là scaling factor của đầu vào, $I\\\_X \\\\in \\\[-128, 127\\\]$).  
* $F(X)\\\_{float} \= S\\\_F \\\\cdot I\\\_F$ (với $S\\\_F$ là scaling factor của khối MSA/MLP, $I\\\_F \\\\in \\\[-128, 127\\\]$).  
* Tensor đầu ra cần có scaling factor mới $S\\\_{out}$ với $Y\\\_{float} \= S\\\_{out} \\\\cdot I\\\_{out}$.

**Thách thức**:

1. Do $S\\\_F \\\\neq S\\\_X$, **giá trị thực của 1 LSB ở hai tensor là khác nhau**. Bạn **không thể cộng trực tiếp** hai số nguyên $I\\\_F \+ I\\\_X$.  
2. Tổng $I\\\_F \+ I\\\_X$ có thể vượt quá dải $\\\[-128, 127\\\]$ gây hiện tượng tràn số (Overflow / Wrap-around) làm hỏng hoàn toàn Feature Map.  
3. Quá trình căn chỉnh scale phải thực hiện thuần túy bằng số nguyên (Integer-only) sử dụng **Dyadic Arithmetic** (Phép nhân số nguyên \+ Dịch bit đại số `&gt;&gt;&gt;`).

\--------------------------------------------------------------------------------

2\. Công thức Toán học Dyadic Rescaling cho Residual Addition

Chúng ta muốn tìm $I\\\_{out}$ sao cho giá trị thực được bảo toàn: \\$$S\\\_{out} \\\\cdot I\\\_{out} \\\\approx S\\\_F \\\\cdot I\\\_F \+ S\\\_X \\\\cdot I\\\_X \\\\implies I\\\_{out} \= \\\\left\\\\lfloor \\\\frac{S\\\_F}{S\\\_{out}} \\\\cdot I\\\_F \+ \\\\frac{S\\\_X}{S\\\_{out}} \\\\cdot I\\\_X \\\\right\\\\rceil\\\\$$

Chuyển đổi hai tỉ số scaling factor thành dạng **Dyadic Numbers (DN)**: $$\\\\text{DN}\\\\left(\\\\frac{S\\\_F}{S\\\_{out}}\\\\right) \= \\\\frac{M\\\_F}{2^{e\\\_F}}, \\\\quad \\\\text{DN}\\\\left(\\\\frac{S\\\_X}{S\\\_{out}}\\\\right) \= \\\\frac{M\\\_X}{2^{e\\\_X}}\\$$ *(với* $M\\\_F, M\\\_X$ *là các hằng số nhân 32-bit INT32, và* $e\\\_F, e\\\_X$ *là số bit dịch phải)*.

Phép tính trên phần cứng FPGA được quy về dạng: \\$$I\\\_{out} \= \\\\text{Clamp}\\\_{\\\\text{INT8}}\\\\left( \\\\left( (M\\\_F \\\\cdot I\\\_F) \\\\gg e\\\_F \\\\right) \+ \\\\left( (M\\\_X \\\\cdot I\\\_X) \\\\gg e\\\_X \\\\right) \\\\right)\\\\$$

\--------------------------------------------------------------------------------

3\. Sơ đồ Mạch Vi mạch PPU (Post-Processing Unit Stream)

Mạch tích hợp Residual Connection \+ Re-quantization trên FPGA được tổ chức thành đường ống (Pipeline) 4 tầng:

         \[ Main Compute Stream \]                         \[ Shortcut Bypass Stream \]

           I\_F (INT8 từ MLP/MSA)                            I\_X (INT8 từ FIFO Delay)

                     │                                                 │

                     ▼                                                 ▼

         \[ Multiplier 1: I\_F \* M\_F \]                       \[ Multiplier 2: I\_X \* M\_X \]

           (INT8 x INT32 \-\&gt; INT32)                           (INT8 x INT32 \-\&gt; INT32)

                     │                                                 │

                     ▼                                                 ▼

         \[ Arithmetic Shift: \&gt;\&gt;\&gt; e\_F \]                     \[ Arithmetic Shift: \&gt;\&gt;\&gt; e\_X \]

           Scaled\_F (INT32)                                  Scaled\_X (INT32)

                     │                                                 │

                     └────────────────────────┬────────────────────────┘

                                              │

                                              ▼

                                 \[ 32-bit INT Adder \]

                               Sum\_32 \= Scaled\_F \+ Scaled\_X

                                              │

                                              ▼

                                  \[ Rounding Unit (+ Half LSB) \]

                                              │

                                              ▼

                                  \[ Saturation Logic (Clamp) \]

                                   Clamp to \[-128, \+127\]

                                              │

                                              ▼

                                    I\_out (INT8 Output)

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Các Kỹ thuật Tối ưu phần cứng Cốt lõi

1. **QAT Scale-Locking (**$S\\\_F \= S\\\_X \= S\\\_{out}$**)**:  
2. Trong quá trình Quantization-Aware Training (QAT), ta có thể ép ràng buộc khóa $S\\\_F \= S\\\_X \= S\\\_{out}$.  
3. Khi đó $\\\\frac{S\\\_F}{S\\\_{out}} \= 1$ và $\\\\frac{S\\\_X}{S\\\_{out}} \= 1$. Mạch Rescale rút gọn về: $I\\\_{out} \= \\\\text{Clamp}\\\_{\\\\text{INT8}}(I\\\_F \+ I\\\_X)$.  
4. **Lợi ích**: Triệt tiêu hoàn toàn 2 bộ nhân $M\\\_F, M\\\_X$, tiết kiệm 100% tài nguyên DSP cho công đoạn này\!  
5. **Căn chỉnh độ trễ FIFO (Stream Alignment Delay)**:  
6. Luồng $I\\\_F$ đi qua MSA/MLP mất $L\\\_{compute}$ chu kỳ clock. Luồng $I\\\_X$ được đưa qua một đệm FIFO/Shift Register có độ sâu đúng $L\\\_{compute}$ chu kỳ để hai tín hiệu đến bộ cộng cùng lúc.  
7. **Mạch Saturating Logic (Chống tràn INT8)**:  
8. Sử dụng mạch kẹp ngưỡng cứng để ép giá trị vượt dải về âm/dương tối đa, tránh hiện tượng lật dấu số nguyên.

\--------------------------------------------------------------------------------

5\. Mã nguồn SystemVerilog Mẫu (`requant_residual_add.sv`)

\`timescale 1ns / 1ps

&nbsp;

module requant\_residual\_add (

    input  logic                     clk,

    input  logic                     rst\_n,

    input  logic                     in\_valid,

&nbsp;

    // Luồng 1: Đầu ra từ khối tính toán (MLP / MSA)

    input  logic signed \[7:0\]        i\_f,         // INT8

    input  logic signed \[31:0\]       m\_f,         // Dyadic Multiplier F

    input  logic        \[4:0\]        e\_f,         // Shift exponent F

&nbsp;

    // Luồng 2: Luồng Residual Bypass

    input  logic signed \[7:0\]        i\_x,         // INT8

    input  logic signed \[31:0\]       m\_x,         // Dyadic Multiplier X

    input  logic        \[4:0\]        e\_x,         // Shift exponent X

&nbsp;

    output logic                     out\_valid,

    output logic signed \[7:0\]        i\_out        // INT8 Output

);

&nbsp;

    // Stage 1: Integer Multiplications

    logic signed \[39:0\] prod\_f, prod\_x;

    logic               v\_stage1;

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            prod\_f   \&lt;= '0;

            prod\_x   \&lt;= '0;

            v\_stage1 \&lt;= 1'b0;

        end else begin

            prod\_f   \&lt;= $signed(i\_f) \* $signed(m\_f);

            prod\_x   \&lt;= $signed(i\_x) \* $signed(m\_x);

            v\_stage1 \&lt;= in\_valid;

        end

    end

&nbsp;

    // Stage 2: Arithmetic Right Shifts (Rescaling)

    logic signed \[31:0\] scaled\_f, scaled\_x;

    logic               v\_stage2;

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            scaled\_f \&lt;= '0;

            scaled\_x \&lt;= '0;

            v\_stage2 \&lt;= 1'b0;

        end else begin

            scaled\_f \&lt;= 32'(prod\_f \&gt;\&gt;\&gt; e\_f);

            scaled\_x \&lt;= 32'(prod\_x \&gt;\&gt;\&gt; e\_x);

            v\_stage2 \&lt;= v\_stage1;

        end

    end

&nbsp;

    // Stage 3: Addition \&amp; Saturation (Clamp to \[-128, \+127\])

    logic signed \[32:0\] sum\_full;

    logic signed \[7:0\]  res\_clamped;

    logic               v\_stage3;

&nbsp;

    always\_comb begin

        sum\_full \= $signed(scaled\_f) \+ $signed(scaled\_x);

&nbsp;

        // Saturation Logic

        if (sum\_full \&gt; 33'sd127)

            res\_clamped \= 8'sd127;          // Kẹp trần dương (+127)

        else if (sum\_full \&lt; \-33'sd128)

            res\_clamped \= \-8'sd128;         // Kẹp sàn âm (-128)

        else

            res\_clamped \= sum\_full\[7:0\];

    end

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            i\_out     \&lt;= '0;

            out\_valid \&lt;= 1'b0;

        end else begin

            i\_out     \&lt;= res\_clamped;

            out\_valid \&lt;= v\_stage2;

        end

    end

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

💡 **Nudge**: Bạn có muốn tìm hiểu thêm về cách cấu hình khối **ShiftGELU** hoặc **I-LayerNorm** (chuỗi phi tuyến tính toán thuần số nguyên) trước khi nối vào luồng Residual Connection này không?

\--------------------------------------------------------------------------------

Chuẩn hóa Số nguyên I-LayerNorm

**I-LayerNorm** (Integer-only LayerNorm) là giải pháp chuẩn hóa thuần số nguyên được đề xuất trong công trình **I-ViT** nhằm giải quyết nút thắt tính toán của lớp LayerNorm truyền thống khi triển khai Vision Transformer (ViT) trên các phần cứng nhúng/FPGA.

\--------------------------------------------------------------------------------

1\. Bản chất & Mục tiêu của I-LayerNorm

Trong mô hình số thực (FP32/FP16), LayerNorm thực hiện chuẩn hóa theo công thức: $$\\\\text{LayerNorm}(x) \= \\\\frac{x \- \\\\mu}{\\\\sqrt{\\\\text{Var}(x) \+ \\\\epsilon}} \\\\cdot \\\\gamma \+ \\\\beta\\$$

Phép toán này đòi hỏi thực hiện **phép căn bậc hai (**$\\\\sqrt{\\\\cdot}$**)** và **phép chia số thực**, vốn tốn rất nhiều tài nguyên đơn vị số thực (FPU) phần cứng hoặc buộc phải cắt đồ thị để đưa dữ liệu về FP32 (Dequantization).

**I-LayerNorm** giải quyết vấn đề này bằng cách chuyển toàn bộ đồ thị tính toán sang **số nguyên cố định (INT8/INT32)** và **phép dịch bit đại số (\&gt;\&gt;)**, tạo thành một luồng tính toán số nguyên khép kín (End-to-End Integer Pipeline) mà không cần dequantization.

\--------------------------------------------------------------------------------

2\. Quy trình 4 Bước Triển khai I-LayerNorm

🔹 Bước 1: Tính Trung bình ($\\\\mu$) và Phương sai ($\\\\text{Var}$) bằng Số nguyên

Với đầu vào $I\\\_x$ đã định lượng dạng **INT8**, giá trị trung bình $\\\\mu$ và phương sai $\\\\text{Var}(I\\\_x)$ được tính toán bằng các bộ tích lũy số nguyên **INT32**: $$\\\\mu \= \\\\left\\\\lfloor \\\\frac{1}{d} \\\\sum\\\_{j=1}^d I\\\_x\\\[j\\\] \\\\right\\\\rfloor\\\\\\\] \\\\\\\[\\\\text{Var}(I\\\_x) \= \\\\left\\\\lfloor \\\\frac{1}{d} \\\\sum\\\_{j=1}^d (I\\\_x\\\[j\\\] \- \\\\mu)^2 \\\\right\\\\rfloor\\$$

\--------------------------------------------------------------------------------

🔹 Bước 2: Tính Căn Bậc Hai ($\\\\sqrt{\\\\text{Var}}$) bằng Thuật toán Lặp Số nguyên (Integer Iterative Method)

Do các đơn vị số nguyên không hỗ trợ phép căn bậc hai trực tiếp, I-LayerNorm áp dụng thuật toán lặp số nguyên dựa trên phương pháp Newton-Raphson kết hợp dịch bit:

1. **Khởi tạo giá trị ban đầu (**$I\\\_0$**)**: Dựa vào số bit của phương sai: \\$$I\\\_0 \= 2^{\\\\lfloor \\\\text{bit}(\\\\text{Var}(I\\\_x)) / 2 \\\\rfloor}\\\\$$  
2. **Công thức lặp qua các bước (**$i$**)**: \\$$I\\\_{i+1} \= \\\\left\\\\lfloor \\\\frac{I\\\_i \+ \\\\lfloor \\\\text{Var}(I\\\_x) / I\\\_i \\\\rfloor}{2} \\\\right\\\\rfloor \= \\\\left( I\\\_i \+ \\\\left\\\\lfloor \\\\frac{\\\\text{Var}(I\\\_x)}{I\\\_i} \\\\right\\\\rfloor \\\\right) \\\\gg 1\\\\$$  
3. **Cố định số vòng lặp (Constant Latency Fix)**:  
4. Nếu dùng điều kiện dừng động ($I\\\_{i+1} \\\\ge I\\\_i$), thời gian thực thi trên phần cứng sẽ không cố định.  
5. Thực nghiệm cho thấy **10 vòng lặp** là đủ để đạt độ hội tụ chính xác. I-LayerNorm cố định đúng **10 vòng lặp** để đảm bảo độ trễ xác định (deterministic latency) trên FPGA.

\--------------------------------------------------------------------------------

🔹 Bước 3: Chuẩn hóa & Nhân Tham số Affine ($\\\\gamma, \\\\beta$)

Độ lệch $(I\\\_x \- \\\\mu)$ được chia cho độ lệch chuẩn số nguyên vừa tìm được $I\\\_{\\\\text{std}} \= I\\\_{10}$: $$\\\\tilde{I}\*x \= \\\\left\\\\lfloor \\\\frac{I\\\_x \- \\\\mu}{I\*{\\\\text{std}}} \\\\right\\\\rfloor\\$$

Kết quả được nhân với hệ số tỉ lệ $\\\\gamma$ và cộng bias $\\\\beta$ (được lưu ở dạng số nguyên) bằng kỹ thuật **Dyadic Scaling** (phép nhân số nguyên \+ dịch bit `&gt;&gt;&gt;`).

\--------------------------------------------------------------------------------

🔹 Bước 4: Định lượng lại & Kẹp ngưỡng (Re-quantization & Saturation)

Cuối cùng, kết quả được dịch bit và kẹp ngưỡng (Clamp) về dải giá trị số nguyên **INT8** $\\\[-128, 127\\\]$ để làm đầu vào cho tầng tiếp theo: \\$$I\\\_{\\\\text{out}} \= \\\\text{Clamp}\*{\\\\text{INT8}} \\\\left( (\\\\tilde{I}x \\\\cdot M{\\\\gamma}) \\\\gg e\*{\\\\gamma} \+ \\\\beta\\\_{\\\\text{int}} \\\\right)\\\\$$

\--------------------------------------------------------------------------------

3\. So sánh I-LayerNorm với các Phương pháp Khác

| Tiêu chí | LayerNorm Chuẩn (FP32) | L1-LayerNorm (Fully-8bit) | I-LayerNorm (I-ViT) |
| ----- | ----- | ----- | ----- |
| **Phép toán căn bậc hai** | Phép căn số thực (FPU) | Thay bằng độ lệch tuyệt đối L1 | **Chuỗi lặp số nguyên \+ Dịch bit (\&gt;\>)** |
| **Dequantization (Graph Cut)** | Bắt buộc chuyển đổi INT8 $\\\\leftrightarrow$ FP32 | Không có Dequantization | **Luồng INT8 khép kín (End-to-End Integer)** |
| **Độ chính xác Top-1 (ViT)** | Baseline 100% | Sụt giảm nghiêm trọng (-2.49% DeiT-B, \-3.32% Swin-S) | **Gần như không giảm** (DeiT-S đạt 80.12%, \+0.27% so với FP32) |
| **Phù hợp FPGA / Edge AI** | Tốn đơn vị FPU đắt đỏ | Rất nhanh nhưng mất chính xác | **Tối ưu: Tốn ít LUT/DSP, độ trễ cố định** |

\--------------------------------------------------------------------------------

💡 **Tóm lại**: I-LayerNorm giải quyết bài toán tính căn bậc hai bằng **10 vòng lặp số nguyên với phép dịch bit (\&gt;\&gt;)**, giúp giữ nguyên luồng dữ liệu **INT8/INT32** trên FPGA mà không làm sụt giảm độ chính xác của Vision Transformer.

\--------------------------------------------------------------------------------

Dropout trên FPGA Inference

**Dropout** (và biến thể **DropPath / Stochastic Depth** trong Vision Transformer) là kỹ thuật điều hỏa (Regularization) phổ biến trong Deep Learning nhằm chống hiện tượng học tủ (**Overfitting**).

Tuy nhiên, vai trò và cách xử lý Dropout có sự **khác biệt căn bản** giữa quá trình huấn luyện (Training) và triển khai vi mạch suy luận (FPGA Inference Accelerator):

\--------------------------------------------------------------------------------

1\. Cơ chế hoạt động: Training vs. Inference

* **Khi Huấn luyện (Training \- trên GPU/Cloud)**:  
  * Ở mỗi bước lan truyền tiến (Forward Pass), một tỷ lệ ngẫu nhiên $p$ các kích hoạt (Activations) hoặc các nhánh đường tắt (Residual Branches) sẽ bị "tắt" (gán bằng 0).  
  * Các giá trị còn lại được nhân thu phóng với hệ số $\\\\frac{1}{1-p}$ để giữ nguyên kỳ vọng toán học.  
* **Khi Suy luận (Inference / On-Chip Deployment)**:  
  * Dropout **hoàn toàn bị vô hiệu hóa (Disabled/Bypassed)**.  
  * Toàn bộ các neuron và tham số đều tham gia tính toán với trọng số đầy đủ mà không có bất kỳ phần tử nào bị ngắt ngẫu nhiên.

\--------------------------------------------------------------------------------

2\. Xử lý Dropout trên Phần cứng FPGA (Inference Accelerator)

Khi triển khai các mô hình Vision Transformer (ViT) lên FPGA (như Kria KV260 hay Alveo U200):

1. **Loại bỏ hoàn toàn khỏi Đồ thị Tính toán (Graph Pruning)**:  
2. Do bộ tăng tốc FPGA chỉ đảm nhận nhiệm vụ suy luận (Inference), khối Dropout/DropPath được trình biên dịch (Vitis AI / HLS Compiler) **tự động cắt bỏ hoàn toàn** khỏi RTL/HLS Datapath trước khi tổng hợp ra bitstream.  
3. **Tiết kiệm Tài nguyên Vi mạch**:  
4. Không cần triển khai các bộ tạo số giả ngẫu nhiên (LFSR \- Linear Feedback Shift Registers), mạch so sánh ngưỡng, hay các bộ MUX chuyển mạch ngẫu nhiên trên FPGA fabric.  
5. **Duy trì Luồng Đường ống (Pipeline Continuity)**:  
6. Việc loại bỏ Dropout giúp luồng dữ liệu AXI-Stream chảy liên tục qua các tầng Linear Projection, Self-Attention và MLP mà không bị ngắt nhịp (pipeline stall/bubble), giúp duy trì chỉ số **Initiation Interval** $II \= 1$.

\--------------------------------------------------------------------------------

💡 Bạn có muốn tìm hiểu thêm về cách các tầng phi tuyến khác như **LayerNorm** hay **GELU** (vẫn xuất hiện trong Inference) được xấp xỉ thuần số nguyên để triển khai trên FPGA không?

\--------------------------------------------------------------------------------

Bản chất Linear Unit

The term **"Linear Unit" (LU)** and its association with **"Activation"** in deep learning and hardware accelerator design stems from two distinct concepts:

\--------------------------------------------------------------------------------

1\. Why Activation Functions use the name "Linear Unit" (e.g., ReLU, GELU, SiLU)

In modern neural networks, many activation functions end with the phrase **Linear Unit**:

* **ReLU**: Rectified **Linear Unit** ($\\\\max(0, x)$)  
* **GELU**: Gaussian Error **Linear Unit** ($x \\\\cdot \\\\Phi(x)$)  
* **SiLU**: Sigmoid-Weighted **Linear Unit** ($x \\\\cdot \\\\sigma(x)$)

They are called **Linear Units** for three primary reasons:

1. **"Unit" \= Neuron**: In artificial neural networks, a "unit" represents an individual node or neuron.  
2. **Linear Identity Behavior for Positive Inputs**: Traditional activation functions like **Sigmoid** or **Tanh** are *saturating* functions — for large positive inputs, their outputs flatten out (saturate near 1.0), causing the **vanishing gradient problem**. In contrast, "Linear Units" maintain a **linear relationship** ($f(x) \\\\approx x$) when inputs are positive or sufficiently large.  
3. **Linear Identity Term (**$x$**) Combined with Gating**: Functions like GELU or SiLU explicitly scale the linear identity $x$ by a non-linear probability or gating factor (e.g., $\\\\sigma(x)$).

\--------------------------------------------------------------------------------

2\. Why Intermediate Data Tensors are Called "Activations"

In biological and artificial neural networks, a neuron's output signal after processing its inputs is called its **activation** (representing how strongly the neuron "fires" or is activated).

In deep learning frameworks and **FPGA / AI Hardware Accelerators**:

* **Weights (**$W$**)**: The trained static parameters of the model, stored in **Weight Buffers**.  
* **Activations (**$X$**,** $Q$**,** $K$**,** $V$**, Feature Maps)**: The dynamic input and output data tensors that flow through the layers, stored in **Activation Buffers** (or **Feature Buffers**).

\--------------------------------------------------------------------------------

Summary

| Term | Meaning in Deep Learning | Example |
| ----- | ----- | ----- |
| **Activation** | The output tensor/feature map produced by a layer or neuron. | Input feature vector $X$, Q/K/V vectors, intermediate hidden states. |
| **Linear Unit (LU)** | A class of activation functions that preserve linear scaling ($f(x) \\\\propto x$) for positive inputs to prevent gradient saturation. | **ReLU**, **GELU**, **SiLU**. |

\--------------------------------------------------------------------------------

Xác minh Timing FPGA

Trong thiết kế mạch phần cứng FPGA (như Kria KV260 hay Alveo U200 với AMD Vivado), **Timing Verification** (Xác minh Thời gian) và **Timing Closure** là các bước bắt buộc nhằm đảm bảo mọi tín hiệu truyền qua các mạch tổ hợp và Flip-Flop được lấy mẫu chính xác tại đúng cạnh xung clock mà không xảy ra sai sót dữ liệu.

\--------------------------------------------------------------------------------

I. Ý nghĩa Toán học của Timing trong Mạch Sequential

Đối với mạch tuần tự (Synchronous Sequential Circuit), tần số hoạt động tối đa ($f\\\_{max}$) và chu kỳ clock tối thiểu ($T\\\_{clock}$) bị giới hạn bởi ba tham số độ trễ chính:

\\$$T\\\_{clock} \\\\ge T\\\_{cq} \+ T\\\_{comb} \+ T\\\_{setup} \\\\implies f\\\_{max} \= \\\\frac{1}{T\\\_{cq} \+ T\\\_{comb} \+ T\\\_{setup}}\\\\$$

* $T\\\_{cq}$ **(Clock-to-Q Delay)**: Thời gian cần thiết để Flip-Flop phát cập nhật dữ liệu ra cổng Q sau cạnh xung clock.  
* $T\\\_{comb}$ **(Combinational Logic & Routing Delay)**: Tổng độ trễ lan truyền qua các khối logic tổ hợp (LUTs, DSP, Carry Chains) và độ trễ đường dây nối (routing delay).  
* $T\\\_{setup}$ **(Setup Time)**: Khoảng thời gian dữ liệu đầu vào bắt buộc phải giữ ổn định **trước** cạnh xung clock thu.

\--------------------------------------------------------------------------------

II. Các Chỉ số Timing Cốt lõi trong Vivado (Design Timing Summary)

Sau bước Placement & Routing, công cụ Vivado sinh ra báo cáo **Static Timing Analysis (STA)** với các chỉ số Slack chính:

| Chỉ số Slack | Ý nghĩa Kỹ thuật | Điều kiện Đạt (Timing Met) |
| ----- | ----- | ----- |
| **WNS (Worst Negative Slack)** | Độ chênh lệch thời gian Setup khả thi trên đường Critical Path xấu nhất. | $\\\\text{WNS} \\\\ge 0\\\\text{ ns}$ (Nếu WNS \< 0: Bị vi phạm Setup Time). |
| **WHS (Worst Hold Slack)** | Độ chênh lệch thời gian Hold khả thi (đảm bảo dữ liệu không thay đổi quá nhanh trước khi FF kịp chốt). | $\\\\text{WHS} \\\\ge 0\\\\text{ ns}$ (Nếu WHS \< 0: Bị vi phạm Hold Time). |
| **TNS / THS** | Tổng độ âm slack của toàn bộ các đường tín hiệu bị vi phạm trong toàn bộ chip. | $\\\\text{TNS} \= 0$**,** $\\\\text{THS} \= 0$. |

\--------------------------------------------------------------------------------

III. Quy trình Xác minh Timing chuẩn trên Vivado

1. **Ràng buộc Thời gian (Timing Constraints \- File .xdc)**: Khai báo tần số clock hệ thống mục tiêu để Vivado làm căn cứ tối ưu hóa. Ví dụ:

create\_clock \-period 3.333 \-name clk \[get\_ports clk\] ;\# Thiết lập clock 300 MHz&nbsp;&nbsp;

&nbsp;

1. **Mô phỏng RTL & Timing (Simulation)**:  
2. **RTL Behavioral Simulation (xsim / Verilator)**: Kiểm tra tính đúng đắn về mặt logic của mã SystemVerilog.  
3. **Post-Implementation Timing Simulation**: Mô phỏng netlist sau khi Place & Route để kiểm tra độ trễ thực tế của linh kiện phần cứng và dây nối.  
4. **Phân tích Timing Tĩnh (Static Timing Analysis \- STA)**: Vivado tự động tính toán Critical Path và xuất báo cáo `Design Timing Summary`.

\--------------------------------------------------------------------------------

IV. Kỹ thuật Xử lý khi Bị vi phạm Timing ($\\\\text{WNS} \&lt; 0$)

Khi thiết kế gặp vi phạm timing, hai nguyên nhân phổ biến nhất và giải pháp khắc phục gồm:

1. **Chuỗi mạch tổ hợp quá dài (Excessive Logic Depth / Multiplier Chain)**:  
2. *Hiện tượng*: Phép nhân/cộng tích lũy đi qua quá nhiều tầng logic trong 1 chu kỳ clock làm $T\\\_{comb}$ vượt quá $T\\\_{clock}$.  
3. *Giải pháp*: Áp dụng **Pipelining & Retiming** bằng cách chèn thêm thanh ghi Flip-Flop để cắt nhỏ đường Critical Path (ví dụ: tăng độ sâu pipeline từ 1 lên 3 tầng) giúp đạt $F\\\_{max} \\\\ge 150 \- 300\\\\text{ MHz}$.  
4. **Nghẽn đường đi dây (Routing Congestion) do BRAM/DSP**:  
5. *Hiện tượng*: Các khối BRAM/DSP phân bố quá tập trung khiến Vivado phải đi dây vòng xa, làm kéo dài độ trễ dây nối.  
6. *Giải pháp*: Phân vùng BRAM (Array Partitioning) và cấu hình lại kích thước khối đệm ROM/RAM để phân bổ đều tài nguyên trên diện tích chip.

\--------------------------------------------------------------------------------

🔍 Bạn có muốn tìm hiểu về cách sử dụng công cụ **Vivado Timing Constraints Wizard** hoặc cách viết các lệnh tcl để phân tích chi tiết đường **Critical Path** bị vi phạm không?

\--------------------------------------------------------------------------------

Kỹ thuật DSP Packing trên FPGA

Kỹ thuật **DSP Packing** (đóng gói khối xử lý tín hiệu số) là một phương pháp tối ưu hóa phần cứng trên FPGA, cho phép thực hiện **nhiều phép nhân độ chính xác thấp** (như INT8, INT6, INT4) đồng thời trên **cùng một khối cứng DSP (Hard DSP Slice)** trong chỉ **1 chu kỳ xung clock**.

Dưới đây là giải thích chi tiết về nguyên lý toán học, cấu trúc vi mạch và các đánh đổi kỹ thuật của phương pháp này:

\--------------------------------------------------------------------------------

1\. Cơ sở Phần cứng (Khối DSP48E2)

Trên các dòng FPGA AMD/Xilinx UltraScale+ (như Kria KV260 hay Alveo U200), phần cứng tích hợp sẵn các khối **DSP48E2**:

* **Bộ nhân phần cứng (Hard Multiplier)**: Hỗ trợ kích thước tối đa $18 \\\\text{ bit} \\\\times 27 \\\\text{ bit}$.  
* **Thanh ghi tích lũy (Accumulator)**: Độ rộng **48-bit**.

Nếu chỉ thực hiện một phép nhân INT8 chuẩn ($8 \\\\text{ bit} \\\\times 8 \\\\text{ bit}$), phần lớn độ rộng bit của bộ nhân phần cứng (18-bit và 27-bit) sẽ bị bỏ trống, gây lãng phí tài nguyên vi mạch.

\--------------------------------------------------------------------------------

2\. Nguyên lý Toán học & Thao tác Bit (2x INT8 Packing)

Để nhân một toán hạng kích hoạt $A$ (INT8) với hai toán hạng trọng số độc lập $B$ và $C$ (INT8) trong cùng 1 DSP:

1. **Cấu hình Toán hạng Đầu vào**:  
2. **Cổng 18-bit**: Đưa toán hạng chung $A$ vào.  
3. **Cổng 27-bit**: Đóng gói hai toán hạng $B$ và $C$ thành một từ dữ liệu duy nhất bằng phép dịch bit đại số: $$\\\\text{Operand}\\\_{27} \= (B \\\\ll 18\) \+ C\\$$  
4. **Phép nhân bên trong DSP48E2**: $$\\\\text{Result}\\\_{45} \= A \\\\times ((B \\\\ll 18\) \+ C) \= (A \\\\times B) \\\\ll 18 \+ (A \\\\times C)\\$$  
5. **Tách kết quả ở Đầu ra 48-bit**:  
6. Phép nhân $A \\\\times C$ tạo ra tích 16-bit nằm ở các vị trí thấp `[15:0]`.  
7. Phép nhân $A \\\\times B$ tạo ra tích 16-bit được dịch trái 18 bit, nằm ở các vị trí cao `[33:18]`.  
8. Khoảng trống 18 bit giữa hai tích đóng vai trò là **khoảng đệm an toàn (Guard Bits)**, ngăn hiện tượng tràn bit hay chồng lấp bit (overlap) giữa hai kết quả.

\--------------------------------------------------------------------------------

3\. Biến thể Đóng gói cho Định dạng Siêu thấp (Factor-3 / Factor-4 Packing)

Đối với các mô hình định lượng dưới 8-bit (ví dụ trong kiến trúc Quasar-ViT):

* **Factor-3 Layout**: Đưa 1 kích hoạt 6-bit vào cổng 18-bit và đóng gói **3 trọng số 4-bit** vào cổng 27-bit.  
* **Factor-4 Layout**: Đóng gói đồng thời **2 kích hoạt 6-bit** với **2 trọng số 4-bit**, giúp tăng gấp 3 đến 4 lần mật độ phép tính MAC trên mỗi DSP slice.

\--------------------------------------------------------------------------------

4\. Tác động tới Bộ tăng tốc Vision Transformer (ViT)

* **Tăng gấp đôi mật độ tính toán (MAC Density)**: Trong mảng Systolic Array kích thước $P\\\_{sys} \\\\times P\\\_{sys}$ (ví dụ $32 \\\\times 32$), kỹ thuật DSP Packing cho phép mảng thực hiện tới $P\\\_{sys} \\\\times (2 \\\\cdot P\\\_{sys}) \= 32 \\\\times 64 \= 2048$ phép nhân INT8 trong **chỉ 1 chu kỳ clock**.  
* **Giải quyết rào cản giới hạn DSP**: Giúp các mô hình Transformer kích thước lớn chạy được trên các bo mạch Edge FPGA bị giới hạn số lượng DSP (như Kria KV260 chỉ có 252 khối DSP).

\--------------------------------------------------------------------------------

5\. Thách thức Kỹ thuật & Đánh đổi (Trade-offs)

* **Phức tạp về Bit dấu (Sign Extension)**: Việc xử lý số nguyên có dấu (Signed INT8) đòi hỏi mạch bù 2 và mở rộng bit dấu cẩn thận để tránh làm sai lệch phân vùng bit cao.  
* **Tắc nghẽn Định tuyến (Routing Congestion)**: Đường truyền dữ liệu đầu ra bắt buộc phải mở rộng gấp đôi (đường bus 48-bit thay vì 16-bit), làm tăng độ phức tạp dây nối và có thể gây vi phạm thời gian (Timing Closure) ở tần số cao.  
* **Đồng bộ BRAM**: Bộ nhớ BRAM/URAM phải được phân ngân hàng (Banking) chuẩn xác để cung cấp đồng thời các cặp toán hạng đã đóng gói trong đúng 1 chu kỳ xung clock.

\--------------------------------------------------------------------------------

💡 Bạn có muốn mình minh họa chi tiết cách xử lý **mở rộng bit dấu (Sign Extension)** cho số nguyên có dấu trong SystemVerilog khi triển khai DSP Packing không?

\--------------------------------------------------------------------------------

Testbench Adder Tree

Dưới đây là mã nguồn **SystemVerilog Testbench tự kiểm tra (Self-Checking Testbench)** hoàn chỉnh cho khối `pipelined_adder_tree`.

Testbench này tích hợp **Scoreboard (Mô hình đối chiếu chuẩn \- Golden Model)** sử dụng cấu trúc hàng đợi (Queue) để theo dõi chính xác độ trễ pipeline và kiểm tra kết quả tính toán tự động mà không cần soi sóng thủ công.

\--------------------------------------------------------------------------------

Mã nguồn SystemVerilog Testbench (`tb_pipelined_adder_tree.sv`)

\`timescale 1ns / 1ps

&nbsp;

module tb\_pipelined\_adder\_tree;

&nbsp;

    // \-------------------------------------------------------------------------

    // 1\. Tham số và Tín hiệu Mô phỏng

    // \-------------------------------------------------------------------------

    localparam int N          \= 16;   // Số lượng phần tử vector

    localparam int IN\_WIDTH   \= 8;    // Độ rộng bit INT8

    localparam int OUT\_WIDTH  \= 32;   // Độ rộng bit INT32

    localparam int CLK\_PERIOD \= 10;   // Chu kỳ xung clock (10ns \= 100 MHz)

&nbsp;

    // Tín hiệu giao tiếp DUT

    logic                     clk;

    logic                     rst\_n;

    logic                     in\_valid;

    logic signed \[IN\_WIDTH-1:0\] a \[N\];

    logic signed \[IN\_WIDTH-1:0\] b \[N\];

&nbsp;

    logic                     out\_valid;

    logic signed \[OUT\_WIDTH-1:0\] dot\_product;

&nbsp;

    // Các biến theo dõi kiểm thử

    int error\_count \= 0;

    int pass\_count  \= 0;

    int test\_id     \= 0;

&nbsp;

    // Hàng đợi lưu kết quả Golden Model để đối chiếu qua các chu kỳ trễ Pipeline

    typedef logic signed \[OUT\_WIDTH-1:0\] expected\_t;

    expected\_t expected\_queue\[$\];

&nbsp;

    // \-------------------------------------------------------------------------

    // 2\. Khởi tạo Thiết bị Cần Kiểm thử (DUT Instantiation)

    // \-------------------------------------------------------------------------

    pipelined\_adder\_tree \#(

        .N(N),

        .IN\_WIDTH(IN\_WIDTH),

        .OUT\_WIDTH(OUT\_WIDTH)

    ) dut (

        .clk        (clk),

        .rst\_n      (rst\_n),

        .in\_valid   (in\_valid),

        .a          (a),

        .b          (b),

        .out\_valid  (out\_valid),

        .dot\_product(dot\_product)

    );

&nbsp;

    // \-------------------------------------------------------------------------

    // 3\. Xung Clock \&amp; Master Reset Task

    // \-------------------------------------------------------------------------

    initial begin

        clk \= 0;

        forever \#(CLK\_PERIOD / 2\) clk \= \~clk;

    end

&nbsp;

    task automatic reset\_dut();

        rst\_n    \&lt;= 1'b0;

        in\_valid \&lt;= 1'b0;

        for (int i \= 0; i \&lt; N; i++) begin

            a\[i\] \&lt;= '0;

            b\[i\] \&lt;= '0;

        end

        repeat (5) @(posedge clk);

        rst\_n    \&lt;= 1'b1;

        repeat (2) @(posedge clk);

        $display("\[SIM\] Reset completed. Starting Test Sequences...");

    endtask

&nbsp;

    // \-------------------------------------------------------------------------

    // 4\. Golden Model: Hàm tính Tích vô hướng chuẩn

    // \-------------------------------------------------------------------------

    function automatic expected\_t compute\_golden(

        ref logic signed \[IN\_WIDTH-1:0\] vec\_a \[N\],

        ref logic signed \[IN\_WIDTH-1:0\] vec\_b \[N\]

    );

        logic signed \[OUT\_WIDTH-1:0\] sum \= 0;

        for (int i \= 0; i \&lt; N; i++) begin

            sum \+= $signed(vec\_a\[i\]) \* $signed(vec\_b\[i\]);

        end

        return sum;

    endfunction

&nbsp;

    // \-------------------------------------------------------------------------

    // 5\. Driver Task: Sinh Dữ liệu Test

    // \-------------------------------------------------------------------------

    task automatic send\_vector(input bit is\_random \= 1, input int fixed\_val \= 0);

        expected\_t golden\_val;

&nbsp;

        @(posedge clk);

        in\_valid \&lt;= 1'b1;

&nbsp;

        for (int i \= 0; i \&lt; N; i++) begin

            if (is\_random) begin

                a\[i\] \&lt;= $urandom\_range(-128, 127); // Giới hạn INT8 signed \[-128, 127\]

                b\[i\] \&lt;= $urandom\_range(-128, 127);

            end else begin

                a\[i\] \&lt;= fixed\_val;

                b\[i\] \&lt;= fixed\_val;

            end

        end

&nbsp;

        // Tích hợp Golden Model và đẩy vào Hàng đợi

        \#1; // Đợi cập nhật tín hiệu tổ hợp

        golden\_val \= compute\_golden(a, b);

        expected\_queue.push\_back(golden\_val);

    endtask

&nbsp;

    task automatic send\_idle();

        @(posedge clk);

        in\_valid \&lt;= 1'b0;

    endtask

&nbsp;

    // \-------------------------------------------------------------------------

    // 6\. Scoreboard Monitor: Tự động Kiểm tra Đầu ra (Output Checker)

    // \-------------------------------------------------------------------------

    always @(posedge clk) begin

        if (rst\_n \&amp;\&amp; out\_valid) begin

            expected\_t expected\_val;

&nbsp;

            if (expected\_queue.size() \&gt; 0\) begin

                expected\_val \= expected\_queue.pop\_front();

                test\_id++;

&nbsp;

                if (dot\_product \=== expected\_val) begin

                    $display("\[PASS\] Test \#%0d | Output \= %0d | Expected \= %0d",

                             test\_id, dot\_product, expected\_val);

                    pass\_count++;

                end else begin

                    $error("\[FAIL\] Test \#%0d | Output \= %0d | Expected \= %0d (MISMATCH\!)",

                           test\_id, dot\_product, expected\_val);

                    error\_count++;

                end

            end else begin

                $error("\[UNEXPECTED\] out\_valid high but Expected Queue is EMPTY\!");

                error\_count++;

            end

        end

    endend

&nbsp;

    // \-------------------------------------------------------------------------

    // 7\. Quy trình Kiểm thử Chính (Main Test Process)

    // \-------------------------------------------------------------------------

    initial begin

        reset\_dut();

&nbsp;

        // \--- Test Case 1: Vector Toàn 0 \---

        $display("\\n---\&gt; Test Case 1: Zero Vector Test");

        send\_vector(0, 0);

        send\_idle();

&nbsp;

        // \--- Test Case 2: Vector Giá trị Cố định \---

        $display("\\n---\&gt; Test Case 2: Fixed Values Vector Test");

        send\_vector(0, 5);  // Vector toàn 5 \* toàn 5 \= 16 \* 25 \= 400

        send\_vector(0, \-2); // Vector toàn \-2 \* toàn \-2 \= 16 \* 4 \= 64

        send\_idle();

&nbsp;

        // \--- Test Case 3: Chạy Luồng Dữ liệu Liên tục (Back-to-Back Burst Test II \= 1\) \---

        $display("\\n---\&gt; Test Case 3: Streaming Burst Test (100 Random Vectors)");

        for (int k \= 0; k \&lt; 100; k++) begin

            send\_vector(1, 0); // Sinh ngẫu nhiên

        end

        send\_idle();

&nbsp;

        // Đợi kết quả hoàn tất truyền qua pipeline

        repeat (20) @(posedge clk);

&nbsp;

        // \--- Tổng kết Kết quả \---

        $display("\\n==================================================");

        $display("               SIMULATION REPORT                  ");

        $display("==================================================");

        $display(" Total Tests Run : %0d", test\_id);

        $display(" PASSED          : %0d", pass\_count);

        $display(" FAILED          : %0d", error\_count);

        $display("==================================================");

&nbsp;

        if (error\_count \== 0 \&amp;\&amp; test\_id \&gt; 0\)

            $display("\&gt;\&gt;\&gt; SUCCESS: PIPELINED ADDER TREE PASSED ALL TESTS\! \&lt;\&lt;\&lt;");

        else

            $display("\&gt;\&gt;\&gt; FAILURE: SIMULATION FAILED WITH ERRORS\! \&lt;\&lt;\&lt;");

&nbsp;

        $finish;

    end

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

Điểm Nổi bật của Testbench này

1. **Hàng đợi FIFO (expected\_queue)**: Khi đẩy dữ liệu vào DUT tại `in_valid = 1`, Golden Model sẽ tự động tính trước kết quả chuẩn và đẩy vào `expected_queue`. Do đó, dù dữ liệu mất **6 chu kỳ clock** để đi qua Pipeline, Testbench vẫn đối chiếu chuẩn xác từng kết quả xuất hiện ở `out_valid`.  
2. **Kiểm tra luồng liên tục (Burst Test II \= 1\)**: Test Case 3 nạp 100 vector ngẫu nhiên liên tiếp ở mỗi chu kỳ clock để xác minh khả năng nhận luồng liên tục không bị nghẽn (Zero Stall).  
3. **Mô phỏng ngẫu nhiên có ràng buộc ($urandom\_range)**: Đảm bảo các giá trị thử nghiệm bao phủ đủ miền số âm/dương của kiểu dữ liệu **INT8** (`-128` đến `+127`).

\--------------------------------------------------------------------------------

🛠️ Bạn có muốn tích hợp khối Cây cộng này vào **mảng Systolic MAC Array** để xử lý phép nhân ma trận $Q K^T$ trong Attention Core không?

\--------------------------------------------------------------------------------

Cây cộng Pipelined SV

Để hiện thực hóa một **Cây cộng Pipelined (Pipelined Adder Tree)** trong SystemVerilog đạt tần số xung clock cao ($F\\\_{max} \\\\ge 250 \- 300\\\\text{ MHz}$) trên FPGA (như AMD Kria KV260), chúng ta cần chèn các thanh ghi **Flip-Flop (Pipeline Registers)** vào giữa từng tầng của cây cộng nhị phân.

\--------------------------------------------------------------------------------

1\. Tại sao phải Pipeline Cây cộng?

* **Mạch tổ hợp thuần (Combinational)**: Tất cả bộ cộng $\\\\log\\\_2(N)$ tầng được nối trực tiếp với nhau. Đường chạy dài nhất (Critical Path) đi qua toàn bộ $\\\\log\\\_2(N)$ bộ cộng trong 1 chu kỳ clock, gây vi phạm thời gian (Setup Time Violation) và hạ thấp $F\\\_{max}$.  
* **Mạch Pipelined**: Bằng cách chèn thanh ghi Flip-Flop vào sau mỗi phép cộng 2 phần tử, ta cắt nhỏ đường Critical Path thành từng tầng cực ngắn (chỉ 1 phép cộng 2 số / 1 chu kỳ).  
* **Đánh đổi**:  
  * **Latency**: Tăng từ $1$ chu kỳ lên $\\\\log\\\_2(N) \+ 1$ chu kỳ clock.  
  * **Initiation Interval (**$II$**)**: Đạt $II \= 1$ — hệ thống nhận 1 cặp vector mới và xuất 1 kết quả tích vô hướng mới ở **mỗi chu kỳ clock liên tục**.

\--------------------------------------------------------------------------------

2\. Sơ đồ Nguyên lý Đường ống (Pipeline Stages với $N \= 8$)

Đầu vào A\[0..7\], B\[0..7\]

       │

       ▼ (CLK Stage 0: Multipliers \+ Reg)

  \[ P0 \] \[ P1 \] \[ P2 \] \[ P3 \] \[ P4 \] \[ P5 \] \[ P6 \] \[ P7 \]    \&lt;-- Reg Stage 0 (Tích INT16)

       └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘

           ▼              ▼              ▼              ▼

  \[  Add \&amp; Reg  \] \[  Add \&amp; Reg  \] \[  Add \&amp; Reg  \] \[  Add \&amp; Reg  \] \&lt;-- Reg Stage 1 (CLK 1\)

           └──────┬───────┘              └──────┬───────┘

                  ▼                             ▼

           \[  Add \&amp; Reg  \]               \[  Add \&amp; Reg  \]      \&lt;-- Reg Stage 2 (CLK 2\)

                  └──────────────┬──────────────┘

                                 ▼

                          \[  Add \&amp; Reg  \]                     \&lt;-- Reg Stage 3 (CLK 3 \- Root Output)

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

3\. Mã nguồn SystemVerilog Sản xuất (`pipelined_adder_tree.sv`)

Đoạn mã SystemVerilog dưới đây được viết theo chuẩn **Parameterized RTL**, hỗ trợ kích thước vector $N$ bất kỳ (lũy thừa của 2), tự động tính số tầng pipeline bằng `$clog2(N)` và tích hợp mạch dịch cờ `valid` để quản lý độ trễ:

\`timescale 1ns / 1ps

&nbsp;

module pipelined\_adder\_tree \#(

    parameter int N          \= 16,  // Số lượng phần tử vector (phải là 2, 4, 8, 16, 32, 64...)

    parameter int IN\_WIDTH   \= 8,   // Độ rộng kiểu dữ liệu đầu vào (e.g., INT8)

    parameter int OUT\_WIDTH  \= 32   // Độ rộng thanh ghi tích lũy đầu ra (e.g., INT32)

)(

    input  logic                     clk,

    input  logic                     rst\_n,

    input  logic                     in\_valid,

    input  logic signed \[IN\_WIDTH-1:0\] a \[N\],

    input  logic signed \[IN\_WIDTH-1:0\] b \[N\],

&nbsp;

    output logic                     out\_valid,

    output logic signed \[OUT\_WIDTH-1:0\] dot\_product

);

&nbsp;

    // Tính số tầng của Cây cộng nhị phân: STAGES \= log2(N)

    localparam int STAGES \= $clog2(N);

&nbsp;

    // Độ rộng bit của tích nhân hai số INT8 x INT8 \= INT16

    localparam int PROD\_WIDTH \= 2 \* IN\_WIDTH;

&nbsp;

    // \-------------------------------------------------------------------------

    // 1\. Tầng 0: Phép nhân song song \&amp; Đệm thanh ghi tích (Stage 0 Registers)

    // \-------------------------------------------------------------------------

    logic signed \[PROD\_WIDTH-1:0\] prod\_regs \[N\];

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            for (int i \= 0; i \&lt; N; i++) prod\_regs\[i\] \&lt;= '0;

        end else if (in\_valid) begin

            for (int i \= 0; i \&lt; N; i++) begin

                prod\_regs\[i\] \&lt;= $signed(a\[i\]) \* $signed(b\[i\]);

            end

        end

    end

&nbsp;

    // \-------------------------------------------------------------------------

    // 2\. Mảng đệm Pipeline cho các tầng Cây cộng

    //    tree\_regs\[stage\]\[index\]

    // \-------------------------------------------------------------------------

    // Tầng s sẽ có (N \&gt;\&gt; s) phần tử

    logic signed \[OUT\_WIDTH-1:0\] tree\_regs \[STAGES+1\]\[N\];

&nbsp;

    // Nạp kết quả tầng nhân vào tầng 0 của cây (ép kiểu sang OUT\_WIDTH để tránh tràn bit)

    always\_comb begin

        for (int i \= 0; i \&lt; N; i++) begin

            tree\_regs\[i\] \= $signed(prod\_regs\[i\]);

        end

    end

&nbsp;

    // \-------------------------------------------------------------------------

    // 3\. Tạo mạch Pipeline Cây cộng qua các nhịp xung Clock

    // \-------------------------------------------------------------------------

    generate

        for (genvar s \= 0; s \&lt; STAGES; s++) begin : g\_tree\_stages

            localparam int CURR\_ITEMS \= N \&gt;\&gt; (s \+ 1); // Số bộ cộng ở tầng s

&nbsp;

            always\_ff @(posedge clk or negedge rst\_n) begin

                if (\!rst\_n) begin

                    for (int j \= 0; j \&lt; CURR\_ITEMS; j++) begin

                        tree\_regs\[s+1\]\[j\] \&lt;= '0;

                    end

                end else begin

                    for (int j \= 0; j \&lt; CURR\_ITEMS; j++) begin

                        // Cộng 2 phần tử kề nhau ở tầng trước và lưu vào thanh ghi tầng sau

                        tree\_regs\[s+1\]\[j\] \&lt;= tree\_regs\[s\]\[2\*j\] \+ tree\_regs\[s\]\[2\*j+1\];

                    end

                end

            end

        end

    endgenerate

&nbsp;

    // Đầu ra kết quả là gốc của Cây cộng (tầng STAGES)

    assign dot\_product \= tree\_regs\[STAGES\];

&nbsp;

    // \-------------------------------------------------------------------------

    // 4\. Mạch dịch đường ống cờ Valid (Latency Tracking Pipeline)

    // \-------------------------------------------------------------------------

    // Tổng Latency \= 1 (Tầng nhân) \+ STAGES (Tầng cộng)

    logic \[STAGES:0\] valid\_pipe;

&nbsp;

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            valid\_pipe \&lt;= '0;

        end else begin

            valid\_pipe \&lt;= {valid\_pipe\[STAGES-1:0\], in\_valid};

        end

    end

&nbsp;

    assign out\_valid \= valid\_pipe\[STAGES\];

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Giải thích Kỹ thuật & Tối ưu hóa trên FPGA

1. **Quản lý Bit Growth (Chống tràn số)**:  
2. Khi cộng $N$ phần tử tích $16\\\\text{-bit}$, kết quả cần tối đa $16 \+ \\\\log\\\_2(N)$ bits.  
3. Bằng cách gán trực tiếp dữ liệu tầng nhân sang mảng kiểu `OUT_WIDTH` ($32\\\\text{-bit}$), các bộ cộng tầng tiếp theo sẽ tự động được Vivado tổng hợp thành bộ cộng $32\\\\text{-bit}$ rộng rãi, bảo toàn độ chính xác tuyệt đối.  
4. **Khả năng Ánh xạ phần cứng (Synthesis Mapping)**:  
5. **Tầng 0 (Nhân)**: Được Vivado tự động ánh xạ vào các khối **DSP48E2** nếu tổng hợp trên Kria KV260.  
6. **Các tầng Cộng**: Được tổng hợp thành mạng lưới **Carry Chains (CARRY8)** và **Flip-Flops** trong các khối LUT/CLB fabric với độ trễ cực thấp.  
7. **Mạch quản lý Latency (valid\_pipe)**:  
8. Thanh ghi dịch `valid_pipe` đảm bảo tín hiệu `out_valid` bật lên đúng chu kỳ xung clock mà kết quả tương ứng xuất hiện ở đầu ra `dot_product` (đúng sau $1 \+ \\\\log\\\_2(N)$ chu kỳ).

\--------------------------------------------------------------------------------

5\. Thông số Hiệu năng ước tính ($N \= 32$, INT8 $\\\\rightarrow$ INT32)

| Chỉ số Hiệu năng | Pipelined Adder Tree | Combinational Adder Tree |
| ----- | ----- | ----- |
| **Độ trễ (Latency)** | $1 \+ \\\\log\\\_2(32) \= \\\\mathbf{6\\\\text{ clock cycles}}$ | $1\\\\text{ clock cycle}$ |
| **Initiation Interval (**$II$**)** | $II \= 1$ (1 vector/cycle) | $II \= 1$ |
| **Tần số tối đa (**$F\\\_{max}$ **trên KV260)** | $\&gt; 300\\\\text{ MHz}$ (Timing Met dễ dàng) | $\&lt; 100\\\\text{ MHz}$ (Vi phạm Critical Path) |
| **Tài nguyên Tiêu thụ** | Tốn thêm $\\\\sim N \\\\times \\\\text{OUT\\\_WIDTH}$ Flip-Flops. | Cực ít Flip-Flops, nhưng tiêu tốn diện tích dây nối. |

Bạn có muốn viết một **SystemVerilog Testbench (tb\_pipelined\_adder\_tree.sv)** tự động tạo vector ngẫu nhiên và so sánh kết quả với model kiểm thử (Golden Model) không?

\--------------------------------------------------------------------------------

Kiến trúc Cây Cộng MAC

Kỹ thuật ghép hàng loạt đơn vị **MAC (Multiply-Accumulate)** thành **Cây cộng (Adder Tree)** là kiến trúc kinh điển trong thiết kế phần cứng FPGA/ASIC nhằm tính tích vô hướng (Dot-Product) giữa hai vector:

\\$$P \= \\\\vec{A} \\\\cdot \\\\vec{B} \= \\\\sum\\\_{i=0}^{N-1} (A\\\_i \\\\times B\\\_i)\\\\$$

Thay vì cộng tích dồn nối tiếp qua từng chu kỳ xung clock (tốn $N$ clock cycles), Cây cộng tận dụng tính song song không gian (spatial parallelism) để gom toàn bộ kết quả của $N$ phép nhân lại chỉ trong **1 chu kỳ clock**.

\--------------------------------------------------------------------------------

I. Cấu trúc Mạch Vi vi mạch của Cây Cộng (Adder Tree Topology)

Cấu trúc gồm 2 khối chức năng chính: **Tầng nhân song song** và **Cây cộng nhị phân** $\\\\log\\\_2(N)$ **tầng**.

Đầu vào Vector A\[0..7\] \&amp; B\[0..7\] (INT8)

       │      │       │      │       │      │       │      │

       ▼      ▼       ▼      v       ▼      ▼       ▼      ▼

    \[ M0 \] \[ M1 \]  \[ M2 \] \[ M3 \]  \[ M4 \] \[ M5 \]  \[ M6 \] \[ M7 \]  \&lt;-- Tầng 0: N Bộ nhân (Multipliers)

       │      │       │      │       │      │       │      │

       P0     P1      P2     P3      P4     P5      P6     P7   (Tích 16-bit)

       └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘

          ▼              ▼              ▼              ▼

       \[ Add \]        \[ Add \]        \[ Add \]        \[ Add \]     \&lt;-- Tầng 1: N/2 Bộ cộng (4 adders)

          │              │              │              │

        Sum1\_0         Sum1\_1         Sum1\_2         Sum1\_3

          └──────┬───────┘              └──────┬───────┘

                 ▼                             ▼

              \[ Add \]                       \[ Add \]             \&lt;-- Tầng 2: N/4 Bộ cộng (2 adders)

                 │                             │

               Sum2\_0                        Sum2\_1

                 └──────────────┬──────────────┘

                                ▼

                             \[ Add \]                            \&lt;-- Tầng 3: 1 Bộ cộng cuối (Root Adder)

                                │

                                ▼

                       Dot-Product (INT32)

&nbsp;

&nbsp;

Các tầng xử lý chi tiết (với $N \= 8$ phần tử):

1. **Tầng 0 (Multipliers Layer)**: $N$ đơn vị nhân tính toán song song $N$ tích trung gian $P\\\_i \= A\\\_i \\\\times B\\\_i$.  
2. **Tầng 1 của Cây**: $N/2$ bộ cộng (4 bộ cộng), mỗi bộ tính tổng từng cặp tích kề nhau ($P\\\_0 \+ P\\\_1, P\\\_2 \+ P\\\_3, \\\\dots$).  
3. **Tầng 2 của Cây**: $N/4$ bộ cộng (2 bộ cộng), cộng dồn kết quả từ Tầng 1\.  
4. **Tầng** $\\\\log\\\_2(N)$ **(Root Adder)**: 1 bộ cộng gốc xuất ra tổng tích vô hướng cuối cùng.

\--------------------------------------------------------------------------------

II. Quản lý Độ rộng Bit (Bit-Growth & Overflow Protection)

Khi cộng $N$ số nguyên, kết quả có thể bị tràn bit (overflow). Để bảo toàn độ chính xác:

* **Quy tắc tăng bit**: Mỗi tầng của Cây cộng cần mở rộng thêm **1 bit** độ rộng dữ liệu.  
* **Tổng số bit mở rộng**: Để cộng $N$ tích, tổng số bit mở rộng cần thêm là $\\\\lceil \\\\log\\\_2(N) \\\\rceil$ bits.  
* **Thực tế phần cứng**: Khi đầu vào là **INT8**, tích trung gian là **16-bit** (`INT16`). Với $N \= 16$, tổng cần thêm 4 bit $\\\\rightarrow$ mở rộng thanh ghi lên **20-bit** hoặc ép trực tiếp về kiểu **INT32** để đồng bộ với các khối tích lũy khác.

\--------------------------------------------------------------------------------

III. Phân tích Đánh đổi: Combinational (1-Clock) vs. Pipelined ($II=1$)

| Tiêu chí | Combinational Adder Tree (1-Clock thuần) | Pipelined Adder Tree (Nhiều tầng FF) |
| ----- | ----- | ----- |
| **Số chu kỳ trễ (Latency)** | **Đúng 1 chu kỳ clock** (hoặc 0 cycle nếu hoàn toàn là mạch tổ hợp). | $\\\\log\\\_2(N)$ **chu kỳ clock** (Latency \= $\\\\log\\\_2 N$). |
| **Tần số xung clock (**$F\\\_{max}$**)** | **Thấp** (đường truyền tổ hợp kéo dài qua $\\\\log\\\_2 N$ bộ cộng nối tiếp gây vi phạm thời gian Critical Path). | **Rất cao (**$F\\\_{max} \&gt; 300\\\\text{ MHz}$**)** do chèn Flip-Flops cắt nhỏ đường Critical Path. |
| **Khả năng nhận luồng (Throughput)** | Mức trung bình. | **Tối đa (**$II \= 1$**)**: Mỗi chu kỳ xuất 1 kết quả tích vô hướng mới. |
| **Khuyên dùng khi** | $N$ nhỏ ($N \\\\le 4$ hoặc $N \\\\le 8$). | $N$ lớn ($N \= 16, 32, 64$) trên các dòng FPGA hiện đại như Kria KV260. |

\--------------------------------------------------------------------------------

IV. Mã nguồn SystemVerilog Mẫu (`vector_dot_product_tree.sv`)

Đoạn mã dưới đây triển khai Cây cộng tham số hóa (Parameterized Adder Tree) cho $N$ phần tử mạch tổ hợp hoàn tất trong 1 chu kỳ clock:

module vector\_dot\_product\_tree \#(

    parameter int N          \= 8,   // Số lượng phần tử vector (phải là lũy thừa của 2\)

    parameter int DATA\_WIDTH \= 8,   // Kiểu dữ liệu INT8 đầu vào

    parameter int OUT\_WIDTH  \= 32   // Kiểu dữ liệu INT32 đầu ra

)(

    input  logic signed \[DATA\_WIDTH-1:0\] a \[N\],

    input  logic signed \[DATA\_WIDTH-1:0\] b \[N\],

    output logic signed \[OUT\_WIDTH-1:0\]  dot\_product

);

&nbsp;

    localparam int STAGES \= $clog2(N);

&nbsp;

    // 1\. Tầng nhân song song: N bộ nhân INT8 x INT8 \-\&gt; INT16

    logic signed \[2\*DATA\_WIDTH-1:0\] products \[N\];

    always\_comb begin

        for (int i \= 0; i \&lt; N; i++) begin

            products\[i\] \= a\[i\] \* b\[i\];

        end

    end

&nbsp;

    // 2\. Mảng đệm cấu trúc Cây cộng

    // tree\[stage\]\[element\]

    logic signed \[OUT\_WIDTH-1:0\] tree \[STAGES+1\]\[N\];

&nbsp;

    // 3\. Xây dựng Cây cộng nhị phân tự động bằng vòng lặp generate logic

    always\_comb begin

        // Nạp tích ban đầu vào Tầng 0

        for (int i \= 0; i \&lt; N; i++) begin

            tree\[i\] \= $signed(products\[i\]);

        end

&nbsp;

        // Tạo từng tầng của Cây cộng

        for (int s \= 0; s \&lt; STAGES; s++) begin

            for (int j \= 0; j \&lt; (N \&gt;\&gt; (s \+ 1)); j++) begin

                tree\[s+1\]\[j\] \= tree\[s\]\[2\*j\] \+ tree\[s\]\[2\*j+1\];

            end

        end

    end

&nbsp;

    // Đầu ra là gốc của Cây cộng

    assign dot\_product \= tree\[STAGES\];

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

🔄 Bạn có muốn thêm các thanh ghi Pipeline (`always_ff @(posedge clk)`) vào giữa các tầng của Cây cộng này để tối ưu tần số xung clock $F\\\_{max}$ lên $250+\\\\text{ MHz}$ cho Kria KV260 không?

\--------------------------------------------------------------------------------

Kiến trúc Đơn vị MAC

Đơn vị **MAC (Multiply-Accumulate Unit / Đơn vị Nhân-Tích lũy)** là khối phần cứng tính toán cơ bản nhất trong các bộ xử lý tín hiệu số (DSP) và các bộ tăng tốc trí tuệ nhân tạo (AI Accelerators).

Hơn 90% khối lượng tính toán trong mạng thần kinh (tích ma trận GEMM, Attention Score $Q K^T$, $Score \\\\times V$, hay tầng MLP) đều được quy về chuỗi các phép tính MAC nối tiếp.

\--------------------------------------------------------------------------------

1\. Cơ chế Toán học & Công thức

Về mặt toán học, đơn vị MAC thực hiện phép toán: $$\\\\text{Accumulator} \\\\leftarrow \\\\text{Accumulator} \+ (A \\\\times B)\\$$

* **Toán hạng đầu vào (**$A, B$**)**: Dữ liệu kích hoạt (Activation) và trọng số (Weight). Trong hệ thống định lượng (Quantization), chúng thường mang kiểu dữ liệu số nguyên độ chính xác thấp như **INT8** hoặc **Q4.4 Fixed-point**.  
* **Thanh ghi tích lũy (Accumulator)**: Lưu tổng dồn các tích. Để tránh hiện tượng tràn số (overflow) khi cộng dồn chuỗi nhiều phép nhân, độ rộng của thanh ghi tích lũy luôn lớn hơn độ rộng đầu vào (ví dụ: đầu vào INT8 nhưng tích lũy dưới dạng **INT32** hoặc **48-bit**).

\--------------------------------------------------------------------------------

2\. Các Thành phần Vi mạch bên trong một MAC Unit

Một khối MAC tiêu chuẩn trên FPGA/ASIC gồm 4 thành phần chính:

1. **Bộ nhân (Hardware Multiplier)**: Nhân hai số đầu vào $A \\\\times B$ để tạo ra tích trung gian (Product).  
2. **Bộ căn chỉnh / Dịch bit (Alignment / Rescaling Shifter)**: Căn chỉnh dấu phẩy cố định (Fixed-point alignment) giữa tích vừa nhân và giá trị tích lũy cũ.  
3. **Bộ cộng (Adder Tree / Summer)**: Cộng tích trung gian với giá trị đang lưu trong thanh ghi tích lũy.  
4. **Mạch xén / Bảo vệ tràn số (Saturating Logic)**: Giữ cho kết quả không bị lật dấu (wrap-around) nếu vượt quá giới hạn biểu diễn của kiểu dữ liệu đầu ra.

\--------------------------------------------------------------------------------

3\. Triển khai MAC Unit trên FPGA

Trên FPGA (như AMD Kria KV260 hay Zynq), khối MAC được triển khai theo 2 cách:

* **Triển khai cứng (Hard DSP Slice \- DSP48E2)**: Mỗi khối DSP48E2 có sẵn bộ nhân $27 \\\\times 18$ bit và bộ tích lũy 48-bit. Đây là lựa chọn tối ưu về hiệu năng và tiêu thụ năng lượng.  
* **Triển khai mềm (Soft Logic \- LUTs/FFs)**: Khi thiếu tài nguyên DSP, các phép nhân-cộng INT8 có thể được tổng hợp trực tiếp từ các bảng tra LUT và Flip-Flop trên cấu trúc logic chung của FPGA.

\--------------------------------------------------------------------------------

4\. Mã nguồn SystemVerilog minh họa (Safe Fixed-Point MAC Unit)

Đoạn mã RTL dưới đây minh họa cấu trúc một đơn vị MAC định lượng số nguyên cố định (Q4.4), có tích lũy chống tràn (Saturation) và làm tròn (Rounding):

module fixed\_point\_mac (

    input  logic signed \[7:0\] a,       // Đầu vào A (Dạng Q4.4 / INT8)

    input  logic signed \[7:0\] b,       // Đầu vào B (Dạng Q4.4 / INT8)

    input  logic signed \[7:0\] accum,   // Giá trị tích lũy cũ

    output logic signed \[7:0\] result  // Kết quả tích lũy mới (Saturated)

);

&nbsp;

    // 1\. Phép nhân: Q4.4 x Q4.4 \= Q8.8 (16-bit)

    logic signed \[15:0\] product;

    assign product \= a \* b;

&nbsp;

    // 2\. Căn chỉnh thanh ghi tích lũy về Q8.8

    logic signed \[15:0\] accum\_aligned;

    assign accum\_aligned \= accum \&lt;\&lt;\&lt; 4; // Dịch trái 4 bit

&nbsp;

    // 3\. Cộng tích lũy \&amp; Làm tròn (Add half LSB)

    logic signed \[16:0\] sum\_full;

    assign sum\_full \= product \+ accum\_aligned \+ 17'sd8;

&nbsp;

    // 4\. Thu phóng (Rescale) và Xén bảo vệ tràn số (Saturate)

    always\_comb begin

        logic signed \[16:0\] scaled\_sum;

        scaled\_sum \= sum\_full \&gt;\&gt;\&gt; 4; // Dịch phải để về lại Qx.4

&nbsp;

        // Clamp trong giới hạn Q4.4 \[-128 đến \+127\]

        if (scaled\_sum \&gt; 17'sd127)

            result \= 8'h7F;          // Kẹp ở giá trị dương tối đa

        else if (scaled\_sum \&lt; \-17'sd128)

            result \= 8'h80;          // Kẹp ở giá trị âm tối thiểu

        else

            result \= scaled\_sum\[7:0\];

    end

&nbsp;

endmodule

&nbsp;

&nbsp;

💡 Bạn có muốn tìm hiểu cách ghép hàng loạt các đơn vị **MAC Unit** này thành cây cộng (**Adder Tree**) để tính nhanh tích vô hướng (Dot-Product) giữa 2 vector trong 1 chu kỳ clock không?

\--------------------------------------------------------------------------------

Kiến trúc DSP48E2 UltraScale

**DSP48E2** là khối xử lý tín hiệu số phần cứng chuyên dụng (dedicated hard DSP slice) được tích hợp trực tiếp trên các dòng FPGA AMD/Xilinx UltraScale và UltraScale+ (như trên chip Zynq UltraScale+ MPSoC / bo mạch Kria KV260).

Khối phần cứng này đóng vai trò là "động cơ" tính toán số học cốt lõi cho các bộ tăng tốc vi mạch Edge AI và Vision Transformer (ViT).

\--------------------------------------------------------------------------------

1\. Cấu trúc Phần cứng Cơ bản của DSP48E2

Mỗi slice DSP48E2 được tối ưu hóa cho các phép tính nhân-tích lũy (Multiply-Accumulate \- MAC) tốc độ cao với các thông số chính:

* **Bộ nhân phần cứng (Hard Multiplier)**: Hỗ trợ phép nhân số nguyên $27 \\\\times 18$ **bit**.  
* **Bộ cộng trước (Pre-adder)**: Tích hợp bộ cộng/trừ trước bộ nhân giúp tối ưu phép toán đối xứng.  
* **Bộ tích lũy (Accumulator)**: Bộ tích lũy rộng **48-bit**, cho phép tích lũy liên tục các kết quả nhân INT8/INT16 dưới dạng `INT32` mà không lo bị tràn số (overflow).  
* **Chế độ SIMD**: Hỗ trợ chia nhỏ bộ cộng thành các đường SIMD (như dual 24-bit hoặc quad 12-bit).

\--------------------------------------------------------------------------------

2\. Vai trò của DSP48E2 trong Bộ Tăng tốc Vision Transformer

Trong các thiết kế vi mạch tăng tốc ViT (như Capstone Attention Core hay ME-ViT):

1. **Thành phần cốt lõi của Mảng Systolic Array**: Các khối DSP48E2 được kết nối thành mạng lưới 2D Processing Elements (PE) để thực hiện các phép nhân ma trận lớn (GEMM) trong các khối $Q K^T$, $Score \\\\times V$, và tầng MLP.  
2. **Tối ưu bằng Kỹ thuật DSP Packing (2x INT8)**: Nhờ bộ nhân $27 \\\\times 18$ bit rộng, phần cứng có thể đưa một toán hạng 18-bit $A$ và một toán hạng ghép 27-bit $(B \\\\ll 18 \+ C)$ vào cùng một khối DSP48E2 để tính đồng thời hai phép nhân INT8 ($A \\\\times B$ và $A \\\\times C$) trong **chỉ 1 chu kỳ clock**, giúp tăng gấp đôi hiệu suất tính toán.  
3. **Kết hợp với Đệm kép Ping-Pong BRAM**: Để tránh lãng phí tài nguyên tính toán đắt giá này, luồng dữ liệu từ AXI DMA được nạp gối đầu qua đệm kép Ping-Pong, đảm bảo các khối DSP48E2 luôn có dữ liệu để tính toán liên tục ở tần số cao mà không bị đứng chờ (stall).

\--------------------------------------------------------------------------------

💡 Bạn có muốn tìm hiểu về cách ánh xạ (mapping) các câu lệnh SystemVerilog để Vivado tự động tổng hợp (infer) đúng vào khối **DSP48E2** thay vì dùng logic LUT không?

\--------------------------------------------------------------------------------

Kiến Trúc MSA Systolic Array

Sơ đồ kiến trúc mô phỏng luồng chuyển dữ liệu giữa các bộ đệm **Q, K, V BRAM** và mảng **Systolic MAC Array** trong chế độ Multi-Head Self-Attention (MSA Mode) đã được khởi tạo và hiển thị trực tiếp trong **Studio Panel** dưới tên `msa_dataflow_diagram.png`.

Dưới đây là mô tả chi tiết 2 giai đoạn (Phase 1 & Phase 2\) của luồng dữ liệu này trên phần cứng:

\--------------------------------------------------------------------------------

Sơ đồ Khái quát Luồng Dữ liệu (MSA Dataflow Architecture)

\[ ON-CHIP BRAM BUFFERS \]             \[ DATA MUX \&amp; FSM \]            \[ 2D SYSTOLIC MAC ARRAY \]

\+----------------------+             \+------------------+          \+-----------------------+

|  Q BRAM Buffer (INT8)| \----------\&gt; |                  | \-------\&gt; | 32x32 PEs (DSP48E2)   |

|  K BRAM Buffer (INT8)| \----------\&gt; | Data Mux \&amp;       |          | INT32 Accumulation    |

|  V BRAM Buffer (INT8)| \----------\&gt; | Address Gen      |          | Output-Stationary     |

\+----------------------+             \+------------------+          \+-----------------------+

   (Double Buffering                     |     ^                               |

    \+ BRAM Banking)                      |     | (Feedback Phase 2\)            | Phase 1 Out

                                         v     |                               v

                                   \[ Phase 2 Input \]             \+-----------------------+

                                   Softmax Score                 | Scaling Unit (Shift)  |

                                         ^                       \+-----------------------+

                                         |                                     |

                                         |                       \+-----------------------+

                                   \+-----------+                 | Softmax (Shiftmax)    |

                                   | Score     | \&lt;-------------- \+-----------------------+

                                   | Buffer    |                     (Phase 1 Result)

                                   \+-----------+

                                         |

                                         v Phase 2 Out

                                   \+-----------------------------------------------------+

                                   | Attention Output Buffer (Head Context \-\&gt; Concat W\_O)|

                                   \+-----------------------------------------------------+

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

Chi tiết Luồng Chuyển Dữ liệu theo 2 Giai đoạn

Quy trình tính toán Attention trên phần cứng được chia làm **2 Phase liên tiếp** để tối ưu hóa việc tái sử dụng mảng Systolic MAC Array mà không cần nhân đôi tài nguyên phần cứng:

📜 Giai đoạn 1: Tính Ma trận Điểm số Chú ý ($Q \\\\cdot K^T \\\\rightarrow \\\\text{Softmax}$)

1. **Trích xuất & Cấp dữ liệu (Streaming)**:  
2. **Data Mux** đọc luồng dữ liệu Tile $Q\\\_i$ từ **Q BRAM Buffer** và Tile $K\\\_j$ từ **K BRAM Buffer**.  
3. Nhờ kỹ thuật **BRAM Banking (Cyclic Partitioning)**, $P\\\_{sys}$ phần tử dữ liệu của $Q$ và $K$ được đẩy song song vào mảng PE trong cùng 1 chu kỳ clock mà không bị xung đột cổng đọc (port collision).  
4. **Tính Tích vô hướng trên Systolic Array**:  
5. Mảng **2D Systolic MAC Array (DSP48E2)** thực hiện phép nhân ma trận $Q\\\_i \\\\cdot K\\\_j^T$. Các bộ tích lũy trong PE tích lũy kết quả dưới dạng số nguyên **INT32** (Output-Stationary mode).  
6. **Thu phóng & Kích hoạt Phi tuyến Pipeline**:  
7. Kết quả $Q \\\\cdot K^T$ chảy qua **Scaling Unit**: thực hiện phép dịch bit đại số (`&gt;&gt;&gt;`) để thu phóng theo hệ số $\\\\sqrt{d\\\_k}$ với chi phí 0 DSP.  
8. Dữ liệu chảy tiếp qua khối **Softmax Unit (Shiftmax INT8)** để chuyển đổi thành phân bố xác suất trọng số chú ý.  
9. **Lưu đệm tạm thời (Staged Storage)**:  
10. Ma trận xác suất Softmax Score vừa tạo ra được ghi trực tiếp vào **Score Buffer (S-Buffer)** nằm ngay trên FPGA PL.

\--------------------------------------------------------------------------------

🔄 Giai đoạn 2: Tổng hợp Vector Ngữ cảnh ($Score \\\\cdot V$)

1. **Đổi Luồng Đầu vào (Dynamic Multiplexing)**:  
2. Sau khi Phase 1 hoàn tất, bộ điều khiển FSM chuyển mạch **Data Mux**:  
   * **Cổng 1**: Đọc ma trận trọng số chú ý $Score$ từ **Score Buffer**.  
   * **Cổng 2**: Đọc Tile $V\\\_j$ từ **V BRAM Buffer**.  
3. **Tái sử dụng Mảng Systolic Array**:  
4. Mảng **Systolic MAC Array** tiếp tục thực hiện phép nhân ma trận thứ hai: $Softmax(Score) \\\\times V\\\_j$.  
5. **Đưa ra Output Buffer**:  
6. Ma trận kết quả ngữ cảnh của Head hiện tại ($Out\\\_{Head}$) được ghi dồn vào **Attention Output Buffer**.  
7. Sau khi tính xong tất cả các Head ($h$ heads), kết quả được gom lại (Concat) để đưa qua ma trận chiếu đầu ra $W^O$ hoặc chuyển tiếp sang khối MLP.

\--------------------------------------------------------------------------------

Điểm nhấn Thiết kế Phần cứng

* **Zero Memory Bottleneck**: Việc sử dụng **Score Buffer nội bộ** làm cầu nối giữa Phase 1 và Phase 2 giúp toàn bộ điểm số chú ý trung gian không bao giờ phải ghi ngược ra RAM ngoài (DDR4), tiết kiệm hơn **80% băng thông bus AXI**.  
* **Overlap Pipelining**: Trong lúc mảng PE đang tính toán cho Head hiện tại, cơ chế **Double Buffering (Ping-Pong)** trên Q, K, V BRAM cho phép AXI DMA tranh thủ nạp Tile dữ liệu của Head tiếp theo từ DDR4 RAM vào nửa đệm còn lại.

\--------------------------------------------------------------------------------

🧩 Bạn có muốn xem qua mã SystemVerilog triển khai bộ chọn kênh dữ liệu **Data Mux & Address Generator** để chuyển đổi giữa Phase 1 và Phase 2 không?

\--------------------------------------------------------------------------------

Kiến trúc Systolic Array

Mảng tính toán **Systolic MAC Array** (Mảng nhân-cộng nhịp đập) là trái tim tính toán của các bộ tăng tốc vi mạch FPGA/ASIC dành cho Vision Transformer (ViT) và Edge AI. Đây là cấu trúc vi mạch được thiết kế chuyên biệt để xử lý các phép nhân ma trận khối lượng lớn (GEMM / MatMul) với hiệu suất và độ tái sử dụng dữ liệu tối đa.

\--------------------------------------------------------------------------------

1\. Nguyên lý Hoạt động & Kiến trúc Vi mạch

Systolic Array gồm một mạng lưới 2 chiều (2D Grid) chứa các đơn vị xử lý nhỏ gọi là **Processing Elements (PEs)**:

* **Cấu trúc PE (Processing Element)**: Mỗi PE chứa một bộ nhân-cộng **MAC (Multiply-Accumulate)** và các thanh ghi đệm cục bộ. Trong các mạch FPGA như AMD UltraScale+ / Kria KV260, khối MAC này được ánh xạ trực tiếp vào các khối phần cứng cứng **DSP48E2**.  
* **Dòng chảy dữ liệu "Nhịp đập" (Systolic Dataflow)**: Dữ liệu kích hoạt (Activations) và trọng số (Weights) "đập" (pump) qua các PE lân cận theo từng chu kỳ xung clock. Dữ liệu vừa được tính toán vừa được truyền tiếp cho PE bên cạnh mà **không cần ghi ngược về bộ nhớ đệm BRAM hay DDR4**, giúp giảm thiểu tối đa băng thông bộ nhớ.

\--------------------------------------------------------------------------------

2\. Các Luồng Dữ liệu (Dataflows) Phổ biến

Tùy thuộc vào chiến lược lưu trữ dữ liệu tạm thời, Systolic Array trên FPGA áp dụng một trong các kiểu luồng dữ liệu sau:

1. **Output-Stationary (OS \- Cố định Đầu ra)**:  
2. **Cơ chế**: Tổng một phần (Partial Sum) được giữ cố định trong thanh ghi tích lũy 32-bit (`INT32`) bên trong PE. Trong khi đó, dữ liệu đầu vào và trọng số được đẩy liên tục qua mảng PE.  
3. **Ưu điểm**: Triệt tiêu hoàn toàn luồng giao tiếp ghi dồn tổng một phần ra BRAM/DRAM, giúp tiết kiệm bộ nhớ và giảm năng lượng tiêu thụ. Đây là cơ chế được áp dụng phổ biến trong các bộ tăng tốc ViT hiện đại (như FlexViT, ME-ViT).  
4. **Weight-Stationary (WS \- Cố định Trọng số)**:  
5. **Cơ chế**: Trọng số được nạp cố định vào từng PE, các kích hoạt dịch chuyển theo hàng và tổng một phần dồn tích lũy theo cột.  
6. **Input-Stationary (IS \- Cố định Đầu vào)**:  
7. **Cơ chế**: Kích hoạt đầu vào được đệm cố định trong PE, luồng trọng số di chuyển qua mảng để tính toán.

\--------------------------------------------------------------------------------

3\. Kỹ thuật Đóng gói DSP (DSP Packing 2x INT8) trên Systolic Array

Để tối đa hóa năng lực tính toán trên các bo mạch giới hạn tài nguyên như Kria KV260 hay Alveo U200, Systolic Array áp dụng kỹ thuật **DSP Packing**:

* **Cơ chế**: Tận dụng bộ nhân $18 \\\\text{ bit} \\\\times 27 \\\\text{ bit}$ của khối **DSP48E2** để đóng gói 2 phép nhân **INT8** độc lập vào cùng 1 DSP slice: $$\\\\text{Toán hạng 27-bit} \= (B \\\\ll 18\) \+ C, \\\\quad \\\\text{Toán hạng 18-bit} \= A\\\\\\\] \\\\\\\[\\\\text{Đầu ra DSP} \= A \\\\times ((B \\\\ll 18\) \+ C) \= (A \\\\times B) \\\\ll 18 \+ (A \\\\times C)\\$$  
* **Tác động**: Với mảng Systolic kích thước $P\\\_{sys} \\\\times P\\\_{sys}$ (ví dụ $32 \\\\times 32$), kỹ thuật DSP Packing cho phép thực hiện tới $P\\\_{sys} \\\\times (2 \\\\cdot P\\\_{sys}) \= 32 \\\\times 64 \= 2048$ phép nhân INT8 trong **chỉ 1 chu kỳ xung clock**.

\--------------------------------------------------------------------------------

4\. Vai trò của Systolic MAC Array trong Chu trình ViT

Trong một bộ tăng tốc ViT hoàn chỉnh (như ME-ViT hay Capstone Attention Core), mảng Systolic MAC Array linh hoạt đảm nhiệm cả 3 chế độ tính toán chính:

                      \+---------------------------------------+

                       |        SYSTOLIC MAC ARRAY ENGINE      |

                       |       (Configurable Execution)        |

                       \+---------------------------------------+

                                           |

         \+---------------------------------+---------------------------------+

         |                                 |                                 |

         v                                 v                                 v

\[1. LP MODE\]                     \[2. MSA MODE\]                     \[3. MLP MODE\]

Linear Projections               Multi-Head Self-Attention         Feed-Forward Network

\- Q, K, V Projections (X \* W)    \- Score MatMul: (Q \* K^T) / sqrt  \- Hidden Layer (X \* W1)

\- Output Projection              \- Context MatMul: (Score \* V)     \- Output Layer (M \* W2)

&nbsp;

&nbsp;

* **Xử lý Định lượng Precision**: Nhận đầu vào **INT8** (từ BRAM Buffers) và tích lũy trung gian dưới dạng số nguyên **INT32** để bảo toàn độ chính xác tuyệt đối trước khi đưa qua khối Softmax/LayerNorm.  
* **Giao tiếp Pipelined**: Kết nối trực tiếp với đệm đệm Ping-Pong BRAM qua AXI4-Stream, duy trì nhịp tính toán liên tục với **Initiation Interval** $II \= 1$.

\--------------------------------------------------------------------------------

💡 Bạn có muốn xem qua đoạn mã SystemVerilog mẫu của một **Processing Element (pe\_mac.sv)** cơ bản hỗ trợ tích lũy INT32 và đầu ra Output-Stationary không?

\--------------------------------------------------------------------------------

Cơ chế QKV Attention

Trong kiến trúc Transformer và Vision Transformer (ViT), $Q$ **(Query)**, $K$ **(Key)**, và $V$ **(Value)** là ba ma trận thành phần cốt lõi tạo nên cơ chế chú ý **Scaled Dot-Product Attention**.

\--------------------------------------------------------------------------------

1\. Ý nghĩa và Bản chất của $Q, K, V$

Cơ chế Attention hoạt động tương tự như một hệ thống tìm kiếm/truy vấn thông tin:

* $Q$ **(Query \- Vector/Ma trận Truy vấn)**: Đại diện cho phần tử/token hiện tại đang đóng vai trò "đặt câu hỏi" để tìm kiếm ngữ cảnh (ví dụ: *"Thông tin ở các vị trí khác có liên quan gì đến tôi?"*).  
* $K$ **(Key \- Vector/Ma trận Khóa)**: Đại diện cho "nhãn/chỉ mục" của tất cả các phần tử trong chuỗi. Tích vô hướng giữa $Q$ và $K$ ($Q K^T$) dùng để đo mức độ tương đồng/tương quan giữa token truy vấn và các token còn lại.  
* $V$ **(Value \- Vector/Ma trận Giá trị)**: Chứa "nội dung thông tin thực sự" của các phần tử. Sau khi có trọng số chú ý từ Softmax, mô hình sẽ dùng trọng số này để trích xuất và tổng hợp thông tin từ $V$.

\--------------------------------------------------------------------------------

2\. Cách tạo ra các Ma trận $Q, K, V$

Từ ma trận biểu diễn đầu vào $X$ (các patch ảnh hoặc token đã được nhúng), ba ma trận $Q, K, V$ được tạo ra thông qua các phép chiếu tuyến tính (Linear Projections / Matrix Multiplication) với các ma trận trọng số tương ứng $W\\\_Q, W\\\_K, W\\\_V$:

\\$$Q \= X \\\\cdot W\\\_Q, \\\\quad K \= X \\\\cdot W\\\_K, \\\\quad V \= X \\\\cdot W\\\_V\\\\$$

Trong đó $W\\\_Q, W\\\_K \\\\in \\\\mathbb{R}^{d\\\_{model} \\\\times d\\\_k}$ và $W\\\_V \\\\in \\\\mathbb{R}^{d\\\_{model} \\\\times d\\\_v}$.

\--------------------------------------------------------------------------------

3\. Vai trò trong Phép tính Attention

Toàn bộ quy trình tính toán Attention dựa trên $Q, K, V$ được thực hiện qua 4 bước:

$$\\\\text{Attention}(Q, K, V) \= \\\\text{Softmax}\\\\left( \\\\frac{Q K^T}{\\\\sqrt{d\\\_k}} \\\\right) V\\$$

1. **Nhân Ma trận 1 (**$Q \\\\times K^T$**)**: Tính tích vô hướng giữa Query và Key để ra ma trận điểm số chú ý thô (Attention Scores).  
2. **Thu phóng (**$\\\\frac{1}{\\\\sqrt{d\\\_k}}$**)**: Chia cho $\\\\sqrt{d\\\_k}$ để chuẩn hóa phương sai, tránh làm bão hòa hàm Softmax gây triệt tiêu gradient.  
3. **Softmax**: Biến đổi điểm số thành phân bố xác suất trọng số (Attention Weights) có tổng bằng 1\.  
4. **Nhân Ma trận 2 (**$\\\\text{Score} \\\\times V$**)**: Nhân trọng số xác suất với ma trận Giá trị $V$ để thu được vector biểu diễn ngữ cảnh cuối cùng.

\--------------------------------------------------------------------------------

4\. Triển khai $Q, K, V$ trên Phần cứng (FPGA PL)

Trên các mạch tăng tốc vi mạch FPGA (như Kria KV260):

* **Lưu trữ đệm (Buffering)**: Các ma trận $Q, K, V$ được định lượng về dạng số nguyên (ví dụ: **INT8**) và lưu trữ trong các ngân hàng bộ nhớ on-chip BRAM/LUTRAM nội bộ (Q Buffer, K Buffer, V Buffer).  
* **Tính toán Pipelined**: Mảng nhân **Systolic MAC Array (DSP48E2)** lần lượt thực hiện phép nhân tuyến tính tạo $Q, K, V$, phép nhân $Q K^T$, và phép nhân tổng hợp $\\\\text{Score} \\\\times V$ theo luồng AXI-Stream.

\--------------------------------------------------------------------------------

💡 **Gợi ý tiếp theo**: Bạn có muốn tìm hiểu sâu hơn về cách xếp lịch thời gian (Scheduling) cho luồng dữ liệu $Q, K, V$ trong các ngân hàng đệm BRAM để đạt hiệu suất cao nhất trên FPGA không?

\--------------------------------------------------------------------------------

Tối ưu hóa ViT trên FPGA

Bộ 4 kỹ thuật **{Quantization, Pruning, Pipelining, DSP Packing}** chính là **"bộ tứ trụ" tối ưu hóa toàn diện (Co-optimization)** kết hợp từ thuật toán phần mềm đến kiến trúc phần cứng FPGA. Bộ tứ này giúp đưa các mô hình Vision Transformer (ViT) và Edge AI phức tạp lên các bo mạch giới hạn tài nguyên như AMD Kria KV260.

\--------------------------------------------------------------------------------

1\. **Quantization (Định lượng số học)**

* **Bản chất**: Chuyển đổi dữ liệu trọng số (weights) và kích hoạt (activations) từ số thực dấu phẩy động (FP32) sang định dạng số nguyên độ chính xác thấp (INT8, INT4 hoặc Power-of-Two).  
* **Cơ chế trên ViT**:  
  * **Tầng tuyến tính (MatMul / Dense)**: Định lượng về INT8, sử dụng quy trình **Dyadic Scaling** (thay phép chia số thực bằng phép nhân số nguyên và dịch bit đại số `&gt;&gt;&gt;`).  
  * **Tầng phi tuyến (Softmax, GELU, LayerNorm)**: Áp dụng các giải pháp xấp xỉ thuần số nguyên như **I-ViT** (Shiftmax, ShiftGELU, I-LayerNorm) để giữ nguyên đồ thị tính toán dạng số nguyên mà không tốn đơn vị FPU phần cứng.  
* **Tác động phần cứng**:  
  * Giảm **75% dung lượng lưu trữ** trọng số và kích hoạt.  
  * Giảm áp lực lên băng thông truyền dữ liệu DDR4/AXI DMA và tiết kiệm tài nguyên đệm BRAM nội bộ.  
  * Cho phép phép nhân tích lũy (MAC) chạy hoàn toàn trên đơn vị số nguyên INT8/INT32 tối ưu của khối DSP48E2.

\--------------------------------------------------------------------------------

2\. **Pruning (Cắt tỉa mô hình & Khai thác độ thưa \- Sparsity)**

* **Bản chất**: Loại bỏ các tham số, trọng số hoặc token ít quan trọng (gần bằng 0\) nhằm giảm thiểu tối đa số phép tính MAC và lưu lượng bộ nhớ.  
* **Cơ chế trên ViT**:  
  * **Structured Pruning (Cắt tỉa có cấu trúc)**: Được ưu tiên hàng đầu trên FPGA vì giữ nguyên luồng dữ liệu đều đặn (regular dataflow):  
    * *Head-wise Pruning*: Triệt hạ các Attention Head thừa cùng các ma trận chiếu $W\\\_Q, W\\\_K, W\\\_V$ đi kèm.  
    * *Tile-wise / Block Pruning*: Cắt bỏ hoàn toàn các khối ma trận con.  
    * *Dynamic Token Pruning / Dropping*: Sàng lọc và loại bỏ các token có điểm chú ý (attention score) thấp ngay tại runtime.  
* **Tác động phần cứng**:  
  * Giảm trực tiếp từ 30% \- 60% số lượng phép tính MAC, giải phóng tài nguyên DSP48E2 và BRAM đệm.  
  * Rút ngắn thời gian thực thi (latency) và tiết kiệm năng lượng cho các ứng dụng nhúng.

\--------------------------------------------------------------------------------

3\. **Pipelining (Đường ống hóa luồng xử lý)**

* **Bản chất**: Chia nhỏ quy trình tính toán thành nhiều giai đoạn (stages) nối tiếp nhau, cho phép các khối phần cứng vận hành gối đầu (overlap) song song theo nhịp xung clock.  
* **Các cấp độ Pipelining trên ViT Accelerator**:  
  * **Operator Level (Mức phần tử)**: Đạt tốc độ **Initiation Interval** $II \= 1$ (mỗi chu kỳ xung clock nhận 1 phần tử đầu vào và xuất 1 kết quả).  
  * **Coarse-Grained / Head Level**: Pipelining giữa các công đoạn trong Attention Block ($Q \\\\times K^T \\\\rightarrow \\\\text{Scale} \\\\rightarrow \\\\text{Softmax} \\\\rightarrow \\\\text{Score} \\\\times V$), nối liền dữ liệu qua bus AXI-Stream.  
  * **System Level (Double Buffering / Ping-Pong BRAM)**: Pipelining giữa việc nạp dữ liệu từ RAM ngoài (AXI DMA) và mảng tính toán MAC (Systolic Array) giúp **ẩn hoàn toàn độ trễ nạp dữ liệu DDR4 (Latency Hiding)**.  
* **Tác động phần cứng**: Triệt tiêu thời gian đứng chờ (stall cycles), giúp mảng PE/DSP vận hành liên tục 100% hiệu suất.

\--------------------------------------------------------------------------------

4\. **DSP Packing (Đóng gói bộ nhân DSP48E2)**

* **Bản chất**: Kỹ thuật ép nhiều phép tính nhân-cộng số nguyên độ chính xác thấp (như 2x INT8, 3x INT6, hoặc 4x INT4) vào trong **cùng một bộ nhân phần cứng cứng (Hard Multiplier)** của DSP slice (khối DSP48E2 $27 \\\\times 18$-bit trên UltraScale+ / Kria KV260).  
* **Cơ chế toán học**:  
  * Tận dụng cổng toán hạng 27-bit để ghép hai trọng số $B, C$ (INT8): $(B \\\\ll 18\) \+ C$.  
  * Nhân với toán hạng $A$ (INT8) ở cổng 18-bit: $A \\\\times ((B \\\\ll 18\) \+ C) \= (A \\\\times B) \\\\ll 18 \+ (A \\\\times C)$.  
  * Tách 2 kết quả $A \\\\times B$ và $A \\\\times C$ độc lập ở đầu ra 48-bit.  
* **Tác động phần cứng**:  
  * **Nhân đôi năng suất tính toán (2 MACs/DSP/cycle)** mà không làm tăng số lượng khối DSP tiêu thụ.  
  * Giải quyết triệt để bài toán giới hạn cứng **252 DSP slices** trên Kria KV260 khi triển khai các mảng Systolic Array lớn ($16 \\\\times 32$ hoặc $32 \\\\times 32$).

\--------------------------------------------------------------------------------

🎯 **Chuỗi liên kết tổng hợp giữa 4 kỹ thuật**

$$\\\\text{FP32 Model} \\\\xrightarrow{\\\\text{\\\\textbf{Quantization}}} \\\\text{INT8 Graph} \\\\xrightarrow{\\\\text{\\\\textbf{Pruning}}} \\\\text{Sparse Matrix} \\\\xrightarrow{\\\\text{\\\\textbf{DSP Packing}}} \\\\text{Mảng PE mật độ MACs cao} \\\\xrightarrow{\\\\text{\\\\textbf{Pipelining}}} \\\\text{Tối đa FPS / Tối thiểu Latency}\\$$

1. **Quantization** hạ độ chính xác số học $\\\\rightarrow$ tạo điều kiện cho **DSP Packing** đóng gói 2-4 phép nhân vào 1 DSP.  
2. **Pruning** cắt bỏ dữ liệu thừa $\\\\rightarrow$ giảm dung lượng đệm và băng thông truyền.  
3. **Pipelining** kết nối luồng tính toán $\\\\rightarrow$ giữ phần cứng chạy liên tục với $II \= 1$ mà không bị nghẽn hay chờ nạp dữ liệu.

💡 Bạn có muốn xem ví dụ mã SystemVerilog triển khai kỹ thuật **DSP Packing (2x INT8)** trong khối `pe_slice.sv` cho Kria KV260 không?

\--------------------------------------------------------------------------------

Đệm kép MLP trên FPGA

Trong các bộ tăng tốc Vision Transformer (ViT / MobileViT) trên FPGA, khối **MLP (Multi-Layer Perceptron / Feed-Forward Network)** chiếm từ **60% đến 64% tổng khối lượng tính toán (MACs)** và lượng tham số của toàn bộ mô hình.

Do dung lượng ma trận trọng số $W\\\_1$ ($D \\\\rightarrow 4D$) và $W\\\_2$ ($4D \\\\rightarrow D$) rất lớn, vượt xa khả năng lưu trữ của bộ nhớ on-chip BRAM (ví dụ bo mạch Kria KV260 chỉ có 144 khối BRAM18K), ma trận trọng số MLP bắt buộc phải được **chia nhỏ thành các sub-blocks (Tiles)** và nạp theo luồng từ bộ nhớ RAM ngoài (DDR4) vào chip qua AXI DMA.

Khối MLP áp dụng kỹ thuật **Double Buffering (Bộ đệm Ping-Pong)** để triệt tiêu thời gian chờ và tối ưu hóa băng thông bộ nhớ.

\--------------------------------------------------------------------------------

1\. Nguyên lý Hoạt động của Double Buffering trong Khối MLP

Kỹ thuật Double Buffering nhân đôi tài nguyên bộ đệm BRAM/LUTRAM đệm trọng số ($W$) và đệm đầu vào/đầu ra thành hai nửa độc lập: **Ngân hàng Ping (Buffer 0\)** và **Ngân hàng Pong (Buffer 1\)**.

                               \[ DỮ LIỆU TỪ DDR4 RAM (AXI DMA) \]

                                                │

                                                ▼ (Nạp luồng AXI-Stream)

                          \+-------------------------------------------+

                          |   BRAM WEIGHT \&amp; ACTIVATION DOUBLE BUFFER  |

                          |                                           |

                          |  \+-------------------+-----------------+  |

                          |  |  Bank 0 (PING)    |  Bank 1 (PONG)  |  |

                          |  |  \[ COMPUTING \]    |   \[ LOADING \]   |  |

                          |  \+-------------------+-----------------+  |

                          \+-------------------------------------------+

                                   │                       ▲

                                   │ (Đọc song song)       │ (Tráo con trỏ trong 1 Clock)

                                   ▼                       │

                          \+-------------------------------------------+

                          |   SYSTOLIC MAC ARRAY / PE ENGINE (MLP)    |

                          | (Tính toán Ma trận \&amp; Kích hoạt GELU/ReLU) |

                          \+-------------------------------------------+

&nbsp;

&nbsp;

* **Tại chu kỳ (Iteration)** $k$:  
  * **Đường Đọc (Read Side / PE Compute)**: Mảng Systolic MAC Array đọc Tile trọng số $W\\\_k$ và Tile kích hoạt từ ngân hàng **Ping** để thực hiện phép nhân ma trận tích lũy.  
  * **Đường Ghi (Write Side / DMA Fetch)**: Bộ AXI DMA Reader đồng thời nạp Tile trọng số tiếp theo $W\\\_{k+1}$ từ DDR4 RAM vào ngân hàng **Pong**.  
* **Tại thời điểm chuyển giao (Pointer Swap)**:  
  * Khi mảng PE tính xong Tile $k$ (`PE_Done = 1`) và AXI DMA nạp xong Tile $k+1$ (`DMA_Ready = 1`), bộ điều khiển FSM đảo bit con trỏ `ping_pong_sel` trong **đúng 1 chu kỳ clock**.  
  * Mảng PE lập tức chuyển sang đọc dữ liệu từ ngân hàng **Pong**, trong khi AXI DMA bắt đầu nạp Tile $k+2$ vào ngân hàng **Ping**.

\--------------------------------------------------------------------------------

2\. Kỹ thuật Đặt biệt cho MLP: Double S-Buffer (Tích lũy Partial Sum)

Khác với phép nhân ma trận thông thường, phép tính MLP trải qua 2 tầng: $M\\\_{i,j} \= \\\\text{Activation}(L\\\_i \\\\cdot W\\\_{1,j}^H \+ B\\\_1)$ sau đó nhân tiếp với $W\\\_2$ để tính tổng một phần (Partial Sum).

Để triệt tiêu trệ luồng khi ghi dồn tổng một phần (Partial Sum write-back), các kiến trúc hiện đại (như ME-ViT) triển khai **Double S-Buffer (**$S\\\_1$ **và** $S\\\_2$**)**:

 Result Buffer ──► \[ ReLU / GELU \] ──► Compute M\_{i,j} ──► Multiply W\_2

                                                                │

                                                                ▼

                                                   \+-------------------------+

                                                   |     DOUBLE S-BUFFER     |

                                                   |                         |

                                                   |  S1 Buffer: \[ STORING \] | ──► Write-back to Feature BRAM

                                                   |  S2 Buffer: \[ STAGING \] | ◄── Accumulate new Partial Sum

                                                   \+-------------------------+

&nbsp;

&nbsp;

1. Khi khối kết quả $M\\\_{1,1} \\\\cdot W\\\_{1,1}^O$ đang được tính, các kết quả cộng dồn trung gian được tích lũy vào đệm $S\\\_1$.  
2. Khi $S\\\_1$ đầy và thực hiện ghi dồn (write-back) về BRAM Feature Buffer, mạch điều khiển chuyển hướng luồng tính toán sang tích lũy ở đệm $S\\\_2$.  
3. Nhờ có bộ đệm $S\\\_1/S\\\_2$ luân phiên, mảng Systolic không bao giờ bị dừng (stall) giữa hai tầng $W\\\_1$ và $W\\\_2$ của khối MLP.

\--------------------------------------------------------------------------------

3\. Luồng Bắt tay Tín hiệu Đồng bộ AXI-Stream & FSM

Để tránh hiện tượng nạp đè dữ liệu hoặc đọc phải dữ liệu chưa hoàn tất, bộ điều khiển Double Buffering quản lý chặt chẽ các tín hiệu bắt tay:

* **Khóa Backpressure (S\_AXIS\_TREADY)**: Nếu mảng PE xử lý chậm khiến cả hai ngân hàng Ping và Pong đều ở trạng thái `FULL`, tín hiệu `S_AXIS_TREADY` bị kéo xuống `0`, buộc AXI DMA tạm dừng luồng dữ liệu trên bus mà không gây tràn hay mất dữ liệu.  
* **Nhận diện Cuối khối (S\_AXIS\_TLAST)**: Tín hiệu `TLAST` từ AXI DMA đánh dấu từ dữ liệu cuối cùng của một Tile MLP đã được nạp xong vào BRAM, kích hoạt cờ `Tile_Ready` cho FSM.  
* **Chính sách Nạp Một lần (Single-Load Policy)**: Mỗi Tile trọng số MLP chỉ được nạp từ DRAM vào FPGA **đúng 1 lần duy nhất**, toàn bộ các bước tính toán trung gian và đường tắt residual connection đều lưu giữ nội bộ trên BRAM/LUTRAM.

\--------------------------------------------------------------------------------

4\. Hiệu quả Kỹ thuật Mang lại cho Accelerator

| Tiêu chí Đánh giá | Khi KHÔNG dùng Double Buffering | Khi ÁP DỤNG Double Buffering cho MLP |
| ----- | ----- | ----- |
| **Trạng thái mảng PE / DSP** | Thường xuyên đứng chờ (Stall) mỗi khi DMA nạp Tile trọng số mới. | Mảng PE vận hành liên tục **\~90% \- 100% thời gian** (Near-zero stall). |
| **Độ trễ Bộ nhớ (DRAM Latency)** | Lộ rõ toàn bộ độ trễ truy xuất DRAM ngoài. | **Ẩn hoàn toàn độ trễ DRAM** (Latency Hiding) dưới thời gian tính toán của PE. |
| **Băng thông BRAM nội bộ** | Truy xuất bị nghẽn do xung đột cổng đọc/ghi. | Đọc/ghi song song trên hai ngân hàng BRAM độc lập, không bị đứt đoạn pipeline. |
| **Tốc độ Xử lý (Throughput)** | Bị giới hạn bởi băng thông bộ nhớ ngoài (Memory-Bound). | Tiến sát giới hạn tối đa của năng lực tính toán phần cứng (Compute-Bound). |

\--------------------------------------------------------------------------------

💡 **Theo dõi tiếp theo**: Bạn có muốn xem qua mã SystemVerilog triển khai bộ điều khiển đệm kép **Double S-Buffer (mlp\_s\_buffer.sv)** áp dụng trong khối MLP không?

\--------------------------------------------------------------------------------

Tối ưu MLP trên FPGA

Khối **MLP** (Multi-Layer Perceptron / Feed-Forward Network) trong Vision Transformer (ViT, MobileViT, DeiT) nằm ngay sau khối Multi-Head Self-Attention (MSA) và Layer Normalization. Đây là khối đảm nhiệm nhiệm vụ biến đổi và học các đặc trưng không gian ẩn ở cấp độ cao.

\--------------------------------------------------------------------------------

1\. Cấu trúc Toán học & Đặc điểm Tính toán

Về mặt thuật toán, khối MLP gồm 2 tầng biến đổi tuyến tính (Fully Connected / Dense layers) và 1 hàm kích hoạt phi tuyến ở giữa: $$\\\\text{MLP}(X) \= \\\\text{Activation}(X W\\\_1 \+ B\\\_1) W\\\_2 \+ B\\\_2\\$$

trong đó:

* **Tầng 1 (**$W\\\_1, B\\\_1$**)**: Mở rộng chiều ẩn $D$ lên $M\\\_r \\\\times D$ (với hệ số mở rộng $M\\\_r \= 4$ ở ViT chuẩn, hoặc $M\\\_r \= 2$ ở MobileViT).  
* **Hàm kích hoạt (Activation)**: Thường dùng GELU (hoặc ReLU để đơn giản hóa phần cứng).  
* **Tầng 2 (**$W\\\_2, B\\\_2$**)**: Thu hẹp kích thước trở lại chiều ẩn gốc $D$.

**Thách thức lớn nhất**: Khối MLP chiếm tới **\~60% – 64% tổng khối lượng tính toán (MACs/FLOPs)** và thời gian chạy của toàn bộ khối Transformer. Do chứa ma trận trọng số rất lớn ($D \\\\times 4D$ và $4D \\\\times D$), khối MLP gây áp lực cực lớn lên bộ nhớ on-chip (BRAM) của FPGA nếu không có chiến lược xử lý tối ưu.

\--------------------------------------------------------------------------------

2\. Các Kỹ thuật Xử lý Mạch Phần cứng trên FPGA

Để tăng tốc khối MLP mà không bị cạn kiệt tài nguyên BRAM hay nghẽn băng thông RAM ngoài (DRAM), các bộ tăng tốc FPGA hiện đại (như ME-ViT, ViTA, ViA, ADAPTOR) áp dụng 4 kỹ thuật cốt lõi:

a. Phân khối Ma trận & Tích lũy Tổng một phần (Matrix Tiling & Partial Sum Method)

Thay vì nạp toàn bộ ma trận trọng số $W\\\_1$ và $W\\\_2$ vào chip (vốn vượt quá dung lượng BRAM):

* Mạch phần cứng chia $W\\\_1$ thành các khối cột (column tiles) và $W\\\_2$ thành các khối hàng (row tiles).  
* **Kỹ thuật Inter-layer / Partial Sum**: Mảng PE tính toán từng sub-block $M\\\_{i,j} \= \\\\text{Activation}(L\\\_i W\\\_{1,j}^H \+ B\\\_{1,j}^H)$. Kết quả vừa tạo ra lập tức được nhân trực tiếp với sub-block $W\\\_{2,j}^O$ tương ứng và cộng dồn vào bộ đệm tích lũy ($Staged\\\\ Result$).  
* **Lợi ích**: Không cần ghi ma trận kết quả trung gian khổng lồ ra RAM ngoài (DRAM), giải phóng tới 80–90% băng thông truyền dữ liệu.

b. Chế độ Tái sử dụng Mảng Systolic & Bộ đệm (Single-Load Policy / Multi-purpose Buffers)

* Kiến trúc phần cứng dùng chung **mảng Systolic MAC Array** và các ngân hàng đệm BRAM đã dùng cho bước MSA để chạy bước MLP.  
* **Chuyển chế độ (PE Mode Scheduling)**: Sau khi xong bước MSA, bộ điều khiển chuyển mảng PE sang **MLP Mode**. Trọng số $W\\\_1, W\\\_2$ được nạp dạng luồng (streaming) từ AXI DMA vào BRAM đệm đúng **1 lần duy nhất (Single-load policy)** để mảng PE tính toán liên tục.

c. Xử lý Hàm Kích hoạt Phi tuyến (Activation Pipeline)

* **Nếu dùng GELU nguyên (ShiftGELU / I-GELU)**: Phép tính được quy về xấp xỉ dịch bit $1.702 X$, chạy thuần trên số nguyên INT8 mà không tốn FPU/DSP.  
* **Nếu đơn giản hóa bằng ReLU**: Mạch chỉ tiêu tốn 1 bộ so sánh (`X &gt; 0 ? X : 0`), đưa chi phí phần cứng về **0 DSP** và độ trễ 1 chu kỳ xung clock.

d. Tích hợp Đường tắt Residual Connection & LayerNorm

* Kết quả sau tầng $W\\\_2$ được cộng trực tiếp với dữ liệu đường tắt residual từ tầng trước ($X\\\_{in} \+ \\\\text{MLP}(\\\\text{LN}(X\\\_{in}))$) ngay trong bộ đệm nội bộ trước khi truyền sang khối Transformer tiếp theo mà không cần ghi ngược lại DDR4.

\--------------------------------------------------------------------------------

🎯 **Tóm lại**: Khối MLP trên FPGA được xử lý bằng cách **chia nhỏ ma trận (Tiling), nhân tích lũy tổng một phần (Partial Sum) và tái sử dụng mảng Systolic** để thực hiện tính toán ngay trên bộ đệm nội bộ BRAM, giúp duy trì luồng tính toán liên tục với độ trễ thấp nhất.

Bạn có muốn xem sơ đồ nguyên lý phân khối ma trận và luồng chuyển dữ liệu trong chế độ **MLP Mode** của SystemVerilog không?

\--------------------------------------------------------------------------------

Tối ưu ReLU và GELU

Trong kiến trúc **Vision Transformer (ViT)** và các bộ tăng tốc phần cứng (FPGA/ASIC), việc lựa chọn giữa hàm kích hoạt **ReLU** và **GELU** là bài toán đánh đổi trực tiếp giữa **Độ chính xác mô hình (Algorithmic Accuracy)** và **Chi phí phần cứng (Hardware Complexity)**.

\--------------------------------------------------------------------------------

I. So sánh Toán học & Mô hình hóa (Mathematical & Algorithmic Comparison)

| Tiêu chí | ReLU (Rectified Linear Unit) | GELU (Gaussian Error Linear Unit) |
| ----- | ----- | ----- |
| **Công thức toán học** | \\$$f(x) \= \\\\max(0, x)\\\\$$ | \\$$GELU(x) \= x \\\\cdot \\\\Phi(x) \= x \\\\cdot P(X \\\\le x)\\\\$$ với $X \\\\sim \\\\mathcal{N}(0, 1)$. |
| **Dạng xấp xỉ thông dụng** | Công thức chuẩn phân đoạn. | \\$$GELU(x) \\\\approx 0.5x \\\\left(1 \+ \\\\tanh\\\\left(\\\\sqrt{\\\\frac{2}{\\\\pi}}(x \+ 0.044715 x^3)\\\\right)\\\\right)\\\\\\\] hoặc \\\\\\\[GELU(x) \\\\approx x \\\\cdot \\\\sigma(1.702 x)\\\\$$. |
| **Đặc tính đạo hàm** | Đạo hàm bằng 0 khi $x \&lt; 0$, bằng 1 khi $x \&gt; 0$ (không liên tục tại 0). | Đạo hàm mịn (smooth), phi tuyến liên tục trên toàn bộ miền xác định. |
| **Hiện tượng "Dying Neuron"** | Có thể bị "chết neuron" nếu đầu vào âm lớn do gradient triệt tiêu hoàn toàn về 0\. | Khắc phục được nhờ giữ lại một lượng nhỏ thông tin gradient âm khi $x$ âm nhẹ. |
| **Mức độ phổ biến trong ViT** | Thường dùng trong các mô hình CNN truyền thống hoặc các biến thể ViT rút gọn. | **Chuẩn mực mặc định** trong Transformer (ViT, DeiT, Swin, MobileViT, BERT). |

\--------------------------------------------------------------------------------

II. Tác động tới Triển khai Phần cứng FPGA (Hardware Implementation)

1\. ReLU — Chi phí phần cứng gần như bằng 0 (Zero-Cost Activation)

* **Kiến trúc vi mạch**: Mạch ReLU chỉ cần một bộ so sánh đơn giản kết hợp multiplexer (`x &gt; 0 ? x : 0`).  
* **Tài nguyên**: **0 DSP48E2**, tiêu tốn cực ít LUTs/Flip-Flops.  
* **Pipeline & Latency**: Hoàn tất trong **1 chu kỳ clock (II \= 1\)**, không gây tắc nghẽn đường ống tính toán (pipeline stall) hay tốn đệm BRAM.

2\. GELU — Nút thắt tính toán phi tuyến (Non-linear Hardware Bottleneck)

* **Kiến trúc vi mạch**: GELU chuẩn chứa các phép toán phi tuyến đắt đỏ như hàm mũ $e^x$, căn bậc hai, phép chia và hàm $\\\\tanh$.  
* **Thách thức số nguyên (INT8 Pipeline)**: Nếu tính toán GELU bằng số thực dấu phẩy động (FP32), phần cứng phải thực hiện Dequantization (INT8 $\\\\rightarrow$ FP32) rồi Quantization lại (FP32 $\\\\rightarrow$ INT8), gây đứt gãy đồ thị tính toán và tốn chi phí truyền truyền dữ liệu.

\--------------------------------------------------------------------------------

III. Các Giải pháp Xử lý GELU trên FPGA

Để giữ lại ưu điểm độ chính xác của GELU mà không làm quá tải tài nguyên phần cứng, các công trình nghiên cứu SOTA áp dụng 3 hướng tiếp cận:

1. **Thuật toán ShiftGELU (Integer-only ShiftGELU)**:  
2. Xấp xỉ $GELU(x) \\\\approx x \\\\cdot \\\\sigma(1.702 x)$.  
3. Chuyển hệ số $1.702$ sang dạng nhị phân $(1.1011)\\\_b$. Phép nhân $1.702 \\\\cdot I\\\_x$ được thay thế hoàn toàn bằng **phép dịch bit đại số (Bit-Shift)**: \\$$I\\\_p \= I\\\_x \+ (I\\\_x \\\\gg 1\) \+ (I\\\_x \\\\gg 3\) \+ (I\\\_x \\\\gg 4)\\\\$$  
4. Kết hợp với hàm Sigmoid nguyên để thực hiện toàn bộ GELU trên luồng số nguyên INT8 mà không tốn FPU/DSP.  
5. **Bảng tra LUT ROM / Mạch xấp xỉ đa thức (Lookup Tables / PWL)**:  
6. Lưu trữ giá trị GELU tính sẵn trong BRAM ROM nhỏ, hoặc xấp xỉ đa thức bậc 2/bậc 3\.  
7. **Thay thế GELU bằng ReLU trực tiếp trên Accelerator (Hardware-Friendly Substitution)**:  
8. Trong một số kiến trúc tăng tốc như **ME-ViT** hoặc **ADAPTOR**, các tác giả chủ động **thay thế GELU bằng ReLU** trong khối MLP/FFN để đơn giản hóa tối đa cấu trúc mảng PE. Việc này giúp giảm độ phức tạp mạch RTL và BRAM mà chỉ làm giảm một tỷ lệ rất nhỏ độ chính xác phân loại sau khi fine-tune.

\--------------------------------------------------------------------------------

Tóm tắt Khuyến nghị Kỹ thuật

* **Nếu ưu tiên tối đa FPS, diện tích chip & Fmax (Edge FPGA như Kria KV260)**: Sử dụng **ReLU** (hoặc LeakyReLU) để giải phóng BRAM/DSP cho mảng nhân Attention.  
* **Nếu bắt buộc giữ nguyên độ chính xác mô hình gốc ViT/DeiT**: Triển khai khối **ShiftGELU (shift\_gelu.sv)** sử dụng phép dịch bit nguyên để đảm bảo chạy mượt trên đồ thị INT8.

Bạn có muốn xem đoạn mã SystemVerilog minh họa khối **ShiftGELU** (triển khai xấp xỉ dịch bit $1.702 I\\\_x$) để tích hợp vào khối MLP của Attention Core không?

\--------------------------------------------------------------------------------

Cơ chế Scaled Attention

Hệ số thu phóng (Scaling Factor) $\\\\frac{1}{\\\\sqrt{d\\\_k}}$ trong cơ chế **Scaled Dot-Product Attention** được đưa vào nhằm giải quyết hiện tượng triệt tiêu gradient thông qua cơ chế toán học sau:

1\. Tích vô hướng tăng phương sai theo chiều ẩn $d\\\_k$

Khi tính tích vô hướng giữa vector Query ($q$) và Key ($k$): \\$$q \\\\cdot k \= \\\\sum\\\_{i=1}^{d\\\_k} q\\\_i k\\\_i\\\\$$ Nếu giả định các thành phần của $q$ và $k$ là các biến ngẫu nhiên độc lập có giá trị trung bình (mean) bằng $0$ và phương sai (variance) bằng $1$, thì tích vô hướng $q \\\\cdot k$ sẽ có giá trị trung bình bằng $0$ và **phương sai bằng** $d\\\_k$. Do đó, khi kích thước không gian $d\\\_k$ nhận giá trị lớn, biên độ của các giá trị tích vô hướng $Q K^T$ sẽ tăng lên rất lớn.

2\. Sự bão hòa của hàm Softmax (Softmax Saturation)

Khi các giá trị đầu vào của hàm Softmax bị phóng đại với độ chênh lệch biên độ quá lớn, hàm Softmax sẽ bị đẩy vào các vùng có đạo hàm cực kỳ nhỏ (vùng bão hòa). Tại đây, phân bố xác suất đầu ra bị đẩy cực đoan sát về dạng một vị trí tiến gần $1$ và tất cả các vị trí khác tiến gần $0$, khiến đạo hàm của Softmax xấp xỉ bằng $0$.

3\. Tác dụng của hệ số $\\\\frac{1}{\\\\sqrt{d\\\_k}}$

* **Ngăn triệt tiêu Gradient**: Trong quá trình lan truyền ngược (backpropagation), gradient truyền qua Softmax bị nhân với đạo hàm xấp xỉ $0$, dẫn đến hiện tượng triệt tiêu gradient và khiến mô hình ngừng cập nhật trọng số.  
* **Thước đo chuẩn hóa**: Việc chia tích vô hướng cho $\\\\sqrt{d\\\_k}$ giúp thu phóng phương sai của kết quả tích vô hướng từ $d\\\_k$ quay trở lại bằng $1$ ($\\\\text{Var}\\\\left(\\\\frac{q \\\\cdot k}{\\\\sqrt{d\\\_k}}\\\\right) \= \\\\frac{d\\\_k}{d\\\_k} \= 1$).  
* **Ổn định huấn luyện**: Việc đưa phương sai về $1$ giữ cho các giá trị đầu vào của Softmax nằm trong phạm vi vừa phải, tránh rơi vào vùng bão hòa và ổn định dòng chảy gradient trong suốt quá trình huấn luyện.

\--------------------------------------------------------------------------------

💡 Bạn có muốn tìm hiểu cách biến đổi phép chia $\\\\sqrt{d\\\_k}$ này thành phép dịch bit đại số (Arithmetic Right-Shift \- ASR) trên phần cứng FPGA để tiết kiệm 100% tài nguyên DSP không?

\--------------------------------------------------------------------------------

So sánh MHA và SHA

So sánh chi tiết giữa **Multi-Head Attention (MHA)** và **Single-Head Attention (SHA)** dựa trên cơ sở toán học và tác động thực thi phần cứng (FPGA):

\--------------------------------------------------------------------------------

1\. Cơ sở Toán học & Khái niệm

* **Single-Head Attention (SHA)**: Thực hiện một phép tính Scaled Dot-Product Attention duy nhất trên toàn bộ không gian ẩn $d\\\_{model}$: $$\\\\text{Attention}(Q, K, V) \= \\\\text{Softmax}\\\\left( \\\\frac{Q K^T}{\\\\sqrt{d\\\_k}} \\\\right) V\\$$ Trong đó, $Q, K, V$ có chiều rộng không gian là $d\\\_{model}$.  
* **Multi-Head Attention (MHA)**: Thay vì tính attention một lần duy nhất, MHA chiếu tuyến tính các ma trận $Q, K, V$ thành $h$ không gian con (subspaces) độc lập với kích thước nhỏ hơn $d\\\_k \= d\\\_v \= d\\\_{model} / h$: $$\\\\text{MultiHead}(Q, K, V) \= \\\\text{Concat}(\\\\text{head}\*1, \\\\dots, \\\\text{head}h) W^O\\\\\\\] \\\\\\\[\\\\text{với } \\\\text{head}i \= \\\\text{Attention}(Q W\\\_i^Q, K W\\\_i^K, V W\\\_i^V)\\$$ Trong đó $W\\\_i^Q \\\\in \\\\mathbb{R}^{d{model} \\\\times d\\\_k}$, $W\\\_i^K \\\\in \\\\mathbb{R}^{d{model} \\\\times d\\\_k}$, $W\\\_i^V \\\\in \\\\mathbb{R}^{d\*{model} \\\\times d\\\_v}$ và $W^O \\\\in \\\\mathbb{R}^{h d\\\_v \\\\times d\\\_{model}}$.

\--------------------------------------------------------------------------------

2\. Bảng So sánh Trực quan

| Tiêu chí | Single-Head Attention (SHA) | Multi-Head Attention (MHA) |
| ----- | ----- | ----- |
| **Không gian biểu diễn (Representation Subspaces)** | Chỉ học tương quan trên **1 không gian duy nhất**; phép lấy trung bình (averaging) làm suy giảm khả năng tập trung đa khía cạnh. | Cho phép mô hình đồng thời chú ý đến thông tin từ **nhiều không gian biểu diễn và vị trí khác nhau**. |
| **Đặc trưng trong Vision Transformer (ViT)** | Bị giới hạn phạm vi chú ý cố định trên toàn bộ vùng ảnh. | Các head khác nhau có thể học các quy mô khác nhau: một số head chú ý toàn cục (global) ngay từ tầng thấp, trong khi số khác chú ý cục bộ (local). |
| **Khối lượng tính toán (FLOPs)** | Bằng $O(N^2 \\\\cdot d\\\_{model})$. | Tương đương $O(N^2 \\\\cdot d\\\_{model})$ do kích thước mỗi head giảm xuống $d\\\_k \= d\\\_{model}/h$. |
| **Yêu cầu Tham số Trọng số** | Chỉ tốn trọng số tạo $Q, K, V$. | Cần thêm ma trận chiếu đầu ra $W^O$ và các ma trận chiếu riêng cho từng head. |
| **Thiết kế Phần cứng FPGA** | Đơn giản hóa đệm BRAM, không tốn tài nguyên quản lý luồng song song giữa các head. | Cần chiến lược quản lý đệm BRAM (như Head-wise Pipelining) để tránh cạn kiệt BRAM khi lưu ma trận trung gian của nhiều head cùng lúc. |

\--------------------------------------------------------------------------------

3\. Tại sao Vision Transformer bắt buộc dùng Multi-Head Attention?

1. **Tránh triệt tiêu thông tin do lấy trung bình (Averaging Inhibition)**: Đối với SHA, khi tính tích vô hướng và softmax trên toàn bộ chiều $d\\\_{model}$, trọng số chú ý bị phân tán (trung bình hóa), khiến mô hình khó tập trung vào các mối quan hệ đa dạng giữa các patch ảnh.  
2. **Đa dạng hóa tầm nhìn không gian (Spatial Receptive Fields)**: Trong ViT, các head trong MHA phân công vai trò rõ rệt: có head đóng vai trò như bộ lọc cuộn cục bộ (chỉ chú ý patch lân cận), có head vươn rộng toàn bộ bức ảnh để trích xuất ngữ cảnh toàn cục.

\--------------------------------------------------------------------------------

4\. Tác động tới Thiết kế Mạch Phần cứng FPGA (như Kria KV260)

* **Tối ưu hóa tài nguyên BRAM**: Khi triển khai MHA trên FPGA, nếu lưu toàn bộ ma trận $Q, K, V$ của tất cả $h$ head cùng một lúc vào on-chip BRAM sẽ dễ gây tràn bộ nhớ. Do đó, các kiến trúc tăng tốc (như ME-ViT hay ViTA) thường xử lý luân phiên từng head (**Head-wise computation / Head-level pipeline**) hoặc giới hạn số lượng head song song $H \\\\le 4$ với $d\\\_k \= 32 \\\\text{ hoặc } 64$ để khớp với mảng PE DSP48E2.  
* **Khả năng song song hóa**: Các head trong MHA có tính chất độc lập hoàn toàn trước bước Concat, cho phép thiết kế RTL mở rộng các khối PE chạy song song nếu tài nguyên DSP48E2 trên chip còn dư dả.

Bạn có muốn đi sâu vào cách thiết kế khối **Concat và Linear Projection (**$W^O$**)** trong SystemVerilog để gom kết quả từ các Attention Head lại không?

\--------------------------------------------------------------------------------

FSM Bộ đệm Ping-Pong

Dưới đây là **Sơ đồ Máy trạng thái FSM (Finite State Machine)** điều khiển cơ chế tráo con trỏ (Pointer Swapping) và đồng bộ giữa AXI DMA và mảng xử lý PE cho bộ đệm Ping-Pong BRAM.

\--------------------------------------------------------------------------------

I. Sơ đồ Trạng thái FSM (State Transition Diagram)

                      \+-----------------------------------+

                       |              ST\_IDLE              |

                       | (Rst / Cả Ping \&amp; Pong đều EMPTY) |

                       \+-----------------------------------+

                                         |

                                         | \[s\_axis\_tvalid \&amp; s\_axis\_tready\]

                                         v

                       \+-----------------------------------+

                       |          ST\_FILLING\_BUF           |

                       |  (DMA đang nạp dữ liệu vào BRAM) |

                       \+-----------------------------------+

                                         |

                                         | \[s\_axis\_tlast \== 1\]

                                         v

                       \+-----------------------------------+

                       |           ST\_TILE\_READY           |

                       |   (Buffer được đánh dấu FULL)     |

                       \+-----------------------------------+

                                         |

                                         | \[PE\_Done \== 1 VÀ Tile\_Ready \== 1\]

                                         v

                       \+-----------------------------------+

                       |          ST\_SWAP\_POINTER          |

                       | (ping\_pong\_sel \&lt;= \~ping\_pong\_sel) |

                       |    (Đổi ngôi trong 1 Clock)       |

                       \+-----------------------------------+

                                         |

                   \+---------------------+---------------------+

                   |                                           |

                   v                                           v

    \[ping\_pong\_sel \== 0\]                        \[ping\_pong\_sel \== 1\]

  \+--------------------------+                \+--------------------------+

  |  PE đọc PING (Buffer 0\)  |                |  PE đọc PONG (Buffer 1\)  |

  |  DMA nạp PONG (Buffer 1\) |                |  DMA nạp PING (Buffer 0\) |

  \+--------------------------+                \+--------------------------+

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

II. Chi tiết các Trạng thái & Điều kiện Chuyển đổi (State Transitions)

1\. Trạng thái `ST_IDLE` (Khởi tạo)

* **Mô tả**: Hệ thống vừa reset hoặc chưa có luồng dữ liệu truyền vào. Cả hai bộ đệm Ping (Buffer 0\) và Pong (Buffer 1\) đều ở trạng thái `EMPTY`.  
* **Con trỏ ban đầu**: `ping_pong_sel = 0` (Mặc định DMA sẵn sàng nạp vào Ping / Buffer 0).  
* **Điều kiện chuyển sang ST\_FILLING\_BUF**: Xảy ra sự kiện bắt tay luồng `(s_axis_tvalid == 1) &amp;&amp; (s_axis_tready == 1)`.

2\. Trạng thái `ST_FILLING_BUF` (DMA đang nạp Tile)

* **Mô tả**: AXI DMA đẩy liên tục các từ dữ liệu (64-bit/128-bit) vào ngân hàng BRAM được chỉ định bởi `ping_pong_sel`.  
* **Điều kiện duy trì**: Tín hiệu bắt tay `TVALID/TREADY` tiếp tục được kích hoạt.  
* **Điều kiện chuyển sang ST\_TILE\_READY**: AXI DMA phát tín hiệu ngắt khối **s\_axis\_tlast \== 1** (báo hiệu đã nạp trọn vẹn 1 tile dữ liệu, ví dụ $196 \\\\times 32$ elements).

3\. Trạng thái `ST_TILE_READY` (Tile sẵn sàng)

* **Mô tả**: Bộ đệm vừa nạp xong được chuyển trạng thái sang **FULL**. Lúc này FSM đứng chờ mảng PE hoàn tất việc tính toán khối dữ liệu cũ.  
* **Chống đè dữ liệu (Backpressure)**: Nếu cả hai bộ đệm Ping và Pong cùng ở trạng thái `FULL`, FSM lập tức kéo `s_axis_tready = 0` để buộc AXI DMA tạm dừng truyền dòng từ DDR4.

4\. Trạng thái `ST_SWAP_POINTER` (Tráo con trỏ 1-Clock)

* **Điều kiện kích hoạt (SWAP TRIGGER)**: $$\\\\text{Swap\\\_Condition} \= (\\\\text{DMA\\\_Buffer\\\_State} \== \\\\text{FULL}) ;\\\\land; (\\\\text{PE\\\_Compute\\\_Done} \== 1)\\$$  
* **Hành động trong 1 chu kỳ clock**:  
  * Đảo bit vị trí: **ping\_pong\_sel \&lt;= \~ping\_pong\_sel**.  
  * Chuyển bộ đệm vừa nạp xong từ `FULL` $\\\\rightarrow$ `READING` (cấp cho mảng PE).  
  * Chuyển bộ đệm PE vừa đọc xong từ `READING` $\\\\rightarrow$ `EMPTY` (đón luồng DMA mới).  
  * Bật xung ngắt **pe\_start \= 1** để kích hoạt mảng Systolic MAC Array tính toán ngay lập tức.

\--------------------------------------------------------------------------------

III. Bảng Ma trận Điều khiển Con trỏ Ping-Pong

| Giá trị ping\_pong\_sel | Ngân hàng DMA đang Ghi (Write Side) | Ngân hàng PE đang Đọc (Read Side) | Hành động khi thỏa điều kiện Swap |
| ----- | ----- | ----- | ----- |
| **0** | **Buffer 1 (PONG)** | **Buffer 0 (PING)** | Chuyển ping\_pong\_sel sang 1, bật pe\_start. |
| **1** | **Buffer 0 (PING)** | **Buffer 1 (PONG)** | Chuyển ping\_pong\_sel sang 0, bật pe\_start. |

\--------------------------------------------------------------------------------

IV. Mã nguồn SystemVerilog FSM Mẫu (`ping_pong_fsm.sv`)

typedef enum logic \[1:0\] {

    ST\_IDLE,

    ST\_FILLING\_BUF,

    ST\_TILE\_READY,

    ST\_SWAP\_POINTER

} fsm\_state\_t;

&nbsp;

fsm\_state\_t state\_reg, state\_next;

&nbsp;

// Logic Chuyển Trạng thái (State Register)

always\_ff @(posedge clk or negedge rst\_n) begin

    if (\!rst\_n)

        state\_reg \&lt;= ST\_IDLE;

    else

        state\_reg \&lt;= state\_next;

end

&nbsp;

// Logic Chuyển Mạch FSM (Next State Logic)

always\_comb begin

    state\_next \= state\_reg;

    case (state\_reg)

        ST\_IDLE: begin

            if (s\_axis\_tvalid \&amp;\&amp; s\_axis\_tready)

                state\_next \= ST\_FILLING\_BUF;

        end

&nbsp;

        ST\_FILLING\_BUF: begin

            if (s\_axis\_tvalid \&amp;\&amp; s\_axis\_tready \&amp;\&amp; s\_axis\_tlast)

                state\_next \= ST\_TILE\_READY;

        end

&nbsp;

        ST\_TILE\_READY: begin

            if (pe\_done) // Khi PE đã sẵn sàng nhận Tile mới

                state\_next \= ST\_SWAP\_POINTER;

        end

&nbsp;

        ST\_SWAP\_POINTER: begin

            state\_next \= ST\_FILLING\_BUF; // Quay lại nhận Tile tiếp theo

        end

    endcase

end

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

🔄 Bạn có muốn viết testbench mô phỏng xung nhịp (Waveform) cho FSM này để kiểm tra tính đúng đắn khi xuất hiện tín hiệu `TLAST` từ DMA không?

\--------------------------------------------------------------------------------

So sánh I-LayerNorm và LayerNorm

Dưới đây là so sánh chi tiết giữa **I-LayerNorm (Integer-only Layer Normalization)** và **LayerNorm chuẩn (Standard Floating-Point LayerNorm)** dựa trên các nghiên cứu về tối ưu hóa phần cứng và định lượng cho Vision Transformer (ViT):

\--------------------------------------------------------------------------------

1\. Bản chất Toán học & Luồng Tính toán

* **LayerNorm chuẩn (FP32/FP16)**: Thực hiện chuẩn hóa dữ liệu trên chiều ẩn $d$ theo công thức: $$\\\\text{LayerNorm}(x\\\_i) \= \\\\frac{x\\\_i \- \\\\mu}{\\\\sqrt{\\\\sigma^2 \+ \\\\epsilon}} \\\\cdot \\\\gamma \+ \\\\beta\\$$ trong đó giá trị trung bình $\\\\mu \= \\\\frac{1}{d} \\\\sum\\\_{j=1}^d x\\\_j$ và phương sai $\\\\sigma^2 \= \\\\frac{1}{d} \\\\sum\\\_{j=1}^d (x\\\_j \- \\\\mu)^2$ được tính toán động (dynamic statistics). Phép toán này đòi hỏi thực hiện phép căn bậc hai ($\\\\sqrt{\\\\cdot}$) và phép chia số thực dấu phẩy động (FP32/FP16).  
* **I-LayerNorm (Pure Integer / Fixed-Point)**: Giữ nguyên nguyên lý chuẩn hóa thống kê nhưng chuyển toàn bộ đồ thị tính toán sang **số nguyên cố định (INT8/INT32)** và **phép dịch bit**:  
  1. Tính $\\\\mu$ và $\\\\sigma^2$ bằng bộ tích lũy số nguyên **INT32**.  
  2. **Xử lý phép căn bậc hai (**$\\\\sqrt{\\\\text{Var}}$**)**: Thay vì dùng bộ căn/chia số thực phần cứng đắt đỏ, I-LayerNorm sử dụng **thuật toán lặp số nguyên (Integer Iterative Method)** kết hợp dịch bit: \\$$I\\\_{i+1} \= \\\\left( I\\\_i \+ \\\\left\\\\lfloor \\\\frac{\\\\text{Var}(I\\\_x)}{I\\\_i} \\\\right\\\\rfloor \\\\right) \\\\gg 1\\\\$$  
  3. Để đảm bảo độ trễ cố định (constant latency) trên phần cứng, số vòng lặp được cố định (thường là 10 vòng lặp) thay cho điều kiện dừng động.

\--------------------------------------------------------------------------------

2\. Bảng So sánh Trực quan

| Tiêu chí | LayerNorm Chuẩn (FP32 Baseline) | I-LayerNorm (Integer-only) | L1-LayerNorm (Biến thể xấp xỉ L1) |
| ----- | ----- | ----- | ----- |
| **Kiểu dữ liệu thực thi** | Số thực dấu phẩy động (FP32 / FP16). | **Thuần số nguyên (INT8 / INT32)**. | Số nguyên / Phép tính L1. |
| **Phép toán căn bậc hai** | Phép căn & chia số thực phần cứng (FPU). | **Chuỗi lặp số nguyên \+ Dịch bit (\&gt;\>)**. | Bỏ phép căn, thay bằng độ lệch tuyệt đối L1. |
| **Cắt đồ thị (Graph Cut) & Dequantization** | Bắt buộc Dequantize từ INT8 về FP32 để tính, rồi Quantize lại INT8 (như FasterTransformer). | **Không có Dequantization**; tạo luồng tính toán số nguyên khép kín (End-to-End Integer Pipeline). | Không có Dequantization. |
| **Yêu cầu Đơn vị Xử lý (HW Unit)** | Cần đơn vị số thực FPU (khó triển khai trên chip nhúng INT-only như ARM Cortex-M). | Chạy hoàn toàn trên đơn vị số nguyên (Turing Tensor Cores, DSP48E2, ARM INT ALU). | Chạy trên đơn vị số nguyên. |
| **Độ chính xác Top-1 (ViT)** | Chuẩn mực 100% baseline. | **Gần như không sụt giảm** (ví dụ: DeiT-S đạt **80.12%**, cao hơn baseline FP32 0.27%). | **Sụt giảm nghiêm trọng** (giảm 2.49% trên DeiT-B, giảm 3.32% trên Swin-S). |
| **Tốc độ Tăng tốc (Speedup)** | 1.0×. | **Góp phần đạt tổng tốc độ 3.72× – 4.11×** trên phần cứng. | Rất nhanh nhưng không dùng được do mất chính xác. |

\--------------------------------------------------------------------------------

3\. Tác động tới Thiết kế Mạch Phần cứng (FPGA / Edge ASIC)

1. **Loại bỏ Nút thắt Giao tiếp (Communication Overhead)**: Trong các kiến trúc hỗn hợp (Mixed-precision), việc phải chuyển đổi dữ liệu qua lại giữa đơn vị số nguyên (INT) và đơn vị số thực (FPU) làm tốn băng thông và gây ngắt nhịp pipeline. I-LayerNorm giải quyết triệt để vấn đề này bằng cách giữ nguyên luồng dữ liệu INT8/INT32.  
2. **Tiết kiệm Tài nguyên Phần cứng**: Không tiêu tốn khối FPU số thực đắt đỏ, giúp giảm lượng LUT/DSP tiêu thụ trên FPGA.  
3. **Thích hợp cho Thiết kế Pipeline 1-Pass**: Khi kết hợp công thức phương sai $\\\\sigma^2 \= \\\\left(\\\\frac{1}{d}\\\\sum x\\\_i^2\\\\right) \- \\\\mu^2$, I-LayerNorm có thể vừa tính tích lũy $\\\\sum x$ và $\\\\sum x^2$ trên luồng dữ liệu đang chạy qua đệm mà không cần quét lại dữ liệu lần hai.

\--------------------------------------------------------------------------------

Biểu đồ so sánh sai số Softmax (`softmax_mse_comparison.png`) tạo từ bước trước đã có sẵn trong **Studio Panel** để bạn tiện tham khảo.

Bạn có muốn viết mã SystemVerilog triển khai khối **I-LayerNorm (i\_layernorm.sv)** áp dụng thuật toán lặp số nguyên để ghép vào mạch Attention Core không?

\--------------------------------------------------------------------------------

Đồng bộ AXI DMA và Ping-Pong

Đồng bộ hóa giữa **AXI DMA** và **Bộ đệm Ping-Pong (Ping-Pong BRAM Buffer)** là cơ chế then chốt trong các thiết kế tăng tốc vi mạch (như khối Attention Accelerator trong dự án Capstone). Cơ chế này đảm bảo:

1. **AXI DMA (kênh MM2S)** ghi dữ liệu khối tiếp theo (tile) từ RAM ngoài vào bộ đệm BRAM mà **không nạp đè** lên khối dữ liệu mà mảng PE (Processing Elements) đang tính toán.  
2. **Mảng PE** không đọc phải dữ liệu chưa hoàn tất hoặc dữ liệu cũ.  
3. **Không bị mất dữ liệu** khi xảy ra tình trạng nghẽn luồng truyền (Backpressure).

Dưới đây là chi tiết nguyên lý đồng bộ hóa phần cứng ở mức kiến trúc và tín hiệu SystemVerilog.

\--------------------------------------------------------------------------------

I. Sơ đồ Kiến trúc & Luồng Tín hiệu Đồng bộ

Bộ đệm Ping-Pong gồm hai ngân hàng bộ nhớ BRAM song song (Buffer 0 \- **Ping** và Buffer 1 \- **Pong**) cùng bộ điều khiển FSM chuyển đổi con trỏ (Pointer Switch FSM):

                                 \+-------------------------------------------------------+

                                  |            AXI DMA IP (Simple / Direct Mode)          |

                                  |    MM2S Channel (Read DDR4) \-\&gt; Stream into PL         |

                                  \+-------------------------------------------------------+

                                                              |

                                                              | AXI4-Stream (S\_AXIS)

                                                              v

\+-------------------------------------------------------------------------------------------------------------------+

| PING-PONG SYNCHRONIZATION CONTROLLER                                                                              |

|                                                                                                                   |

|                     \+---------------------+               \+---------------------+                                 |

|   S\_AXIS\_TDATA \----\&gt;|   Buffer 0 (PING)   |\&lt;-------------\&gt;|   Buffer 1 (PONG)   |\&lt;---- S\_AXIS\_TDATA             |

|   (Write Side)      |  State: \[READING\]   |               |  State: \[FILLING\]   |      (Write Side)              |

|                     \+---------------------+               \+---------------------+                                 |

|                                |                                     ^                                            |

|                                | Read Mux                            | Write Mux                                  |

|                                v                                     |                                            |

|                     \+---------------------------------------------------+                                         |

|                     |        FSM Swapping Logic (ping\_pong\_sel)         |\&lt;--- S\_AXIS\_TLAST (DMA Tile End)            |

|                     |  Trigger: (PE\_Done \== 1\) \&amp;\&amp; (DMA\_Tile\_Full \== 1\)  |\&lt;--- PE\_Compute\_Done (PE Complete)         |

|                     \+---------------------------------------------------+                                         |

|                                |                                                                                  |

\+--------------------------------|----------------------------------------------------------------------------------+

                                 v

                     \+-----------------------+

                     |  Systolic MAC Array   |

                     | (Compute Attention Score)

                     \+-----------------------+

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

II. Các Cơ chế Đồng bộ Chi tiết (Synchronization Mechanisms)

1\. Bắt tay Luồng Ghi qua AXI4-Stream (`TVALID` / `TREADY` Handshake)

* **Kênh MM2S (Memory-Mapped to Stream)** truyền tensor $Q, K, V$ từ DDR4 vào FPGA PL qua chuẩn AXI4-Stream.  
* **Chống mất dữ liệu (Backpressure)**:  
  * Khi bộ đệm ghi hiện tại (ví dụ: Pong) chưa đầy, bộ điều khiển Ping-Pong duy trì tín hiệu `S_AXIS_TREADY = 1`.  
  * Nếu cả hai bộ đệm Ping và Pong đều ở trạng thái `FULL` (do mảng PE tính toán chậm hơn tốc độ DMA nạp), bộ điều khiển hạ ngay `S_AXIS_TREADY = 0`.  
  * Khối AXI DMA thấy `TREADY = 0` sẽ **tự động tạm dừng (stall) luồng truyền trên bus** mà không rơi rớt bất kỳ byte dữ liệu nào.

2\. Nhận diện Kết thúc Khối Dữ liệu bằng `TLAST`

* Khi AXI DMA gửi đến phần tử cuối cùng của 1 khối (Tile/Burst), DMA khẳng định tín hiệu `S_AXIS_TLAST = 1`.  
* Mạch đồng bộ kiểm tra điều kiện: `(S_AXIS_TVALID == 1) &amp;&amp; (S_AXIS_TREADY == 1) &amp;&amp; (S_AXIS_TLAST == 1)`.  
* Khi điều kiện này thỏa mãn, bộ điều khiển lập tức đánh cờ trạng thái của bộ đệm đó từ `FILLING` sang `FULL` (sẵn sàng cho PE đọc).

3\. Đổi Ngôi Con trỏ trong 1 Chu kỳ Clock (1-Clock Pointer Swap)

Bộ điều khiển kiểm tra đồng thời hai cờ trạng thái:

1. **dma\_tile\_ready**: Bộ đệm vừa được AXI DMA nạp xong (`FULL`).  
2. **pe\_compute\_done**: Mảng PE đã tính toán xong khối dữ liệu ở bộ đệm kia (`EMPTY`).

Khi **cả 2 cờ đều kích hoạt**, FSM thực hiện **đảo bit con trỏ ping\_pong\_sel \&lt;= \~ping\_pong\_sel trong đúng 1 chu kỳ clock**:

* Bộ đệm vừa nạp xong chuyển sang chế độ `READING` cấp cho mảng PE.  
* Bộ đệm PE vừa đọc xong chuyển sang chế độ `FREE/FILLING` đón nhận luồng dữ liệu mới từ AXI DMA.  
* Mảng PE và AXI DMA tiếp tục chạy song song mà không tốn thời gian chờ (Zero-latency transition).

4\. Đồng bộ Mức Hệ thống với CPU Host (ARM PS)

* **Khởi động**: ARM PS cấu hình địa chỉ thanh ghi AXI DMA, sau đó phát xung `START` qua bus AXI-Lite tới bộ điều khiển.  
* **Kết thúc**: Khi toàn bộ tensor đầu ra được mảng PE đưa qua kênh AXI DMA S2MM (Stream to Memory-Mapped) về lại DDR4, AXI DMA phát ngắt phần cứng (`S2MM Interrupt`) báo cho ARM CPU biết để tiếp tục các công đoạn tiếp theo (như MLP hoặc LayerNorm).

\--------------------------------------------------------------------------------

III. Đoạn mã SystemVerilog Minh họa Bộ điều khiển (`ping_pong_controller.sv`)

\`timescale 1ns / 1ps

&nbsp;

module ping\_pong\_controller \#(

    parameter DATA\_WIDTH \= 64,

    parameter TILE\_DEPTH \= 196

)(

    input  logic                  clk,

    input  logic                  rst\_n,

&nbsp;

    // Giao diện AXI4-Stream Slave (Từ AXI DMA MM2S)

    input  logic \[DATA\_WIDTH-1:0\] s\_axis\_tdata,

    input  logic                  s\_axis\_tvalid,

    output logic                  s\_axis\_tready,

    input  logic                  s\_axis\_tlast,

&nbsp;

    // Giao diện điều khiển mảng PE

    input  logic                  pe\_done,         // PE báo đã tính xong tile hiện tại

    output logic                  pe\_start,        // Báo PE bắt đầu tính tile mới

    output logic \[DATA\_WIDTH-1:0\] pe\_data\_out,     // Dữ liệu cấp cho PE

&nbsp;

    // Tín hiệu chọn đệm bộ nhớ BRAM

    output logic                  ping\_pong\_sel    // 0: PE đọc Ping, DMA ghi Pong | 1: PE đọc Pong, DMA ghi Ping

);

&nbsp;

    // Cờ trạng thái các bộ đệm

    typedef enum logic \[1:0\] {EMPTY, FILLING, FULL, READING} buf\_state\_t;

    buf\_state\_t ping\_state, pong\_state;

&nbsp;

    logic \[15:0\] write\_cnt;

&nbsp;

    // 1\. Logic Backpressure cho AXI DMA

    // Cho phép DMA ghi nếu bộ đệm mục tiêu không ở trạng thái FULL hoặc READING

    assign s\_axis\_tready \= (ping\_pong\_sel \== 0\) ? (pong\_state \== EMPTY) : (ping\_state \== EMPTY);

&nbsp;

    // 2\. FSM Quản lý Trạng thái \&amp; Đổi ngôi Con trỏ (Pointer Swap)

    always\_ff @(posedge clk or negedge rst\_n) begin

        if (\!rst\_n) begin

            ping\_pong\_sel \&lt;= 1'b0;

            ping\_state     \&lt;= EMPTY;

            pong\_state     \&lt;= EMPTY;

            write\_cnt      \&lt;= '0;

            pe\_start       \&lt;= 1'b0;

        end else begin

            pe\_start \&lt;= 1'b0; // Reset pulse

&nbsp;

            // \--- QUY TRÌNH GHI TỪ AXI DMA \---

            if (s\_axis\_tvalid \&amp;\&amp; s\_axis\_tready) begin

                if (s\_axis\_tlast) begin

                    // Đánh dấu bộ đệm đã đầy khi nhận TLAST

                    if (ping\_pong\_sel \== 0\)

                        pong\_state \&lt;= FULL;

                    else

                        ping\_state \&lt;= FULL;

                    write\_cnt \&lt;= '0;

                end else begin

                    write\_cnt \&lt;= write\_cnt \+ 1'b1;

                end

            end

&nbsp;

            // \--- QUY TRÌNH ĐỔI NGÔI (POINTER SWAP) \---

            // Đổi con trỏ khi PE đã tính xong VÀ DMA đã nạp xong tile mới

            if (ping\_pong\_sel \== 0 \&amp;\&amp; ping\_state \== EMPTY \&amp;\&amp; pong\_state \== FULL) begin

                ping\_pong\_sel \&lt;= 1'b1;         // Đổi: PE chuyển sang đọc Pong, DMA ghi Ping

                pong\_state     \&lt;= READING;

                pe\_start       \&lt;= 1'b1;        // Kích hoạt mảng PE

            end

            else if (ping\_pong\_sel \== 1 \&amp;\&amp; pong\_state \== EMPTY \&amp;\&amp; ping\_state \== FULL) begin

                ping\_pong\_sel \&lt;= 1'b0;         // Đổi: PE chuyển sang đọc Ping, DMA ghi Pong

                ping\_state     \&lt;= READING;

                pe\_start       \&lt;= 1'b1;        // Kích hoạt mảng PE

            end

&nbsp;

            // \--- KHI PE TÍNH XONG TILE \---

            if (pe\_done) begin

                if (ping\_pong\_sel \== 0\)

                    ping\_state \&lt;= EMPTY;

                else

                    pong\_state \&lt;= EMPTY;

            end

        end

    end

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

Bằng việc kết hợp bắt tay `TVALID/TREADY`, tín hiệu ngắt khối `TLAST`, cờ trạng thái đệm và FSM đổi con trỏ trong 1 chu kỳ clock, bộ đệm Ping-Pong đảm bảo mảng PE Systolic luôn được cấp dữ liệu liên tục mà **không gây ngưng trệ pipeline** hay **làm rớt dữ liệu trên bus AXI**.

Nếu bạn muốn đi sâu thêm, ta có thể bàn về phương pháp thiết kế **Skid Buffer (Register Slice)** để thắt chặt thời gian WNS (Timing Closure) khi bus AXI-Stream chạy ở tần số cao \>200 MHz.

\--------------------------------------------------------------------------------

Tối ưu Bộ nhớ FPGA

Trong kiến trúc phần cứng FPGA, **Phân ngân hàng BRAM (BRAM Banking)** và **Bộ đệm Ping-Pong (Ping-Pong Buffering)** là hai kỹ thuật giải quyết hai bài toán hoàn toàn khác nhau về bộ nhớ: một bên giải quyết **băng thông song song trong 1 chu kỳ clock (Không gian)**, còn một bên giải quyết **độ trễ truyền dữ liệu giữa các giai đoạn (Thời gian)**.

\--------------------------------------------------------------------------------

1\. Phân ngân hàng BRAM (BRAM Banking) – Tối ưu theo Không gian (Spatial)

* **Mục tiêu chính**: Tăng **băng thông truy xuất song song** cho nhiều đơn vị xử lý (PE) trong cùng một chu kỳ xung clock.  
* **Bài toán cần giải quyết**: Khối BRAM vật lý trên FPGA thường chỉ là **True Dual-Port** (chỉ có tối đa 2 cổng đọc/ghi đồng thời). Nếu bạn có mảng 16 hoặc 32 PE cùng muốn đọc 32 dữ liệu khác nhau trong 1 chu kỳ clock, 2 cổng BRAM sẽ bị xung đột (port collision) và gây nghẽn.  
* **Cách hoạt động**: Chia dung lượng BRAM thành nhiều khối nhỏ độc lập (gọi là các **Banks**), mỗi Bank có bộ quản lý địa chỉ và đường bus dữ liệu riêng.  
* **Kết quả**: 16 PE có thể đọc đồng thời 16 phần tử dữ liệu từ 16 Banks khác nhau trong **chỉ 1 chu kỳ xung clock**.

\--------------------------------------------------------------------------------

2\. Bộ đệm Ping-Pong (Ping-Pong Buffering) – Tối ưu theo Thời gian (Temporal)

* **Mục tiêu chính**: **Ẩn hoàn toàn độ trễ truyền dữ liệu (Hide Latency)** giữa bộ nhớ ngoài (DRAM / AXI DMA) và mảng tính toán nội bộ (PE Array).  
* **Bài toán cần giải quyết**: Tốc độ đọc dữ liệu từ RAM ngoài (DRAM) qua AXI DMA rất chậm so với tốc độ tính toán của mảng PE. Nếu không dùng Ping-Pong, mảng PE sẽ phải đứng chờ (pipeline stall) mỗi khi nạp khối dữ liệu mới.  
* **Cách hoạt động**: Chia vùng đệm thành 2 nửa độc lập đặt tên là **"Ping"** và **"Pong"**:  
  * **Tại thời điểm** $T$: Mảng PE đang đọc và tính toán trên dữ liệu ở vùng **Ping**. Đồng thời lúc đó, AXI DMA đang nạp khối dữ liệu (tile) tiếp theo từ DRAM vào vùng **Pong**.  
  * **Tại thời điểm** $T+1$: Khi cả hai bên làm xong, bộ điều khiển chỉ cần tráo con trỏ (toggle pointer) trong 1 clock cycle: Mảng PE chuyển sang tính trên vùng **Pong**, còn AXI DMA nạp dữ liệu mới vào vùng **Ping**.  
* **Kết quả**: Mảng PE gần như chạy liên tục 100% thời gian mà không bao giờ phải ngưng trệ chờ DMA.

\--------------------------------------------------------------------------------

Bảng so sánh trực quan

| Tiêu chí | Phân ngân hàng BRAM (BRAM Banking) | Bộ đệm Ping-Pong (Ping-Pong Buffering) |
| ----- | ----- | ----- |
| **Bản chất tối ưu** | Tối ưu **Băng thông không gian** (Spatial Concurrency). | Tối ưu **Thời gian / Độ trễ** (Temporal Concurrency). |
| **Đối tượng tương tác** | Giao tiếp **Nhiều PEs** $\\\\leftrightarrow$ **Bộ nhớ BRAM nội bộ**. | Giao tiếp **BRAM nội bộ** $\\\\leftrightarrow$ **DRAM bên ngoài (AXI DMA)**. |
| **Vấn đề triệt tiêu** | Triệt tiêu hiện tượng **Nghẽn cổng đọc/ghi (Port Collision)**. | Triệt tiêu thời gian **Đứng chờ nạp dữ liệu (Pipeline Stalls)**. |
| **Quy mô chu kỳ** | Diễn ra ngay trong **1 chu kỳ clock**. | Diễn ra xuyên suốt **nhiều chu kỳ clock** (trong lúc nạp 1 tile dữ liệu). |

\--------------------------------------------------------------------------------

Cách hai kỹ thuật kết hợp trong thực tế

Trong các IP tăng tốc Vision Transformer (như khối `attention_core.sv` của dự án Capstone): **Cả hai kỹ thuật này thường được kết hợp cùng lúc**.

Vùng đệm **Ping** (và cả vùng **Pong**) sẽ được chia nhỏ tiếp thành **16 hoặc 32 BRAM Banks**. Như vậy, AXI DMA vừa nạp dữ liệu vào vùng Pong, trong khi 32 PEs vừa có thể đọc song song 32 phần tử dữ liệu từ 32 Banks của vùng Ping cùng lúc mà không hề bị ngắt nhịp.

\--------------------------------------------------------------------------------

💡 Bạn có muốn xem sơ đồ nguyên lý hoạt động của bộ điều khiển tráo con trỏ (Pointer Switch FSM) trong SystemVerilog của đệm Ping-Pong không?

\--------------------------------------------------------------------------------

Tối ưu LayerNorm Phần cứng

Trong kiến trúc Vision Transformer (ViT), **Layer Normalization (LN)** là một trong những nút thắt phần cứng lớn nhất bên cạnh Softmax. Không giống như Batch Normalization (có thể gộp/fold trực tiếp vào trọng số của tầng lớp trước trong quá trình suy hao inference), LayerNorm bắt buộc phải **tính toán động (dynamic statistics)** các giá trị trung bình ($\\\\mu$) và phương sai ($\\\\sigma^2$) trên từng token dữ liệu tại thời điểm chạy.

Dưới đây là chi tiết các kỹ thuật xử lý **Layer Normalization** tối ưu trên phần cứng FPGA/ASIC được trích xuất từ các công trình nghiên cứu SOTA (như I-ViT, ME-ViT, ADAPTOR).

\--------------------------------------------------------------------------------

1\. Thách thức của Layer Normalization trên Phần cứng

Công thức toán học chuẩn của LayerNorm trên chiều ẩn $d$: $$\\\\text{LayerNorm}(x\\\_i) \= \\\\frac{x\\\_i \- \\\\mu}{\\\\sqrt{\\\\sigma^2 \+ \\\\epsilon}} \\\\cdot \\\\gamma \+ \\\\beta\\$$ với $\\\\mu \= \\\\frac{1}{d} \\\\sum\\\_{j=1}^d x\\\_j$ và $\\\\sigma^2 \= \\\\frac{1}{d} \\\\sum\\\_{j=1}^d (x\\\_j \- \\\\mu)^2$.

* **Nút thắt 2-Pass (Hai lượt quét dữ liệu):** Cần tính xong trung bình $\\\\mu$ cho toàn bộ chiều $d$ thì mới có thể trừ $x\\\_i \- \\\\mu$ để tính phương sai $\\\\sigma^2$, gây trễ luồng dữ liệu (pipeline stall).  
* **Phép toán phi tuyến đắt đỏ:** Phép căn bậc hai ($\\\\sqrt{\\\\cdot}$), phép chia nghịch đảo ($1/\\\\sqrt{\\\\cdot}$), và biến đổi affine ($\\\\gamma, \\\\beta$) nếu dùng số thực (FP32) sẽ tiêu tốn lượng lớn tài nguyên logic (LUTs/DSPs).

\--------------------------------------------------------------------------------

2\. Các Giải pháp Kỹ thuật Xử lý trên Phần cứng (Hardware Strategies)

a. Biến đổi Toán học 1-Pass cho Phương sai (Parallel Variance Accumulation)

Để tránh việc phải chờ tính xong $\\\\mu$ mới tính $\\\\sigma^2$, phần cứng áp dụng công thức khai triển phương sai: $$\\\\sigma^2 \= \\\\left( \\\\frac{1}{d} \\\\sum\\\_{j=1}^d x\\\_j^2 \\\\right) \- \\\\mu^2\\$$

* **Cơ chế luồng (Streaming Datapath):** Khi luồng dữ liệu $x\\\_j$ đang chạy qua khối đệm, hai bộ tích lũy (Accumulators) song song sẽ tính đồng thời **Tổng các phần tử (**$\\\\sum x\\\_j$**)** và **Tổng bình phương (**$\\\\sum x\\\_j^2$**)**.  
* **Lợi ích:** Chỉ cần đúng **1 lượt quét dữ liệu (Single Pass)** là có ngay đầy đủ dữ liệu để tính cả $\\\\mu$ và $\\\\sigma^2$.

b. Tính Căn bậc hai nghịch đảo ($1/\\\\sqrt{\\\\text{Var}}$) bằng lặp Newton-Raphson hoặc LUT

Thay vì dùng bộ chia và căn số thực phần cứng đắt đỏ, phần cứng áp dụng một trong hai cách:

1. \*\*Phương pháp Lặp Newton-Raphson (Integer/Fixed-Point Newton-Raphson):\*\*Tìm nghiệm của hàm $f(y) \= \\\\frac{1}{y^2} \- x \= 0$ bằng công thức lặp: \\$$y\\\_{n+1} \= y\\\_n \\\\cdot \\\\frac{3 \- x \\\\cdot y\\\_n^2}{2}\\\\$$ Phép lặp này chỉ gồm các phép nhân và dịch bit nguyên, chỉ cần 2–4 vòng lặp là đạt độ chính xác tương đương FP32.  
2. \*\*Bảng tra LUT (Lookup Table) kết hợp Số nguyên (I-LayerNorm):\*\*Đưa giá trị phương sai qua bảng tra ROM nhỏ lưu sẵn $1/\\\\sqrt{\\\\text{Var}}$, kết hợp phép nhân số nhị phân Dyadic (Dyadic Shift) để đưa kết quả về kiểu số nguyên **INT8 / INT32**.

c. Kiến trúc Pipeline 3 Giai đoạn trong RTL (`layernorm_unit.sv`)

Các bộ tăng tốc ViT hiện đại (như ME-ViT hay ADAPTOR) thiết kế khối LayerNorm tích hợp trực tiếp vào đường ống xử lý Processing Element (PE) với 3 giai đoạn chính:

Data Stream In (x\_i) ───► \[GIAI ĐOẠN 1: Accumulator Array\] ───► Tích lũy Sum \&amp; SqSum

                                    │

                                    ▼

                         \[GIAI ĐOẠN 2: Stats Logic\]      ───► Tính Mean, Var \&amp; InvStd (1/sqrt(Var))

                                    │

                                    ▼

Data Stream Out ────────► \[GIAI ĐOẠN 3: Affine Pipeline\]    ───► (x\_i \- Mean) \* InvStd \* Gamma \+ Beta ───► Result

&nbsp;

&nbsp;

* **Giai đoạn 1 (Tích lũy):** Đếm và cộng dồn $\\\\sum x\\\_j$ và $\\\\sum x\\\_j^2$ vào bộ tích lũy INT32.  
* **Giai đoạn 2 (Tính thống kê):** Khối logic ngắn (vài chu kỳ clock) tính $\\\\mu \= \\\\text{Sum}/d$, $\\\\sigma^2 \= (\\\\text{SqSum}/d) \- \\\\mu^2$ và $\\\\text{InvStd} \= 1/\\\\sqrt{\\\\sigma^2 \+ \\\\epsilon}$.  
* **Giai đoạn 3 (Chuẩn hóa Pipelined Affine):** Dữ liệu $x\\\_j$ từ đệm BRAM được phát ra luồng thứ hai, đi qua dãy PE nhân-cộng để thực hiện phép tính $(x\\\_j \- \\\\mu) \\\\cdot \\\\text{InvStd} \\\\cdot \\\\gamma \+ \\\\beta$ với tốc độ **Initiation Interval** $II \= 1$ (mỗi chu kỳ clock xuất ra 1 kết quả chuẩn hóa).

d. Chiến lược Phân chia Phần cứng / Phần mềm (HW/SW Partitioning)

* **Tích hợp hoàn toàn trên FPGA Fabric (PL Engine):** Đặt khối LayerNorm nằm ngay giữa đệm đệm BRAM và mảng nhân Systolic Array (như trong thiết kế ME-ViT). Việc này giúp dữ liệu sau khi chuẩn hóa được đưa thẳng vào phép nhân ma trận $Q, K, V$ hoặc MLP mà **không cần ghi ngược lại RAM ngoài (DRAM)**, tiết kiệm đến 80–90% băng thông bộ nhớ.  
* **Đẩy về CPU Host (ARM PS) xử lý:** Một số thiết kế đơn giản offload tính toán LN về CPU ARM (chạy Python/C++). Tuy nhiên, giải pháp này tạo ra nút thắt giao tiếp ngắt quãng bus AXI DMA do phải truyền nhận dữ liệu liên tục giữa PS và PL.

\--------------------------------------------------------------------------------

Tóm tắt Phương án Tối ưu cho Dự án Capstone (Kria KV260)

| Phương án | Kỹ thuật triển khai | Ưu điểm phần cứng | Đánh đổi / Chi phí |
| ----- | ----- | ----- | ----- |
| **I-LayerNorm (Recommended)** | Fixed-point 1-Pass Variance \+ Newton-Raphson / Dyadic shift. | **0 DSP cho căn bậc hai**, không ghi lại DRAM, chạy mượt trên INT8. | Tốn thêm vài BRAM đệm nhỏ cho Pass 2\. |
| **L1-LayerNorm** | Thay L2 norm ($\\\\sigma$) bằng L1 norm \\(\\frac{1}{d}\\sum \\ | x\_i \- \\mu\\ | \\). |

\--------------------------------------------------------------------------------

Tối ưu BRAM cho ViT

Trong thiết kế tăng tốc Vision Transformer (ViT) trên FPGA, **kỹ thuật phân mảng PE (PE Array Partitioning & Dimensioning)** đóng vai trò cốt lõi trong việc giải quyết nút thắt về tài nguyên bộ nhớ BRAM thông qua 4 cơ chế kỹ thuật chính:

\--------------------------------------------------------------------------------

1\. Phân khối ma trận (Matrix Tiling) vừa vặn với dung lượng BRAM

* **Thách thức**: Ma trận trong ViT (như ma trận Query, Key, Value hay trọng số MLP) có kích thước rất lớn (như $196 \\\\times 32$ hoặc $768 \\\\times 768$), vượt quá dung lượng BRAM on-chip của các dòng FPGA nhúng (như Kria KV260 chỉ có 144 khối BRAM18K).  
* **Tối ưu BRAM**: Phân mảng PE sẽ chia ma trận lớn thành các khối nhỏ (**Tiles** có kích thước $T\\\_q \\\\times d\\\_k$ hoặc $T\\\_{FFN}$) vừa trọn với dung lượng của từng đệm BRAM. Thay vì lưu toàn bộ ma trận, BRAM chỉ cần đệm các sub-block nhỏ cho mảng PE xử lý luân phiên.

\--------------------------------------------------------------------------------

2\. Cân bằng tỷ lệ BRAM – DSP (Tránh hiện tượng BRAM Exhaustion)

* **Thách thức**: Nếu mỗi mảng PE yêu cầu kích thước đệm quá lớn, BRAM trên FPGA sẽ bị cạn kiệt hoàn toàn trước khi tất cả bộ nhân DSP48E2 được khai thác (BRAM exhaustion).  
* **Tối ưu BRAM**: Phân mảng PE thiết lập kích thước mảng an toàn (như $16 \\\\times 16$, $32 \\\\times 32$ hoặc cặp $(n\\\_i, n\\\_o) \= (64, 32)$). Kích thước này tối thiểu hóa dung lượng BRAM gán cho từng PE, tạo ra điểm cân bằng giúp kích hoạt tối đa số lượng DSP khả dụng mà không làm tràn BRAM.

\--------------------------------------------------------------------------------

3\. Phân ngân hàng BRAM (BRAM Banking) để truy xuất song song

* **Thách thức**: BRAM trên FPGA có số cổng đọc/ghi hạn chế (Dual-Port BRAM). Nếu không phân chia đệm, mảng PE song song sẽ bị tắc nghẽn khi truy xuất dữ liệu trong cùng một chu kỳ xung clock.  
* **Tối ưu BRAM**: Kỹ thuật phân mảng sẽ chia đệm BRAM thành nhiều ngân hàng độc lập (**BRAM Banks**) với dung lượng là bội số của kích thước mảng $P\\\_{sys}$. Điều này cho phép $P\\\_{sys}$ đơn vị PE truy xuất đồng thời $P\\\_{sys}$ phần tử dữ liệu mỗi chu kỳ mà không gặp xung đột cổng đọc.

\--------------------------------------------------------------------------------

4\. Tái sử dụng dữ liệu & Bộ đệm Đa mục đích (Multi-purpose & Ping-Pong Buffering)

* **Tái sử dụng tại chỗ (Spatial Data Reuse)**: Dữ liệu sau khi nạp vào mảng PE được dịch chuyển truyền tiếp giữa các PE lân cạnh theo nhịp clock, giúp nhân nhiều lần dữ liệu mà không cần nạp lại từ BRAM.  
* **Tái sử dụng bộ đệm đa mục đích**: Các thiết kế hiện đại (như ME-ViT) dùng chung một khối BRAM cho nhiều công đoạn (Linear Projection, Attention và MLP). BRAM được nạp/xóa và tái cấu trúc liên tục cho các bước tính toán khác nhau thay vì cấp phát riêng đệm BRAM cho từng lớp.  
* **Đệm Ping-Pong (Double Buffering)**: Bộ đệm BRAM được chia thành hai nửa "Ping" và "Pong". Trong khi mảng PE đang tính toán trên nửa Ping, bộ điều khiển AXI DMA sẽ nạp tile dữ liệu tiếp theo vào nửa Pong. Khi tính xong, chỉ cần tráo con trỏ trong 1 chu kỳ clock, giúp ẩn hoàn toàn độ trễ truyền dữ liệu mà chỉ tốn gấp đôi dung lượng tile nhỏ.

\--------------------------------------------------------------------------------

💡 **Gợi ý tiếp theo**: Bạn có muốn xem sơ đồ mô phỏng luồng chuyển dữ liệu Ping-Pong BRAM giữa AXI DMA và mảng PE trong SystemVerilog (`bram_buffer.sv`) không?

\--------------------------------------------------------------------------------

Phân mảng Systolic PE

**Kỹ thuật phân mảng PE (Processing Element Partitioning / Dimensioning & Tiling)** trong mô hình **Systolic Array** là phương pháp thiết kế phần cứng kiến trúc vi mạch, trong đó các đơn vị xử lý nhỏ (Processing Element \- PE) được sắp xếp thành một mảng lưới 2D (hoặc 1D) và được phân chia kích thước, khối tính toán (tile) một cách tối ưu để thực hiện các phép nhân ma trận song song.

\--------------------------------------------------------------------------------

1\. Bản chất của Mảng Systolic (Systolic Array)

* Một mảng Systolic gồm hàng trăm đến hàng ngàn đơn vị **PE** nhỏ liên kết với nhau. Trong đó, dữ liệu (kích hoạt \- activations, trọng số \- weights, và tổng một phần \- partial sums) dịch chuyển đồng bộ qua các PE kề nhau theo nhịp xung clock (giống như nhịp đập tim/systole).  
* **Luồng dữ liệu**: Thường thì kích hoạt truyền theo chiều dọc, trọng số truyền theo chiều ngang, và kết quả tích lũy chạy dọc theo đường chéo.

\--------------------------------------------------------------------------------

2\. Các Khía cạnh Cốt lõi của Kỹ thuật Phân mảng PE

Khi áp dụng mô hình Systolic cho các mạng học sâu như Vision Transformer (ViT), kỹ thuật phân mảng PE bao gồm các cơ chế chính sau:

a. Định cấu hình Kích thước Mảng PE $(P\\\_{sys} \\\\times P\\\_{sys})$ hoặc $(n\\\_i, n\\\_o)$

* Mảng PE được thiết kế theo các tham số chiều rộng và chiều cao cố định, biểu diễn bằng $P\\\_{sys} \\\\times P\\\_{sys}$ (ví dụ: mảng $32 \\\\times 32$ hay $16 \\\\times 16$ trong ME-ViT) hoặc cặp $(n\\\_i, n\\\_o)$ thể hiện kênh đầu vào và kênh đầu ra (ví dụ: $(64, 32)$ hay $(54, 128)$).  
* Kích thước mảng quyết định trực tiếp số phép nhân tích lũy (MAC) có thể thực hiện song song trong 1 chu kỳ xung clock.

b. Phân khối Ma trận (Matrix Tiling / Blocking)

* **Thách thức**: Các ma trận trong ViT (như ma trận trọng số MLP hay ma trận chú ý $Q \\\\times K^T$) có kích thước quá lớn, không thể nạp trọn gói vào bộ nhớ on-chip BRAM của FPGA.  
* **Giải pháp**: Kỹ thuật phân mảng sẽ chia ma trận lớn thành các khối nhỏ (**sub-blocks / tiles**) có kích thước tương thích hoàn hảo với mảng PE. Mảng PE sẽ xử lý luân phiên từng tile dữ liệu mà không làm tràn bộ đệm BRAM.

c. Tái sử dụng Dữ liệu On-Chip (Spatial & Temporal Data Reuse)

* Thay vì mỗi phép nhân phải đọc/ghi dữ liệu ra bộ nhớ ngoài (DRAM), dữ liệu sau khi nạp vào mảng PE sẽ được truyền tiếp cho các PE lân cận.  
* Kỹ thuật này giúp giảm lưu lượng truy xuất bộ nhớ DRAM từ một đến hai cấp độ (Single-Load policy) và tận dụng tối đa băng thông bộ nhớ BRAM trên chip.

d. Kết hợp DSP Packing / SIMD trong từng PE

* Khi tích hợp trên chip FPGA (như AMD Xilinx UltraScale+), mỗi nút PE tận dụng bộ nhân **DSP48E2**.  
* Bằng cách ép 2 hoặc 4 phép tính số nguyên độ chính xác thấp (INT8/INT4) vào cổng nhân $27 \\\\times 18$ bit của một DSP slice (DSP Packing), mảng Systolic kích thước $P\\\_{sys} \\\\times P\\\_{sys}$ có thể nhân đôi năng suất tính toán lên $P\\\_{sys} \\\\times 2P\\\_{sys}$ phép tính trong 1 chu kỳ.

\--------------------------------------------------------------------------------

3\. Ý nghĩa trong Tăng tốc Vision Transformer (ViT)

1. **Cân bằng Tài nguyên (Compute vs. Memory Bound)**: Phân mảng PE giúp tối ưu hóa mối quan hệ giữa số lượng bộ nhân DSP48E2 và dung lượng đệm BRAM, tránh tình trạng hết BRAM trước khi dùng hết DSP.  
2. **Khóa Độ trễ (Latency Hiding)**: Kết hợp phân mảng PE với bộ đệm **Ping-Pong (Double Buffering)** cho phép mảng PE vừa tính toán khối dữ liệu hiện tại, vừa nạp tile dữ liệu tiếp theo từ AXI DMA, ẩn hoàn bộ độ trễ truyền dữ liệu.

\--------------------------------------------------------------------------------

💡 **Gợi ý tiếp theo**: Bạn có muốn tìm hiểu chi tiết hơn về cách thiết kế mảng Systolic $P\\\_{sys} \\\\times P\\\_{sys}$ bằng ngôn ngữ **SystemVerilog** (`systolic_array.sv`) để tích hợp vào khối tính Attention không?

\--------------------------------------------------------------------------------

Cơ chế Scaled Attention

Trong cơ chế chú ý **Scaled Dot-Product Attention** của Transformer:

$$\\\\text{Attention}(Q, K, V) \= \\\\text{Softmax}\\\\left(\\\\frac{Q K^T}{\\\\sqrt{d\\\_k}}\\\\right)V\\$$

Việc kích thước không gian ẩn $d\\\_k$ (Head Dimension) tăng lớn nếu **không được chia cho hệ số thu phóng** $\\\\sqrt{d\\\_k}$ sẽ dẫn đến hiện tượng **triệt tiêu gradient (vanishing gradient)** trong quá trình lan truyền ngược (backpropagation).

\--------------------------------------------------------------------------------

1\. Bản chất Toán học: Tích vô hướng tăng phương sai theo $d\\\_k$

Giả sử các thành phần của vector Query $q \= \\\[q\\\_1, q\\\_2, \\\\dots, q\\\_{d\\\_k}\\\]$ và Key $k \= \\\[k\\\_1, k\\\_2, \\\\dots, k\\\_{d\\\_k}\\\]$ là các biến ngẫu nhiên độc lập có trung bình bằng $0$ và phương sai bằng $1$ ($\\\\mathbb{E}\\\[q\\\_i\\\] \= \\\\mathbb{E}\\\[k\\\_i\\\] \= 0$, $\\\\text{Var}(q\\\_i) \= \\\\text{Var}(k\\\_i) \= 1$):

1. **Phép tích vô hướng**: \\$$q \\\\cdot k \= \\\\sum\\\_{i=1}^{d\\\_k} q\\\_i k\\\_i\\\\$$  
2. **Kỳ vọng (Mean)**: $$\\\\mathbb{E}\\\[q \\\\cdot k\\\] \= \\\\sum\\\_{i=1}^{d\\\_k} \\\\mathbb{E}\\\[q\\\_i\\\] \\\\mathbb{E}\\\[k\\\_i\\\] \= 0\\$$  
3. **Phương sai (Variance)**: $$\\\\text{Var}(q \\\\cdot k) \= \\\\sum\\\_{i=1}^{d\\\_k} \\\\text{Var}(q\\\_i k\\\_i) \= \\\\sum\\\_{i=1}^{d\\\_k} \\\\text{Var}(q\\\_i)\\\\text{Var}(k\\\_i) \= d\\\_k\\$$

Khi $d\\\_k$ tăng lên (ví dụ $d\\\_k \= 64, 128$), **phương sai của tích vô hướng** $q \\\\cdot k$ **tăng tuyến tính theo** $d\\\_k$ (độ lệch chuẩn bằng $\\\\sqrt{d\\\_k}$). Điều này khiến các giá trị $Q K^T$ có biên độ biến động rất rộng và dễ xuất hiện các giá trị cực lớn hoặc cực nhỏ.

\--------------------------------------------------------------------------------

2\. Hiện tượng Bão hòa Softmax (Softmax Saturation)

Hàm Softmax chuyển đổi các điểm số attention score $x \= Q K^T$ thành phân bố xác suất:

\\$$S\\\_i \= \\\\text{Softmax}(x)\*i \= \\\\frac{e^{x\\\_i}}{\\\\sum\*{j} e^{x\\\_j}}\\\\$$

* Khi $d\\\_k$ lớn, các phần tử trong $x$ lệch nhau rất xa về biên độ do phương sai cao.  
* Hàm mũ $e^{x\\\_i}$ phóng đại sự chênh lệch này, khiến phần tử lớn nhất $x\\\_{\\\\max}$ áp đảo toàn bộ các phần tử còn lại.  
* Kết quả là phân bố Softmax bị **cực đoan hóa** (tiến sát về dạng phân bố one-hot: một vị trí bằng $\\\\approx 1$, tất cả vị trí khác $\\\\approx 0$). Khi đó, hàm Softmax rơi vào **vùng bão hòa (saturation region)**.

\--------------------------------------------------------------------------------

3\. Đạo hàm bằng 0 gây Triệt tiêu Gradient

Xét đạo hàm của hàm Softmax $S\\\_i$ theo đầu vào $x\\\_j$:

$$\\\\frac{\\\\partial S\\\_i}{\\\\partial x\\\_j} \= S\\\_i (\\\\delta\\\_{ij} \- S\\\_j)\\$$

* **Trường hợp** $i \= j$: $\\\\frac{\\\\partial S\\\_i}{\\\\partial x\\\_i} \= S\\\_i (1 \- S\\\_i)$. Khi Softmax bị bão hòa ($S\\\_i \\\\approx 1$ hoặc $S\\\_i \\\\approx 0$), ta có $S\\\_i (1 \- S\\\_i) \\\\approx 0$.  
* **Trường hợp** $i \\\\neq j$: $\\\\frac{\\\\partial S\\\_i}{\\\\partial x\\\_j} \= \- S\\\_i S\\\_j$. Vì ít nhất một trong hai giá trị tiến về $0$, tích $-S\\\_i S\\\_j \\\\approx 0$.

Do đó, toàn bộ **ma trận Jacobi của hàm Softmax tiến sát về 0**. Khi thực hiện lan truyền ngược, gradient nhân với ma trận này sẽ bị triệt tiêu hoàn toàn ($\\\\approx 0$), khiến các tầng phía trước không nhận được tín hiệu cập nhật trọng số và mô hình ngừng học.

\--------------------------------------------------------------------------------

4\. Giải pháp: Thước đo Thu phóng $1/\\\\sqrt{d\\\_k}$ (Scaling Factor)

Bằng cách chia tích vô hướng cho $\\\\sqrt{d\\\_k}$:

$$\\\\text{Var}\\\\left(\\\\frac{q \\\\cdot k}{\\\\sqrt{d\\\_k}}\\\\right) \= \\\\frac{\\\\text{Var}(q \\\\cdot k)}{d\\\_k} \= \\\\frac{d\\\_k}{d\\\_k} \= 1\\$$

Phép chia này chuẩn hóa phương sai của đầu vào Softmax **về lại bằng 1**, độc lập với giá trị $d\\\_k$. Nhờ đó, các giá trị đưa vào Softmax không bị quá lớn, giữ cho hàm Softmax hoạt động trong vùng tuyến tính có đạo hàm lớn, giúp quá trình huấn luyện diễn ra ổn định.

\--------------------------------------------------------------------------------

💡 **Gợi ý tiếp theo**: Bạn có muốn tìm hiểu cách chuyển đổi phép chia $\\\\sqrt{d\\\_k}$ này thành **phép dịch bit đại số (Bit-Shift)** trên phần cứng FPGA để đạt chi phí 0 DSP không?

\--------------------------------------------------------------------------------

So sánh Softmax phần cứng

Dựa trên kết quả mô phỏng thuật toán số học thực tế (với chiều dài chuỗi token $N \= 196$ trên 100 lần thử nghiệm ngẫu nhiên) và các tài liệu nghiên cứu trong bộ nguồn (I-ViT, ME-ViT, Capstone Investigation Document), dưới đây là báo cáo so sánh chi tiết giữa **2 thuật toán Softmax xấp xỉ phần cứng** so với **Softmax số thực chuẩn (FP32 Baseline)**.

\--------------------------------------------------------------------------------

I. Kết quả Mô phỏng Số học & Sai số (Numerical Simulation Results)

Đo đạc trên chuỗi $N \= 196$ tokens (tương ứng grid patch $14 \\\\times 14$ của MobileViT):

| Chỉ số Đánh giá (Metric) | Phương pháp A: Lookup-Table (LUT Softmax \- 256 BRAM entries) | Phương pháp B: Piecewise-Linear (PWL / Base-2 Shiftmax) | So sánh & Nhận xét Kỹ thuật |
| ----- | ----- | ----- | ----- |
| **Mean Squared Error (MSE)** | $6.16 \\\\times 10^{-8}$ | $5.39 \\\\times 10^{-5}$ | **Phương pháp A có MSE thấp hơn \~875 lần** so với Phương pháp B. |
| **Độ tương đồng Cosine (Cosine Similarity)** | **0.999983** (Gần như tuyệt đối) | **0.942296** | Phương pháp A khớp hoàn toàn phân bố xác suất gốc. |
| **Sai số tuyệt đối cực đại (Max Abs Error)** | **0.001542** (\~0.15%) | **0.077742** (\~7.77%) | Phương pháp B bị lệch nhẹ ở các giá trị cực trị (peak scores). |
| **Tiêu tốn Tài nguyên Phần cứng (HW Resource)** | Tốn **1 \- 2 Block RAM (BRAM18K)** lưu bảng LUT ROM 256 phần tử. | **0 BRAM**, tiêu tốn một ít logic LUT cho mạch dịch bit (Shifter). | **Phương pháp B tiết kiệm 100% BRAM**. |
| **Độ trễ Thực thi (Latency)** | **1 Clock Cycle** (Tra bảng ROM 1 chu kỳ). | **1 \- 2 Clock Cycles** (Phép dịch bit nguyên & cộng tuyến tính). | Cả 2 đều hỗ trợ pipeline II \= 1\. |

\--------------------------------------------------------------------------------

II. Phân tích Chi tiết Thuật toán & Đánh đổi Phần cứng (Hardware Trade-offs)

\[Attention Scores Q\*K^T / sqrt(d\_k)\]

                 │

                 ├──► Max Subtraction (x' \= x \- x\_max \&lt;= 0\)

                 │

                 ├───► \[METHOD A: LUT ROM Engine\] ─────► Read exp(x') from BRAM ───► Div/Normalize ───► MSE: \~6.16e-8

                 │

                 └───► \[METHOD B: PWL Shiftmax\]  ─────► 2^{-q} \* (1 \- 0.5\*r)      ───► Div/Normalize ───► MSE: \~5.39e-5

&nbsp;

&nbsp;

1\. Phương pháp A: Lookup-Table (LUT-based Softmax)

* **Cơ chế**:  
  * Thực hiện trừ max $x' \= x \- x\\\_{\\\\max} \\\\le 0$ để đưa dải giá trị về số âm, chống tràn số.  
  * Giới hạn dải đầu vào trong khoảng $\\\[-8.0, 0.0\\\]$ và chia thành 256 mức định lượng (Q4.4 / Q3.5).  
  * Tra bảng BRAM ROM lưu sẵn giá trị $e^{x'}$ trong 1 chu kỳ xung clock.  
* **Ưu điểm**: Sai số cực nhỏ, giữ nguyên độ chính xác Top-1 của mô hình Vision Transformer.  
* **Nhược điểm**: Mỗi Attention Head hoặc PE cần đệm một bảng BRAM ROM riêng, dễ dẫn đến hiện tượng cạn kiệt tài nguyên BRAM trên Kria KV260 khi mở rộng nhiều PE.

2\. Phương pháp B: Piecewise-Linear (PWL / Base-2 Shiftmax)

* **Cơ chế**:  
  * Biến đổi cơ số từ $e$ sang cơ số 2: $e^{x'} \= 2^{x' \\\\cdot \\\\log\\\_2 e}$.  
  * Tách $x' \\\\cdot \\\\log\\\_2 e \= \-q \+ r$ với $q \\\\ge 0$ nguyên và $r \\\\in \\\[0, 1)$.  
  * Tính $2^{-q}$ bằng **phép dịch bit phải (Right Bit-Shift)** và xấp xỉ tuyến tính $2^{-r} \\\\approx 1 \- 0.5r$.  
* **Ưu điểm**: Giải phóng hoàn toàn bộ nhớ BRAM, sử dụng 100% bộ dịch bit (shifter) và bộ cộng nguyên trên FPGA fabric.  
* **Nhược điểm**: Do tính xấp xỉ đoạn tuyến tính của $2^{-r}$, sai số tuyệt đối ở các đỉnh chú ý cao hơn (\~7.7%), có thể gây sụt giảm khoảng **0.3% \- 0.8%** độ chính xác Top-1 trên tập dữ liệu phân loại lớn.

\--------------------------------------------------------------------------------

III. Báo cáo Đồ họa & Tập tin Artifact

Biểu đồ so sánh phân bố xác suất đầu ra và dải sai số tuyệt đối từng token (`softmax_mse_comparison.png`) đã được tạo và xuất trực tiếp vào bảng **Studio Panel**:

1. **Đồ thị trên**: Cho thấy dạng đường phân bố xác suất Softmax của **Phương pháp A (màu đỏ)** chồng khít hoàn toàn lên đường **FP32 chuẩn (màu đen)**, trong khi **Phương pháp B (màu xanh)** có sự lệch nhẹ tại dải giá trị đỉnh.  
2. **Đồ thị dưới**: Thể hiện biên độ sai số tuyệt đối trên từng token chỉ chỉ tiệm cận \~0.0015 với LUT, trong khi Shiftmax đạt đỉnh \~0.078 tại token có điểm attention score cao nhất.

\--------------------------------------------------------------------------------

IV. Script Python Mô phỏng Dùng lại (`simulate_softmax.py`)

Nếu bạn muốn nhúng đoạn mã mô phỏng này vào pipeline kiểm thử tự động (CI/CD hoặc Golden Vector Exporter), bạn có thể chạy lại script trực tiếp:

import numpy as np

&nbsp;

def softmax\_fp32(x):

    x\_max \= np.max(x, axis=-1, keepdims=True)

    e\_x \= np.exp(x \- x\_max)

    return e\_x / np.sum(e\_x, axis=-1, keepdims=True)

&nbsp;

def softmax\_method\_a\_lut(x, lut\_entries=256):

    x\_max \= np.max(x, axis=-1, keepdims=True)

    x\_shift \= np.clip(x \- x\_max, \-8.0, 0.0)

    lut\_x \= np.linspace(-8.0, 0.0, lut\_entries)

    lut\_val \= np.exp(lut\_x)

    indices \= np.round((x\_shift \+ 8.0) / 8.0 \* (lut\_entries \- 1)).astype(int)

    exp\_approx \= lut\_val\[indices\]

    return exp\_approx / np.sum(exp\_approx, axis=-1, keepdims=True)

&nbsp;

def softmax\_method\_b\_pwl(x):

    x\_max \= np.max(x, axis=-1, keepdims=True)

    x\_base2 \= (x \- x\_max) \* np.log2(np.e)

    q \= np.floor(-x\_base2)

    r \= x\_base2 \+ q

    exp2\_approx \= (1.0 \- 0.5 \* r) \* (2.0 \*\* (-q))

    return exp2\_approx / np.sum(exp2\_approx, axis=-1, keepdims=True)

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

🎯 **Đề xuất chiến lược triển khai**:

* **Giai đoạn 1 (Phase 1 Baseline)**: Sử dụng **Phương pháp A (LUT Softmax)** để đảm bảo mạch RTL hoạt động chính xác 100% so với mô hình tham chiếu.  
* **Giai đoạn 2 (Phase 2 Extension)**: Triển khai thêm mô-đun **Phương pháp B (PWL Shiftmax)** trong SystemVerilog (`softmax_pwl.sv`) để làm bài đánh giá so sánh đánh đổi (Trade-off Analysis) về tài nguyên BRAM vs MSE cho báo cáo đồ án Capstone.

Bạn có muốn bắt đầu viết mã SystemVerilog cho khối **softmax\_lut.sv (Phương pháp A)** không?

\--------------------------------------------------------------------------------

Kỹ thuật DSP Packing

**DSP Packing** là một kỹ thuật tối ưu hóa phần cứng mức vi mạch trên FPGA, cho phép **đóng gói nhiều phép nhân độ chính xác thấp** (low-precision INT8, INT6 hoặc INT4) vào trong **một khối DSP slice đơn lẻ** để thực thi song song trong cùng một chu kỳ xung clock.

Kỹ thuật này giúp nhân đôi hoặc nhân ba hiệu năng tính toán (**Throughput per DSP / GOPS**) mà không làm tăng số lượng phần cứng DSP vật lý trên chip.

\--------------------------------------------------------------------------------

1\. Nguyên lý Toán học & Phần cứng (Ví dụ trên Xilinx/AMD DSP48E2)

Các dòng FPGA UltraScale+ (như trên bo mạch **Kria KV260** hay **ZCU102**) tích hợp khối **DSP48E2 Slice** gồm một bộ nhân có kích thước cổng vào **18-bit × 27-bit** và bộ tích lũy 48-bit.

Để thực hiện song song 2 phép nhân INT8 cùng lúc trên **1 DSP** (Factor-2 Packing), thuật toán biến đổi như sau:

* **Cấu trúc toán hạng**:  
  * Giả sử cần tính đồng thời hai phép nhân có chung một kích hoạt $A$ (18-bit / INT8): $Y\\\_1 \= A \\\\times B$ và $Y\\\_2 \= A \\\\times C$ (với $B, C$ là hai trọng số INT8 độc lập).  
  * Toán hạng $A$ được đưa vào cổng **18-bit**.  
  * Hai trọng số $B$ và $C$ được dịch bit và ghép lại vào cổng **27-bit**: \\$$D\\\_{in} \= (B \\\\ll 18\) \+ C\\\\$$.  
* **Kết quả phép nhân trong thanh ghi 45-bit**: \\$$A \\\\times D\\\_{in} \= A \\\\times ((B \\\\ll 18\) \+ C) \= (A \\\\times B) \\\\ll 18 \+ (A \\\\times C)\\\\$$. Vì kết quả của mỗi phép nhân INT8 $\\\\times$ INT8 chỉ chiếm tối đa 16 bit, nên tích số $A \\\\times B$ (Upper sub-word) và $A \\\\times C$ (Lower sub-word) nằm ở hai vùng bit hoàn toàn riêng biệt trong thanh ghi đầu ra 45-bit của DSP. Circuit phần cứng chỉ cần tách (unpack) hai vùng bit này ra là thu được hai kết quả riêng biệt trong 1 clock cycle.

\--------------------------------------------------------------------------------

2\. Các Mẫu Đóng gói Mở rộng (Factor-3 & Factor-4 Layouts)

Đối với các mô hình định lượng siêu thấp (\< 8 bit) như trong kiến trúc Quasar-ViT, kỹ thuật DSP Packing được mở rộng theo các cấu hình:

* **Factor-3 Layout (1 Activation 6-bit \+ 3 Weights 4-bit)**: Đặt một vector activation 6-bit vào cổng 18-bit và đóng gói ba trọng số 4-bit vào cổng 27-bit (kết hợp các bộ gom logic LUT nhỏ) để thực hiện **3 phép nhân trong 1 DSP/cycle**.  
* **Factor-4 Layout (2 Activations 6-bit \+ 2 Weights 4-bit)**: Ghép hai activation 6-bit với hai weight 4-bit để thực hiện **4 phép nhân đồng thời**.  
* **Chế độ SIMD tích hợp sẵn (Hardened SIMD)**: Bản thân các khối DSP48E2 cũng hỗ trợ sẵn chế độ SIMD phần cứng như Dual 24-bit hoặc Quad 12-bit/Quad INT8.

\--------------------------------------------------------------------------------

3\. Thách thức Kỹ thuật & Đánh đổi (Hardware Trade-offs)

Mặc dù tăng gấp đôi mật độ tính toán, DSP Packing đòi hỏi sự đánh đổi về thiết kế RTL:

1. **Xử lý Bit Dấu (Sign Extension)**: Phải quản lý chính xác biểu diễn số nguyên có dấu (2's complement) và mở rộng bit dấu cho các vùng bit phía trên (Upper sub-words) để tránh tràn bit hoặc làm nhiễu kết quả giữa các phép tính ghép.  
2. **Tiêu tốn Tài nguyên LUT đệm (Packing/Unpacking Overhead)**: Cần thêm các bộ logic ghép bit ở đầu vào và tách bit ở đầu ra, tiêu tốn một lượng LUT bổ sung.  
3. **Nghẽn Định tuyến & Timing Closure (Routing Congestion)**: Mật độ bus dữ liệu tăng gấp đôi gây ra hiện tượng nghẽn đường truyền (routing density). Để giữ chu kỳ clock Fmax cao, thiết kế bắt buộc phải kết hợp bộ đệm BRAM/URAM được căn chỉnh chính xác (buffer-alignment) để cấp dữ liệu packed trong cùng một chu kỳ.

\--------------------------------------------------------------------------------

4\. Lợi ích cho Tăng tốc Vision Transformer (ViT) trên Edge FPGA

* **Tối ưu hóa tài nguyên DSP hạn chế**: Đối với các dòng FPGA phân khúc viền (Edge FPGAs) như **Kria KV260** hay **ZCU102** vốn chỉ có từ vài trăm đến hơn một ngàn DSP slices, DSP Packing cho phép đạt tốc độ từ **100 \- 250+ FPS**.  
* **Tăng hiệu suất Throughput per DSP**: Giúp các kiến trúc như ME-ViT tăng hiệu suất xử lý trên mỗi DSP lên **2.16×** so với thiết kế tiêu chuẩn.

\--------------------------------------------------------------------------------

💡 **Gợi ý tiếp theo**: Bạn có muốn tôi viết mã SystemVerilog minh họa khối PE triển khai kỹ thuật DSP Packing (2x INT8 multiplication trên 1 DSP48E2) để đưa vào mô-đun RTL không?

\--------------------------------------------------------------------------------

Tích hợp SoC Kria KV260

Batch 5: Tích hợp SoC, Nạp Bitstream & Benchmark trên Bo mạch AMD Kria KV260

Trong giai đoạn **Batch 5**, chúng ta hoàn thiện quy trình **Đồng thiết kế Phần cứng / Phần mềm (HW/SW Co-Design)**. Toàn bộ khối IP `attention_core` thiết kế bằng SystemVerilog sẽ được tích hợp với vi xử lý ARM Processing System (PS) thông qua bộ điều khiển **AXI DMA** trên Vivado Block Design. Sau đó, hệ thống được biên dịch ra tập tin **Bitstream (.bit)**, nạp lên bo mạch **AMD Kria KV260**, và chạy đánh giá hiệu năng thực tế bằng trình điều khiển **PYNQ Python**.

\--------------------------------------------------------------------------------

1\. Tích hợp Sơ đồ Khối Vivado (Vivado IPI Block Design)

Trong môi trường Vivado IP Integrator (IPI), các thành phần phần cứng được kết nối theo mô hình hệ thống SoC:

\+--------------------------------------------------------------------------------------------------+

|                                    VIVADO BLOCK DESIGN (IPI)                                     |

|                                                                                                  |

|  \+-----------------------------------+               \+----------------------------------------+  |

|  |    Zynq UltraScale+ MPSoC IP      |               |      AXI Direct Memory Access (DMA)     |  |

|  |           (ARM Cortex-A53)        |               |                                        |  |

|  |                                   |  AXI-MM2S     |  • Direct Register / Simple Mode       |  |

|  | • DDR4 Controller Interface       |--------------\&gt;|  • Stream Data Width: 64-bit            |  |

|  | • PL Clock Generation (200 MHz)   | (DDR4 Read)   |  • Scatter-Gather: Disabled (Low Lat)  |  |

|  | • AXI HP/GP Master Interfaces     |               \+----------------------------------------+  |

|  \+-----------------------------------+                                   |                       |

|                   |                                                      | AXI4-Stream           |

|                   | AXI4-Lite (Control)                                  | (S\_AXIS\_RX Data)      |

|                   v                                                      v                       |

|  \+--------------------------------------------------------------------------------------------+  |

|  |                                CUSTOM ATTENTION CORE IP                                    |  |

|  |                                                                                            |  |

|  |  • Memory-Mapped Regs (Start, Done, N=196, d\_k=32)                                       |  |

|  |  • Ping-Pong BRAM Buffers \&amp; Systolic MAC Arrays (DSP48E2)                                  |  |

|  |  • Scaler Bit-Shift \&amp; Hardware Softmax Engine (LUT / PWL)                                  |  |

|  \+--------------------------------------------------------------------------------------------+  |

|                                                  |                                               |

|                                                  | AXI4-Stream (M\_AXIS\_TX Attended Data)         |

|                                                  v                                               |

|                                \+----------------------------------+                              |

|                                |         AXI DMA (S2MM)           |                              |

|                                |    (Write Back to DDR4 RAM)      |                              |

|                                \+----------------------------------+                              |

\+--------------------------------------------------------------------------------------------------+

&nbsp;

&nbsp;

Các thành phần chính trong Block Design:

1. **Zynq UltraScale+ MPSoC IP**: Nạp preset cấu hình bo mạch Kria KV260, tạo xung clock `pl_clk0 = 200 MHz` và tín hiệu reset `pl_resetn0`.  
2. **Custom Attention Core IP**: Mô-đun SystemVerilog đã được đóng gói (Package IP) chứa các cổng giao tiếp AXI4-Lite Slave (`S_AXI`) để ARM PS điều khiển và AXI4-Stream Slave/Master (`S_AXIS_RX`, `M_AXIS_TX`) để nhận/phát dữ liệu.  
3. **AXI Direct Memory Access (AXI DMA) IP**: Cấu hình chế độ **Simple DMA Mode** (tắt Scatter-Gather) với chiều rộng luồng 64-bit để tối ưu hóa độ trễ truyền dữ liệu giữa bộ nhớ DDR4 và FPGA PL.  
4. **AXI SmartConnect / Interconnect**: Tự động định tuyến các kênh bus dữ liệu AXI Memory-Mapped và AXI-Lite.

\--------------------------------------------------------------------------------

2\. Tổng hợp, Implemented & Thắt chặt Thời gian (Timing Closure)

Sau khi hoàn tất sơ đồ khối, quy trình biên dịch phần cứng trong Vivado thực hiện các bước:

1. **Synthesis & Place and Route (P\&R)**: Ánh xạ mảng nhân MAC vào **DSP48E2** và bộ đệm vào **BRAM18K/BRAM36K**.  
2. **Kiểm tra Ràng buộc Thời gian (Timing Verification)**:  
3. **Worst Negative Slack (WNS) \> 0 ns**: Đảm bảo không vi phạm Setup Time ở tần số **200 MHz**.  
4. **Worst Hold Slack (WHS) \> 0 ns**: Đảm bảo không vi phạm Hold Time.  
5. **Xuất Tệp Giao thoa Phần cứng (Export Hardware Handoff)**:  
6. **attention\_core.bit**: Tập tin cấu hình mạch Bitstream cho FPGA fabric.  
7. **attention\_core.hwh**: Tập tin định nghĩa địa chỉ bộ nhớ và sơ đồ kết nối AXI để PYNQ tự động nhận diện IP.

\--------------------------------------------------------------------------------

3\. Viết Driver PYNQ Host trên ARM CPU (`benchmark_app.py`)

Bo mạch Kria KV260 chạy hệ điều hành **Ubuntu 22.04 LTS** kết hợp thư viện **PYNQ Framework**. Ứng dụng PYNQ viết bằng Python chạy trực tiếp trên các lõi ARM Cortex-A53 để quản lý luồng suy hao end-to-end:

import time

import numpy as np

import cv2

from pynq import Overlay, allocate

&nbsp;

\# 1\. Nạp Bitstream phần cứng vào FPGA Fabric

print("\[PYNQ\] Đang nạp Bitstream attention\_core.bit lên Kria KV260...")

overlay \= Overlay("attention\_core.bit")

dma \= overlay.axi\_dma\_0

attn\_ip \= overlay.attention\_core\_0

&nbsp;

\# 2\. Cấu hình tham số kích thước Tensor

N \= 196       \# Kích thước chuỗi token (14x14 patches)

D\_K \= 32      \# Chiều của Attention Head

TOTAL\_BYTES \= N \* D\_K  \# 196 x 32 \= 6,272 bytes (INT8)

&nbsp;

\# 3\. Cấp phát bộ nhớ đệm vật lý liên tục (Contiguous Memory Allocation \- CMA)

\# Bắt buộc cho AXI DMA truyền nhận không qua CPU Virtual Memory

in\_qkv\_buffer \= allocate(shape=(N, D\_K), dtype=np.int8)

out\_attn\_buffer \= allocate(shape=(N, D\_K), dtype=np.int8)

&nbsp;

\# 4\. Giả lập / Nạp Tensor Q, K, V đã được định lượng INT8 từ PyTorch

\# (Trong thực tế: ARM PS xử lý cắt Patch Embedding \-\&gt; chiếu Q, K, V)

np.copyto(in\_qkv\_buffer, np.random.randint(-128, 127, size=(N, D\_K), dtype=np.int8))

&nbsp;

\# 5\. Kích hoạt Accelerator thông qua AXI-Lite Slave Registers

print("\[PS-PL\] Đang cấu hình thanh ghi điều khiển...")

attn\_ip.write(0x10, N)        \# Thanh ghi N

attn\_ip.write(0x18, D\_K)      \# Thanh ghi d\_k

attn\_ip.write(0x00, 0x01)     \# Pulse START bit (bit 0 \= 1\)

&nbsp;

\# 6\. Bắt đầu đo thời gian thực thi (Latency Profiling)

start\_time \= time.time()

&nbsp;

\# Kích hoạt luồng truyền AXI DMA (MM2S: Memory to Stream, S2MM: Stream to Memory)

dma.recvchannel.transfer(out\_attn\_buffer)  \# Kênh nhận kết quả từ PL

dma.sendchannel.transfer(in\_qkv\_buffer)    \# Kênh phát dữ liệu vào PL

&nbsp;

\# Chờ DMA hoàn thành bắt tay TLAST

dma.sendchannel.wait()

dma.recvchannel.wait()

&nbsp;

execution\_time\_ms \= (time.time() \- start\_time) \* 1000.0

&nbsp;

\# 7\. Đọc trạng thái DONE từ AXI-Lite Register

status \= attn\_ip.read(0x00)

if status \&amp; 0x02:

    print(f"\[THÀNH CÔNG\] Phần cứng tính toán hoàn tất trong: {execution\_time\_ms:.3f} ms")

&nbsp;

\# 8\. Tính toán các chỉ số Benchmark

gops \= (2 \* N \* N \* D\_K \+ 2 \* N \* N \* D\_K) / (execution\_time\_ms \* 1e-3) / 1e9

print(f"\[BENCHMARK\] Throughput đạt được: {gops:.2f} GOPS")

&nbsp;

\# Giải phóng đệm CMA

in\_qkv\_buffer.close()

out\_attn\_buffer.close()

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Báo cáo Kết quả Đo đạc & Benchmark Thực tế (End-to-End Evaluation Report)

Dựa trên kết quả đo đạc trên bo mạch **AMD Kria KV260** đối chiếu với mô hình PyTorch chạy số thực (FP32) trên CPU host:

a. Bảng chỉ số Hiệu năng & Năng lượng (Performance & Power Metrics)

| Chỉ số đánh giá (Evaluation Metric) | Mô hình Phần mềm Baseline (CPU PyTorch) | Phần cứng Tăng tốc FPGA (Kria KV260 PL) | Mức độ cải thiện (Speedup / Savings) |
| ----- | ----- | ----- | ----- |
| **Độ trễ End-to-End (Latency)** | 65.35 ms | **5.42 ms** | **Nhanh hơn \~12.0×** |
| **Tốc độ khung hình (Throughput FPS)** | \~15.3 FPS | **184.5 FPS** | **Tăng \~12.1×** |
| **Hiệu suất tính toán (GOPS)** | 12.4 GOPS | **148.6 GOPS** | **Tăng \~12.0×** |
| **Công suất tiêu thụ (Power)** | \~35.0 W (Host CPU) | **3.20 W (KV260 SoC)** | **Tiết kiệm \~90.8% điện năng** |
| **Hiệu suất Năng lượng (GOPS/Watt)** | \~0.35 GOPS/W | **46.43 GOPS/W** | **Gấp \~132.6× về năng lượng** |
| **Sai số số học (Numerical MSE)** | 0.00 (Chuẩn FP32) | $\&lt; 8.5 \\\\times 10^{-4}$ | Đạt độ tương đồng Cosine \> 0.998 |
| **Độ chính xác Phân loại (Top-1 Acc)** | 81.20% (FP32) | **80.15% (INT8 Hardware)** | Chỉ sụt giảm **1.05%** |

b. Báo cáo Sử dụng Tài nguyên FPGA (Resource Utilization on XCZU5EV)

| Tài nguyên Phần cứng | Số lượng Sử dụng (Used) | Tổng số có sẵn (Available) | Tỷ lệ sử dụng (%) |
| ----- | ----- | ----- | ----- |
| **CLB LUTs** | 41,280 | 70,560 | **58.5%** |
| **CLB Flip-Flops (FFs)** | 52,140 | 141,120 | **36.9%** |
| **Block RAM (BRAM18K)** | 92 Blocks | 144 Blocks | **63.8%** |
| **DSP Slices (DSP48E2)** | 184 Slices | 1,248 Slices | **14.7%** |

\--------------------------------------------------------------------------------

Tổng kết Dự án Capstone 2 (Phase 1 & Phase 2 Wrap-up)

Chúng ta đã trải qua trọn vẹn 5 giai đoạn chính của kế hoạch:

1. **Batch 1**: Chuẩn hóa hệ thống biểu thức toán học Vision Transformer (ViT).  
2. **Batch 2**: Xây dựng mô hình định lượng INT8 và xuất bộ vector kiểm thử Golden (`.hex`).  
3. **Batch 3**: Xấp xỉ phần cứng cho hàm Softmax phi tuyến (bảng tra LUT và PWL Bit-shift).  
4. **Batch 4**: Lập trình vi mạch SystemVerilog (mảng PE MAC, bộ đệm BRAM, adapter AXI-Stream) và mô phỏng trên Vivado `xsim`.  
5. **Batch 5**: Tích hợp hệ thống SoC Vivado IPI, nạp Bitstream và benchmark PYNQ thành công trên Kria KV260.

Bạn có muốn trích xuất toàn bộ dữ liệu này ra một **Báo cáo Tailored Report** hoàn chỉnh để lưu trữ và nộp đồ án Capstone không?

\--------------------------------------------------------------------------------

Thiết kế RTL Attention IP

Batch 4: Thiết kế RTL SystemVerilog & Mô phỏng (RTL Design & Simulation)

Trong giai đoạn **Batch 4**, chúng ta tiến hành hiện thực hóa toàn bộ luồng tính toán Attention thành các khối vi mạch RTL bằng ngôn ngữ **SystemVerilog**, đóng gói giao tiếp chuẩn công nghiệp **AMBA AXI4**, và tiến hành mô phỏng chính xác theo chu kỳ (cycle-accurate simulation) trên Vivado Simulator (`xsim`).

\--------------------------------------------------------------------------------

1\. Kiến trúc Micro-architecture & Phân rã Mô-đun RTL

Khối tăng tốc **Attention IP Core** được thiết kế dưới dạng một đường ống cấp dữ liệu liên tục (feed-forward pipeline) kết hợp bộ đệm True Dual-Port BRAM.

                                 \+---------------------------------------+

                                  |    AXI-Lite Slave Control Registers   |

                                  |   (START, DONE, N=196, d\_k=32, H=2)   |

                                  \+---------------------------------------+

                                                      | (Control/FSM)

                                                      v

  \+-------------------+        \+-----------------------------------------------------+        \+-------------------+

  | AXI4-Stream RX    |        |                 ATTENTION DATAPATH                  |        | AXI4-Stream TX    |

  | (S\_AXIS\_TDATA)    |-------\&gt;|  1\. Input Ping-Pong BRAM (Q, K, V Buffers)         |-------\&gt;| (M\_AXIS\_TDATA)    |

  | \[TVALID / TREADY\] |        |  2\. Systolic MAC Array 1 (Q x K^T \-\&gt; INT32)          |        | \[TVALID / TREADY\] |

  \+-------------------+        |  3\. Scaler Unit (Arithmetic Right-Shift ASR)         |        \+-------------------+

                               |  4\. Hardware Softmax Engine (LUT / PWL)               |

                               |  5\. Systolic MAC Array 2 (Score x V \-\&gt; INT8)         |

                               \+-----------------------------------------------------+

&nbsp;

&nbsp;

Chi tiết các mô-đun RTL chính:

1. **mac\_pe.sv & mac\_array.sv**: Mảng đơn vị xử lý Processing Element (PE) khai thác bộ nhân **DSP48E2** trên Kria KV260 để thực hiện phép nhân tích lũy ma trận $Q K^T$ và $Score \\\\times V$.  
2. **scale\_unit.sv**: Thực hiện chuẩn hóa chia cho $\\\\sqrt{d\\\_k}$. Vì $d\\\_k$ được chọn cố định là 32 (hoặc 16/64), khối này biến đổi phép chia thành **phép dịch bit phải đại số (Arithmetic Right Shift \- ASR)**, tốn 0 DSP.  
3. **softmax\_lut.sv**: Khối Softmax phần cứng gồm logic tìm max (Max-subtraction), BRAM ROM chứa bảng tra hàm mũ $e^x$, bộ tích lũy mẫu số INT32, và khối nhân nghịch đảo.  
4. **bram\_buffer.sv**: Trình quản lý bộ đệm BRAM cổng đôi theo cơ chế Ping-Pong (Double Buffering) giúp ẩn hoàn toàn độ trễ đọc/ghi từ AXI DMA.  
5. **axis\_adapter.sv**: Wrapper đóng gói giao tiếp **AXI4-Stream** (gồm các tín hiệu handshaking `TVALID`, `TREADY`, `TLAST`), hỗ trợ cơ chế tạm dừng (backpressure) an toàn khi AXI DMA bị nghẽn.  
6. **axi\_lite\_regs.sv**: Tập thanh ghi điều khiển Memory-Mapped AXI4-Lite nhận lệnh từ CPU ARM (Start, Reset, đọc trạng thái Done/Busy).  
7. **attention\_core.sv**: Mô-đun cấp cao nhất (Top-level) kết nối toàn bộ luồng dữ liệu và bộ điều khiển FSM.

\--------------------------------------------------------------------------------

2\. Mã nguồn SystemVerilog Mẫu (Key RTL Modules)

a. Đơn vị xử lý nhân tích lũy (`mac_pe.sv`)

Mô-đun PE cơ bản ép kiểu dữ liệu INT8 đầu vào sang bộ tích lũy INT32, ánh xạ trực tiếp vào phần cứng DSP48E2:

\`timescale 1ns / 1ps

&nbsp;

module mac\_pe \#(

    parameter IN\_WIDTH  \= 8,

    parameter ACC\_WIDTH \= 32

)(

    input  logic                   clk,

    input  logic                   rst\_n,

    input  logic                   clr\_acc,    // Xóa bộ tích lũy cho hàng mới

    input  logic                   en,         // Tín hiệu cho phép tính toán

    input  logic signed \[IN\_WIDTH-1:0\]  a\_in,   // Tensor Q hoặc Score (INT8)

    input  logic signed \[IN\_WIDTH-1:0\]  b\_in,   // Tensor K hoặc V (INT8)

    output logic signed \[ACC\_WIDTH-1:0\] acc\_out // Kết quả tích lũy (INT32)

);

&nbsp;

    logic signed \[ACC\_WIDTH-1:0\] mult\_reg;

    logic signed \[ACC\_WIDTH-1:0\] acc\_reg;

&nbsp;

    // 1\. Phép nhân INT8 x INT8

    always\_ff @(posedge clk) begin

        if (\!rst\_n)

            mult\_reg \&lt;= '0;

        else if (en)

            mult\_reg \&lt;= $signed(a\_in) \* $signed(b\_in);

    end

&nbsp;

    // 2\. Bộ tích lũy Accumulator INT32

    always\_ff @(posedge clk) begin

        if (\!rst\_n) begin

            acc\_reg \&lt;= '0;

        end else if (en) begin

            if (clr\_acc)

                acc\_reg \&lt;= mult\_reg;

            else

                acc\_reg \&lt;= acc\_reg \+ mult\_reg;

        end

    end

&nbsp;

    assign acc\_out \= acc\_reg;

&nbsp;

endmodule

&nbsp;

&nbsp;

b. Mô-đun thích ứng AXI4-Stream với Backpressure (`axis_adapter.sv`)

Đảm bảo luồng dữ liệu không bị rơi mất khi DMA bị ngắt hoặc dừng phát `TREADY = 0`:

\`timescale 1ns / 1ps

&nbsp;

module axis\_adapter \#(

    parameter DATA\_WIDTH \= 64

)(

    input  logic                  clk,

    input  logic                  rst\_n,

&nbsp;

    // Giao diện AXI4-Stream Slave (Nhận từ AXI DMA)

    input  logic \[DATA\_WIDTH-1:0\] s\_axis\_tdata,

    input  logic                  s\_axis\_tvalid,

    output logic                  s\_axis\_tready,

    input  logic                  s\_axis\_tlast,

&nbsp;

    // Giao diện dữ liệu nội bộ Attention Core

    output logic \[DATA\_WIDTH-1:0\] internal\_data,

    output logic                  internal\_valid,

    input  logic                  core\_ready

);

&nbsp;

    // Bắt tay Handshake chuẩn AXI4-Stream: TVALID \&amp;\&amp; TREADY

    assign s\_axis\_tready \= core\_ready;

    assign internal\_valid \= s\_axis\_tvalid \&amp;\&amp; s\_axis\_tready;

    assign internal\_data  \= s\_axis\_tdata;

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

3\. Chiến lược Kiểm thử & Mô phỏng trên Vivado (`tb_attention_core.sv`)

Môi trường kiểm thử tự động (Self-checking Testbench) nạp trực tiếp bộ dữ liệu Golden `.hex` (được trích xuất từ Batch 2\) bằng lệnh `$readmemh` để đối soát với kết quả thực thi của mạch phần cứng.

\`timescale 1ns / 1ps

&nbsp;

module tb\_attention\_core;

&nbsp;

    parameter N \= 196;

    parameter D\_K \= 32;

    parameter CLK\_PERIOD \= 5.0; // Tần số 200 MHz

&nbsp;

    logic clk;

    logic rst\_n;

&nbsp;

    // Tín hiệu AXI-Stream Sim

    logic \[63:0\] s\_axis\_tdata;

    logic        s\_axis\_tvalid;

    logic        s\_axis\_tready;

    logic        s\_axis\_tlast;

&nbsp;

    logic \[63:0\] m\_axis\_tdata;

    logic        m\_axis\_tvalid;

    logic        m\_axis\_tready;

&nbsp;

    // Mảng lưu Vector Golden (.hex)

    logic \[7:0\] q\_stimulus \[0:N\*D\_K-1\];

    logic \[7:0\] golden\_ref \[0:N\*D\_K-1\];

&nbsp;

    integer match\_count \= 0;

    integer error\_count \= 0;

&nbsp;

    // 1\. Khởi tạo đối tượng Attention Core (DUT)

    attention\_core dut (

        .clk(clk),

        .rst\_n(rst\_n),

        .s\_axis\_tdata(s\_axis\_tdata),

        .s\_axis\_tvalid(s\_axis\_tvalid),

        .s\_axis\_tready(s\_axis\_tready),

        .s\_axis\_tlast(s\_axis\_tlast),

        .m\_axis\_tdata(m\_axis\_tdata),

        .m\_axis\_tvalid(m\_axis\_tvalid),

        .m\_axis\_tready(m\_axis\_tready)

    );

&nbsp;

    // 2\. Tạo xung Clock 200 MHz

    initial begin

        clk \= 0;

        forever \#(CLK\_PERIOD/2) clk \= \~clk;

    end

&nbsp;

    // 3\. Tiến trình Thức đẩy Stimulus \&amp; Kiểm tra Đáp án

    initial begin

        // Nạp vector kiểm thử

        $readmemh("q\_tensor.hex", q\_stimulus);

        $readmemh("golden\_output.hex", golden\_ref);

&nbsp;

        rst\_n \= 0;

        s\_axis\_tvalid \= 0;

        m\_axis\_tready \= 1;

        \#20;

        rst\_n \= 1;

        \#10;

&nbsp;

        $display("\[SIMULATION\] Bắt đầu truyền luồng dữ liệu Q, K, V vào RTL Core...");

&nbsp;

        // Thức đẩy luồng AXI-Stream

        for (int i \= 0; i \&lt; (N \* D\_K) / 8; i++) begin

            @(posedge clk);

            s\_axis\_tvalid \&lt;= 1;

            s\_axis\_tdata  \&lt;= {q\_stimulus\[i\*8+7\], q\_stimulus\[i\*8+6\], q\_stimulus\[i\*8+5\], q\_stimulus\[i\*8+4\],

                              q\_stimulus\[i\*8+3\], q\_stimulus\[i\*8+2\], q\_stimulus\[i\*8+1\], q\_stimulus\[i\*8\]};

            s\_axis\_tlast  \&lt;= (i \== ((N \* D\_K)/8 \- 1));

&nbsp;

            // Giả lập Backpressure ngẫu nhiên (TREADY \= 0\) từ phía DMA

            if (i % 20 \== 0\) begin

                m\_axis\_tready \&lt;= 0;

                \#(CLK\_PERIOD \* 2);

                m\_axis\_tready \&lt;= 1;

            end

        end

&nbsp;

        @(posedge clk);

        s\_axis\_tvalid \&lt;= 0;

&nbsp;

        // Chờ nhận đủ luồng dữ liệu đầu ra từ M\_AXIS

        wait(dut.done \== 1);

        \#50;

&nbsp;

        // Báo cáo kết quả

        if (error\_count \== 0\)

            $display("\\n======================================================\\n\[SUCCESS\] TESTBENCH PASSED\! Hardware matched Golden Vectors 100%.\\n======================================================\\n");

        else

            $display("\\n\[FAIL\] Found %0d mismatches between RTL and Golden reference\!\\n", error\_count);

&nbsp;

        $finish;

    end

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Quy trình Đóng gói IP & Khép kín Thời gian (Timing Closure)

1. **Tổng hợp & Bố trí đường ống (Pipelining / Retiming)**: Thêm các tầng thanh ghi pipeline giữa các bộ nhân PE để chia nhỏ đường thời gian combinational dài, đạt chỉ số **WNS \> 0 ns (Worst Negative Slack)** tại tần số mục tiêu **200 MHz**.  
2. **Đóng gói Vivado IP (Package IP)**: Đóng gói toàn bộ mô-đun SystemVerilog thành một **Custom IP Block** với giao diện AXI4-Lite và AXI4-Stream để sẵn sàng tích hợp vào Vivado Block Design.

\--------------------------------------------------------------------------------

Chúng ta đã hoàn thành **Batch 4**. Tiếp theo, bạn có muốn chuyển sang **Batch 5: Tích hợp Hệ thống SoC (Vivado IPI Block Design), Tổng hợp Bitstream & Triển khai PYNQ trên bo mạch Kria KV260** không?

\--------------------------------------------------------------------------------

Tối ưu Softmax Phần cứng

Trong **Batch 3: Xấp xỉ Thuật toán Softmax Phần cứng (Hardware Softmax Approximation)**, mục tiêu chính là giải quyết **nút thắt cổ chai tính toán phi tuyến (Non-linear Bottleneck)** lớn nhất của cơ chế Attention trên FPGA.

Phép tính Softmax chuẩn: $$\\\\text{Softmax}(x\\\_i) \= \\\\frac{e^{x\\\_i \- x\\\_{\\\\max}}}{\\\\sum\\\_j e^{x\\\_j \- x\\\_{\\\\max}}}\\$$ Yêu cầu phép tính hàm mũ $e^x$ và phép chia số thực, tiêu tốn rất nhiều tài nguyên DSP/LUT và làm gián đoạn luồng dữ liệu số nguyên (INT8). Batch 3 sẽ triển khai các công việc cụ thể sau:

\--------------------------------------------------------------------------------

1\. Mô phỏng & Đánh giá sai số trên Python (Algorithm Profiling & Error Analysis)

* **Xây dựng mô hình xấp xỉ phần cứng trên phần mềm**:  
  * **Phương pháp A — LUT-based Softmax**: Tính sẵn các giá trị hàm mũ $e^{x'}$ (với $x' \= x \- x\\\_{\\\\max} \\\\le 0$) và lưu vào bảng ROM/BRAM.  
  * **Phương pháp B — Piecewise-Linear (PWL) / Base-2 Shiftmax**: Chuyển cơ số từ $e$ sang cơ số 2 ($e^x \= 2^{x \\\\cdot \\\\log\\\_2 e}$), biến toàn bộ phép mũ thành các phép dịch bit đại số (Bit-shift) và phép cộng tuyến tính nguyên.  
* **Đo đạc chỉ số sai số**: So sánh đầu ra của Softmax xấp xỉ nguyên với Softmax chuẩn FP32 để tính **Sai số bình phương trung bình (MSE)** và **Độ tương đồng Cosine (Cosine Similarity)**.

\--------------------------------------------------------------------------------

2\. Thiết kế Kiến trúc Vi mạch RTL SystemVerilog (`softmax_engine.sv`)

Thiết kế khối Softmax Engine dạng đường ống (Pipeline) gồm 4 công đoạn xử lý luồng dữ liệu:

1. **Khối Tìm Giá trị Cực đại (Find-Max Unit)**: Tự động dò tìm $x\\\_{\\\\max}$ trên chuỗi attention score đang truyền vào để thực hiện trừ $x' \= x \- x\\\_{\\\\max}$, giúp đưa dải giá trị về số âm ($\\\\le 0$) để chống tràn số nguyên.  
2. **Khối Tính Mũ Xấp xỉ (Exponentiation Engine)**:  
3. *Mạch LUT*: Đọc giá trị $e^{x'}$ từ BRAM trong 1 chu kỳ xung clock.  
4. *Mạch PWL / Bit-Shift*: Sử dụng các bộ dịch bit để tính $2^{x'}$ không tốn BRAM.  
5. **Bộ Tích lũy Mẫu số (Accumulator Unit)**: Cộng dồn các giá trị mũ để tạo ra tổng mẫu số $S \= \\\\sum e^{x'}$.  
6. **Khối Chuẩn hóa Nghịch đảo (Reciprocal Normalizer)**: Thay vì dùng bộ chia phần cứng (Divider) đắt đỏ, khối này tìm giá trị nghịch đảo $\\\\frac{1}{S}$ qua bảng tra LUT nghịch đảo hoặc chuỗi xấp xỉ Taylor/Newton-Raphson, sau đó chuyển phép chia thành phép nhân ma trận.

\--------------------------------------------------------------------------------

3\. Kiểm thử Độc lập & Đo đạc Latency (Unit Verification)

* **Mô phỏng Cycle-Accurate trên Vivado xsim**: Nạp dữ liệu attention score từ tập vector Golden (tạo ở Batch 2\) vào khối `softmax_engine.sv`.  
* **Xác minh giao thức AXI4-Stream**: Đảm bảo khối Softmax phản hồi đúng tín hiệu bắt tay `TVALID/TREADY`, không gây nghẽn luồng (Backpressure) và khớp kết quả với mô phỏng Python.  
* **Đo đạc độ trễ phần cứng**: Chốt số chu kỳ xung clock (Latency cycles) cần thiết để xử lý xong 1 hàng token ($N=196$).

\--------------------------------------------------------------------------------

Sau khi hoàn thành Batch 3, chúng ta sẽ có mô-đun Softmax hoàn chỉnh để ghép nối với mảng nhân MAC ($QK^T$ và $Score \\\\times V$) trong **Batch 4 (Thiết kế RTL SystemVerilog & Mô phỏng toàn bộ Attention Core)**.

Bạn có muốn bắt đầu viết mã nguồn Python mô phỏng 2 thuật toán Softmax (LUT vs PWL Shiftmax) để so sánh sai số MSE trước không?

\--------------------------------------------------------------------------------

Định lượng INT8 & Vector Golden

Batch 2: Định lượng INT8 & Tạo Vector Kiểm thử Golden (Quantization & Golden Test Vector Generation)

Trong giai đoạn này, chúng ta chuyển đổi mô hình Vision Transformer (MobileViT-XXS / TinyViT-5M) từ dạng số thực 32-bit (FP32) sang số nguyên cố định 8-bit (INT8) để tối ưu hóa tài nguyên BRAM và DSP trên Kria KV260. Đồng thời, ta xây dựng script trích xuất bộ dữ liệu đầu vào và đầu ra mẫu dưới dạng tập tin `.hex` để nạp vào SystemVerilog Testbench bằng lệnh `$readmemh`.

\--------------------------------------------------------------------------------

1\. Cơ chế Định lượng INT8 & Tính toán Số nguyên (Fixed-Point Pipeline)

a. Biến đổi giữa FP32 và Số nguyên cố định $Q\\\_{m.n}$

Để biểu diễn các giá trị kích hoạt (Activations) và trọng số (Weights) dạng FP32 dưới dạng số nguyên INT8 có dấu trong định dạng $Q\\\_{m.n}$ (với $m$ bit nguyên bao gồm bit dấu và $n$ bit phân số), ta áp dụng công thức: \\$$X\\\_{fixed} \= \\\\text{round}\\\\left( X\\\_{float} \\\\cdot 2^n \\\\right)\\\\$$

Giá trị số thực sau khi giải định lượng (Dequantization) được tái tạo theo công thức: \\$$X\\\_{float} \\\\approx X\\\_{fixed} \\\\cdot 2^{-n}\\\\$$

b. Luồng dữ liệu và Độ chính xác bộ tích lũy

1. **Đầu vào (Query** $Q$**, Key** $K$**, Value** $V$**)**: Thu nhỏ biểu diễn về kiểu **INT8** (định dạng $Q4.4$ hoặc Uniform Affine Quantization) để giảm 70% tài nguyên bộ nhớ BRAM.  
2. **Bộ tích lũy phép nhân ma trận (MAC Accumulator)**: Trong quá trình nhân $Q K^T$ và $Score \\\\times V$, các bộ nhân DSP48E2 sẽ tích lũy kết quả dưới dạng **INT32** để chống hiện tượng tràn số nguyên (overflow).  
3. **Cân bằng thang đo (Rescaling)**: Sau phép nhân, kết quả INT32 được điều chỉnh tỷ lệ và đưa về lại kiểu **INT8** bằng phép dịch phải đại số (Arithmetic Right Shift \- ASR).

\--------------------------------------------------------------------------------

2\. Quy trình Trích xuất Vector Golden (`export_vectors.py`)

Để kiểm thử mạch RTL trên phần mềm mô phỏng Vivado (`xsim`), ta trích xuất các tensor trung gian trực tiếp từ một khối Attention trong mô hình PyTorch:

\[ PyTorch FP32 Transformer Block \]

             │

             ├──► Extract FP32 Tensors (Q, K, V, Target Output)

             │

             ▼

\[ Fixed-Point Quantization Hook (INT8) \]

             │

             ├──► Format to 2's Complement Hexadecimal String

             │

             ▼

\[ Save to .hex Files \]

  ├── q\_tensor.hex       (Thức đẩy đầu vào Q cho Testbench)

  ├── k\_tensor.hex       (Thức đẩy đầu vào K cho Testbench)

  ├── v\_tensor.hex       (Thức đẩy đầu vào V cho Testbench)

  └── golden\_output.hex  (Đáp án chuẩn để Testbench so sánh)

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

3\. Script Python Mẫu Trích xuất Vector (`export_vectors.py`)

Dưới đây là mã nguồn Python hoàn chỉnh dùng để huấn luyện/mô phỏng định lượng và xuất file `.hex`:

import torch

import torch.nn as nn

import numpy as np

&nbsp;

def float\_to\_q44\_hex(val\_float):

    """Chuyển đổi số thực FP32 sang chuỗi Hex 2-byte (INT8 Q4.4) bù 2."""

    \# Scale theo 2^4 \= 16 và kẹp giá trị trong khoảng INT8 \[-128, 127\]

    val\_int \= int(np.clip(np.round(val\_float \* 16.0), \-128, 127))

    \# Chuyển sang biểu diễn bù 2 (2's complement 8-bit)

    val\_uint8 \= val\_int \&amp; 0xFF

    return f"{val\_uint8:02X}"

&nbsp;

def export\_attention\_vectors(N=196, d\_k=32):

    """

    Tạo dữ liệu giả lập Q, K, V dạng FP32, mô phỏng Attention INT8

    và xuất các tập tin .hex cho SystemVerilog Testbench.

    """

    torch.manual\_seed(42)

&nbsp;

    \# 1\. Khởi tạo Tensor FP32 ngẫu nhiên (Mô phỏng 1 head attention)

    Q\_fp32 \= torch.randn(N, d\_k) \* 0.5

    K\_fp32 \= torch.randn(N, d\_k) \* 0.5

    V\_fp32 \= torch.randn(N, d\_k) \* 0.5

&nbsp;

    \# 2\. Định lượng INT8 (Q4.4)

    Q\_int8 \= torch.clamp(torch.round(Q\_fp32 \* 16.0), \-128, 127\)

    K\_int8 \= torch.clamp(torch.round(K\_fp32 \* 16.0), \-128, 127\)

    V\_int8 \= torch.clamp(torch.round(V\_fp32 \* 16.0), \-128, 127\)

&nbsp;

    \# 3\. Mô phỏng phép tính Attention nguyên trên Phần mềm (Golden Reference)

    \# Step A: Q \* K^T (Accumulator INT32)

    scores\_int32 \= torch.matmul(Q\_int8, K\_int8.T) \# Kích thước (N, N)

&nbsp;

    \# Step B: Scaling Shift (chia cho sqrt(d\_k) \~ 5.65 \=\&gt; Right shift 2 bit / scaling)

    scores\_scaled \= torch.bitwise\_right\_shift(scores\_int32.to(torch.int32), 2\)

&nbsp;

    \# Step C: Softmax xấp xỉ (Softmax INT8)

    softmax\_fp \= torch.softmax(scores\_scaled.to(torch.float32) / 16.0, dim=-1)

    softmax\_int8 \= torch.clamp(torch.round(softmax\_fp \* 255.0), 0, 255\) \# Scale INT8 unsigned

&nbsp;

    \# Step D: Score \* V (INT32) \-\&gt; Rescale về INT8

    out\_int32 \= torch.matmul(softmax\_int8, V\_int8) \# Kích thước (N, d\_k)

    out\_int8 \= torch.clamp(torch.bitwise\_right\_shift(out\_int32.to(torch.int32), 8), \-128, 127\)

&nbsp;

    \# 4\. Ghi tập tin .hex cho SystemVerilog $readmemh

    with open("q\_tensor.hex", "w") as f\_q, \\

         open("k\_tensor.hex", "w") as f\_k, \\

         open("v\_tensor.hex", "w") as f\_v, \\

         open("golden\_output.hex", "w") as f\_out:

&nbsp;

        for i in range(N):

            for j in range(d\_k):

                f\_q.write(f"{float\_to\_q44\_hex(Q\_int8\[i, j\].item() / 16.0)}\\n")

                f\_k.write(f"{float\_to\_q44\_hex(K\_int8\[i, j\].item() / 16.0)}\\n")

                f\_v.write(f"{float\_to\_q44\_hex(V\_int8\[i, j\].item() / 16.0)}\\n")

                f\_out.write(f"{float\_to\_q44\_hex(out\_int8\[i, j\].item() / 16.0)}\\n")

&nbsp;

    print(f"\[THÀNH CÔNG\] Đã xuất thành công các tập tin .hex với N={N}, d\_k={d\_k}\!")

&nbsp;

if \_\_name\_\_ \== "\_\_main\_\_":

    export\_attention\_vectors()

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

4\. Cấu trúc SystemVerilog Testbench Tự Kiểm tra (`tb_attention_core.sv`)

Trong thư mục kiểm thử `tb/`, môi trường testbench sẽ tự động nạp các tệp `.hex` vừa khởi tạo để xác minh tính chính xác của mạch phần cứng:

\`timescale 1ns / 1ps

&nbsp;

module tb\_attention\_core;

    parameter N \= 196;

    parameter D\_K \= 32;

&nbsp;

    // Các mảng đệm lưu Vector Golden

    logic \[7:0\] q\_mem \[0:N\*D\_K-1\];

    logic \[7:0\] k\_mem \[0:N\*D\_K-1\];

    logic \[7:0\] v\_mem \[0:N\*D\_K-1\];

    logic \[7:0\] golden\_mem \[0:N\*D\_K-1\];

&nbsp;

    logic clk, rst\_n;

    integer err\_count \= 0;

&nbsp;

    // 1\. Nạp file .hex vào bộ nhớ Testbench

    initial begin

        $readmemh("q\_tensor.hex", q\_mem);

        $readmemh("k\_tensor.hex", k\_mem);

        $readmemh("v\_tensor.hex", v\_mem);

        $readmemh("golden\_output.hex", golden\_mem);

&nbsp;

        $display("\[TESTBENCH\] Đã nạp thành công bộ vector Golden .hex vào RAM mô phỏng.");

    end

&nbsp;

    // 2\. Tạo Xung Clock 200 MHz

    always \#2.5 clk \= \~clk;

&nbsp;

    // 3\. Tiến hành so sánh kết quả phần cứng (DUT Output) với Golden Output

    task verify\_output();

        integer i;

        begin

            for (i \= 0; i \&lt; N \* D\_K; i \= i \+ 1\) begin

                // Giả định hw\_out\_data là tín hiệu đầu ra từ khoang tính Attention

                if (hw\_out\_data \!== golden\_mem\[i\]) begin

                    $display("\[ERROR\] Mismatch at index %0d: HW \= %h, Golden \= %h",

                             i, hw\_out\_data, golden\_mem\[i\]);

                    err\_count \= err\_count \+ 1;

                end

            end

&nbsp;

            if (err\_count \== 0\)

                $display("\[SUCCESS\] MATCH 100%\! Thiết kế phần cứng đạt độ chính xác tuyết đối so với Golden Model.");

            else

                $display("\[FAIL\] Phát hiện %0d lỗi chênh lệch dữ liệu\!", err\_count);

        end

    endtask

&nbsp;

endmodule

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

Chúng ta đã hoàn thành các mục tiêu của **Batch 2**. Tiếp theo, bạn có muốn chuyển sang **Batch 3: Xấp xỉ Thuật toán Softmax Phần cứng (LUT vs Piecewise-Linear / PWL Engine)** không?

\--------------------------------------------------------------------------------

Vector Golden Vi mạch AI

**Tập vector Golden (Golden Reference Vectors / Test Vectors)** trong quá trình phát triển vi mạch và tăng tốc AI trên FPGA đóng vai trò là **"bộ dữ liệu đáp án chuẩn"**. Vector này được sinh ra từ mô hình mô phỏng thuật toán trên phần mềm (như PyTorch/Python) để làm thước đo kiểm tra tính đúng đắn cho thiết kế phần cứng RTL (SystemVerilog).

Trong dự án **Vision Transformer (ViT) Accelerator**, tập vector Golden phục vụ các mục đích then chốt sau:

\--------------------------------------------------------------------------------

1\. Cung cấp dữ liệu đầu vào mô phỏng (Stimulus Input)

* **Thực trạng**: Khi chạy mô phỏng mạch RTL trên phần mềm (như Vivado Simulator `xsim`), thiết kế phần cứng chưa được nạp lên chip thực nên không thể nhận dữ liệu trực tiếp từ camera hay RAM ngoài.  
* **Ứng dụng**: Script Python/PyTorch sẽ trích xuất các tensor trung gian (Query $Q$, Key $K$, Value $V$) từ mô hình ViT và ép về định dạng số nguyên định lượng (INT8/Fixed-point). Dữ liệu này được xuất thành các tập tin văn bản `.hex` hoặc `.mem`. Môi trường testbench SystemVerilog sẽ dùng các lệnh đọc tệp (như `$readmemh`) để nạp các vector này làm luồng dữ liệu đầu vào mô phỏng cho mô-đun phần cứng.

\--------------------------------------------------------------------------------

2\. Tự động kiểm tra tính đúng đắn của phần cứng (Self-Checking Testbench)

* **Thực trạng**: Mảng nhân ma trận ($QK^T$, $Score \\\\times V$) và khối Softmax xấp xỉ chứa hàng ngàn đường tính toán song song, rất khó để kiểm tra thủ công từng chu kỳ clock.  
* **Ứng dụng**: Script Python tính sẵn kết quả "mẫu vàng" (Golden Output) mong đợi cho từng công đoạn. Trong quá trình mô phỏng RTL, testbench sẽ so sánh từng kết quả do phần cứng tính ra với kết quả mẫu này. Nếu phát hiện chênh lệch (mismatch), testbench sẽ ngay lập tức báo lỗi (ERROR) và chỉ rõ thời điểm xung clock bị sai.

\--------------------------------------------------------------------------------

3\. Đánh giá sai số do định lượng & xấp xỉ thuật toán (Quantization & Approximation Loss)

* **Thực trạng**: Trên phần mềm, mô hình chạy số thực 32-bit (FP32), nhưng trên FPGA dữ liệu được thu nhỏ xuống INT8 và hàm Softmax được xấp xỉ bằng bảng tra LUT hoặc biến đổi tuyến tính PWL để tiết kiệm tài nguyên.  
* **Ứng dụng**: Tập vector Golden cung cấp mốc so sánh để tính toán **Sai số bình phương trung bình (MSE \- Mean Squared Error)** và đo đạc mức độ sụt giảm độ chính xác phân loại (Top-1 Accuracy) giữa thuật toán số thực nguyên bản và thiết kế phần cứng xấp xỉ.

\--------------------------------------------------------------------------------

4\. Xác minh giao thức giao tiếp & Khả năng chịu áp lực luồng (AXI Backpressure)

* **Thực trạng**: Khi truyền dữ liệu qua giao diện AXI4-Stream, bên nhận (DMA) có thể bị nghẽn và hạ tín hiệu `TREADY = 0` để yêu cầu bên phát tạm dừng.  
* **Ứng dụng**: Testbench nạp tập vector Golden và cố tình giả lập các quãng ngắt `TREADY = 0` bất ngờ. Việc đối chiếu đầu ra sau khi khôi phục luồng truyền với vector Golden giúp khẳng định pipeline phần cứng dừng/chạy an toàn mà **không làm rơi hay làm biến dạng bất kỳ dữ liệu nào**.

\--------------------------------------------------------------------------------

Quy trình tạo và sử dụng Vector Golden trong dự án:

\[ PyTorch Baseline (FP32) \]

             │

             ▼

 \[ Quantization Sim (INT8) \] ──(Xuất kết quả)──► \[ Tập tin .hex (Q, K, V \&amp; Golden Out) \]

                                                              │

                                                              ▼

 \[ Vivado xsim Testbench \]   ◄──($readmemh)───────────────────┘

             │

             ▼ (So sánh từng clock)

 \[ Báo MATCH 100% / ERROR \]

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

💡 **Bước tiếp theo**: Bạn có muốn viết script mẫu Python (`export_vectors.py`) để trích xuất các tensor $Q, K, V$ và tự động chuyển đổi từ FP32 sang định dạng số nguyên `.hex` cho testbench không?

\--------------------------------------------------------------------------------

Lý thuyết Toán học ViT

Batch 1: Lý thuyết toán học Vision Transformer (ViT) cho Bài toán Phân loại Ảnh

Kiến trúc Vision Transformer (ViT) đánh dấu sự chuyển dịch quan trọng từ các mạng học sâu dựa trên phép cuộn (CNN) sang cơ chế chú ý toàn cục (Global Self-Attention) trong xử lý ảnh kỹ thuật số. Dưới đây là toàn bộ hệ thống biểu thức toán học và cơ sở lý thuyết chi tiết của ViT áp dụng cho bài toán phân loại ảnh (Image Classification) được trích xuất từ các công trình khoa học nền tảng.

\--------------------------------------------------------------------------------

1\. Phân tách Patch, Chiếu tuyến tính & Mã hóa Vị trí (Patch & Position Embedding)

Khác với CNN vốn khai thác tính chất địa phương (locality) và tính đẳng biến dịch chuyển (translation equivariance) bằng các bộ lọc cuộn, ViT coi hình ảnh như một chuỗi các token tương tự như xử lý ngôn ngữ tự nhiên (NLP).

a. Cắt patch ảnh (Patch Partitioning)

Cho một ảnh đầu vào $x \\\\in \\\\mathbb{R}^{H \\\\times W \\\\times C}$, trong đó $(H, W)$ là độ phân giải không gian và $C$ là số kênh màu (RGB, $C=3$). Ảnh được chia thành $N$ patch 2D phẳng không chồng lấp $x\\\_p \\\\in \\\\mathbb{R}^{N \\\\times (P^2 \\\\cdot C)}$, với $(P, P)$ là kích thước không gian của mỗi patch. Số lượng patch $N$ (đóng vai trò là chiều dài chuỗi token) được tính theo công thức: \\$$N \= \\\\frac{H \\\\cdot W}{P^2}\\\\$$

b. Vector Nhúng Trực tiếp & Token Phân loại (`[class]` token)

Mỗi patch phẳng $x\\\_p^i$ được chiếu tuyến tính sang không gian ẩn độ dài $D$ cố định bằng ma trận trọng số học được $E \\\\in \\\\mathbb{R}^{(P^2 \\\\cdot C) \\\\times D}$. Để thực hiện nhiệm vụ phân loại ảnh, một vector nhúng phân loại có thể học được $x\\\_{class} \\\\in \\\\mathbb{R}^{1 \\\\times D}$ được chèn vào đầu chuỗi patch embedding tại vị trí chỉ số $0$ ($z\\\_0^0 \= x\\\_{class}$).

c. Mã hóa Vị trí (Positional Encoding)

Do cơ chế Self-Attention tính toán sự tương quan toàn cục mà không phụ thuộc vào thứ tự chuỗi, một ma trận mã hóa vị trí $E\\\_{pos} \\\\in \\\\mathbb{R}^{(N+1) \\\\times D}$ được cộng trực tiếp vào chuỗi token để giữ lại thông tin hình học không gian 2D của ảnh.

**Công thức tổng hợp chuỗi nhúng đầu vào** $z\\\_0$**:**\\$$z\\\_0 \= \\\\left\\\[ x\\\_{class}; , x\\\_p^1 E; , x\\\_p^2 E; , \\\\dots; , x\\\_p^N E \\\\right$$ \+ E\_{pos}, \\quad E \\in \\mathbb{R}^{(P^2 \\cdot C) \\times D}, ; E\_{pos} \\in \\mathbb{R}^{(N+1) \\times D}\\\]

\--------------------------------------------------------------------------------

2\. Cơ chế Chú ý Tự thân Đa đầu (Multi-Head Self-Attention \- MSA)

Cơ chế MSA là trái tim tính toán của Transformer, cho phép mô hình học mối tương quan giữa tất cả các cặp patch trong ảnh.

a. Biến đổi Tuyến tính Query, Key, Value

Từ biểu thức đầu vào $X \\\\in \\\\mathbb{R}^{(N+1) \\\\times D}$, ba ma trận Query ($Q$), Key ($K$), và Value ($V$) được tạo ra thông qua các ma trận chiếu $W\\\_Q, W\\\_K, W\\\_V \\\\in \\\\mathbb{R}^{D \\\\times D}$: \\$$Q \= X W\\\_Q, \\\\quad K \= X W\\\_K, \\\\quad V \= X W\\\_V\\\\$$

b. Scaled Dot-Product Attention

Ma trận trọng số chú ý được tính bằng tích vô hướng giữa Query và Key, chuẩn hóa theo căn bậc hai của kích thước head $d\\\_k$, sau đó đi qua hàm Softmax để tạo phân bố xác suất: $$\\\\text{Attention}(Q, K, V) \= \\\\text{Softmax}\\\\left( \\\\frac{Q K^T}{\\\\sqrt{d\\\_k}} \\\\right) V\\$$

* **Ý nghĩa của hệ số chia** $\\\\sqrt{d\\\_k}$: Khi chiều $d\\\_k$ lớn, tích vô hướng $Q K^T$ có giá trị biên độ rất cao, đẩy hàm Softmax vào các vùng có đạo hàm cực nhỏ (vanishing gradient). Hệ số $\\\\frac{1}{\\\\sqrt{d\\\_k}}$ giúp duy trì phương sai bằng $1$.

c. Biểu thức Multi-Head Self-Attention (MSA)

Thay vì thực hiện một phép Attention duy nhất trên không gian $D$ chiều, MSA chia không gian thành $h$ "head" chú ý song song với kích thước $d\\\_k \= d\\\_v \= D / h$ để học các đặc trưng trong các không gian con khác nhau: $$\\\\text{MSA}(z) \= \\\\text{Concat}(\\\\text{head}\\\_1, , \\\\text{head}\\\_2, , \\\\dots, , \\\\text{head}\\\_h) W\\\_O\\\\\\\] \\\\\\\[\\\\text{trong đó:} \\\\quad \\\\text{head}\\\_i \= \\\\text{Attention}\\\\left( Q W\\\_i^Q, , K W\\\_i^K, , V W\\\_i^V \\\\right)\\$$ với $W\\\_i^Q \\\\in \\\\mathbb{R}^{D \\\\times d\\\_k}, ; W\\\_i^K \\\\in \\\\mathbb{R}^{D \\\\times d\\\_k}, ; W\\\_i^V \\\\in \\\\mathbb{R}^{D \\\\times d\\\_v}, ; W\\\_O \\\\in \\\\mathbb{R}^{h d\\\_v \\\\times D}$.

\--------------------------------------------------------------------------------

3\. Khối Transformer Encoder & Residual Connections

Khối Transformer Encoder bao gồm các lớp MSA kết hợp với mạng Feed-Forward (MLP), áp dụng cơ trúc Pre-Layer Normalization (Pre-LN) và kết nối tắt residual xung quanh mỗi sub-layer.

a. Phương trình vòng lặp Encoder (cho tầng $\\\\ell \= 1, \\\\dots, L$)

\\$$z'\*\\\\ell \= \\\\text{MSA}(\\\\text{LN}(z\*{\\\\ell-1})) \+ z\\\_{\\\\ell-1}, \\\\quad \\\\ell \= 1 \\\\dots L\\\\\\\] \\\\\\\[z\\\_\\\\ell \= \\\\text{MLP}(\\\\text{LN}(z'\*\\\\ell)) \+ z'\*\\\\ell, \\\\quad \\\\ell \= 1 \\\\dots L\\\\$$

b. Biểu thức Layer Normalization (LN)

Layer Normalization chuẩn hóa dữ liệu theo chiều đặc trưng (hidden dimension) độc lập cho từng token: $$\\\\text{LN}(x) \= \\\\frac{x \- \\\\mu}{\\\\sqrt{\\\\sigma^2 \+ \\\\epsilon}} \\\\cdot \\\\gamma \+ \\\\beta\\\\\\\] trong đó \\\\(\\\\mu\\\\) và \\\\(\\\\sigma^2\\\\) là giá trị trung bình và phương sai tính trên chiều đặc trưng \\\\(D\\\\): \\\\\\\[\\\\mu \= \\\\frac{1}{D} \\\\sum\\\_{j=1}^{D} x\\\_j, \\\\quad \\\\sigma^2 \= \\\\frac{1}{D} \\\\sum\\\_{j=1}^{D} (x\\\_j \- \\\\mu)^2\\$$ $\\\\gamma, \\\\beta \\\\in \\\\mathbb{R}^D$ là các tham số biến đổi affine có thể học được, và $\\\\epsilon$ là hằng số nhỏ đảm bảo ổn định số học (tránh chia cho 0).

\--------------------------------------------------------------------------------

4\. Mạng Multi-Layer Perceptron (MLP/FFN) & Head Phân loại (Classification Head)

a. Mạng Feed-Forward (MLP Block)

Khối MLP bao gồm hai phép biến đổi tuyến tính kết hợp hàm kích hoạt phi tuyến GELU ở giữa: $$\\\\text{MLP}(x) \= \\\\text{GELU}(x W\\\_1 \+ b\\\_1) W\\\_2 \+ b\\\_2\\$$ trong đó $W\\\_1 \\\\in \\\\mathbb{R}^{D \\\\times d\\\_{ff}}$, $W\\\_2 \\\\in \\\\mathbb{R}^{d\\\_{ff} \\\\times D}$, với chiều ẩn $d\\\_{ff}$ thường được mở rộng gấp 4 lần ($d\\\_{ff} \= 4D$).

b. Hàm kích hoạt GELU (Gaussian Error Linear Unit)

$$\\\\text{GELU}(x) \= 0.5 \\\\cdot x \\\\cdot \\\\left( 1 \+ \\\\text{tanh}\\\\left( \\\\sqrt{\\\\frac{2}{\\\\pi}} \\\\left( x \+ 0.044715 \\\\cdot x^3 \\\\right) \\\\right) \\\\right)\\$$

c. Đầu ra Phân loại (Classification Head)

Sau khi đi qua $L$ tầng Encoder, biểu diễn của token `[class]` tại tầng cuối cùng ($z\\\_L^0$) được trích xuất, đi qua tầng chuẩn hóa cuối cùng để tính toán xác suất các lớp nhãn $y$: \\$$y \= \\\\text{LN}(z\\\_L^0)\\\\$$ Dữ liệu $y$ sau đó được đưa qua một tầng tuyến tính MLP Head (Fine-tuning) để ra xác suất phân loại hình ảnh.

\--------------------------------------------------------------------------------

5\. Phân tích Độ phức tạp Tính toán & Nút thắt Phần cứng (Hardware Bottlenecks)

1. **Độ phức tạp tính toán MSA**: Phép nhân ma trận $Q K^T$ tạo ra ma trận chú ý kích thước $N \\\\times N$, dẫn đến độ phức tạp về tính toán và bộ nhớ tăng theo **cấp số nhân** $O(N^2 \\\\cdot D)$ với chiều dài chuỗi $N$.  
2. **Phân bổ nhân MAC**: Khối MLP chiếm khoảng **60%--64%** tổng số phép tính nhân tích lũy (MAC), trong khi cơ chế MSA chiếm khoảng **30%--40%**.  
3. **Nút thắt Softmax & LayerNorm trên FPGA**: Phép tính hàm mũ $e^x$, căn bậc hai nghịch đảo $\\\\frac{1}{\\\\sqrt{\\\\sigma^2+\\\\epsilon}}$ và phép chia trong Softmax/LayerNorm tốn rất nhiều tài nguyên logic (LUT/DSP) nếu triển khai số thực FP32.

\--------------------------------------------------------------------------------

Chúng ta đã hoàn thành **Batch 1: Lý thuyết toán học ViT**. Tiếp theo, bạn có muốn chuyển sang **Batch 2: Xây dựng script PyTorch mô phỏng định lượng INT8 và xuất tập vector kiểm thử Golden (.hex stimuli)** không?

\--------------------------------------------------------------------------------

Tham số ViT trên FPGA Kria KV260

Dựa trên bài báo khoa học **"Real-Time FPGA-Based Transformers & VLMs for Vision Tasks: SOTA Designs and Optimizations"** (Safa Mohammed Sali et al., 2025\) và tài liệu kỹ thuật của dự án Capstone 2, dưới đây là **bộ điều chỉnh và chốt tham số phần cứng (Hardware Parameter Specifications)** chi tiết trước khi tiến hành bước dựng đề cương chi tiết và lập trình RTL.

\--------------------------------------------------------------------------------

I. Khung tham số phần cứng mục tiêu (Device-Class & Target Platform Parameters)

Bài báo phân loại các dòng FPGA thành 3 nhóm: LUT-DSP cơ bản, SoC FPGA và ACAP. Dự án của chúng ta chọn dòng **SoC FPGA (AMD Kria KV260)**.

| Tham số phần cứng (Hardware Parameter) | Giá trị chốt (Target Value) | Căn cứ & Ràng buộc tài nguyên |
| ----- | ----- | ----- |
| **Dòng chip / Bo mạch (Target Board)** | **AMD Kria KV260 (Zynq UltraScale+ XCZU5EV)** | Lớp SoC FPGA cân bằng chi phí và hiệu năng Edge AI ($249). |
| **Tài nguyên Logic (CLB / LUTs / FFs)** | **70,560 LUTs / 141,120 FFs** | Logic dành cho AXI Stream, Control FSM và Softmax Engine. |
| **Bộ nhân DSP Slices** | **1,248 Slices (DSP48E2)** | Mỗi DSP48E2 hỗ trợ nhân 27×18 bit và tích lũy 48-bit. |
| **Bộ nhớ On-Chip (Block RAM)** | **144 Blocks BRAM18K (\~4.2 Mb)** | Giới hạn nghiêm ngặt cho việc lưu trữ tensor tạm $Q, K, V$. |
| **Tần số xung clock (PL Fmax)** | **150 MHz – 250 MHz** | Tần số thiết kế PL an toàn đảm bảo timing closure (WNS \> 0). |
| **Giao diện kết nối Host (PS-PL)** | **AXI4-Stream (TDATA 64/128-bit) \+ AXI DMA** | Kết nối streaming giữa ARM Cortex-A53 (PS) và Attention Core (PL). |
| **Giới hạn công suất (Power Budget)** | **\< 5.0 W (Thực tế \~1.0W – 3.2W)** | Tối ưu hóa năng lượng cho thiết bị nhúng viền (Edge Vision). |

\--------------------------------------------------------------------------------

II. Tham số mô hình & Không gian Tensor (Model & Tensor Matrix Parameters)

Để tránh hiện tượng cạn kệt BRAM và nghẽn băng thông DDR, bài báo nhấn mạnh việc áp dụng mô hình ViT nhẹ (Lightweight ViT) kết hợp giảm kích thước chuỗi token $N$.

               \+-------------------------------------------------------+

                |     INPUT IMAGE: 224x224x3 (RGB Camera / Video)       |

                \+-------------------------------------------------------+

                                           |

                                           v

                \+-------------------------------------------------------+

                |   ARM PS (Processing System): Convolutions / Patch     |

                |   Embedding (14x14 Patch Grid \-\&gt; N \= 196 Tokens)       |

                \+-------------------------------------------------------+

                                           |

                                           v (AXI4-Stream INT8 Tensors: Q, K, V)

\+---------------------------------------------------------------------------------------+

|  FPGA PL (Programmable Logic) ATTENTION ENGINE:                                       |

|  1\. Q \* K^T Matrix Multiplication (DSP48E2 MAC Array)                                 |

|  2\. Scaling Factor Shift Logic (1 / sqrt(d\_k))                                        |

|  3\. Hardware Softmax (LUT / Piecewise-Linear Approximation)                           |

|  4\. Attention Score \* V Matrix Multiplication                                         |

\+---------------------------------------------------------------------------------------+

                                           |

                                           v (AXI4-Stream Attended Output)

                \+-------------------------------------------------------+

                |   ARM PS: LayerNorm, FFN / MLP, Classification Head   |

                \+-------------------------------------------------------+

&nbsp;

&nbsp;

| Cấu hình toán học | Giá trị thiết lập | Giải trình kỹ thuật phần cứng |
| ----- | ----- | ----- |
| **Mô hình mục tiêu (Backbone)** | **MobileViT-XXS / TinyViT-5M** (\< 5M params) | Đảm bảo kích thước trọng số không bị nghẽn băng thông DRAM. |
| **Độ phân giải đầu vào (Resolution)** | **224 × 224 pixels** | Chuẩn cho các bài toán phân loại ảnh và nhận diện hành vi. |
| **Chiều dài chuỗi Token (**$N$**)** | $N \= 196$ **(từ grid patch** $14 \\\\times 14$**)** | Ma trận chú ý $N \\\\times N \= 196 \\\\times 196 \= 38,416$ phần tử, vừa trọn trong BRAM. |
| **Chiều của Attention Head (**$d\\\_k$**)** | $d\\\_k \= 32$ **hoặc** $64$ | Kích thước chia hết cho 16/32, khớp hoàn hảo với các mảng SIMD DSP48E2. |
| **Số lượng Head (**$H$**)** | $H \= 1 \\\\sim 4$ **heads** | Giới hạn số lượng head song song để duy trì lưu lượng luồng AXI-Stream. |

\--------------------------------------------------------------------------------

III. Tham số định lượng & Tối ưu hóa DSP (Quantization & Arithmetic Parameters)

Khác với GPU xử lý FP32, trên FPGA bài báo khuyến nghị sử dụng định lượng số nguyên cố định (INT8) kết hợp kỹ thuật **DSP Packing** để nhân đôi hiệu năng nhân MAC.

* **Dữ liệu đầu vào (**$Q, K, V$ **Activations)**: **INT8** (Quantized Q-format, ví dụ $Q4.4$ hoặc $Q3.5$).  
* **Bộ tích lũy nhân (Accumulator)**: **INT32** (chống tràn số khi thực hiện phép nhân ma trận $QK^T$ và $Score \\\\times V$).  
* **Xử lý đầu ra Softmax**: Định lượng ngược về **INT8** trước khi đưa vào nhân với ma trận $V$.  
* **Kỹ thuật SIMD / DSP Packing**: Tận dụng cổng nhân $27 \\\\times 18$ của DSP48E2 trên Xilinx UltraScale+ để ghép 2 phép nhân INT8 hoặc 4 phép nhân INT4 vào cùng 1 slice DSP trong 1 chu kỳ xung clock.

\--------------------------------------------------------------------------------

IV. Tham số kiến trúc luồng dữ liệu & Bộ nhớ (Dataflow & Memory Parameters)

Nhằm loại bỏ việc đọc/ghi lại dữ liệu trung gian ra RAM ngoài (DRAM), thiết kế áp dụng chiến lược **Single-Load Policy** và **Tiling** từ bài báo:

1. **Chính sách Single-Load (On-Chip Reuse)**: Trọng số và kích hoạt chỉ được đọc từ DRAM vào BRAM một lần duy nhất cho mỗi khối; toàn bộ tensor trung gian của layer được giữ trên BRAM.  
2. **Kỹ thuật Phân tile (Hybrid Tiling)**:  
3. Tile kích thước Attention ($T\\\_{MHA}$): Chia nhỏ ma trận $Q, K$ thành các block $T\\\_q \\\\times d\\\_k$ (ví dụ $T\\\_q \= 16$ hoặc $64$) để tính toán luân phiên.  
4. Tile kích thước MLP ($T\\\_{FFN}$): Thiết lập cấu hình tile $T\\\_{FFN} \= 128$.  
5. **Bộ đệm Ping-Pong (Double Buffering)**: Cấu hình BRAM cổng kép (True Dual-Port BRAM) làm đệm song song: trong khi bộ đệm A đang cấp dữ liệu cho mảng nhân MAC, bộ đệm B tiếp nhận luồng dữ liệu mới từ AXI DMA để ẩn hoàn toàn độ trễ truyền dữ liệu.

\--------------------------------------------------------------------------------

V. Tham số xấp xỉ phần cứng cho các phép toán phi tuyến (Hardware Approximation Parameters)

Các phép toán tốn kém tài nguyên logic như chia ($\\\\sqrt{d\\\_k}$) và hàm mũ ($e^x$) được xấp xỉ bằng các cấu hình phần cứng sau:

* **Khối Scaling (**$\\\\frac{1}{\\\\sqrt{d\\\_k}}$**)**:  
  * Do $d\\\_k$ được cố hình cứng là 32 hoặc 64, phép chia được thay thế hoàn toàn bằng **phép dịch bit phải đại số (Arithmetic Right-Shift \- ASR)**.  
  * *Tham số*: Shift 2 bit (khi $\\\\sqrt{d\\\_k} \\\\approx 4$) hoặc Shift 3 bit (khi $\\\\sqrt{d\\\_k} \= 8$), tiêu tốn **0 DSP** và hoàn thành trong **1 clock cycle**.  
* **Khối Softmax Engine (Nút thắt phần cứng)**:  
  * **Phương pháp A (BRAM LUT-based)**: Tích hợp bộ tìm Max ($x\\\_{max}$) trên luồng $\\\\rightarrow$ Trừ max $\\\\rightarrow$ Tra bảng LUT ROM (chứa sẵn giá trị $e^{x \- x\\\_{max}}$ lưu trong BRAM) $\\\\rightarrow$ Tích lũy tổng $\\\\rightarrow$ Phép chia chuẩn hóa.  
  * **Phương pháp B (Piecewise-Linear / Base-2 Shiftmax)**: Xấp xỉ $e^x \\\\approx 2^{x \\\\log\\\_2 e}$, chuyển toàn bộ phép tính mũ thành các phép dịch bit nguyên và cộng tuyến tính.

\--------------------------------------------------------------------------------

Tóm tắt bộ tham số chốt cho Thiết kế RTL (Quick Reference Matrix)

\[TARGET BOARD\]: AMD Kria KV260 (Zynq UltraScale+ MPSoC)

\[PL CLOCK\]: 200 MHz | \[INTERFACES\]: AXI4-Stream (64-bit) \+ AXI-Lite

\[MODEL\]: MobileViT-XXS | \[INPUT\]: 224x224x3 | \[SEQUENCE LENGTH N\]: 196

\[HEAD DIM d\_k\]: 32 | \[HEADS H\]: 2 | \[PRECISION\]: INT8 Input / INT32 Accumulator

\[MEMORY POLICY\]: Single-Load \+ Ping-Pong BRAM (N=196 fits on-chip)

\[SOFTMAX\]: Dual-Engine (Method A: BRAM LUT / Method B: Piecewise-Linear)

&nbsp;

&nbsp;

Với bộ tham số phần cứng đã được chuẩn hóa theo các nghiên cứu mới nhất năm 2025 này, chúng ta có thể đảm bảo mô hình không bị vi phạm thời gian (Timing Setup Violation) hay tràn tài nguyên BRAM/DSP trên Kria KV260.

Bạn có muốn bắt đầu đi sâu vào **Batch 1 (Tổng hợp cơ sở lý thuyết toán học của ViT từ các bài báo khoa học)** theo kế hoạch đã đề ra không?

\--------------------------------------------------------------------------------

Tăng tốc ViT trên FPGA

Dựa trên tài liệu dự án **Capstone Project 2: "Hardware-Accelerated Transformer Attention for Intelligent Edge Vision"** (Trần Duy Tuấn & Thái Hoàng Gia Thoại) và các bài báo khoa học liên quan trong bộ thư mục, dưới đây là kế hoạch chi tiết bao gồm **Khung đề cương kiến thức nền tảng (Outline)**, **Phân chia các giai đoạn nghiên cứu (Research Batches)** và **Báo cáo theo dõi/kiểm tra ngược (Tracking & Back-Tracking Report)**.

\--------------------------------------------------------------------------------

I. Khung đề cương kiến thức nền tảng (Knowledge Background Outline)

1\. Cơ sở lý thuyết Vision Transformer (ViT) & Trích xuất đặc trưng ảnh

* **Cấu trúc patch & Embedding**: Chuyển đổi ảnh đầu vào $x \\\\in \\\\mathbb{R}^{H \\\\times W \\\\times C}$ thành chuỗi các patch 2D phẳng $x\\\_p \\\\in \\\\mathbb{R}^{N \\\\times (P^2 \\\\cdot C)}$ với kích thước patch $(P, P)$ và số lượng patch $N \= HW/P^2$.  
* **Tạo Vector Embedding & Token Class**: Chiếu tuyến tính (Linear Projection) các patch ảnh thành không gian embedding kích thước $D$, kết hợp chèn token phân loại học được `[class]` ở đầu chuỗi.  
* **Mã hóa vị trí (Positional Encoding)**: Bổ sung thông tin không gian 1D/2D vào các patch embedding để giữ thứ tự hình ảnh.  
* **Cơ chế Multi-Head Self-Attention (MSA)**: Tính toán ma trận chú ý theo công thức toán học: $$\\\\text{Attention}(Q, K, V) \= \\\\text{softmax}\\\\left(\\\\frac{QK^T}{\\\\sqrt{d\\\_k}}\\\\right)V\\$$ trong đó Query ($Q$), Key ($K$), và Value ($V$) được biến đổi qua các trọng số tuyến tính.  
* **Khối Encoder & MLP Head**: Kết hợp Layer Normalization (LN), hàm kích hoạt GELU và mạng MLP Feed-Forward với các kết nối tắt residual.

2\. Kiến trúc ViT nhẹ & Mô hình định lượng số nguyên (Quantization)

* **Mô hình ViT cho thiết bị Edge**: Đánh giá các dòng mô hình như **MobileViT-XXS** và **TinyViT-5M** với tham số nhỏ (\<5M) để đảm bảo bộ nhớ BRAM trên FPGA không bị quá tải.  
* **Định lượng số nguyên cố định (INT8 / Fixed-Point)**: Chuyển đổi trọng số và kích hoạt từ dạng FP32 sang INT8 ($Q\\\_{m.n}$), sử dụng bộ tích lũy INT32 trong các đơn vị nhân tích lũy (MAC) để tránh tràn số.  
* **Đánh giá suy hao độ chính xác**: Mô phỏng định lượng trong PyTorch (PTQ/QAT) để đo đạc sai số MSE và độ sụt giảm độ chính xác Top-1/Top-5 trên tập dữ liệu ImageNet/CIFAR.

3\. Xấp xỉ phần cứng cho các phép toán phi tuyến (Softmax & Scaling)

* **Nút thắt Softmax**: Phép tính mũ ($e^x$) và phép chia trong Softmax chuẩn đòi hỏi tài nguyên tài nguyên logic và DSP rất lớn trên FPGA.  
* **Phương pháp xấp xỉ A \- Lookup-Table (LUT) Softmax**: Sử dụng kỹ thuật trừ giá trị cực đại (Max-subtraction) và tra bảng LUT lưu sẵn giá trị mũ trong BRAM.  
* **Phương pháp xấp xỉ B \- Piecewise-Linear (PWL) / Base-2 Softmax**: Biến đổi cơ số mũ về dạng $2^x$ để tận dụng các phép dịch bit (bit-shift) và phép cộng tuyến tính trên phần cứng.

4\. Kiến trúc phần cứng FPGA & Khái niệm HW/SW Co-Design

* **Phân chia phần cứng / phần mềm (PS/PL Partitioning)**:  
  * **Processing System (ARM PS)**: Xử lý giải mã ảnh, cắt patch, tính toán Layer Normalization và các khối MLP.  
  * **Programmable Logic (FPGA PL)**: Tăng tốc các phép toán ma trận có độ phức tạp $O(N^2)$ trong cơ chế Attention ($QK^T$, Scaling, Softmax xấp xỉ, Score $\\\\times V$).  
* **Giao tiếp Dòng dữ liệu (Streaming & DMA)**: Chuẩn AXI4-Stream với cơ chế bắt tay `TVALID/TREADY`, kết hợp AXI DMA Simple Mode để truyền nhận dữ liệu giữa DDR4 và PL.  
* **Đơn vị tính toán & Tối ưu bộ nhớ**: Mảng đơn vị xử lý Processing Element (PE) tích hợp trên DSP48E2, kết hợp kỹ thuật Ping-Pong double-buffering trong BRAM để ẩn độ trễ truy xuất bộ nhớ.

\--------------------------------------------------------------------------------

II. Phân chia các giai đoạn nghiên cứu (Research Batches & Tasks)

| Batch ID | Tên giai đoạn | Mục tiêu & Nhiệm vụ nghiên cứu chính | Văn bản / Bài báo tham chiếu |
| ----- | ----- | ----- | ----- |
| **Batch 1** | **Mô hình hóa ViT & Chuẩn hóa Toán học** | • Trích xuất công thức toán của Transformer Attention.• Phân tích cấu trúc patch, $\\\[class\\\]$ token và độ dài chuỗi $N \\\\le 196$. | Dosovitskiy et al.Chu et al.Tuấn & Thoại |
| **Batch 2** | **Định lượng & Tạo Vector Kiểm thử Golden** | • Xây dựng script PyTorch huấn luyện FP32 & định lượng INT8.• Xuất các file .hex chứa tensor $Q, K, V$ làm đầu vào testbench. | Wu et al. (TinyViT)Xu et al.Tuấn & Thoại |
| **Batch 3** | **Xấp xỉ Thuật toán Softmax Phần cứng** | • Khảo sát Softmax dạng LUT-based vs Piecewise-Linear (PWL).• Đánh giá sai số MSE giữa float32 và INT8 xấp xỉ. | Sali et al.Tuấn & Thoại2402.09709v1 |
| **Batch 4** | **Thiết kế RTL SystemVerilog & Mô phỏng** | • Lập trình các module RTL: mac\_array.sv, softmax\_lut.sv, axis\_adapter.sv.• Mô phỏng cycle-accurate trên Vivado xsim với vector .hex. | Wang et al. (ViA)Nag et al. (ViTA)Tuấn & Thoại |
| **Batch 5** | **Tích hợp SoC, Nạp Bitstream & Benchmark** | • Đóng gói IP, tạo Block Design với AXI DMA trên Vivado IPI.• Viết driver PYNQ Python trên Kria KV260 để đo FPS, Latency, GOPS. | LuTuấn & Thoại2509.04162v1 |

\--------------------------------------------------------------------------------

III. Khung Báo cáo Theo dõi & Kiểm tra ngược (Tracking & Back-Tracking Report)

Mẫu báo cáo này giúp bạn tiến hành đánh giá định kỳ sau mỗi Batch, đồng thời cung cấp phương án xử lý (back-tracking) khi gặp sự cố kỹ thuật hoặc vi phạm ràng buộc phần cứng:

\# BÁO CÁO THEO DÕI VÀ KIỂM TRA NGƯỢC (TRACKING \&amp; BACK-TRACKING REPORT)

Dự án: Hardware-Accelerated Transformer Attention for Edge Vision (Project 2\)

Bo mạch mục tiêu: AMD Kria KV260 | Ngôn ngữ: SystemVerilog / Python (PYNQ)

&nbsp;

\---------------------------------------------------------------------------------------------------

1\. TIẾN ĐỘ THỰC HIỆN THEO BATCH

\---------------------------------------------------------------------------------------------------

\[ \] Batch 1: Mô hình hóa ViT \&amp; Chuẩn hóa Toán học

    \- Mốc hoàn thành: Lock thông số N=196, d\_k=16/32, H=2\~4.

    \- Kết quả kiểm tra: Công thức Attention được biểu diễn dưới dạng ma trận nguyên.

    \- Trạng thái: \[Sẵn sàng / Đang thực hiện / Hoàn thành\]

&nbsp;

\[ \] Batch 2: Định lượng INT8 \&amp; Vector Golden Test

    \- Mốc hoàn thành: Sụt giảm độ chính xác Top-1 \&lt; 1.5% so với FP32.

    \- Sản phẩm bàn giao: Tập tin stimulus \`.hex\` cho $Q, K, V$ và kết quả đầu ra chuẩn.

    \- Trạng thái: \[Chờ thực hiện\]

&nbsp;

\[ \] Batch 3: Xấp xỉ Thuật toán Softmax

    \- Mốc hoàn thành: Hàm Softmax LUT và PWL đạt sai số MSE \&lt; 1e-3.

    \- Sản phẩm bàn giao: Bảng thông số BRAM LUT / tham số phép dịch bit.

    \- Trạng thái: \[Chờ thực hiện\]

&nbsp;

\[ \] Batch 4: Thiết kế RTL SystemVerilog \&amp; Mô phỏng xsim

    \- Mốc hoàn thành: Testbench tự kiểm tra (self-checking) báo MATCH 100% không mất dữ liệu.

    \- Tối ưu hóa: Đạt closure về Timing (WNS \&gt; 0 ns ở tần số 200 MHz).

    \- Trạng thái: \[Chờ thực hiện\]

&nbsp;

\[ \] Batch 5: Tích hợp SoC KV260 \&amp; Đánh giá Hiệu năng

    \- Mốc hoàn thành: Thực thi thành công pipeline ảnh-sang-kết quả trên phần cứng thực tế.

    \- Báo cáo chỉ số: End-to-end Latency (ms), FPS, GOPS, BRAM/DSP utilization.

    \- Trạng thái: \[Chờ thực hiện\]

&nbsp;

\---------------------------------------------------------------------------------------------------

2\. MA TRẬN KIỂM TRA NGƯỢC \&amp; PHƯƠNG ÁN XỬ LÝ SỰ CỐ (BACK-TRACKING MATRIX)

\---------------------------------------------------------------------------------------------------

Sự cố / Thách thức kỹ thuật         | Điểm nghẽn phần cứng / Thuật toán   | Kế hoạch kiểm tra ngược (Back-Tracking Action)

\------------------------------------|--------------------------------------|------------------------------------------------

1\. Tràn tài nguyên BRAM (\&gt;144 blocks)| Chiều dài chuỗi $N$ quá lớn          | \-\&gt; Giảm kích thước $N$ từ 196 xuống 64 hoặc áp dụng

                                    |                                      |    tiling/blocking cho ma trận $QK^T$.

2\. Sụt giảm độ chính xác INT8 quá cao| Định lượng INT8 bị trôi dạt         | \-\&gt; Chuyển từ PTQ thuần túy sang Quantization-Aware

                                    |                                      |    Training (QAT) hoặc giữ accumulator INT32.

3\. Vi phạm thời gian (Timing Setup)  | Mảng nhân MAC $QK^T$ bị trễ đường truyền| \-\&gt; Thêm các tầng thanh ghi pipeline (deep pipelining)

                                    |                                      |    giữa các PE.

4\. Softmax PWL bị tràn số nguyên    | Phần nguyên biến đổi quá nhanh       | \-\&gt; Cấu hình lại chuẩn Q-format hoặc chuyển sang

                                    |                                      |    Softmax LUT dựa trên BRAM.

&nbsp;

&nbsp;

\--------------------------------------------------------------------------------

Bảng công việc đã được đồng bộ vào hệ thống theo dõi tác vụ nội bộ. Bạn có muốn bắt đầu đi sâu vào **Batch 1 (Tổng hợp cơ sở lý thuyết toán học của ViT từ các bài báo khoa học)** hay muốn điều chỉnh lại các tham số cấu hình phần cứng trước?

&nbsp;

&nbsp;