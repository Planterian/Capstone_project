# Báo Cáo Phân Tích Hệ Thống Triển Khai Vivado KV260 (mha_kv260_deploy)

---

## 1. Giới Thiệu Tổng Quan Thư Mục `vivado_kv260`

Thư mục `c:\Users\tuan2\Desktop\vivado_kv260` là không gian làm việc (workspace) chuyên biệt phục vụ cho giai đoạn **triển khai thực tế trên phần cứng (Hardware Deployment & Hardware-in-the-Loop)** của lõi gia tốc **Multi-Head Attention (MHA)** trên nền tảng **AMD Kria KV260 Vision AI Starter Kit** (sử dụng chip SoC **Zynq UltraScale+ MPSoC `xck26-sfvc784-2LV-c`**).

Thư mục gốc bao gồm 2 phân hệ chính:
1. **`mha_kv260_deploy/`**: Thư mục dự án Vivado (Vivado Project Workspace). Chứa toàn bộ cấu trúc tích hợp hệ thống phần cứng trên chip (System-on-Chip - SoC) thông qua công cụ Vivado IP Integrator với Block Design mang tên **`mha_system`**.
2. **`pynq_driver/`**: Thư mục lưu trữ driver phần mềm máy chủ (Host Software Drivers) chạy trên môi trường **Ubuntu / PYNQ Linux (ARM Cortex-A53 PS)**, phối hợp cùng các tệp bitstream nạp vi mạch (`attention_core_top.bit`) và tệp đặc tả phần cứng (`attention_core_top.hwh`).

---

## 2. Kiến Trúc Toàn Cục Block Design `mha_system` (HW/SW Co-Design)

Cấu trúc cây thư mục trong `mha_kv260_deploy.srcs/sources_1/bd/mha_system/ip/` phản ánh trung thực một thiết kế SoC hoàn chỉnh theo mô hình **Đồng thiết kế Phần cứng / Phần mềm (HW/SW Co-Design)** giữa phân hệ xử lý ARM (PS) và phân hệ logic khả trình FPGA (PL):

```text
===================================================================================================================
                                KIẾN TRÚC TÍCH HỢP HỆ THỐNG MHA TRÊN AMD KRIA KV260
===================================================================================================================

       +-----------------------------------------------------------------------------------------+
       |                  ARM PROCESSING SYSTEM (PS) - Zynq UltraScale+ MPSoC                    |
       |                  (mha_system_zynq_ps_0: Quad Cortex-A53 @ 1.33 GHz)                     |
       |                                                                                         |
       |   [ Ứng dụng PYNQ / Python ]                [ Bộ nhớ DDR4 Hệ Thống (4GB PS DDR4) ]     |
       |   - Driver AXI DMA & CSR MMIO               - Khối đệm liên tục (CMA Buffers)           |
       |   - Thu nhận Camera OpenCV                  - Lưu trữ ma trận Q, K, V (18,816 Bytes)    |
       |   - Hậu xử lý MLP & Hiển thị UI             - Lưu kết quả Attention Context (6,272 B)   |
       +--------------+-------------------------------------------+------------------------------+
                      | M_AXI_HPM0_FPD (AXI-Lite Control)         | S_AXI_HP0_FPD (High-Perf 128-bit)
                      | (Thanh ghi điều khiển)                   | (Truy xuất DDR4 băng thông cao)
                      v                                           ^
       +-------------------------------+                          |
       | mha_system_axi_interconnect_0 |                          |
       | (Interconnect: xbar + auto_pc)|                          |
       +-------+---------------+-------+                          |
               |               |                                  |
               | S_AXI_LITE    | S_AXI (CSRs)                     |
               v               |                                  |
       +---------------+       |                     +------------+--------------------+
       |  axi_dma_0_0  |       |                     |     mha_system_axi_smc_hp_0     |
       |  (AXI DMA)    |       |                     |     (AXI SmartConnect Bridge)   |
       +-------+-------+       |                     +------------+--------------------+
               |               |                                  ^
               | MM2S/S2MM     |                                  | M_AXI_MM2S & M_AXI_S2MM
               | Streams       |                                  | (Truyền đọc/ghi DMA)
               v               v                                  |
       +-------+---------------+----------------------------------+--------------------+
       |                                                                               |
       |                  mha_system_mha_core_0 (Module Ref: attention_core_top)       |
       |                  [ KHỐI GIA TỐC PHẦN CỨNG MULTI-HEAD ATTENTION (PL) ]         |
       |                                                                               |
       |   ├── AXI4-Lite CSR: 0x00=CTRL, 0x04=NBYTES, 0x10=N, 0x18=D_K, 0x20=SHIFT     |
       |   ├── S_AXIS: Kênh nhận luồng Q, K, V từ DMA MM2S -> Nạp Input Ping-Pong BRAM |
       |   ├── Engine Tính Toán:                                                       |
       |   │     ├── Systolic MAC Array 1: Tính Q * K^T (INT8 x INT8 -> INT32)        |
       |   │     ├── Scaler Unit: Chuẩn hóa đại số 1/sqrt(d_k) bằng ASR Barrel Shifter |
       |   │     ├── Hardware Softmax: Tra bảng LUT / Xấp xỉ PWL -> Xác suất INT8     |
       |   │     └── Systolic MAC Array 2: Tính Softmax_Scores * V -> Kết quả INT8     |
       |   └── M_AXIS: Kênh đẩy kết quả Attended Context ra DMA S2MM -> Ghi về DDR4    |
       |                                                                               |
       +-------------------------------------------------------------------------------+

       +-----------------------------------+       +-----------------------------------+
       |   mha_system_proc_sys_reset_0_0   |       |      mha_system_irq_concat_0      |
       |   (Đồng bộ hóa tín hiệu Reset     |       |   (Gộp 2 đường ngắt s2mm_introut  |
       |    từ PS pl_resetn0 cho toàn PL)  |       |    và mm2s_introut vào pl_ps_irq0)|
       +-----------------------------------+       +-----------------------------------+
```

---

## 3. Danh Mục & Chức Năng Từng Khối IP Trong Block Design

Dựa trên vết cấu trúc thư mục con tại `mha_kv260_deploy.srcs/sources_1/bd/mha_system/ip/`:

| Tên Khối IP | Loại IP Chuẩn / Phân Loại | Vai Trò & Chức Năng Kỹ Thuật Trong Hệ Thống |
| :--- | :--- | :--- |
| **`mha_system_zynq_ps_0`** | `zynq_ultra_ps_e` (Zynq MPSoC) | Lõi điều khiển trung tâm trên Kria K26 SOM. Chạy Linux/PYNQ, quản lý bộ nhớ ngoài DDR4, cấp phát xung nhịp hệ thống (`pl_clk0` @ 100/200 MHz), tạo reset hệ thống (`pl_resetn0`), tiếp nhận tín hiệu ngắt phần cứng (`pl_ps_irq0`), và làm AXI Master để cấu hình các thanh ghi CSR qua cổng `M_AXI_HPM0_FPD`. |
| **`mha_system_mha_core_0`** | RTL Module Reference (`attention_core_top`) | Lõi gia tốc phần cứng MHA độc quyền của đề tài. Tích hợp trực tiếp mã RTL SystemVerilog (`attention_core_top.sv`). Chứa bộ điều khiển AXI-Lite, bộ nhận/phát AXI4-Stream, bộ đệm Ping-Pong BRAM, 2 mảng tính tích vô hướng Systolic/Vector Streaming, khối chuẩn hóa Scaler ASR và khối phi tuyến Softmax phần cứng. |
| **`mha_system_axi_dma_0_0`** | `axi_dma` (Xilinx LogiCORE IP PG021) | Động cơ DMA Direct Register / Scatter-Gather trực tiếp di chuyển dữ liệu lớn giữa bộ nhớ vật lý DDR4 của ARM PS và lõi gia tốc PL mà không làm tắc nghẽn CPU Cortex-A53. Kênh MM2S đẩy dữ liệu $Q, K, V$ vào lõi; kênh S2MM kéo ma trận kết quả ngữ cảnh trả về DDR4. |
| **`mha_system_axi_smc_hp_0`** | `smartconnect` (AXI SmartConnect) | Cầu nối chuyển đổi giao thức và tốc độ bus hiệu năng cao. Gom 2 cổng Master tốc độ cao của DMA (`M_AXI_MM2S` và `M_AXI_S2MM`) vào cổng Slave hiệu năng cao `S_AXI_HP0_FPD` (High Performance 128-bit) của Zynq MPSoC, tối ưu hóa băng thông truyền dữ liệu bộ nhớ liên tục (CMA). |
| **`mha_system_axi_interconnect_ctrl_0`** | `axi_interconnect` | Mạng chuyển mạch phân phối điều khiển Memory-Mapped. Nối cổng Master AXI-Lite của ARM PS (`M_AXI_HPM0_FPD`) tới các cổng Slave AXI-Lite của `axi_dma_0` (thanh ghi DMA) và `mha_core_0` (thanh ghi CSR). |
| **`...imp_xbar_0`** | `axi_crossbar` | Khối chuyển mạch ma trận nội bộ trong AXI Interconnect, thực hiện giải mã không gian địa chỉ để định tuyến lệnh đọc/ghi từ PS tới đúng địa chỉ IP đích. |
| **`...imp_auto_pc_0`** | `axi_protocol_converter` | Bộ chuyển đổi giao thức tự động (chuyển từ chuẩn AXI4 đầy đủ sang AXI4-Lite tinh giản) nhằm tiết kiệm tài nguyên logic trên FPGA. |
| **`mha_system_proc_sys_reset_0_0`** | `proc_sys_reset` | Khối sinh và đồng bộ tín hiệu Reset. Chuyển đổi tín hiệu reset không đồng bộ từ PS thành tín hiệu reset đồng bộ cực tính thấp (`peripheral_aresetn`, `interconnect_aresetn`) tương thích với xung clock `pl_clk0` của PL. |
| **`mha_system_irq_concat_0`** | `xlconcat` | Khối ghép đường ngắt (Interrupt Concatenation). Ghép 2 đường ngắt độc lập `mm2s_introut` (hoàn tất truyền phát) và `s2mm_introut` (hoàn tất tiếp nhận) của AXI DMA thành bus ngắt vector 2-bit đưa vào cổng `pl_ps_irq0` của ARM PS để kích hoạt hàm xử lý ngắt (ISR) trong kernel Linux. |

---

## 4. Bản Đồ Thanh Ghi Điều Khiển AXI4-Lite (CSR Specification)

Theo thiết kế phần cứng của lõi MHA (`attention_core_top` / `axi_lite_regs`), các thanh ghi cấu hình được ánh xạ địa chỉ MMIO như sau:

| Địa chỉ Offset | Tên Thanh Ghi | Quyền | Giá Trị Tiêu Chuẩn (ViT / Patch 14x14) | Mô Tả Chức Năng |
| :---: | :---: | :---: | :---: | :--- |
| `0x00` | `REG_CTRL` | R/W | `0x00000001` (Pulse START) | **Thanh ghi điều khiển chính**.<br>• Bit 0: `START` (Kích hoạt lõi MHA bắt đầu xử lý).<br>• Bit 1: `CLEAR` (Xóa mềm trạng thái/Reset pipeline). |
| `0x04` | `REG_NBYTES` | R/W | `18,816` ($196 \times 32 \times 3$) | **Tổng số byte dữ liệu đầu vào** của khối $Q, K, V$ (INT8) cần nạp qua luồng DMA MM2S. |
| `0x0C` | `REG_STATUS` | RO | `Bit 0: BUSY, Bit 1: DONE` | **Thanh ghi trạng thái hoạt động**.<br>• Bit 0 = 1: Lõi đang bận tính toán.<br>• Bit 1 = 1: Lõi đã hoàn thành phép tính Attention cho toàn bộ chuỗi token. |
| `0x10` | `REG_TOKENS` | R/W | `196` ($N = 14 \times 14$) | **Số lượng token trong chuỗi (Sequence Length $N$)**.<br>Mỗi ảnh 224x224 chia thành $14 \times 14 = 196$ patches. |
| `0x18` | `REG_HDIM` | R/W | `32` ($d_k = 32$) | **Kích thước một Attention Head (Head Dimension $d_k$)**.<br>Mỗi vector biểu diễn token gồm 32 phần tử INT8. |
| `0x20` | `REG_SHIFT` | R/W | `2` ($S = 2$) | **Hệ số dịch bit phải đại số (ASR)** của bộ Scaler Unit.<br>Tương đương phép nhân gần đúng với hệ số tỉ lệ $1/\sqrt{d_k} = 1/\sqrt{32} \approx 1/5.66$. |

---

## 5. Cơ Chế Truyền Nhận Dữ Liệu Bộ Nhớ Liên Tục (CMA & DMA Streaming)

Hệ thống tận dụng cơ chế **Contiguous Memory Allocation (CMA)** trên nhân Linux PYNQ để đạt thông lượng bộ nhớ tối đa:

### 5.1. Định dạng gói dữ liệu vào (Ingress Buffer - InBuffer)
* Kích thước mảng: $N \times d_k \times 3 = 196 \times 32 \times 3 = 18,816 \text{ Bytes}$ (Kiểu số nguyên có dấu `int8`).
* Cấu trúc ghép kênh: Các vector $Q$, $K$, và $V$ của từng token được sắp xếp liên tiếp trong bộ nhớ DDR4 trước khi kích hoạt kênh DMA MM2S truyền xuống bộ đệm Ping-Pong BRAM trên FPGA.

### 5.2. Định dạng gói dữ liệu ra (Egress Buffer - OutBuffer)
* Kích thước mảng: $N \times d_k = 196 \times 32 = 6,272 \text{ Bytes}$ (Kiểu số nguyên có dấu `int8`).
* Kết quả sau khi tính ma trận xác suất Attention và nhân chập với ma trận $V$:
  $$\text{Context} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \times V$$
* Dữ liệu được đẩy ra bus AXI4-Stream Master của vi mạch, qua kênh DMA S2MM ghi thẳng về mảng `out_buffer` trên RAM DDR4.

### 5.3. Trình tự bắt tay phần cứng (Execution Protocol)
1. **Host PS**: Cấp phát `in_buffer` và `out_buffer` bằng hàm `pynq.allocate`.
2. **Host PS**: Ghi cấu hình các thanh ghi $N=196$, $d_k=32$, $S=2$, $N_{\text{bytes}}=18,816$ qua AXI-Lite CSR.
3. **Host PS**: Khởi động kênh nhận DMA S2MM (`dma.recvchannel.transfer(out_buffer)`).
4. **Host PS**: Khởi động kênh truyền DMA MM2S (`dma.sendchannel.transfer(in_buffer)`).
5. **Host PS**: Kích xung `START = 1` tại địa chỉ thanh ghi `0x00`.
6. **Hardware PL**: Lõi MHA tự động nạp BRAM $\rightarrow$ tính toán theo đường ống $\rightarrow$ xuất luồng dữ liệu ra AXI-Stream.
7. **Host PS**: Đợi tín hiệu ngắt DMA hoàn tất (`dma.wait()`), đọc trực tiếp kết quả từ `out_buffer` mà không phải tốn chu kỳ sao chép bộ nhớ (Zero-Copy Transfer).

---

## 6. Phân Tích Hiện Trạng Thư Mục `vivado_kv260`

Qua quét kiểm tra toàn bộ cây thư mục `c:\Users\tuan2\Desktop\vivado_kv260`:
* **Số lượng thư mục con**: **220 thư mục** cấu trúc chuẩn của Vivado 2026.
* **Số lượng file**: Hiện tại bên trong thư mục này chứa **0 file** (toàn bộ các tệp `.bd`, `.xci`, `.bit`, mã nguồn RTL, tệp nhật ký `.log`, `.jou` hoặc báo cáo `.rpt` đã bị lược bỏ hoặc chỉ được tạo ra dạng khung xương thư mục từ máy `DESKTOP-297OVLQ`).
* **Vết tích môi trường thực thi**:
  * Thư mục `mha_kv260_deploy.runs/synth_1/.Xil/Vivado-7924-DESKTOP-297OVLQ` cho thấy dự án Vivado từng được khởi tạo và chạy synthesis trên máy trạm mang tên **`DESKTOP-297OVLQ`**.
  * Trong khi đó, máy tính hiện tại đang phát triển là **`DESKTOP-L58T74E`**.
  * Toàn bộ mã nguồn RTL chuẩn hóa (Variant 1A, Variant 2A, Variant 3B), các script synthesis (`run_synth_kv260.bat`), ràng buộc chân/xung nhịp (`timing_kv260.xdc`), và mã driver Python hoàn chỉnh (`deploy_kv260.py`, `mha_kv260_demo.py`) hiện đang được lưu trữ toàn vẹn và quản lý chính xác trong kho lưu trữ **`c:\Users\tuan2\Desktop\Capstone_Project`**.

---

## 7. Khuyến Nghị & Quy Trình Tái Thiết Lập Hệ Thống Triển Khai

Để hoàn thiện quy trình triển khai từ mã nguồn `Capstone_Project` sang bo mạch vật lý AMD Kria KV260:

### 📌 Bước 1: Khởi tạo lại Vivado Project & Block Design Tự Động
Sử dụng công cụ Vivado 2026.1 trên máy trạm hiện tại (`C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat`) để đóng gói RTL `attention_core_top` thành IP hoặc import dưới dạng Module Reference:
1. Đặt mục tiêu phần cứng: Board `AMD Kria KV260 Vision AI Starter Kit` (Part: `xck26-sfvc784-2LV-c`).
2. Thêm các IP Core: `zynq_ultra_ps_e`, `axi_dma`, `smartconnect`, `axi_interconnect`, `proc_sys_reset`, `xlconcat`.
3. Thêm lõi MHA từ thư mục RTL của Variant tối ưu nhất (Khuyến nghị **Variant 2A** hoặc **Variant 3B** vì Variant 1A gặp vấn đề về Negative Slack trên Softmax pipeline).
4. Kết nối bus tự động (Block Automation & Connection Automation).

### 📌 Bước 2: Biên Dịch Ra Bitstream & Hardware Handoff
Chạy quy trình:
$$\text{Generate Block Design} \longrightarrow \text{Run Synthesis} \longrightarrow \text{Run Implementation} \longrightarrow \text{Generate Bitstream}$$
Kết quả đầu ra cần thu được:
* `attention_core_top.bit`: Tệp bitstream nạp cấu hình logic vào FPGA PL.
* `attention_core_top.hwh`: Tệp mô tả phần cứng (Hardware Handoff XML) để thư viện PYNQ tự động nhận diện thanh ghi và cổng DMA.

### 📌 Bước 3: Đồng Bộ Vào Thư Mục `pynq_driver/` & Chuyển Sang KV260
Copy 3 tệp thiết yếu vào thư mục `vivado_kv260/pynq_driver/` và chuyển qua bo mạch KV260 (qua SCP/SFTP):
1. `attention_core_top.bit`
2. `attention_core_top.hwh`
3. `deploy_kv260.py` (Script chạy thực nghiệm)

### 📌 Bước 4: Chạy Suy Luận Trên Bo Mạch Kria KV260
Từ terminal của Kria KV260 (kết nối qua SSH `ubuntu@192.168.1.179` hoặc jump host lab `100.124.168.26`):
```bash
cd ~/mha_driver
sudo python3 deploy_kv260.py
```
Hệ thống sẽ nạp overlay, khởi chạy AXI DMA, đo đạc độ trễ phần cứng từng frame và in kết quả FPS thời gian thực.
