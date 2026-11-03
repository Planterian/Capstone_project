# Hướng Dẫn Quy Trình Làm Việc Vivado GUI (AMD Kria KV260 - MHA Variant 2A)

> **Mục tiêu tài liệu**: Tài liệu hóa quy trình xây dựng, cấu hình, kiểm tra và thu thập bằng chứng thị giác (Visual Evidence) trên môi trường **Vivado GUI** phục vụ cho **bài báo cáo tốt nghiệp / đồ án Capstone** của hệ thống SoC gia tốc Multi-Head Attention (MHA) Variant 2A (`attention_core_top_2A`) trên kit **AMD Kria KV260 Vision AI Starter Kit**.

---

## 1. Tổng Quan Kỹ Thuật Hệ Thống

| Thuộc Tính | Thông Số Thiết Kế | Ghi Chú Kỹ Thuật |
| :--- | :--- | :--- |
| **Nền tảng mục tiêu** | AMD Kria KV260 Starter Kit | Chip Zynq UltraScale+ MPSoC `xck26-sfvc784-2LV-c` |
| **Lõi gia tốc (PL)** | Variant 2A (`attention_core_top_2A`) | Streaming Vector Engine + DSP Packing INT8xINT8 |
| **Xung nhịp hệ thống** | **150.000 MHz** (`pl_clk0`) | Cấp từ Zynq PS PLL, đảm bảo Timing Closure (Slack dương) |
| **Kênh dữ liệu DMA** | AXI DMA v7.1 (Direct Register Mode) | Bus MM2S / S2MM rộng **128 bit** (16 tokens INT8/chu kỳ) |
| **Cầu nối bộ nhớ DDR4**| AXI SmartConnect v1.0 | Nối vào cổng High-Performance `S_AXI_HP0_FPD` (128-bit) |
| **Kênh điều khiển MMIO**| AXI SmartConnect v1.0 (`axi_smc_ctrl`) | Giao tiếp từ `M_AXI_HPM0_FPD` (tối ưu hóa đường dẫn, loại bỏ giới hạn 260 ký tự) |
| **Bản đồ địa chỉ** | `0xA000_0000` (DMA), `0xA001_0000` (MHA) | Phân vùng 64 KB cho mỗi IP Slave |

---

## 2. Quy Trình 1: Phương Pháp "GUI Automation" Toàn Diện (Khuyên Dùng)

Phương pháp này kết hợp sức mạnh tự động hóa của script TCL với giao diện đồ họa trực quan của Vivado GUI. Script `setup_kv260_gui_2A.tcl` được thiết kế để tự động hóa trọn gói: **Dựng Block Design SoC $\to$ Tạo Wrapper $\to$ Tổng hợp (Synthesis) $\to$ Định tuyến (Implementation) $\to$ Tạo Bitstream $\to$ Xuất Artifacts Overlay $\to$ Mở Implemented Design & Block Design ngay trên GUI** (không đóng Vivado).

### Bước 1: Khởi động Vivado GUI
- Mở Vivado (phiên bản 2022.2+ hoặc 2026.1) từ Start Menu hoặc màn hình Desktop.

### Bước 2: Chạy Script Tự Động Từ Tcl Console
- Ở thanh **Tcl Console** dưới đáy màn hình Vivado, nhập lệnh sau và nhấn `Enter`:
  ```tcl
  cd C:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/synthesis
  source setup_kv260_gui_2A.tcl
  ```

> [!TIP]
> **Chế độ tùy chọn (Nếu chỉ muốn dừng lại xem Block Design trước khi build):**
> Trước khi `source`, bạn có thể gõ `set AUTO_BUILD_BITSTREAM 0` vào Tcl Console. Script sẽ dừng lại sau khi dựng xong Block Design để bạn ngắm layout. Khi sẵn sàng sinh Bitstream, chỉ cần gõ lệnh `run_generate_bitstream`.

### Bước 3: Quy Trình Diễn Ra Hoàn Toàn Tự Động Trên Vivado GUI
1. **Khởi tạo Project & Nạp Source**:
   - Vivado tự tạo project `mha_kv260_deploy` tại thư mục `./vivado_prj_2A`.
   - Nạp toàn bộ RTL SystemVerilog (`attention_core_top_2A.sv`, `dsp_packed_adder_tree.sv`, `softmax_lut.sv`,...).
   - Đăng ký bảng tra cứu Softmax LUT `exp_rom_all_stages.hex`.
   - Nạp ràng buộc định thời hệ thống `timing_kv260_system.xdc` (loại bỏ vi phạm quasi-static false path).
2. **Dựng Block Design `mha_system`**:
   - Tự động tích hợp IP: `zynq_ps` (150 MHz `pl_clk0`, HP0 128-bit, HPM0 AXI-Lite), `axi_dma_0` (128-bit Direct Register Mode), `axi_smc_hp` (128-bit), `axi_smc_ctrl`, `irq_concat`, và RTL Module Reference `mha_core_0`.
   - Tự nối toàn bộ bus AXI4-Stream 128-bit, AXI4 128-bit, AXI-Lite 32-bit, xung nhịp 150 MHz và ngắt IRQ.
   - Phân bổ địa chỉ: `0xA000_0000` (DMA) và `0xA001_0000` (MHA Core).
   - Tự động căn chỉnh layout (`regenerate_bd_layout`) và kiểm tra hợp lệ (`validate_bd_design`).
3. **Sinh Top-Level HDL Wrapper**:
   - Tự tạo file `mha_system_wrapper.v` và gán làm Top của project.
4. **Synthesis, Implementation & Write Bitstream**:
   - Chạy tự động với **`jobs = 4`** (`set_param general.maxThreads 4`) bảo đảm tốc độ cao và giữ nhiệt độ máy tính an toàn (~79°C).
   - Tự động đợi quá trình chạy hoàn tất (`wait_on_run impl_1`).
5. **Xuất Hardware Handoff & Overlay**:
   - Tự động copy `attention_core_top_2A.bit`, `attention_core_top_2A.hwh`, và `attention_core_top_2A.xsa` sang `Project/src/Software/` và các thư mục overlay.
6. **Mở Trực Quan Trên Màn Hình GUI (Không Đóng Ứng Dụng)**:
   - Mở sẵn cửa sổ **Implemented Design** (Device Floorplan hiển thị vị trí các khối DSP/BRAM/LUT).
   - Nạp sẵn bảng báo cáo **Timing Summary** (`timing_1`) và **Utilization** (`utilization_1`).
   - Mở sẵn tab sơ đồ **Block Design Diagram** (`mha_system.bd`) để sinh viên chụp hình ảnh minh chứng.

---

## 3. Checklist 6 Hình Ảnh "Đắt Giá" Cần Thu Thập Cho Báo Cáo

Khi Block Design đã sẵn sàng trên màn hình GUI, sinh viên tiến hành chụp lại 6 hình ảnh sau để chèn vào tài liệu thuyết minh đồ án / slide thuyết trình:

| STT | Tên Hình Ảnh Minh Chứng | Vị Trí / Thao Tác Mở Trên GUI | Giá Trị & Ý Nghĩa Kỹ Thuật Đối Với Hội Đồng Chấm |
| :---: | :--- | :--- | :--- |
| **1** | **Toàn cảnh Block Design (`mha_system.bd`)** | Mở tab **Diagram** chính của Block Design. Nhấn `F6` để xuất hiện thanh thông báo *"Validation successful"*. | Minh chứng kiến trúc **HW/SW Co-Design**: liên kết đồng bộ giữa ARM Cortex-A53 PS $\leftrightarrow$ AXI DMA 128-bit $\leftrightarrow$ Lõi gia tốc MHA PL. |
| **2** | **Bảng Phân Bổ Địa Chỉ (Address Editor)** | Click tab **Address Editor** (nằm ngay cạnh tab *Diagram*). | Chứng minh không gian địa chỉ vật lý MMIO chuẩn xác:<br>• `axi_dma_0`: `0xA000_0000` (Size 64KB)<br>• `mha_core_0`: `0xA001_0000` (Size 64KB) |
| **3** | **Cửa Sổ Cấu Hình IP AXI DMA** | Nhấp đúp chuột (Double click) vào khối **`axi_dma_0`** trên sơ đồ. | Chứng minh giải pháp DMA Streaming hiệu năng cao:<br>• Vô hiệu hóa Scatter-Gather Engine (*Direct Register Mode*).<br>• Stream Data Width & Memory Map Width = **128 bit** (16 INT8 tokens/beat). |
| **4** | **Cửa Sổ Cấu Hình Zynq MPSoC** | Nhấp đúp chuột vào khối **`zynq_ps`** $\rightarrow$ xem mục *Clock Configuration* và *PS-PL Configuration*. | Chứng minh cấp phát xung nhịp **150.000 MHz** (`pl_clk0`) và kích hoạt bus dữ liệu High-Performance **128-bit** `S_AXI_HP0_FPD` giao tiếp DDR4. |
| **5** | **Báo Cáo Timing Summary (Sau Implementation)** | Trong Flow Navigator: Chọn **Open Implemented Design** $\rightarrow$ chọn mục **Report Timing Summary**. | Minh chứng thiết kế đạt **Timing Closure 100%** tại 150 MHz:<br>• Slack âm tồi nhất dương ($WNS > 0\text{ ns}$).<br>• Tổng thời gian trễ âm bằng 0 ($TNS = 0.000\text{ ns}$). |
| **6** | **Bản Đồ Phân Bổ Chip (Device Floorplan Layout)** | Trong cửa sổ Implemented Design, chuyển sang tab **Device**. | Hiển thị mật độ phân bố phần cứng thực tế trên silicon chip Kria KV260: các khối DSP48E2 (màu xanh lục) và RAMB36E2 (màu tím) của lõi MHA 2A. |

---

## 4. Quy Trình 2: Thao Tác Chuột Thủ Công 100% (Phục Vụ Vấn Đáp)

Dành cho trường hợp hội đồng chấm thi yêu cầu thao tác trực tiếp trên giao diện GUI:

1. **Khởi tạo Project**:
   - `File` $\rightarrow$ `Project` $\rightarrow$ `New...` $\rightarrow$ Đặt tên `mha_kv260_deploy`.
   - Chọn kiểu *RTL Project* $\rightarrow$ Chọn Part: `xck26-sfvc784-2LV-c`.
2. **Nạp Mã Nguồn**:
   - `File` $\rightarrow$ `Add Sources` $\rightarrow$ `Add or create design sources` $\rightarrow$ Duyệt chọn toàn bộ file `.sv`, `.svh` trong `Project/src/Hardware/rtl_ver02_2A/`.
   - Thêm file `exp_rom_all_stages.hex` $\rightarrow$ Nhấp chuột phải chọn *Set File Type* là **Memory Initialization Files**.
3. **Dựng Sơ Đồ Khối (Create Block Design)**:
   - Click `Create Block Design` $\rightarrow$ Đặt tên `mha_system`.
   - Nhấn icon `+` (Add IP) để thêm lần lượt:
     - `Zynq UltraScale+ MPSoC`: Bật `M_AXI_HPM0_FPD`, `S_AXI_HP0_FPD`, `pl_clk0` = 150 MHz, `IRQ0`.
     - `AXI Direct Memory Access`: Bỏ chọn SG, đặt độ rộng MM2S & S2MM thành 128-bit.
     - `AXI SmartConnect`: Đặt 2 Slave Interfaces (SI), 1 Master Interface (MI).
     - `AXI Interconnect`: Đặt 1 Slave Interface (SI), 2 Master Interfaces (MI).
     - `Processor System Reset` & `Concat` (2 ports).
   - Nhấp chuột phải vào khoảng trống canvas $\rightarrow$ Chọn **Add Module...** $\rightarrow$ Chọn `attention_core_top_2A`.
4. **Kết Nối Bus & Tín Hiệu**:
   - Click `Run Connection Automation` hoặc nối dây thủ công theo đúng kiến trúc mục 1.
5. **Định Địa Chỉ & Sinh Wrapper**:
   - Chuyển sang tab `Address Editor`, gán địa chỉ `0xA000_0000` (DMA) và `0xA001_0000` (MHA Core).
   - Nhấn `F6` để Validate Design.
   - Nhấp chuột phải vào `mha_system.bd` trong tab Sources $\rightarrow$ Chọn **Create HDL Wrapper** (chọn *Let Vivado manage wrapper and auto-update*).

---

## 5. Sinh Bitstream Trực Tiếp Từ Vivado GUI

1. Trên thanh công cụ bên trái (**Flow Navigator**), cuộn xuống mục **Program and Debug**.
2. Click chọn **Generate Bitstream**.
3. Trong hộp thoại hiển thị:
   - **Number of jobs**: Thiết lập là **`4`** (để giữ hệ thống ổn định, tránh quá nhiệt CPU và treo máy).
   - Bấm **OK**.
4. Quá trình chạy mất khoảng **15 - 25 phút** qua các bước: *Synthesis* $\rightarrow$ *Implementation* $\rightarrow$ *Generate Bitstream*.
5. Khi hoàn thành, chọn *View Reports* hoặc *Open Implemented Design* để xem báo cáo tài nguyên (Utilization) và Timing.

---

## 6. Xuất File & Kiểm Thử Trên Kit Phần Cứng Kria KV260 Thật

### Bước 1: Xuất File Bitstream và Phần Cứng
Trong thanh **Tcl Console** của Vivado GUI, thực hiện lệnh sao chép:
```tcl
file copy -force ./vivado_prj_2A/mha_kv260_deploy.runs/impl_1/mha_system_wrapper.bit ../../Software/attention_core_top_2A.bit
file copy -force ./vivado_prj_2A/mha_kv260_deploy.gen/sources_1/bd/mha_system/hw_handoff/mha_system.hwh ../../Software/attention_core_top_2A.hwh
puts "-> DA XUAT FILE THANH CONG VAO Project/src/Software!"
```

### Bước 2: Kích Hoạt Kiểm Thử Từ Xa (Remote Tailscale Verification)
Mở cửa sổ dòng lệnh PowerShell trên laptop và kích hoạt script kiểm thử:
```powershell
cd C:\Users\tuan2\Desktop\Capstone_Project\Project\src\Software
python test_client_laptop.py --ip 192.168.1.179 --jump ktmt@100.124.168.26 --user ubuntu
```

### Tiêu Chí Đánh Giá Thành Công:
1. **Nạp Bitstream**: Lõi FPGA trên board Kria KV260 nhận diện driver PYNQ thành công qua file `.bit` và `.hwh`.
2. **Truyền nhận DMA**: Kênh MM2S đẩy 18,816 bytes ($Q, K, V$) và kênh S2MM nhận về 6,272 bytes (Attended Context) qua bộ đệm liên tục (CMA) không bị timeout.
3. **Độ chính xác số học (Bit-Exact)**: Kết quả tính toán của phần cứng FPGA trùng khớp hoàn toàn ($100\%$ Bit-Exact, Sai số $MAE = 0.0$) so với mô hình tham chiếu PyTorch Golden Model của mạng IViT_MobileViT.
