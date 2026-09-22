# Môi trường ARM PS Zynq

**Không nhất thiết.** ARM PS (Processing System) trên Zynq UltraScale+ MPSoC / AMD Kria KV260 có thể vận hành linh hoạt dưới nhiều môi trường phần mềm khác nhau tùy thuộc vào mục tiêu dự án (Nghiên cứu / Prototyping hay Sản phẩm thương mại nhúng sâu).

Các tùy chọn môi trường thực thi trên ARM PS và phạm vi ứng dụng phù hợp bao gồm:

---

##### 1. Các Tùy Chọn Môi Trường Thực Thi Cho ARM PS

###### 1️⃣ Ubuntu OS (kèm PYNQ Framework)
* **Bản chất**: Hệ điều hành Linux đầy đủ (Full Desktop/Server OS) tích hợp Python PYNQ library (`pynq.Overlay`, `pynq.allocate`).
* **Ưu điểm**: **Phát triển cực kỳ nhanh**. Hỗ trợ sẵn Python 3, OpenCV, Jupyter Notebook, SSH và trình quản lý gói pip/apt. Giúp cấp phát bộ nhớ liên tục (CMA) và gọi AXI DMA chỉ với vài dòng lệnh Python.
* **Nhược điểm**: Dung lượng lớn (~2–4 GB), thời gian khởi động chậm (20–30 giây), tốn nhiều tài nguyên RAM/CPU của ARM PS cho các service nền.
* **Phù hợp nhất cho**: **Đồ án tốt nghiệp (Capstone Project), Nghiên cứu R&D, Đánh giá PoC**.

###### 2️⃣ PetaLinux (Custom Embedded Linux)
* **Bản chất**: Linux nhúng siêu nhẹ được tùy biến riêng bằng công cụ Yocto/PetaLinux của AMD/Xilinx.
* **Ưu điểm**: Khởi động nhanh (3–5 giây), dung lượng nhỏ gọn (< 100MB), tối ưu tài nguyên RAM/CPU.
* **Phù hợp nhất cho**: Sản phẩm nhúng thương mại.

###### 3️⃣ Bare-Metal C / FreeRTOS
* **Bản chất**: Chương trình C/C++ biên dịch bằng SDK/Vitis IDE chạy trực tiếp trên lõi ARM mà không qua bất kỳ OS nào.
* **Ưu điểm**: **Zero Overhead**, thời gian khởi động tức thì, kiểm soát 100% thanh ghi phần cứng và không gian bộ nhớ.
* **Nhược điểm**: Phải tự viết toàn bộ driver điều khiển AXI DMA, GIC Interrupt Controller, Ethernet/UART ở mức thanh ghi thô.
* **Phù hợp nhất cho**: Kiểm thử vi mạch đơn giản hoặc các ứng dụng siêu mỏng đòi hỏi tốc độ phản hồi tuyệt đối.

---

##### 📊 Bảng So Sánh Lựa Chọn

| Môi trường | Thời gian Boot | Dung lượng / Tài nguyên | Tốc độ Lập trình | Độ trễ Real-Time |
| ------ | ------ | ------ | ------ | ------ |
| **Ubuntu + PYNQ** | Chậm (~25s) | Lớn (Cần SD Card 16GB+) | **Rất nhanh (Python API)** | Thấp (Soft Real-Time) |
| **PetaLinux** | Nhanh (~3s) | Nhẹ (Vài chục MB Flash) | Trung bình (C++/Python) | Trung bình |
| **FreeRTOS** | Tức thì (<1s) | Siêu nhẹ (<1MB RAM) | Chậm (C/C++ Drivers) | **Rất cao (Hard Real-time)** |
| **Bare-Metal** | Tức thì (0s) | Siêu nhẹ | Chậm nhất (C thô) | **Tuyệt đối** |

---

##### 💡 Khuyên Dùng Cho Dự Án
* **Nếu đang làm Đồ án Capstone / PoC**: **Nên dùng Ubuntu OS + PYNQ**. Lựa chọn này giúp bạn tập trung hoàn toàn vào việc thiết kế vi mạch SystemVerilog RTL, truyền nhận AXI DMA và đánh giá mô hình ViT mà không bị mất hàng tuần cấu hình build-system PetaLinux hay viết driver Bare-metal C.
* **Nếu chuyển sang sản phẩm thực tế**: Bạn có thể giữ nguyên Bitstream `.bit` trên FPGA PL và biên dịch lại phần phần mềm ARM PS sang **PetaLinux** hoặc **Bare-metal C** để tối ưu tốc độ khởi động và năng lượng.
