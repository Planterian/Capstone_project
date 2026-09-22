# Hướng dẫn Kria KV260

Dưới đây là hướng dẫn chi tiết từ A–Z về **cách chuẩn bị, kết nối dây cáp vật lý và quy trình lập trình** giữa Laptop và bo mạch **AMD Kria KV260** cho đồ án.

---

##### 1. Danh mục thiết bị cần có (Hardware Checklist)
1. **Laptop / PC phát triển**: Dùng để viết mã SystemVerilog/Python, chạy phần mềm **Vivado** (biên dịch ra file phần cứng .bit) và điều khiển bo mạch từ xa.
2. **Bo mạch AMD Kria KV260 Vision AI Starter Kit**: Chứa chip Zynq UltraScale+ MPSoC (gồm lõi **ARM PS** để chạy Ubuntu/PYNQ và **FPGA PL** để tăng tốc AI).
3. **Thẻ nhớ MicroSD (Tối thiểu 16GB / 32GB Class 10)**: Chứa hệ điều hành **Ubuntu Desktop / PYNQ Linux Image** cho Kria KV260.
4. **Bộ nguồn DC (12V – 3A)**: Nguồn cấp điện cho KV260.
5. **Cáp mạng Ethernet (RJ45)**: Kết nối KV260 vào cùng mạng Wi-Fi/Router với Laptop (hoặc cắm trực tiếp vào cổng LAN của Laptop).
6. **Cáp Micro-USB (hoặc USB-C tùy phiên bản)**: Cắm từ KV260 sang Laptop để đọc cổng Serial Console (UART/COM port) khi cần debug hạ tầng.
7. **USB Webcam (Chuẩn UVC)**: Cắm trực tiếp vào cổng USB trên KV260 để quay hình ảnh real-time.
8. **Màn hình HDMI / DisplayPort & Cáp** *(Tùy chọn)*: Để hiển thị trực tiếp giao diện nhận diện hình ảnh từ KV260.

---

##### 2. Sơ đồ kết nối vật lý (Physical Connection Diagram)

```text
===================================================================================================
                               SƠ ĐỒ KẾT NỐI HỆ THỐNG (DEVELOPMENT MODE)
===================================================================================================

       +------------------------------------+
       |   LAPTOP / PC PHÁT TRIỂN           |
       |   - Chạy Vivado (Tạo file .bit)    |
       |   - VS Code / SSH Terminal         |
       +-----------------+------------------+
                         |
      +------------------+------------------+
      | (Cáp Mạng LAN)                      | (Cáp Micro-USB UART Debug)
      v                                     v
  +---+-------------------------------------+--------------------------------------------------+
  |   Cổng RJ45                             Cổng Micro-USB (COM Port)                          |
  |                                                                                            |
  |                        BO MẠCH AMD KRIA KV260 STARTER KIT                                  |
  |                                                                                            |
  |   [ Chip Zynq UltraScale+ MPSoC ]                                                          |
  |     ├── ARM PS (Cortex-A53): Chạy Ubuntu/PYNQ Linux, OpenCV, AXI DMA Driver                |
  |     └── FPGA PL: Nạp Bitstream Attention Core (SystemVerilog RTL)                          |
  |                                                                                            |
  |   Cổng USB 3.0                          Cổng HDMI / DP             Cổng Nguồn DC 12V        |
  +---+-------------------------------------+--------------------------+-----------------------+
      |                                     |                          |
      v                                     v                          v
  [ USB Webcam ]                   [ Màn hình HDMI ]           [ Nguồn Điện 12V-3A ]
  (Quay ảnh 224x224)               (Hiển thị FPS / Bounding)
```

---

##### 3. Quy trình 4 bước thiết lập & vận hành chi tiết

###### 📌 BƯỚC 1: Flash Hệ Điều Hành vào Thẻ Nhớ MicroSD
1. Tải file **PYNQ / Ubuntu Image for Kria KV260** (.img) từ trang chủ PYNQ.io.
2. Dùng phần mềm **BalenaEtcher** hoặc **Rufus** trên Laptop để nạp file .img này vào thẻ nhớ MicroSD.
3. Cắm thẻ nhớ MicroSD vào khe cắm thẻ ở mặt dưới bo mạch KV260.

###### 📌 BƯỚC 2: Kết Nối Cáp Vật Lý
1. **Cắm USB Webcam** vào 1 trong các cổng USB 3.0 trên KV260.
2. **Cắm cáp Mạng LAN (RJ45)** nối từ KV260 vào Router Wi-Fi nhà bạn (sao cho Laptop và KV260 ở chung một mạng nội bộ).
3. **Cắm cáp Micro-USB** từ cổng Debug của KV260 sang Laptop.
4. **Cắm Nguồn 12V** vào KV260 để bật máy (Đèn LED trên bo mạch sẽ sáng).

###### 📌 BƯỚC 3: Truy Cập & Điều Khiển KV260 Từ Laptop
Sau khi KV260 khởi động xong (khoảng 20–30 giây), bạn ngồi trên Laptop và điều khiển KV260 bằng 1 trong 3 cách:
* **Cách 1 (Ưu tiên - VS Code Remote SSH)**: Mở VS Code trên Laptop $\rightarrow$ Cài extension Remote - SSH $\rightarrow$ Kết nối tới `xilinx@<Địa_chỉ_IP_KV260>`. Bạn có thể sửa code Python, chạy script ngay trên VS Code của Laptop như làm việc cục bộ.
* **Cách 2 (Trình duyệt Web - Jupyter Notebook)**: Mở Chrome trên Laptop $\rightarrow$ Truy cập `http://<Địa_chỉ_IP_KV260>:9090` (Mật khẩu mặc định: `xilinx`).
* **Cách 3 (Serial Console Debug)**: Dùng phần mềm **PuTTY** trên Laptop mở cổng COM (Baudrate 115200) để xem nhật ký khởi động Linux thô.

###### 📌 BƯỚC 4: Luồng Lập Trình & Thực Thi Đồ Án Hàng Ngày

```text
[Lập trình RTL trên Laptop] ──(Vivado Build)──> Tạo file phần cứng: attention_core.bit & .hwh
                                                                  │
                                                        (Chuyển file qua SSH/SFTP)
                                                                  ▼
[Lập trình Python trên KV260] <──(SSH / VS Code)── [Đẩy file .bit vào KV260]
  - pynq.Overlay("attention_core.bit") ──> Nạp vi mạch lên FPGA PL
  - OpenCV đọc Webcam ──> PYNQ AXI DMA ──> FPGA PL tăng tốc ──> Hiển thị kết quả UI
```
