# Vận Hành Độc Lập KV260 v2.0 (Standalone Edge AI Execution)

> **Dự án:** Triển khai End-to-End HW-SW Codesign MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo mô hình MobileViT $256 \times 256$ và Bitstream `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/van_hanh_doc_lap_kv260.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/van_hanh_doc_lap_kv260.md)

---

## 1. PHÂN BIỆT 2 GIAI ĐOẠN: PHÁT TRIỂN VS. VẬN HÀNH ĐỘC LẬP

### 1.1 Giai đoạn Phát triển (Development Phase):
Cần Laptop/PC phát triển để:
- Tổng hợp vi mạch (Vivado Synthesis & Implementation 200 MHz trên part `xck26-sfvc784-2LV-c`).
- Kiểm thử số học Bit-Exact và mô phỏng RTL (Icarus Verilog / Vivado Simulator).
- Tinh chỉnh mô hình lượng tử hóa (I-ViT Surgery, PTQ, QAT).

### 1.2 Giai đoạn Vận hành Độc lập (Standalone Demonstration Phase):
Bo mạch **AMD Kria KV260 có thể vận hành hoàn toàn độc lập mà không cần kết nối với Laptop**:
1. Cắm USB Webcam vào cổng USB 3.0 trên KV260.
2. Cắm màn hình hiển thị trực tiếp vào cổng HDMI/DisplayPort.
3. Cấp nguồn 12V-3A: Hệ thống Ubuntu/PYNQ tự khởi động qua systemd service:
   - Tự động nạp file `attention_core_kv260.bit` lên FPGA PL.
   - Chạy pipeline nhận diện MobileViT-XXS thời gian thực (chụp ảnh $256 \times 256$, chạy CNN Stem trên ARM PS, offload Attention Core sang FPGA PL, và xuất nhãn phân loại lên màn hình).
