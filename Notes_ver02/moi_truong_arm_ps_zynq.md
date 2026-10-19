# Môi Trường ARM PS Zynq v2.0 (Ubuntu PYNQ & HW-SW Runtime)

> **Dự án:** Triển khai End-to-End HW-SW Codesign MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo mô hình MobileViT $256 \times 256$ và Bitstream `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/moi_truong_arm_ps_zynq.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/moi_truong_arm_ps_zynq.md)

---

## 1. MÔI TRƯỜNG THỰC THI KHUYẾN NGHỊ: UBUNTU 22.04 LTS + PYNQ v3.0

Đối với đồ án Capstone và nghiên cứu HW-SW Codesign, môi trường **Ubuntu Linux kết hợp framework PYNQ** là giải pháp tối ưu nhất:
- **Python 3.10+ / PyTorch CPU:** Thực thi các phần không tăng tốc (CNN Stem MobileNetV2, LayerNorm, Projections, MLP) một cách linh hoạt.
- **PYNQ DMA & Contiguous Memory Allocation (CMA):** Cấp phát các khối bộ nhớ liên tục trong RAM DDR4, cho phép DMA truyền trực tiếp dữ liệu ma trận $Q, K, V$ xuống FPGA PL với tốc độ phần cứng (Zero Copy).
- **Thư viện OpenCV & Media Pipeline:** Đọc webcam, tiền xử lý ảnh và hiển thị kết quả phân loại thời gian thực.
