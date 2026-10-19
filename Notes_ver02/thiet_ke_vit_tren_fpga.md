# Thiết Kế ViT Trên FPGA AMD Kria KV260 v2.0

> **Dự án:** Thiết kế bộ tăng tốc Attention Core cho MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/thiet_ke_vit_tren_fpga.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/thiet_ke_vit_tren_fpga.md)

---

## 1. CÁC NGUYÊN TẮC THIẾT KẾ CỐT LÕI CHO VI MẠCH VI-T

Khi đưa mô hình Vision Transformer lên chip FPGA biên Kria KV260, 3 thách thức lớn nhất là: **Băng thông bộ nhớ**, **Tài nguyên DSP hạn chế** và **Các hàm kích hoạt phi tuyến tính**. Dự án giải quyết triệt để thông qua 3 nguyên tắc:

### 1.1 Tái sử dụng Phần cứng bằng Sequential Heads Looping
Thay vì khởi tạo 4 lõi Attention song song (tiêu tốn hơn $80\%$ DSP và BRAM của chip KV260), thiết kế sử dụng **1 lõi Attention duy nhất** và dùng máy trạng thái FSM quét tuần tự lần lượt từng Head ($h = 0 \dots H-1$):
- Tiết kiệm **>70% diện tích logic và BRAM**.
- Đáp ứng khả năng mở rộng linh hoạt cho các mô hình có số lượng Head tùy biến ($H = 4, 8$).

### 1.2 Triệt tiêu phép chia dấu phẩy động bằng Dyadic Scaling
Toàn bộ các phép chia cho $\sqrt{d_k}$ và nhân hệ số lượng tử hóa được gom thành phép nhân dịch nguyên:
$$\text{scaled\_score} = (\text{raw\_score} \times M) \ggg E$$
Tất cả các tham số $M, E$ được nạp qua thanh ghi CSR AXI-Lite, cho phép chip xử lý các kích thước $d_k = 16, 20, 24$ mà không cần thay đổi cấu trúc phần cứng.

### 1.3 Phần cứng hóa hàm Softmax không dùng FPU
Thay vì dùng Floating-Point Unit (FPU) cồng kềnh, khối Softmax dùng bảng tra BRAM 1024 từ $\times$ 20 bits kết hợp với bộ chia nguyên 32 chu kỳ, đảm bảo tần số hoạt động đạt $200\text{ MHz}$ và đạt bit-exact với mô hình PyTorch I-ViT.
