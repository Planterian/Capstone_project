# Kỹ thuật DSP Packing

**DSP Packing** là một kỹ thuật tối ưu hóa phần cứng mức vi mạch trên FPGA, cho phép **đóng gói nhiều phép nhân độ chính xác thấp** (low-precision INT8, INT6 hoặc INT4) vào trong **một khối DSP slice đơn lẻ** để thực thi song song trong cùng một chu kỳ xung clock.
Kỹ thuật này giúp nhân đôi hoặc nhân ba hiệu năng tính toán (**Throughput per DSP / GOPS**) mà không làm tăng số lượng phần cứng DSP vật lý trên chip.

---

##### 1. Nguyên lý Toán học & Phần cứng (Ví dụ trên Xilinx/AMD DSP48E2)
Các dòng FPGA UltraScale+ (như trên bo mạch **Kria KV260** hay **ZCU102**) tích hợp khối **DSP48E2 Slice** gồm một bộ nhân có kích thước cổng vào **18-bit × 27-bit** và bộ tích lũy 48-bit.
Để thực hiện song song 2 phép nhân INT8 cùng lúc trên **1 DSP** (Factor-2 Packing), thuật toán biến đổi như sau:
* **Cấu trúc toán hạng**:
    * Giả sử cần tính đồng thời hai phép nhân có chung một kích hoạt $A$ (18-bit / INT8): $Y_1 = A \times B$ và $Y_2 = A \times C$ (với $B, C$ là hai trọng số INT8 độc lập).
    * Toán hạng $A$ được đưa vào cổng **18-bit**.
    * Hai trọng số $B$ và $C$ được dịch bit và ghép lại vào cổng **27-bit**: $$D_{in} = (B \ll 18) + C$$.
* **Kết quả phép nhân trong thanh ghi 45-bit**: $$A \times D_{in} = A \times ((B \ll 18) + C) = (A \times B) \ll 18 + (A \times C)$$. Vì kết quả của mỗi phép nhân INT8 $\times$ INT8 chỉ chiếm tối đa 16 bit, nên tích số $A \times B$ (Upper sub-word) và $A \times C$ (Lower sub-word) nằm ở hai vùng bit hoàn toàn riêng biệt trong thanh ghi đầu ra 45-bit của DSP. Circuit phần cứng chỉ cần tách (unpack) hai vùng bit này ra là thu được hai kết quả riêng biệt trong 1 clock cycle.

---

##### 2. Các Mẫu Đóng gói Mở rộng (Factor-3 & Factor-4 Layouts)
Đối với các mô hình định lượng siêu thấp (< 8 bit) như trong kiến trúc Quasar-ViT, kỹ thuật DSP Packing được mở rộng theo các cấu hình:
* **Factor-3 Layout (1 Activation 6-bit + 3 Weights 4-bit)**: Đặt một vector activation 6-bit vào cổng 18-bit và đóng gói ba trọng số 4-bit vào cổng 27-bit (kết hợp các bộ gom logic LUT nhỏ) để thực hiện **3 phép nhân trong 1 DSP/cycle**.
* **Factor-4 Layout (2 Activations 6-bit + 2 Weights 4-bit)**: Ghép hai activation 6-bit với hai weight 4-bit để thực hiện **4 phép nhân đồng thời**.
* **Chế độ SIMD tích hợp sẵn (Hardened SIMD)**: Bản thân các khối DSP48E2 cũng hỗ trợ sẵn chế độ SIMD phần cứng như Dual 24-bit hoặc Quad 12-bit/Quad INT8.

---

##### 3. Thách thức Kỹ thuật & Đánh đổi (Hardware Trade-offs)
Mặc dù tăng gấp đôi mật độ tính toán, DSP Packing đòi hỏi sự đánh đổi về thiết kế RTL:
1. **Xử lý Bit Dấu (Sign Extension)**: Phải quản lý chính xác biểu diễn số nguyên có dấu (2's complement) và mở rộng bit dấu cho các vùng bit phía trên (Upper sub-words) để tránh tràn bit hoặc làm nhiễu kết quả giữa các phép tính ghép.
2. **Tiêu tốn Tài nguyên LUT đệm (Packing/Unpacking Overhead)**: Cần thêm các bộ logic ghép bit ở đầu vào và tách bit ở đầu ra, tiêu tốn một lượng LUT bổ sung.
3. **Nghẽn Định tuyến & Timing Closure (Routing Congestion)**: Mật độ bus dữ liệu tăng gấp đôi gây ra hiện tượng nghẽn đường truyền (routing density). Để giữ chu kỳ clock Fmax cao, thiết kế bắt buộc phải kết hợp bộ đệm BRAM/URAM được căn chỉnh chính xác (buffer-alignment) để cấp dữ liệu packed trong cùng một chu kỳ.

---

##### 4. Lợi ích cho Tăng tốc Vision Transformer (ViT) trên Edge FPGA
* **Tối ưu hóa tài nguyên DSP hạn chế**: Đối với các dòng FPGA phân khúc viền (Edge FPGAs) như **Kria KV260** hay **ZCU102** vốn chỉ có từ vài trăm đến hơn một ngàn DSP slices, DSP Packing cho phép đạt tốc độ từ **100 - 250+ FPS**.
* **Tăng hiệu suất Throughput per DSP**: Giúp các kiến trúc như ME-ViT tăng hiệu suất xử lý trên mỗi DSP lên **2.16×** so với thiết kế tiêu chuẩn.
