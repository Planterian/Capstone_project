# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỐT NGHIỆP - TUẦN 5

* **Đề tài**: Triển khai và tăng tốc mô hình Vision Transformer (ViT) trên nền tảng nhúng AMD Kria KV260
* **Sinh viên thực hiện**: [Họ và tên sinh viên]
* **Thời gian báo cáo**: Tuần 5
* **Giảng viên hướng dẫn**: [Họ và tên GVHD]

---

### 1. Công việc đã làm trong tuần qua (Tuần 5)

* **Nghiên cứu và hiện thực các kỹ thuật lượng tử hóa (Quantization) cho ViT**:
  * Tìm hiểu cơ sở lý thuyết và cơ chế hoạt động của hai phương pháp lượng tử hóa số nguyên 8-bit (INT8): Lượng tử hóa sau huấn luyện (**PTQ - Post-Training Quantization**) và Huấn luyện nhận biết lượng tử hóa (**QAT - Quantization-Aware Training**).
  * Tái cấu trúc các khối mạng của Vision Transformer (`ScratchMultiheadAttention`, `HWFriendlyTransformerEncoder`) để tương thích với luồng phần cứng của FPGA; tích hợp các mô-đun lượng tử hóa của PyTorch (`FloatFunctional`, `QuantStub`, `DeQuantStub`) nhằm xử lý an toàn các phép toán ghép nối ma trận và cộng phần dư (Residual Add).
  * Thực hiện calibration cho PTQ và fine-tune 5 epochs cho QAT dựa trên mô hình ViT baseline (10 layers, 8 heads, embed_dim 256) đã huấn luyện trên bộ dữ liệu CIFAR-10.

* **Tổng hợp và so sánh sơ bộ về độ chính xác (Accuracy) và hiệu năng (Performance)**:
  * Trích xuất số liệu đánh giá thực nghiệm sơ bộ trên tập test CIFAR-10 giữa 3 phiên bản mô hình:

| Tiêu chí đánh giá | FP32 Baseline | PTQ INT8 | QAT INT8 | Nhận xét / Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **Độ chính xác (Test Acc)** | ~76.33% | ~71.85% | **~75.42%** | QAT bảo toàn độ chính xác tốt hơn PTQ (chỉ giảm ~0.91% so với FP32). |
| **Dung lượng file trọng số** | 21.26 MB | 5.93 MB | **5.65 MB** | Dung lượng giảm **~73.4%**, tối ưu bộ nhớ lưu trữ trên chip. |
| **Định dạng dữ liệu** | Float32 | INT8 | **INT8** | Tiết kiệm 4 lần băng thông truyền nhận dữ liệu. |
| **Bộ nhớ đệm (CMA BRAM ước tính)**| ~199.7 KB | ~49.9 KB | **~48.8 KB** | Giảm tải đáng kể dung lượng BRAM cần dùng trên FPGA. |

  * *Đánh giá sơ bộ*: QAT chứng minh được ưu thế vượt trội so với PTQ trên cấu trúc mạng Transformer, giúp mô hình thích nghi tốt với việc làm tròn số nguyên ở các hàm phi tuyến nhạy cảm như Softmax và LayerNorm.

* **Thiết lập pipeline video streaming mô phỏng cấu hình Kria KV260 (Pure Software Local)**:
  * Xây dựng luồng thu nhận video từ USB Webcam ở chuẩn độ phân giải **720p (1280x720 @ 30fps)** bằng kỹ thuật đa luồng (`ThreadedCameraStream`), giải quyết triệt để hiện tượng trễ bộ đệm (buffer lag) của OpenCV.
  * Hiện thực hóa mô hình phân chia công việc phần cứng/phần mềm (**HW/SW Partitioning**) mô phỏng kiến trúc Kria KV260:
    * *ARM Processing System (PS)*: Đảm nhận thu thập video, cắt vùng vuông trung tâm (Center ROI 720x720), co ảnh bicubic về kích thước $32 \times 32 \times 3$, chuẩn hóa dữ liệu theo CIFAR-10 Mean/Std, hậu xử lý MLP Head và xuất giao diện đồ họa.
    * *FPGA Programmable Logic (PL)*: Thiết lập lớp mô phỏng bộ nhớ liên tục CMA BRAM và tính toán độ trễ truyền dữ liệu qua kênh bus AXI4-Stream (băng thông lý thuyết 1.6 GB/s @ 200 MHz).
  * Tích hợp giao diện **Heads-Up Display (HUD)** trực quan hóa thời gian thực: hiển thị khung ngắm mục tiêu, bảng phân loại Top-3 Bar Charts và bảng phân tách độ trễ (Latency Breakdown).
  * Hiện tại toàn bộ hệ thống đang được chạy thực nghiệm thuần phần mềm (**pure software**) trên CPU máy tính cá nhân để thiết lập mốc đánh giá cơ sở (baseline) trước khi chuyển đổi sang bitstream phần cứng thật.

---

### 2. Khó khăn gặp phải

* **Nút thắt cổ chai về độ trễ khi chạy thuần phần mềm trên CPU (CPU Latency Bottleneck)**:
  * Do cấu trúc Vision Transformer có tới 10 khối Encoder với độ phức tạp tính toán cơ chế tự chú ý là $\mathcal{O}(N^2 \cdot d_k)$, việc thực thi thuần phần mềm trên CPU máy tính khiến độ trễ suy luận dao động trong khoảng 25–35 ms/khung hình. Mặc dù luồng đọc camera đạt tốc độ tối đa 74.7 FPS, nhưng tốc độ xử lý toàn chu trình (End-to-End) hiện chỉ đạt xấp xỉ 23–28 FPS, chưa tận dụng hết thông lượng 30 FPS danh định của camera nếu không có phần cứng chuyên dụng tăng tốc.
* **Sự nhạy cảm của các toán tử Attention đối với việc lượng tử hóa**:
  * Khác với các mạng tích chập (CNN), mạng Transformer có độ nhạy cảm rất cao tại phép nhân ma trận $Q \times K^T$ và hàm tính xác suất Softmax. Trong quá trình thử nghiệm PTQ ban đầu, độ chính xác bị suy giảm đáng kể do hiện tượng tràn số và mất mát thông tin phân phối xác suất chú ý, buộc phải chuyển sang triển khai QAT với đồ thị tính toán tùy biến (`ScratchMultiheadAttention`) để giữ kết quả ổn định.
* **Vấn đề tương thích backend lượng tử hóa PyTorch**:
  * Quá trình nạp trọng số lượng tử hóa INT8 phụ thuộc chặt chẽ vào backend của hệ điều hành máy host (`onednn`, `fbgemm` hoặc `qnnpack`), dẫn đến một số cảnh báo deprecation và sai lệch nhỏ trong việc ánh xạ định dạng đóng gói `_packed_params` giữa môi trường huấn luyện Colab và môi trường chạy local.

---

### 3. Công việc tuần tới

Tiếp tục tối ưu hóa mô hình, nghiên cứu thêm các tài liệu liên quan đến phần cứng và chuẩn bị cho các bước kiểm thử tiếp theo trong đồ án.
