# Kế Hoạch Triển Khai Ứng Dụng Video Image Classification ViT (vit_qat_int8) Mô Phỏng Bo Mạch AMD Kria KV260

---

## 1. Tổng Quan Dự Án & Bối Cảnh Hệ Thống

* **Mục tiêu**: Xây dựng ứng dụng phân loại ảnh thời gian thực (Real-time Video Image Classification) sử dụng luồng camera USB 720p @ 30 fps dựa trên mô hình **Vision Transformer đã lượng tử hóa nhận biết (QAT INT8 - `vit_qat_int8.pth`)**.
* **Môi trường giả lập**: Replicate & Simulate môi trường bo mạch nhúng **AMD Kria KV260 Vision AI Starter Kit** bằng phần mềm (Software-Only Simulation), phản ánh đúng cơ chế phân chia phần cứng/phần mềm (**HW/SW Co-design & Partitioning**):
  - **ARM Processing System (PS)**: Đảm nhận thu thập video OpenCV, tiền xử lý khung hình (Center crop, Resize 32x32, Chuẩn hóa CIFAR-10), điều khiển truyền nhận AXI DMA, hậu xử lý MLP Head và hiển thị giao diện HUD thời gian thực.
  - **FPGA Programmable Logic (PL)**: Mô phỏng khối tăng tốc phần cứng **Attention Core** (Systolic MAC Array nhân $Q \times K^T$, bộ dịch bit đại số ASR Scaler, bảng tra phần cứng Hardware Softmax LUT, và nhân $Score \times V$ bằng INT8) kết nối qua bộ nhớ liên tục CMA và thanh ghi AXI-Lite.
* **Cấu hình mô hình đích (`vit_qat_int8`)**:
  - Dataset: CIFAR-10 (10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
  - Độ phân giải đầu vào: $32 \times 32 \times 3$, kích thước Patch: $4 \times 4$ ($N = 64$ patch tokens + 1 CLS token = 65 tokens).
  - Kiến trúc: `embed_dim = 256`, `num_heads = 8` ($d_k = 32$), `depth = 10`, `mlp_dim = 512`.
  - Trọng số định lượng: `./models_cache/vit_qat_int8.pth` (PyTorch INT8 Quantized Model).

---

## 2. Phân Chia Khối Lượng Công Việc (Batches of Workload)

Hệ thống được chia thành **4 Batches** logic và độc lập để đảm bảo chất lượng kiểm thử, tính mô-đun và dễ dàng rà soát/phê duyệt:

```text
+---------------------------------------------------------------------------------------------------+
| BATCH 1: KV260 HARDWARE SIMULATION ENGINE & QUANTIZED MODEL INFERENCE LOADER                      |
| - Tái cấu trúc mô hình QuantizableVisionTransformer & HWFriendlyTransformerEncoder                |
| - Nạp trọng số vit_qat_int8.pth vào engine lượng tử hóa INT8 (onednn/fbgemm)                     |
| - Xây dựng lớp KriaKV260HardwareSimulator (Mô phỏng AXI DMA, CMA BRAM, Chu kỳ tính toán PL)       |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| BATCH 2: REAL-TIME 720P@30FPS VIDEO STREAM PIPELINE & PS PREPROCESSING                            |
| - Khởi tạo luồng USB Webcam 720p (1280x720 @ 30 fps) đa luồng (Threaded VideoCapture)            |
| - Chế độ kép (Dual Mode): Hỗ trợ Webcam vật lý + Fallback Video tổng hợp/Slide ảnh CIFAR-10       |
| - Pipeline tiền xử lý ARM PS: Center ROI Crop, Resize 32x32, RGB, CIFAR-10 Normalization         |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| BATCH 3: EDGE AI HUD OVERLAY, HARDWARE LATENCY PROFILER & TELEMETRY MONITOR                       |
| - Thiết kế giao diện Dashboard chuẩn Edge AI Kria KV260 (HUD - Heads-Up Display)                  |
| - Hiển thị Bounding Target Reticle, Top-1/Top-3 Confidence Bar Chart                              |
| - Bộ đo phân tách độ trễ (Latency Breakdown): T_pre (PS), T_dma, T_pl (Attention), T_post (PS)   |
| - Bộ điều khiển tương tác bàn phím ('q': Thoát, 's': Chụp ảnh, 'd': Debug Mode, 'l': Lock 30fps) |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| BATCH 4: JUPYTER NOTEBOOK INTEGRATION, COMPREHENSIVE BENCHMARK & FINAL REPORT                     |
| - Xuất bản mã nguồn chuẩn sang Video_image_classification_VIT.py (với cú pháp # %% [markdown])     |
| - Đồng bộ hóa và sinh file hoàn chỉnh Video_image_classification_VIT.ipynb                        |
| - Viết báo cáo thực nghiệm so sánh hiệu năng vi xử lý x86/ARM vs FPGA PL ước tính                 |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Chi Tiết Các Phần Tử Từng Batch

### 📌 Batch 1: KV260 Hardware Simulation Engine & Model Loader
1. **Thành phần**:
   - `PatchEmbedding`, `ScratchMultiheadAttention`, `HWFriendlyTransformerEncoder`, `QuantizableVisionTransformer`.
   - Engine lượng tử hóa tương thích PyTorch (`onednn` / `fbgemm`) nạp file `./models_cache/vit_qat_int8.pth`.
   - Lớp `KriaKV260Simulator`:
     - Mô phỏng cấp phát bộ nhớ vật lý liên tục CMA (`allocate` shape `(65, 256)` kiểu `np.int8`).
     - Đo lường và mô phỏng độ trễ truyền nhận AXI DMA kênh Tx/Rx qua bus AXI4-Stream (băng thông 64-bit @ 200 MHz trên KV260 PL).
     - Đo lường và tính toán chu kỳ phần cứng ước tính của mảng Systolic MAC Array ($O(N^2 \cdot d_k)$).

### 📌 Batch 2: Real-time 720p Video Pipeline & Preprocessing
1. **Thành phần**:
   - Lớp `ThreadedCameraStream`:
     - Đọc camera USB với độ phân giải `1280x720`, tốc độ yêu cầu `30 FPS`.
     - Sử dụng buffer ring/queue độc lập để triệt tiêu hiện tượng nghẽn luồng đọc camera OpenCV trên Windows/Linux.
     - Tự động kiểm tra thiết bị: nếu không tìm thấy Webcam vật lý (hoặc chạy trong môi trường headless/server), tự động kích hoạt chế độ **Synthetic Video / CIFAR-10 Loop Test Stream** đảm bảo không bao giờ bị dừng ứng dụng đột ngột.
   - Lớp `PSPreprocessor`:
     - Trích xuất vùng quan tâm trung tâm (Center ROI Square $720 \times 720$).
     - Resize bicubic xuống $32 \times 32 \times 3$.
     - Chuẩn hóa tensor với CIFAR Mean `(0.4914, 0.4822, 0.4465)` và Std `(0.2470, 0.2435, 0.2616)`.

### 📌 Batch 3: Edge AI HUD Overlay & Telemetry Dashboard
1. **Thành phần**:
   - Lớp `KV260HUDVisualizer`:
     - Vẽ khung nhắm trung tâm (Target Reticle & ROI Box) trên khung hình gốc 720p.
     - Hiển thị bảng phân loại: Nhãn hàng đầu (Top-1 Class) với độ tin cậy phần trăm và thanh đo Top-3 (Bar Charts).
     - Bảng thông số phần cứng (Hardware Telemetry Panel):
       - Chỉ số FPS thực tế và FPS mục tiêu của KV260.
       - Phân rã độ trễ (Latency Breakdown):
         * $T_{\text{read+prep}}$: Thời gian đọc và tiền xử lý khung hình trên ARM PS (ms).
         * $T_{\text{axi\_dma}}$: Thời gian mô phỏng truyền dữ liệu qua AXI DMA (ms).
         * $T_{\text{pl\_attn}}$: Thời gian tính toán Attention Core trên FPGA PL (ms).
         * $T_{\text{post}}$: Thời gian tính toán MLP Head & Softmax trên ARM PS (ms).
         * $T_{\text{total}}$: Độ trễ toàn chu trình từ Camera đến Màn hình (End-to-End Latency).
     - Banner định danh: `[AMD Kria KV260 Simulation | vit_qat_int8 | PL Accelerator: ACTIVE]`.

### 📌 Batch 4: Notebook & Final Reporting
1. **Thành phần**:
   - File Python chính `model_notebook/Video_image_classification_VIT.py` được chú thích cực kỳ tỉ mỉ theo cấu trúc cell `# %% [code]` và `# %% [markdown]`.
   - Sinh file Jupyter Notebook `model_notebook/Video_image_classification_VIT.ipynb` chuẩn cấu trúc JSON.
   - Báo cáo phân tích chuyên sâu tại `model_notebook/markdowns/bao_cao_danh_gia_hieu_nang_vit_kv260.md`.

---

## 4. Các Mega-Prompts Cho Từng Batch (Dùng Cho Các Lần Trao Đổi Tiếp Theo)

Dưới đây là các **Mega-Prompts** hoàn chỉnh, chuẩn kỹ sư Senior AI/Embedded Systems, sẵn sàng để bạn duyệt và kích hoạt từng bước:

---

### 🚀 MEGA-PROMPT CHO BATCH 1:
```text
Thực hiện BATCH 1: Xây dựng KV260 Hardware Simulation Engine & Model Loader cho mô hình vit_qat_int8.
Mục tiêu cụ thể:
1. Tạo file mã nguồn 'model_notebook/kv260_hardware_engine.py' chứa định nghĩa hoàn chỉnh của mô hình Vision Transformer lượng tử hóa (QuantizableVisionTransformer, HWFriendlyTransformerEncoder, ScratchMultiheadAttention, PatchEmbedding, MLP).
2. Viết hàm nạp an toàn và chuẩn xác trọng số INT8 từ 'model_notebook/models_cache/vit_qat_int8.pth' vào engine lượng tử hóa của PyTorch (hỗ trợ onednn / fbgemm / qnnpack).
3. Xây dựng lớp 'KriaKV260Simulator' mô phỏng kiến trúc vi mạch FPGA Zynq UltraScale+ MPSoC:
   - Mô phỏng cấp phát bộ nhớ CMA liên tục (contiguous memory allocation) tương đương pynq.allocate.
   - Mô phỏng thanh ghi điều khiển AXI-Lite (Control registers: START, N=65, d_k=32, H=8).
   - Đo lường và mô phỏng độ trễ DMA transfer (AXI4-Stream 64-bit @ 200MHz) và chu kỳ tính toán của Systolic MAC Array trên FPGA PL.
4. Viết kịch bản tự kiểm thử (Self-test verification) với vector ngẫu nhiên để xác nhận tính toán thành công và in ra output shape [1, 10] cùng bảng phân tích chu kỳ phần cứng ước tính.
5. Quy tắc code: Chú thích chi tiết từng dòng (WHAT/WHY/FUNCTIONALITY) bằng tiếng Việt hoặc tiếng Anh kỹ thuật, định dạng cell comment (# %%) để dễ dàng chuyển sang Jupyter Notebook.
```

---

### 🚀 MEGA-PROMPT CHO BATCH 2:
```text
Thực hiện BATCH 2: Xây dựng Pipeline Video Stream 720p @ 30 FPS và Tiền xử lý ARM PS.
Mục tiêu cụ thể:
1. Tạo module 'model_notebook/video_pipeline.py' quản lý luồng video camera:
   - Lớp 'ThreadedCameraStream' đọc video từ USB Webcam ở chuẩn độ phân giải 1280x720 và tốc độ 30 FPS trên một luồng nền riêng biệt (background thread) để không bị drop frame hoặc nghẽn I/O.
   - Tích hợp chế độ dự phòng thông minh (Synthetic Fallback Mode): Nếu không có webcam cắm ngoài hoặc index 0 không khả dụng, tự động chuyển sang đọc video demo hoặc sinh luồng ảnh kiểm thử CIFAR-10 liên tục kèm hiệu ứng chuyển động.
2. Lớp 'PSPreprocessor':
   - Cắt khung hình vùng trung tâm (Center-Crop 720x720) để bảo toàn tỷ lệ khung hình chuẩn vuông.
   - Resize xuống kích thước 32x32 bằng nội suy chuẩn cv2.INTER_AREA / cv2.INTER_CUBIC.
   - Chuyển đổi không gian màu BGR sang RGB và chuẩn hóa theo CIFAR-10 Mean/Std.
   - Trả về tensor PyTorch và mảng mượn bộ nhớ mô phỏng CMA INT8.
3. Tích hợp chạy thử luồng đọc video và hiển thị FPS khung hình thô trước khi nạp vào AI.
4. Quy tắc code: Đầy đủ chú thích giải thích WHAT/WHY/FUNCTIONALITY, định dạng cell chuẩn (# %%).
```

---

### 🚀 MEGA-PROMPT CHO BATCH 3:
```text
Thực hiện BATCH 3: Thiết kế Giao diện Kria KV260 Edge AI HUD Overlay & Telemetry Monitor.
Mục tiêu cụ thể:
1. Xây dựng module 'model_notebook/hud_overlay.py' đảm nhận việc vẽ giao diện điều khiển thời gian thực chuẩn công nghiệp Edge AI trên khung hình 720p:
   - Vẽ khung nhắm Reticle ngắm mục tiêu ở vùng trung tâm (Center ROI).
   - Vẽ bảng Dashboard kết quả phân loại: Top-1 Class với confidence %, Top-3 Bar Charts trực quan.
   - Bảng thông số đo đạc phần cứng (Hardware Telemetry Monitor):
     * FPS thời gian thực (hiển thị màu xanh lá > 25 FPS, vàng 15-25 FPS, đỏ < 15 FPS).
     * Bảng phân tích trễ (Latency Breakdown Table): T_read_prep (PS), T_axi_dma (Bus), T_pl_attn (FPGA PL), T_post (PS), và T_total (End-to-End).
     * Trạng thái phần cứng KV260: [KV260 PL INT8 Engine: ACTIVE | CMA Buffer: LOCKED].
2. Tích hợp các phím tắt điều khiển bàn phím linh hoạt:
   - 'q': Thoát ứng dụng an toàn và giải phóng tài nguyên.
   - 's': Chụp ảnh màn hình lưu vào thư mục 'captures/'.
   - 'd': Bật/Tắt chế độ Debug hiển thị chi tiết ma trận Attention.
   - 'f': Bật/Tắt chế độ giới hạn FPS (FPS Lock 30 FPS vs Unlocked).
3. Đảm bảo giao diện hiện đại, đường nét sắc sảo, phông chữ và bảng màu trực quan (glassmorphism/dark HUD style).
```

---

### 🚀 MEGA-PROMPT CHO BATCH 4:
```text
Thực hiện BATCH 4: Đóng gói toàn diện ứng dụng vào file Video_image_classification_VIT.py, sinh Jupyter Notebook hoàn chỉnh Video_image_classification_VIT.ipynb và Lập Báo cáo Đánh giá Hiệu năng.
Mục tiêu cụ thể:
1. Tích hợp tất cả các thành phần từ Batch 1, 2, 3 vào một file hoàn chỉnh duy nhất: 'model_notebook/Video_image_classification_VIT.py' với các cell tách biệt rõ ràng (# %% [markdown] và # %% [code]).
2. Sử dụng kịch bản tự động chuyển đổi file .py sang file Jupyter Notebook 'model_notebook/Video_image_classification_VIT.ipynb' chuẩn JSON đầy đủ metadata và markdown diễn giải.
3. Chạy thử nghiệm đánh giá (Benchmarking Run) để đo lường thực tế:
   - Tốc độ suy luận FP32 vs QAT INT8.
   - Tỷ lệ giảm kích thước mô hình (5.9 MB vs 21.2 MB).
   - Phân tích hiệu năng giả lập Kria KV260 (chu kỳ tính toán phần cứng PL vs xử lý thuần trên CPU).
4. Tạo file báo cáo hoàn chỉnh tại 'model_notebook/markdowns/bao_cao_danh_gia_hieu_nang_vit_kv260.md' ghi chép chi tiết kết quả thực nghiệm, hướng dẫn vận hành và nhật ký kiểm thử.
```

---

## 5. Quy Chuẩn Đóng Gói Và Theo Dõi Tiến Độ

| Hạng mục | Tệp tin đích | Trạng thái |
| :--- | :--- | :--- |
| **Kế hoạch & Mega-Prompts** | `model_notebook/markdowns/ke_hoach_trien_khai_video_vit.md` | ✅ Đã hoàn thành |
| **Kiến trúc KV260 Simulation** | `model_notebook/markdowns/kien_truc_phan_cung_va_mo_phong_kv260.md` | ✅ Đang khởi tạo |
| **Engine Mô Phỏng & Model** | `model_notebook/kv260_hardware_engine.py` | ⏳ Chờ phê duyệt |
| **Pipeline Video Stream** | `model_notebook/video_pipeline.py` | ⏳ Chờ phê duyệt |
| **Giao diện Edge AI HUD** | `model_notebook/hud_overlay.py` | ⏳ Chờ phê duyệt |
| **File Python Notebook-Ready** | `model_notebook/Video_image_classification_VIT.py` | ⏳ Chờ phê duyệt |
| **File Jupyter Notebook** | `model_notebook/Video_image_classification_VIT.ipynb` | ⏳ Chờ phê duyệt |
| **Báo cáo Đánh Giá Toàn Diện** | `model_notebook/markdowns/bao_cao_danh_gia_hieu_nang_vit_kv260.md`| ⏳ Chờ phê duyệt |

---
*Tài liệu được thiết kế bởi Senior AI/Embedded Systems Engineer cho Đồ Án Tốt Nghiệp Kỹ Thuật Máy Tính/Khoa Học Máy Tính.*
