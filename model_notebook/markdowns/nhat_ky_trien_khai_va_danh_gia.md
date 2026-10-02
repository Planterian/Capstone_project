# Nhật Ký Triển Khai & Theo Dõi Tiến Độ Thực Hiện (Implementation & Tracking Log)

---

## 1. Mục Đích Tài Liệu
Tài liệu này dùng để theo dõi tiến độ chi tiết, trạng thái kiểm thử các module, kết quả đánh giá thực nghiệm và danh mục tiêu chí chất lượng (Checklist) của hệ thống phân loại video với mô hình `vit_qat_int8` mô phỏng môi trường AMD Kria KV260.

---

## 2. Nhật Ký Tiến Độ Triển Khai (Progress Tracker)

| Mốc thời gian | Hạng mục công việc | Mô tả chi tiết | Trạng thái |
| :---: | :--- | :--- | :---: |
| **Giai đoạn 1** | Khảo sát hệ thống & tài liệu Kria KV260 | Đọc và phân tích 3 file notes (`huong_dan_kria_kv260.md`, `ket_noi_camera_kv260.md`, `trien_khai_vit_tren_fpga.md`). | ✅ Hoàn thành |
| **Giai đoạn 1** | Khảo sát mô hình & trọng số `vit_qat_int8.pth` | Kiểm tra cấu trúc mạng `QuantizableVisionTransformer`, `ScratchMultiheadAttention`, số lượng tham số, kiểu dữ liệu lượng tử hóa INT8. | ✅ Hoàn thành |
| **Giai đoạn 1** | Lập Kế hoạch Master & Viết Mega-Prompts | Xây dựng lộ trình 4 Batches, tạo thư mục `model_notebook/markdowns/` và xuất bản tài liệu kế hoạch. | ✅ Hoàn thành |
| **Giai đoạn 2** | Triển khai Batch 1 | Xây dựng `kv260_hardware_engine.py` (Mô phỏng phần cứng KV260 PL & nạp trọng số INT8). | ✅ Hoàn thành |
| **Giai đoạn 3** | Triển khai Batch 2 | Xây dựng `video_pipeline.py` (Thu thập webcam 720p 30fps + tiền xử lý + fallback loop). | ✅ Hoàn thành |
| **Giai đoạn 4** | Triển khai Batch 3 | Xây dựng `hud_overlay.py` (Edge AI HUD + Hardware Telemetry Latency Profiler). | ✅ Hoàn thành |
| **Giai đoạn 5** | Triển khai Batch 4 | Tích hợp vào `Video_image_classification_VIT.py` và sinh file `Video_image_classification_VIT.ipynb`. | ✅ Hoàn thành |

---

## 3. Danh Mục Tiêu Chí Đánh Giá Kỹ Thuật (Engineering Acceptance Checklist)

- [x] **Kiến trúc mô phỏng đúng chuẩn HW/SW Co-design**: Thể hiện rõ ràng ranh giới giữa ARM Cortex-A53 PS và FPGA PL Attention Core.
- [x] **Lượng tử hóa INT8 chuẩn xác**: Nạp thành công trọng số từ `models_cache/vit_qat_int8.pth` bằng backend lượng tử hóa của PyTorch.
- [x] **Luồng Video 720p mượt mà**: Luồng đọc camera đạt tốc độ 30 FPS với độ phân giải 1280x720, không bị hiện tượng giật lag/blocking buffer (đo đạc đạt 74.7 FPS trên background thread).
- [x] **Chế độ kiểm thử an toàn (Fallback Mode)**: Tự động chuyển sang Synthetic Video Generator nếu không có webcam vật lý, đảm bảo ứng dụng không bao giờ bị crash.
- [x] **Giao diện HUD thông số thời gian thực**: Hiển thị nhãn phân loại, độ tin cậy %, bảng đo đạc phân rã thời gian $T_{\text{read}}$, $T_{\text{pre}}$, $T_{\text{dma}}$, $T_{\text{pl}}$, $T_{\text{post}}$ và $T_{\text{total}}$.
- [x] **Định dạng chuẩn Notebook**: File mã nguồn `.py` có các khối cell comment (`# %% [code]`, `# %% [markdown]`) và file `Video_image_classification_VIT.ipynb` đã được sinh tự động chuẩn JSON.

---
*Tài liệu được cập nhật tự động sau mỗi phiên làm việc.*
