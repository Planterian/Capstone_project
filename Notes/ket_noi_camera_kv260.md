# Kết Nối Camera KV260 v2.0 (MobileViT-XXS Input Pipeline)

> **Dự án:** Triển khai End-to-End HW-SW Codesign MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo kích thước ảnh MobileViT $256 \times 256 \times 3$)  
> **Tài liệu gốc tham chiếu:** [`Notes/ket_noi_camera_kv260.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/ket_noi_camera_kv260.md)

---

## 1. TIỀN XỬ LÝ KHUNG HÌNH CHO MOBILEVIT-XXS

Khác với các mô hình CNN thông thường (224x224), **MobileViT-XXS** nhận đầu vào chuẩn kích thước **$256 \times 256 \times 3$**:
```python
import cv2
import numpy as np

# Mở Webcam UVC trên bo mạch Kria KV260
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Tiền xử lý ảnh theo chuẩn ImageNet và MobileViT
frame_resized = cv2.resize(frame, (256, 256))
frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
normalized = (frame_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
tensor_input = np.transpose(normalized, (2, 0, 1))[np.newaxis, :].astype(np.float32)
```
Dữ liệu tensor này sau đó được nạp vào CNN Stem trên ARM PS, tạo thành 256 tokens đưa xuống bộ tăng tốc Attention trên FPGA PL.
