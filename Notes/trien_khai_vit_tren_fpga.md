# Triển khai ViT trên FPGA

Xây dựng ứng dụng demo **Real-time Edge AI / Hardware-in-the-Loop (HIL)**!
Sau khi huấn luyện mô hình ViT trên Google Colab, bạn chỉ cần xuất file trọng số (trọng số đã định lượng INT8 như .pth, .bin hoặc .hex), lưu về máy cục bộ (Laptop hoặc thẻ nhớ SD trên bo mạch Kria/Zynq), sau đó chạy script Python đọc luồng camera từ OpenCV.

---

##### 1. Luồng dữ liệu Thực thi Real-time (Webcam $\rightarrow$ ARM PS $\rightarrow$ FPGA PL)

```text
[ LAPTOP WEBCAM / USB CAM ]
            │ (cv2.VideoCapture)
            ▼
[ 1. FRAME READ & PREPROCESS (ARM PS / Laptop Python) ]
    ├── Bắt khung hình RGB -> Resize/Crop 224x224x3
    ├── Patch Embedding: Cắt grid 14x14 -> N = 196 Tokens
    └── QKV Projection + Quantize -> Chuẩn bị mảng INT8 (Q, K, V)
            │
            ▼ (AXI4-Stream via AXI DMA)
[ 2. HARDWARE ATTENTION ACCELERATOR (FPGA PL Engine) ]
    ├── AXI-Lite: ARM PS ghi thanh ghi điều khiển (START=1, N=196, d_k=32, H=2)
    ├── RX Stream -> Nạp dữ liệu vào Input Ping-Pong BRAM (Q, K, V)
    ├── Systolic MAC 1: Tính Q * K^T (INT32)
    ├── Scaler Unit: Dịch bit phải đại số (ASR 1/sqrt(d_k))
    ├── Hardware Softmax: Tra bảng LUT / Xấp xỉ PWL
    ├── Systolic MAC 2: Tính Score * V (INT8)
    └── TX Stream -> Đẩy kết quả Attended Output ra M_AXIS_TDATA
            │
            ▼ (AXI4-Stream via AXI DMA)
[ 3. POST-PROCESSING & DISPLAY (ARM PS / Laptop Python) ]
    ├── Nhận Attended Output từ FPGA PL
    ├── Tính toán phần còn lại: LayerNorm + MLP + Classification Head
    └── Hiển thị nhãn dự đoán (Label & FPS) đè lên khung hình webcam (cv2.imshow)
```

---

##### 2. Kịch bản Mã nguồn Python Mẫu (realtime_vit_inference.py)

```python
import cv2
import numpy as np
import time
from pynq import Overlay, allocate

# 1. Nạp Bitstream phần cứng & Khai báo IP Core
ol = Overlay("attention_core.bit")
dma = ol.axi_dma_0
attn_ip = ol.attention_core_0

# Kích thước dữ liệu chuẩn hóa
N = 196      # 14x14 Patch Grid
D_K = 32     # Head dimension
INT8_BYTES = N * D_K

# 2. Cấp phát bộ nhớ vật lý liên tục (CMA) cho AXI DMA
in_q = allocate(shape=(N, D_K), dtype=np.int8)
in_k = allocate(shape=(N, D_K), dtype=np.int8)
in_v = allocate(shape=(N, D_K), dtype=np.int8)
out_attn = allocate(shape=(N, D_K), dtype=np.int8)

# 3. Khởi tạo Webcam với OpenCV
cap = cv2.VideoCapture(0) # 0: Camera mặc định của laptop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Đã khởi tạo Webcam & FPGA IP Core. Bắt đầu luồng suy luận Real-time...")

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # --- STEP A: Preprocessing (ARM PS) ---
        img_resized = cv2.resize(frame, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        q_mat, k_mat, v_mat = preprocess_to_qkv_int8(img_rgb)

        np.copyto(in_q, q_mat)
        np.copyto(in_k, k_mat)
        np.copyto(in_v, v_mat)

        # --- STEP B: FPGA Execution (Attention Engine PL) ---
        attn_ip.write(0x10, N)     # Ghi N=196
        attn_ip.write(0x18, D_K)   # Ghi d_k=32
        attn_ip.write(0x00, 0x01)  # START Signal

        dma.recvchannel.transfer(out_attn)
        dma.sendchannel.transfer(in_q)

        dma.sendchannel.wait()
        dma.recvchannel.wait()

        # --- STEP C: Post-Processing & Class Inference (ARM PS) ---
        class_id, confidence = run_mlp_head_int8(out_attn)

        # --- STEP D: Render kết quả đè lên Webcam ---
        fps = 1.0 / (time.time() - start_time)
        label_text = f"Class: {class_id} ({confidence*100:.1f}%) | FPS: {fps:.1f}"

        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("FPGA ViT Real-time Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng luồng Webcam.")
```

---

##### 3. Ưu điểm của giải pháp này đối với Đề tài Capstone
1. **Khắc phục triệt để hạn chế nạp lại Dataset trên Colab**: Bạn chỉ cần train 1 lần trên Colab, tải file weights .pth về máy. Khi test/demo chỉ cần chạy script local với webcam mà không tốn thời gian tải lại dữ liệu.
2. **Minh chứng tính ứng dụng thực tế (Live Demo)**: Việc hiển thị kết quả phân loại ảnh trực tiếp từ webcam với chỉ số FPS realtime là bằng chứng thuyết phục nhất cho thấy khối hardware accelerator trên FPGA hoạt động chính xác và có độ trễ thấp.
3. **Phân chia khối lượng công việc hoàn hảo (HW/SW Partitioning)**:
    * **Phần mềm (Python/OpenCV)**: Đảm nhận các việc biến đổi linh hoạt (đọc camera, resize, vẽ giao diện UI).
    * **Phần cứng (FPGA PL Attention Core)**: Đảm nhận phần tính toán nhân ma trận nặng nề nhất $O(N^2)$ với tốc độ cao nhờ **Systolic MAC Array** và **AXI4-Stream Pipeline**.
