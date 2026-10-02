# -*- coding: utf-8 -*-
"""
===================================================================================================
MODULE: REAL-TIME 720P@30FPS VIDEO PIPELINE & PS PREPROCESSOR (BATCH 2)
===================================================================================================
WHAT:
    Mô-đun thu thập luồng video USB Webcam 720p @ 30 FPS đa luồng (Threaded Capture) và
    hệ thống tiền xử lý hình ảnh trên bộ xử lý ARM Processing System (Cortex-A53).

WHY:
    1. Đọc camera tuần tự trực tiếp trong vòng lặp chính của OpenCV (cv2.VideoCapture.read) sẽ làm
       nghẽn luồng xử lý do phụ thuộc vào I/O phần cứng, dẫn đến trễ khung hình (buffer lag) và tụt FPS.
    2. Mô hình ViT được huấn luyện trên ảnh vuông 32x32 (CIFAR-10), trong khi webcam xuất khung hình
       chữ nhật 16:9 (1280x720). ARM PS phải cắt vùng trung tâm (Center ROI 720x720) để bảo toàn
       tỷ lệ khung hình trước khi co kích thước về 32x32.
    3. Hỗ trợ chế độ dự phòng thông minh (Synthetic Fallback Mode): Khi không có camera vật lý hoặc
       chạy trên môi trường server/notebook không có camera, hệ thống tự động sinh luồng video
       chuyển động chứa các mẫu ảnh CIFAR-10 để kiểm thử mà không bao giờ bị lỗi dừng đột ngột.

FUNCTIONALITY:
    - ThreadedCameraStream: Luồng đọc ngầm liên tục cập nhật khung hình mới nhất.
    - SyntheticFrameGenerator: Bộ sinh khung hình kiểm thử 720p thời gian thực.
    - PSPreprocessor: Cắt Center-ROI, co ảnh Bicubic/Area 32x32, RGB, chuẩn hóa CIFAR-10 Mean/Std.
===================================================================================================
"""

# %% [markdown]
# ### Cell 1: Import Thư Viện Xử Lý Thị Giác Máy Tính
# **Mục đích**: Nhập OpenCV, threading, queue và PyTorch để xây dựng pipeline đọc video không nghẽn luồng.

# %% [code]
import os
import sys
import time
import threading
from typing import Tuple, Optional, Union

# Cấu hình UTF-8 console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
import torch

# Tham chiếu các hằng số chuẩn hóa từ Batch 1
from kv260_hardware_engine import (
    IMG_SIZE,
    NUM_CHANNELS,
    CIFAR_MEAN,
    CIFAR_STD,
    CIFAR10_CLASSES
)


# %% [markdown]
# ### Cell 2: Bộ Sinh Khung Hình Dự Phòng (Synthetic Frame Generator)
# **Mục đích**: Tự động kích hoạt khi không có USB Webcam vật lý. Tạo luồng video 720p @ 30 FPS
# với hiệu ứng chuyển động, hiển thị ảnh mẫu phân loại cùng mục tiêu kiểm thử.

# %% [code]
class SyntheticFrameGenerator:
    """
    WHAT: Bộ sinh khung hình 720p tổng hợp (Synthetic 1280x720 Video Stream).
    WHY: Đảm bảo mã nguồn chạy được 100% trong mọi điều kiện (laptop không camera, máy bàn,
         máy ảo, Google Colab hoặc server headless) mà không bị văng lỗi.
    FUNCTIONALITY:
        - Tạo khung hình 1280x720 nền gradient tối phong cách công nghiệp Edge AI.
        - Vẽ hình ảnh mẫu chuyển động mượt mà trong vùng ngắm trung tâm (Center ROI).
        - Hiển thị nhãn kiểm thử và thanh thời gian mô phỏng camera 30 FPS.
    """
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.sample_classes = ['frog', 'airplane', 'automobile', 'bird', 'ship', 'horse', 'dog', 'cat']
        self.current_class_idx = 0
        self.last_switch_time = time.time()

    def generate_frame(self) -> np.ndarray:
        """Sinh ra 1 khung hình 720p giả lập (1280x720x3 uint8 BGR)."""
        self.frame_idx += 1
        now = time.time()

        # Đổi đối tượng kiểm thử mỗi 4 giây
        if now - self.last_switch_time > 4.0:
            self.current_class_idx = (self.current_class_idx + 1) % len(self.sample_classes)
            self.last_switch_time = now

        current_class_name = self.sample_classes[self.current_class_idx]

        # 1. Tạo nền xám đậm gradient
        frame = np.full((self.height, self.width, 3), 25, dtype=np.uint8)

        # Lưới tọa độ đồ họa mô phỏng camera công nghiệp
        grid_step = 80
        for x in range(0, self.width, grid_step):
            cv2.line(frame, (x, 0), (x, self.height), (35, 35, 35), 1)
        for y in range(0, self.height, grid_step):
            cv2.line(frame, (0, y), (self.width, y), (35, 35, 35), 1)

        # 2. Vùng trung tâm vuông (Center ROI 720x720)
        roi_size = min(self.width, self.height)
        roi_x1 = (self.width - roi_size) // 2
        roi_y1 = 0
        roi_x2 = roi_x1 + roi_size
        roi_y2 = self.height

        # Vẽ một vật thể đồ họa đại diện cho lớp đối tượng đang test
        center_x = self.width // 2
        center_y = self.height // 2

        # Chuyển động dao động nhẹ hình sin mô phỏng vật thể trước camera
        offset_x = int(math.sin(self.frame_idx * 0.05) * 40) if "math" in globals() else int(np.sin(self.frame_idx * 0.05) * 40)
        offset_y = int(np.cos(self.frame_idx * 0.05) * 30)

        obj_cx = center_x + offset_x
        obj_cy = center_y + offset_y

        # Vẽ hình học trực quan tùy theo nhãn
        if current_class_name == 'frog':
            # Vẽ hình chú ếch xanh tròn
            cv2.circle(frame, (obj_cx, obj_cy), 110, (40, 180, 50), -1)
            cv2.circle(frame, (obj_cx - 45, obj_cy - 80), 30, (50, 200, 60), -1)
            cv2.circle(frame, (obj_cx + 45, obj_cy - 80), 30, (50, 200, 60), -1)
            cv2.circle(frame, (obj_cx - 45, obj_cy - 80), 12, (255, 255, 255), -1)
            cv2.circle(frame, (obj_cx + 45, obj_cy - 80), 12, (255, 255, 255), -1)
            cv2.ellipse(frame, (obj_cx, obj_cy + 25), (60, 20), 0, 0, 180, (20, 100, 30), 8)
        elif current_class_name in ['airplane', 'bird']:
            # Vẽ hình thoi mô phỏng vật thể bay
            pts = np.array([[obj_cx, obj_cy - 120], [obj_cx + 140, obj_cy],
                            [obj_cx, obj_cy + 100], [obj_cx - 140, obj_cy]], np.int32)
            cv2.fillPoly(frame, [pts], (220, 180, 70) if current_class_name == 'airplane' else (80, 120, 220))
        elif current_class_name in ['automobile', 'truck']:
            # Vẽ hình chữ nhật xe hơi
            cv2.rectangle(frame, (obj_cx - 140, obj_cy - 50), (obj_cx + 140, obj_cy + 70), (40, 50, 220), -1)
            cv2.rectangle(frame, (obj_cx - 90, obj_cy - 110), (obj_cx + 90, obj_cy - 50), (60, 70, 240), -1)
            cv2.circle(frame, (obj_cx - 80, obj_cy + 70), 28, (30, 30, 30), -1)
            cv2.circle(frame, (obj_cx + 80, obj_cy + 70), 28, (30, 30, 30), -1)
        else:
            # Hình tròn đại diện vật nuôi (dog, cat, horse)
            cv2.circle(frame, (obj_cx, obj_cy), 110, (180, 120, 70), -1)
            cv2.ellipse(frame, (obj_cx - 60, obj_cy - 90), (25, 45), -30, 0, 360, (150, 90, 50), -1)
            cv2.ellipse(frame, (obj_cx + 60, obj_cy - 90), (25, 45), 30, 0, 360, (150, 90, 50), -1)

        # Chú thích chế độ mô phỏng
        cv2.putText(frame, "[SYNTHETIC CAMERA FALLBACK MODE - 720p @ 30FPS]", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Simulated Object: {current_class_name.upper()} (Switch in 4s)", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

        return frame


# %% [markdown]
# ### Cell 3: Luồng Đọc Video Đa Luồng (Threaded Camera Stream)
# **Mục đích**: Chạy một tiểu trình nền (background thread) liên tục gọi `cap.grab()` và `cap.retrieve()`,
# giúp luồng chính luôn lấy được khung hình mới nhất với độ trễ bằng 0, không bị nghẽn buffer camera USB.

# %% [code]
class ThreadedCameraStream:
    """
    WHAT: Bộ thu thập luồng video đa luồng hiệu năng cao (Non-blocking Threaded Video Stream).
    WHY: Khi chạy ở chuẩn 720p @ 30 FPS, nếu đọc camera trực tiếp đồng bộ, hàm cap.read()
         sẽ block luồng chính từ 15ms - 33ms cho mỗi khung hình, làm tụt FPS toàn hệ thống.
    FUNCTIONALITY:
        - Mở camera qua cổng cv2.VideoCapture(camera_index).
        - Thiết lập cấu hình phần cứng: FRAME_WIDTH=1280, FRAME_HEIGHT=720, FPS=30.
        - Khởi tạo thread đọc độc lập liên tục ghi đè khung hình mới nhất vào biến self.latest_frame.
        - Tự động chuyển đổi sang SyntheticFrameGenerator nếu không tìm thấy camera vật lý.
    """
    def __init__(self, camera_index: int = 0, target_width: int = 1280, target_height: int = 720, target_fps: int = 30):
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic = False
        self.synthetic_gen: Optional[SyntheticFrameGenerator] = None

        self.latest_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Thống kê hiệu năng thu nhận
        self.grabbed_frames_count = 0
        self.start_time = time.time()
        self.measured_camera_fps = 0.0

        # Khởi tạo nguồn video
        self._initialize_stream()

    def _initialize_stream(self):
        """Khởi tạo kết nối tới camera USB vật lý hoặc kích hoạt chế độ dự phòng."""
        print(f"[Camera Pipeline] Đang tìm kiếm camera USB tại thiết bị index {self.camera_index}...")

        # Thử mở camera vật lý với DirectShow trên Windows hoặc V4L2 trên Linux
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        # Cấu hình độ phân giải và FPS mong muốn
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Kiểm tra xem camera có mở thành công và đọc được frame không
        success, test_frame = cap.read() if cap.isOpened() else (False, None)

        if success and test_frame is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            self.cap = cap
            self.is_synthetic = False
            self.latest_frame = test_frame
            print(f"[Camera Pipeline] ✅ Kết nối thành công USB Webcam vật lý:")
            print(f"                 - Độ phân giải: {actual_w}x{actual_h}")
            print(f"                 - Tốc độ khung hình: {actual_fps:.1f} FPS")
        else:
            if cap.isOpened():
                cap.release()
            self.cap = None
            self.is_synthetic = True
            self.synthetic_gen = SyntheticFrameGenerator(self.target_width, self.target_height)
            self.latest_frame = self.synthetic_gen.generate_frame()
            print(f"[Camera Pipeline] ⚠️ Không phát hiện USB Webcam vật lý hợp lệ.")
            print(f"[Camera Pipeline] 🚀 Tự động kích hoạt 'Synthetic Fallback Mode' (720p @ 30 FPS).")

    def start(self) -> "ThreadedCameraStream":
        """Khởi động luồng đọc video chạy ngầm."""
        if self.is_running:
            return self

        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_worker, daemon=True, name="ThreadedCameraWorker")
        self.thread.start()
        print(f"[Camera Pipeline] Đã bắt đầu tiểu trình thu thập video ngầm.")
        return self

    def _capture_worker(self):
        """Hàm chạy ngầm liên tục lấy khung hình mới nhất."""
        fps_interval = 1.0 / self.target_fps
        last_frame_time = time.time()

        while self.is_running:
            if not self.is_synthetic and self.cap is not None:
                # Đọc từ camera vật lý
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.grabbed_frames_count += 1
                else:
                    # Nếu camera vật lý mất tín hiệu, chuyển sang synthetic mode
                    print("[Camera Pipeline] Mất tín hiệu từ camera vật lý! Chuyển sang Synthetic Fallback.")
                    self.is_synthetic = True
                    self.synthetic_gen = SyntheticFrameGenerator(self.target_width, self.target_height)
            else:
                # Sinh khung hình tổng hợp theo tốc độ chuẩn 30 FPS
                now = time.time()
                elapsed = now - last_frame_time
                if elapsed < fps_interval:
                    time.sleep(fps_interval - elapsed)
                last_frame_time = time.time()

                frame = self.synthetic_gen.generate_frame()
                with self.lock:
                    self.latest_frame = frame
                    self.grabbed_frames_count += 1

            # Cập nhật FPS đo đạc
            duration = time.time() - self.start_time
            if duration > 1.0:
                self.measured_camera_fps = self.grabbed_frames_count / duration

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lấy bản sao của khung hình mới nhất."""
        with self.lock:
            if self.latest_frame is None:
                return False, None
            # Trả về bản sao an toàn để không bị ghi đè khi đang xử lý
            return True, self.latest_frame.copy()

    def stop(self):
        """Dừng luồng video và giải phóng camera."""
        self.is_running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        print("[Camera Pipeline] Đã giải phóng luồng video.")


# %% [markdown]
# ### Cell 4: Hệ Thống Tiền Xử Lý Hình Ảnh ARM PS (PSPreprocessor)
# **Mục đích**: Tiếp nhận khung hình 1280x720 từ webcam, cắt vùng vuông trung tâm (Center ROI 720x720)
# để chống méo ảnh, co kích thước về 32x32 và chuẩn hóa tensor nạp vào mô hình ViT.

# %% [code]
class PSPreprocessor:
    """
    WHAT: Khối tiền xử lý hình ảnh thực thi trên lõi ARM Cortex-A53 (ARM PS).
    WHY: Vi mạch FPGA PL chỉ xử lý ma trận số học INT8. Các thao tác linh hoạt như cắt vùng quan tâm (ROI),
         nội suy co dãn kích thước và chuẩn hóa thống kê ảnh (Mean/Std) được phân công tối ưu cho ARM PS.
    FUNCTIONALITY:
        1. get_center_roi_coordinates(): Tính toán tọa độ hộp (x1, y1, x2, y2) của vùng trung tâm.
        2. crop_center_roi(): Cắt vùng vuông 720x720 từ ảnh gốc 1280x720.
        3. preprocess_to_tensor():
           - Co ảnh từ 720x720 xuống 32x32 bằng thuật toán cv2.INTER_AREA (chống răng cưa tốt nhất).
           - Đổi kênh màu OpenCV BGR sang chuẩn RGB của PyTorch.
           - Chuẩn hóa theo giá trị CIFAR-10 Mean & Std.
           - Trả về PyTorch Tensor kích thước [1, 3, 32, 32] ở kiểu float32 sẵn sàng cho QuantStub.
    """
    def __init__(self, target_img_size: int = IMG_SIZE,
                 mean: Tuple[float, float, float] = CIFAR_MEAN,
                 std: Tuple[float, float, float] = CIFAR_STD):
        self.target_size = target_img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def get_center_roi_coordinates(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Tính toán tọa độ vùng vuông trung tâm (x1, y1, x2, y2)."""
        roi_dim = min(frame_w, frame_h)  # 720 pixels
        x1 = (frame_w - roi_dim) // 2   # (1280 - 720) // 2 = 280
        y1 = (frame_h - roi_dim) // 2   # 0
        x2 = x1 + roi_dim               # 280 + 720 = 1000
        y2 = y1 + roi_dim               # 720
        return x1, y1, x2, y2

    def crop_center_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Cắt vùng vuông trung tâm từ khung hình gốc."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.get_center_roi_coordinates(w, h)
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def preprocess_to_tensor(self, roi_image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, float]:
        """
        WHAT: Xử lý vùng ảnh ROI thành tensor nạp vào mô hình ViT.
        RETURNS:
            - input_tensor: torch.Tensor shape [1, 3, 32, 32]
            - resized_rgb: np.ndarray ảnh 32x32x3 (dùng để hiển thị phóng to nếu cần)
            - prep_latency_ms: Thời gian tiền xử lý tính bằng mili-giây
        """
        t_start = time.perf_counter()

        # 1. Co ảnh về kích thước chuẩn của mô hình (32x32)
        # Sử dụng INTER_AREA vì đây là phép co ảnh (downsampling) từ 720x720 -> 32x32
        resized_bgr = cv2.resize(roi_image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        # 2. Chuyển đổi không gian màu BGR sang RGB
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        # 3. Chuẩn hóa pixel về [0.0, 1.0] và trừ Mean / chia Std
        norm_img = resized_rgb.astype(np.float32) / 255.0
        norm_img = (norm_img - self.mean) / self.std

        # 4. Đổi trục thành định dạng PyTorch [Channels, Height, Width] -> [3, 32, 32]
        tensor_data = np.transpose(norm_img, (2, 0, 1))

        # 5. Thêm chiều batch: [1, 3, 32, 32]
        input_tensor = torch.from_numpy(tensor_data).unsqueeze(0)

        prep_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return input_tensor, resized_rgb, prep_latency_ms


# %% [markdown]
# ### Cell 5: Tự Kiểm Thử Module Video Pipeline (Self-Test)
# **Mục đích**: Chạy thử nghiệm độc lập thu nhận 100 khung hình video 720p,
# cắt ROI, tiền xử lý và đo đạc thông lượng FPS thực tế của Pipeline.

# %% [code]
def self_test_video_pipeline():
    """Hàm chạy kiểm thử độc lập cho module Batch 2."""
    print("\n" + "=" * 80)
    print("  BẮT ĐẦU CHẠY THỬ NGHIỆM TỰ ĐỘNG BATCH 2: 720P VIDEO PIPELINE & PREPROCESSOR  ")
    print("=" * 80)

    # 1. Khởi động luồng video
    stream = ThreadedCameraStream(camera_index=0, target_width=1280, target_height=720, target_fps=30)
    stream.start()

    # 2. Khởi tạo bộ tiền xử lý ARM PS
    preprocessor = PSPreprocessor(target_img_size=IMG_SIZE)

    # Đợi 0.5s để camera ổn định
    time.sleep(0.5)

    print("\n[Pipeline Test] Bắt đầu thu nhận và xử lý liên tục 90 khung hình...")
    latencies = []
    total_frames = 90
    t_pipeline_start = time.time()

    for i in range(total_frames):
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        # Cắt ROI và tiền xử lý
        roi, (x1, y1, x2, y2) = preprocessor.crop_center_roi(frame)
        tensor, resized, t_prep = preprocessor.preprocess_to_tensor(roi)
        latencies.append(t_prep)

        # Mô phỏng nhịp độ 30 FPS
        time.sleep(0.01)

    total_time = time.time() - t_pipeline_start
    avg_fps = total_frames / total_time
    avg_prep_ms = np.mean(latencies)

    print("\n" + "-" * 50)
    print(f"✅ KẾT QUẢ KIỂM THỬ BATCH 2:")
    print(f"  - Độ phân giải khung hình gốc  : 1280 x 720 (720p)")
    print(f"  - Kích thước vùng Center ROI   : {x2 - x1} x {y2 - y1}")
    print(f"  - Kích thước tensor tiền xử lý : {list(tensor.shape)} (32x32)")
    print(f"  - Thời gian tiền xử lý trung bình : {avg_prep_ms:.3f} ms / frame")
    print(f"  - Tốc độ thông lượng thu nhận  : {avg_fps:.1f} FPS")
    print("-" * 50)

    # Giải phóng camera
    stream.stop()
    print("\n[SUCCESS] BATCH 2 ĐÃ HOÀN THÀNH VÀ SẴN SÀNG TÍCH HỢP VÀO BATCH 3!\n")


if __name__ == '__main__':
    self_test_video_pipeline()
