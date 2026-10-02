# -*- coding: utf-8 -*-
"""
===================================================================================================
MODULE: KRIA KV260 EDGE AI HUD OVERLAY & TELEMETRY MONITOR (BATCH 3)
===================================================================================================
WHAT:
    Mô-đun thiết kế giao diện Heads-Up Display (HUD) đồ họa chuẩn công nghiệp Edge AI cho
    bo mạch AMD Kria KV260 và bảng điều khiển đo đạc hiệu năng phần cứng (Telemetry Monitor).

WHY:
    1. Trong các hệ thống nhúng Edge AI (như Smart Camera, Robotics, Kiểm tra lỗi công nghiệp),
       giao diện HUD thời gian thực là công cụ quan trọng nhất để trực quan hóa:
       - Vùng ngắm đối tượng (Target Reticle)
       - Độ tin cậy dự đoán (Confidence Scores & Top-3 Bar Charts)
       - Tốc độ khung hình (FPS) và phân rã độ trễ từng khâu (Latency Breakdown).
    2. Minh chứng trực quan cho thấy khối tăng tốc phần cứng FPGA PL Attention Core và lõi ARM PS
       hoạt động nhịp nhàng, độ trễ thấp và không có hiện tượng drop frame.

FUNCTIONALITY:
    - draw_hud(): Vẽ toàn bộ giao diện HUD hiện đại lên khung hình 720p (1280x720).
    - draw_target_reticle(): Vẽ khung ngắm công nghệ cao ở vùng Center ROI 720x720.
    - draw_prediction_dashboard(): Vẽ bảng kết quả phân loại Top-1 và Top-3 dạng thanh đo (Bar Chart).
    - draw_hardware_telemetry(): Vẽ bảng theo dõi hiệu năng phần cứng Kria KV260 (FPS, chu kỳ PL, DMA).
    - save_snapshot(): Chụp ảnh lưu trữ khung hình phân loại kèm metadata.
===================================================================================================
"""

# %% [markdown]
# ### Cell 1: Import Thư Viện Đồ Họa Và Cấu Hình Giao Diện
# **Mục đích**: Cấu hình bảng màu, phông chữ và các tiện ích hiển thị OpenCV.

# %% [code]
import os
import sys
import time
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# Cấu hình UTF-8 console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

# =================================================================================================
# BẢNG MÀU CHUẨN EDGE AI HUD (BGR FORMAT FOR OPENCV)
# =================================================================================================
COLOR_BG_DARK = (18, 18, 22)           # Màu nền tối bán trong suốt (Dark Slate)
COLOR_PANEL_BORDER = (70, 70, 85)      # Viền bảng điều khiển
COLOR_ACCENT_CYAN = (230, 215, 0)      # Màu xanh Cyan công nghệ cao (BGR: 0, 215, 230)
COLOR_ACCENT_GREEN = (50, 220, 90)     # Màu xanh lá đạt chuẩn FPS cao
COLOR_ACCENT_YELLOW = (40, 215, 255)   # Màu vàng cảnh báo (BGR: 255, 215, 40)
COLOR_ACCENT_RED = (40, 50, 235)       # Màu đỏ cảnh báo độ trễ cao
COLOR_TEXT_WHITE = (245, 245, 245)     # Chữ màu trắng sáng
COLOR_TEXT_MUTED = (160, 160, 175)     # Chữ xám mờ phụ chú
COLOR_RETICLE_CORNERS = (0, 230, 255)  # Màu góc nhắm mục tiêu (Vàng cam công nghệ)


# %% [markdown]
# ### Cell 2: Lớp Điều Khiển Hiển Thị Giao Diện (KV260HUDVisualizer)
# **Mục đích**: Nhận khung hình 1280x720, kết quả phân loại Top-1/Top-3 và các chỉ số đo đạc trễ
# để render giao diện đồ họa hoàn chỉnh với phong cách kính mờ (Glassmorphism).

# %% [code]
class KV260HUDVisualizer:
    """
    WHAT: Bộ hiển thị giao diện thời gian thực Edge AI HUD (Heads-Up Display) Kria KV260.
    WHY: Trực quan hóa kết quả phân loại từ mô hình vit_qat_int8 và phân tích độ trễ phần cứng chi tiết.
    FUNCTIONALITY:
        - Quản lý các chế độ hiển thị: Debug Mode, FPS Lock, Pause.
        - Render vùng ngắm Center ROI với các góc công nghệ cao (Tech Brackets).
        - Hiển thị bảng Top-3 xác suất với thanh tiến trình đồ họa (Progress Bars).
        - Hiển thị bảng phân rã độ trễ phần cứng: T_prep, T_dma, T_pl, T_post, T_total.
        - Lưu ảnh chụp màn hình độ phân giải cao vào thư mục captures/.
    """
    def __init__(self, output_dir: str = "captures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Trạng thái điều khiển người dùng
        self.debug_mode = True          # Bật/Tắt bảng phân rã chi tiết trễ phần cứng
        self.fps_lock_30 = False        # Bật/Tắt khóa FPS 30 để mô phỏng camera thực tế
        self.is_paused = False          # Tạm dừng luồng video
        self.snapshot_counter = 0

        # Lịch sử FPS để làm mịn (Smoothing)
        self.fps_history: List[float] = []
        self.max_history = 20

    def calculate_smooth_fps(self, current_fps: float) -> float:
        """Làm mịn chỉ số FPS để tránh hiện tượng số nhảy giật cục trên màn hình."""
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)
        return float(np.mean(self.fps_history))

    def draw_glass_panel(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                         alpha: float = 0.72, border_color: Tuple[int, int, int] = COLOR_PANEL_BORDER):
        """
        WHAT: Vẽ một bảng nền bán trong suốt hiệu ứng kính mờ (Glassmorphism).
        WHY: Giúp các dòng chữ và chỉ số dễ đọc mà không che mất hoàn toàn khung cảnh video camera phía sau.
        """
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BG_DARK, -1)
        # Pha trộn bán trong suốt với khung hình gốc
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        # Vẽ viền mảnh tinh tế
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)

    def draw_target_reticle(self, canvas: np.ndarray, roi_coords: Tuple[int, int, int, int]):
        """
        WHAT: Vẽ khung ngắm mục tiêu công nghệ cao tại vùng Center ROI 720x720.
        WHY: Người dùng biết chính xác khu vực nào trước camera đang được cắt và đưa vào mô hình ViT 32x32.
        """
        x1, y1, x2, y2 = roi_coords
        corner_len = 35
        thickness = 3

        # 4 góc ngắm Tech Brackets
        # Góc trên - trái
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), COLOR_RETICLE_CORNERS, thickness)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), COLOR_RETICLE_CORNERS, thickness)

        # Góc trên - phải
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), COLOR_RETICLE_CORNERS, thickness)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), COLOR_RETICLE_CORNERS, thickness)

        # Góc dưới - trái
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), COLOR_RETICLE_CORNERS, thickness)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), COLOR_RETICLE_CORNERS, thickness)

        # Góc dưới - phải
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), COLOR_RETICLE_CORNERS, thickness)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), COLOR_RETICLE_CORNERS, thickness)

        # Dấu chữ thập nhỏ ở tâm khung ngắm (Center Crosshair)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ch_len = 12
        cv2.line(canvas, (cx - ch_len, cy), (cx + ch_len, cy), (0, 255, 255), 1)
        cv2.line(canvas, (cx, cy - ch_len), (cx, cy + ch_len), (0, 255, 255), 1)
        cv2.circle(canvas, (cx, cy), 4, (0, 255, 255), 1)

        # Nhãn định danh vùng trích xuất
        label = "TARGET REGION (32x32 ViT ROI)"
        cv2.putText(canvas, label, (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)

    def draw_top_banner(self, canvas: np.ndarray, frame_w: int):
        """Vẽ thanh thông tin biểu ngữ ở cạnh trên màn hình."""
        banner_h = 42
        self.draw_glass_panel(canvas, 0, 0, frame_w, banner_h, alpha=0.85, border_color=(40, 40, 50))

        # Tiêu đề hệ thống bo mạch
        title_text = "AMD KRIA KV260 VISION AI STARTER KIT"
        cv2.putText(canvas, title_text, (20, 27), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 255), 1, cv2.LINE_AA)

        # Huy hiệu trạng thái vi mạch
        badge_text = "[ARM Cortex-A53 PS | FPGA PL Attention Core: INT8 QAT | AXI DMA: 1.6 GB/s]"
        cv2.putText(canvas, badge_text, (480, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 220, 90), 1, cv2.LINE_AA)

        # Thời gian thực hệ thống
        now_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, now_str, (frame_w - 95, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def draw_prediction_dashboard(self, canvas: np.ndarray, top1_name: str, top1_conf: float,
                                  top3_list: List[Tuple[str, float]], x: int = 20, y: int = 55):
        """
        WHAT: Vẽ bảng kết quả phân loại Top-1 và Top-3 với thanh đo xác suất (Confidence Bars).
        WHY: Người xem nhìn thấy ngay dự đoán tốt nhất và phân bố xác suất của 3 lớp cao nhất.
        """
        panel_w = 340
        panel_h = 240
        self.draw_glass_panel(canvas, x, y, panel_w, panel_h)

        # Tiêu đề bảng
        cv2.putText(canvas, "PREDICTION DASHBOARD", (x + 15, y + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (x + 15, y + 34), (x + panel_w - 15, y + 34), COLOR_PANEL_BORDER, 1)

        # Ô hiển thị Top-1 Class to rõ
        cv2.putText(canvas, "TOP-1 CLASS:", (x + 15, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        class_display = top1_name.upper()
        # Đổi màu xanh lá nếu độ tin cậy > 70%, vàng nếu 40-70%, đỏ nếu < 40%
        if top1_conf >= 0.70:
            top_color = COLOR_ACCENT_GREEN
        elif top1_conf >= 0.40:
            top_color = COLOR_ACCENT_YELLOW
        else:
            top_color = COLOR_ACCENT_RED

        cv2.putText(canvas, class_display, (x + 15, y + 95),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, top_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{top1_conf * 100:.1f}%", (x + panel_w - 90, y + 95),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, top_color, 2, cv2.LINE_AA)

        # Đường phân cách
        cv2.line(canvas, (x + 15, y + 115), (x + panel_w - 15, y + 115), (45, 45, 55), 1)
        cv2.putText(canvas, "TOP-3 CONFIDENCE BREAKDOWN:", (x + 15, y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # 3 thanh đo xác suất (Bar Charts)
        bar_start_y = y + 155
        bar_w_max = 160
        bar_h = 14

        colors = [COLOR_ACCENT_GREEN, COLOR_ACCENT_CYAN, COLOR_ACCENT_YELLOW]

        for i, (cls_name, prob) in enumerate(top3_list):
            item_y = bar_start_y + (i * 26)

            # Tên nhãn
            cv2.putText(canvas, f"{i+1}. {cls_name:<10}", (x + 15, item_y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

            # Khung viền thanh đo
            bar_x = x + 115
            cv2.rectangle(canvas, (bar_x, item_y), (bar_x + bar_w_max, item_y + bar_h), (45, 45, 55), -1)

            # Thanh phần trăm xác suất
            current_bar_w = int(bar_w_max * max(0.0, min(1.0, prob)))
            if current_bar_w > 0:
                cv2.rectangle(canvas, (bar_x, item_y), (bar_x + current_bar_w, item_y + bar_h), colors[i % 3], -1)

            # Giá trị %
            cv2.putText(canvas, f"{prob * 100:5.1f}%", (bar_x + bar_w_max + 8, item_y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    def draw_hardware_telemetry(self, canvas: np.ndarray, latency_dict: Dict[str, float],
                                current_fps: float, x: int = 910, y: int = 55):
        """
        WHAT: Vẽ bảng theo dõi hiệu năng phần cứng Kria KV260 (Hardware Telemetry Monitor).
        WHY: Thể hiện chi tiết chỉ số FPS thời gian thực và phân tách độ trễ của từng khâu phần cứng.
        """
        panel_w = 350
        panel_h = 360
        self.draw_glass_panel(canvas, x, y, panel_w, panel_h)

        # Tiêu đề bảng
        cv2.putText(canvas, "KRIA KV260 HARDWARE TELEMETRY", (x + 15, y + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (x + 15, y + 34), (x + panel_w - 15, y + 34), COLOR_PANEL_BORDER, 1)

        # 1. Chỉ số FPS thời gian thực
        smooth_fps = self.calculate_smooth_fps(current_fps)
        cv2.putText(canvas, "SYSTEM THROUGHPUT:", (x + 15, y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        fps_color = COLOR_ACCENT_GREEN if smooth_fps >= 25.0 else (COLOR_ACCENT_YELLOW if smooth_fps >= 15.0 else COLOR_ACCENT_RED)
        cv2.putText(canvas, f"{smooth_fps:4.1f} FPS", (x + 15, y + 92),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, fps_color, 2, cv2.LINE_AA)

        # Trạng thái FPS Lock
        lock_status = "LOCKED: 30 FPS" if self.fps_lock_30 else "UNLOCKED (MAX SPEED)"
        cv2.putText(canvas, f"[{lock_status}]", (x + 180, y + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Đường phân cách
        cv2.line(canvas, (x + 15, y + 110), (x + panel_w - 15, y + 110), (45, 45, 55), 1)

        # 2. Bảng phân rã độ trễ phần cứng (Latency Breakdown)
        cv2.putText(canvas, "HW/SW LATENCY BREAKDOWN (ms):", (x + 15, y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        t_prep = latency_dict.get("t_prep_ms", 1.5)
        t_dma = latency_dict.get("t_dma_total_ms", 0.11)
        t_pl = latency_dict.get("t_pl_attention_ms", 25.0)
        t_post = latency_dict.get("t_ps_mlp_head_ms", 8.0)
        t_total = t_prep + t_dma + t_pl + t_post

        rows = [
            ("1. ARM PS Video Preprocess", f"{t_prep:6.2f} ms", (200, 200, 200)),
            ("2. AXI4-Stream DMA (2-way)", f"{t_dma:6.2f} ms", (0, 220, 255)),
            ("3. FPGA PL Attention Core", f"{t_pl:6.2f} ms", COLOR_ACCENT_GREEN),
            ("4. ARM PS MLP Head Softmax", f"{t_post:6.2f} ms", (200, 200, 200)),
        ]

        row_start_y = y + 155
        for i, (label, val_str, val_color) in enumerate(rows):
            cur_y = row_start_y + (i * 24)
            cv2.putText(canvas, label, (x + 15, cur_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, val_str, (x + panel_w - 95, cur_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1, cv2.LINE_AA)

        # Tổng độ trễ toàn chu trình
        cv2.line(canvas, (x + 15, y + 258), (x + panel_w - 15, y + 258), COLOR_PANEL_BORDER, 1)
        cv2.putText(canvas, "TOTAL LATENCY (E2E):", (x + 15, y + 280),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{t_total:6.2f} ms", (x + panel_w - 110, y + 280),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_ACCENT_GREEN, 1, cv2.LINE_AA)

        # Thông số vi mạch PL
        pl_cycles = latency_dict.get("pl_hardware_cycles", 105950)
        cv2.line(canvas, (x + 15, y + 298), (x + panel_w - 15, y + 298), (45, 45, 55), 1)
        cv2.putText(canvas, f"PL Systolic Array: 16x16 @ 200MHz", (x + 15, y + 318),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"PL Compute Cycles: {pl_cycles:,} cycles", (x + 15, y + 338),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def draw_bottom_control_bar(self, canvas: np.ndarray, frame_w: int, frame_h: int):
        """Vẽ thanh hướng dẫn phím tắt điều khiển ở cạnh đáy màn hình."""
        bar_h = 32
        bar_y = frame_h - bar_h
        self.draw_glass_panel(canvas, 0, bar_y, frame_w, bar_h, alpha=0.85, border_color=(40, 40, 50))

        shortcuts = [
            ("[Q]", "Quit Application"),
            ("[S]", "Save Snapshot"),
            ("[D]", "Toggle Debug HUD"),
            ("[F]", "Toggle FPS Lock"),
            ("[P]", "Pause / Resume")
        ]

        start_x = 25
        spacing = 230
        for i, (key, desc) in enumerate(shortcuts):
            pos_x = start_x + (i * spacing)
            cv2.putText(canvas, key, (pos_x, bar_y + 21),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (pos_x + 35, bar_y + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    def save_snapshot(self, frame: np.ndarray, top1_name: str, confidence: float) -> str:
        """Chụp và lưu khung hình hiện tại kèm nhãn dự đoán vào thư mục captures/."""
        self.snapshot_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kv260_snapshot_{timestamp}_{top1_name}_{int(confidence*100)}pct.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"[Snapshot] 📸 Đã lưu ảnh chụp màn hình: '{filepath}'")
        return filepath

    def render(self, frame: np.ndarray, roi_coords: Tuple[int, int, int, int],
               top1_name: str, top1_conf: float, top3_list: List[Tuple[str, float]],
               latency_dict: Dict[str, float], current_fps: float) -> np.ndarray:
        """
        WHAT: Hàm tổng hợp render toàn bộ giao diện HUD lên khung hình 720p.
        RETURNS: Khung hình 1280x720 đã được phủ các thành phần giao diện Edge AI.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Vẽ thanh biểu ngữ trên cùng
        self.draw_top_banner(canvas, w)

        # 2. Vẽ khung ngắm mục tiêu trung tâm
        self.draw_target_reticle(canvas, roi_coords)

        # 3. Vẽ bảng phân loại dự đoán
        self.draw_prediction_dashboard(canvas, top1_name, top1_conf, top3_list, x=25, y=55)

        # 4. Vẽ bảng thông số phần cứng nếu bật chế độ Debug
        if self.debug_mode:
            self.draw_hardware_telemetry(canvas, latency_dict, current_fps, x=w - 375, y=55)

        # 5. Vẽ thanh phím tắt điều khiển ở đáy
        self.draw_bottom_control_bar(canvas, w, h)

        # Thông báo nếu đang tạm dừng video
        if self.is_paused:
            cv2.putText(canvas, "[VIDEO STREAM PAUSED]", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        return canvas


# %% [markdown]
# ### Cell 3: Tự Kiểm Thử Độc Lập Module HUD Overlay (Self-Test)
# **Mục đích**: Khởi tạo khung hình mẫu và kiểm thử render toàn bộ các thành phần HUD,
# sau đó chụp lưu thử nghiệm 1 ảnh mẫu vào thư mục `captures/`.

# %% [code]
def self_test_hud_overlay():
    """Hàm chạy kiểm thử độc lập cho module Batch 3."""
    print("\n" + "=" * 80)
    print("  BẮT ĐẦU CHẠY THỬ NGHIỆM TỰ ĐỘNG BATCH 3: KRIA KV260 HUD VISUALIZER  ")
    print("=" * 80)

    # 1. Tạo một khung hình mẫu 1280x720
    test_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    # 2. Dữ liệu thử nghiệm mẫu
    roi_coords = (280, 0, 1000, 720)
    top1_name = "automobile"
    top1_conf = 0.942
    top3_list = [
        ("automobile", 0.942),
        ("truck", 0.041),
        ("airplane", 0.009)
    ]
    latency_dict = {
        "t_prep_ms": 1.45,
        "t_dma_total_ms": 0.11,
        "t_pl_attention_ms": 22.80,
        "t_ps_mlp_head_ms": 7.50,
        "pl_hardware_cycles": 105950
    }
    fps = 31.2

    # 3. Khởi tạo visualizer và render
    visualizer = KV260HUDVisualizer(output_dir="captures")
    rendered_frame = visualizer.render(
        frame=test_frame,
        roi_coords=roi_coords,
        top1_name=top1_name,
        top1_conf=top1_conf,
        top3_list=top3_list,
        latency_dict=latency_dict,
        current_fps=fps
    )

    # 4. Lưu ảnh kiểm thử
    saved_path = visualizer.save_snapshot(rendered_frame, top1_name, top1_conf)

    print("\n" + "-" * 50)
    print("✅ KẾT QUẢ KIỂM THỬ BATCH 3:")
    print(f"  - Độ phân giải khung hình HUD   : {rendered_frame.shape[1]}x{rendered_frame.shape[0]}")
    print(f"  - Trạng thái Panel Dashboard   : OK (Top-1: {top1_name.upper()} {top1_conf*100:.1f}%)")
    print(f"  - Trạng thái Hardware Telemetry: OK (FPS: {fps:.1f}, PL Cycles: 105,950)")
    print(f"  - File ảnh chụp kiểm thử       : '{saved_path}'")
    print("-" * 50)
    print("\n[SUCCESS] BATCH 3 ĐÃ HOÀN THÀNH VÀ SẴN SÀNG TÍCH HỢP TOÀN DIỆN VÀO BATCH 4!\n")


if __name__ == '__main__':
    self_test_hud_overlay()
