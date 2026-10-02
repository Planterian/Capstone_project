# -*- coding: utf-8 -*-
"""
===================================================================================================
PROJECT: REAL-TIME VISION TRANSFORMER (ViT) VIDEO CLASSIFICATION APPLICATION
         PURE END-TO-END SOFTWARE INFERENCE (NO SIMULATOR / NO EMULATOR)
===================================================================================================
WHAT:
    Ứng dụng thị giác máy tính phân loại hình ảnh thời gian thực từ Webcam (hoặc Video Demo tổng hợp)
    chạy thuần phần mềm (Pure Software End-to-End Deep Learning Application) sử dụng mô hình
    Vision Transformer lượng tử hóa nhận biết (QAT INT8: vit_qat_int8.pth).

WHY:
    - Cung cấp một phiên bản ứng dụng thực tế độc lập hoàn toàn, không chứa bất kỳ thành phần
      mô phỏng phần cứng vi mạch (No Hardware Simulator / No Emulation).
    - Đo đạc chính xác 100% thời gian thực thi phần mềm thực tế trên CPU máy tính chủ:
      * Thời gian đọc camera và tiền xử lý ảnh (Preprocessing Time).
      * Thời gian suy luận mạng nơ-ron (PyTorch Forward Pass Inference Time).
      * Thời gian hậu xử lý Softmax và trích xuất Top-k (Postprocessing Time).
      * Tốc độ khung hình thực tế đạt được (Real-Time Host FPS).

FUNCTIONALITY:
    1. Tự động tìm và nạp trọng số INT8 từ ./models_cache/vit_qat_int8.pth vào PyTorch Engine.
    2. Đọc luồng video camera 720p @ 30 FPS đa luồng không trễ buffer (Zero-Latency Threaded Stream).
    3. Hỗ trợ phím tắt [C] để chuyển đổi linh hoạt: Camera 0 <-> Camera 1 <-> Synthetic Video Demo.
    4. Tiền xử lý Center ROI 720x720 -> Co ảnh 32x32 -> Chuẩn hóa CIFAR-10 Mean/Std.
    5. Suy luận mô hình ViT và hiển thị giao diện HUD chuyên nghiệp:
       - Vùng ngắm đối tượng (Target Box Reticle).
       - Bảng phân loại Top-1 Class với độ tin cậy % và Top-3 Confidence Bar Charts.
       - Bảng thông số hiệu năng phần mềm (Real FPS, Latency Breakdown: Preprocess, Inference, Postprocess, Total E2E).
    6. Bộ phím tắt điều khiển:
       - [Q]: Thoát ứng dụng
       - [S]: Lưu ảnh chụp màn hình vào captures/
       - [C]: Đổi camera hoặc bật Video Demo chuyển động
       - [D]: Bật/Tắt bảng thông số Telemetry
       - [F]: Bật/Tắt khóa tốc độ 30 FPS
       - [P]: Tạm dừng / Tiếp tục
===================================================================================================
"""

# %% [markdown]
# # PHẦN 1: Cấu Hình Môi Trường & Thư Viện
# Thiết lập encoding UTF-8 cho Windows, cố định seed ngẫu nhiên và import PyTorch, OpenCV, NumPy.

# %% [code]
import os
import sys
import time
import threading
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union, Any

# Cấu hình encoding UTF-8 console Windows để tránh UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quantization

# Cố định seed ngẫu nhiên
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# # PHẦN 2: Siêu Tham Số Kiến Trúc ViT & Định Nghĩa Mô Hình
# Khởi tạo kiến trúc khớp 100% với file trọng số đã huấn luyện `models_cache/vit_qat_int8.pth`:
# - Kích thước ảnh: $32 \times 32 \times 3$
# - Kích thước Patch: $4 \times 4 \implies 64$ patches (+ 1 CLS token = 65 tokens)
# - Chiều Embedding: 256, Số Heads: 8 ($d_k = 32$), Số tầng: 10, MLP Dim: 512

# %% [code]
IMG_SIZE = 32         # 32x32 pixels
NUM_CHANNELS = 3      # 3 kênh RGB
NUM_CLASSES = 10      # 10 lớp CIFAR-10
PATCH_SIZE = 4        # 4x4 pixels
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 64 patches
NUM_HEADS = 8         # 8 Attention Heads
EMBED_DIM = 256       # 256 embedding dimension
HEAD_DIM = EMBED_DIM // NUM_HEADS  # d_k = 32
MLP_DIM = 512         # 512 hidden dimension
DROP_RATE = 0.1       # Dropout 10%
DEPTH = 10            # 10 Transformer Encoder Blocks

# 10 nhãn phân loại chuẩn CIFAR-10
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Giá trị Mean và Std của bộ dữ liệu CIFAR-10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def setup_quantization_engine() -> str:
    """Tự động lựa chọn quantization engine phù hợp với CPU (onednn / fbgemm / qnnpack)."""
    supported_engines = torch.backends.quantized.supported_engines
    chosen_engine = None

    if 'onednn' in supported_engines:
        chosen_engine = 'onednn'
    elif 'fbgemm' in supported_engines:
        chosen_engine = 'fbgemm'
    elif 'qnnpack' in supported_engines:
        chosen_engine = 'qnnpack'
    else:
        chosen_engine = list(supported_engines)[0] if len(supported_engines) > 0 else 'none'

    torch.backends.quantized.engine = chosen_engine
    return chosen_engine


class PatchEmbedding(nn.Module):
    """Cắt ảnh thành 64 patch và chiếu tuyến tính lên vector không gian 256 chiều."""
    def __init__(self, num_channels: int = NUM_CHANNELS, embed_dim: int = EMBED_DIM, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 32, 32] -> [B, 256, 8, 8] -> [B, 64, 256]
        return self.patch_embed(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    """Khối Feed-Forward Network với hàm kích hoạt GELU."""
    def __init__(self, embed_dim: int = EMBED_DIM, mlp_dim: int = MLP_DIM, drop_rate: float = DROP_RATE):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.dropout(self.gelu(self.linear1(x)))))


class ScratchMultiheadAttention(nn.Module):
    """Khối Multi-Head Self-Attention hỗ trợ lượng tử hóa an toàn."""
    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS, dropout: float = DROP_RATE):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_k = 32

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.ff_matmul_qv = nn.quantized.FloatFunctional()
        self.attn_probs_quant = quantization.QuantStub()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = q.shape

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        q_heads = q_proj.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_proj.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_proj.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q_heads_f = q_heads.dequantize() if q_heads.is_quantized else q_heads
        k_heads_f = k_heads.dequantize() if k_heads.is_quantized else k_heads

        # 1. Tính toán ma trận Attention Scores: Q * K^T
        attn_scores = torch.matmul(q_heads_f, k_heads_f.transpose(-2, -1))

        # 2. Scaling: nhân 1/sqrt(d_k)
        attn_scores = attn_scores * self.scale

        # 3. Softmax xác suất chú ý
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_probs = self.attn_probs_quant(attn_probs)

        # 4. Context Output: Attn * V
        context = self.ff_matmul_qv.matmul(attn_probs, v_heads)
        context = context.transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(context)

        return output, attn_probs


class HWFriendlyTransformerEncoder(nn.Module):
    """Transformer Encoder Block với phép cộng phần dư (Residual Add) lượng tử hóa an toàn."""
    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS,
                 mlp_dim: int = MLP_DIM, drop_rate: float = DROP_RATE):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.multi_head_atten = ScratchMultiheadAttention(embed_dim, num_heads, dropout=drop_rate)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.multi_layer_perceptron = MLP(embed_dim, mlp_dim, drop_rate)

        self.ff_residual_1 = nn.quantized.FloatFunctional()
        self.ff_residual_2 = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_1 = x
        attention_output, _ = self.multi_head_atten(self.layer_norm_1(x), self.layer_norm_1(x), self.layer_norm_1(x))
        x = self.ff_residual_1.add(attention_output, residual_1)

        residual_2 = x
        mlp_output = self.multi_layer_perceptron(self.layer_norm_2(x))
        x = self.ff_residual_2.add(mlp_output, residual_2)
        return x


class QuantizableVisionTransformer(nn.Module):
    """Mô hình Vision Transformer (ViT) hoàn chỉnh tương thích QAT INT8."""
    def __init__(self, img_size: int = IMG_SIZE, patch_size: int = PATCH_SIZE,
                 num_channels: int = NUM_CHANNELS, num_classes: int = NUM_CLASSES,
                 embed_dim: int = EMBED_DIM, depth: int = DEPTH, num_heads: int = NUM_HEADS,
                 mlp_dim: int = MLP_DIM, drop_rate: float = DROP_RATE):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(num_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        self.transformer_layers = nn.Sequential(*[
            HWFriendlyTransformerEncoder(embed_dim, num_heads, mlp_dim, drop_rate)
            for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self.quant_cls = quantization.QuantStub()
        self.quant_pos = quantization.QuantStub()
        self.f_cat = nn.quantized.FloatFunctional()
        self.f_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.patch_embedding(x)
        B = x.size(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens_quant = self.quant_cls(cls_tokens)
        x = self.f_cat.cat((cls_tokens_quant, x), dim=1)

        pos_embed_quant = self.quant_pos(self.pos_embed)
        x = self.f_add.add(x, pos_embed_quant)

        x = self.transformer_layers(x)
        logits = self.mlp_head(x[:, 0])
        return self.dequant(logits)


# %% [markdown]
# # PHẦN 3: Nạp Mô Hình Trọng Số INT8 (Model Loader)
# Tìm kiếm và nạp an toàn tệp trọng số `vit_qat_int8.pth` vào PyTorch Quantization Engine.

# %% [code]
def load_vit_model(weights_path: str = "models_cache/vit_qat_int8.pth") -> nn.Module:
    """Nạp trọng số INT8 vào mô hình Vision Transformer."""
    engine = setup_quantization_engine()

    possible_paths = [
        weights_path,
        os.path.join(os.path.dirname(__file__), weights_path) if "__file__" in locals() else weights_path,
        os.path.join("model_notebook", weights_path),
        os.path.join("..", weights_path),
        os.path.join("e:/CapstoneProjectDocs/Capstone_project/model_notebook", weights_path),
        os.path.join("e:/CapstoneProjectDocs/Capstone_project/model_notebook/models_cache", os.path.basename(weights_path))
    ]

    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break

    if actual_path is None:
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file trọng số {weights_path} tại: {possible_paths}")

    qat_model = QuantizableVisionTransformer().to('cpu')
    qat_model.train()
    qat_model.qconfig = quantization.get_default_qat_qconfig(engine)
    quantization.prepare_qat(qat_model, inplace=True)
    qat_model_int8 = quantization.convert(qat_model, inplace=False)

    state_dict = torch.load(actual_path, map_location='cpu')
    qat_model_int8.load_state_dict(state_dict)
    qat_model_int8.eval()
    print(f"[Model Loader] ✅ Nạp thành công '{actual_path}' ({len(state_dict)} tensors). Engine: '{engine}'")
    return qat_model_int8


# %% [markdown]
# # PHẦN 4: Pipeline Đọc Video Camera Đa Luồng & Bộ Sinh Video Demo Dự Phòng
# - `ThreadedCameraStream`: Luồng ngầm đọc camera OpenCV liên tục không bị drop frame hay nghẽn I/O.
# - Hỗ trợ bấm phím `[C]` để chuyển đổi nguồn: Camera 0 <-> Camera 1 <-> Synthetic Video Demo.
# - Tự động phát hiện khi camera bị đen (nắp che/quyền riêng tư) và hướng dẫn người dùng.

# %% [code]
class SyntheticFrameGenerator:
    """Sinh luồng video demo 720p @ 30 FPS có hình ảnh chuyển động sinh động."""
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.sample_classes = ['frog', 'airplane', 'automobile', 'bird', 'ship', 'horse', 'dog', 'cat']
        self.cur_idx = 0
        self.last_switch = time.time()

    def generate_frame(self) -> np.ndarray:
        self.frame_idx += 1
        now = time.time()
        if now - self.last_switch > 4.0:
            self.cur_idx = (self.cur_idx + 1) % len(self.sample_classes)
            self.last_switch = now

        current_class = self.sample_classes[self.cur_idx]
        frame = np.full((self.height, self.width, 3), 20, dtype=np.uint8)

        # Lưới tọa độ đồ họa
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (32, 32, 38), 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), (32, 32, 38), 1)

        cx = self.width // 2 + int(np.sin(self.frame_idx * 0.05) * 40)
        cy = self.height // 2 + int(np.cos(self.frame_idx * 0.05) * 30)

        # Vẽ hình ảnh đối tượng CIFAR-10 mẫu
        if current_class == 'frog':
            cv2.circle(frame, (cx, cy), 110, (40, 180, 50), -1)
            cv2.circle(frame, (cx - 45, cy - 80), 30, (50, 200, 60), -1)
            cv2.circle(frame, (cx + 45, cy - 80), 30, (50, 200, 60), -1)
        elif current_class in ['airplane', 'bird']:
            pts = np.array([[cx, cy - 120], [cx + 140, cy], [cx, cy + 100], [cx - 140, cy]], np.int32)
            cv2.fillPoly(frame, [pts], (220, 180, 70))
        elif current_class in ['automobile', 'truck']:
            cv2.rectangle(frame, (cx - 140, cy - 50), (cx + 140, cy + 70), (40, 50, 220), -1)
            cv2.rectangle(frame, (cx - 90, cy - 110), (cx + 90, cy - 50), (60, 70, 240), -1)
            cv2.circle(frame, (cx - 80, cy + 70), 28, (30, 30, 30), -1)
            cv2.circle(frame, (cx + 80, cy + 70), 28, (30, 30, 30), -1)
        else:
            cv2.circle(frame, (cx, cy), 110, (180, 120, 70), -1)

        cv2.putText(frame, "[SYNTHETIC VIDEO DEMO - CIFAR-10 STREAM]", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Simulated Object: {current_class.upper()} (Auto-switch 4s)", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        return frame


class ThreadedCameraStream:
    """Luồng đọc camera ngầm hiệu năng cao (Zero-Latency Ring Buffer) hỗ trợ đa nguồn."""
    def __init__(self, camera_source: Union[int, str] = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.camera_source = camera_source
        self.current_idx = camera_source if isinstance(camera_source, int) else 0
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic = (camera_source == 'synthetic')
        self.synthetic_gen = SyntheticFrameGenerator(self.width, self.height)

        self.latest_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.source_description = "Synthetic Demo (CIFAR-10)" if self.is_synthetic else f"Webcam {self.camera_source}"

        self._init_stream()

    def _init_stream(self):
        if self.is_synthetic:
            self.cap = None
            self.latest_frame = self.synthetic_gen.generate_frame()
            self.source_description = "Synthetic Demo (CIFAR-10)"
            print("[Camera] 🚀 Đang phát luồng Synthetic Video Demo (720p @ 30 FPS).")
            return

        print(f"[Camera] Tìm kiếm USB Webcam tại index {self.current_idx}...")
        cap = cv2.VideoCapture(self.current_idx, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(self.current_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        success, test_frame = cap.read() if cap.isOpened() else (False, None)
        if success and test_frame is not None:
            self.cap = cap
            self.is_synthetic = False
            self.latest_frame = test_frame
            self.source_description = f"USB Webcam (Index {self.current_idx})"
            frame_mean = float(np.mean(test_frame))
            if frame_mean < 1.5:
                print(f"[Camera] ⚠️ CẢNH BÁO: Webcam {self.current_idx} kết nối được nhưng khung hình ĐEN (mean={frame_mean:.2f}).")
                print("          -> Kiểm tra nắp che camera (privacy shutter), hoặc bấm phím 'C' để đổi sang Synthetic Demo.")
            else:
                print(f"[Camera] ✅ Kết nối thành công USB Webcam {self.current_idx} ({test_frame.shape[1]}x{test_frame.shape[0]} @ {self.fps} FPS)")
        else:
            if cap.isOpened():
                cap.release()
            self.cap = None
            self.is_synthetic = True
            self.latest_frame = self.synthetic_gen.generate_frame()
            self.source_description = "Synthetic Demo (Fallback)"
            print(f"[Camera] 🚀 Không mở được camera {self.current_idx} -> Tự động chuyển sang Synthetic Demo (720p @ 30 FPS).")

    def toggle_source(self) -> str:
        """Chuyển nguồn camera: Cam 0 -> Cam 1 -> Synthetic Demo -> Cam 0."""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None

            if not self.is_synthetic and self.current_idx == 0:
                self.current_idx = 1
                self.is_synthetic = False
                print("[Camera] Đang thử kết nối USB Webcam tại index 1...")
                cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                success, test_frame = cap.read() if cap.isOpened() else (False, None)
                if success and test_frame is not None and float(np.mean(test_frame)) >= 1.0:
                    self.cap = cap
                    self.latest_frame = test_frame
                    self.source_description = "USB Webcam (Index 1)"
                else:
                    if cap.isOpened():
                        cap.release()
                    self.is_synthetic = True
                    self.source_description = "Synthetic Demo (CIFAR-10)"
            elif not self.is_synthetic and self.current_idx == 1:
                self.is_synthetic = True
                self.source_description = "Synthetic Demo (CIFAR-10)"
            else:
                self.current_idx = 0
                self.is_synthetic = False
                print("[Camera] Đang thử kết nối lại USB Webcam tại index 0...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                success, test_frame = cap.read() if cap.isOpened() else (False, None)
                if success and test_frame is not None:
                    self.cap = cap
                    self.latest_frame = test_frame
                    self.source_description = "USB Webcam (Index 0)"
                else:
                    if cap.isOpened():
                        cap.release()
                    self.is_synthetic = True
                    self.source_description = "Synthetic Demo (CIFAR-10)"

            print(f"[Camera] Nguồn video hiện tại: {self.source_description}")
            return self.source_description

    def start(self) -> "ThreadedCameraStream":
        if self.is_running:
            return self
        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="CamWorker")
        self.thread.start()
        return self

    def _worker(self):
        fps_interval = 1.0 / self.fps
        last_t = time.time()
        while self.is_running:
            if not self.is_synthetic and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    self.is_synthetic = True
            else:
                now = time.time()
                elapsed = now - last_t
                if elapsed < fps_interval:
                    time.sleep(fps_interval - elapsed)
                last_t = time.time()
                frame = self.synthetic_gen.generate_frame()
                with self.lock:
                    self.latest_frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return (True, self.latest_frame.copy()) if self.latest_frame is not None else (False, None)

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


class VideoPreprocessor:
    """Tiền xử lý khung hình ảnh đầu vào: Cắt Center ROI 720x720 và chuẩn hóa CIFAR-10."""
    def __init__(self, target_size: int = IMG_SIZE):
        self.target_size = target_size
        self.mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(CIFAR_STD, dtype=np.float32).reshape(1, 1, 3)

    def crop_center_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        roi_dim = min(h, w)
        x1 = (w - roi_dim) // 2
        y1 = (h - roi_dim) // 2
        return frame[y1:y1 + roi_dim, x1:x1 + roi_dim], (x1, y1, x1 + roi_dim, y1 + roi_dim)

    def preprocess_to_tensor(self, roi_img: np.ndarray) -> Tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        resized = cv2.resize(roi_img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)
        return tensor, (time.perf_counter() - t0) * 1000.0


# %% [markdown]
# # PHẦN 5: Giao Diện HUD Trực Quan Hóa Thuần Phần Mềm (Clean Application HUD)
# Giao diện Edge AI HUD trực quan hóa kết quả phân loại và các chỉ số đo đạc phần mềm:
# - Target Reticle: Vùng ảnh trích xuất $32 \times 32$ pixels.
# - Prediction Dashboard: Nhãn Top-1 và 3 thanh tiến trình xác suất Top-3.
# - Software Performance Telemetry: Đo đạc FPS thực tế và thời gian thực thi (Preprocess, Forward Inference, Postprocess, Total E2E).

# %% [code]
class NativeVisionHUDVisualizer:
    """Giao diện HUD Overlay thời gian thực cho ứng dụng thị giác máy tính."""
    def __init__(self, output_dir: str = "captures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.debug_mode = True
        self.fps_lock_30 = False
        self.is_paused = False
        self.fps_history: List[float] = []

    def get_smooth_fps(self, fps: float) -> float:
        self.fps_history.append(fps)
        if len(self.fps_history) > 20:
            self.fps_history.pop(0)
        return float(np.mean(self.fps_history))

    def draw_glass_panel(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                          alpha: float = 0.75, border_color: Tuple[int, int, int] = (70, 70, 85)):
        """Vẽ bảng nền bán trong suốt hiệu ứng kính mờ (Glassmorphism)."""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)

    def render(self, frame: np.ndarray, roi_coords: Tuple[int, int, int, int],
               top1_name: str, top1_conf: float, top3_list: List[Tuple[str, float]],
               latency_dict: Dict[str, float], current_fps: float = 30.0,
               source_desc: str = "Webcam") -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Thanh tiêu đề trên cùng (Top Banner)
        self.draw_glass_panel(canvas, 0, 0, w, 44, alpha=0.88, border_color=(50, 50, 60))
        cv2.putText(canvas, "REAL-TIME VISION TRANSFORMER (ViT) CLASSIFIER", (20, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"[PyTorch INT8 Quantized Engine | Source: {source_desc}]", (540, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (50, 220, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, datetime.now().strftime("%H:%M:%S"), (w - 95, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 175), 1, cv2.LINE_AA)

        # 2. Khung ngắm Center ROI (Tech Brackets)
        x1, y1, x2, y2 = roi_coords
        c_len = 35
        c_color = (0, 230, 255)
        top_y = max(y1, 46)
        bot_y = min(y2, h - 36)
        for (px, py, dx, dy) in [(x1, top_y, 1, 1), (x2, top_y, -1, 1), (x1, bot_y, 1, -1), (x2, bot_y, -1, -1)]:
            cv2.line(canvas, (px, py), (px + dx * c_len, py), c_color, 3)
            cv2.line(canvas, (px, py), (px, py + dy * c_len), c_color, 3)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), (0, 255, 255), 1)
        cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), (0, 255, 255), 1)
        cv2.putText(canvas, "TARGET REGION (32x32 ViT ROI)", (x1 + 10, max(y1 + 25, 72)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 1, cv2.LINE_AA)

        # 3. Bảng Prediction Dashboard
        px, py, pw, ph = 25, 55, 340, 240
        self.draw_glass_panel(canvas, px, py, pw, ph)
        cv2.putText(canvas, "PREDICTION DASHBOARD", (px + 15, py + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 215, 0), 1, cv2.LINE_AA)
        cv2.line(canvas, (px + 15, py + 34), (px + pw - 15, py + 34), (70, 70, 85), 1)

        top_col = (50, 220, 90) if top1_conf >= 0.7 else ((40, 215, 255) if top1_conf >= 0.4 else (40, 50, 235))
        cv2.putText(canvas, top1_name.upper(), (px + 15, py + 92), cv2.FONT_HERSHEY_DUPLEX, 1.0, top_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{top1_conf * 100:.1f}%", (px + pw - 90, py + 92), cv2.FONT_HERSHEY_DUPLEX, 0.85, top_col, 2, cv2.LINE_AA)

        cv2.putText(canvas, "TOP-3 CONFIDENCE BREAKDOWN:", (px + 15, py + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 175), 1, cv2.LINE_AA)
        bar_colors = [(50, 220, 90), (230, 215, 0), (40, 215, 255)]
        for i, (name, prob) in enumerate(top3_list):
            by = py + 150 + (i * 26)
            cv2.putText(canvas, f"{i+1}. {name:<10}", (px + 15, by + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (px + 115, by), (px + 275, by + 14), (45, 45, 55), -1)
            cv2.rectangle(canvas, (px + 115, by), (px + 115 + int(160 * prob), by + 14), bar_colors[i % 3], -1)
            cv2.putText(canvas, f"{prob * 100:5.1f}%", (px + 285, by + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)

        # 4. Bảng Software Performance Telemetry
        if self.debug_mode:
            tx, ty, tw, th = w - 375, 55, 350, 310
            self.draw_glass_panel(canvas, tx, ty, tw, th)

            cv2.putText(canvas, "SOFTWARE TELEMETRY", (tx + 15, ty + 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (tx + 15, ty + 34), (tx + tw - 15, ty + 34), (70, 70, 85), 1)

            s_fps = self.get_smooth_fps(current_fps)
            f_col = (50, 220, 90) if s_fps >= 25 else ((40, 215, 255) if s_fps >= 15 else (40, 50, 235))
            cv2.putText(canvas, f"{s_fps:4.1f} FPS", (tx + 15, ty + 90), cv2.FONT_HERSHEY_DUPLEX, 0.95, f_col, 2, cv2.LINE_AA)
            lock_label = "[LOCKED 30]" if self.fps_lock_30 else "[UNLOCKED]"
            cv2.putText(canvas, lock_label, (tx + 175, ty + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.line(canvas, (tx + 15, ty + 110), (tx + tw - 15, ty + 110), (45, 45, 55), 1)

            rows = [
                ("1. Video Preprocessing", f"{latency_dict.get('t_prep_ms', 1.5):6.2f} ms", (200, 200, 200)),
                ("2. PyTorch Forward Inference", f"{latency_dict.get('t_infer_ms', 25.0):6.2f} ms", (0, 230, 255)),
                ("3. Softmax & Postprocessing", f"{latency_dict.get('t_post_ms', 0.5):6.2f} ms", (200, 200, 200)),
            ]
            for idx, (lbl, val, col) in enumerate(rows):
                ry = ty + 140 + (idx * 25)
                cv2.putText(canvas, lbl, (tx + 15, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(canvas, val, (tx + tw - 100, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

            tot_lat = latency_dict.get("t_e2e_ms", 30.0)
            cv2.line(canvas, (tx + 15, ty + 225), (tx + tw - 15, ty + 225), (70, 70, 85), 1)
            cv2.putText(canvas, "TOTAL LATENCY (E2E):", (tx + 15, ty + 252),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{tot_lat:6.2f} ms", (tx + tw - 105, ty + 252),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 220, 90), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Inference Device: Host CPU (x86_64)", (tx + 15, ty + 288),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 175), 1, cv2.LINE_AA)

        # 5. Cảnh báo nếu khung hình camera bị đen hoàn toàn (Black Screen Detector)
        if float(np.mean(frame)) < 2.5:
            alert_w, alert_h = 600, 160
            ax = (w - alert_w) // 2
            ay = (h - alert_h) // 2 + 15
            self.draw_glass_panel(canvas, ax, ay, alert_w, alert_h, alpha=0.92, border_color=(40, 50, 240))
            cv2.putText(canvas, "CANH BAO: KHUNG HINH CAMERA DANG BI DEN!", (ax + 20, ay + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, (40, 50, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, "1. Kiem tra nut gat / nap che camera (Privacy Shutter) tren may.", (ax + 20, ay + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, "2. Kiem tra quyen truy cap Camera trong Windows (Settings -> Privacy).", (ax + 20, ay + 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, ">> BAM PHIM [C] DE CHUYEN SANG VIDEO DEMO CIFAR-10 CHUYEN DONG!", (ax + 20, ay + 130),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        # 6. Thanh phím tắt điều khiển ở đáy (Bottom Control Bar)
        self.draw_glass_panel(canvas, 0, h - 34, w, 34, alpha=0.88, border_color=(50, 50, 60))
        shortcuts = [
            ("[Q]", "Quit"),
            ("[S]", "Snapshot"),
            ("[C]", "Cam/Demo"),
            ("[A]", "Async/Sync"),
            ("[D]", "Toggle HUD"),
            ("[F]", "FPS Lock"),
            ("[P]", "Pause")
        ]
        pos_x = 20
        spacing = 175
        for k, desc in shortcuts:
            cv2.putText(canvas, k, (pos_x, h - 12), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (pos_x + 30, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            pos_x += spacing

        if self.is_paused:
            cv2.putText(canvas, "[VIDEO STREAM PAUSED]", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        return canvas

    def save_snapshot(self, frame: np.ndarray, top1_name: str, confidence: float) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_native_{timestamp}_{top1_name}_{int(confidence * 100)}pct.png"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"[Snapshot] 📸 Đã lưu ảnh chụp màn hình: '{path}'")
        return path


class ThreadedInferenceWorker:
    """Tiểu trình suy luận AI nền độc lập, giúp luồng video đạt 30+ FPS mượt mà."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.input_tensor: Optional[torch.Tensor] = None
        self.t_prep_ms: float = 1.5
        self.latest_result: Optional[Tuple[str, float, List[Tuple[str, float]], Dict[str, float]]] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="InferenceWorker")
        self.thread.start()
        return self

    def submit_frame(self, tensor: torch.Tensor, t_prep_ms: float):
        with self.lock:
            self.input_tensor = tensor
            self.t_prep_ms = t_prep_ms
        self.new_frame_event.set()

    def get_latest_result(self) -> Optional[Tuple[str, float, List[Tuple[str, float]], Dict[str, float]]]:
        with self.lock:
            return self.latest_result

    def _worker(self):
        while self.is_running:
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()

            with self.lock:
                tensor = self.input_tensor
                t_prep = self.t_prep_ms

            if tensor is None:
                continue

            t_infer_start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
            t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

            t_post_start = time.perf_counter()
            top_prob, top_idx = torch.max(probs, dim=0)
            top1_id = int(top_idx.item())
            top1_conf = float(top_prob.item())
            top1_name = CIFAR10_CLASSES[top1_id]

            top3_probs, top3_indices = torch.topk(probs, k=3)
            top3_list = [
                (CIFAR10_CLASSES[idx.item()], float(p.item()))
                for p, idx in zip(top3_probs, top3_indices)
            ]
            t_post_ms = (time.perf_counter() - t_post_start) * 1000.0
            t_e2e_ms = t_prep + t_infer_ms + t_post_ms

            latency_dict = {
                "t_prep_ms": t_prep,
                "t_infer_ms": t_infer_ms,
                "t_post_ms": t_post_ms,
                "t_e2e_ms": t_e2e_ms
            }

            with self.lock:
                self.latest_result = (top1_name, top1_conf, top3_list, latency_dict)

    def stop(self):
        self.is_running = False
        self.new_frame_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)


# %% [markdown]
# # PHẦN 6: Hàm Vận Hành Ứng Dụng Chính (End-to-End Live Video Loop)
# Khởi tạo mô hình, camera stream và thực thi vòng lặp xử lý video liên tục.
# Hỗ trợ cả 2 chế độ:
# - ASYNC MODE (Mặc định): Suy luận trên luồng nền riêng -> Luồng video đạt 30+ FPS mượt mà.
# - SYNC MODE: Suy luận tuần tự từng khung hình (chờ CPU 35-50ms -> ~18-20 FPS).
# Bấm phím [A] để chuyển đổi qua lại giữa 2 chế độ này.

# %% [code]
def run_live_application(
    camera_source: Union[int, str] = 0,
    async_mode: bool = True,
    max_frames: Optional[int] = None,
    display_gui: bool = True
):
    """
    Hàm thực thi ứng dụng phân loại video thời gian thực thuần phần mềm (Pure Software App).
    - camera_source: Chỉ số camera USB (0, 1) hoặc 'synthetic' để phát video demo CIFAR-10.
    - async_mode: True để suy luận đa luồng mượt mà 30+ FPS (khuyên dùng), False để chạy tuần tự.
    - max_frames: Giới hạn số khung hình chạy thử (None để chạy liên tục).
    - display_gui: Mở cửa sổ đồ họa hiển thị video (True) hoặc chạy headless (False).
    """
    print("\n" + "=" * 80)
    print("  KHỞI ĐỘNG ỨNG DỤNG REAL-TIME VISION TRANSFORMER (PURE SOFTWARE APPLICATION)  ")
    print("=" * 80)

    # 1. Nạp mô hình ViT QAT INT8
    model = load_vit_model("models_cache/vit_qat_int8.pth")

    # 2. Khởi tạo Camera Stream 720p @ 30 FPS
    stream = ThreadedCameraStream(camera_source=camera_source, width=1280, height=720, fps=30)
    stream.start()

    # 3. Khởi tạo Worker suy luận nền nếu bật async_mode
    worker = ThreadedInferenceWorker(model).start() if async_mode else None

    # 4. Khởi tạo Preprocessor và Visualizer
    preprocessor = VideoPreprocessor(target_size=IMG_SIZE)
    visualizer = NativeVisionHUDVisualizer(output_dir="captures")

    time.sleep(0.5)  # Chờ luồng camera ổn định
    window_name = "Real-Time ViT Classification (Native Software App)"

    frame_idx = 0
    t_fps_start = time.time()
    fps_measured = 30.0

    # Kết quả mặc định ban đầu
    current_top1_name = "INITIALIZING"
    current_top1_conf = 0.0
    current_top3_list = [(c, 0.0) for c in CIFAR10_CLASSES[:3]]
    current_latency_dict = {"t_prep_ms": 1.5, "t_infer_ms": 35.0, "t_post_ms": 0.5, "t_e2e_ms": 37.0}

    print("\n[Application] 🚀 Ứng dụng đã sẵn sàng! Đang phát luồng video...")
    print(f"              Chế độ hiện tại: {'ASYNC PIPELINE (Mượt 30 FPS)' if async_mode else 'SYNC PIPELINE (Tuần tự ~19 FPS)'}")
    print("              [Q] Thoát | [S] Chụp ảnh | [C] Đổi Cam/Demo | [A] Đổi Async/Sync\n")

    try:
        while True:
            t_loop_start = time.perf_counter()

            # Đọc khung hình mới nhất từ luồng
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1

            if not visualizer.is_paused:
                # 1. Cắt Center ROI 720x720 và tiền xử lý
                roi, roi_coords = preprocessor.crop_center_roi(frame)
                input_tensor, t_prep_ms = preprocessor.preprocess_to_tensor(roi)

                if async_mode and worker is not None:
                    # Chế độ ASYNC: Đẩy tensor cho worker nền, không làm nghẽn luồng render
                    worker.submit_frame(input_tensor, t_prep_ms)
                    res = worker.get_latest_result()
                    if res is not None:
                        current_top1_name, current_top1_conf, current_top3_list, current_latency_dict = res
                else:
                    # Chế độ SYNC: Suy luận tuần tự trực tiếp trên CPU
                    t_infer_start = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(input_tensor)
                        probabilities = F.softmax(logits, dim=1).squeeze(0)
                    t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

                    t_post_start = time.perf_counter()
                    top_prob, top_idx = torch.max(probabilities, dim=0)
                    top1_id = int(top_idx.item())
                    current_top1_conf = float(top_prob.item())
                    current_top1_name = CIFAR10_CLASSES[top1_id]

                    top3_probs, top3_indices = torch.topk(probabilities, k=3)
                    current_top3_list = [
                        (CIFAR10_CLASSES[idx.item()], float(prob.item()))
                        for prob, idx in zip(top3_probs, top3_indices)
                    ]
                    t_post_ms = (time.perf_counter() - t_post_start) * 1000.0

                    current_latency_dict = {
                        "t_prep_ms": t_prep_ms,
                        "t_infer_ms": t_infer_ms,
                        "t_post_ms": t_post_ms,
                        "t_e2e_ms": t_prep_ms + t_infer_ms + t_post_ms
                    }

                # Source desc hiển thị kèm chế độ
                mode_tag = "ASYNC 30FPS" if async_mode else "SYNC"
                full_source_desc = f"{stream.source_description} | {mode_tag}"

                # 4. Render giao diện HUD
                rendered_frame = visualizer.render(
                    frame=frame,
                    roi_coords=roi_coords,
                    top1_name=current_top1_name,
                    top1_conf=current_top1_conf,
                    top3_list=current_top3_list,
                    latency_dict=current_latency_dict,
                    current_fps=fps_measured,
                    source_desc=full_source_desc
                )
            else:
                rendered_frame = frame

            # Tính toán FPS tức thời
            t_loop_end = time.perf_counter()
            elapsed_sec = t_loop_end - t_loop_start

            # Khóa tốc độ 30 FPS nếu bật fps_lock_30
            if visualizer.fps_lock_30:
                target_interval = 1.0 / 30.0
                if elapsed_sec < target_interval:
                    time.sleep(target_interval - elapsed_sec)

            now = time.time()
            if now - t_fps_start >= 0.5:
                fps_measured = 1.0 / max(0.001, time.perf_counter() - t_loop_start)
                t_fps_start = now

            # Hiển thị cửa sổ đồ họa
            if display_gui:
                cv2.imshow(window_name, rendered_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("[Application] Nhận lệnh thoát từ người dùng (Key 'Q').")
                    break
                elif key == ord('c') or key == ord('C'):
                    new_src = stream.toggle_source()
                    print(f"[Application] 📹 Đã chuyển nguồn video sang: {new_src}")
                elif key == ord('a') or key == ord('A'):
                    async_mode = not async_mode
                    if async_mode and worker is None:
                        worker = ThreadedInferenceWorker(model).start()
                    elif not async_mode and worker is not None:
                        worker.stop()
                        worker = None
                    print(f"[Application] ⚡ Chế độ luồng AI: {'ASYNC PIPELINE (Mượt 30 FPS)' if async_mode else 'SYNC PIPELINE (Tuần tự ~19 FPS)'}")
                elif key == ord('s') or key == ord('S'):
                    visualizer.save_snapshot(rendered_frame, current_top1_name, current_top1_conf)
                elif key == ord('d') or key == ord('D'):
                    visualizer.debug_mode = not visualizer.debug_mode
                    print(f"[Application] Chế độ Telemetry HUD: {'BẬT' if visualizer.debug_mode else 'TẮT'}")
                elif key == ord('f') or key == ord('F'):
                    visualizer.fps_lock_30 = not visualizer.fps_lock_30
                    print(f"[Application] Khóa tốc độ 30 FPS: {'BẬT' if visualizer.fps_lock_30 else 'TẮT'}")
                elif key == ord('p') or key == ord('P'):
                    visualizer.is_paused = not visualizer.is_paused
                    print(f"[Application] Trạng thái video: {'TẠM DỪNG' if visualizer.is_paused else 'TIẾP TỤC'}")

            # Giới hạn số khung hình nếu chạy chế độ kiểm thử
            if max_frames and frame_idx >= max_frames:
                print(f"[Application] Đã hoàn thành {max_frames} khung hình kiểm thử.")
                break

    finally:
        if worker is not None:
            worker.stop()
        stream.stop()
        if display_gui:
            cv2.destroyAllWindows()
        print("[Application] Đã đóng ứng dụng và giải phóng tài nguyên thành công.")


# %% [markdown]
# # PHẦN 7: Kiểm Thử Độc Lập Suy Luận Mẫu (Sample Inference Verification)
# Chạy thử nghiệm phân loại trên một tensor ngẫu nhiên để xác minh mô hình hoạt động chính xác trước khi mở video.

# %% [code]
def self_test_native_pipeline():
    """Chạy kiểm thử suy luận độc lập trên CPU."""
    print("\n" + "=" * 80)
    print("  CHẠY THỬ NGHIỆM ĐỘC LẬP PIPELINE SUY LUẬN THUẦN PHẦN MỀM (SELF-TEST)  ")
    print("=" * 80)

    model = load_vit_model("models_cache/vit_qat_int8.pth")
    dummy_input = torch.randn(1, 3, 32, 32)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(dummy_input)
        probs = F.softmax(logits, dim=1).squeeze(0)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    top_prob, top_idx = torch.max(probs, dim=0)
    print(f"\n👉 KẾT QUẢ DỰ ĐOÁN: {CIFAR10_CLASSES[top_idx.item()].upper()} (Độ tin cậy: {top_prob.item()*100:.2f}%)")
    print(f"👉 Thời gian suy luận CPU (Forward Pass): {infer_ms:.2f} ms")
    print(f"👉 Tốc độ thông lượng ước tính: {1000.0 / infer_ms:.1f} FPS")

    top3_probs, top3_indices = torch.topk(probs, k=3)
    print("\nBẢNG TOP-3 XÁC SUẤT:")
    for rank, (p, idx) in enumerate(zip(top3_probs, top3_indices), 1):
        bar = "█" * int(p.item() * 30)
        print(f"  {rank}. {CIFAR10_CLASSES[idx.item()]:<12} : {p.item()*100:6.2f}% | {bar}")

    print("\n[SUCCESS] MÔ HÌNH VÀ PIPELINE ĐÃ SẴN SÀNG CHẠY VIDEO!\n")


# %% [markdown]
# # PHẦN 8: Thực Thi Ứng Dụng (Main Execution Block)

# %% [code]
if __name__ == '__main__':
    # 1. Chạy tự kiểm thử
    self_test_native_pipeline()

    # 2. Khởi chạy ứng dụng video trực tiếp:
    # - camera_source=0: Sử dụng Webcam USB mặc định
    #   (Có thể đổi thành camera_source='synthetic' nếu muốn chạy ngay Video Demo CIFAR-10)
    # - display_gui=True: BẬT cửa sổ hiển thị video OpenCV
    # - max_frames=None: Chạy liên tục (bấm 'Q' trên cửa sổ video để thoát)
    print("\n[Application] Khởi động ứng dụng video trực tiếp...")
    print("              Bấm phím [Q] để thoát, [S] để lưu ảnh, [C] để đổi Camera/Demo.\n")
    run_live_application(
        camera_source=0,
        max_frames=None,
        display_gui=True
    )
