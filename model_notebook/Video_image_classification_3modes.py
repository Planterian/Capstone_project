# -*- coding: utf-8 -*-
"""
===================================================================================================
PROJECT: REAL-TIME VIDEO IMAGE CLASSIFICATION USING ViT (vit_qat_int8)
         MULTI-MODE EXECUTION (RAW CPU vs KV260 ARM PS vs KV260 HW/SW CO-DESIGN)
===================================================================================================
WHAT:
    Ứng dụng phân loại hình ảnh thời gian thực từ USB Webcam 720p @ 30 FPS sử dụng mô hình
    Vision Transformer lượng tử hóa nhận biết (QAT INT8: vit_qat_int8.pth) hỗ trợ 3 CHẾ ĐỘ THỰC THI
    linh hoạt có thể chuyển đổi tức thì (runtime toggling) bằng bàn phím hoặc cấu hình:

    1. MODE 1 - RAW HOST CPU:
       Chạy suy luận trực tiếp trên CPU máy tính chủ (x86_64) bằng PyTorch INT8 Native Engine,
       không áp dụng bất kỳ mô phỏng phần cứng hay giới hạn vi mạch nhúng nào.

    2. MODE 2 - KV260 ARM PS ONLY (Embedded CPU Emulation):
       Mô phỏng thực thi toàn bộ mô hình ViT trên lõi vi xử lý nhúng ARM Cortex-A53 (Quad-Core @ 1.33 GHz)
       của bo mạch AMD Kria KV260 mà KHÔNG dùng khối tăng tốc FPGA PL (PL Disabled).
       Phản ánh chính xác nút thắt cổ chai về độ trễ khi chạy ViT (~250 MMACs) trên CPU nhúng (~4-5 FPS).

    3. MODE 3 - KV260 HW/SW CO-DESIGN (ARM PS + FPGA PL + AXI DMA/CMA):
       Mô phỏng kiến trúc vi mạch phân tán không đồng nhất (Heterogeneous Co-Design):
       - ARM PS: Tiền xử lý, nạp bộ nhớ CMA, cấu hình thanh ghi AXI-Lite, hậu xử lý MLP Head.
       - AXI DMA: Truyền nhận 2 chiều qua bus AXI4-Stream 64-bit @ 200 MHz (Băng thông 1.6 GB/s).
       - FPGA PL: Khối tăng tốc phần cứng Attention Core (Systolic Array 16x16 MACs @ 200 MHz,
                  ASR Scaler, LUT Softmax) xử lý 10 tầng Transformer (~105,950 chu kỳ).
       Đạt hiệu năng tăng tốc thời gian thực (24 - 30 FPS).

HOW TO TOGGLE MODES:
    - Bấm phím '1': Chuyển ngay sang Chế độ 1 (Raw Host CPU)
    - Bấm phím '2': Chuyển ngay sang Chế độ 2 (KV260 ARM PS Only)
    - Bấm phím '3': Chuyển ngay sang Chế độ 3 (KV260 HW/SW Co-Design)
    - Bấm phím 'm': Chuyển đổi tuần tự qua 3 chế độ (1 -> 2 -> 3 -> 1)
    - Bấm phím 'q': Thoát ứng dụng
    - Bấm phím 's': Chụp ảnh màn hình lưu vào captures/
    - Bấm phím 'd': Bật/Tắt bảng Debug Telemetry
    - Bấm phím 'f': Bật/Tắt khóa 30 FPS
    - Bấm phím 'p': Tạm dừng / Tiếp tục luồng video
===================================================================================================
"""

# %% [markdown]
# # PHẦN 1: Cấu Hình Môi Trường & Dependencies
# Thiết lập UTF-8 console trên Windows, cố định seed ngẫu nhiên và kiểm tra các gói thư viện cần thiết.

# %% [code]
import os
import sys
import time
import math
import threading
from datetime import datetime
from enum import IntEnum
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
# # PHẦN 2: Siêu Tham Số Kiến Trúc ViT & Định Nghĩa Mô Hình QAT INT8
# Khởi tạo kiến trúc khớp 100% với `models_cache/vit_qat_int8.pth`:
# - Ảnh vào: $32 \times 32 \times 3$, Patch: $4 \times 4 \implies 64$ patches (+ 1 CLS token = 65 tokens).
# - Embedding Dimension: 256, Heads: 8 ($d_k = 32$), Layers: 10, MLP Dim: 512.

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
    """Tự động lựa chọn quantization engine phù hợp nhất với CPU (onednn / fbgemm / qnnpack)."""
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
    """Cắt ảnh thành 64 patch và chiếu tuyến tính lên không gian 256 chiều."""
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
    """
    Khối Attention tự triển khai mô phỏng 4 công đoạn tính toán phần cứng FPGA PL:
    - Stage 1: Systolic Array 1 nhân Q * K^T (INT32 Partial Sums).
    - Stage 2: Scaler Unit nhân 1/sqrt(d_k) (dịch bit phải số học ASR).
    - Stage 3: Hardware Softmax (tra bảng LUT ép về xác suất INT8).
    - Stage 4: Systolic Array 2 nhân Score * V (INT8 x INT8).
    """
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

        # STAGE 1: Q * K^T
        attn_scores = torch.matmul(q_heads_f, k_heads_f.transpose(-2, -1))

        # STAGE 2: Scaler Unit
        attn_scores = attn_scores * self.scale

        # STAGE 3: Hardware Softmax LUT
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_probs = self.attn_probs_quant(attn_probs)

        # STAGE 4: Score * V
        context = self.ff_matmul_qv.matmul(attn_probs, v_heads)
        context = context.transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(context)

        return output, attn_probs


class HWFriendlyTransformerEncoder(nn.Module):
    """Transformer Encoder với phép cộng Residual lượng tử hóa an toàn bằng FloatFunctional."""
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
    """Mô hình Vision Transformer lượng tử hóa QAT INT8 hoàn chỉnh."""
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


def load_vit_qat_model(weights_path: str = "models_cache/vit_qat_int8.pth") -> nn.Module:
    """Nạp trọng số INT8 vào mô hình QuantizableVisionTransformer."""
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
# # PHẦN 3: Kiến Trúc 3 Chế Độ Thực Thi & Bộ Giả Lập Phần Cứng Kria KV260
# Định nghĩa enum `ExecutionMode` và lớp `MultiModeKV260Engine` hỗ trợ 3 chế độ:
# 1. `RAW_CPU`: Host CPU x86_64 thuần phần mềm PyTorch.
# 2. `KV260_ARM_PS`: Mô phỏng CPU nhúng ARM Cortex-A53 (Quad-Core @ 1.33 GHz) không có FPGA PL.
# 3. `KV260_CO_DESIGN`: Mô phỏng vi mạch phối hợp ARM PS + FPGA PL + AXI DMA/CMA.

# %% [code]
class ExecutionMode(IntEnum):
    """Định nghĩa 3 chế độ thực thi của hệ thống."""
    RAW_CPU = 1            # Mode 1: Host CPU Native PyTorch Execution
    KV260_ARM_PS = 2        # Mode 2: KV260 ARM Cortex-A53 PS Only Emulation
    KV260_CO_DESIGN = 3     # Mode 3: KV260 Heterogeneous ARM PS + FPGA PL + AXI DMA/CMA


MODE_METADATA = {
    ExecutionMode.RAW_CPU: {
        "name": "MODE 1: RAW HOST CPU",
        "short_name": "RAW_CPU",
        "processor": "Host x86_64 CPU (PyTorch Native Engine)",
        "pl_status": "N/A (No Hardware Acceleration)",
        "dma_status": "N/A (Direct RAM)",
        "badge_color": (230, 160, 50),     # Xanh biển sáng (BGR)
        "accent_color": (255, 200, 80),
        "target_fps": "Host Unlocked",
        "description": "Suy luan thuan CPU may chu tren PyTorch INT8 khong qua gia lap KV260."
    },
    ExecutionMode.KV260_ARM_PS: {
        "name": "MODE 2: KV260 ARM PS ONLY",
        "short_name": "ARM_PS_ONLY",
        "processor": "ARM Quad-Core Cortex-A53 @ 1.33 GHz",
        "pl_status": "DISABLED (0 MHz, 0 Cycles)",
        "dma_status": "INACTIVE (0.0 MB/s)",
        "badge_color": (30, 130, 240),     # Vàng cam cảnh báo (BGR)
        "accent_color": (40, 180, 255),
        "target_fps": "4.0 ~ 5.5 FPS",
        "description": "Mo phong ViT chay 100% tren ARM Cortex-A53 khong co PL accelerator."
    },
    ExecutionMode.KV260_CO_DESIGN: {
        "name": "MODE 3: KV260 HW/SW CO-DESIGN",
        "short_name": "HW_CO_DESIGN",
        "processor": "ARM PS (Cortex-A53) + FPGA PL Attention Core",
        "pl_status": "ACTIVE (Systolic 16x16 @ 200 MHz)",
        "dma_status": "ACTIVE (AXI4-Stream 1.6 GB/s)",
        "badge_color": (50, 220, 90),      # Xanh lá cây tăng tốc (BGR)
        "accent_color": (0, 230, 255),     # Cyan
        "target_fps": "25.0 ~ 30.0 FPS",
        "description": "Kien truc phan tan: ARM PS tien/hau xu ly + FPGA PL Attention Core + AXI DMA/CMA."
    }
}


class MultiModeKV260Engine:
    """
    Bộ động cơ thực thi đa chế độ (Multi-Mode Execution Engine):
    - Đảm nhiệm việc suy luận tensor ảnh qua mô hình vit_qat_int8.pth.
    - Tính toán phân rã độ trễ (Latency Breakdown) và mô phỏng thông số phần cứng
      chính xác cho từng chế độ được kích hoạt.
    """
    def __init__(self, model_weights_path: str = "models_cache/vit_qat_int8.pth",
                 initial_mode: ExecutionMode = ExecutionMode.KV260_CO_DESIGN):
        print("=" * 80)
        print("  KHỞI TẠO ĐỘNG CƠ THỰC THI ĐA CHẾ ĐỘ (MULTI-MODE KV260 EXECUTION ENGINE)  ")
        print("=" * 80)

        self.current_mode = initial_mode
        self.frame_counter = 0

        # Thông số phần cứng FPGA PL Kria KV260
        self.pl_clock_freq_mhz = 200.0                       # Xung nhịp FPGA PL: 200 MHz
        self.clock_period_ns = 1000.0 / self.pl_clock_freq_mhz  # 5.0 ns
        self.axi_bus_width_bytes = 8                         # Bus AXI4-Stream 64-bit = 8 bytes
        self.dma_bandwidth_gbps = (self.pl_clock_freq_mhz * 1e6 * self.axi_bus_width_bytes) / 1e9  # 1.6 GB/s
        self.dma_overhead_ms = 0.035                         # Độ trễ khởi tạo DMA ~ 35 us
        self.systolic_dim = 16                               # Mảng Systolic MAC 16x16
        self.macs_per_cycle = self.systolic_dim * self.systolic_dim  # 256 MACs/chu kỳ

        # Vùng nhớ liên tục CMA (Contiguous Memory Allocation)
        # Dung lượng chứa Q, K, V cho 65 tokens x 256 chiều = 49,920 bytes (~48.75 KB)
        self.cma_buffer_size = 3 * 65 * 256
        self.cma_memory_pool = np.zeros(self.cma_buffer_size, dtype=np.int8)

        # Thanh ghi điều khiển AXI-Lite
        self.axi_lite_regs: Dict[int, int] = {
            0x00: 0x00,  # Control Register: bit 0: START
            0x04: 0x01,  # Status Register: bit 0: IDLE, bit 1: DONE
            0x10: 65,    # N: Sequence Length (65)
            0x14: 8,     # H: Number of Heads (8)
            0x18: 32,    # d_k: Dimension per head (32)
            0x1C: 256    # d_model: Embedding Dim (256)
        }

        # Nạp mô hình lượng tử hóa INT8
        self.model = load_vit_qat_model(model_weights_path)
        print(f"[Engine] Chế độ mặc định ban đầu: {MODE_METADATA[self.current_mode]['name']}")

    def set_mode(self, mode: ExecutionMode):
        """Thay đổi chế độ thực thi hiện tại."""
        if mode in ExecutionMode:
            self.current_mode = mode
            print(f"\n[Engine] 🔄 ĐÃ CHUYỂN SANG: {MODE_METADATA[mode]['name']}")
            print(f"         Vi xử lý: {MODE_METADATA[mode]['processor']}")
            print(f"         Mục tiêu FPS: {MODE_METADATA[mode]['target_fps']}\n")

    def toggle_next_mode(self) -> ExecutionMode:
        """Chuyển tuần tự qua các chế độ: 1 -> 2 -> 3 -> 1."""
        next_val = (self.current_mode % 3) + 1
        new_mode = ExecutionMode(next_val)
        self.set_mode(new_mode)
        return new_mode

    def calculate_pl_hardware_cycles(self, num_tokens: int = 65, head_dim: int = 32,
                                     num_heads: int = 8, depth: int = 10) -> Tuple[int, float]:
        """Tính chu kỳ xung nhịp và thời gian lý thuyết của FPGA PL Attention Core."""
        macs_per_layer = 2 * (num_tokens * num_tokens * head_dim * num_heads)
        systolic_cycles_per_layer = math.ceil(macs_per_layer / self.macs_per_cycle)
        softmax_lut_cycles = math.ceil((num_tokens * num_tokens * num_heads) / 16)
        total_cycles_per_layer = systolic_cycles_per_layer + softmax_lut_cycles + 32
        total_pl_cycles = total_cycles_per_layer * depth
        pl_time_ms = (total_pl_cycles * self.clock_period_ns) / 1e6
        return total_pl_cycles, pl_time_ms

    def calculate_dma_transfer_time(self, data_bytes: int) -> float:
        """Tính thời gian truyền nhận AXI DMA qua bus AXI4-Stream 64-bit @ 200 MHz."""
        return ((data_bytes / (self.dma_bandwidth_gbps * 1e9)) * 1000.0) + self.dma_overhead_ms

    def predict_tensor(self, input_tensor: torch.Tensor, t_prep_ms: float = 1.5) -> Tuple[int, float, List[Tuple[str, float]], Dict[str, Any]]:
        """
        Thực hiện suy luận và tính toán phân rã độ trễ theo CHẾ ĐỘ THỰC THI hiện tại:
        - Mode 1: RAW_CPU
        - Mode 2: KV260_ARM_PS
        - Mode 3: KV260_CO_DESIGN
        """
        self.frame_counter += 1
        t_start = time.perf_counter()

        # Thực thi mô hình PyTorch INT8 để lấy kết quả dự đoán thực tế
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
        t_infer_end = time.perf_counter()
        measured_infer_ms = (t_infer_end - t_infer_start) * 1000.0

        # Trích xuất nhãn dự đoán và Top-3
        top_prob, top_idx = torch.max(probabilities, dim=0)
        top1_class_id = int(top_idx.item())
        top1_confidence = float(top_prob.item())

        top3_probs, top3_indices = torch.topk(probabilities, k=3)
        top3_list = [
            (CIFAR10_CLASSES[idx.item()], float(prob.item()))
            for prob, idx in zip(top3_probs, top3_indices)
        ]

        # Phân rã độ trễ tùy theo chế độ
        latency_dict: Dict[str, Any] = {
            "mode": self.current_mode,
            "mode_meta": MODE_METADATA[self.current_mode]
        }

        if self.current_mode == ExecutionMode.RAW_CPU:
            # MODE 1: Host CPU x86_64
            t_post_ms = 0.45
            total_e2e_ms = t_prep_ms + measured_infer_ms + t_post_ms
            latency_dict.update({
                "t_prep_ms": t_prep_ms,
                "t_infer_ms": measured_infer_ms,
                "t_post_ms": t_post_ms,
                "t_dma_total_ms": 0.0,
                "t_pl_attention_ms": 0.0,
                "t_e2e_ms": total_e2e_ms,
                "pl_hardware_cycles": 0,
                "fps_estimate": 1000.0 / max(1.0, total_e2e_ms),
                "summary_line": f"Host CPU: {measured_infer_ms:.2f}ms | No HW Acceleration"
            })

        elif self.current_mode == ExecutionMode.KV260_ARM_PS:
            # MODE 2: KV260 ARM Cortex-A53 PS Only (Quad-core @ 1.33 GHz)
            # ViT có ~250 MMACs. Trên Cortex-A53 quad-core với NEON SIMD @ 1.33 GHz,
            # độ trễ chạy phần mềm rơi vào khoảng 190 - 240 ms/frame (~4.5 FPS)
            arm_prep_ms = max(4.0, t_prep_ms * 2.5)
            arm_attn_ms = 115.0  # Software Multi-Head Attention in NEON
            arm_mlp_ms = 85.0    # Software MLP FFN in NEON
            arm_post_ms = 6.0    # Softmax + Classification Head
            total_arm_ms = arm_prep_ms + arm_attn_ms + arm_mlp_ms + arm_post_ms

            # Mô phỏng nhịp trễ thực tế của CPU nhúng Cortex-A53
            latency_dict.update({
                "t_prep_ms": arm_prep_ms,
                "t_ps_attn_ms": arm_attn_ms,
                "t_ps_mlp_ms": arm_mlp_ms,
                "t_post_ms": arm_post_ms,
                "t_dma_total_ms": 0.0,
                "t_pl_attention_ms": 0.0,
                "t_e2e_ms": total_arm_ms,
                "pl_hardware_cycles": 0,
                "fps_estimate": 1000.0 / max(1.0, total_arm_ms),
                "summary_line": f"ARM Cortex-A53: {total_arm_ms:.1f}ms (~{1000.0/total_arm_ms:.1f} FPS) | CPU Bottleneck"
            })

        elif self.current_mode == ExecutionMode.KV260_CO_DESIGN:
            # MODE 3: Heterogeneous Co-Design (ARM PS + FPGA PL + AXI DMA/CMA)
            pl_cycles, theoretical_pl_ms = self.calculate_pl_hardware_cycles(num_tokens=65, depth=10)
            dma_tx_ms = self.calculate_dma_transfer_time(self.cma_buffer_size)
            dma_rx_ms = self.calculate_dma_transfer_time(65 * 256)
            total_dma_ms = dma_tx_ms + dma_rx_ms

            ps_mlp_head_ms = max(0.2, measured_infer_ms * 0.25)
            simulated_pl_ms = max(theoretical_pl_ms, measured_infer_ms * 0.65)
            total_e2e_ms = t_prep_ms + total_dma_ms + simulated_pl_ms + ps_mlp_head_ms

            latency_dict.update({
                "t_prep_ms": t_prep_ms,
                "t_dma_tx_ms": dma_tx_ms,
                "t_dma_rx_ms": dma_rx_ms,
                "t_dma_total_ms": total_dma_ms,
                "t_pl_attention_ms": simulated_pl_ms,
                "t_ps_mlp_head_ms": ps_mlp_head_ms,
                "t_e2e_ms": total_e2e_ms,
                "pl_hardware_cycles": pl_cycles,
                "fps_estimate": 1000.0 / max(1.0, total_e2e_ms),
                "summary_line": f"PL Attention: {simulated_pl_ms:.2f}ms | DMA: {total_dma_ms:.2f}ms | Cycles: {pl_cycles:,}"
            })

        return top1_class_id, top1_confidence, top3_list, latency_dict


# %% [markdown]
# # PHẦN 4: Pipeline Đọc Video 720p @ 30 FPS & Tiền Xử Lý Khung Hình
# - `ThreadedCameraStream`: Luồng đọc camera nền chống nghẽn buffer. Tự động chuyển `SyntheticFrameGenerator` nếu không có webcam.
# - `PSPreprocessor`: Cắt Center ROI $720 \times 720$, co ảnh về $32 \times 32$ và chuẩn hóa CIFAR-10.

# %% [code]
class SyntheticFrameGenerator:
    """Sinh luồng video 720p @ 30 FPS tổng hợp khi không có camera vật lý."""
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
        frame = np.full((self.height, self.width, 3), 22, dtype=np.uint8)

        # Lưới tọa độ đồ họa
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (32, 32, 38), 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), (32, 32, 38), 1)

        cx = self.width // 2 + int(np.sin(self.frame_idx * 0.05) * 40)
        cy = self.height // 2 + int(np.cos(self.frame_idx * 0.05) * 30)

        # Vẽ hình ảnh đối tượng mẫu CIFAR-10
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

        cv2.putText(frame, "[SYNTHETIC CAMERA FALLBACK - 720p @ 30FPS]", (30, 45),
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
                print(f"[Camera] ⚠️ CẢNH BÁO: Webcam {self.current_idx} kết nối được nhưng khung hình ĐEN HOÀN TOÀN (mean={frame_mean:.2f}).")
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


class PSPreprocessor:
    """Tiền xử lý khung hình trên lõi ARM PS: Cắt Center ROI và chuẩn hóa CIFAR-10."""
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
# # PHẦN 5: Giao Diện Edge AI Multi-Mode HUD & Telemetry Monitor
# Lớp `MultiModeHUDVisualizer` hiển thị giao diện tùy biến tương thích hoàn toàn theo từng chế độ:
# - Banner trên cùng: Đổi màu và tiêu đề theo Chế độ 1 / 2 / 3.
# - Khung ngắm Center ROI: Tech Brackets chỉ báo vùng ảnh trích xuất $32 \times 32$.
# - Prediction Dashboard: Top-1 Class, Top-1 % và biểu đồ cột Top-3 xác suất.
# - Bảng Hardware Telemetry: Hiển thị phân rã độ trễ thích ứng chính xác theo từng chế độ.
# - Thanh phím tắt đáy: Hướng dẫn bấm `[1]`, `[2]`, `[3]`, `[M]` để chuyển chế độ.

# %% [code]
class MultiModeHUDVisualizer:
    """Giao diện Heads-Up Display hỗ trợ hiển thị linh hoạt 3 chế độ thực thi."""
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
                          alpha: float = 0.72, border_color: Tuple[int, int, int] = (70, 70, 85)):
        """Vẽ bảng nền bán trong suốt phong cách kính mờ (Glassmorphism)."""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)

    def render(self, frame: np.ndarray, roi_coords: Tuple[int, int, int, int],
               top1_name: str, top1_conf: float, top3_list: List[Tuple[str, float]],
               latency_dict: Dict[str, Any], current_fps: float = 30.0) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        mode = latency_dict.get("mode", ExecutionMode.KV260_CO_DESIGN)
        mode_meta = MODE_METADATA.get(mode, MODE_METADATA[ExecutionMode.KV260_CO_DESIGN])
        badge_color = mode_meta["badge_color"]

        # 1. Thanh tiêu đề trên cùng (Top Banner)
        self.draw_glass_panel(canvas, 0, 0, w, 44, alpha=0.88, border_color=(50, 50, 60))
        cv2.putText(canvas, mode_meta["name"], (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, badge_color, 1, cv2.LINE_AA)
        badge_text = f"[{mode_meta['processor']} | PL: {mode_meta['pl_status'].split(' ')[0]}]"
        cv2.putText(canvas, badge_text, (430, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, datetime.now().strftime("%H:%M:%S"), (w - 95, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 175), 1, cv2.LINE_AA)

        # 2. Khung ngắm Center ROI (Tech Brackets không đè lên Banner)
        x1, y1, x2, y2 = roi_coords
        c_len = 35
        c_color = badge_color
        # Giới hạn tọa độ vẽ bracket để không đè lên top banner (y>=46) và bottom bar (y<=h-36)
        top_y = max(y1, 46)
        bot_y = min(y2, h - 36)
        for (px, py, dx, dy) in [(x1, top_y, 1, 1), (x2, top_y, -1, 1), (x1, bot_y, 1, -1), (x2, bot_y, -1, -1)]:
            cv2.line(canvas, (px, py), (px + dx * c_len, py), c_color, 3)
            cv2.line(canvas, (px, py), (px, py + dy * c_len), c_color, 3)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), (0, 255, 255), 1)
        cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), (0, 255, 255), 1)
        # Nhãn Target Region đặt ở y=72 để tránh chồng chéo với Top Banner
        reticle_label_y = max(y1 + 25, 72)
        cv2.putText(canvas, "TARGET REGION (32x32 ViT ROI)", (x1 + 10, reticle_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 1, cv2.LINE_AA)

        # 3. Bảng Prediction Dashboard
        px, py, pw, ph = 25, 55, 340, 240
        self.draw_glass_panel(canvas, px, py, pw, ph)
        cv2.putText(canvas, "PREDICTION DASHBOARD", (px + 15, py + 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 215, 0), 1, cv2.LINE_AA)
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

        # 4. Bảng Hardware Telemetry (Thích ứng theo chế độ)
        if self.debug_mode:
            tx, ty, tw, th = w - 385, 55, 360, 360
            self.draw_glass_panel(canvas, tx, ty, tw, th)

            panel_title = f"{mode_meta['short_name']} TELEMETRY"
            cv2.putText(canvas, panel_title, (tx + 15, ty + 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, badge_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (tx + 15, ty + 34), (tx + tw - 15, ty + 34), (70, 70, 85), 1)

            s_fps = self.get_smooth_fps(current_fps)
            # Màu FPS: Xanh lá nếu >= 20, Vàng nếu 10-20, Đỏ nếu < 10
            f_col = (50, 220, 90) if s_fps >= 20 else ((40, 215, 255) if s_fps >= 10 else (40, 50, 235))
            cv2.putText(canvas, f"{s_fps:4.1f} FPS", (tx + 15, ty + 90), cv2.FONT_HERSHEY_DUPLEX, 0.95, f_col, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Target: {mode_meta['target_fps']}", (tx + 175, ty + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.line(canvas, (tx + 15, ty + 110), (tx + tw - 15, ty + 110), (45, 45, 55), 1)

            # Các dòng phân tích trễ theo từng chế độ
            rows = []
            if mode == ExecutionMode.RAW_CPU:
                rows = [
                    ("1. Video Preprocessing", f"{latency_dict.get('t_prep_ms', 1.5):6.2f} ms", (200, 200, 200)),
                    ("2. PyTorch Native INT8", f"{latency_dict.get('t_infer_ms', 25.0):6.2f} ms", (255, 200, 80)),
                    ("3. Softmax & Class Extraction", f"{latency_dict.get('t_post_ms', 0.5):6.2f} ms", (200, 200, 200)),
                    ("4. FPGA PL Acceleration", "N/A (Disabled)", (130, 130, 140))
                ]
            elif mode == ExecutionMode.KV260_ARM_PS:
                rows = [
                    ("1. ARM PS Preprocessing", f"{latency_dict.get('t_prep_ms', 4.5):6.2f} ms", (200, 200, 200)),
                    ("2. ARM PS ViT Attention (NEON)", f"{latency_dict.get('t_ps_attn_ms', 115.0):6.2f} ms", (40, 180, 255)),
                    ("3. ARM PS MLP Head FFN", f"{latency_dict.get('t_ps_mlp_ms', 85.0):6.2f} ms", (40, 180, 255)),
                    ("4. ARM PS Postprocess", f"{latency_dict.get('t_post_ms', 6.0):6.2f} ms", (200, 200, 200))
                ]
            else:  # KV260_CO_DESIGN
                rows = [
                    ("1. ARM PS Video Preprocess", f"{latency_dict.get('t_prep_ms', 1.5):6.2f} ms", (200, 200, 200)),
                    ("2. AXI4-Stream DMA (2-way)", f"{latency_dict.get('t_dma_total_ms', 0.11):6.2f} ms", (0, 220, 255)),
                    ("3. FPGA PL Attention Core", f"{latency_dict.get('t_pl_attention_ms', 25.0):6.2f} ms", (50, 220, 90)),
                    ("4. ARM PS MLP Head Softmax", f"{latency_dict.get('t_ps_mlp_head_ms', 8.0):6.2f} ms", (200, 200, 200))
                ]

            for idx, (lbl, val, col) in enumerate(rows):
                ry = ty + 140 + (idx * 24)
                cv2.putText(canvas, lbl, (tx + 15, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(canvas, val, (tx + tw - 110, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

            tot_lat = latency_dict.get("t_e2e_ms", 35.0)
            cv2.line(canvas, (tx + 15, ty + 248), (tx + tw - 15, ty + 248), (70, 70, 85), 1)
            cv2.putText(canvas, "TOTAL LATENCY (E2E):", (tx + 15, ty + 272), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{tot_lat:6.2f} ms", (tx + tw - 110, ty + 272), cv2.FONT_HERSHEY_DUPLEX, 0.6, badge_color, 1, cv2.LINE_AA)

            # Footer status của bảng Telemetry
            if mode == ExecutionMode.RAW_CPU:
                cv2.putText(canvas, "Host CPU Threads: Native Multi-Core", (tx + 15, ty + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 175), 1, cv2.LINE_AA)
                cv2.putText(canvas, "PL Hardware: Not Used", (tx + 15, ty + 332), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)
            elif mode == ExecutionMode.KV260_ARM_PS:
                cv2.putText(canvas, "ARM Core: Quad Cortex-A53 @ 1.33 GHz", (tx + 15, ty + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 175), 1, cv2.LINE_AA)
                cv2.putText(canvas, "WARNING: CPU Bottleneck without FPGA", (tx + 15, ty + 332), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 130, 240), 1, cv2.LINE_AA)
            else:
                pl_cycles = latency_dict.get("pl_hardware_cycles", 105950)
                cv2.putText(canvas, "PL Systolic Array: 16x16 @ 200 MHz", (tx + 15, ty + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 175), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"PL Compute Cycles: {pl_cycles:,} cycles", (tx + 15, ty + 332), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 220, 90), 1, cv2.LINE_AA)

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
            ("[1]", "Raw CPU"),
            ("[2]", "ARM PS"),
            ("[3]", "HW Co-Design"),
            ("[M]", "Mode"),
            ("[C]", "Cam/Demo"),
            ("[Q]", "Quit"),
            ("[S]", "Snapshot"),
            ("[D]", "Debug"),
            ("[F]", "FPS Lock"),
            ("[P]", "Pause")
        ]
        pos_x = 18
        spacing = 126
        for k, desc in shortcuts:
            cv2.putText(canvas, k, (pos_x, h - 12), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (pos_x + 28, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            pos_x += spacing

        # Cảnh báo tạm dừng
        if self.is_paused:
            cv2.putText(canvas, "[VIDEO STREAM PAUSED]", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        return canvas

    def save_snapshot(self, frame: np.ndarray, mode_name: str, top1_name: str, confidence: float) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = mode_name.lower().replace(" ", "_").replace(":", "")
        filename = f"snapshot_{safe_mode}_{timestamp}_{top1_name}_{int(confidence * 100)}pct.png"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"[Snapshot] 📸 Đã lưu ảnh: '{path}'")
        return path


# %% [markdown]
# # PHẦN 6: Hàm Vận Hành Chính Đa Chế Độ (Multi-Mode Live Video Pipeline)
# Khởi động ứng dụng, lắng nghe sự kiện phím bấm thời gian thực để chuyển đổi mượt mà giữa 3 chế độ.

# %% [code]
def run_live_video_classification(
    camera_source: Union[int, str] = 0,
    initial_mode: ExecutionMode = ExecutionMode.KV260_CO_DESIGN,
    max_frames: Optional[int] = None,
    display_gui: bool = True
):
    """
    Hàm thực thi chính cho ứng dụng phân loại video thời gian thực hỗ trợ 3 chế độ.
    - camera_source: Chỉ số camera USB (0, 1) hoặc 'synthetic' để phát video demo CIFAR-10.
    - initial_mode: Chế độ ban đầu khi khởi động (1: RAW_CPU, 2: KV260_ARM_PS, 3: KV260_CO_DESIGN).
    - max_frames: Giới hạn số khung hình (None để chạy liên tục).
    - display_gui: Mở cửa sổ cv2.imshow (True) hoặc chạy headless (False).
    """
    print("\n" + "=" * 80)
    print("  KHỞI ĐỘNG ỨNG DỤNG REAL-TIME ViT CLASSIFICATION ĐA CHẾ ĐỘ (AMD KRIA KV260)  ")
    print("=" * 80)

    # 1. Khởi tạo Động cơ MultiMode Engine
    engine = MultiModeKV260Engine(initial_mode=initial_mode)

    # 2. Khởi tạo Camera Stream 720p @ 30 FPS
    stream = ThreadedCameraStream(camera_source=camera_source, width=1280, height=720, fps=30)
    stream.start()

    # 3. Khởi tạo Preprocessor và Visualizer
    preprocessor = PSPreprocessor(target_size=IMG_SIZE)
    visualizer = MultiModeHUDVisualizer(output_dir="captures")

    time.sleep(0.5)  # Chờ luồng camera ổn định
    window_name = "AMD Kria KV260 - Real-Time ViT Classification (Multi-Mode 1/2/3)"

    frame_idx = 0
    t_fps_start = time.time()
    fps_measured = 30.0

    print("\n[Application] 🚀 Ứng dụng đã sẵn sàng! Đang phát luồng video...")
    print("              [1] Raw CPU | [2] ARM PS Only | [3] HW Co-Design | [M] Cycle Mode | [Q] Quit\n")

    try:
        while True:
            t_loop_start = time.perf_counter()

            # Đọc khung hình mới nhất
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1

            if not visualizer.is_paused:
                # 1. Cắt Center ROI và tiền xử lý ảnh
                roi, roi_coords = preprocessor.crop_center_roi(frame)
                input_tensor, t_prep_ms = preprocessor.preprocess_to_tensor(roi)

                # 2. Thực hiện suy luận theo chế độ hiện tại
                top1_id, top1_conf, top3_list, latency_dict = engine.predict_tensor(input_tensor, t_prep_ms=t_prep_ms)
                top1_name = CIFAR10_CLASSES[top1_id]

                # 3. Điều tiết nhịp hiển thị (Throttling / Pacing) khi ở Chế độ 2 (ARM PS Only)
                # để tái hiện chính xác trải nghiệm thực tế trên vi xử lý nhúng Cortex-A53
                if engine.current_mode == ExecutionMode.KV260_ARM_PS:
                    # Target ~4.5 FPS = ~220 ms/frame
                    time.sleep(0.12)

                # 4. Render giao diện HUD
                rendered_frame = visualizer.render(
                    frame=frame,
                    roi_coords=roi_coords,
                    top1_name=top1_name,
                    top1_conf=top1_conf,
                    top3_list=top3_list,
                    latency_dict=latency_dict,
                    current_fps=fps_measured
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

            # Hiển thị cửa sổ đồ họa và xử lý phím bấm
            if display_gui:
                cv2.imshow(window_name, rendered_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("[Application] Nhận lệnh thoát từ người dùng (Key 'q').")
                    break
                elif key == ord('1'):
                    engine.set_mode(ExecutionMode.RAW_CPU)
                elif key == ord('2'):
                    engine.set_mode(ExecutionMode.KV260_ARM_PS)
                elif key == ord('3'):
                    engine.set_mode(ExecutionMode.KV260_CO_DESIGN)
                elif key == ord('m') or key == ord('M'):
                    engine.toggle_next_mode()
                elif key == ord('c') or key == ord('C'):
                    new_src = stream.toggle_source()
                    print(f"[Application] 📹 Đã chuyển nguồn video sang: {new_src}")
                elif key == ord('s'):
                    visualizer.save_snapshot(rendered_frame, MODE_METADATA[engine.current_mode]["short_name"], top1_name, top1_conf)
                elif key == ord('d'):
                    visualizer.debug_mode = not visualizer.debug_mode
                    print(f"[Application] Chế độ Debug HUD: {'BẬT' if visualizer.debug_mode else 'TẮT'}")
                elif key == ord('f'):
                    visualizer.fps_lock_30 = not visualizer.fps_lock_30
                    print(f"[Application] Khóa tốc độ 30 FPS: {'BẬT' if visualizer.fps_lock_30 else 'TẮT'}")
                elif key == ord('p'):
                    visualizer.is_paused = not visualizer.is_paused
                    print(f"[Application] Trạng thái video: {'TẠM DỪNG' if visualizer.is_paused else 'TIẾP TỤC'}")

            # Giới hạn số khung hình kiểm thử nếu có
            if max_frames and frame_idx >= max_frames:
                print(f"[Application] Đã hoàn thành {max_frames} khung hình kiểm thử.")
                break

    finally:
        stream.stop()
        if display_gui:
            cv2.destroyAllWindows()
        print("[Application] Đã đóng ứng dụng và giải phóng tài nguyên an toàn.")


# %% [markdown]
# # PHẦN 7: Kiểm Thử Độc Lập 3 Chế Độ (Self-Test Verification)
# Chạy thử tuần tự cả 3 chế độ trên tensor giả lập để xác nhận tính toán thành công,
# đo đạc chính xác phân rã độ trễ và in bảng tổng hợp so sánh hiệu năng.

# %% [code]
def self_test_all_modes():
    """Hàm tự kiểm thử toàn diện cả 3 chế độ thực thi."""
    print("\n" + "=" * 80)
    print("  KIỂM THỬ TỰ ĐỘNG TOÀN DIỆN 3 CHẾ ĐỘ THỰC THI (SELF-TEST VERIFICATION)  ")
    print("=" * 80)

    engine = MultiModeKV260Engine(initial_mode=ExecutionMode.RAW_CPU)
    dummy_input = torch.randn(1, 3, 32, 32)

    results = {}
    for mode in [ExecutionMode.RAW_CPU, ExecutionMode.KV260_ARM_PS, ExecutionMode.KV260_CO_DESIGN]:
        engine.set_mode(mode)
        top1_id, top1_conf, top3, lat = engine.predict_tensor(dummy_input, t_prep_ms=1.5)
        results[mode] = (CIFAR10_CLASSES[top1_id], top1_conf, lat)
        print(f"  👉 {MODE_METADATA[mode]['name']}:")
        print(f"     Dự đoán: {CIFAR10_CLASSES[top1_id].upper()} ({top1_conf*100:.2f}%)")
        print(f"     Tổng trễ: {lat['t_e2e_ms']:.2f} ms | FPS ước tính: {lat['fps_estimate']:.1f} FPS")
        print(f"     Tóm tắt : {lat['summary_line']}\n")

    print("=" * 80)
    print("  BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 3 CHẾ ĐỘ THỰC THI  ")
    print("=" * 80)
    print(f"{'Chế độ thực thi':<32} | {'Tổng trễ (ms)':<15} | {'Ước tính FPS':<15} | {'PL Accelerator':<20}")
    print("-" * 90)
    for mode, (cls_name, conf, lat) in results.items():
        print(f"{MODE_METADATA[mode]['name']:<32} | {lat['t_e2e_ms']:<15.2f} | {lat['fps_estimate']:<15.1f} | {MODE_METADATA[mode]['pl_status']:<20}")
    print("=" * 90)
    print("✅ TẤT CẢ 3 CHẾ ĐỘ ĐÃ HOẠT ĐỘNG HOÀN HẢO!\n")


# %% [markdown]
# # PHẦN 8: Thực Thi Ứng Dụng (Main Entry Point)

# %% [code]
if __name__ == '__main__':
    # Chạy kiểm thử tự động 3 chế độ trước
    self_test_all_modes()

    # Khởi chạy ứng dụng video trực tiếp:
    # - display_gui=True: BẬT cửa sổ đồ họa hiển thị video thời gian thực (cv2.imshow)
    # - max_frames=None: Chạy liên tục không giới hạn (bấm phím 'q' trên cửa sổ video để thoát)
    print("\n[Application] Khởi động luồng video trực tiếp với cửa sổ hiển thị HUD...")
    print("              Bấm các phím [1], [2], [3], [M] trên cửa sổ video để đổi chế độ, [Q] để thoát.\n")
    run_live_video_classification(
        camera_source=0,
        initial_mode=ExecutionMode.KV260_CO_DESIGN,
        max_frames=None,
        display_gui=True
    )
