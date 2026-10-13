"""
Export Dyadic Scales & Bit-Exact Golden Test Vectors (.hex)
============================================================
BƯỚC 1: Chuẩn hóa Python Golden Model & Trích xuất Golden Test Vectors.
Đồng bộ Bit-Exact giữa PyTorch I-ViT (MobileViT-XXS QAT 67.28%)
và Kiến trúc phần cứng tăng tốc trên AMD Kria KV260.

Script thực hiện:
  1. Nạp mô hình MobileViT-XXS + I-ViT và checkpoint QAT tốt nhất.
  2. [Trích xuất Scale] Phân rã tham số Dyadic Scaling chính xác cho 3 Stages:
       - STAGE 2 (dim=16): M=18308, E=22
       - STAGE 3 (dim=20): M=22559, E=22
       - STAGE 4 (dim=24): M=19947, E=22
       - Xuất dyadic_scale_config.json và hw_scale_params.svh.
  3. [Trích xuất Vector] Chạy forward hook trên tầng Transformer Stage 2:
       - q_matrix.hex              (256 x 16, INT8 signed)
       - k_matrix.hex              (256 x 16, INT8 signed)
       - v_matrix.hex              (256 x 16, INT8 signed)
       - golden_gemm_qk.hex        (256 x 256, INT32 signed)
       - golden_scaled_score.hex   (256 x 256, INT16 signed)
       - golden_softmax_prob.hex   (256 x 256, UINT16 in [0, 32767])
       - golden_softmax_prob_u8.hex(256 x 256, UINT8 in [0, 255])
       - golden_context_out.hex    (256 x 16, INT8 signed)
  4. Kiểm chứng tính toàn vẹn (0-LSB / 1-LSB match) với mô hình gốc.
"""

import os
import sys
import json
import math
import numpy as np
import torch
from pathlib import Path

# Đảm bảo mã hóa UTF-8 an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được ivit_surgery
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_NOTEBOOK_DIR = os.path.dirname(_CURRENT_DIR)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
if _NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, _NOTEBOOK_DIR)

import patch_ivit_device  # noqa: F401
from ivit_surgery.mobilevit_ivit_surgery import replace_mobilevit_blocks
from ivit_surgery.mobilevit_ivit_qat import load_calibrated_weights
from transformers import MobileViTForImageClassification


def decompose_dyadic_16bit(scale_float: float, max_bit: int = 15):
    """
    Phân rã scale thập phân thành cặp Dyadic (M, E):
        S ~= M * 2^(-E)
    với M là số nguyên 16-bit có dấu [0, 32767], E là số dịch bit [0, 31].
    """
    if scale_float <= 0:
        return 0, 0
    mantissa, exp = math.frexp(scale_float)
    m_int = int(round(mantissa * (2 ** max_bit)))
    e_shift = max_bit - exp
    return m_int, int(e_shift)


def write_hex_file(filepath: str, data_array: np.ndarray, bit_width: int = 8, is_signed: bool = True):
    """
    Ghi dữ liệu mảng numpy thành tệp .hex tương thích $readmemh của Verilog.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    flat_data = data_array.flatten()
    
    with open(filepath, "w") as f:
        for val in flat_data:
            val_int = int(val)
            if bit_width == 8:
                if is_signed and val_int < 0:
                    val_int = (val_int + 256) & 0xFF
                f.write(f"{val_int:02X}\n")
            elif bit_width == 16:
                if is_signed and val_int < 0:
                    val_int = (val_int + 65536) & 0xFFFF
                f.write(f"{val_int:04X}\n")
            elif bit_width == 32:
                if is_signed and val_int < 0:
                    val_int = (val_int + 4294967296) & 0xFFFFFFFF
                f.write(f"{val_int:08X}\n")
            else:
                f.write(f"{val_int:X}\n")
    print(f"  [HEX] Đã xuất {len(flat_data)} phần tử ({bit_width}-bit) -> {filepath}")


def run_export_pipeline(ckpt_path: str, output_dirs: list):
    print("=" * 70)
    print("[BƯỚC 1] CHUẨN HÓA PYTHON GOLDEN MODEL & XUẤT GOLDEN VECTORS (.HEX)")
    print("=" * 70)
    device = torch.device("cpu")

    # 1. Khởi tạo mô hình
    print("\n[1/4] Khởi tạo mô hình MobileViT-XXS + I-ViT...")
    model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-xx-small", low_cpu_mem_usage=True
    )
    model, _ = replace_mobilevit_blocks(model)
    model.eval()

    # 2. Nạp checkpoint
    print(f"\n[2/4] Nạp checkpoint QAT: {ckpt_path}")
    assert os.path.exists(ckpt_path), f"Không tìm thấy file: {ckpt_path}"
    load_calibrated_weights(model, ckpt_path, device=device)

    # 3. Trích xuất chính xác hệ số Dyadic Scale cho cả 3 Stages
    print("\n[3/4] Trích xuất chính xác bộ tham số Dyadic Scale (M, E)...")
    stages = {
        "STAGE2": {"layer": model.mobilevit.encoder.layer[2].transformer.layer[0].attn, "dim": 16},
        "STAGE3": {"layer": model.mobilevit.encoder.layer[3].transformer.layer[0].attn, "dim": 20},
        "STAGE4": {"layer": model.mobilevit.encoder.layer[4].transformer.layer[0].attn, "dim": 24},
    }

    stage_params = {}
    svh_lines = [
        "// ============================================================================",
        "// AUTO-GENERATED HARDWARE SCALE PARAMETERS FOR KRIA KV260 ATTENTION CORE",
        "// Generated from MobileViT + I-ViT QAT Model (Accuracy: 67.28%)",
        "// Remediation Step 1: Bit-Exact Alignment with PyTorch I-ViT",
        "// ============================================================================\n",
        "`ifndef HW_SCALE_PARAMS_SVH",
        "`define HW_SCALE_PARAMS_SVH\n",
    ]

    for stage_name, info in stages.items():
        attn_mod = info["layer"]
        d_k = info["dim"]
        sq = attn_mod.qact1.act_scaling_factor.item()
        s_score = attn_mod.qact_attn1.act_scaling_factor.item()
        
        # S_effective = (Sq * Sk) / (S_score * sqrt(d_k))
        s_eff = (sq * sq) / (s_score * math.sqrt(d_k))
        m_eff, e_eff = decompose_dyadic_16bit(s_eff, max_bit=15)

        # BƯỚC 2.5: Tiền tính toán bảng Lookup Table int_exp_shift 256 phần tử cho từng Stage
        def calc_exp_shift_entry(x_int, sf, n=15):
            x_int = x_int + math.floor(x_int / 2.0) - math.floor(x_int / 16.0)
            x0_int = math.floor(-1.0 / sf)
            x_int = max(x_int, n * x0_int)
            q = math.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int = r / 2.0 - x0_int
            exp_int = max(math.floor(exp_int * (2.0 ** (n - q))), 0)
            return int(exp_int)

        stage_exp_table = [calc_exp_shift_entry(-d, s_score, n=15) for d in range(256)]

        stage_params[stage_name] = {
            "head_dim": d_k,
            "S_Q": sq,
            "S_score": s_score,
            "S_effective": s_eff,
            "M_scale": m_eff,
            "E_scale": e_eff,
            "exp_rom_table": stage_exp_table,
        }

        svh_lines.append(f"// {stage_name}: Head Dim = {d_k}, Effective Scale = {s_eff:.6f}")
        svh_lines.append(f"`define {stage_name}_HEAD_DIM       16'd{d_k}")
        svh_lines.append(f"`define {stage_name}_SCALE_M        16'd{m_eff}")
        svh_lines.append(f"`define {stage_name}_SCALE_E        5'd{e_eff}\n")

    svh_lines.append("// DEFAULT SETTINGS (MobileViT-XXS Stage 2)")
    svh_lines.append("`define DEFAULT_SEQ_LEN       16'd256")
    svh_lines.append("`define DEFAULT_HEAD_DIM      16'd16")
    svh_lines.append("`define DEFAULT_NUM_HEADS     8'd4")
    svh_lines.append(f"`define DEFAULT_SCALE_M       16'd{stage_params['STAGE2']['M_scale']}")
    svh_lines.append(f"`define DEFAULT_SCALE_E       5'd{stage_params['STAGE2']['E_scale']}\n")
    svh_lines.append("`endif // HW_SCALE_PARAMS_SVH\n")

    # 4. Trích xuất Golden Test Vectors bằng forward hook chân thực
    print("\n[4/4] Trích xuất Golden Test Vectors cho Single Head (Stage 2: 256x16)...")
    # Cố định hoàn toàn running_stat trên toàn bộ mô hình để dùng đúng tham số đã hiệu chuẩn (calibrated)
    for m in model.modules():
        if hasattr(m, 'running_stat'):
            m.running_stat = False

    target_layer = stages["STAGE2"]["layer"]
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 3, 256, 256)[:, [2, 1, 0], :, :]

    captured = {}
    def hook_attn(module, inp, out):
        x_in, sf_in = inp[0], inp[1]
        B, N, C = x_in.shape
        H = module.num_attention_heads
        d_k = module.attention_head_size
        
        # Projection QKV
        x_qkv, sf_qkv = module.qkv(x_in, sf_in)
        x_qact1, sf_qact1 = module.qact1(x_qkv, sf_qkv)
        
        # Tách Head 0 của Batch 0
        qkv = x_qact1.reshape(B, N, 3, H, d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn, sf_attn = module.matmul_1(q, sf_qact1, k.transpose(-2, -1), sf_qact1)
        attn = attn * module.scale
        sf_attn = sf_attn * module.scale
        attn_q, sf_attn_q = module.qact_attn1(attn, sf_attn)
        attn_sm, sf_sm = module.int_softmax(attn_q, sf_attn_q)
        out_ctx, sf_ctx = module.matmul_2(attn_sm, sf_sm, v, sf_qact1)
        
        captured["q"] = q[0, 0].detach().cpu()
        captured["k"] = k[0, 0].detach().cpu()
        captured["v"] = v[0, 0].detach().cpu()
        captured["sf_q"] = sf_qact1.detach().cpu()
        captured["sf_score"] = sf_attn_q.detach().cpu()
        captured["attn_q"] = attn_q[0, 0].detach().cpu()
        captured["attn_sm"] = attn_sm[0, 0].detach().cpu()
        captured["sf_sm"] = sf_sm.detach().cpu()
        captured["out_ctx"] = out_ctx[0, 0].detach().cpu()
        captured["sf_ctx"] = sf_ctx.detach().cpu()

    h = target_layer.register_forward_hook(hook_attn)
    with torch.no_grad():
        _ = model(pixel_values=dummy_input)
    h.remove()

    q = captured["q"]
    k = captured["k"]
    v = captured["v"]
    sf_q = captured["sf_q"].item()
    sf_sm = captured["sf_sm"].item()
    sf_ctx = captured["sf_ctx"].item()

    # Định lượng đầu vào INT8 [-128, 127]
    q_int8 = torch.clamp(torch.round(q / sf_q), -128, 127).to(torch.int8).numpy()
    k_int8 = torch.clamp(torch.round(k / sf_q), -128, 127).to(torch.int8).numpy()
    v_int8 = torch.clamp(torch.round(v / sf_q), -128, 127).to(torch.int8).numpy()

    # 1. Raw GEMM: Q * K^T -> INT32 [256, 256]
    gemm_qk_int32 = np.matmul(q_int8.astype(np.int32), k_int8.astype(np.int32).T)

    # 2. Scaled score bằng Dyadic Rescaler chuẩn: (Raw * M) >>> E -> INT16 [256, 256]
    m_scale = stage_params["STAGE2"]["M_scale"]
    e_scale = stage_params["STAGE2"]["E_scale"]
    scaled_score_int64 = (gemm_qk_int32.astype(np.int64) * m_scale) >> e_scale
    scaled_score_int16 = np.clip(scaled_score_int64, -32768, 32767).astype(np.int16)

    # 3. Softmax Probabilities:
    # 16-bit integer chuẩn xác theo toán tử IntSoftmax của I-ViT ứng dụng trực tiếp từ scaled_score_int16
    def int_exp_shift(x_int, sf, n=15):
        x_int = x_int + np.floor(x_int / 2.0) - np.floor(x_int / 16.0)
        x0_int = np.floor(-1.0 / sf)
        x_int = np.maximum(x_int, n * x0_int)
        q = np.floor(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2.0 - x0_int
        exp_int = np.maximum(np.floor(exp_int * (2.0 ** (n - q))), 0)
        return exp_int

    s_score_stage2 = stage_params["STAGE2"]["S_score"]
    exp_rom_table = np.array([int(int_exp_shift(-d, s_score_stage2, n=15)) for d in range(256)], dtype=np.int64)
    softmax_prob_uint16 = np.zeros_like(scaled_score_int16, dtype=np.uint16)
    for r in range(256):
        row_s = scaled_score_int16[r].astype(np.int64)
        s_max = np.max(row_s)
        delta = -(row_s - s_max)
        delta = np.clip(delta, 0, 255)
        exp_vals = exp_rom_table[delta]
        sum_exp = int(np.sum(exp_vals))
        factor = (2**31 - 1) // sum_exp
        prob = (exp_vals * factor) >> 16
        softmax_prob_uint16[r] = np.clip(prob, 0, 65535).astype(np.uint16)

    # Bản UINT8 chuẩn hóa phục vụ các kiểm thử 8-bit
    prob_float = softmax_prob_uint16.astype(float)
    prob_normalized = prob_float / np.sum(prob_float, axis=-1, keepdims=True)
    softmax_prob_uint8 = np.clip(np.round(prob_normalized * 255.0), 0, 255).astype(np.uint8)

    # 4. Context Output:
    # Context = Softmax_16 * V -> INT32
    context_int32 = np.matmul(softmax_prob_uint16.astype(np.int32), v_int8.astype(np.int32))
    # Chuẩn hóa về INT8 (dịch phải 15 do tổng xác suất ~ 32768)
    context_out_int8 = np.clip(context_int32 >> 15, -128, 127).astype(np.int8)

    # Đối soát Bit-Exact với mô hình PyTorch
    model_ctx_fp = captured["out_ctx"].numpy()
    model_ctx_int = np.round(model_ctx_fp / sf_ctx)
    diff_ctx = np.abs(context_int32 - model_ctx_int)
    print(f"\n[ĐỐI SOÁT BIT-EXACT]: Context INT32 vs PyTorch I-ViT:")
    print(f"  Max Error: {diff_ctx.max()} LSB")
    print(f"  Tỷ lệ khớp chính xác 100% (0 LSB): {np.all(diff_ctx == 0)}")

    # Ghi tệp ra các thư mục đích
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        # JSON config
        json_path = os.path.join(out_dir, "dyadic_scale_config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"stages": stage_params}, f, indent=2)
        print(f"  [SAVED] Dyadic JSON -> {json_path}")

        # SVH header
        svh_path = os.path.join(out_dir, "hw_scale_params.svh")
        with open(svh_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svh_lines))
        print(f"  [SAVED] Header SVH -> {svh_path}")

        # Hex vectors
        write_hex_file(os.path.join(out_dir, "q_matrix.hex"), q_int8, bit_width=8, is_signed=True)
        write_hex_file(os.path.join(out_dir, "k_matrix.hex"), k_int8, bit_width=8, is_signed=True)
        write_hex_file(os.path.join(out_dir, "v_matrix.hex"), v_int8, bit_width=8, is_signed=True)
        write_hex_file(os.path.join(out_dir, "golden_gemm_qk.hex"), gemm_qk_int32, bit_width=32, is_signed=True)
        write_hex_file(os.path.join(out_dir, "golden_scaled_score.hex"), scaled_score_int16, bit_width=16, is_signed=True)
        write_hex_file(os.path.join(out_dir, "golden_softmax_prob.hex"), softmax_prob_uint16, bit_width=16, is_signed=False)
        write_hex_file(os.path.join(out_dir, "golden_softmax_prob_u8.hex"), softmax_prob_uint8, bit_width=8, is_signed=False)
        write_hex_file(os.path.join(out_dir, "golden_context_out.hex"), context_out_int8, bit_width=8, is_signed=True)

        # BƯỚC 2.5: Xuất tệp nhị phân exp_rom_all_stages.hex (1024 words x 20-bit)
        rom_all_stages = (
            stage_params["STAGE2"]["exp_rom_table"]
            + stage_params["STAGE3"]["exp_rom_table"]
            + stage_params["STAGE4"]["exp_rom_table"]
            + stage_params["STAGE2"]["exp_rom_table"]  # Reserved slot (copy Stage 2)
        )
        rom_hex_path = os.path.join(out_dir, "exp_rom_all_stages.hex")
        with open(rom_hex_path, "w", encoding="utf-8") as f:
            for val in rom_all_stages:
                f.write(f"{int(val):05X}\n")
        print(f"  [SAVED] Multi-Stage Exp ROM Hex (1024 words) -> {rom_hex_path}")

    print("\n" + "=" * 70)
    print(f"[HOÀN TẤT] BƯỚC 1 & BƯỚC 2.5 ĐÃ XUẤT TOÀN BỘ GOLDEN VECTORS & MULTI-STAGE EXP ROM.")
    print("=" * 70)


if __name__ == "__main__":
    default_ckpt = os.path.join(_CURRENT_DIR, "mobilevit_ivit_qat_best.pth")
    output_targets = [
        os.path.join(_CURRENT_DIR, "golden_vectors"),
        r"c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02",
    ]
    run_export_pipeline(default_ckpt, output_targets)
