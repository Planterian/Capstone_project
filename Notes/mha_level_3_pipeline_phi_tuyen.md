# Kiến Trúc MHA Level 3 v2.0: Pipelines Phi Tuyến — Scaler Unit, Softmax BRAM & Context GEMM

> **Dự án:** Tăng tốc Attention Core MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mha_level_3_pipeline_phi_tuyen.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mha_level_3_pipeline_phi_tuyen.md)

---

## 1. TỔNG QUAN TẦNG XỬ LÝ PHI TUYẾN

Tầng xử lý phi tuyến biến đổi các điểm số tích vô hướng thô (Raw Scores INT32) thành ma trận xác suất phân bố chuẩn hóa (UINT16 Probabilities), sau đó nhân tích lũy với ma trận V (Context Aggregation) để cho ra vector ngữ cảnh đầu ra của lớp Self-Attention.

```text
========================================================================================================================
                              LEVEL 3: NON-LINEAR ATTENTION PIPELINE ARCHITECTURE
========================================================================================================================

    Raw Attention Scores: stage1_score [31:0] (Signed INT32, 1 score/cycle)
            │
            v
   +-------------------------------------------------------------------------------------------------------------------+
   | STAGE 2: DYADIC SCALER UNIT (scale_unit.sv)                                                                      |
   |                                                                                                                   |
   |   1. Phép nhân 48-bit: prod_stage = in_score * scale_m  (M = 27640 cho Stage 2)                                  |
   |   2. Dịch bit đại số:  shifted = prod_stage >>> scale_e (E = 22 cho Stage 2)                                     |
   |   3. Kẹp ngưỡng bão hòa: [-32768, +32767] -> INT16 Scaled Score                                                  |
   +----------------------------------------------------+--------------------------------------------------------------+
                                                        |
                                                        | scaled_score [15:0] (Signed INT16)
                                                        v
   +-------------------------------------------------------------------------------------------------------------------+
   | STAGE 3: RECONFIGURABLE HARDWARE SOFTMAX (softmax_lut.sv)                                                         |
   |                                                                                                                   |
   |   [Pass 1: Quét Max] S_max = max(Score_0 .. Score_255) (256 cycles)                                               |
   |   [Pass 2: Tra BRAM] Delta = S_max - Score_i; Exp_val = exp_rom [Stage_offset + Delta]; Tích lũy mẫu số Sum_Exp     |
   |   [Pass 3: Bộ chia]  inv_sum = (2^31 - 1) / Sum_Exp (32-step Restoring Integer Divider)                          |
   |   [Pass 4: Chuẩn hóa]out_prob = (Exp_val * inv_sum) >>> 16 -> UINT16 trong dải [0, 32767]                        |
   +----------------------------------------------------+--------------------------------------------------------------+
                                                        |
                                                        | out_prob [15:0] (UINT16 [0, 32767], Sum ~= 32768)
                                                        v
   +-------------------------------------------------------------------------------------------------------------------+
   | STAGE 4: SCORE x V CONTEXT GEMM ENGINE (score_v_gemm.sv)                                                          |
   |                                                                                                                   |
   |   1. Tích lũy N tokens:   ctx_acc[d] += prob_data * V[token_idx, d]  (32 bộ tích lũy song song, N in {256, 64, 16}) |
   |   2. Chuẩn hóa dịch phải: ctx_shifted[d] = ctx_acc[d] >>> 15                                                      |
   |   3. Kẹp bão hòa INT8:    [-128, +127]                                                                            |
   |   4. Đóng gói Egress:     16B/beat; Xuất N beats (dk<=16) hoặc 2N beats (dk>16); tlast tại beat cuối               |
   +-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. CHI TIẾT CÁC KHỐI CHỨC NĂNG

### 2.1 Khối Dyadic Scaler Unit (`scale_unit.sv`)
- **Toán học I-ViT:** 
  $$S_{\text{effective}} = \frac{S_Q \cdot S_K}{S_{\text{score}} \cdot \sqrt{d_k}} \approx M \times 2^{-E}$$
- **Phần cứng:** 
  - Tầng nhân 1: `prod_stage <= 48'($signed(in_score)) * 48'($signed(scale_m));`
  - Tầng dịch 2: `clamped_score <= sat_clamp(prod_stage >>> scale_e);`
  - Đảm bảo độ trễ 2 chu kỳ clock có thanh ghi pipeline chốt tín hiệu, không gây suy giảm tần số hoạt động ($F_{\max} \ge 250\text{ MHz}$).

### 2.2 Khối Softmax Engine Tái Cấu Hình BRAM (`softmax_lut.sv`)
- **BRAM Phân vùng đa tầng:**
  - Kích thước: 1024 từ $\times$ 20 bits.
  - Phân vùng tự động theo tín hiệu `head_dim`:
    - `head_dim == 16`: Trỏ Offset `0x000` (Stage 2)
    - `head_dim == 20`: Trỏ Offset `0x100` (Stage 3)
    - `head_dim == 24`: Trỏ Offset `0x200` (Stage 4)
- **Bộ chia lặp 32 chu kỳ (Restoring Divider):**
  - Khắc phục hoàn toàn lỗi chia tổ hợp và lỗi tràn số dấu phẩy tĩnh.
  - Sử dụng tử số chuẩn hóa $2^{31} - 1$, đầu ra xác suất đạt chuẩn UINT16 $[0, 32767]$ khớp từng bit (0-LSB error) với mô hình PyTorch I-ViT.

### 2.3 Khối Context GEMM (`score_v_gemm.sv`) & Giao Diện AXI4-Stream TX 128-Bit
- **32 Bộ tích lũy song song:** Tích lũy liên tục qua $N$ tokens động (`ctx_acc[d] += prob_data * V[k, d]`, với $N \in \{256, 64, 16\}$ theo từng Stage), dịch phải 15 bits và kẹp bão hòa số nguyên INT8 $[-128, +127]$.
- **Đóng gói Egress Native 128-bit (`AXIS_DATA_WIDTH = 128`) thích ứng Dynamic Sequence Length $N$:**
  - **Với Stage 2 ($N=256, d_k = 16$):** Mỗi hàng context xuất trong đúng **1 beat duy nhất** (`ST_OUT_BEAT0` ghép 16 bytes `ctx_safe[15:0]`). Toàn bộ 256 hàng xuất trọn vẹn trong **256 beats / head**, tín hiệu `m_axis_tlast` kích hoạt chính xác tại beat thứ 255.
  - **Với Stage 3 ($N=64, d_k = 20$):** Mỗi hàng xuất trong **2 beats** (`ST_OUT_BEAT0` 16 bytes thấp, `ST_OUT_BEAT1` 4 bytes cao kèm mask `m_axis_tkeep = 16'h000F`). Xuất trọn vẹn trong **128 beats / head** ($64 \times 2$), `m_axis_tlast` kích hoạt chính xác tại beat thứ 127.
  - **Với Stage 4 ($N=16, d_k = 24$):** Mỗi hàng xuất trong **2 beats** (`ST_OUT_BEAT0` 16 bytes thấp, `ST_OUT_BEAT1` 8 bytes cao kèm mask `m_axis_tkeep = 16'h00FF`). Xuất trọn vẹn trong **32 beats / head** ($16 \times 2$), `m_axis_tlast` kích hoạt chính xác tại beat thứ 31.
- Hỗ trợ đầy đủ tín hiệu bắt tay backpressure AXI-Stream Egress (`m_axis_tready` / `m_axis_tvalid`).

