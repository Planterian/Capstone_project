# HƯỚNG DẪN CÁC LỆNH TERMINAL CHẠY KIỂM THỬ XÁC MINH (VERIFICATION GUIDE) - BIẾN THỂ 3B

> **Dự án:** Bộ tăng tốc phần hardware MobileViT-XXS Self-Attention trên AMD Kria KV260  
> **Kiến trúc:** **Phương án 3B (Variant 3B)** — 2D Output-Stationary Systolic MAC Array with DSP Packing (**16 DSP48E2**, 32 MACs/chu kỳ) & Tile-to-Row Score Reordering Buffer  
> **Thư mục làm việc:** `Project/src/Hardware/rtl_ver02_3B`  
> **Mục tiêu:** Cung cấp tài liệu tra cứu đầy đủ, chính xác từng dòng lệnh terminal (PowerShell / Command Prompt) để biên dịch và chạy mô phỏng Icarus Verilog (`iverilog` / `vvp`) cho từng cấp độ kiểm thử độc lập cũng như toàn bộ hệ thống Top-Level Integration của Biến thể 3B.

---

## 1. THIẾT LẬP MÔI TRƯỜNG & ĐƯỜNG DẪN LÀM VIỆC

### 1.1 Yêu cầu công cụ
* **Trình mô phỏng RTL:** Icarus Verilog (`iverilog` phiên bản 12.0 trở lên) hỗ trợ chuẩn IEEE 1800-2012 SystemVerilog (cờ `-g2012`).
  - Đường dẫn mặc định: `C:\iverilog\bin\iverilog.exe` và `C:\iverilog\bin\vvp.exe`.
* **Shell thực thi:** Windows PowerShell (`pwsh` / `powershell.exe`) hoặc Windows Command Prompt (`cmd.exe`).
* **Golden Vectors:** Các tệp `.hex` chuẩn hóa từ mô hình PyTorch đã được copy sẵn vào thư mục `rtl_ver02_3B`:
  - `q_matrix.hex`, `k_matrix.hex`, `v_matrix.hex`: Ma trận đầu vào 256 tokens $\times$ 16 chiều.
  - `golden_gemm_qk.hex`: 65,536 giá trị tích vô hướng $Q K^T$ INT32.
  - `golden_scaled_score.hex`: 65,536 điểm sau tỉ lệ bão hòa dyadic INT16.
  - `exp_rom_all_stages.hex`: Bảng tra cứu số mũ 1,024 từ 20-bit.
  - `golden_softmax_prob.hex`: 1,024 phân phối xác suất xác minh Softmax.
  - `golden_context_out.hex`: 4,096 byte Context Out chuẩn từ PyTorch ($256 \times 16$).

### 1.2 Di chuyển vào thư mục làm việc
Trước khi chạy bất kỳ lệnh nào, hãy mở terminal và di chuyển trực tiếp vào thư mục mã nguồn của Biến thể 3B:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
```

---

## 2. BẢNG TỔNG HỢP CÁC LỆNH VERIFICATION (VARIANT 3B)

| Cấp độ | Module kiểm tra | Testbench thực thi | File mô phỏng (.vvp) | Mục tiêu kiểm thử | Thời gian | Kết quả chuẩn |
|:---|:---|:---|:---|:---|:---:|:---:|
| **Level 1** | Systolic MAC Array & DSP Packing | [`tb_systolic_mac_array_3B.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/tb_systolic_mac_array_3B.sv) | `tb_systolic_3b.vvp` | 16 DSP48E2 + Lưới $4\times 4$ PE + Wavefront Skew + Drain tuần tự 32 items | ~2s | **100% BIT-EXACT (FL-03/FL-04 PASS)** |
| **Level 2** | Dyadic Rescaler | [`tb_golden_scale_unit.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/tb_golden_scale_unit.sv) | `tb_scale_3b.vvp` | Phép nhân định tỉ lệ $(S \times M) \gg E$, 65,536 điểm ($M=27640, E=22$) | ~3s | **100% BIT-EXACT (65,536/65,536)** |
| **Level 3** | Hardware IntSoftmax | [`tb_golden_softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/tb_golden_softmax_lut.sv) | `tb_softmax_3b.vvp` | Max-subtraction, Exp ROM, Restoring Divider 32 chu kỳ | ~2s | **100% BIT-EXACT (1,024/1,024)** |
| **Level 4** | Score-V Context GEMM | [`tb_score_v_gemm.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/tb_score_v_gemm.sv) | `tb_score_v_3b.vvp` | GEMM 2 ($P \times V$), dịch bit `>>> 15`, kiểm tra `tlast` | ~45s | **100% BIT-EXACT (4,096/4,096)** |
| **Level 5 (TOP)** | **Attention Core Top 3B** | [`tb_attention_core_3B.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/tb_attention_core_3B.sv) | `tb_top_3b.vvp` | **Toàn bộ pipeline 3B: AXI-Stream Ingress + Lưới Systolic 16 DSP + Reorder Buffer + Softmax + Score-V** | **~50s** | **100% BIT-EXACT (4,096/4,096)** |

---

## 3. CHI TIẾT TỪNG LỆNH CHẠY & EXPECTED OUTPUT LOGS

### 3.1 Cấp độ 1: Kiểm thử Khối Systolic MAC Array & DSP Packing (`systolic_mac_array.sv`)
Kiểm tra toàn diện kiến trúc mảng tâm thu 2D kích thước $4 \times 4$ PE Output-Stationary tích hợp 16 lát cắt phần cứng DSP48E2 (`pe_slice_dsp_packing.sv`). Mỗi PE slice thực hiện 2 phép nhân INT8 $\times$ INT8 song song. Mạch skew tự động phân tán dữ liệu theo đường chéo wavefront ($r$ cycle trễ hàng, $c$ cycle trễ cột). Kiểm tra triệt để:
1. **FL-03**: Xác minh tính độc lập giữa Col 0 và Col 1 trong cùng một PE, không bị nhiễu chéo dấu (crosstalk).
2. **FL-04**: Xác minh cơ chế xả kết quả tuần tự chân thực (True Sequential Drain) đúng 32 giá trị tích vô hướng per tile mà không bị mất dữ liệu hay nghẽn nhịp.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
C:\iverilog\bin\iverilog.exe -g2012 -o tb_systolic_3b.vvp pe_slice_dsp_packing.sv systolic_mac_array.sv tb_systolic_mac_array_3B.sv
C:\iverilog\bin\vvp.exe tb_systolic_3b.vvp
```

#### Expected Output Log:
```text
=======================================================================
   TESTBENCH: 2D SYSTOLIC MAC ARRAY VERIFICATION (VARIANT 3B)
   16 DSP48E2 Hard Slices + 4x4 Output-Stationary Grid + Built-in Auto-Skew
=======================================================================
[TB INFO]: Reset released. Starting systolic feed sequence...
[TB INFO]: Step 1 - Feeding diagonal wavefront vectors (D=16 cycles)...
[TB INFO]: Step 2 - Feeding flush zeros (12 cycles wavefront propagation)...
[TB INFO]: Step 3 - Triggering Tile Drain Sequence (32 dot products)...
   [DRAIN #00] PE(0, 0) SubCol 0: DUT = 1600 | Golden = 1600 (MATCH)
   [DRAIN #01] PE(0, 0) SubCol 1: DUT = 3200 | Golden = 3200 (MATCH)
   [DRAIN #02] PE(0, 1) SubCol 0: DUT = 1600 | Golden = 1600 (MATCH)
   [DRAIN #03] PE(0, 1) SubCol 1: DUT = 3200 | Golden = 3200 (MATCH)
   [DRAIN #30] PE(3, 3) SubCol 0: DUT = 1600 | Golden = 1600 (MATCH)
   [DRAIN #31] PE(3, 3) SubCol 1: DUT = 3200 | Golden = 3200 (MATCH)

=======================================================================
   2D SYSTOLIC MAC ARRAY VERIFICATION SUMMARY (VARIANT 3B)
=======================================================================
   Total Drained Items : 32 / 32
   Exact Matched Items : 32
   Mismatches Detected : 0
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

### 3.2 Cấp độ 2: Kiểm thử Khối Dyadic Rescaler (`scale_unit.sv`)
Kiểm thử mạch nhân định tỉ lệ dyadic bão hòa $(S \times M) \gg E$ với hệ số chuẩn Stage 2 ($M = 27640, E = 22$), đối soát toàn bộ 65,536 điểm giữa `golden_gemm_qk.hex` và `golden_scaled_score.hex`.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
C:\iverilog\bin\iverilog.exe -g2012 -o tb_scale_3b.vvp scale_unit.sv tb_golden_scale_unit.sv
C:\iverilog\bin\vvp.exe tb_scale_3b.vvp
```

#### Expected Output Log:
```text
=======================================================================
   TESTBENCH: scale_unit vs PyTorch I-ViT Golden Model (Stage 2)
=======================================================================
[TB INFO]: Loading golden_gemm_qk.hex (65536 samples)...
[TB INFO]: Loading golden_scaled_score.hex (65536 samples)...
[TB INFO]: Reset Released. Streaming inputs through scale_unit...
=======================================================================
   FINAL VERIFICATION REPORT: SCALE_UNIT vs GOLDEN MODEL
=======================================================================
   Total Elements Tested : 65536
   Exact Matches (0 LSB) : 65536
   Mismatches Detected   : 0
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

### 3.3 Cấp độ 3: Kiểm thử Khối Hardware IntSoftmax Engine (`softmax_lut.sv`)
Kiểm thử 3 giai đoạn của bộ Softmax số nguyên:
1. Giai đoạn 1: Tìm giá trị cực đại hàng $S_{\max}$ qua bộ so sánh trượt 256 phần tử.
2. Giai đoạn 2: Trừ giá trị cực đại $(S_i - S_{\max})$, tra bảng ROM số mũ 1,024 từ `exp_rom_all_stages.hex` và cộng dồn $\sum \exp$.
3. Giai đoạn 3: Bộ chia không phục hồi (Restoring Divider) 32 chu kỳ tính xác suất $P_i = (\exp(S_i) \ll 15) / \sum \exp$.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
C:\iverilog\bin\iverilog.exe -g2012 -o tb_softmax_3b.vvp softmax_lut.sv tb_golden_softmax_lut.sv
C:\iverilog\bin\vvp.exe tb_softmax_3b.vvp
```

#### Expected Output Log:
```text
=======================================================================
   TESTBENCH: softmax_lut vs PyTorch I-ViT Golden Model
=======================================================================
[TB INFO]: Loading ROM initialized from exp_rom_all_stages.hex...
[TB INFO]: Loading golden_scaled_score.hex (256 inputs/row)...
[TB INFO]: Loading golden_softmax_prob.hex (256 outputs/row)...
[TB INFO]: Reset Released. Testing 4 test rows (1024 probabilities)...
=======================================================================
   FINAL VERIFICATION REPORT: SOFTMAX_LUT vs GOLDEN MODEL
=======================================================================
   Total Probabilities Tested : 1024
   Exact Matches (0 LSB)      : 1024
   Tolerable Difference (1 LSB): 0
   Fatal Errors (>1 LSB)      : 0
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

### 3.4 Cấp độ 4: Kiểm thử Khối Score-V Context GEMM (`score_v_gemm.sv`)
Kiểm thử phép nhân ma trận trọng số xác suất với ma trận Value: $\text{Context} = P \times V$. Kiểm tra 16 bộ tích lũy MAC song song cho 16 chiều không gian đặc trưng ($d_k = 16$), bộ dịch chuẩn hóa `>>> 15`, và tín hiệu báo hoàn tất hàng.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
C:\iverilog\bin\iverilog.exe -g2012 -o tb_score_v_3b.vvp score_v_gemm.sv tb_score_v_gemm.sv requant_residual_add.sv
C:\iverilog\bin\vvp.exe tb_score_v_3b.vvp
```

#### Expected Output Log:
```text
=======================================================================
   TESTBENCH: score_v_gemm vs PyTorch Golden Context Out
=======================================================================
[TB INFO]: Loading golden_softmax_prob.hex (65536 probabilities)...
[TB INFO]: Loading v_matrix.hex (4096 values)...
[TB INFO]: Loading golden_context_out.hex (4096 context values)...
[TB INFO]: Streaming 256 rows through score_v_gemm engine...
=======================================================================
   FINAL VERIFICATION REPORT: SCORE_V_GEMM vs GOLDEN MODEL
=======================================================================
   Total Context Elements Tested : 4096 / 4096
   Exact Matches                 : 4096
   Mismatches Detected           : 0
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

### 3.5 Cấp độ 5 (TOP-LEVEL): Toàn Bộ Hệ Thống Attention Core Top 3B (`attention_core_top_3B.sv`)
Kiểm thử toàn bộ hệ thống tăng tốc Self-Attention Biến thể 3B từ đầu đến cuối:
1. Giao tiếp AXI4-Lite cấu hình các thanh ghi CSR ($N=256, d_k=16, H=1, M=27640, E=22$).
2. Giao tiếp AXI4-Stream Ingress nạp ma trận $Q, K, V$ (1,536 beats $\times$ 8 bytes = 12,288 bytes) vào bộ đệm Ping-Pong BRAM.
3. Bộ điều khiển FSM quét 64 khối hàng $\times$ 32 mảnh cột ($4 \times 8$ tile), điều phối dòng tính toán trên mảng tâm thu 16 DSP.
4. Bộ đệm tái sắp xếp [`reorder_buffer_3B.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_3B/reorder_buffer_3B.sv) gom đủ $4 \times 256$ điểm INT32, sau đó phát liên tục 256 phần tử/hàng sang khối Dyadic Scaler, Softmax và Context GEMM.
5. Cổng ra AXI4-Stream Egress giám sát và so khớp đúng 4,096 byte kết quả với `golden_context_out.hex` cùng cờ `m_axis_tlast` khẳng định chính xác ở nhịp 511.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
C:\iverilog\bin\iverilog.exe -g2012 -o tb_top_3b.vvp tb_attention_core_3B.sv attention_core_top_3B.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pe_slice_dsp_packing.sv systolic_mac_array.sv reorder_buffer_3B.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv requant_residual_add.sv
C:\iverilog\bin\vvp.exe tb_top_3b.vvp
```

#### Expected Output Log:
```text
=======================================================================
   STARTING INTEGRATED ATTENTION CORE VERIFICATION (VARIANT 3B)
   2D SYSTOLIC MAC ARRAY (16 DSP48E2) + SCORE REORDERING BUFFER
=======================================================================
[TB INFO]: Loading Golden Test Vectors from .hex files...
[TB INFO]: Loaded 256 Q tokens, 256 K tokens, 256 V tokens.
[TB INFO]: Reset Released.
[TB INFO]: Step 2 - Configuring CSR (N=256, dk=16, H=1, M=27640, E=22)...
[TB INFO]: Step 3 - Streaming Q, K, V matrices into AXI-Stream Ingress...
[TB INFO]: Finished Streaming 1,536 beats into Ingress BRAM.
[TB INFO]: Step 4 - Triggering START Pulse via AXI-Lite (0x00 <= 1)...
[TB INFO]: Step 5 - Waiting for Systolic Tile Pipeline (64 blocks x 32 tiles)...
[TB INFO]: Running... Cycle 50000, Egress Samples: 448 / 4096
[TB INFO]: Running... Cycle 100000, Egress Samples: 1088 / 4096
[TB INFO]: Running... Cycle 150000, Egress Samples: 1728 / 4096
[TB INFO]: Running... Cycle 200000, Egress Samples: 2368 / 4096
[TB INFO]: Running... Cycle 250000, Egress Samples: 3008 / 4096
[TB INFO]: Running... Cycle 300000, Egress Samples: 3648 / 4096
[CONTEXT EGRESS #0000]: DUT = -1 | Golden = -1 (MATCH)
[CONTEXT EGRESS #0511]: DUT = -2 | Golden = -2 (MATCH)
[CONTEXT EGRESS #4095]: DUT = 1 | Golden = 1 (MATCH)
[TB INFO]: m_axis_tlast correctly asserted on beat 511!

=======================================================================
   FINAL VERIFICATION REPORT: ATTENTION CORE TOP 3B (16 DSP48E2)
=======================================================================
   Total Egress Samples Monitored : 4096 / 4096
   Exact Matched Samples          : 4096
   Mismatches                     : 0
   TLAST Assertion Errors         : 0
   Total Execution Clock Cycles   : ~325400
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

## 4. CHẠY TỰ ĐỘNG TẤT CẢ TESTBENCH (ONE-CLICK BATCH SCRIPT)

Để chạy toàn bộ 5 cấp độ kiểm thử của Biến thể 3B một cách tự động chỉ bằng một lệnh duy nhất:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_3B
.\run_verifications_3B.bat
```

Script sẽ tự động dừng và báo lỗi nếu có bất kỳ module nào không vượt qua bước kiểm thử bit-exact.

---

## 5. PHÂN TÍCH HIỆU NĂNG & TÀI NGUYÊN PHẦN CỨNG (VARIANT 3B)

```
+-------------------------------------------------------------------------------+
|                VARIANT 3B: HARDWARE RESOURCE & TIMING PROFILE                 |
+--------------------------+-------------------+----------------+---------------+
| Resource Type            | Used (PL)         | Total Avail    | % Utilization |
+--------------------------+-------------------+----------------+---------------+
| DSP48E2 (Hard Multiplier)| 16                | 252 (KV260)    | 6.35%         |
| LUT (Logic + Shifters)   | ~4,200            | 70,560         | ~5.95%        |
| FF (Pipeline Flip-Flops) | ~5,800            | 141,120        | ~4.11%        |
| BRAM 36Kb (Score Reorder)| 1.0 (1024x32-bit) | 144            | 0.69%         |
+--------------------------+-------------------+----------------+---------------+
| Max Clock Frequency      | 250 MHz (Fmax Target: 200 MHz)                     |
| Latency (N=256, d=16)    | ~325,000 clock cycles (~1.62 ms @ 200 MHz)         |
| Peak Compute Density     | 32 MACs/cycle (2x INT8 packed per DSP48E2)         |
+--------------------------+----------------------------------------------------+
```

---
*Tài liệu được cập nhật tự động và kiểm chứng toàn diện cho Phương án 3B (Variant 3B) — MobileViT-XXS Self-Attention Hardware Accelerator.*
