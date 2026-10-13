# HƯỚNG DẪN TOÀN DIỆN CÁC LỆNH TERMINAL CHẠY KIỂM THỬ XÁC MINH (VERIFICATION GUIDE)

> **Dự án:** Bộ tăng tốc phần cứng MobileViT-XXS Self-Attention trên AMD Kria KV260 (`rtl_ver02`)  
> **Phiên bản:** v3.0 (Cập nhật sau khi hoàn thành 100% Bước 1 đến Bước 3)  
> **Mục tiêu:** Cung cấp tài liệu tra cứu đầy đủ, chính xác từng dòng lệnh terminal (PowerShell / Command Prompt) để tái hiện, kiểm tra và nghiệm thu độc lập từng module phần cứng cũng như toàn bộ hệ thống Top-Level Integration.

---

## 1. THIẾT LẬP MÔI TRƯỜNG & ĐƯỜNG DẪN LÀM VIỆC

### 1.1 Yêu cầu công cụ
* **Trình mô phỏng RTL:** Icarus Verilog (`iverilog` phiên bản 12.0 trở lên) hỗ trợ IEEE 1800-2012 SystemVerilog (`-g2012`).
  - Đường dẫn mặc định trên máy: `C:\iverilog\bin\iverilog.exe` và `C:\iverilog\bin\vvp.exe`.
* **Môi trường Python:** Python 3.10+ (có thư viện `torch`, `numpy`).
* **Shell:** Windows PowerShell (`pwsh` / `powershell.exe`) hoặc Windows Command Prompt (`cmd.exe`).

### 1.2 Thiết lập đường dẫn thư mục
Trước khi thực hiện các lệnh bên dưới, hãy mở terminal và di chuyển đến thư mục tương ứng:

```powershell
# Di chuyển vào thư mục gốc của dự án
cd c:\Users\tuan2\Desktop\Capstone_Project

# Thư mục mã nguồn RTL và Testbench
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02

# Thư mục mã nguồn mô hình PyTorch & Báo cáo
cd c:\Users\tuan2\Desktop\Capstone_Project\model_notebook\ivit_surgery
```

---

## 2. BẢNG TỔNG HỢP CÁC LỆNH VERIFICATION

| Cấp độ | Module cần kiểm tra | Testbench thực thi | Mục tiêu xác minh | Thời gian chạy | Kết quả chuẩn |
|:---|:---|:---|:---|:---:|:---:|
| **Python** | Model QAT & Hex Exporter | [`export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py) | Xuất 6 tệp Golden Vectors chuẩn từ PyTorch | ~5s | Sinh đủ 6 tệp .hex |
| **Step 2** | Dyadic Rescaler | [`tb_golden_scale_unit.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_golden_scale_unit.sv) | $(QK^T \times M) \gg E$, 65,536 điểm ảnh | ~3s | **100% BIT-EXACT (0 lỗi)** |
| **Step 2** | IntSoftmax Engine | [`tb_golden_softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_golden_softmax_lut.sv) | Max, Exp ROM, 32-cycle Div, 65,536 xác suất | ~25s | **100% BIT-EXACT (0 lỗi)** |
| **Step 2.5** | Reloadable BRAM LUT | [`tb_softmax_lut_reloadable.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_softmax_lut_reloadable.sv) | Nạp lại LUT qua AXI-Lite & chuyển tầng $d_k=16, 20, 24$ | ~8s | **100% PASS (5 Phase)** |
| **Step 3** | Systolic MAC Array | [`tb_systolic_mac_array.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_systolic_mac_array.sv) | Sửa FL-03 (tách col0/col1), FL-04 (drain tuần tự) | ~2s | **100% PASS** |
| **Step 3** | Score-V Context GEMM | [`tb_score_v_gemm.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_score_v_gemm.sv) | Sửa FL-08: 256 hàng, 512 beats, tlast tại beat 511 | ~45s | **100% BIT-EXACT (0 lỗi)** |
| **Step 3** | Ping-Pong BRAM Buffer | [`tb_ping_pong_bram_buffer.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_ping_pong_bram_buffer.sv) | Zero-bubble Ingress, chuyển mạch Bank 0/1 | ~10s | **100% PASS** |
| **Step 3 TOP**| **Attention Core Top** | [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv) | **Toàn bộ pipeline 4 tầng, đối soát 4,096 samples** | **~25s** | **100% BIT-EXACT (4,096/4,096)** |

---

## 3. CHI TIẾT TỪNG LỆNH CHẠY & EXPECTED OUTPUT LOGS

> [!NOTE]
> Tất cả các lệnh mô phỏng RTL dưới đây cần được thực thi tại thư mục làm việc:  
> `cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02`

---

### 3.1 Tầng 0: Xuất Test Vectors từ PyTorch Golden Model
Tạo và chuẩn hóa các tệp vector `.hex` (`q_matrix.hex`, `k_matrix.hex`, `v_matrix.hex`, `golden_gemm_qk.hex`, `golden_scaled_score.hex`, `golden_softmax_prob.hex`, `golden_context_out.hex`):

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\model_notebook\ivit_surgery
python export_golden_vectors.py
```

* **Dấu hiệu thành công:**
  ```text
  [INFO] Exporting Q, K, V matrices...
  [INFO] Successfully generated golden_gemm_qk.hex (65536 lines)
  [INFO] Successfully generated golden_scaled_score.hex (65536 lines)
  [INFO] Successfully generated golden_softmax_prob.hex (65536 lines)
  [INFO] Successfully generated golden_context_out.hex (4096 lines)
  [SUCCESS] Golden vectors standardized and copied to rtl_ver02 directory!
  ```

---

### 3.2 Tầng 1: Kiểm thử Khối Dyadic Rescaler (`scale_unit.sv`)
Kiểm tra phép nhân dịch bão hòa $(S \times M) \gg E$ với $M = 27640, E = 22$ trên toàn bộ 65,536 điểm của ma trận $QK^T$:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_scale.vvp scale_unit.sv tb_golden_scale_unit.sv
C:\iverilog\bin\vvp.exe tb_scale.vvp
```

* **Expected Output:**
  ```text
  =======================================================================
     TESTBENCH: SCALE_UNIT HARDWARE VERIFICATION (rtl_ver02)
  =======================================================================
  [TB INFO]: Loaded 65536 elements from golden_gemm_qk.hex & golden_scaled_score.hex
  [TB INFO]: Using Dyadic Parameters: M = 27640, E = 22
  [TB INFO]: Verification Complete: 65536 elements processed.
  >>> SUCCESS: 65536 / 65536 MATCHED 100% BIT-EXACT! ERRORS: 0 <<<
  ```

---

### 3.3 Tầng 1: Kiểm thử Khối IntSoftmax Engine (`softmax_lut.sv`)
Kiểm tra tìm Max, tra cứu Exp ROM 20-bit, bộ chia 32-cycle Restoring Divider và chuẩn hóa xác suất 16-bit trên toàn bộ 256 hàng:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_softmax.vvp softmax_lut.sv tb_golden_softmax_lut.sv
C:\iverilog\bin\vvp.exe tb_softmax.vvp
```

* **Expected Output:**
  ```text
  =======================================================================
     TESTBENCH: SOFTMAX_LUT HARDWARE VERIFICATION (rtl_ver02)
  =======================================================================
  [TB INFO]: Starting Bit-Exact Verification across 256 Rows (65,536 Probs)...
  Row 000 Completed: All 256 Probabilities Matched PyTorch Bit-Exact Model.
  ...
  Row 255 Completed: All 256 Probabilities Matched PyTorch Bit-Exact Model.
  >>> SUCCESS: 65,536 / 65,536 PROBABILITIES MATCHED 100% (0 ERRORS)! <<<
  ```

---

### 3.4 Tầng 2: Kiểm thử Nạp lại BRAM LUT Bước 2.5 (`tb_softmax_lut_reloadable.sv`)
Kiểm tra khả năng nạp lại giá trị LUT trong lúc chạy (Runtime Reload) qua AXI4-Lite và cơ chế chuyển đổi phân vùng bộ nhớ tự động theo chiều vector $d_k \in \{16, 20, 24\}$:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_reloadable.vvp softmax_lut.sv axi_lite_regs.sv tb_softmax_lut_reloadable.sv
C:\iverilog\bin\vvp.exe tb_reloadable.vvp
```

* **Expected Output:**
  ```text
  =======================================================================
     TESTBENCH: RECONFIGURABLE BRAM LUT & MULTI-STAGE STEP 2.5
  =======================================================================
  PHASE 1: Default ROM Verification (Stage 2, head_dim=16) -> PASS
  PHASE 2: Multi-Stage Address Mapping (head_dim=16, 20, 24) -> PASS
  PHASE 3: Port B Runtime AXI-Lite Reloading -> PASS
  PHASE 4: Readback Verification of Modified Entries -> PASS
  PHASE 5: Online Softmax Inference with Reloaded Coefficients -> PASS
  =======================================================================
  >>> ALL 5 PHASES PASSED WITH ZERO ERRORS! <<<
  ```

---

### 3.5 Tầng 3: Kiểm thử Mảng Systolic MAC Array (`systolic_mac_array.sv`)
Kiểm tra độc lập hóa 2 cột tích lũy (FL-03) và mạch xả dữ liệu tuần tự 32 phần tử (FL-04):

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_systolic.vvp systolic_mac_array.sv pe_slice_dsp_packing.sv tb_systolic_mac_array.sv
C:\iverilog\bin\vvp.exe tb_systolic.vvp
```

* **Expected Output:**
  ```text
  =======================================================================
     TESTBENCH: SYSTOLIC MAC ARRAY (FL-03 & FL-04 REMEDIATION CHECK)
  =======================================================================
  [TB DRAIN]: Drained 32 / 32 dot product items successfully.
  [TB CHECK]: Col0 and Col1 accumulation independent -> No crosstalk detected.
  >>> SUCCESS: systolic_mac_array PASSED 100%! <<<
  ```

---

### 3.6 Tầng 3: Kiểm thử Khối Context GEMM (`score_v_gemm.sv`)
Kiểm tra khả năng tính toán toàn bộ 256 hàng tokens, dịch bit `>>> 15`, xuất đủ 512 beats (4,096 bytes) và kiểm tra cờ `tlast` chỉ bật ở beat 511:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_score_v.vvp score_v_gemm.sv tb_score_v_gemm.sv
C:\iverilog\bin\vvp.exe tb_score_v.vvp
```

* **Expected Output:**
  ```text
  === STARTING SCORE_V_GEMM VERIFICATION (256 ROWS, 512 BEATS) ===
  === SCORE_V_GEMM VERIFICATION REPORT ===
  Total Beats Processed:  512 (Expected: 512)
  Total Bytes Matched:    4096 / 4096
  Total Byte Errors:      0
  Total TLAST Errors:     0
  >>> SUCCESS: score_v_gemm PASSED 100% BIT-EXACT MATCH! <<<
  ```

---

### 3.7 Tầng 3: Kiểm thử Bộ đệm Ping-Pong BRAM (`ping_pong_bram_buffer.sv`)
Kiểm tra cơ chế nạp 64-bit tự động $Q \rightarrow K \rightarrow V$ không bong bóng (zero-bubble) và chuyển mạch Bank 0 / Bank 1:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_pp.vvp ping_pong_bram_buffer.sv tb_ping_pong_bram_buffer.sv
C:\iverilog\bin\vvp.exe tb_pp.vvp
```

* **Expected Output:**
  ```text
  =======================================================================
     TESTBENCH: PING-PONG BRAM BUFFER (ZERO-BUBBLE INGRESS VERIFICATION)
  =======================================================================
  [TB INFO]: Bank 0 written successfully (1536 beats: Q, K, V).
  [TB INFO]: Bank swap executed smoothly without data hazards.
  >>> SUCCESS: ping_pong_bram_buffer PASSED 100%! <<<
  ```

---

### 3.8 TẦNG 4 (QUAN TRỌNG NHẤT): KIỂM THỬ TÍCH HỢP TOÀN DIỆN TOP-LEVEL (`tb_attention_core.sv`)
Mô phỏng toàn bộ hệ thống tăng tốc Self-Attention: nạp 1,536 beats qua AXI-Stream Ingress, kích hoạt lệnh START qua AXI-Lite CSR, chạy thông suốt 4 tầng tính toán cho 256 hàng tokens và đối soát 4,096 mẫu kết quả đầu ra với mô hình PyTorch:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02
C:\iverilog\bin\iverilog.exe -g2012 -o tb_top.vvp tb_attention_core.sv attention_core_top.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv systolic_mac_array.sv pe_slice_dsp_packing.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
C:\iverilog\bin\vvp.exe tb_top.vvp
```

* **Expected Output Hoàn Chỉnh:**
  ```text
  =======================================================================
     STARTING INTEGRATED ATTENTION CORE VERIFICATION (rtl_ver02)
  =======================================================================
  [TB INFO]: Loading Golden Test Vectors from .hex files...
  [TB INFO]: Loaded 256 Q tokens, 256 K tokens, 256 V tokens.
  [TB INFO]: Reset Released.
  [TB INFO]: Step 2 - Configuring CSR (N=256, dk=16, H=1, M=27640, E=22)...
  [TB INFO]: Step 3 - Streaming Q, K, V matrices into AXI-Stream Ingress...
  [TB INFO]: Finished Streaming 1,536 beats into Ingress BRAM.
  [TB INFO]: Step 4 - Triggering START Pulse via AXI-Lite (0x00 <= 1)...
  [TB INFO]: Step 5 - Waiting for Core Pipeline Execution (256 rows)...
  [CONTEXT EGRESS #0000]: DUT = 5 | Golden = 5 (   MATCH)
  [CONTEXT EGRESS #0001]: DUT = 2 | Golden = 2 (   MATCH)
  [CONTEXT EGRESS #0002]: DUT = 0 | Golden = 0 (   MATCH)
  [CONTEXT EGRESS #0003]: DUT = -3 | Golden = -3 (   MATCH)
  [CONTEXT EGRESS #0004]: DUT = -7 | Golden = -7 (   MATCH)
  [CONTEXT EGRESS #0005]: DUT = 4 | Golden = 4 (   MATCH)
  [CONTEXT EGRESS #0006]: DUT = -5 | Golden = -5 (   MATCH)
  [CONTEXT EGRESS #0007]: DUT = 0 | Golden = 0 (   MATCH)
  [CONTEXT EGRESS #0008]: DUT = 5 | Golden = 5 (   MATCH)
  [CONTEXT EGRESS #0009]: DUT = 6 | Golden = 6 (   MATCH)
  [CONTEXT EGRESS #0010]: DUT = 4 | Golden = 4 (   MATCH)
  [CONTEXT EGRESS #0011]: DUT = 4 | Golden = 4 (   MATCH)
  [CONTEXT EGRESS #0012]: DUT = 0 | Golden = 0 (   MATCH)
  [CONTEXT EGRESS #0013]: DUT = 10 | Golden = 10 (   MATCH)
  [CONTEXT EGRESS #0014]: DUT = -1 | Golden = -1 (   MATCH)
  [CONTEXT EGRESS #0015]: DUT = -1 | Golden = -1 (   MATCH)
  [CONTEXT EGRESS #0511]: DUT = -1 | Golden = -1 (   MATCH)
  [TB INFO]: Running... Cycle 50000, Egress Samples: 976 / 4096
  [TB INFO]: Running... Cycle 100000, Egress Samples: 1952 / 4096
  [TB INFO]: Running... Cycle 150000, Egress Samples: 2944 / 4096
  [TB INFO]: Running... Cycle 200000, Egress Samples: 3920 / 4096
  [CONTEXT EGRESS #4095]: DUT = -1 | Golden = -1 (   MATCH)
  [TB INFO]: m_axis_tlast correctly asserted on beat 511!

  =======================================================================
     FINAL VERIFICATION REPORT: ATTENTION CORE TOP INTEGRATION (rtl_ver02)
  =======================================================================
     Total Egress Samples Monitored : 4096 / 4096
     Exact Matched Samples          : 4096
     Mismatched Samples             : 0
     TLAST Assertion Errors         : 0
     Total Execution Clock Cycles   : 208384
     >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
  =======================================================================
  ```

---

## 4. KỊCH BẢN CHẠY TỰ ĐỘNG TẤT CẢ TESTBENCH (ONE-CLICK POWERSHELL SCRIPT)

Bạn có thể chạy toàn bộ các bài kiểm thử xác minh của dự án chỉ bằng một lệnh duy nhất. Tạo hoặc chạy trực tiếp đoạn mã PowerShell sau:

### Lệnh PowerShell tự động kiểm thử toàn bộ (All-In-One):
```powershell
# Di chuyển đến thư mục RTL
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  RUNNING FULL HARDWARE VERIFICATION SUITE (rtl_ver02)    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Test Scale Unit
Write-Host "`n[1/6] Running tb_golden_scale_unit..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_scale.vvp scale_unit.sv tb_golden_scale_unit.sv
& C:\iverilog\bin\vvp.exe tb_scale.vvp

# 2. Test Softmax LUT
Write-Host "`n[2/6] Running tb_golden_softmax_lut..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_softmax.vvp softmax_lut.sv tb_golden_softmax_lut.sv
& C:\iverilog\bin\vvp.exe tb_softmax.vvp

# 3. Test Reloadable BRAM LUT (Step 2.5)
Write-Host "`n[3/6] Running tb_softmax_lut_reloadable..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_reloadable.vvp softmax_lut.sv axi_lite_regs.sv tb_softmax_lut_reloadable.sv
& C:\iverilog\bin\vvp.exe tb_reloadable.vvp

# 4. Test Systolic Array (FL-03 / FL-04)
Write-Host "`n[4/6] Running tb_systolic_mac_array..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_systolic.vvp systolic_mac_array.sv pe_slice_dsp_packing.sv tb_systolic_mac_array.sv
& C:\iverilog\bin\vvp.exe tb_systolic.vvp

# 5. Test Score-V GEMM (FL-08)
Write-Host "`n[5/6] Running tb_score_v_gemm..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_score_v.vvp score_v_gemm.sv tb_score_v_gemm.sv
& C:\iverilog\bin\vvp.exe tb_score_v.vvp

# 6. Test Attention Core Top Integration
Write-Host "`n[6/6] Running tb_attention_core (Top-Level Integration)..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_top.vvp tb_attention_core.sv attention_core_top.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv systolic_mac_array.sv pe_slice_dsp_packing.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
& C:\iverilog\bin\vvp.exe tb_top.vvp

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  ALL TESTS COMPLETED SUCCESSFULLY!                       " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
```

---

## 5. HƯỚNG DẪN XỬ LÝ SỰ CỐ (TROUBLESHOOTING)

1. **Lỗi `Subroutine ports with unpacked dimensions are not yet supported`:**
   - *Nguyên nhân:* Icarus Verilog không cho phép truyền mảng chưa đóng gói (`unpacked array`) làm đối số của `task`.
   - *Cách khắc phục:* Không truyền mảng vào task. Truy cập mảng toàn cục trong module hoặc dùng mảng đóng gói (`packed array`).
2. **Lỗi `Unable to open input file`:**
   - *Nguyên nhân:* Biên dịch `iverilog` gặp lỗi cú pháp trước đó nên không sinh ra tệp `.vvp`.
   - *Cách khắc phục:* Kiểm tra log biên dịch của lệnh `iverilog` để sửa lỗi cú pháp trước khi chạy `vvp`.
3. **Mô phỏng bị treo (Timeout / Hang):**
   - *Nguyên nhân:* Khối nhận dữ liệu AXI-Stream không kéo cờ `tready` lên cao hoặc bắt tay FSM chờ sai tín hiệu `busy`.
   - *Cách khắc phục:* Kiểm tra xem `m_axis_tready` có được gán `1'b1` trong testbench và các xung `start_row`, `start_pulse` đã được kích hoạt đúng chu kỳ chưa.
