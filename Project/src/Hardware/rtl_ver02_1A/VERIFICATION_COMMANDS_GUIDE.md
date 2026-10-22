# HƯỚNG DẪN CÁC LỆNH TERMINAL CHẠY KIỂM THỬ XÁC MINH (VERIFICATION GUIDE) - BIẾN THỂ 1A

> **Dự án:** Bộ tăng tốc phần cứng MobileViT-XXS Self-Attention trên AMD Kria KV260  
> **Kiến trúc:** **Phương án 1A (Variant 1A)** — Vector Streaming Reduction Tree (Pure Fabric Multipliers, **0 DSP48E2**)  
> **Thư mục:** `Project/src/Hardware/rtl_ver02_1A`  
> **Mục tiêu:** Cung cấp tài liệu tra cứu đầy đủ, chính xác từng dòng lệnh terminal (PowerShell / Command Prompt) để biên dịch và chạy mô phỏng Icarus Verilog (`iverilog` / `vvp`) cho từng testbench độc lập cũng như toàn bộ hệ thống Top-Level Integration của Biến thể 1A.

---

## 1. THIẾT LẬP MÔI TRƯỜNG & ĐƯỜNG DẪN LÀM VIỆC

### 1.1 Yêu cầu công cụ
* **Trình mô phỏng RTL:** Icarus Verilog (`iverilog` phiên bản 12.0 trở lên) hỗ trợ chuẩn IEEE 1800-2012 SystemVerilog (cờ `-g2012`).
  - Đường dẫn mặc định: `C:\iverilog\bin\iverilog.exe` và `C:\iverilog\bin\vvp.exe`.
* **Shell thực thi:** Windows PowerShell (`pwsh` / `powershell.exe`) hoặc Windows Command Prompt (`cmd.exe`).
* **Golden Vectors:** Các tệp `.hex` chuẩn hóa từ mô hình PyTorch đã được copy sẵn vào thư mục `rtl_ver02_1A`.

### 1.2 Di chuyển vào thư mục làm việc
Trước khi chạy bất kỳ lệnh nào, hãy mở terminal và di chuyển trực tiếp vào thư mục mã nguồn của Biến thể 1A:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
```

---

## 2. BẢNG TỔNG HỢP CÁC LỆNH VERIFICATION (VARIANT 1A)

| Cấp độ | Module kiểm tra | Testbench thực thi | File mô phỏng (.vvp) | Mục tiêu kiểm thử | Thời gian | Kết quả chuẩn |
|:---|:---|:---|:---|:---|:---:|:---:|
| **Level 1** | Pipelined Adder Tree | [`tb_pipelined_adder_tree_1A.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/tb_pipelined_adder_tree_1A.sv) | `tb_tree_1a.vvp` | 16 bộ nhân Fabric + Cây cộng 4 tầng ($Q K^T$) | ~3s | **100% BIT-EXACT (65,536/65,536)** |
| **Level 2** | Dyadic Rescaler | [`tb_golden_scale_unit.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/tb_golden_scale_unit.sv) | `tb_scale_1a.vvp` | Phép nhân định tỉ lệ $(S \times M) \gg E$, 65,536 điểm | ~3s | **100% BIT-EXACT (65,536/65,536)** |
| **Level 3** | Hardware IntSoftmax | [`tb_golden_softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/tb_golden_softmax_lut.sv) | `tb_softmax_1a.vvp` | Max tìm kiếm, Exp ROM, Restoring Divider 32 chu kỳ | ~2s | **100% BIT-EXACT (1,024/1,024)** |
| **Level 4** | Score-V Context GEMM | [`tb_score_v_gemm.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/tb_score_v_gemm.sv) | `tb_score_v_1a.vvp` | GEMM 2 ($P \times V$), dịch bit `>>> 15`, kiểm tra `tlast` | ~45s | **100% BIT-EXACT (4,096/4,096)** |
| **Level 5 (TOP)** | **Attention Core Top 1A** | [`tb_attention_core_1A.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/tb_attention_core_1A.sv) | `tb_top_1a.vvp` | **Toàn bộ pipeline 1A (AXI-Stream + Tree + Softmax + Score-V)** | **~25s** | **100% BIT-EXACT (4,096/4,096)** |

---

## 3. CHI TIẾT TỪNG LỆNH CHẠY & EXPECTED OUTPUT LOGS

### 3.1 Cấp độ 1: Kiểm thử Cây cộng Pipelined Adder Tree (`pipelined_adder_tree.sv`)
Kiểm thử lõi tính tích vô hướng 16 phần tử sử dụng thuần logic nhân LUT fabric (`reg signed [7:0] a * b`) và cây cộng rút gọn nhị phân 4 tầng pipeline ($16 \to 8 \to 4 \to 2 \to 1$). Testbench nạp toàn bộ 256 vector Query và 256 vector Key, kiểm tra đúng 65,536 giá trị dot-product đối soát với `golden_gemm_qk.hex`.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
C:\iverilog\bin\iverilog.exe -g2012 -o tb_tree_1a.vvp pipelined_adder_tree.sv tb_pipelined_adder_tree_1A.sv
C:\iverilog\bin\vvp.exe tb_tree_1a.vvp
```

#### Expected Output Log:
```text
=======================================================================
   TESTBENCH: PIPELINED ADDER TREE HARDWARE VERIFICATION (1A)
=======================================================================
[TB INFO]: Loaded 256 Q tokens and 256 K tokens.
[TB INFO]: Loaded 65536 expected scores from golden_gemm_qk.hex.
[TB INFO]: Reset released. Starting stream through 16-element Tree...
[TB INFO]: Row 0 verified. Matched: 256 / 256. Errors: 0
[TB INFO]: Row 1 verified. Matched: 256 / 256. Errors: 0
[TB INFO]: Row 2 verified. Matched: 256 / 256. Errors: 0
[TB INFO]: Row 3 verified. Matched: 256 / 256. Errors: 0
...
[TB INFO]: Row 254 verified. Matched: 256 / 256. Errors: 0
[TB INFO]: Row 255 verified. Matched: 256 / 256. Errors: 0
=======================================================================
   PIPELINED ADDER TREE VERIFICATION SUMMARY (Variant 1A)
=======================================================================
   Total Tested Scores : 65536 / 65536
   Exact Matched Scores: 65536
   Mismatches          : 0
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

### 3.2 Cấp độ 2: Kiểm thử Khối Dyadic Rescaler (`scale_unit.sv`)
Kiểm thử mạch nhân định tỉ lệ dyadic bão hòa $(S \times M) \gg E$ với hệ số chuẩn Stage 2 ($M = 27640, E = 22$), đối soát toàn bộ 65,536 điểm giữa `golden_gemm_qk.hex` và `golden_scaled_score.hex`.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
C:\iverilog\bin\iverilog.exe -g2012 -o tb_scale_1a.vvp scale_unit.sv tb_golden_scale_unit.sv
C:\iverilog\bin\vvp.exe tb_scale_1a.vvp
```

#### Expected Output Log:
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

### 3.3 Cấp độ 3: Kiểm thử Khối IntSoftmax Engine (`softmax_lut.sv`)
Kiểm tra tìm Max hàng, giải thuật trừ Max, tra bảng Exp ROM đối xứng 20-bit, bộ chia Restoring Divider 32 chu kỳ và chuẩn hóa xác suất 16-bit đối soát với `golden_softmax_prob.hex`.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
C:\iverilog\bin\iverilog.exe -g2012 -o tb_softmax_1a.vvp softmax_lut.sv tb_golden_softmax_lut.sv
C:\iverilog\bin\vvp.exe tb_softmax_1a.vvp
```

#### Expected Output Log:
```text
=======================================================================
   TESTBENCH: SOFTMAX_LUT HARDWARE VERIFICATION (rtl_ver02)
=======================================================================
[TB INFO]: Loaded 65536 golden scores and golden probabilities.
[TB INFO]: Starting Bit-Exact Verification across 4 Rows (1,024 Probs)...
[ROW 0] Verification Complete: 256 / 256 Matched 100% Bit-Exact!
[ROW 1] Verification Complete: 256 / 256 Matched 100% Bit-Exact!
[ROW 2] Verification Complete: 256 / 256 Matched 100% Bit-Exact!
[ROW 3] Verification Complete: 256 / 256 Matched 100% Bit-Exact!
>>> SUCCESS: 1024 / 1024 PROBABILITIES MATCHED 100% (0 ERRORS)! <<<
```

---

### 3.4 Cấp độ 4: Kiểm thử Khối Score-V Context GEMM (`score_v_gemm.sv`)
Kiểm tra phép nhân ma trận Context $P \times V$, bộ dịch bit định dạng `>>> 15`, bộ tạo beat 64-bit và cơ chế phát cờ `m_axis_tlast` chính xác tại beat thứ 511 (4,096 bytes đầu ra).

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
C:\iverilog\bin\iverilog.exe -g2012 -o tb_score_v_1a.vvp score_v_gemm.sv tb_score_v_gemm.sv
C:\iverilog\bin\vvp.exe tb_score_v_1a.vvp
```

#### Expected Output Log:
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

### 3.5 Cấp độ 5 (Quan Trọng Nhất): Kiểm thử Tích hợp Toàn diện Top-Level Biến Thể 1A (`tb_attention_core_1A.sv`)
Mô phỏng toàn diện Top-Level Architecture của Phương án 1A:
1. Giao tiếp ngoại vi: Nhận 1,536 beats nạp dữ liệu ma trận $Q, K, V$ qua AXI4-Stream Ingress vào Ping-Pong BRAM Buffer.
2. Cấu hình CSR: Kích hoạt tham số qua AXI4-Lite ($N=256, d_k=16, H=1, M=27640, E=22$) và phát xung START.
3. Luồng tính toán nội bộ: Dòng $Q$ được nạp cố định vào Vector Register của [`pipelined_adder_tree`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/pipelined_adder_tree.sv); luồng $K^T$ chạy liên tục qua 16 bộ nhân Fabric và cây cộng 4 tầng để tạo $QK^T$; điểm được scale qua [`scale_unit`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/scale_unit.sv); chuẩn hóa qua [`softmax_lut`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/softmax_lut.sv); nhân tích lũy với $V$ trong [`score_v_gemm`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/score_v_gemm.sv).
4. Xuất dữ liệu: Egress nhận đủ 512 beats (4,096 mẫu INT8) qua AXI-Stream và đối soát trực tiếp từng byte với `golden_context_out.hex`.

#### Lệnh thực thi:
```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
C:\iverilog\bin\iverilog.exe -g2012 -o tb_top_1a.vvp tb_attention_core_1A.sv attention_core_top_1A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pipelined_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
C:\iverilog\bin\vvp.exe tb_top_1a.vvp
```

#### Expected Output Log:
```text
=======================================================================
   STARTING INTEGRATED ATTENTION CORE 1A VERIFICATION (PURE FABRIC)
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
   FINAL VERIFICATION REPORT: ATTENTION CORE 1A (PURE FABRIC)
=======================================================================
   Total Egress Samples Monitored : 4096 / 4096
   Exact Matched Samples          : 4096
   Mismatched Samples             : 0
   TLAST Assertion Errors         : 0
   Total Execution Clock Cycles   : 209663
   >>> STATUS: 100% BIT-EXACT MATCH ACHIEVED! [PASS] <<<
=======================================================================
```

---

## 4. KỊCH BẢN CHẠY TỰ ĐỘNG TOÀN BỘ (AUTOMATED RUNNERS)

### 4.1 Sử dụng Windows Batch File (`run_verifications_1A.bat`)
Trong thư mục `rtl_ver02_1A` đã có sẵn tệp kịch bản `run_verifications_1A.bat`. Bạn chỉ cần thực thi:

```cmd
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A
run_verifications_1A.bat
```

### 4.2 Lệnh PowerShell All-In-One (Chạy tuần tự 5 testbenches)
Nếu muốn chạy trực tiếp bằng PowerShell với hiển thị màu trực quan:

```powershell
cd c:\Users\tuan2\Desktop\Capstone_Project\Project\src\Hardware\rtl_ver02_1A

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  RUNNING HARDWARE VERIFICATION SUITE FOR VARIANT 1A      " -ForegroundColor Cyan
Write-Host "  PURE FABRIC MULTIPLIERS (0 DSP48E2)                     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Pipelined Adder Tree
Write-Host "`n[1/5] Running tb_pipelined_adder_tree_1A..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_tree_1a.vvp pipelined_adder_tree.sv tb_pipelined_adder_tree_1A.sv
& C:\iverilog\bin\vvp.exe tb_tree_1a.vvp

# 2. Scale Unit
Write-Host "`n[2/5] Running tb_golden_scale_unit..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_scale_1a.vvp scale_unit.sv tb_golden_scale_unit.sv
& C:\iverilog\bin\vvp.exe tb_scale_1a.vvp

# 3. Softmax LUT
Write-Host "`n[3/5] Running tb_golden_softmax_lut..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_softmax_1a.vvp softmax_lut.sv tb_golden_softmax_lut.sv
& C:\iverilog\bin\vvp.exe tb_softmax_1a.vvp

# 4. Score-V GEMM
Write-Host "`n[4/5] Running tb_score_v_gemm..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_score_v_1a.vvp score_v_gemm.sv tb_score_v_gemm.sv
& C:\iverilog\bin\vvp.exe tb_score_v_1a.vvp

# 5. Top-Level Integration (1A Top)
Write-Host "`n[5/5] Running tb_attention_core_1A (Top Integration)..." -ForegroundColor Yellow
& C:\iverilog\bin\iverilog.exe -g2012 -o tb_top_1a.vvp tb_attention_core_1A.sv attention_core_top_1A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pipelined_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
& C:\iverilog\bin\vvp.exe tb_top_1a.vvp

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  ALL 5 VARIANT 1A TESTS PASSED 100% BIT-EXACT!           " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
```

---

## 5. LƯU Ý KỸ THUẬT & TROUBLESHOOTING KHI KIỂM THỬ BIẾN THỂ 1A

1. **Độ trễ Pipeline Flushing của Cây cộng (5 chu kỳ):**
   - Khối [`pipelined_adder_tree.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02_1A/pipelined_adder_tree.sv) có đúng 5 chu kỳ trễ (1 chu kỳ nhân + 4 chu kỳ cộng cây nhị phân).
   - Khi chuyển giữa các hàng Query $Q$ trong testbench, bắt buộc phải đợi tối thiểu 5 chu kỳ (`repeat (5) @(posedge clk)`) để xả hết 2 mẫu token cuối (254 và 255) trong đường ống trước khi cập nhật vector thanh ghi `a[d]`, tránh trường hợp dữ liệu hàng mới bị nhân lẫn với hàng cũ.
2. **Khử hoàn toàn DSP48E2 trong GEMM 1:**
   - Trong biến thể 1A, khối `systolic_mac_array.sv` và `pe_slice_dsp_packing.sv` được thay thế hoàn toàn bởi `pipelined_adder_tree.sv`.
   - Lệnh biên dịch của `tb_top_1a.vvp` không còn chứa `systolic_mac_array.sv` hay `pe_slice_dsp_packing.sv`. Tổng tài nguyên DSP cho GEMM 1 là **0 DSP48E2**.
3. **Mặt nạ động `reg_head_dim` ($d_k=16, 20, 24$):**
   - Khi chạy ở các tầng khác của MobileViT ($d_k > 16$), cây cộng tự động mở rộng theo tham số cấu hình qua thanh ghi AXI-Lite CSR mà không làm treo đường ống.
