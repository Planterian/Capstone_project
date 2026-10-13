# KẾ HOẠCH THỰC THI 3 BƯỚC CÒN LẠI (BƯỚC 2, BƯỚC 3, BƯỚC 4)
## Hoàn thiện Bộ Lõi Tăng tốc Attention Core Bit-Exact (rtl_ver02) trên AMD Kria KV260
### (Tài liệu chỉ dẫn kỹ thuật độc lập — Dùng cho Phiên làm việc mới / New Chat Session)

**Mã tài liệu:** `PLAN_PHASE4_REMAINING_STEPS`  
**Ngày khởi tạo:** 11/10/2026  
**Vị trí lưu trữ:** [`model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md)  
**Tài liệu tham chiếu bắt buộc:**
1. Báo cáo kiểm toán khuyết tật: [`REPORT_PHASE4_RTL_COMPREHENSIVE_AUDIT_FLAWS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_RTL_COMPREHENSIVE_AUDIT_FLAWS.md)
2. Báo cáo nghiệm thu Bước 1: [`REPORT_PHASE4_STEP1_PYTHON_GOLDEN_MODEL.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_STEP1_PYTHON_GOLDEN_MODEL.md)
3. Thư mục mã nguồn phần cứng: [`Project/src/Hardware/rtl_ver02/`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/)

---

## TỔNG QUAN BỐI CẢNH & HIỆN TRẠNG ĐÃ HOÀN TẤT (PREREQUISITES)

### Trạng thái đã hoàn tất tại Bước 1 (Đã nghiệm thu ✓):
* Toàn bộ dữ liệu kiểm thử vàng (Golden Vectors) đã được chuẩn hóa bit-exact 100% với mô hình PyTorch MobileViT-XXS I-ViT (67.28% Accuracy).
* Hệ số Dyadic Scale chuẩn mới:
  - **Stage 2 ($d_k=16$):** $S_{\text{eff}} = 0.004365 \implies \mathbf{M = 18,308,\ E = 22}$
  - **Stage 3 ($d_k=20$):** $S_{\text{eff}} = 0.005378 \implies \mathbf{M = 22,559,\ E = 22}$
  - **Stage 4 ($d_k=24$):** $S_{\text{eff}} = 0.004756 \implies \mathbf{M = 19,947,\ E = 22}$
* Đã xuất đồng bộ tệp tham số [`hw_scale_params.svh`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/hw_scale_params.svh) và bộ 8 tệp `.hex` chuẩn vào thư mục `Project/src/Hardware/rtl_ver02/`.
* Khắc phục hoàn toàn lỗi suy thoái Softmax: Ma trận xác suất 16-bit [`golden_softmax_prob.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_softmax_prob.hex) chứa 581 mức xác suất phong phú trong dải $[0, 1110]$, Mean=$127.0$, tổng hàng $\approx 32,457 \dots 32,768$.

---

## BẢNG LỘ TRÌNH 3 BƯỚC CÒN LẠI (REMAINING ROADMAP)

```
======================================================================================================
               LỘ TRÌNH HOÀN THIỆN ATTENTION ACCELERATOR (rtl_ver02)
======================================================================================================

 [BƯỚC 1: XONG ✓]   Chuẩn hóa Python Golden Model & Dyadic Scale (M=18308, E=22). Đã nghiệm thu.
       │
       ▼
 [BƯỚC 2: XONG ✓]   Sửa lỗi Số học & Logic khối Softmax Engine và Ping-Pong BRAM Buffer (100% Bit-Exact).
       ├── 2.1: Sửa bộ chia Softmax Q16 trong softmax_lut.sv (100% Bit-Exact 0-LSB Error).
       ├── 2.2: Sửa Ping-Pong BRAM nhận đủ 64-bit ingress và FSM tự động nạp Q -> K -> V (20/20 PASS).
       └── 2.3: Đã nghiệm thu độc lập tại REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md.
       │
       ▼
 [BƯỚC 2.5: XONG ✓] Kiến trúc True Dual-Port BRAM Softmax Nạp Động (Software-Defined Multi-Stage).
       ├── 2.5.1: Mở rộng dyadic_scale_config.json chứa bảng 3 Stage (dk=16, 20, 24) và hex 1024 words.
       ├── 2.5.2: True Dual-Port BRAM 1024x20-bit trong softmax_lut.sv; auto-select Stage theo head_dim.
       ├── 2.5.3: Cửa sổ ghi BRAM qua AXI-Lite CSR (0x40 addr, 0x44 data) trong axi_lite_regs.sv.
       └── 2.5.4: Đã nghiệm thu độc lập tại REPORT_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md (5/5 PASS).
       │
       ▼
 [BƯỚC 3: TIẾP THEO] Tái tích hợp & Cấu trúc lại Mảng Systolic Array Top-Level
       ├── 3.1: Tách riêng 2 bộ tích lũy Col0/Col1 trong systolic_mac_array.sv; bỏ cây cộng ngang.
       ├── 3.2: Hoàn thiện score_v_gemm.sv tính đủ 256 tokens (4,096 bytes context out).
       ├── 3.3: Gỡ bỏ mã giả initial $readmemh trong attention_core_top.sv, kết nối luồng dataflow thật.
       └── 3.4: Chạy mô phỏng cục bộ khối GEMM -> Tạo Báo cáo Bước 3.
       │
       ▼
 [BƯỚC 4] Hoàn thiện Testbench, Xác minh Toàn diện & Tổng hợp Vivado 200 MHz
       ├── 4.1: Sửa cú pháp và hoàn thiện tb_attention_core.sv (so sánh đủ 4,096 bytes, bỏ PASS giả).
       ├── 4.2: Thêm Clock Constraint 200 MHz (5.0ns) vào base.xdc.
       ├── 4.3: Chạy Vivado Synthesis & Implementation trên chip Kria KV260 (WNS > 0, TNS = 0).
       └── 4.4: Xuất Bitstream .bit và Hardware Handoff .hwh -> Tạo Báo cáo Nghiệm thu Tổng kết.
======================================================================================================
```

---

## BƯỚC 2: SỬA LỖI SỐ HỌC & LOGIC KHỐI SOFTMAX & BRAM

### Mục tiêu kỹ thuật:
Khắc phục triệt để lỗi số học làm đầu ra Softmax bằng 0 tuyệt đối và lỗi làm mất 50% dữ liệu nạp AXI-Stream.

### Chi tiết các công việc cần thực hiện:

#### 1. Sửa đổi module [`softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/softmax_lut.sv):
* **Lỗi cần sửa (FL-05):** Bộ chia lặp 24 bước đang chia $(255 \ll 16) / \text{sum\_exp} \approx 16.7\text{M} / 16.2\text{M} = 1$, sau đó dịch phải `>> 16` làm kết quả triệt tiêu về 0.
* **Quy chuẩn sửa:**
  1. Hỗ trợ tham số cấu hình độ phân giải xác suất `OUT_WIDTH = 16` (mặc định cho I-ViT bit-exact) và `OUT_WIDTH = 8`.
  2. Thiết kế thuật toán tính toán nghịch đảo tổng mũ $\text{inv\_sum}$:
     - Tử số đặt là $2^{31} - 1 = 32'\text{h7FFF\_FFFF}$ (chuẩn I-ViT `factor = (2**31 - 1) / exp_int_sum`).
     - Bộ chia lặp tính thương số 32-bit: $\text{inv\_sum\_reg} = \lfloor (2^{31}-1) / \text{sum\_exp\_reg} \rfloor$.
     - Tại tầng chuẩn hóa:
       $$\text{prod\_prob} = (\text{rom\_data\_exp} \times \text{inv\_sum\_reg}) \ggg (31 - \text{OUT\_WIDTH} + 1)$$
       Với `OUT_WIDTH = 16`: dịch phải $31 - 16 + 1 = 16$ bit. Kết quả `out_prob` là số nguyên 16-bit nằm trong $[0, 32767]$, **khớp từng bit với PyTorch `IntSoftmax`**.
  3. Cập nhật FSM Softmax:
     - Tín hiệu `start_row` chỉ được kích hoạt xung 1 chu kỳ tại đầu mỗi hàng 256 token.
     - Bổ sung thanh ghi đệm `score_buffer[256]` ngăn chặn ghi đè khi đang tính toán hàng trước.

#### 2. Sửa đổi module [`ping_pong_bram_buffer.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/ping_pong_bram_buffer.sv):
* **Lỗi cần sửa (FL-06 & FL-07):** BRAM ghi 32-bit nhưng bus AXI-Stream là 64-bit (bỏ rơi 32 bit cao). `ram_target_sel` bị gán cứng vào RAM Q.
* **Quy chuẩn sửa:**
  1. Nâng cấp RAM nội bộ lưu trữ dữ liệu 64-bit (hoặc bộ chuyển đổi ghép 2 từ 32-bit trong 1 chu kỳ ghi). Khuyến nghị: cấu hình `RAM_WIDTH = 64` hoặc lưu $8 \times \text{INT8}$ mỗi word BRAM.
  2. Bổ sung mạch đếm địa chỉ và FSM tự động định tuyến Ingress:
     - Giai đoạn 1: Nạp đủ $256 \times 16 \text{ bytes}$ vào `Q_RAM`.
     - Giai đoạn 2: Tự động chuyển con trỏ nạp tiếp $256 \times 16 \text{ bytes}$ vào `K_RAM`.
     - Giai đoạn 3: Nạp tiếp $256 \times 16 \text{ bytes}$ vào `V_RAM`.
     - Phát cờ `bank_write_full = 1'b1` và kích hoạt đảo Bank Ping-Pong an toàn.
  3. Sửa lỗi khởi tạo: Trong chu kỳ đầu, cấm đọc Bank 1 cho đến khi Bank 0 được ghi đầy và tráo đổi thành công.

#### 3. Kiểm thử & Tiêu chí nghiệm thu Bước 2:
* Chạy testbench độc lập:
  ```powershell
  iverilog -g2012 -o tb_softmax.vvp Project/src/Hardware/rtl_ver02/tb_golden_softmax_lut.sv Project/src/Hardware/rtl_ver02/softmax_lut.sv
  vvp tb_softmax.vvp
  ```
* **Tiêu chí ĐẠT:** Testbench so sánh 1,024 mẫu đầu tiên của `golden_softmax_prob.hex` đạt **100% Khớp Bit-Exact (Mismatches = 0)**.
* **Lập báo cáo:** Tạo tài liệu `REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md`.

---

## BƯỚC 3: TÁI TÍCH HỢP & CẤU TRÚC LẠI MẢNG SYSTOLIC ARRAY TOP-LEVEL [ĐÃ HOÀN THÀNH 100% - PASS BIT-EXACT]

### Mục tiêu kỹ thuật:
Xóa bỏ hoàn toàn mã giả combinational loop và `$readmemh` trong Top module; đưa mảng Systolic Array thực tế vào vận hành để tính trọn vẹn 256 tokens Attention.

> **Trạng thái thực tế:** **ĐÃ HOÀN THÀNH 100% (2026-10-13)**  
> **Báo cáo nghiệm thu chi tiết:** Xem [`REPORT_PHASE4_STEP3_TOP_INTEGRATION.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_STEP3_TOP_INTEGRATION.md)  
> **Kết quả kiểm thử:**
> - `systolic_mac_array.sv`: Độc lập `pe_acc_col0`/`col1`, drain logic tuần tự hóa, unit test PASS.
> - `score_v_gemm.sv`: Shift `>>> 15`, bộ đếm 256 hàng, 512 beats, `tlast` tại beat 511, unit test đạt **4,096/4,096 bytes matched (100% bit-exact)**.
> - `attention_core_top.sv`: Xóa bỏ 100% mã phi tổng hợp `$readmemh`, thiết lập Real Dataflow Pipeline thông suốt 4 tầng, tích hợp nạp LUT Bước 2.5, hỗ trợ Sequential Heads $H=4$, testbench tích hợp đạt **4,096/4,096 samples matched (100% bit-exact, 0 errors, 0 tlast errors)**.

---

## BƯỚC 4: HOÀN THIỆN TESTBENCH, XÁC MINH TOÀN DIỆN & TỔNG HỢP VIVADO 200 MHZ

### Mục tiêu kỹ thuật:
Xác minh Testbench bit-exact trên toàn bộ 4,096 bytes của ma trận Attention và đóng gói Bitstream phần cứng trên bo mạch AMD Kria KV260 đạt Timing Closure.

### Chi tiết các công việc cần thực hiện:

#### 1. Hoàn thiện Testbench [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv):
* Sửa lỗi cú pháp `automatic int timeout_cycles` thành biến procedural chuẩn.
* Xóa bỏ nhánh điều kiện PASS giả tạo `else if (egress_cnt == 0)`.
* Cập nhật kích thích AXI-Stream:
  - Stream tuần tự toàn bộ dữ liệu $Q$, $K$, $V$ từ các tệp `q_matrix.hex`, `k_matrix.hex`, `v_matrix.hex` qua bus `s_axis_tdata`.
  - Kích hoạt lệnh `START` qua thanh ghi AXI-Lite CSR (`0x00 <= 1`).
* Giám sát luồng Egress:
  - Đón nhận và kiểm tra trọn vẹn **512 beats (4,096 samples INT8)**.
  - So sánh đối soát bit-exact với [`golden_context_out.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_context_out.hex).
  - In báo cáo kiểm thử chân thực: `Total Egress Samples Monitored: 4096`, `Matched Samples: 4096`, `Errors: 0`.

#### 2. Thiết lập ràng buộc thời gian 200 MHz trong [`base.xdc`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/base.xdc):
* Bổ sung định nghĩa xung nhịp chuẩn:
  ```tcl
  create_clock -period 5.000 -name clk [get_ports clk]
  set_input_delay -clock clk -max 1.500 [get_ports {s_axis_* s_axi_*}]
  set_input_delay -clock clk -min 0.500 [get_ports {s_axis_* s_axi_*}]
  set_output_delay -clock clk -max 1.500 [get_ports {m_axis_* s_axi_*}]
  set_output_delay -clock clk -min 0.500 [get_ports {m_axis_* s_axi_*}]
  ```

#### 3. Chạy Tổng hợp & Đóng gói Vivado (Synthesis & Implementation):
* Khởi tạo dự án Vivado nhắm vào part AMD Kria KV260: `xck26-sfvc784-2LV-c`.
* Chạy luồng kiểm tra tự động (Vivado Batch Mode Tcl):
  - **RTL Synthesis:** Kiểm tra không còn cảnh báo biến mất logic (No trimmed registers on systolic core).
  - **Timing Closure:** Kiểm tra báo cáo `report_timing_summary`: Worst Negative Slack $\text{WNS} \ge 0.000\text{ ns}$, Total Negative Slack $\text{TNS} = 0.000\text{ ns}$ tại 200 MHz.
  - **Tài nguyên (Utilization):** Kiểm tra tỷ lệ sử dụng tài nguyên (DSP48E2 $\le 20\%$, BRAM $\le 15\%$, LUT/FF an toàn).
* **Xuất sản phẩm bàn giao cuối cùng:**
  - `attention_core_kv260.bit` (Bitstream nạp PL)
  - `attention_core_kv260.hwh` (Hardware Handoff cho PYNQ Jupyter Lab)
* **Lập báo cáo tổng kết:** Tạo tài liệu `PHASE4_FINAL_VERIFICATION_AND_BITSTREAM_REPORT.md`.

---

## PROMPT MẪU ĐỂ CHẠY TRONG PHIÊN LÀM VIỆC MỚI (NEW CHAT SESSION PROMPT)

Khi bắt đầu một cửa sổ chat mới, người dùng chỉ cần sao chép đoạn prompt dưới đây để Agent mới tiếp nhận công việc ngay lập tức:

```markdown
Chào Antigravity, tôi muốn tiếp tục dự án Capstone tăng tốc MobileViT-XXS trên AMD Kria KV260.

Bước 1 (Chuẩn hóa Python Golden Model) đã hoàn thành và nghiệm thu tại:
@[model_notebook/ivit_surgery/REPORT_PHASE4_STEP1_PYTHON_GOLDEN_MODEL.md]

Hiện tại tôi cần bạn thực hiện BƯỚC 2 theo đúng kế hoạch kỹ thuật đã lưu tại:
@[model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md]

Yêu cầu thực hiện:
1. Sửa lỗi số học bộ chia Softmax trong Project/src/Hardware/rtl_ver02/softmax_lut.sv để xuất đúng xác suất 16-bit khớp với golden_softmax_prob.hex.
2. Sửa lỗi nhận dữ liệu 64-bit và FSM nạp Q->K->V trong Project/src/Hardware/rtl_ver02/ping_pong_bram_buffer.sv.
3. Chạy kiểm thử iverilog với tb_golden_softmax_lut.sv xác nhận PASS 100%.
4. Tạo báo cáo chi tiết cho Bước 2 tại model_notebook/ivit_surgery/REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md trước khi làm bước tiếp theo.
```

---
*Kế hoạch kỹ thuật được lưu trữ chính thức tại `model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md` để đảm bảo tính liên tục và nhất quán của dự án.*
