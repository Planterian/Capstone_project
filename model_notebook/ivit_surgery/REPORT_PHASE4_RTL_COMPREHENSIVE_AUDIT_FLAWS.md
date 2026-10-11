# BÁO CÁO KIỂM TOÁN TOÀN DIỆN PHASE 4: PHÂN TÍCH KHUYẾT TẬT THIẾT KẾ RTL VÀ KẾ HOẠCH KIỂM THỬ (RTL DESIGN & VERIFICATION AUDIT REPORT)

**Mã tài liệu:** `REPORT_PHASE4_RTL_COMPREHENSIVE_AUDIT_FLAWS`  
**Thời gian lập báo cáo:** 11/10/2026 — 17:15:00 (GMT+7)  
**Phạm vi kiểm toán:** Toàn bộ mã nguồn RTL (`Project/src/Hardware/rtl_ver02/`), Testbench, Kịch bản trích xuất Golden Vectors (`export_golden_vectors.py`), Tệp ràng buộc (`base.xdc`), và 2 báo cáo đối chứng tiền nhiệm:
- [`PHASE4_RTL_BIT_CORRECTION_REPORT.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PHASE4_RTL_BIT_CORRECTION_REPORT.md)
- [`REPORT_RTL_BIT_ALIGNMENT_20261010_0950.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_RTL_BIT_ALIGNMENT_20261010_0950.md)

**Yêu cầu thực thi:** Kiểm tra toàn diện, truy tìm toàn bộ các lỗi tiềm ẩn (flaws), không sửa đổi mã nguồn gốc, lập báo cáo chi tiết khách quan phục vụ khắc phục và nghiệm thu đồ án.

---

## MỤC LỤC
1. [TỔNG QUAN ĐIỀU HÀNH & KẾT LUẬN KIỂM TOÁN (EXECUTIVE SUMMARY & VERDICT)](#1-tổng-quan-điều-hành--kết-luận-kiểm-toán)
2. [CÁC KHUYẾT TẬT KIẾN TRÚC & CHỨC NĂNG RTL NGHIÊM TRỌNG (CORE RTL FLAWS)](#2-các-khuyết-tật-kiến-trúc--chức-năng-rtl-nghiêm-trọng)
   - 2.1 [Systolic MAC Array bị ngắt kết nối và thay thế bằng mã giả trong Top-Level](#21-systolic-mac-array-bị-ngắt-kết-nối-và-thay-thế-bằng-mã-giả)
   - 2.2 [Khuyết tật cấu trúc toán học bên trong Systolic MAC Array](#22-khuyết-tật-cấu-trúc-toán-học-bên-trong-systolic-mac-array)
   - 2.3 [Lỗi tính toán dấu phẩy tĩnh làm tê liệt bộ chia Softmax Hardware](#23-lỗi-tính-toán-dấu-phẩy-tĩnh-làm-tê-liệt-bộ-chia-softmax-hardware)
   - 2.4 [Mất mát dữ liệu 50% và khóa chết địa chỉ trong Ping-Pong BRAM Buffer](#24-mất-mát-dữ-liệu-50-và-khóa-chết-địa-chỉ-trong-ping-pong-bram)
   - 2.5 [Khối Stage 4 Score x V chỉ tính 1 Token duy nhất (thiếu 99.6% ma trận)](#25-khối-stage-4-score-x-v-chỉ-tính-1-token-duy-nhất)
   - 2.6 [Cơ chế Sequential Heads Loop chỉ là FSM hình thức, không xử lý đa Head](#26-cơ-chế-sequential-heads-loop-chỉ-là-fsm-hình-thức)
   - 2.7 [Thanh ghi AXI-Lite CSR và các khối RTL ngoại vi không được kết nối](#27-thanh-ghi-axi-lite-csr-và-các-khối-rtl-ngoại-vi)
3. [CÁC KHUYẾT TẬT TRONG TESTBENCH & KẾ HOẠCH KIỂM THỬ (VERIFICATION & TESTBENCH FLAWS)](#3-các-khuyết-tật-trong-testbench--kế-hoạch-kiểm-thử)
   - 3.1 [Lỗi cú pháp SystemVerilog ngăn cản biên dịch trên Simulator chuẩn](#31-lỗi-cú-pháp-systemverilog-ngăn-cản-biên-dịch)
   - 3.2 [Điều kiện PASS giả tạo khi hệ thống bị treo hoặc không có dữ liệu](#32-điều-kiện-pass-giả-tạo-khi-hệ-thống-bị-treo)
   - 3.3 [Ngưỡng dung sai che giấu hoàn toàn lỗi Softmax xuất toàn số 0](#33-ngưỡng-dung-sai-che-giấu-hoàn-toàn-lỗi-softmax)
   - 3.4 [Kích thích Testbench thiếu hụt (chỉ nạp ma trận Q, bỏ qua K và V)](#34-kích-thích-testbench-thiếu-hụt)
   - 3.5 [Bằng chứng nhật ký mô phỏng trong báo cáo Phase 4 không phản ánh thực tế](#35-bằng-chứng-nhật-ký-mô-phỏng-trong-báo-cáo-phase-4)
4. [CÁC KHUYẾT TẬT TRONG QUY TRÌNH TẠO GOLDEN VECTORS (`export_golden_vectors.py`)](#4-các-khuyết-tật-trong-quy-trình-tạo-golden-vectors)
   - 4.1 [Suy thoái Softmax xác suất thành toàn bộ số 1 (All-Ones Matrix)](#41-suy-thoái-softmax-xác-suất-thành-toàn-bộ-số-1)
   - 4.2 [Lệch chuẩn thuật toán với mô hình PyTorch I-ViT QAT](#42-lệch-chuẩn-thuật-toán-với-mô-hình-pytorch-i-vit-qat)
   - 4.3 [Golden Context Out trở thành tổng số học thô của ma trận V](#43-golden-context-out-trở-thành-tổng-số-học-thô)
5. [KHUYẾT TẬT RÀNG BUỘC THỜI GIAN VÀ KHẢ NĂNG TỔNG HỢP (TIMING & SYNTHESIS READINESS)](#5-khuyết-tật-ràng-buộc-thời-gian-và-khả-năng-tổng-hợp)
6. [BẢNG MA TRẬN TRUY VẾT TỔNG HỢP CÁC KHUYẾT TẬT (DEFECT TRACEABILITY MATRIX)](#6-bảng-ma-trận-truy-vết-tổng-hợp-các-khuyết-tật)
7. [LỘ TRÌNH KHẮC PHỤC KỸ THUẬT (ACTIONABLE REMEDIATION ROADMAP)](#7-lộ-trình-khắc-phục-kỹ-thuật)

---

## 1. TỔNG QUAN ĐIỀU HÀNH & KẾT LUẬN KIỂM TOÁN

Qua quá trình rà soát độc lập toàn diện từng dòng mã nguồn, phương trình toán học, cấu trúc phần cứng và thực thi mô phỏng trên công cụ chuẩn IEEE 1800, nhóm kiểm toán đưa ra kết luận:

> [!CAUTION]
> **KẾT LUẬN KIỂM TOÁN CHÍNH THỨC: THIẾT KẾ RTL PHASE 4 (`rtl_ver02`) CHƯA ĐẠT CHUẨN HOÀN THIỆN.**
> Tuyên bố trong tài liệu `PHASE4_RTL_BIT_CORRECTION_REPORT.md` (rằng thiết kế đã đạt *Executive Status: Complete ✓*, *100% Bit-Exact Match PASS*, *Sequential Heads Loop hoàn thành*) **chưa phản ánh chính xác thực tế kỹ thuật**.
> Khi chạy mô phỏng kiểm thử thực tế `tb_attention_core.sv`, kết quả thực tế là **FAIL với tỷ lệ lỗi 93.75% (15/16 mẫu không khớp)**, phần cứng chỉ chạy đúng 1 token đầu tiên rồi ngắt, và khối Softmax Hardware bị lỗi dấu phẩy tĩnh khiến đầu ra **toàn bộ bằng 0**.

### Bảng đối chiếu hiện trạng báo cáo Phase 4 vs Thực tế kiểm toán:

| Hạng mục kỹ thuật | Tuyên bố trong báo cáo Phase 4 | Kết quả kiểm toán thực tế | Đánh giá mức độ |
| :--- | :--- | :--- | :---: |
| **Tính hợp lệ Testbench Top** | Pass 256/256 mẫu Egress | **FAIL: 15/16 mẫu lỗi**, chỉ chạy 16 mẫu rồi tắt | **Nghiêm trọng (Critical)** |
| **Khối Systolic MAC Array** | Tích hợp nhân ma trận Bit-Exact | **Bị ngắt kết nối hoàn toàn** (Floating output), thay bằng vòng lặp gán cứng Row 0 | **Nghiêm trọng (Critical)** |
| **Thuật toán Hardware Softmax** | Timing-Safe, sai số $\le 1$ LSB | **Lỗi chia Fixed-point: Đầu ra Prob = 0 toàn bộ** | **Nghiêm trọng (Critical)** |
| **Dung lượng Vector Egress** | Hoàn thành ma trận Attention | **Chỉ tính 1 Token (16 bytes)** trên tổng số 256 tokens (4,096 bytes) | **Nghiêm trọng (Critical)** |
| **Ping-Pong BRAM Ingress** | Nạp đệm Q, K, V qua AXI-Stream | **Cắt bỏ 50% dữ liệu** (`[63:32]`), khóa cứng chỉ nạp được Q | **Nghiêm trọng (Critical)** |
| **Vòng lặp Head (Sequential Loop)**| Xử lý tuần tự $H=4$ Heads | FSM chạy hình thức, không trỏ địa chỉ Head mới, lặp lại Head 0 | **Cao (High)** |
| **Chất lượng Golden Vectors** | Bộ dữ liệu chuẩn từ PyTorch QAT | **`golden_softmax_prob.hex` suy thoái thành toàn bộ số 1 (65,536 giá trị `0x01`)** | **Nghiêm trọng (Critical)** |
| **Khả năng tổng hợp (Synthesizability)**| Sẵn sàng tạo Bitstream KV260 | Tồn tại `$readmemh` trong Top Module; thiếu Timing Constraints | **Cao (High)** |

---

## 2. CÁC KHUYẾT TẬT KIẾN TRÚC & CHỨC NĂNG RTL NGHIÊM TRỌNG

### 2.1 Systolic MAC Array bị ngắt kết nối và thay thế bằng mã giả trong Top-Level
* **Vị trí tệp:** [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/attention_core_top.sv#L243-L340)
* **Hiện trạng mã nguồn:**
  Tại dòng 308–324, module `systolic_mac_array` được khởi tạo:
  ```systemverilog
  systolic_mac_array #( ... ) u_systolic_mac_array (
      .clk             (clk),
      .rst_n           (rst_n),
      .gemm_start      (start_pulse),
      .clr_acc         (clear_buf_pulse),
      .in_valid        (skid_tvalid),
      .mode_sel        (1'b0),
      .act_row         (act_row_wire),
      .wt_col0         (wt_col0_wire),
      .wt_col1         (wt_col1_wire),
      .out_valid       (systolic_out_valid),
      .dot_product_out (dot_product_out)
  );
  ```
  Tuy nhiên, hai tín hiệu đầu ra `dot_product_out` và `systolic_out_valid` **hoàn toàn không được nối đi bất kỳ đâu (floating)**!
  Ngược lại, khối `scale_unit` tại dòng 327–340 lại nhận đầu vào từ `stage1_score_reg`:
  ```systemverilog
  scale_unit #( ... ) u_scale_unit (
      .in_valid     (stage1_valid_reg),
      .in_score     (stage1_score_reg),
      ...
  );
  ```
  Và `stage1_score_reg` được tính bằng một vòng lặp tổ hợp thuần túy (dòng 278–283):
  ```systemverilog
  always_comb begin
      stage1_score_wire = '0;
      for (int d = 0; d < 16; d++) begin
          stage1_score_wire = stage1_score_wire + 32'($signed(q_matrix_ram[0][d]) * $signed(k_matrix_ram[k_step_cnt][d]));
      end
  end
  ```
* **Hậu quả kỹ thuật:**
  1. Mảng Systolic Array 2D (khối kiến trúc cốt lõi nhất của bộ tăng tốc) thực tế là **"mã chết" (dead logic)**. Khi tổng hợp bằng Vivado Synthesis, toàn bộ khối systolic array sẽ bị bộ tối ưu hóa gỡ bỏ hoàn toàn (trimmed/pruned).
  2. Vòng lặp tính toán gán cứng `q_matrix_ram[0][d]`: **chỉ tính toán duy nhất hàng 0 của ma trận Q**. 255 hàng còn lại ($t = 1 \dots 255$) bị bỏ rơi hoàn toàn.
  3. Chiều vector bị gán cứng `d < 16`: Khi cấu hình sang Stage 3 ($d_k = 20$) hoặc Stage 4 ($d_k = 24$), phần cứng vẫn chỉ nhân 16 chiều đầu tiên, làm sai lệch kết quả hoàn toàn.
  4. Top-level chứa khối `initial` với `$readmemh` nạp trực tiếp file `.hex` từ máy tính vào mảng thanh ghi (`q_raw_init`, `k_raw_init`, `v_raw_init`). Đây là cấu trúc **phi tổng hợp (non-synthesizable)** trên FPGA board thật nếu không được ánh xạ qua ROM/BRAM chuyên dụng có tệp COE/MEM.

---

### 2.2 Khuyết tật cấu trúc toán học bên trong Systolic MAC Array
* **Vị trí tệp:** [`systolic_mac_array.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/systolic_mac_array.sv#L103-L144)
* **Khuyết tật 1: Trộn lẫn 2 cột dữ liệu độc lập vào 1 bộ tích lũy:**
  Tại dòng 109:
  ```systemverilog
  pe_acc_matrix[r][c] <= pe_acc_matrix[r][c] + 32'(pe_prod_ab) + 32'(pe_prod_ac);
  ```
  Trong kỹ thuật DSP Packing của Xilinx UltraScale+, cổng B và cổng C của DSP nhận 2 trọng số thuộc 2 cột khác nhau ($W_{\text{col0}}$ và $W_{\text{col1}}$). Mục đích là tính song song hai phần tử ma trận kết quả $C_{r, 2c}$ và $C_{r, 2c+1}$. Việc cộng gộp `pe_prod_ab + pe_prod_ac` vào cùng một thanh ghi tích lũy tương đương phép tính:
  $$P = A \times (B + C)$$
  Điều này làm triệt tiêu tính độc lập của 2 cột đặc trưng, phá hủy hoàn toàn cấu trúc ma trận kết quả.
* **Khuyết tật 2: Cây cộng Adder Tree giảm chiều sai hướng toán học:**
  Tại dòng 117–144, mảng systolic tiếp tục đưa các bộ tích lũy vào cây cộng `pipelined_adder_tree`:
  ```systemverilog
  assign tree_in_a[c] = pe_acc_matrix[r][c]; // c = 0..ARRAY_SIZE-1
  // Đưa vào adder tree để tính dot_product_out[r]
  ```
  Trong kiến trúc Output-Stationary Systolic Array, mỗi phần tử PE $(r, c)$ tại cuối chu kỳ đã chứa trọn vẹn giá trị tích vô hướng của hàng $r$ và cột $c$. Việc dùng thêm cây cộng theo chiều ngang $c$ sẽ tính tổng:
  $$\text{Out}[r] = \sum_{c=0}^{\text{ARRAY\_SIZE}-1} C_{r, c}$$
  Đây là phép cộng dồn tất cả các cột của ma trận kết quả thành một số vô hướng duy nhất, hoàn toàn vô nghĩa trong phép nhân ma trận GEMM ($Q \times K^T$).

---

### 2.3 Lỗi tính toán dấu phẩy tĩnh làm tê liệt bộ chia Softmax Hardware
* **Vị trí tệp:** [`softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/softmax_lut.sv#L392-L435)
* **Hiện trạng mã nguồn:**
  Tại trạng thái khởi động bộ chia `ST_DIV_START` (dòng 394–398):
  ```systemverilog
  // Khởi tạo bộ chia: tử số = (255 << 16) = 16,711,680
  div_denom_reg <= (sum_exp_reg == 0) ? 32'd1 : sum_exp_reg;
  div_rem_reg   <= {{16'b0, 32'd16711680}};
  div_step_reg  <= 6'd24;
  state_reg     <= ST_DIV_RUN;
  ```
  Khi bộ chia không phục hồi hoàn tất, kết quả được gán (dòng 417):
  ```systemverilog
  inv_sum_reg <= div_rem_reg[23:0];
  ```
  Và tại tầng chuẩn hóa đầu ra `ST_NORM_OUT` (dòng 425–428):
  ```systemverilog
  prod_prob = (rom_data_exp * inv_sum_reg) >> 16;
  out_prob  <= (prod_prob > 32'd255) ? 8'hFF : prod_prob[7:0];
  ```
* **Phân tích cơ chế gây lỗi (Fatal Arithmetic Failure):**
  1. Với $N=256$ tokens, tổng các số mũ $\text{sum\_exp\_reg} = \sum_{i=0}^{255} \text{ROM}[i]$. Giá trị mỗi mục ROM $\le 65535$. Trong thực tế kiểm thử, $\text{sum\_exp\_reg} \approx 16,211,790$ (tương đương $\approx 250 \times 65000$).
  2. Tử số được chọn là $255 \times 2^{16} = 16,711,680$.
  3. Thương số nguyên của phép chia:
     $$\text{inv\_sum\_reg} = \left\lfloor \frac{16,711,680}{16,211,790} \right\rfloor = \mathbf{1}$$
  4. Khi tính xác suất:
     $$\text{prod\_prob} = \frac{\text{rom\_data\_exp} \times 1}{2^{16}} = \frac{\text{rom\_data\_exp}}{65536}$$
     Do $\text{rom\_data\_exp} \le 65535 < 65536$, phép dịch phải số nguyên `>> 16` luôn luôn cho kết quả bằng:
     $$\mathbf{prod\_prob = 0} \implies \mathbf{out\_prob = 8'h00}$$
* **Hậu quả:** Khối `softmax_lut.sv` **xuất toàn bộ xác suất bằng 0 cho tất cả 256 tokens** trên toàn bộ các chu kỳ chạy.

---

### 2.4 Mất mát dữ liệu 50% và khóa chết địa chỉ trong Ping-Pong BRAM Buffer
* **Vị trí tệp:** [`ping_pong_bram_buffer.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/ping_pong_bram_buffer.sv#L168-L208) và [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/attention_core_top.sv#L193-L220)
* **Khuyết tật 1: Bỏ rơi 32 bit cao của bus AXI4-Stream:**
  Trong `ping_pong_bram_buffer.sv`, giao tiếp đầu vào là `DATA_WIDTH = 64` (chứa 8 giá trị INT8). Độ rộng bộ nhớ RAM là `RAM_WIDTH = 32`.
  Tại dòng 173–176:
  ```systemverilog
  case (ram_target_sel)
      2'b00: ram_q_b0[waddr_cnt] <= s_axis_tdata[RAM_WIDTH-1:0]; // Chỉ lấy [31:0]!
      ...
  ```
  Các bit `s_axis_tdata[63:32]` hoàn toàn bị bỏ rơi. Cứ mỗi nhịp truyền AXI-Stream, **50% dữ liệu của token bị vứt bỏ**.
* **Khuyết tật 2: Khóa chết mục tiêu bộ nhớ RAM:**
  Trong `attention_core_top.sv` dòng 206:
  ```systemverilog
  .ram_target_sel (2'b00), // Gán cứng luôn luôn là Q_RAM!
  ```
  Không có cơ chế chuyển vùng `ram_target_sel` sang `2'b01` (K_RAM) hay `2'b10` (V_RAM). Qua bus AXI-Stream, CPU ARM **không thể nào nạp được ma trận K và V vào BRAM**.
* **Khuyết tật 3: Địa chỉ đọc BRAM bị treo ở mức 0:**
  Trong `attention_core_top.sv` dòng 401–403:
  ```systemverilog
  raddr_q <= '0;
  raddr_k <= '0;
  raddr_v <= '0;
  ```
  Các thanh ghi địa chỉ đọc này không bao giờ được tăng trong bất kỳ trạng thái nào của FSM.
* **Khuyết tật 4: Lỗi nhịp khởi tạo Ping-Pong tráo Bank:**
  Biến cờ `pp_bank_reg` khởi tạo bằng `1'b0` (Ghi Bank 0, Đọc Bank 1). Nhưng Bank 1 lúc khởi động hoàn toàn chứa dữ liệu rác (hoặc toàn 0). Vì vậy, trong lần thực thi đầu tiên, khối tính toán sẽ đọc toàn bộ số 0 từ Bank 1.

---

### 2.5 Khối Stage 4 Score x V chỉ tính 1 Token duy nhất
* **Vị trí tệp:** [`score_v_gemm.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/score_v_gemm.sv#L85-L162) & [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/attention_core_top.sv#L422-L438)
* **Chi tiết kỹ thuật:**
  Ma trận Attention Context Out có kích thước chuẩn:
  $$N \times d_k = 256 \times 16 = 4,096 \text{ bytes (yêu cầu 512 beats AXI-Stream 64-bit)}$$
  Tuy nhiên trong `attention_core_top.sv`:
  ```systemverilog
  MHA_RUN_ATTN: begin
      if (softmax_out_valid) begin
          out_token_cnt <= out_token_cnt + 1'b1;
          if (out_token_cnt >= (reg_seq_len - 1)) begin
              mha_state <= MHA_WAIT_EGRESS;
          end
      end
  end
  ```
  Biến `out_token_cnt` đếm 256 xung `softmax_out_valid`. Nhưng 256 xung này chỉ tương ứng với **hàng xác suất thứ 0**.
  Khi chuyển sang `MHA_WAIT_EGRESS`, module `score_v_gemm.sv` xuất ra:
  - Beat 0: 8 bytes (Dimensions 0..7)
  - Beat 1: 8 bytes (Dimensions 8..15) có gắn cờ `m_axis_tlast = 1'b1`.
  Ngay khi nhận `m_axis_tlast`, FSM top-level chuyển ngay sang `MHA_NEXT_HEAD` rồi kết thúc sang `MHA_DONE`!
* **Hậu quả:** Bộ tăng tốc chỉ tạo ra đúng **16 bytes (1 vector đặc trưng của token 0)** thay vì 4,096 bytes. Toàn bộ 255 tokens còn lại (tương đương **99.61% khối lượng tính toán**) bị biến mất hoàn toàn khỏi dòng xuất Egress.

---

### 2.6 Cơ chế Sequential Heads Loop chỉ là FSM hình thức
* **Vị trí tệp:** [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/attention_core_top.sv#L440-L448)
* **Phân tích:**
  Báo cáo Phase 4 khẳng định kiến trúc đã hoàn tất tính năng *Sequential Heads Loop ($h = 0 \dots H-1$)*.
  Tuy nhiên, khi kiểm tra mã FSM:
  ```systemverilog
  MHA_NEXT_HEAD: begin
      out_token_cnt <= '0;
      if (head_cnt + 1'b1 < reg_num_heads[3:0]) begin
          head_cnt  <= head_cnt + 1'b1;
          mha_state <= MHA_LOAD_HEAD;
      end else begin
          mha_state <= MHA_DONE;
      end
  end
  ```
  Khi chuyển sang `head_cnt = 1`:
  - Không có bất kỳ logic tính toán offset địa chỉ nào để nạp dữ liệu của Head 1 từ BRAM (địa chỉ BRAM vẫn bị treo ở 0).
  - Vòng lặp tính toán vẫn chỉ truy cập vào `q_matrix_ram[0][d]`, vốn là mảng dữ liệu chỉ chứa Head 0 được nạp từ `q_matrix.hex`.
  - Kết quả là nếu cấu hình $H=4$, phần cứng chỉ đơn thuần chạy lặp lại đúng 1 token của Head 0 thêm 3 lần nữa.

---

### 2.7 Thanh ghi AXI-Lite CSR và các khối RTL ngoại vi không được kết nối
1. **Khối Re-quantization Residual Adder:** Tệp [`requant_residual_add.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/requant_residual_add.sv) có mặt trong thư mục nhưng **hoàn toàn không được khởi tạo (uninstantiated)** trong `attention_core_top.sv`.
2. **Thanh ghi Mantissa/Exponent tái lượng tử hóa:** `axi_lite_regs` xuất tín hiệu `reg_out_m` và `reg_out_e` (địa chỉ CSR `0x28` và `0x2C`), nhưng trong `attention_core_top.sv`, hai dây này để lơ lửng, không được kết nối vào `score_v_gemm.sv`. Trong khi đó, `score_v_gemm.sv` lại gán cứng phép dịch bit thô `ctx_acc >>> 8`.
3. **Thanh ghi độ dài luồng nạp DMA:** `reg_nbytes` (CSR `0x08`) được ghi từ ARM CPU nhưng để hở, không có mạch đếm kiểm tra dung lượng DMA ingress.

---

## 3. CÁC KHUYẾT TẬT TRONG TESTBENCH & KẾ HOẠCH KIỂM THỬ

### 3.1 Lỗi cú pháp SystemVerilog ngăn cản biên dịch trên Simulator chuẩn
* **Vị trí tệp:** [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv#L218-L224)
* **Mã lỗi:**
  ```systemverilog
  begin : wait_egress
      automatic int timeout_cycles = 0; // LỖI CÚ PHÁP CHUẨN IEEE 1800
      while (egress_cnt == 0 && timeout_cycles < 1500) begin
  ```
* **Phân tích:** Khai báo `automatic` bên trong một khối lệnh (`begin : block_name`) nằm trực tiếp dưới phạm vi module (ngoài `task`/`function`) là vi phạm chuẩn phân rã biến trong nhiều trình biên dịch HDL tiêu chuẩn (như Icarus Verilog: *"sorry: overriding the default variable lifetime is not yet supported"*). Điều này chứng minh testbench này **chưa từng được biên dịch sạch thành công** trên các công cụ mô phỏng chuẩn mà không qua chỉnh sửa.

---

### 3.2 Điều kiện PASS giả tạo khi hệ thống bị treo
* **Vị trí tệp:** [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv#L237-L242)
* **Mã nguồn:**
  ```systemverilog
  if (error_count == 0 && egress_cnt > 0) begin
      $display(" >>> STATUS: ALL GOLDEN VECTORS MATCHED SUCCESSFULLY! [PASS] <<<");
  end else if (egress_cnt == 0) begin
      $display(" >>> STATUS: PIPELINE COMPLETED [PASS - ZERO TIMING/DEADLOCK FAULTS] <<<");
  end else begin
      $display(" >>> STATUS: %0d ERRORS DETECTED [FAIL] <<<", error_count);
  end
  ```
* **Lỗ hổng kiểm thử nghiêm trọng (Critical Testbench Flaw):**
  Nhánh `else if (egress_cnt == 0)` có nghĩa là: **Nếu mạch phần cứng bị Deadlock hoàn toàn, hoặc tín hiệu Valid không bao giờ bật lên khiến không có một byte dữ liệu nào được xuất ra (`egress_cnt == 0`), Testbench vẫn in ra thông báo PASS!**
  Đây là một phản mẫu (anti-pattern) kiểm thử nguy hiểm, biến lỗi treo mạch thành kết quả kiểm thử thành công.

---

### 3.3 Ngưỡng dung sai che giấu hoàn toàn lỗi Softmax
* **Vị trí tệp:** [`tb_golden_softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_golden_softmax_lut.sv#L66) & [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv#L85)
* **Hiện trạng kiểm tra:**
  ```systemverilog
  if (diff <= 1) begin
      pass_count++;
  end else begin
      error_count++;
  end
  ```
* **Cơ chế che giấu lỗi:**
  Như đã phân tích ở Mục 2.3, module `softmax_lut.sv` xuất ra `DUT Prob = 0` cho toàn bộ các phần tử.
  Trong khi đó, tệp `golden_softmax_prob.hex` do lỗi làm tròn từ Python nên toàn bộ 65,536 giá trị đều bằng `1` (Mục 4.1).
  Khoảng cách sai lệch:
  $$\text{diff} = |\text{DUT} - \text{Golden}| = |0 - 1| = 1 \le 1$$
  Vì điều kiện là `diff <= 1`, **Testbench đã tính 1,024 mẫu lỗi sai lệch hoàn toàn thành 1,024 mẫu PASS!**
  Điều này tạo ra báo cáo giả: `100% BIT-EXACT MATCH (<= 1 LSB Tolerance)! [PASS]` trong khi thực chất phần cứng không tính toán được gì.

---

### 3.4 Kích thích Testbench thiếu hụt
* **Vị trí tệp:** [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv#L198-L215)
* Testbench chỉ thực hiện vòng lặp stream dữ liệu của mảng `q_mem` vào bus `s_axis_tdata`:
  ```systemverilog
  for (int i = 0; i < TEST_TOKENS; i++) begin
      s_axis_tvalid <= 1'b1;
      s_axis_tdata  <= {q_mem[...], ...};
  end
  ```
  Testbench **hoàn toàn không stream `k_mem` và không stream `v_mem`**.
  Sở dĩ mạch vẫn "chạy" được trong mô phỏng là vì `attention_core_top.sv` đã tự ý nạp ngầm các tệp hex thông qua khối `$readmemh` phi tổng hợp bên trong Top module. Nếu đưa lên FPGA board thật qua cổng AXI DMA, thiết kế sẽ dừng hoạt động ngay lập tức vì thiếu K và V.

---

### 3.5 Bằng chứng nhật ký mô phỏng trong báo cáo Phase 4 không phản ánh thực tế

Đối chiếu trực tiếp giữa nhật ký công bố trong [`PHASE4_RTL_BIT_CORRECTION_REPORT.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PHASE4_RTL_BIT_CORRECTION_REPORT.md#L249-L269) và kết quả mô phỏng thực thi thực tế khi chạy testbench:

#### So sánh trực diện hai kết quả:

```
+-------------------------------------------------------------+-------------------------------------------------------------+
|   NHẬT KÝ ĐƯỢC CÔNG BỐ TRONG BÁO CÁO PHASE 4                |   KẾT QUẢ MÔ PHỎNG THỰC TẾ TRÊN CÔNG CỤ HDL SIMULATOR      |
+-------------------------------------------------------------+-------------------------------------------------------------+
| =========================================================== | =========================================================== |
|    FINAL VERIFICATION REPORT: ATTENTION CORE (rtl_ver02)    |    FINAL VERIFICATION REPORT: ATTENTION CORE (rtl_ver02)    |
| =========================================================== | =========================================================== |
|    Total Egress Samples Monitored : 256                     |    Total Egress Samples Monitored : 16                      |
|    Matched Samples (<= 1 LSB)     : 256                     |    Matched Samples (<= 1 LSB)     : 1                       |
|    Mismatched Samples             : 0                       |    Mismatched Samples             : 15                      |
|    >>> STATUS: ALL GOLDEN VECTORS MATCHED! [PASS] <<<       |    >>> STATUS: 15 ERRORS DETECTED [FAIL] <<<                |
| =========================================================== | =========================================================== |
+-------------------------------------------------------------+-------------------------------------------------------------+
```

#### Dữ liệu mẫu Egress thực tế in ra từ Simulator:
```text
[CONTEXT EGRESS #0000]: DUT = 0 | Golden = 7   (Diff = 7)  --> FAIL
[CONTEXT EGRESS #0001]: DUT = 0 | Golden = 2   (Diff = 2)  --> FAIL
[CONTEXT EGRESS #0002]: DUT = 0 | Golden = 2   (Diff = 2)  --> FAIL
[CONTEXT EGRESS #0003]: DUT = 0 | Golden = -2  (Diff = 2)  --> FAIL
[CONTEXT EGRESS #0004]: DUT = 0 | Golden = -13 (Diff = 13) --> FAIL
[CONTEXT EGRESS #0005]: DUT = 0 | Golden = 7   (Diff = 7)  --> FAIL
[CONTEXT EGRESS #0006]: DUT = 0 | Golden = -8  (Diff = 8)  --> FAIL
[CONTEXT EGRESS #0007]: DUT = 0 | Golden = 1   (Diff = 1)  --> PASS (do ngưỡng <= 1)
```
*DUT xuất toàn bộ bằng 0 (`DUT = 0`)*, chỉ duy nhất mẫu #0007 có Golden bằng 1 nên tình cờ rơi vào ngưỡng $\text{diff} \le 1$. Toàn bộ 15 mẫu còn lại đều sai lệch nghiêm trọng. Tổng số mẫu giám sát dừng lại ở con số 16 thay vì 256 như báo cáo công bố.

---

## 4. CÁC KHUYẾT TẬT TRONG QUY TRÌNH TẠO GOLDEN VECTORS (`export_golden_vectors.py`)

### 4.1 Suy thoái Softmax xác suất thành toàn bộ số 1 (All-Ones Matrix)
* **Vị trí tệp:** [`export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py#L256-L260)
* **Đoạn mã sinh dữ liệu:**
  ```python
  score_float = scaled_score_int16.astype(np.float32) / 32.0
  exp_score = np.exp(score_float - np.max(score_float, axis=-1, keepdims=True))
  prob_float = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
  softmax_prob_uint8 = np.clip(np.round(prob_float * 255.0), 0, 255).astype(np.uint8)
  ```
* **Phân tích toán học:**
  - Với chuỗi token độ dài $N = 256$, xác suất phân bố đều hoặc phân tán quanh giá trị:
    $$\text{prob\_float} \approx \frac{1}{256} \approx 0.00390625$$
  - Nhân với hệ số tỷ lệ 255.0:
    $$0.00390625 \times 255.0 = 0.99609375$$
  - Khi áp dụng hàm làm tròn `np.round(...)`:
    $$\text{round}(0.99609375) = \mathbf{1.0}$$
* **Kết quả kiểm toán trên tệp:**
  Kiểm tra toàn bộ 65,536 phần tử trong tệp [`golden_softmax_prob.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/golden_softmax_prob.hex):
  - Giá trị lớn nhất: `1`
  - Giá trị nhỏ nhất: `1`
  - Số lượng giá trị bằng `1`: **65,536 (100%)**
  - Số lượng giá trị khác `1`: **0 (0%)**
* **Ý nghĩa:** Ma trận xác suất chú ý Softmax bị suy biến hoàn toàn thành một **ma trận toàn số 1**. Tính chất tập trung trọng số (Attention Focus) của mạng Transformer bị phá hủy 100%.

---

### 4.2 Lệch chuẩn thuật toán với mô hình PyTorch I-ViT QAT
* Mô hình PyTorch đã qua phẫu thuật tại [`mobilevit_ivit_surgery.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/mobilevit_ivit_surgery.py#L105) sử dụng lớp `IntSoftmax(16)` (Shiftmax) với bộ tích lũy 16-bit và thuật toán dịch bit đa thức $x + \lfloor x/2 \rfloor - \lfloor x/16 \rfloor$.
* Trong khi đó, `export_golden_vectors.py` không hề gọi qua `model.int_softmax`, mà lại tự viết một công thức xấp xỉ riêng với phép chia tùy tiện `/ 32.0`.
* Điều này dẫn đến sự mất đồng bộ 3 bên:
  1. **PyTorch QAT:** Dùng toán tử Shiftmax I-ViT (16-bit).
  2. **Golden Generator:** Dùng hàm `np.exp()` float32 chia cho 32.0 rồi ép kiểu sang UINT8.
  3. **RTL Softmax:** Dùng bảng tra LUT $256 \times 16$-bit kết hợp bộ chia lặp 24 bước.
  Ba thuật toán hoàn toàn khác nhau cùng tồn tại khiến cho mục tiêu "Bit-Exact Alignment" về mặt toán học là không thể đạt được.

---

### 4.3 Golden Context Out trở thành tổng số học thô của ma trận V
Tại dòng 263–264 của `export_golden_vectors.py`:
```python
context_int32 = np.matmul(softmax_prob_uint8.astype(np.int32), v_int8.astype(np.int32))
context_out_int8 = np.clip(context_int32 >> 8, -128, 127).astype(np.int8)
```
Do ma trận `softmax_prob_uint8` bị suy thoái thành ma trận toàn số 1 (tất cả các phần tử đều là 1), phép nhân ma trận trên thực chất chỉ là phép tính:
$$\text{context\_int32}[i, d] = \sum_{k=0}^{255} 1 \times V[k, d] = \sum_{k=0}^{255} V[k, d]$$
Vector đầu ra của tất cả 256 tokens trở nên giống hệt nhau, và đơn thuần chỉ là tổng số học của các cột ma trận V. Dữ liệu kiểm thử này hoàn toàn mất đi tính đại diện của cơ chế Attention.

---

## 5. KHUYẾT TẬT RÀNG BUỘC THỜI GIAN VÀ KHẢ NĂNG TỔNG HỢP (TIMING & SYNTHESIS READINESS)

Khi kiểm tra tệp ràng buộc phần cứng [`base.xdc`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/base.xdc):
1. **Thiếu định nghĩa xung nhịp hệ thống (Clock Constraint):**
   Tệp `base.xdc` chỉ chứa các gán chân I/O vật lý cho Camera MIPI và header PMOD kế thừa từ dự án mẫu của bo mạch. **Không có bất kỳ dòng lệnh nào định nghĩa xung nhịp**, ví dụ:
   ```tcl
   create_clock -period 5.000 -name clk [get_ports clk]
   ```
2. **Hậu quả về mặt Timing:** Không có ràng buộc chu kỳ xung nhịp 200 MHz, trình tổng hợp Vivado Timing Engine không thể phân tích độ trễ đường truyền (Static Timing Analysis - STA). Các tuyên bố trong báo cáo Phase 4 về việc thiết kế đạt chuẩn *"Timing Closure WNS > 0, TNS = 0 tại 200 MHz"* là chưa có cơ sở kiểm chứng kỹ thuật.
3. **Cảnh báo tổng hợp (Synthesis Warnings):** Các hàm hiển thị hệ thống `$display` và `$error` xuất hiện rải rác trong các khối `always_ff` của `score_v_gemm.sv`, `softmax_lut.sv` và `ping_pong_bram_buffer.sv` sẽ phát sinh hàng loạt cảnh báo trong quá trình tổng hợp Vivado RTL Elaboration.

---

## 6. BẢNG MA TRẬN TRUY VẾT TỔNG HỢP CÁC KHUYẾT TẬT

| ID | Module / Tệp | Loại khuyết tật | Mức độ | Nguyên nhân cốt lõi (Root Cause) | Hậu quả kỹ thuật |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **FL-01** | `attention_core_top.sv` | Kiến trúc / Tích hợp | **CRITICAL** | `dot_product_out` thả nổi; thay bằng vòng lặp combinational Row 0 | Mảng Systolic Array bị vô hiệu hóa; chỉ tính 1 hàng token Q |
| **FL-02** | `attention_core_top.sv` | Khả năng tổng hợp | **HIGH** | Dùng khối `initial` với `$readmemh` nạp file hex vào mảng thanh ghi | Phi tổng hợp trên FPGA cứng; tiêu tốn tài nguyên FF/LUT bất thường |
| **FL-03** | `systolic_mac_array.sv` | Cấu trúc toán học | **CRITICAL** | Cộng gộp `pe_prod_ab + pe_prod_ac` vào 1 bộ tích lũy | Trộn lẫn 2 cột dữ liệu độc lập; sai công thức nhân ma trận |
| **FL-04** | `systolic_mac_array.sv` | Thuật toán GEMM | **CRITICAL** | Dùng `pipelined_adder_tree` cộng gộp các PE theo hàng ngang | Triệt tiêu ma trận kết quả thành một đại lượng vô hướng |
| **FL-05** | `softmax_lut.sv` | Số học Fixed-Point | **CRITICAL** | Tử số bộ chia quá nhỏ so với mẫu số, sau đó dịch phải `>> 16` | Đầu ra Softmax bằng 0 tuyệt đối cho tất cả 256 tokens |
| **FL-06** | `ping_pong_bram_buffer.sv`| Giao tiếp I/O | **CRITICAL** | Bus dữ liệu 64-bit nhưng BRAM chỉ ghi `[31:0]` | Mất mát 50% dữ liệu nạp từ bus AXI4-Stream |
| **FL-07** | `attention_core_top.sv` | Điều khiển logic | **HIGH** | `ram_target_sel` gán cứng `2'b00`, `raddr_*` gán cứng `0` | Không thể nạp ma trận K và V qua DMA; địa chỉ đọc BRAM bị treo |
| **FL-08** | `score_v_gemm.sv` | Logic tuần tự | **CRITICAL** | FSM phát `m_axis_tlast` và dừng lại ngay sau token đầu tiên | Thiếu 255 tokens (99.6% ma trận đặc trưng đầu ra) |
| **FL-09** | `tb_attention_core.sv` | Tính hợp lệ Testbench| **CRITICAL** | Nhánh `egress_cnt == 0` in ra PASS; ngưỡng sai số `diff <= 1` lỏng lẻo | Báo cáo PASS giả tạo khi mạch bị deadlock hoặc xuất toàn số 0 |
| **FL-10** | `export_golden_vectors.py`| Chất lượng Vector | **CRITICAL** | Ép kiểu xác suất qua UINT8 với $N=256$ làm tất cả phần tử làm tròn thành 1 | Bộ Golden Vectors suy thoái thành ma trận toàn số 1 |
| **FL-11** | `base.xdc` | Ràng buộc vật lý | **MEDIUM** | Thiếu lệnh `create_clock` định nghĩa xung nhịp 200 MHz | Không thể phân tích Timing Closure (WNS/TNS) trong Vivado |
| **FL-12** | `axi_lite_regs.sv` | Khả năng điều khiển | **LOW** | Các thanh ghi CSR `reg_out_m`, `reg_out_e`, `reg_nbytes` không nối vào lõi | Driver PYNQ trên ARM không kiểm soát được các tham số tái lượng tử |

---

## 7. LỘ TRÌNH KHẮC PHỤC KỸ THUẬT (ACTIONABLE REMEDIATION ROADMAP)

Để đưa lõi tăng tốc Multi-Head Attention đạt trạng thái thực thi hoàn hảo (Bit-Exact thực sự và chạy được trên bo mạch AMD Kria KV260), nhóm đề xuất kế hoạch hành động 4 bước kỹ thuật:

```
[BƯỚC 1: CHUẨN HÓA PYTHON GOLDEN MODEL]
  ├── Đồng bộ hàm Softmax trong export_golden_vectors.py đúng chuẩn 16-bit IntSoftmax
  ├── Xuất lại 7 tệp .hex có phân phối xác suất động học chuẩn (không suy thoái thành số 1)
  └── Đảm bảo Golden Context Out phản ánh đúng ma trận Attention

[BƯỚC 2: SỬA LỖI SỐ HỌC VÀ LOGIC KHỐI SOFTMAX & BRAM]
  ├── Sửa bộ chia trong softmax_lut.sv: Định dạng Q16.16 hoặc mở rộng tử số lên 48-bit
  ├── Cập nhật ping_pong_bram_buffer.sv: Đóng gói đúng 64-bit (2 chu kỳ 32-bit hoặc mở rộng RAM thành 64-bit)
  └── Bổ sung FSM điều phối nạp tuần tự Q -> K -> V qua AXI-Stream

[BƯỚC 3: TÁI TÍCH HỢP VÀ SỬA CẤU TRÚC SYSTOLIC ARRAY TOP-LEVEL]
  ├── Tách riêng 2 bộ tích lũy cho Cột 0 và Cột 1 trong systolic_mac_array.sv
  ├── Loại bỏ cây cộng ngang sai quy tắc; bổ sung bộ đệm trích xuất ma trận kết quả
  ├── Xóa bỏ hoàn toàn mã gán cứng initial $readmemh và vòng lặp comb Row 0 trong top-level
  └── Kết nối đúng đầu ra Systolic Array vào scale_unit -> softmax_lut -> score_v_gemm

[BƯỚC 4: HOÀN THIỆN TESTBENCH & TỔNG HỢP VIVADO 200 MHZ]
  ├── Viết lại tb_attention_core.sv: Nạp đủ Q, K, V qua AXI-Stream; so sánh đủ 4,096 bytes Egress
  ├── Bỏ điều kiện PASS giả tạo (egress_cnt == 0); áp dụng kiểm tra nghiêm ngặt
  ├── Bổ sung constraint xung nhịp 200 MHz trong base.xdc
  └── Chạy Vivado Synthesis & Implementation trên chip xck26 kiểm tra WNS > 0 và DSP/BRAM utilization
```

---

*Báo cáo kiểm toán được lưu trữ chính thức tại `model_notebook/ivit_surgery/REPORT_PHASE4_RTL_COMPREHENSIVE_AUDIT_FLAWS.md` nhằm bảo đảm tính minh bạch, chính xác và khoa học cho toàn bộ hệ thống tăng tốc phần cứng của đồ án.*
