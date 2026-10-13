# KẾ HOẠCH THỰC THI BƯỚC 2.5: MÔ HÌNH PRELOAD SOFTWARE-DEFINED LOOKUP TABLE — HARDWARE-ACCELERATED
## Thiết kế Kiến trúc Softmax Nạp Động Linh hoạt Đa Stage (Stage 2, 3, 4) và Hỗ trợ Fine-Tuning trên AMD Kria KV260

**Mã tài liệu:** `PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT`  
**Ngày khởi tạo:** 13/10/2026 — 22:20:00 (GMT+7)  
**Vị trí lưu trữ:** [`model_notebook/ivit_surgery/PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md)  
**Tài liệu tham chiếu:**
1. Kế hoạch gốc: [`PLAN_PHASE4_REMAINING_STEPS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md)
2. Báo cáo nghiệm thu Bước 2: [`REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md)
3. Giải trình LUT: [`EXPLANATION_SOFTMAX_LUT_UPGRADE.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/EXPLANATION_SOFTMAX_LUT_UPGRADE.md)
4. Tham số cấu hình: [`dyadic_scale_config.json`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/dyadic_scale_config.json)

---

## 1. BỐI CẢNH & ĐỘNG LỰC KỸ THUẬT CỦA BƯỚC 2.5

### 1.1 Hiện trạng đã hoàn tất tại Bước 2 (Nền tảng số học Bit-Exact vững chắc):
Trong Bước 2 vừa qua, nhóm kỹ thuật đã xử lý triệt để lỗi số học triệt tiêu đầu ra về 0 (FL-05) và lỗi mất 50% dữ liệu nạp AXI-Stream (FL-06, FL-07). Kết quả kiểm thử iverilog độc lập cho thấy:
* Khối `softmax_lut.sv` đạt **100% Bit-Exact (0-LSB Error)** trên toàn bộ ma trận $256 \times 256$ ($65,536$ tokens).
* Khối `ping_pong_bram_buffer.sv` vượt qua $20/20$ tiêu chí kiểm thử, bảo toàn toàn vẹn $12,288\text{ bytes}$ dữ liệu Ingress $Q \rightarrow K \rightarrow V$.

### 1.2 Nút thắt kiến trúc cần giải quyết tại Bước 2.5:
Mặc dù Bước 2 đã đạt độ chính xác bit-exact tuyệt đối cho **Stage 2 ($d_k=16$)**, bảng tra cứu `exp_rom` trong `softmax_lut.sv` hiện đang được khởi tạo dạng **bảng tĩnh cứng (Static Pre-initialized BRAM)**.
Điều này dẫn đến các hạn chế kiến trúc:
1. **Khó khăn khi chuyển đổi Stage:** MobileViT-XXS có 3 tầng Transformer khác nhau về chiều Head và hệ số thang đo:
   - **Stage 2 ($d_k=16$):** $S_{\text{score}} = 0.1275113 \implies x_0 = -8$
   - **Stage 3 ($d_k=20$):** $S_{\text{score}} = 0.0570586 \implies x_0 = -17$
   - **Stage 4 ($d_k=24$):** $S_{\text{score}} = 0.0503187 \implies x_0 = -19$
   Nếu bảng `exp_rom` bị đóng băng tĩnh theo Stage 2, phần cứng sẽ tính sai đường cong số mũ khi chuyển sang Stage 3 và Stage 4.
2. **Khó khăn khi Fine-Tuning mô hình:** Khi người dùng huấn luyện tinh chỉnh (fine-tune) mô hình trên tập dữ liệu ảnh mới, các hệ số thang đo lượng tử hóa ($S_{\text{score}}$) sẽ thay đổi. Với bảng ROM tĩnh, người dùng bắt buộc phải sửa file SystemVerilog và chạy lại luồng tổng hợp Vivado ($30 \dots 45\text{ phút}$).
3. **Mục tiêu giải pháp Bước 2.5:** Chuyển dịch toàn diện từ *ROM tĩnh* sang kiến trúc công nghiệp chuẩn:  
   **"Software-Defined, Hardware-Accelerated" (Preload Runtime BRAM qua bus AXI)**:
   - Phần mềm (Python PYNQ / C++) tính trước 256 giá trị bảng mũ từ hệ số $S_{\text{score}}$ lúc khởi động mô hình (Model Warm-up).
   - Nạp các giá trị này vào BRAM hai cổng (Dual-Port BRAM) qua bus AXI-Lite trong thời gian micro-giây ($< 2\text{ µs}$).
   - Phần cứng thực thi suy luận thời gian thực với độ trễ tối thiểu $1\text{ chu kỳ}$ ở tần số $200\text{ MHz}$ mà **không cần tổng hợp lại Bitstream**.

---

## 2. THIẾT KẾ KIẾN TRÚC KỸ THUẬT BƯỚC 2.5

```
===================================================================================================
             KIẾN TRÚC SOFTWARE-DEFINED LOOKUP TABLE (RELOADABLE BRAM)
===================================================================================================

 [HOST CPU / ARM Cortex-A53 (PYNQ Python)]
        │
        ├── (1) Đọc dyadic_scale_config.json lúc load_model()
        ├── (2) Tính toán 256 phần tử int_exp_shift cho từng Stage
        │
        ▼ Bus AXI4-Lite CSR
 [axi_lite_regs.sv]
        │
        ├── Ghi thanh ghi: reg_head_dim, reg_scale_m, reg_scale_e
        └── Window ghi cấu hình BRAM (Port B: clk, we, addr, din)
                     │
                     ▼
             [softmax_lut.sv]
        +-----------------------------------------------------------+
        |                  TRUE DUAL-PORT BRAM                      |
        |              1024 words x 19-bit (RAMB18K)                |
        +-----------------------------+-----------------------------+
        |   PORT B: WRITE CONFIG      |    PORT A: READ INFERENCE   |
        |   (Từ Host ARM qua AXI)     |    (Từ Softmax Datapath)    |
        |  - xung nhịp clk            |  - xung nhịp clk (200 MHz)  |
        |  - we_b (ghi 1 word/cycle)  |  - read_addr (1 cycle delay)|
        |  - addr_b (0 .. 1023)       |  - out_exp_data (19-bit)    |
        |  - data_in (19-bit)         |  - Zero DSP, O(1) latency   |
        +-----------------------------+-----------------------------+
                     ▲                              │
                     │                              ▼
         Nạp bảng trước lúc khởi động     Bộ nhân chuẩn hóa 16-bit
         (Model Initialization)           out_prob = (exp * inv_sum) >> 16
===================================================================================================
```

### 2.1 Cấu trúc Phân vùng Bộ nhớ Đa Stage (Memory Map Partitioning)
1 lát RAMB18K trên chip AMD Kria KV260 có cấu hình $1024\text{ words} \times 18\text{b}/36\text{b}$ hoặc $2048\text{ words}$. Ta phân bổ không gian 1024 entries thành 4 vùng phân biệt:

| Vùng nhớ BRAM (Word Addr) | Dung lượng | Mục đích sử dụng | Hệ số $S_{\text{score}}$ mặc định |
| :---: | :---: | :--- | :---: |
| `0x000` $\dots$ `0x0FF` ($0 \dots 255$) | 256 words | **Stage 2 ($d_k=16$)** Attention Core | `0.1275113` |
| `0x100` $\dots$ `0x1FF` ($256 \dots 511$) | 256 words | **Stage 3 ($d_k=20$)** Attention Core | `0.0570586` |
| `0x200` $\dots$ `0x2FF` ($512 \dots 767$) | 256 words | **Stage 4 ($d_k=24$)** Attention Core | `0.0503187` |
| `0x300` $\dots$ `0x3FF` ($768 \dots 1023$)| 256 words | Dự phòng cho Fine-Tuning / Custom Task | Tùy biến |

### 2.2 Cơ chế Tự động chọn Phân vùng theo `head_dim` (Run-time Multi-Stage Routing)
Tại cổng đọc Port A trong `softmax_lut.sv`:
Địa chỉ đọc thực tế được kết hợp giữa mã Stage (dựa trên thanh ghi `head_dim` của CSR) và độ lệch nguyên $\Delta s$:
```systemverilog
logic [9:0] rom_addr_stage;

always_comb begin
    case (head_dim)
        16'd16:  rom_addr_stage = {2'b00, rom_addr}; // Stage 2: Offset 0x000
        16'd20:  rom_addr_stage = {2'b01, rom_addr}; // Stage 3: Offset 0x100
        16'd24:  rom_addr_stage = {2'b10, rom_addr}; // Stage 4: Offset 0x200
        default: rom_addr_stage = {2'b00, rom_addr}; // Mặc định
    endcase
end

// Port A: Đọc siêu tốc 1 chu kỳ phục vụ suy luận
always_ff @(posedge clk) begin
    rom_data_exp <= exp_bram[rom_addr_stage];
end
```

### 2.3 Cổng nạp Cấu hình Port B (Write Configuration Port)
```systemverilog
// Port B: Nạp tham số động từ CPU qua bus AXI-Lite
always_ff @(posedge clk) begin
    if (cfg_bram_we) begin
        exp_bram[cfg_bram_addr] <= cfg_bram_wdata[18:0];
    end
end
```
* **Thời gian nạp:** Ghi liên tiếp 256 words chỉ mất $256 \times 5.0\text{ ns} = \mathbf{1.28\text{ µs}}$.
* **Độc lập hoàn toàn:** Port B và Port A chạy độc lập; việc nạp bảng cấu hình diễn ra trước khi bật cờ `start`, không gây xung đột xung nhịp hay làm giảm tần số $200\text{ MHz}$.

---

## 3. KẾ HOẠCH TÁC ĐỘNG & ĐỒNG BỘ CÁC TÀI LIỆU THIẾT KẾ LIÊN QUAN

Dưới đây là kế hoạch chi tiết về việc cập nhật và điều chỉnh các tài liệu thiết kế của dự án để đảm bảo tính nhất quán của toàn bộ hệ thống:

### 3.1 Kế hoạch sửa đổi [`PLAN_PHASE4_REMAINING_STEPS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md)
* **Bổ sung Bước 2.5 vào sơ đồ tổng quan lộ trình:**
  Chèn Bước 2.5 vào giữa Bước 2 và Bước 3:
  ```
  [BƯỚC 2: XONG ✓]  Sửa lỗi Số học Softmax & Ping-Pong BRAM (Bit-Exact Stage 2). Đã nghiệm thu.
        │
        ▼
  [BƯỚC 2.5: THIẾT KẾ MỚI] Kiến trúc Software-Defined Preload BRAM LUT & Multi-Stage Scalability.
        ├── 2.5.1: Mở rộng BRAM exp_rom thành True Dual-Port 1024 words trong softmax_lut.sv.
        ├── 2.5.2: Thêm AXI-Lite BRAM Write Window trong axi_lite_regs.sv.
        └── 2.5.3: Mở rộng dyadic_scale_config.json chứa bảng 3 Stage phục vụ PYNQ preload.
        │
        ▼
  [BƯỚC 3] Tái tích hợp & Cấu trúc lại Mảng Systolic Array Top-Level...
  ```
* **Cập nhật giao tiếp Bước 3:** Trong Bước 3, khi tích hợp `attention_core_top.sv`, nối các chân Port B của `softmax_lut` sang khối `axi_lite_regs` để CPU PYNQ điều khiển được.

### 3.2 Kế hoạch sửa đổi [`REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/REPORT_PHASE4_STEP2_SOFTMAX_BRAM.md)
* Bổ sung ghi chú chuyển tiếp: Xác nhận Bước 2.0 đã hoàn thành xuất sắc việc giải quyết lỗi số học cơ bản (FL-05) và xác minh bit-exact trên Stage 2.
* Tham chiếu sang `PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md` như là bản nâng cấp kiến trúc mở rộng tính năng Scalable cho toàn bộ hệ thống.

### 3.3 Kế hoạch cập nhật [`dyadic_scale_config.json`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/dyadic_scale_config.json)
* Hiện tại file JSON chỉ lưu các con số vô hướng (`M_scale`, `E_scale`, `S_score`).
* **Kế hoạch mở rộng:** Thêm trường mảng dữ liệu tiền tính toán `exp_rom_table` (256 phần tử) cho cả 3 Stage (`STAGE2`, `STAGE3`, `STAGE4`):
  ```json
  {
    "stages": {
      "STAGE2": {
        "head_dim": 16,
        "M_scale": 27640,
        "E_scale": 22,
        "exp_rom_table": [262144, 245760, 229376, ...]
      },
      "STAGE3": {
        "head_dim": 20,
        "M_scale": 21936,
        "E_scale": 22,
        "exp_rom_table": [262144, 253952, 245760, ...]
      },
      "STAGE4": {
        "head_dim": 24,
        "M_scale": 18810,
        "E_scale": 22,
        "exp_rom_table": [262144, 258048, 249856, ...]
      }
    }
  }
  ```
* Tệp này sẽ đóng vai trò là "Payload nhị phân" để ứng dụng Python trên ARM CPU chỉ việc đọc và ghi thẳng vào phần cứng.

### 3.4 Kế hoạch cập nhật [`export_golden_vectors.py`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/export_golden_vectors.py)
* Bổ sung chức năng tự động sinh dữ liệu `exp_rom_table` cho cả 3 Stage và xuất đồng thời ra:
  1. `dyadic_scale_config.json` (định dạng JSON cho PYNQ Python).
  2. `exp_rom_all_stages.hex` (định dạng Hex cho Testbench Verilog simulation).

---

## 4. KẾ HOẠCH SỬA ĐỔI MÃ NGUỒN RTL & TESTBENCH CHO BƯỚC 2.5

> [!IMPORTANT]
> **Quy tắc thực hiện:** Toàn bộ công việc sửa đổi mã nguồn phần cứng dưới đây được **lên kế hoạch chặt chẽ** và sẽ chỉ được triển khai thực hiện sau khi người dùng phê duyệt tài liệu kế hoạch Bước 2.5 này.

### 4.1 Module [`softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/softmax_lut.sv)
* Nâng tham số `ROM_DEPTH = 1024` (thay vì 256).
* Bổ sung cổng giao tiếp cấu hình Port B:
  ```systemverilog
  // Configuration Port B Interface (AXI-Lite write access)
  input  logic        cfg_we,
  input  logic [9:0]  cfg_addr,
  input  logic [18:0] cfg_wdata,
  input  logic [15:0] head_dim
  ```
* Mảng BRAM được khởi tạo mặc định bằng dữ liệu của cả 3 Stage (vừa hoạt động độc lập trong mô phỏng, vừa cho phép ghi đè động từ CPU).

### 4.2 Module [`axi_lite_regs.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/axi_lite_regs.sv)
* Mở rộng không gian thanh ghi AXI4-Lite:
  - `0x40`: Thanh ghi địa chỉ cấu hình BRAM `REG_LUT_ADDR` (10-bit).
  - `0x44`: Thanh ghi dữ liệu cấu hình BRAM `REG_LUT_DATA` (19-bit).
  - Khi CPU ghi vào `0x44`, module tự động phát xung `cfg_we = 1'b1` trong 1 chu kỳ để ghi vào BRAM.

### 4.3 Testbench Xác minh Độc lập [`tb_softmax_lut_reloadable.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_softmax_lut_reloadable.sv)
* Xây dựng kịch bản kiểm thử 3 kịch bản:
  1. **Kịch bản 1 (Multi-Stage Auto Select):** Giữ nguyên bảng mặc định, đổi `head_dim` lần lượt $16 \rightarrow 20 \rightarrow 24$, xác nhận Softmax tự trỏ sang đúng vùng bảng của từng Stage.
  2. **Kịch bản 2 (Runtime Reloading Test):** Mô phỏng CPU ghi nạp 256 giá trị mới qua Port B, sau đó chạy xử lý một hàng token và kiểm tra đầu ra khớp bit-exact với bảng mới nạp.
  3. **Kịch bản 3 (Zero Collision Test):** Xác nhận việc đọc Port A khi suy luận không bị ảnh hưởng bởi logic Port B, đảm bảo Timing $200\text{ MHz}$.

---

## 5. TIÊU CHÍ NGHIỆM THU BƯỚC 2.5 (ACCEPTANCE CRITERIA)

| Hạng mục kiểm tra | Tiêu chuẩn ĐẠT yêu cầu | Phương pháp kiểm chứng |
| :--- | :--- | :--- |
| **Dung lượng BRAM** | Không vượt quá 1 lát RAMB18K ($1024 \times 19$-bit) | Vivado Elaboration & Resource Report |
| **Độ trễ suy luận Port A** | Giữ vững đúng $1\text{ chu kỳ}$ xung nhịp ($5.0\text{ ns}$) | Iverilog Waveform Simulation |
| **Tốc độ nạp Port B** | Nạp xong 256 words trong vòng $\le 2\text{ µs}$ | Testbench Timing Analysis |
| **Độ chính xác Stage 2** | Duy trì **100% Bit-Exact (0-LSB Error)** như Bước 2 | Chạy lại `tb_golden_softmax_lut.sv` |
| **Khả năng tự đổi Stage** | Đổi `head_dim = 20` hoặc `24` tự động trỏ đúng vùng bảng | Simulation Log Check |

---

## 6. KẾT LUẬN & HƯỚNG DẪN BƯỚC TIẾP THEO

Bản kế hoạch kỹ thuật Bước 2.5 này hoàn thiện một mắt xích thiết kế cực kỳ quan trọng, đưa bộ tăng tốc Attention Core của Capstone từ mức **mạch chuyên dụng tĩnh (Hardcoded)** lên đẳng cấp **IP Core công nghiệp có khả năng tái cấu trúc mềm dẻo (Software-Defined Reconfigurable Core)**.

Sau khi bạn xem xét và phê duyệt bản kế hoạch này:
1. Nhóm kỹ thuật sẽ cập nhật đồng bộ các tài liệu thiết kế ([`PLAN_PHASE4_REMAINING_STEPS.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_REMAINING_STEPS.md) và `dyadic_scale_config.json`).
2. Tiến hành triển khai mã nguồn RTL cho Bước 2.5 và chạy kiểm thử xác nhận.
3. Bàn giao thiết kế hoàn chỉnh vững chắc để tự tin bước vào **BƯỚC 3** (Tái tích hợp Systolic Array Top-Level).

---
*Tài liệu kế hoạch được lưu trữ chính thức tại `model_notebook/ivit_surgery/PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md`.*
