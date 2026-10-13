# BÁO CÁO NGHIỆM THU BƯỚC 2.5: KIẾN TRÚC RELOADABLE DUAL-PORT BRAM SOFTMAX VÀ MULTI-STAGE SCALABILITY
## Hiện thực hóa Mô hình "Software-Defined, Hardware-Accelerated" cho MobileViT-XXS trên AMD Kria KV260

**Mã tài liệu:** `REPORT_PHASE4_STEP2_5_RECONFIGURABLE_LUT`  
**Ngày nghiệm thu:** 13/10/2026 — 22:45:00 (GMT+7)  
**Tác giả:** Đội ngũ Kỹ sư Thiết kế RTL & Hệ thống Tăng tốc AI (Antigravity Senior Agent)  
**Tệp kế hoạch tham chiếu:** [`PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/PLAN_PHASE4_STEP2_5_RECONFIGURABLE_LUT.md)  
**Trạng thái kiểm tra:** **100% PASSED (Toàn bộ 5/5 tiêu chuẩn kiểm tra đạt hoàn hảo)**  

---

## 1. TỔNG QUAN KẾT QUẢ THỰC HIỆN BƯỚC 2.5

Bước 2.5 đã chuyển đổi thành công module tính toán phi tuyến Softmax từ **ROM tĩnh cứng (Static Hardcoded Table)** sang **Kiến trúc BRAM Hai cổng nạp động (True Dual-Port Reconfigurable BRAM)** theo đúng chuẩn thiết kế IP Core công nghiệp.

### Các thành quả kỹ thuật cốt lõi đạt được:
1. **Bảo tồn toàn vẹn 100% cấu hình và nguồn gốc số học:**  
   * Bổ sung trường dữ liệu `exp_rom_table` (256 phần tử số nguyên cho mỗi Stage) vào [`dyadic_scale_config.json`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/dyadic_scale_config.json).
   * Giữ nguyên vẹn 100% các trường tỉ lệ thực gốc: `S_Q`, `S_score`, `S_effective`, `M_scale`, `E_scale`, `head_dim` phục vụ đối soát toán học và tự động tính lại (re-computation).
   * Tự động sinh tệp nhị phân nạp đa Stage [`exp_rom_all_stages.hex`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/exp_rom_all_stages.hex) chứa $1,024\text{ words} \times 20\text{-bit}$.
2. **Nâng cấp True Dual-Port BRAM trong [`softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/softmax_lut.sv):**
   * **Độ sâu:** Mở rộng từ $256 \rightarrow 1,024\text{ words}$ (phân bổ thành 4 phân vùng 256 từ: Stage 2, Stage 3, Stage 4 và Phân vùng dự phòng Fine-Tuning).
   * **Độ rộng từ:** Mở rộng từ $19\text{-bit} \rightarrow 20\text{-bit}$ (đáp ứng trọn vẹn giá trị đỉnh của Stage 3 là $589,824$ và Stage 4 là $655,360$ mà không bị tràn bit).
   * **Port A (Inference Read):** Tự động ánh xạ địa chỉ theo `head_dim` ($16 \rightarrow \text{offset } 0$, $20 \rightarrow \text{offset } 256$, $24 \rightarrow \text{offset } 512$). Độ trễ đọc đúng $1\text{ chu kỳ}$ xung nhịp ($5.0\text{ ns}$ ở $200\text{ MHz}$).
   * **Port B (Config Write):** Nhận bus ghi trực tiếp từ AXI4-Lite (`cfg_we`, `cfg_addr[9:0]`, `cfg_wdata[19:0]`).
3. **Mở rộng thanh ghi CSR AXI4-Lite trong [`axi_lite_regs.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/axi_lite_regs.sv):**
   * Bổ sung thanh ghi `ADDR_LUT_ADDR` (`0x40`): Ghi địa chỉ từ BRAM ($0 \dots 1023$).
   * Bổ sung thanh ghi `ADDR_LUT_DATA` (`0x44`): Ghi dữ liệu 20-bit; tự động phát xung ghi `cfg_lut_we = 1'b1` trong đúng 1 chu kỳ để nạp vào BRAM.
4. **Kiểm chứng độc lập và hồi quy:**
   * Tạo mới testbench [`tb_softmax_lut_reloadable.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_softmax_lut_reloadable.sv) xác minh chuyển đổi Stage và nạp đè thời gian thực: **5/5 Checks PASS**.
   * Chạy lại toàn bộ testbench hồi quy [`tb_golden_softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_golden_softmax_lut.sv): **100% Khớp Bit-Exact (0-LSB Error)** so với PyTorch golden model trên Stage 2.
   * Chạy lại testbench hồi quy [`tb_ping_pong_bram_buffer.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_ping_pong_bram_buffer.sv): **20/20 Checks PASS**.

---

## 2. KẾT QUẢ MÔ PHỎNG KIỂM THỬ THỰC TẾ

### 2.1 Kiểm thử Module Nạp Động [`tb_softmax_lut_reloadable.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_softmax_lut_reloadable.sv)

```
=======================================================================
   TESTBENCH: STEP 2.5 RECONFIGURABLE MULTI-STAGE BRAM SOFTMAX LUT
=======================================================================

[TEST SCENARIO 1]: Multi-Stage Partition Auto-Selection (dk = 16, 20, 24)
  -> STAGE 2 (dk=16): Prob Sum = 32698, Peak Token 0 Prob = 3604
     [CHECK 1.1 PASSED]: Stage 2 Partition (Offset 0x000) Validated.
  -> STAGE 3 (dk=20): Prob Sum = 32525, Peak Token 0 Prob = 1674
     [CHECK 1.2 PASSED]: Stage 3 Partition (Offset 0x100) Validated.
  -> STAGE 4 (dk=24): Prob Sum = 32683, Peak Token 0 Prob = 1520
     [CHECK 1.3 PASSED]: Stage 4 Partition (Offset 0x200) Validated.

[TEST SCENARIO 2]: Runtime Reloading via Port B (Host CPU AXI-Lite write)
  -> Preloading 256 custom uniform words into Stage 2 BRAM via Port B...
  -> Preload completed (256 words written).
  -> Modified STAGE 2 Output: Prob Sum = 32512, Peak Prob = 127
     [CHECK 2.1 PASSED]: Runtime Reload Successful! Exact uniform prob = 127/token (Sum=32512).

[TEST SCENARIO 3]: Restoring Original Stage 2 Calibrated Table from Hex...
  -> Restoration completed. Verifying bit-exact response...
  -> Restored STAGE 2 Output: Prob Sum = 32698, Peak Token 0 Prob = 3604
     [CHECK 3.1 PASSED]: Original Calibrated Weights Restored & Functional.

=======================================================================
   FINAL REPORT: STEP 2.5 RECONFIGURABLE BRAM VERIFICATION
=======================================================================
   Total Checks Evaluated : 5
   Checks Passed          : 5
   Checks Failed          : 0
   >>> STATUS: ALL STEP 2.5 TESTS PASSED (100% SUCCESS)! <<<
=======================================================================
```

### 2.2 Kiểm thử Hồi quy Bit-Exact [`tb_golden_softmax_lut.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_golden_softmax_lut.sv)

```
=======================================================================
   FINAL VERIFICATION REPORT: SOFTMAX_LUT vs GOLDEN MODEL (STAGE 2)
=======================================================================
   Total Softmax Tokens Tested : 1024
   Matched Tokens (0 LSB Error) : 1024
   Mismatched Tokens (> 0 LSB) : 0
   >>> STATUS: 100% BIT-EXACT MATCH (Zero Error)! [PASS] <<<
=======================================================================
```

---

## 3. BẢNG PHÂN BỔ BỘ NHỚ VÀ ĐỊA CHỈ CSR HOÀN CHỈNH

### 3.1 Bản đồ Không gian BRAM Softmax ($1,024\text{ Words} \times 20\text{-bit}$)
| Dải địa chỉ Word | Kích thước | Tầng tương ứng | `head_dim` | Hệ số $S_{\text{score}}$ | Giá trị Đỉnh ($d=0$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0x000` $\dots$ `0x0FF` ($0 \dots 255$) | 256 words | **Stage 2** | `16` | `0.1275113` | $262,144$ ($2^{18}$) |
| `0x100` $\dots$ `0x1FF` ($256 \dots 511$) | 256 words | **Stage 3** | `20` | `0.0570586` | $589,824$ ($18 \times 2^{15}$) |
| `0x200` $\dots$ `0x2FF` ($512 \dots 767$) | 256 words | **Stage 4** | `24` | `0.0503187` | $655,360$ ($20 \times 2^{15}$) |
| `0x300` $\dots$ `0x3FF` ($768 \dots 1023$)| 256 words | **Dự phòng / Fine-Tuning** | Tùy biến | Tùy biến | Tùy biến |

### 3.2 Bổ sung Thanh ghi Giao tiếp AXI4-Lite
| Địa chỉ Byte | Tên thanh ghi | Quyền | Mô tả chức năng |
| :---: | :--- | :---: | :--- |
| `0x40` | `ADDR_LUT_ADDR` | R/W | Địa chỉ word trong BRAM Softmax ($0 \dots 1023$, 10-bit) |
| `0x44` | `ADDR_LUT_DATA` | R/W | Dữ liệu 20-bit ghi vào BRAM. Khi ghi vào địa chỉ này, phần cứng tự động phát xung kích hoạt `cfg_lut_we = 1'b1` trong 1 chu kỳ. |

---

## 4. KẾT LUẬN & SẴN SÀNG CHO BƯỚC 3

* **Nhiệm vụ Bước 2.5 đã hoàn thành xuất sắc 100%.**
* Khối Softmax và BRAM Ingress hiện đã trở thành các module độc lập, vững chắc, chuẩn công nghiệp, tự động co giãn theo 3 Stage của MobileViT và cho phép nạp động không cần biên dịch lại bitstream.
* **Hệ thống đã đạt mức độ sẵn sàng và an toàn cao nhất để bước vào BƯỚC 3 (Tái tích hợp Systolic Array & Cấu trúc lại Top-Level).**
