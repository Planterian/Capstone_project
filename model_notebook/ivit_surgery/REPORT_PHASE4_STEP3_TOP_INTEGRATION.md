# BÁO CÁO NGHIỆM THU BƯỚC 3: TÁI TÍCH HỢP & CẤU TRÚC LẠI TOP-LEVEL MHA ACCELERATOR (`rtl_ver02`)

> **Dự án:** Tăng tốc phần cứng MobileViT-XXS Self-Attention trên bo mạch AMD Kria KV260  
> **Thời gian hoàn thành:** 2026-10-13  
> **Trạng thái:** **100% PASS - HOÀN THÀNH TOÀN BỘ TIÊU CHÍ BƯỚC 3 & ĐẠT BIT-EXACT 4,096/4,096 SAMPLES**

---

## 1. TỔNG QUAN KẾT QUẢ THỰC HIỆN BƯỚC 3

Bước 3 đã giải quyết triệt để toàn bộ các khuyết tật kiến trúc cốt lõi (FL-03, FL-04, FL-08) và tái tích hợp hoàn chỉnh hệ thống RTL Top-Level từ luồng Ingress đến Egress:

1. **Khắc phục FL-03 & FL-04 trong [`systolic_mac_array.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/systolic_mac_array.sv):**
   - Tách độc lập bộ tích lũy `pe_acc_col0` (cột chẵn) và `pe_acc_col1` (cột lẻ) cho từng Processing Element (PE).
   - Loại bỏ hoàn toàn cây cộng ngang `pipelined_adder_tree` sai bản chất toán học.
   - Xây dựng mạch giải tỏa tuần tự `dot_product_stream` với điều khiển `drain_start` / `drain_done`.
   - **Kết quả Unit Test:** 100% PASS trên [`tb_systolic_mac_array.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_systolic_mac_array.sv).

2. **Khắc phục FL-08 trong [`score_v_gemm.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/score_v_gemm.sv):**
   - Chuẩn hóa độ rộng xác suất `PROB_WIDTH = 16` và dịch bit số học `>>> 15` đồng bộ với tổng xác suất IntSoftmax ($\approx 32,768$).
   - Triển khai bộ đếm hàng `row_token_cnt` chạy trọn vẹn $0 \dots 255$ tokens.
   - Xuất dữ liệu chuẩn giao thức AXI4-Stream với kiểm soát bắt tay `tready`: 2 beats 64-bit cho mỗi hàng token, tổng cộng **512 beats = 4,096 bytes**.
   - Cờ `m_axis_tlast` được định thời chính xác chỉ bật lên ở beat 511 (beat cuối cùng của ma trận).
   - **Kết quả Unit Test:** 100% BIT-EXACT trên [`tb_score_v_gemm.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_score_v_gemm.sv) (4,096/4,096 bytes matched, 0 errors, 0 tlast errors).

3. **Tái cấu trúc tổng thể Top-Level [`attention_core_top.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/attention_core_top.sv):**
   - **Xóa bỏ 100% mã phi tổng hợp:** Loại bỏ các mảng RAM nạp bằng `$readmemh("q_matrix.hex", ...)`. Thay thế bằng bộ đệm Block RAM trên chip tổng hợp được (`(* ram_style = "block" *)`), nạp động tự động qua bus AXI-Stream Ingress (`s_axis_tdata`).
   - **Thiết lập Real Dataflow Pipeline xuyên suốt:**
     $$\text{AXI-Stream RX} \rightarrow \text{Skid Buffer} \rightarrow \text{Ping-Pong BRAM} \rightarrow \text{Systolic MAC / GEMM} \rightarrow \text{scale\_unit} \rightarrow \text{softmax\_lut} \rightarrow \text{score\_v\_gemm} \rightarrow \text{AXI-Stream TX}$$
   - **Tích hợp cổng nạp lại LUT Bước 2.5:** Đấu nối trực tiếp các tín hiệu AXI-Lite CSR `cfg_lut_we`, `cfg_lut_addr`, `cfg_lut_wdata` vào Port B của `softmax_lut`.
   - **Đồng bộ hóa tham số Dyadic Scale:** Đấu nối `reg_scale_m` ($M=27640$), `reg_scale_e` ($E=22$), và `reg_head_dim` ($d_k=16, 20, 24$) vào các khối đích.
   - **Chuẩn hóa FSM Sequential Heads ($H=4$):** Quản lý chu trình chạy tuần tự 256 hàng và 4 heads theo kiến trúc Single-Head Core tiết kiệm tài nguyên FPGA.

---

## 2. BẢNG ĐỐI SOÁT & KẾT QUẢ MÔ PHỎNG TÍCH HỢP HỆ THỐNG

### 2.1 Kết quả mô phỏng Top-Level [`tb_attention_core.sv`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Project/src/Hardware/rtl_ver02/tb_attention_core.sv):

```text
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

### 2.2 So sánh chi tiết mẫu dữ liệu (Trích xuất ngẫu nhiên từ kết quả chạy thực tế):

| Token / Chiều | Chỉ số Egress | Kết quả RTL (DUT) | PyTorch Golden Model | Độ sai lệch (Diff) | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Token 0, Dim 0 | `#0000` | `5` | `5` | 0 | **MATCH** |
| Token 0, Dim 1 | `#0001` | `2` | `2` | 0 | **MATCH** |
| Token 0, Dim 2 | `#0002` | `0` | `0` | 0 | **MATCH** |
| Token 0, Dim 3 | `#0003` | `-3` | `-3` | 0 | **MATCH** |
| Token 0, Dim 4 | `#0004` | `-7` | `-7` | 0 | **MATCH** |
| Token 0, Dim 5 | `#0005` | `4` | `4` | 0 | **MATCH** |
| Token 0, Dim 6 | `#0006` | `-5` | `-5` | 0 | **MATCH** |
| Token 0, Dim 7 | `#0007` | `0` | `0` | 0 | **MATCH** |
| Token 0, Dim 8 | `#0008` | `5` | `5` | 0 | **MATCH** |
| Token 0, Dim 9 | `#0009` | `6` | `6` | 0 | **MATCH** |
| Token 0, Dim 10| `#0010` | `4` | `4` | 0 | **MATCH** |
| Token 0, Dim 11| `#0011` | `4` | `4` | 0 | **MATCH** |
| Token 0, Dim 12| `#0012` | `0` | `0` | 0 | **MATCH** |
| Token 0, Dim 13| `#0013` | `10` | `10` | 0 | **MATCH** |
| Token 0, Dim 14| `#0014` | `-1` | `-1` | 0 | **MATCH** |
| Token 0, Dim 15| `#0015` | `-1` | `-1` | 0 | **MATCH** |
| Token 31, Dim 15| `#0511` | `-1` | `-1` | 0 | **MATCH** |
| Token 255, Dim 15| `#4095` | `-1` | `-1` | 0 | **MATCH** |

---

## 3. TIÊU CHÍ NGHIỆM THU BƯỚC 3: ĐÁNH GIÁ CHI TIẾT

| Tiêu chuẩn kỹ thuật | Mục tiêu thiết kế | Kết quả đạt được | Kết luận |
|:---|:---|:---|:---:|
| **Loại bỏ mã phi tổng hợp** | Xóa hoàn toàn `$readmemh` trong Top synthesizable RTL | 100% RTL tổng hợp được, dữ liệu nạp qua AXI-Stream Ingress | **ĐẠT** |
| **Loại bỏ cổng thả nổi** | Không còn cổng unconnected / dead logic | Toàn bộ các module con đều được nối dây đầy đủ | **ĐẠT** |
| **Sửa đổi FL-03 & FL-04** | Độc lập col0/col1, tuần tự hóa drain logic | Unit test `tb_systolic_mac_array` PASS 100% | **ĐẠT** |
| **Sửa đổi FL-08** | Egress 256 tokens, 512 beats, tlast tại beat 511 | Unit test `tb_score_v_gemm` PASS 100% | **ĐẠT** |
| **Tích hợp Top-Level** | Toàn bộ 4 tầng thông suốt | `tb_attention_core` đạt 4,096/4,096 mẫu khớp chính xác | **ĐẠT** |
| **Độ chính xác số học** | Sai số 0 LSB so với PyTorch Bit-Exact Model | 0 lỗi trên toàn bộ 4,096 mẫu đầu ra | **ĐẠT** |

---

## 4. KẾT LUẬN & SẴN SÀNG CHO BƯỚC 4

Toàn bộ hệ thống phần cứng RTL của Self-Attention Accelerator (`rtl_ver02`) hiện đã đạt trạng thái **hoàn thiện, đồng bộ hóa tuyệt đối và sẵn sàng 100% để tiến hành Bước 4**:
- Đóng gói file ràng buộc thời gian 200 MHz (`base.xdc`).
- Tổng hợp logic (Synthesis) và Implementation trên AMD Vivado nhằm đóng gói Bitstream hoàn chỉnh cho Kria KV260.
