# Kiến trúc Vi mạch MHA v2.0 (MobileViT-XXS Attention Accelerator)

> **Dự án:** Bộ tăng tốc phần cứng Scaled-Dot Attention Core cho Apple MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật đồng bộ hóa Bit-Exact với I-ViT QAT Model và RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/mha_kien_truc_vi_mach.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/mha_kien_truc_vi_mach.md)

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG TOP-LEVEL (attention_core_top)

Sơ đồ kết nối phân định ranh giới giữa **Processing System (ARM Cortex-A53 PS)** và **Programmable Logic (FPGA PL)** trên bo mạch AMD Kria KV260:

```text
========================================================================================================================
                                     TOP-LEVEL RTL ARCHITECTURE: attention_core_top (v2.0)
========================================================================================================================

    ARM PS (Host / PYNQ Driver / Jupyter Lab)
   +---------------------------------------+
   |  Memory-Mapped AXI-Lite               |
   |  Control, Status & BRAM Reload Regs   |
   +-------------------+-------------------+
                       |
                       | (AWADDR, WDATA, ARADDR, RDATA / AXI4-Lite 32-bit)
                       v
   +-------------------------------------------------------------------------------------------------------------------+
   | AXI4-Lite Slave Register File & Central Controller (axi_lite_regs.sv)                                             |
   |   - Reg 0x00: CONTROL [0: START, 1: CLEAR_BUF, 2: SOFT_RESET]  - Reg 0x18: NUM_HEADS (H = 4)                     |
   |   - Reg 0x04: STATUS  [0: BUSY, 1: DONE, 2: ERROR, 3: PP_BANK] - Reg 0x20: SCALE_M (Dyadic Mantissa M = 27640)   |
   |   - Reg 0x10: SEQ_LEN (N = 256)                                - Reg 0x24: SCALE_E (Dyadic Exponent E = 22)      |
   |   - Reg 0x14: HEAD_DIM (d_k = 16, 20, 24)                      - Reg 0x40/0x44: BRAM LUT Reload Window (Step 2.5) |
   +---------------------------------------------------+---------------------------------------------------------------+
                                                       | (Control Signals: start_pulse, cfg_lut_*, reg_scale_*)
                                                       v
   +-------------------------------------------------------------------------------------------------------------------+
   |                                            PIPELINED ATTENTION DATAPATH                                           |
   |                                                                                                                   |
   |  AXI DMA MM2S (DDR4 -> PL)                                                                                        |
   |  +--------------------+                                                                                           |
   |  | AXI4-Stream Ingress|=== (64-bit S_AXIS_TDATA / TVALID / TREADY / TLAST) ===+                                   |
   |  +--------------------+                                                       |                                   |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           | Skid Buffer & Multi-Bank Ingress BRAM  |              |
   |                                                           | (FSM nạp tuần tự Q -> K -> V INT8)     |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               |                                   |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 1: QK^T GEMM Streaming Engine   |              |
   |                                                           |  (Nhân song song d_k chiều -> INT32)   |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT32 Raw Scores: 1 score/cycle) |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 2: Dyadic Scaler Unit (HW-04)   |              |
   |                                                           |  ((Score x M) >>> E -> INT16 Sat Clamped)|            |
   |                                                           +-------------------+--------------------+              |
   |                                                           | (INT16 Scaled Scores [-32768, 32767])                 |
   |                                                           v                                                       |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 3: Reconfigurable Softmax Engine|              |
   |                                                           |  (Max-Sub + True Dual-Port 1024w BRAM  |              |
   |                                                           |   + 32-step Restoring Div -> UINT16)   |              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (UINT16 Probabilities [0, 32767])  |
   |                                                                               v                                   |
   |                                                           +----------------------------------------+              |
   |                                                           |  STAGE 4: Score x V Context GEMM       |              |
   |                                                           |  (Tích lũy 256 tokens, >>> 15, Sat INT8|              |
   |                                                           +-------------------+--------------------+              |
   |                                                                               | (INT8 Context Vectors)            |
   |                                                                               v                                   |
   |  AXI DMA S2MM (PL -> DDR4)                                                    |                                   |
   |  +--------------------+                                                       |                                   |
   |  | AXI4-Stream Egress |<== (64-bit M_AXIS_TDATA / TVALID / TREADY / TLAST) ===+                                   |
   |  +--------------------+                                                                                           |
   +-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. LUỒNG DỮ LIỆU PIPELINE 4 TẦNG (PIPELINE DATAPATH)

Sơ đồ chi tiết luồng xử lý đồng bộ chu kỳ xung nhịp giữa các khối chức năng:

```text
========================================================================================================================
                                     MHA CORE PIPELINE & DATAFLOW ARCHITECTURE
========================================================================================================================

   AXI4-Stream RX  
   S_AXIS_TDATA [63:0]  
   ======/======================================================================================================+  
         |                                                                                                      |  
         v                                                                                                      v  
   +------------------+     +-------------------+     +-------------------+     +------------------+     +------------------+
   |   INPUT BUFFER   |     |  STAGE 1: QK^T    |     |  STAGE 2: DYADIC  |     |  STAGE 3: BRAM   |     |  STAGE 4: SCORE*V|
   |   SKID & BRAM    |     |  DOT-PRODUCT      |     |  SCALER UNIT      |     |  SHIFTMAX ENGINE |     |  CONTEXT GEMM    |
   |                  |     |                   |     |                   |     |                  |     |                  |
   |  +------------+  |     |  +-------------+  |     |  +-------------+  |     |  +------------+  |     |  +------------+  |
   |  | Q_MEM      |  |     |  | Parallel d_k|  |     |  | 48-bit Mult |  |     |  | Max-Sub    |  |     |  | 32-bit Acc |  |
   |  | 256x32 INT8|  |     |  | Multipliers |  |     |  | (Score x M) |  |     |  | Buffer 256 |  |     |  | 32 Chans   |  |
   |  +------------+  |     |  +------+------+  |     |  +------+------+  |     |  +-----+------+  |     |  +-----+------+  |
   |  | K_MEM      |  |     |         |         |     |         |         |     |        |         |     |        |         |
   |  | 256x32 INT8|  |     |         v         |     |         v         |     |        v         |     |        v         |
   |  +------------+  |     |  +-------------+  |     |  +-------------+  |     |  +------------+  |     |  +------------+  |
   |  | V_MEM      |  |     |  | Reduction   |  |     |  | ASR >>> E   |  |     |  | 1024w BRAM |  |     |  | >>> 15      |  |
   |  | 256x32 INT8|  |     |  | Adder Tree  |  |     |  | Sat Clamp   |  |     |  | Exp ROM    |  |     |  | Sat Clamp   |  |
   |  +------------+  |     |  +------+------+  |     |  +------+------+  |     |  +-----+------+  |     |  +-----+------+  |
   |                  |     |         |         |     |         |         |     |        |         |     |        |         |
   |                  |     |         |         |     |         |         |     |        v         |     |        |         |
   |                  |     |         |         |     |         |         |     |  +------------+  |     |        |         |
   |                  |     |         |         |     |         |         |     |  | 32-step Div|  |     |        |         |
   |                  |     |         |         |     |         |         |     |  | 2^31 / Sum |  |     |        |         |
   |                  |     |         |         |     |         |         |     |  +-----+------+  |     |        |         |
   |                  |     |         |         |     |         |         |     |        |         |     |        |         |
   +--------+---------+     +---------+---------+     +---------+---------+     +--------+---------+     +--------+---------+
            | (Read Port)             |                         |                        |                        |
            |                         | INT32 Raw Score         | INT16 Scaled Score     | UINT16 Prob [0..32767] | 64-bit Beats
            +-------------------------+-------------------------+------------------------+------------------------+
                                                                                                                  |
                                                                                                                  v
                                                                                                          AXI4-Stream TX
                                                                                                          M_AXIS_TDATA [63:0]
                                                                                                          (512 Beats / Head)
```

---

## 3. BẢNG ĐỐI CHIẾU SỬA ĐỔI KIẾN TRÚC TỪ v1.0 SANG v2.0

| Hạng mục thiết kế | Phiên bản cũ (`Notes/`) | Phiên bản chuẩn hóa hiện tại (`Notes_ver02/`) | Lý do kỹ thuật / Căn cứ sửa đổi |
|---|---|---|---|
| **Mô hình mục tiêu** | ViT tiêu chuẩn ($N=196, d_k=32, H=12$) | **Apple MobileViT-XXS** ($N=256, d_k \in \{16, 20, 24\}, H=4$) | Khớp cấu trúc MobileViT trích xuất từ Hugging Face |
| **Cơ chế thu phóng (Scaler)** | Dịch phải thuần túy $\text{ASR} \gg 2$ hoặc $3$ (0 DSPs) | **Dyadic Rescaler:** $(S \times M) \ggg E$ với $M=27640, E=22$ | $1/\sqrt{20}$ và $1/\sqrt{24}$ không thể biểu diễn bằng phép dịch bit |
| **Độ phân giải Softmax** | UINT8 $[0, 255]$ (bộ chia xấp xỉ tổ hợp) | **UINT16 $[0, 32767]$** (Bộ chia lặp 32 chu kỳ khôi phục số dư) | Khớp 100% Bit-Exact với toán tử `IntSoftmax` của I-ViT |
| **Bảng tra hàm mũ Exp ROM** | ROM tĩnh 256 từ (gán cứng $e^x$) | **True Dual-Port BRAM 1024 từ $\times$ 20-bit** có nạp lại qua CSR | Hỗ trợ 3 Stages linh hoạt thông qua phần mềm cấu hình |
| **Tích lũy Systolic MAC** | Gộp chung `col0 + col1` vào 1 thanh ghi | **Tách độc lập 2 bộ tích lũy `pe_acc_col0` và `col1`** | Bảo toàn tính độc lập của 2 cột đặc trưng trong DSP packing |
| **Khối Context GEMM** | Chỉ tính 1 token đầu tiên (thiếu 99.6%) | **Tính đủ 256 tokens, 512 beats AXI-Stream, `tlast` beat 511** | Xuất trọn vẹn 4,096 bytes ma trận Context Output |
| **Vòng lặp Head (Multi-Head)** | 4 lõi song song hoặc FSM lặp Head 0 | **Single-Head Core tuần tự hóa 4 Heads** qua FSM trung tâm | Tiết kiệm >70% tài nguyên DSP và BRAM trên chip Kria KV260 |
| **Mã đọc tệp `$readmemh`** | Nằm trong Top synthesizable RTL | **Loại bỏ 100% trong synthesizable RTL**, nạp qua AXI-Stream | Đảm bảo khả năng tổng hợp phần cứng trên bo mạch vật lý |
