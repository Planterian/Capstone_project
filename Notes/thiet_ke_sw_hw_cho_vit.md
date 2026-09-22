# Thiết kế SW-HW cho ViT

Dưới đây là bộ sơ đồ kiến trúc vi mạch và hệ thống **SW-HW Co-Design** đầy đủ cho dự án tăng tốc **Vision Transformer (ViT) trên SoC (AMD Kria KV260)**. Bộ sơ đồ này bao gồm kiến trúc SoC, phân tầng SW-HW, luồng dữ liệu (Dataflow) end-to-end, và quy trình lập trình/điều khiển thực thi (Programming & Execution Procedure).

---

##### 1. SoC & SYSTEM DESIGN ARCHITECTURE

Sơ đồ thể hiện toàn bộ hệ thống trên chip (SoC), ranh giới giữa **Processing System (ARM PS)** và **Programmable Logic (FPGA PL)**, hệ thống bus AXI4, và phân cấp bộ nhớ.

```text
========================================================================================================================
                                     SOC & SYSTEM DESIGN ARCHITECTURE (AMD Kria KV260)
========================================================================================================================

  +------------------------------------------------------------------------------------------------------------------+
  |                                      PROCESSING SYSTEM (ARM PS - Application Subsystem)                          |
  |                                                                                                                  |
  |  +-----------------------------------+     +----------------------------------+     +-------------------------+  |
  |  |  ARM Cortex-A53 Multi-Core CPU    |     |  System Memory Controller        |     |  Peripherals & I/O      |  |
  |  |  (Ubuntu OS / PYNQ Runtime / App) |     |  (DDR4 Controller / Interconnect)|     |  (USB 3.0 / DisplayPort)|  |
  |  +-----------------+-----------------+     +----------------+-----------------+     +------------+------------+  |
  |                    |                                        |                                    |               |
  |                    v                                        v                                    v               |
  |            AXI Interconnect                         DDR4 DRAM (4GB)                     USB Webcam / Display     |
  |                    |                        +-------------------------------+                                    |
  |                    |                        | - Operating System & Drivers  |                                    |
  |                    |                        | - CMA Buffer (Quantized INT8) |                                    |
  |                    |                        +---------------+---------------+                                    |
  +--------------------+----------------------------------------|----------------------------------------------------+
                       |                                        |
      AXI4-Lite Bus    | (Control/Status Regs)                  | AXI4 High-Performance (HP) Bus
      [32-bit Memory]  |                                        | (DMA Data Transfers)
                       v                                        v
  ====================================================================================================================
                       |                                        |
  +--------------------+----------------------------------------|----------------------------------------------------+
  |                    v                                        v                                                    |
  |   +----------------------------------+            +----------------------------------+                           |
  |   |  AXI4-Lite Slave Register File   |            |   AXI DMA Controller (HW Core)   |                           |
  |   |  (Control FSM, N, d_k, Config)   |            |   - MM2S Engine (Read DDR4 -> PL)|                           |
  |   +----------------+-----------------+            |   - S2MM Engine (Write PL -> DDR)|                           |
  |                    |                              +----------------+-----------------+                           |
  |                    | (Control Signals)                             | (AXI4-Stream 64-bit)                         |
  |                    v                                               v                                             |
  |   +----------------------------------------------------------------------------------+                           |
  |   |                        FPGA PL ACCELERATOR CORE (Attention Engine)               |                           |
  |   |                                                                                  |                           |
  |   |   +-------------------+    +--------------------+    +-----------------------+   |                           |
  |   |   | Ping-Pong BRAM    |--->| Systolic GEMM Array|--->| Non-linear Pipeline   |   |                           |
  |   |   | (Q, K, V Buffers) |    | (DSP Packing 2xINT8|    | (ShiftGELU / Softmax) |   |                           |
  |   |   +-------------------+    +--------------------+    +-----------------------+   |                           |
  |   +----------------------------------------------------------------------------------+                           |
  |                                                                                                                  |
  |                                       PROGRAMMABLE LOGIC (FPGA PL)                                               |
  +------------------------------------------------------------------------------------------------------------------+
```

---

##### 2. SW-HW CO-DESIGN LAYERED ARCHITECTURE

Phân tầng kiến trúc phần mềm và phần cứng, thể hiện cách ứng dụng người dùng tương tác qua Linux Kernel, PYNQ Framework xuống driver DMA và vi mạch PL.

```text
========================================================================================================================
                                SW-HW CO-DESIGN LAYERED ARCHITECTURE
========================================================================================================================

  +------------------------------------------------------------------------------------------------------------------+
  | APPLICATION LAYER (User Space - Python / C++)                                                                    |
  |   - Host Application (`realtime_vit_inference.py`)                                                               |
  |   - Image Preprocessing (OpenCV Resize 224x224, Patch Partitioning, INT8 Quantization)                          |
  |   - Postprocessing & Inference Result Visualization (Bounding Box, Classification Label, FPS Render)            |
  +---------------------------------------------------+--------------------------------------------------------------+
                                                      |
                                                      v
  +------------------------------------------------------------------------------------------------------------------+
  | SW FRAMEWORK & API LAYER (PYNQ Runtime Environment)                                                              |
  |   - `pynq.Overlay`: Loads Bitstream (`.bit`) & Device Tree Overlay (`.hwh`)                                       |
  |   - `pynq.allocate`: Allocates Physically Contiguous Memory (CMA - Contiguous Memory Allocator)                   |
  |   - `pynq.lib.axidma`: Controls AXI DMA Channel Transfer (`sendchannel`, `recvchannel`)                          |
  |   - Register I/O Access API: MMIO Write/Read (`attn_ip.write(offset, val)`)                                      |
  +---------------------------------------------------+--------------------------------------------------------------+
                                                      |
                                                      v
  +------------------------------------------------------------------------------------------------------------------+
  | OS & DRIVER LAYER (Linux Kernel Space)                                                                           |
  |   - UIO (Userspace I/O) Driver: Direct Register Map Access for AXI-Lite Control Registers                         |
  |   - Xilinx ZynqMP DMA Driver (`xilinx-axidma`): Handles DMA Ring Buffers & Hardware Interrupts                     |
  |   - CMA Driver (`/dev/cma`): Guarantees Zero-Copy Buffer Allocation for Direct DMA Access                         |
  +---------------------------------------------------+--------------------------------------------------------------+
                                                      |
                                                      v  (Hardware Boundary: AXI Interconnect / Memory Bus)
  ====================================================================================================================
                                                      |
  +---------------------------------------------------+--------------------------------------------------------------+
  | HARDWARE LAYER (FPGA PL - Physical Circuits)                                                                     |
  |   - AXI4-Lite Control Slave Registers (Offsets: 0x00=START/DONE, 0x10=N, 0x18=d_k)                              |
  |   - AXI DMA IP Core (MM2S & S2MM Stream Channels)                                                                |
  |   - Ping-Pong On-Chip BRAM / LUTRAM Buffers                                                                      |
  |   - Pipelined GEMM & Attention Core (DSP48E2 Packing, ShiftGELU, I-LayerNorm, Softmax LUT)                       |
  +------------------------------------------------------------------------------------------------------------------+
```

---

##### 3. END-TO-END SW-HW DATAFLOW ARCHITECTURE

Luồng truyền dữ liệu chi tiết qua từng công đoạn xử lý từ Camera vào ARM PS, qua bus AXI DMA sang FPGA PL, và quay trở lại hiển thị.

```text
========================================================================================================================
                                     END-TO-END SW-HW DATAFLOW ARCHITECTURE
========================================================================================================================

  [ USB Webcam / Video Stream ]
               |
               v (Raw BGR Image: 640x480x3 @ 30fps)
  +-------------------------------------------------------------------+
  | SW STAGE 1: ARM PS Preprocessing                                  |
  |   1. OpenCV Resize & Center Crop -> 224x224x3                      |
  |   2. Patch Partitioning (P=16) -> N = 196 Patches                 |
  |   3. Linear Projection & Positional Encoding -> FP32 Feature Map  |
  |   4. Uniform Symmetric INT8 Quantization: I_X = Clamp(X / Scale)  |
  +------------------------------------+------------------------------+
                                       |
                                       v (INT8 Input Tensors: Q, K, V)
  +-------------------------------------------------------------------+
  | SW STAGE 2: Zero-Copy CMA Buffer Mapping                          |
  |   - Write Q, K, V Matrices into Allocated CMA Memory Space        |
  |   - Memory Addresses: Physical Addr (PA) mapped for AXI DMA Access|
  +------------------------------------+------------------------------+
                                       |
                                       v (AXI4 MM2S Memory Read Burst)
  ===========================================================================================================
  | BUS BOUNDARY: AXI4 High-Performance (HP) Bus (64-bit Stream)                                           |
  ===========================================================================================================
                                       |
                                       v
  +-------------------------------------------------------------------+
  | HW STAGE 1: FPGA PL On-Chip Streaming & Ping-Pong Buffering       |
  |   - Stream RX Engine receives INT8 Data via AXI4-Stream Interface |
  |   - FSM stores incoming blocks into Ping-Pong BRAM (Buffer A/B)   |
  +------------------------------------+------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------+
  | HW STAGE 2: Pipelined Execution Engine                            |
  |   1. QK^T GEMM: Systolic Array (DSP Packing 2xINT8) -> INT32 Acc  |
  |   2. Scaler Unit: Arithmetic Right Shift (ASR) 1/sqrt(d_k)        |
  |   3. Softmax Engine: Max-Subtraction -> Exp LUT -> Inverse Div    |
  |   4. Score x V GEMM: Aggregation -> Saturating INT8 Output        |
  +------------------------------------+------------------------------+
                                       |
                                       v (Attended INT8 Token Output)
  +-------------------------------------------------------------------+
  | HW STAGE 3: Stream TX & DMA S2MM Transfer                         |
  |   - Stream TX Engine packs result into AXI4-Stream Burst          |
  |   - AXI DMA S2MM writes INT8 Output Tensor back to DDR4 CMA Buffer|
  +------------------------------------+------------------------------+
                                       |
                                       v (AXI4 S2MM Memory Write Burst)
  ===========================================================================================================
  | BUS BOUNDARY: AXI4 High-Performance (HP) Bus                                                           |
  ===========================================================================================================
                                       |
                                       v
  +-------------------------------------------------------------------+
  | SW STAGE 3: ARM PS Postprocessing & Visualization                 |
  |   1. De-quantization & MLP / Classification Head Execution        |
  |   2. ArgMax Prediction Label Extraction                           |
  |   3. OpenCV Render Frame with Prediction Label & Real-time FPS UI |
  +------------------------------------+------------------------------+
                                       |
                                       v
  [ HDMI / DisplayPort Real-time Screen Display ]
```

---

##### 4. SW-HW PROGRAMMING PROCEDURE & EXECUTION SEQUENCE

Sơ đồ trình tự thời gian (Sequence Diagram) thể hiện chính xác các bước giao tiếp, gửi lệnh bắt đầu, kích hoạt DMA, FSM phần cứng xử lý, và bắt tay hoàn tất (Handshake/Interrupt).

```text
========================================================================================================================
                            SW-HW PROGRAMMING & EXECUTION SEQUENCE PROCEDURE
========================================================================================================================

  SW (ARM PS Host Script)                  AXI DMA Controller                  FPGA PL Attention Core FSM
           |                                       |                                       |
           | 1. pynq.Overlay("vit_core.bit")        |                                       |
           |-------------------------------------->| (Load Bitstream & Program PL Logic)   |
           |                                       |                                       |
           | 2. Allocate CMA Memory Buffers        |                                       |
           |    in_buf  = allocate((N,d_k), int8)  |                                       |
           |    out_buf = allocate((N,d_k), int8)  |                                       |
           |-------------------------------------->| (Reserve Physical Continuous Memory)  |
           |                                       |                                       |
           | 3. Preprocess & Write INT8 Input Data |                                       |
           |-------------------------------------->| (Fill in_buf with INT8 Image Tokens)  |
           |                                       |                                       |
           | 4. Configure AXI-Lite Regs            |                                       |
           |    attn.write(0x10, N=196)            |                                       |
           |    attn.write(0x18, d_k=32)           |-------------------------------------->| (Set Register Context)
           |                                       |                                       |
           | 5. Initiate DMA Receive Channel       |                                       |
           |    dma.recvchannel.transfer(out_buf)  |                                       |
           |-------------------------------------->| (Prepare S2MM RX Ring Buffers)        |
           |                                       |                                       |
           | 6. Initiate DMA Send Channel          |                                       |
           |    dma.sendchannel.transfer(in_buf)   |                                       |
           |-------------------------------------->| (Start MM2S Memory Burst Read)        |
           |                                       |                                       |
           | 7. Assert Hardware Start Pulse        |                                       |
           |    attn.write(0x00, 0x01) [START=1]   |-------------------------------------->| 8. FSM Transition:
           |                                       |                                       |    IDLE -> FETCH_BRAM
           |                                       |                                       |    -> COMPUTE_QKT
           |                                       |                                       |    -> SOFTMAX_PIPELINE
           |                                       |                                       |    -> COMPUTE_SV
           |                                       |                                       |    -> WRITEBACK
           |                                       |                                       |
           |                                       |<======================================| 9. Stream TX Done
           |                                       |    (AXI4-Stream TLAST Asserted)       |    FSM State -> DONE
           |                                       |                                       |    Assert Reg 0x00 Bit 1
           |                                       |                                       |
           | 10. Wait DMA Channels Complete        |                                       |
           |     dma.sendchannel.wait()            |                                       |
           |     dma.recvchannel.wait()            |                                       |
           |<--------------------------------------| (DMA Transfer Finished / Interrupt)   |
           |                                       |                                       |
           | 11. Read Output & Postprocess         |                                       |
           |     res = out_buf.copy()              |                                       |
           |     render_ui(res)                    |                                       |
           |                                       |                                       |
```

---

##### 💡 Tóm tắt các đặc tính SW-HW Co-Design chính:
1. **Zero-Copy Memory Access**: Sử dụng **Linux CMA (Contiguous Memory Allocator)** qua API `pynq.allocate` giúp AXI DMA truy xuất trực tiếp vùng nhớ DDR4 mà không cần copy qua Kernel Space, giảm 100% chi phí overhead bộ nhớ.
2. **Double-Buffering (Ping-Pong Hiding)**: Kết hợp Ping-Pong BRAM trên PL và DMA Ring Buffers giúp **gối đầu thời gian truyền DMA và thời gian tính toán của mảng MAC**, đạt hiệu suất thực thi sát ngưỡng lý thuyết.
3. **Hardware Control via AXI-Lite**: ARM PS điều khiển FSM vi mạch linh hoạt thông qua tập thanh ghi cấu hình (Start, Done, Sequence Length $N$, Head Dimension $d_k$), cho phép chuyển đổi quy mô mô hình mà không cần nạp lại Bitstream.
