# Phase 3: End-to-End HW/SW Co-Design on AMD Kria KV260

This folder contains the complete runtime harness for the **Partitioned Hardware/Software Co-Design** deployment of MobileViT-XXS on the AMD Kria KV260.

## Hardware/Software Partitioning
- **ARM Cortex-A53 CPU (PS):** Executes the MobileNetV2 CNN stem, inverted residual blocks, layer norms, intermediate MLPs, and linear classifier head.
- **FPGA PL Accelerator (150 MHz):** Accelerates the Stage 2 Multi-Head Attention kernel ($N=256, d_k=16, H=1$) over 128-bit AXI DMA (`axi_dma_0`) and AXI4-Lite CSR control.
- **Stages 3 & 4:** Seamlessly handled via software fallback on ARM CPU.

## Quickstart on Standalone KV260

### Run with default sample image:
```bash
./run_benchmark.sh
```

### Run on a custom image:
```bash
sudo -E python3 run_ivit_hw_kv260.py --image /home/ubuntu/test_images/my_image.jpg --iterations 50
```

## Architectural Characteristics
- **FPGA Bitstream:** `overlay/attention_core_top_2A.bit` (Variant 2A: DSP packed INT8 x INT8 + INT16 x INT8)
- **Clock Frequency:** 150.000 MHz (`pl_clk0`)
- **Measured PL Kernel Latency:** ~2.08 ms (481.3 FPS!)
- **DMA Packet Size:** 12,288 Bytes (Ingress) / 4,096 Bytes (Egress)
- **Output Artifacts:** `phase3_metrics.json`

