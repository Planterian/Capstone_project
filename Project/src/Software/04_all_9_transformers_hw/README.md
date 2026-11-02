# Phase 4: Full Network Acceleration of All 9 Transformers on AMD Kria KV260

This folder contains the complete runtime harness for **Full-Network Attention Acceleration** where **ALL 9 Multi-Head Attention blocks** across the 3 MobileViT stages are offloaded to the physical 150 MHz FPGA PL Core.

## Scope of Acceleration
- **Stage 2 (2 Layers):** $N=256, d_k=16$ ($M=27640, E=22$) $\rightarrow$ Ingress: 12,288 B, Egress: 4,096 B
- **Stage 3 (4 Layers):** $N=64, d_k=20$ ($M=22559, E=22$) $\rightarrow$ Ingress: 3,840 B, Egress: 1,280 B
- **Stage 4 (3 Layers):** $N=16, d_k=24$ ($M=19947, E=22$) $\rightarrow$ Ingress: 1,152 B, Egress: 384 B
- **Total Invocations:** **9 hardware offloads per image inference pass!**

## Quickstart on Standalone KV260

### Run with default sample image:
```bash
./run_benchmark.sh
```

### Run on a custom image:
```bash
sudo -E python3 run_all_transformers_hw_kv260.py --image /home/ubuntu/test_images/my_image.jpg --iterations 50
```

## Architectural Characteristics
- **FPGA Bitstream:** `overlay/attention_core_top_2A.bit` @ 150.000 MHz
- **Memory Optimization:** Unified pre-allocated CMA DDR4 buffers (max 12,288 B ingress, 4,096 B egress) reused with zero-copy buffer slicing across all 9 layers.
- **Dynamic CSR:** Reconfigures $N$, $d_k$, $M$, $E$, and payload length before each invocation.
- **Output Artifacts:** `phase4_metrics.json`

