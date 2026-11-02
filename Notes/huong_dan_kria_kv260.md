# Hướng Dẫn Vận Hành Bo Mạch AMD Kria KV260 v2.0 (MobileViT Accelerator)

> **Dự án:** Triển khai End-to-End HW-SW Codesign MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo mô hình MobileViT $256 \times 256$ và Bitstream `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/huong_dan_kria_kv260.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/huong_dan_kria_kv260.md)

---

## 1. DANH MỤC THIẾT BỊ VÀ THÔNG SỐ CHUẨN

1. **Bo mạch AMD Kria KV260 Vision AI Starter Kit**:
   - Chip: Zynq UltraScale+ MPSoC (`xck26-sfvc784-2LV-c`).
   - Lõi ARM PS: Quad-core ARM Cortex-A53 @ 1.33 GHz, chạy Ubuntu 22.04 LTS / PYNQ Linux v3.0+.
   - FPGA PL: 256K System Logic Cells, 1,248 DSP Slices, 144 BRAM Blocks (36Kb).
2. **Thẻ nhớ MicroSD**: Dung lượng tối thiểu 32GB Class 10 (khuyến nghị SanDisk Extreme U3/V30).
3. **Nguồn cấp DC**: Nguồn adapter 12V – 3A (jack tròn 2.1mm $\times$ 5.5mm, cực dương trong).
4. **Cáp mạng Ethernet (RJ45)**: Kết nối KV260 vào switch/router nội bộ với Laptop phát triển.
5. **Cáp Micro-USB Debug**: Đọc console UART qua cổng COM (`Baudrate: 115200`).
6. **USB Webcam**: Chuẩn UVC (720p/1080p), chụp ảnh đầu vào chuẩn $256 \times 256 \times 3$ cho MobileViT-XXS.

---

## 2. SƠ ĐỒ KẾT NỐI VẬT LÝ

```text
===================================================================================================
                               SƠ ĐỒ KẾT NỐI HỆ THỐNG (DEVELOPMENT MODE v2.0)
===================================================================================================

       +------------------------------------+
       |   LAPTOP / PC PHÁT TRIỂN           |
       |   - Vivado 2026.1 (Build Bitstream)|
       |   - VS Code Remote SSH / SCP       |
       +-----------------+------------------+
                         |
      +------------------+------------------+
      | (Cáp Mạng LAN - 1Gbps)              | (Cáp Micro-USB Debug UART)
      v                                     v
  +---+-------------------------------------+--------------------------------------------------+
  |   Cổng RJ45                             Cổng Micro-USB (COM Port)                          |
  |                                                                                            |
  |                        BO MẠCH AMD KRIA KV260 STARTER KIT                                  |
  |                                                                                            |
  |   [ Chip Zynq UltraScale+ MPSoC ]                                                          |
  |     ├── ARM PS (Cortex-A53): Chạy Ubuntu, PYNQ v3.0, PyTorch CPU, AXI DMA Driver           |
  |     └── FPGA PL: Attention Core Top (rtl_ver02 @ 200 MHz, 4096B Bit-Exact)                |
  |                                                                                            |
  |   Cổng USB 3.0                          Cổng HDMI / DP             Cổng Nguồn DC 12V        |
  +---+-------------------------------------+--------------------------+-----------------------+
      |                                     |                          |
      v                                     v                          v
  [ USB Webcam ]                   [ Màn hình HDMI ]           [ Nguồn Điện 12V-3A ]
  (Ảnh 256x256x3)                  (Hiển thị Top-K Classes)
```

---

## 3. QUY TRÌNH DEPLOY VÀ CHẠY JUPYTER LAB TRÊN KV260

### 3.1 Chuyển giao sản phẩm phần cứng (Hardware Handoff):
Từ máy tính phát triển, sao chép 2 tệp đóng gói vào KV260 qua lệnh SCP:
```bash
scp Project/src/Hardware/rtl_ver02/attention_core_kv260.bit xilinx@<IP_KV260>:/home/xilinx/jupyter_notebooks/
scp Project/src/Hardware/rtl_ver02/attention_core_kv260.hwh xilinx@<IP_KV260>:/home/xilinx/jupyter_notebooks/
scp model_notebook/ivit_surgery/golden_vectors/dyadic_scale_config.json xilinx@<IP_KV260>:/home/xilinx/jupyter_notebooks/
```

### 3.2 Khởi động Jupyter Lab và nạp mạch:
1. Mở trình duyệt web truy cập `http://<IP_KV260>:9090` (mật khẩu: `xilinx`).
2. Mở file notebook kiểm thử và thực thi:
```python
from pynq import Overlay, allocate
import json
import numpy as np

# Nạp Overlay phần cứng vào FPGA PL
ol = Overlay("attention_core_kv260.bit")
print("[INFO] Bitstream Loaded Successfully into Kria KV260 PL!")

# Đọc cấu hình Dyadic Scales trích xuất từ QAT
with open("dyadic_scale_config.json") as f:
    cfg = json.load(f)

stage2_m = cfg["stages"]["STAGE2"]["M_scale"] # 27640
stage2_e = cfg["stages"]["STAGE2"]["E_scale"] # 22

# Cấu hình thanh ghi CSR
ol.attention_core_top.write(0x10, 256)      # SEQ_LEN = 256
ol.attention_core_top.write(0x14, 16)       # HEAD_DIM = 16
ol.attention_core_top.write(0x18, 4)        # NUM_HEADS = 4
ol.attention_core_top.write(0x20, stage2_m) # SCALE_M
ol.attention_core_top.write(0x24, stage2_e) # SCALE_E
```
