# Triển Khai ViT Trên FPGA Kria KV260 Với PYNQ v2.0

> **Dự án:** Triển khai End-to-End HW-SW Codesign MobileViT-XXS trên AMD Kria KV260  
> **Phiên bản:** v2.0 (Cập nhật chuẩn hóa theo RTL `rtl_ver02`)  
> **Tài liệu gốc tham chiếu:** [`Notes/trien_khai_vit_tren_fpga.md`](file:///c:/Users/tuan2/Desktop/Capstone_Project/Notes/trien_khai_vit_tren_fpga.md)

---

## 1. QUY TRÌNH TRIỂN KHAI VỚI PYNQ VÀ JUPYTER LAB

Hệ điều hành PYNQ trên AMD Kria KV260 cung cấp giao diện lập trình Python trực quan để giao tiếp với phần cứng FPGA PL qua các lớp trừu tượng `Overlay` và `allocate`:

```python
from pynq import Overlay, allocate
import numpy as np

# 1. Nạp Bitstream và tệp Hardware Handoff
ol = Overlay("attention_core_kv260.bit")
dma = ol.axi_dma
attn_core = ol.attention_core_top

# 2. Cấu hình các thanh ghi AXI-Lite CSR cho Stage 2 (dk=16)
attn_core.write(0x10, 256)     # SEQ_LEN = 256
attn_core.write(0x14, 16)      # HEAD_DIM = 16
attn_core.write(0x18, 4)       # NUM_HEADS = 4
attn_core.write(0x20, 27640)   # SCALE_M = 27640
attn_core.write(0x24, 22)      # SCALE_E = 22

# 3. Cấp phát vùng đệm bộ nhớ liên tục Contiguous DMA Buffers
in_buffer = allocate(shape=(4, 3, 256, 16), dtype=np.int8)   # 4 heads x (Q, K, V)
out_buffer = allocate(shape=(4, 256, 16), dtype=np.int8)     # 4 heads x Context Out

# 4. Truyền nhận dữ liệu qua AXI DMA
dma.sendchannel.transfer(in_buffer)
dma.recvchannel.transfer(out_buffer)

# 5. Kích hoạt tính toán trên FPGA
attn_core.write(0x00, 1)       # START pulse

# 6. Đợi DMA hoàn thành
dma.sendchannel.wait()
dma.recvchannel.wait()
```

---

## 2. QUẢN LÝ TÀI NGUYÊN VÀ THỜI GIAN ĐÓNG GÓI VIVADO

1. **Ràng buộc thời gian (Timing Closure):**
   - Tần số xung nhịp danh định: $200\text{ MHz}$ ($T_{\text{clk}} = 5.0\text{ ns}$).
   - Mục tiêu: Worst Negative Slack $\text{WNS} \ge 0.000\text{ ns}$, Total Negative Slack $\text{TNS} = 0.000\text{ ns}$.
2. **Dự báo tài nguyên trên Kria KV260 (`xck26-sfvc784-2LV-c`):**
   - **LUTs:** $\approx 12,000 / 117,120$ ($10.2\%$).
   - **BRAM (RAMB36K):** $\approx 12 / 144$ ($8.3\%$).
   - **DSP48E2:** $\approx 36 / 1,248$ ($2.9\%$).
   - Hệ thống vận hành cực kỳ mát mẻ, an toàn nhiệt và tiêu thụ công suất thấp ($< 5\text{W}$).
