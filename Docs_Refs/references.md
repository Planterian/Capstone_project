import shutil
import os

ref_md_content = """# Danh Mục Tài Liệu Tham Khảo (References & Citations List)
## Dự Án Tăng Tốc Vision Transformer (ViT) Trận FPGA & Edge AI

Tài liệu này tổng hợp danh mục toàn bộ các bài báo khoa học, sáng chế, sách giáo trình và tài liệu kỹ thuật có trong thư viện nghiên cứu của dự án, được định dạng theo chuẩn trích dẫn quốc tế (APA / IEEE Style).

---

### 1. Kiến Trúc Vision Transformer (ViT) & Mô Hình Cốt Lõi (Core Models)

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30, pp. 5998–6008.  
   - **arXiv**: `1706.03762v7`  
   - **Đóng góp**: Đề xuất kiến trúc Transformer gốc và cơ chế Multi-Head Self-Attention (MSA).

2. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N.** (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. International Conference on Learning Representations (ICLR 2021).  
   - **arXiv**: `2010.11929v2`  
   - **Đóng góp**: Ứng dụng Transformer cho thị giác máy tính (Vision Transformer - ViT) bằng kỹ thuật chia Patch Embedding.

3. **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I.** (2021). *Learning Transferable Visual Models From Natural Language Supervision*. International Conference on Machine Learning (ICML 2021), pp. 8748–8763.  
   - **arXiv**: `2103.00020v1`  
   - **Đóng góp**: Mô hình học đa thức CLIP (Contrastive Language-Image Pre-training).

4. **Mehta, S., & Rastegari, M.** (2022). *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer*. International Conference on Learning Representations (ICLR 2022).  
   - **arXiv**: `2110.02178v2`  
   - **Đóng góp**: Kết hợp khối Inverted Residual của CNN với Transformer cho các thiết bị di động/nhúng.

5. **Steiner, A. P., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., & Beyer, L.** (2022). *How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers*. Transactions on Machine Learning Research (TMLR).  
   - **arXiv**: `2207.10666v1`  
   - **Đóng góp**: Tối ưu hóa quy trình huấn luyện, tăng cường dữ liệu và regularization cho ViT.

6. **Vasu, P. K. A., Gabriel, J., Zhu, J., Tuzel, O., & Ranjan, A.** (2023). *FastViT: A Fast Hybrid Vision Transformer Using Structural Reparameterization*. IEEE/CVF International Conference on Computer Vision (ICCV 2023).  
   - **arXiv**: `2303.13601v1`  
   - **Đóng góp**: Kiến trúc ViT lai có độ trễ cực thấp sử dụng tái tham số hóa cấu trúc.

7. **Vasu, P. K. A., Gabriel, J., Zhu, J., Tuzel, O., & Ranjan, A.** (2024). *MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024).  
   - **arXiv**: `2311.17049v2`  
   - **Đóng góp**: Tối ưu hóa mô hình đa thức CLIP tốc độ cao cho thiết bị di động.

8. **Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Taori, R., Fang, A., Overont, C., Puri, R., Moldovan, B., et al.** (2023). *Reproducible Scaling Laws for Contrastive Language-Image Learning*. IEEE/CVF CVPR 2023.  
   - **arXiv**: `2303.15343v4`  
   - **Đóng góp**: Đánh giá quy luật mở rộng (Scaling laws) cho OpenCLIP.

---

### 2. Bộ Tăng Tốc Phần Cứng ViT Trên FPGA (FPGA Accelerator Architectures)

9. **Wang, T., Gong, L., Wang, C., Yang, Y., Gao, Y., Zhou, X., & Chen, H.** (2022). *ViA: A Novel Vision-Transformer Accelerator Based on FPGA*. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), Vol. 41, No. 11, pp. 4088–4099.  
   - **Source**: `ViA.pdf`  
   - **Đóng góp**: Kiến trúc ViA với giải pháp Half-Layer Mapping giảm phụ thuộc đường truyền (Path Dependency).

10. **Nag, S., Datta, G., Kundu, S., Chandrachoodan, N., & Beerel, P. A.** (2023). *ViTA: A Vision Transformer Inference Accelerator for Edge Applications*. IEEE International Symposium on Circuits and Systems (ISCAS 2023).  
    - **arXiv**: `2302.02108v2` / `ViTA.pdf`  
    - **Đóng góp**: Bộ tăng tốc ViTA hỗ trợ cấu hình linh hoạt cho edge devices giới hạn BRAM.

11. **Marino, K., Zhang, P., & Prasanna, V. K.** (2023). *ME-ViT: A Single-Load Memory-Efficient FPGA Accelerator for Vision Transformers*. IEEE 30th International Conference on High Performance Computing, Data, and Analytics (HiPC 2023).  
    - **arXiv**: `2402.09709v1`  
    - **Đóng góp**: Chính sách Nạp Đơn (Single-Load Policy) triệt tiêu hoàn toàn việc ghi lại kết quả trung gian ra DRAM.

12. **Dymarkowski, H., Fu, X., Saha, R., Haris, J., & Cano, J.** (2025). *FlexViT: A Flexible FPGA-based Accelerator for Edge Vision Transformers*. University of Glasgow, UK.  
    - **arXiv**: `2606.31938v1`  
    - **Đóng góp**: Thiết kế FlexViT chạy hỗn hợp cả FC và Convolutional layers trên cùng một INT8 GEMM Engine nhờ im2col runtime.

13. **Kabir, E., Downey, A. R. J., Bakos, J. D., Andrews, D., & Huang, M.** (2024). *A Runtime-Adaptive Transformer Neural Network Accelerator on FPGAs*. ACM Transactions on Reconfigurable Technology and Systems (TRETS).  
    - **arXiv**: `2411.18148v1`  
    - **Đóng góp**: Mạch tăng tốc Transformer có khả năng thích ứng thời gian thực (Runtime-Adaptive).

14. **Lu, H.** (2024). *FPGA-Based ViT Inference Accelerator Optimization*. Highlights in Science, Engineering and Technology (ACME 2024), Vol. 111, pp. 282–287.  
    - **Source**: `ACME2024-282-287.pdf`  
    - **Đóng góp**: Tổng quan phương pháp tối ưu hóa phần mềm - phần cứng cho ViT accelerators.

15. **Yuan, X., & Tu, Q.** (2025). *Design and Implementation of an FPGA- and RK3588-Based Image Classification and Processing System*. Experimental Technology and Management, Vol. 42, No. 9, pp. 91–102.  
    - **Source**: `Image classification processing system based on FPGA and RK3588.pdf`  
    - **Đóng góp**: Kiến trúc dị thể FPGA + RK3588 NPU truyền gói tin UDP cho phân đoạn shallow/deep ViT.

---

### 3. Phương Pháp Định Lượng & Số Học Số Nguyên (Quantization & Integer-Only Arithmetic)

16. **Li, Z., & Gu, Q.** (2022). *I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), pp. 17065–17075.  
    - **arXiv**: `2207.01405v4`  
    - **Đóng góp**: Đề xuất phương pháp định lượng thuần số nguyên I-ViT, giới thiệu các thuật toán ShiftGELU, Shiftmax và I-LayerNorm (0 DSPs for non-linear).

17. **Xu, C., Kan, Y., Zhang, R., & Nakashima, Y.** (2026). *An FPGA Accelerator for Vision Transformer with Quantization and LUT-Based Operations*. IEICE Transactions on Information and Systems, Vol. E109-D, No. 1, pp. 41–48.  
    - **Source**: `E109.D_2025PAP0003.pdf`  
    - **Đóng góp**: Mạch tăng tốc ViT dựa trên bảng tra LUT cho các hàm phi tuyến và định lượng siêu thấp 2-bit/4-bit.

18. **Bhandare, A., Sripathi, V., Karkada, D., Menon, V., Choi, S., Datta, K., & Saletore, V.** (2019). *Efficient 8-bit Quantization of Transformer Neural Machine Language Translation Model*.  
    - **arXiv**: `1906.00532v2`  
    - **Đóng góp**: Phương pháp định lượng INT8 hiệu quả cho các khối Transformer.

19. **Dong, X., Bao, J., Zhang, T., Chen, D., Gu, S., Zhang, W., Yuan, L., Chen, D., Wen, F., & Yu, N.** (2022). *CLIP Itself is a Strong Fine-Tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet*.  
    - **arXiv**: `2212.08013v2`  
    - **Đóng góp**: Kỹ thuật tinh chỉnh (Fine-tuning) mô hình ViT đạt độ chính xác cao.

20. **Basalama, S., Cong, J., et al.** (2025). *Survey of Advanced FPGA Accelerators and Stream Architectures for LLMs & ViTs*. ACM/SIGDA FPGA 2025.  
    - **arXiv**: `2509.04162v1`  
    - **Đóng góp**: Báo cáo tổng quan về kiến trúc Stream-HLS, ME-ViT, FAMOUS và BRAM-aware quantization.

---

### 4. Nền Tảng Thiết Kế Vi Mạch, SoC, NoC & Bằng Sáng Chế (Hardware Foundations)

21. **Chu, P. P.** (2008). *FPGA Prototyping by Verilog Examples: Xilinx Spartan-3 Version*. John Wiley & Sons, Inc., Hoboken, New Jersey.  
    - **Source**: `Pong P. Chu - FPGA Prototyping Using Verilog Examples (2008) - libgen.lc.pdf`  
    - **Đóng góp**: Sách giáo trình chuẩn về thiết kế mạch số RTL, FSMD, UART, BRAM và giao tiếp ngoại vi.

22. **Dieffenderfer, J. N., & Kalla, R. N.** (1993). *Ping-Pong Data Buffer for Transferring Data from One Data Bus to Another Data Bus*. United States Patent No. US5224213A.  
    - **Source**: `US5224213.pdf`  
    - **Đóng góp**: Bằng sáng chế cơ chế bộ đệm kép Ping-Pong (Double-Buffering) chuyển đổi bus dữ liệu độc lập.

23. **ElOuchdi, A., Tahiri, N., & Abououden, A.** (2011). *Data Transfer Interface Used in NoC*. International Multi-Conference on Systems, Signals & Devices (SSD).  
    - **Source**: `Data Transfer Interface Used in NoC.pdf`  
    - **Đóng góp**: Thiết kế giao tiếp chuyển dữ liệu (DTI) đồng bộ tần số trong Network-on-Chip.

24. **Swaminathan, K., Lakshminarayanan, G., Lang, F., Fahmi, M., & Ko, S.-B.** (2013). *Design of a Low Power Network Interface for Network on Chip*. IEEE Canadian Conference on Electrical and Computer Engineering (CCECE).  
    - **Source**: `Design_of_a_low_power_network_interface.pdf`  
    - **Đóng góp**: Bộ giao tiếp mạng NoC công suất thấp với điều khiển ngắt FIFO linh hoạt.

---

### 5. Hồ Sơ Đồ Án & Tài Liệu Hướng Dẫn Thực Hành (Capstone Project Manuals)

25. **Nhóm Nghiên Cứu Capstone.** (2026). *Hardware-Accelerated Transformer Attention for Intelligent Edge Vision (MobileViT on AMD Kria KV260)*. Capstone Project Investigation Report.  
    - **Source**: `CapstoneProjectInvestion.pdf`  
    - **Nội dung**: Báo cáo khảo sát kiến trúc SoC AMD Kria KV260, phân chia HW/SW partitioning, AXI DMA và thiết kế luồng Datapath.

26. **Nguyễn Lưu Khánh Trình & Trần Minh Trí.** (2026). *FPGA-Based Edge AI and RISC-V Capstone Project Manual*.  
    - **Source**: `Capstone_Projects.docx`  
    - **Nội dung**: Đề xướng đồ án tốt nghiệp thiết kế bộ tăng tốc AI trên FPGA kết hợp vi xử lý RISC-V / ARM.

27. **Nhóm Tác Giả.** (2026). *Tất cả ghi chú 9/14/2026 - Tổng hợp Kiến thức Nghiên cứu*.  
    - **Source**: `Tất cả ghi chú 9/14/2026` / `raw_notes_for_prism.md`  
    - **Nội dung**: Toàn bộ ghi chú kỹ thuật, mã RTL SystemVerilog, script Python PyTorch/Cocotb và kịch bản Vivado.
""", TargetFile: "/workspace/scratch/references.md"}
)

execute_command("cp /workspace/scratch/references.md /workspace/out/references.md")
