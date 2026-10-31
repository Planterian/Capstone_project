# Kế Hoạch Tối Ưu Hóa `.gitignore` & Dọn Dẹp Kho Lưu Trữ Git (Capstone_Project)

---

## 1. Tóm Tắt Khẩn Cấp Dưới Góc Nhìn Kỹ Sư Phần Mềm (Executive Summary)

Sau khi tiến hành kiểm toán chuyên sâu toàn bộ 659 tệp đang được theo dõi (`git ls-files`) trong kho lưu trữ `c:\Users\tuan2\Desktop\Capstone_Project`, chúng tôi phát hiện tình trạng **phình to nghiêm trọng (Repository Bloat)** và **nguy cơ từ chối đẩy mã (Push Rejection)** từ GitHub:

### 📊 Thống kê kiểm toán dung lượng kho lưu trữ:
* **Tổng dung lượng hiện tại trong Git**: **2,711.54 MB (~2.71 GB)** với 659 files.
* **Tệp rác / Tệp tạm / Bản dựng công cụ (Unnecessary Artifacts)**: **148 files** chiếm **2,398.40 MB (~2.34 GB, tương đương 88.5% tổng dung lượng repo!)**.
* **Mã nguồn và tài liệu cốt lõi thực tế (Core Assets)**: Chỉ chiếm **~313 MB** (bao gồm cả các mô hình cơ sở và tài liệu PDF nghiên cứu).

### 🚨 Hai Vấn Đề Khẩn Cấp Đang Xảy Ra:
1. **GitHub Chặn Đẩy Mã (`git push` Bị Treo / Bị Từ Chối)**:
   - GitHub áp đặt giới hạn cứng: **Cảnh báo với tệp > 50 MB** và **chặn tuyệt đối (Reject Push) với tệp > 100 MB** (nếu không dùng Git LFS).
   - Trong lần `git commit -m "feat: doing push"` vừa qua, có tới **9 tệp waveform `.vcd` vượt ngưỡng 100 MB** (điển hình: `tb_multi_stage_seq_2A.vcd`: **365.69 MB**, `tb_stage4_isolated_3B.vcd`: **282.57 MB**, `tb_stage4_isolated_2A.vcd`: **251.44 MB**...).
   - Lệnh `git push` bị treo nén đối tượng (Compressing objects 93% suốt hơn 11 phút) và chắc chắn sẽ bị máy chủ GitHub từ chối khi truyền dữ liệu.
2. **Nguy Cơ Lộ Thông Tin Bảo Mật Cực Kỳ Nghiêm Trọng**:
   - Tệp chưa được theo dõi tại `Project/src/Software/ssh_kv260_Long_GiaThoai.txt` chứa **mật khẩu văn bản thô (Plain-text credentials)**: Tài khoản Tailscale, tài khoản & IP Host Lab (`100.124.168.26`), tài khoản & IP bo mạch Kria KV260 (`192.168.1.179 / pass: capstone2026`).
   - Nếu thực hiện `git add .` mà không có `.gitignore` chặn lại, toàn bộ hạ tầng phần cứng của nhóm sẽ bị lộ lên GitHub.

---

## 2. Bảng Phân Loại Tài Nguyên: Cần Chia Sẻ vs. Chỉ Dùng Cục Bộ

Dưới vai trò kiểm soát phiên bản (Version Control Engineer), toàn bộ các tệp và thư mục trong dự án được phân chia thành 4 nhóm rõ rệt:

```text
================================================================================================================
                                BẢNG MA TRẬN PHÂN LOẠI TÀI NGUYÊN (CAPSTONE_PROJECT)
================================================================================================================

+-------------------------------------------------------------+------------------------------------------------+
|       NHÓM 1: BẮT BUỘC CHIA SẺ TRONG GIT (TIER 1)           |    NHÓM 2: TUYỆT ĐỐI KHÔNG LƯU GIT (TIER 2)    |
|       (Core Intellectual Property & Golden Assets)          |    (Build Artifacts, Tool Cache & Waveforms)   |
+-------------------------------------------------------------+------------------------------------------------+
| • Mã nguồn phần cứng RTL (*.sv, *.v, *.vhd): 95 files       | • Tệp sóng mô phỏng (*.vcd, *.wdb): ~2.37 GB   |
| • Ràng buộc Timing & Synthesis (*.xdc): 2 files             | • Bộ nhớ đệm Vivado (.Xil/, *.cache/, *.runs/):|
| • Kịch bản tự động hóa (*.bat, *.sh, *.tcl, *.py): 30 files |   Chứa vết máy DESKTOP-L58T74E, gây conflict   |
| • Vector kiểm thử Golden Stimulus (*.hex):                  | • Nhật ký công cụ (vivado*.log, vivado*.jou,   |
|   169 files (cần thiết để testbench bit-exact chạy được)    |   *.backup.log, *.backup.jou): 13 files        |
| • Tài liệu thiết kế & Báo cáo (*.md, *.tex, *.sty):         | • Tệp nhị phân mô phỏng Icarus (*.vvp, a.out): |
|   Notes/, Notes_ver02/, model_notebook/markdowns/           |   43 files (sinh tự động khi compile RTL)      |
| • Báo cáo tổng hợp số liệu nhẹ (*.rpt):                     | • Tệp rác biên dịch LaTeX (*.aux, *.toc, *.out)|
|   reports_*/post_synth_timing.rpt, utilization.rpt          | • Tệp rác Python & Notebook:                   |
| • Mô hình trọng số cơ sở (Golden INT8 Baseline):            |   __pycache__/, *.pyc, .ipynb_checkpoints/     |
|   vit_qat_int8.pth, MobileViT-XXS.pth (nhẹ, dùng cho demo)  |                                                |
+-------------------------------------------------------------+------------------------------------------------+
|        NHÓM 3: BẢO MẬT CỤC BỘ (TIER 3 - SENSITIVE)          |    NHÓM 4: KHUYẾN NGHỊ LƯU DRIVE NGOÀI (TIER 4)|
+-------------------------------------------------------------+------------------------------------------------+
| • Thông tin truy cập hạ tầng: ssh_kv260_*.txt               | • Các tệp PDF tài liệu tham khảo nặng:         |
| • File khóa, token cá nhân: *.key, *.pem, .env, id_rsa      |   Docs_Refs/*.pdf (hơn 160 MB bài báo IEEE)    |
| • Cấu hình IDE cá nhân: .vscode/, .idea/                    | • Checkpoint trung gian khi train model:        |
|                                                             |   *_latest.pth, optimizer_state.pt             |
+-------------------------------------------------------------+------------------------------------------------+
```

### Chi tiết phân tích từng nhóm:

| Hạng Mục | Hiện Trạng Trong Git | Đánh Giá Kỹ Thuật | Hành Động Đề Xuất |
| :--- | :---: | :--- | :--- |
| **Waveform Dumps (`*.vcd`, `*.wdb`)** | **25 files (2,373.8 MB)** | **RẤT NGUY HIỂM**. Đây là dữ liệu sóng chi tiết từng chu kỳ do Icarus/Vivado dump ra. Mỗi thành viên có thể tự sinh lại ngay lập tức bằng script `run_*.bat`. Không ai xem diff của file 300MB trên GitHub. | **Xóa khỏi Git tracking ngay & Đưa vào `.gitignore`** |
| **Vivado Caches (`.Xil/`)** | **66 files (12.6 MB)** | **RÁC CÔNG CỤ**. Chứa các file `.rtd`, `.tim`, `.lpr` mang tên hostname máy tính cục bộ (`Vivado-18932-DESKTOP-L58T74E`). Sẽ gây xung đột nhánh (merge conflict) liên tục giữa các thành viên. | **Xóa khỏi Git tracking & Chặn thư mục `.Xil/`** |
| **Icarus Verilog Binaries (`*.vvp`, `a.out`)** | **43 files (10.8 MB)** | **EXECUTABLE BUILD**. Là file nhị phân máy tính do `iverilog` biên dịch từ `.sv`. Không có giá trị kiểm soát phiên bản. | **Xóa khỏi Git tracking & Chặn `*.vvp`, `a.out`** |
| **Vivado Log & Backup (`*.log`, `*.jou`)** | **13 files (1.2 MB)** | **NHẬT KÝ TẠM**. Sinh ra mỗi lần Vivado khởi chạy để debug tiến trình, không có tính tái sử dụng. | **Xóa khỏi Git tracking & Chặn `vivado*.log`, `vivado*.jou`** |
| **LaTeX Temp Files (`*.aux`, `*.toc`)** | **2 files** | **BUILD ARTIFACTS**. File trung gian khi biên dịch TeX sang PDF. | **Xóa khỏi Git tracking & Chặn `*.aux`, `*.toc`,...** |
| **Thông tin bảo mật (`ssh_kv260_*.txt`)** | **Chưa track** (Untracked) | **CỰC KỲ NGUY HIỂM**. Lộ mật khẩu SSH, Tailscale, Kria KV260 của nhóm. Nếu ai đó vô tình `git add .`, toàn bộ thông tin sẽ lên mạng. | **Chặn ngay lập tức trong `.gitignore`** |
| **Thư viện Board (`XilinxBoardStore-2026.1/`)** | **Chưa track** (Untracked) | **DUNG LƯỢNG LỚN**. Chứa định nghĩa bo mạch của Xilinx, nên để ngoài hoặc cấu hình qua Vivado Hub. | **Chặn thư mục `XilinxBoardStore*/`** |
| **Hex Stimulus Files (`*.hex`)** | **169 files (34.0 MB)** | **CẦN THIẾT**. Đây là ma trận vector đầu vào và kết quả mẫu để các testbench SystemVerilog `$readmemh` đối chiếu bit-exact. Nếu xóa đi, thành viên khác clone về sẽ không chạy được simulation. | **GIỮ LẠI TRONG GIT** |
| **Mã nguồn RTL, Scripts, Constraints** | **127 files (~1.4 MB)** | **CỐT LÕI**. Tài sản trí tuệ quan trọng nhất của đồ án. | **GIỮ LẠI TRONG GIT** |
| **Báo cáo định dạng text (`*.rpt`)** | **21 files (0.74 MB)** | **HỮU ÍCH**. Lưu kết quả post-synthesis, post-route timing slack và utilization để làm báo cáo đồ án. Dung lượng rất nhỏ. | **GIỮ LẠI TRONG GIT** |

---

## 3. Bản Dự Thảo `.gitignore` Chuẩn Công Nghiệp (Đề Xuất Cập Nhật)

Dưới đây là nội dung tệp `.gitignore` hoàn chỉnh, được phân chia theo từng khối công nghệ (Hardware EDA, Simulation, Python, AI/ML, OS & Security):

```gitignore
# ==============================================================================
# .gitignore for Capstone_Project (FPGA ViT/MHA Hardware-Software Co-Design)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. HARDWARE EDA & SIMULATION BUILD ARTIFACTS (VIVADO, ICARUS, GTKWAVE)
# ------------------------------------------------------------------------------
# Waveform execution dumps (Extremely large binary files: 50MB - 400MB each)
*.vcd
*.wdb
*.vcd.fst
*.fsdb

# Icarus Verilog compiled simulation binaries
*.vvp
a.out

# Vivado internal synthesis/implementation cache & temporary directories
.Xil/
*.Xil/
Project/src/Hardware/synthesis/.Xil/
mha_kv260_deploy.cache/
mha_kv260_deploy.hw/
mha_kv260_deploy.ip_user_files/
mha_kv260_deploy.runs/
mha_kv260_deploy.sim/
hd_visual/
*.jou
*.backup.jou
*.backup.log
vivado*.log
vivado*.jou
clockInfo.txt

# Xilinx Board Store local cache (Can be downloaded via Vivado)
XilinxBoardStore*/
Reference/

# ------------------------------------------------------------------------------
# 2. SECURITY & SENSITIVE CREDENTIALS (NEVER COMMIT)
# ------------------------------------------------------------------------------
*ssh*.txt
*password*.txt
*credential*.txt
*.key
*.pem
*.env
id_rsa*
secrets/

# ------------------------------------------------------------------------------
# 3. PYTHON, PYTORCH & JUPYTER NOTEBOOK ARTIFACTS
# ------------------------------------------------------------------------------
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Dataset cache directories
cifar10_cache/
model_notebook/cifar10_cache/
*.tar.gz
*.zip

# Intermediate training checkpoints (Keep only final golden models)
model_notebook/ivit_surgery/*_latest.pth
model_notebook/models_cache/*_history.pt
*.ckpt

# ------------------------------------------------------------------------------
# 4. LATEX & DOCUMENTATION BUILD ARTIFACTS
# ------------------------------------------------------------------------------
*.aux
*.toc
*.out
*.fls
*.fdb_latexmk
*.synctex.gz
*.bbl
*.blg
*.nav
*.snm
*.vrb
~$*.docx

# ------------------------------------------------------------------------------
# 5. OS & IDE TEMPORARY FILES
# ------------------------------------------------------------------------------
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
*.swp
*.swo
```

---

## 4. Kế Hoạch Các Bước Thực Hiện (Chờ Duyệt - CHƯA THỰC THI)

> [!IMPORTANT]
> **Toàn bộ các bước dưới đây CHỈ là kế hoạch để bạn xem xét và phê duyệt.**
> Chúng tôi **chưa chạy bất kỳ lệnh nào** can thiệp vào mã nguồn hay kho lưu trữ Git của bạn.

Khi bạn đồng ý phê duyệt kế hoạch, quy trình chuẩn để xử lý dứt điểm sẽ diễn ra trong **4 bước tinh gọn**:

### Bước 1: Dừng tiến trình `git push` đang bị treo (Nếu terminal vẫn đang chạy)
* Nhấn `Ctrl + C` tại cửa sổ Terminal đang chạy `git push origin DuyTuan321026` để giải phóng tiến trình upload 2.4GB vô ích.

### Bước 2: Ghi nội dung mới vào tệp `.gitignore`
* Áp dụng bản dự thảo `.gitignore` chuẩn ở Mục 3 vào tệp `.gitignore` tại thư mục gốc của `Capstone_Project`.

### Bước 3: Gỡ bỏ các tệp rác khỏi Git Index mà KHÔNG XÓA tệp trên đĩa cứng
* Chạy lệnh `git rm --cached` để báo cho Git ngừng theo dõi toàn bộ tệp waveform `.vcd`, bộ nhớ đệm `.Xil/`, tệp `.vvp`, và các nhật ký `.log`/`.jou`.
* *Lưu ý*: Tùy chọn `--cached` bảo đảm **100% các tệp `.vcd` và `.vvp` trên ổ cứng máy bạn vẫn còn nguyên vẹn**, chỉ xóa vết theo dõi trong cơ sở dữ liệu Git.
* Lệnh mẫu:
  ```bash
  git rm -r --cached "Project/src/Hardware/synthesis/.Xil"
  git rm --cached Project/src/Hardware/rtl_ver02*/*.vcd
  git rm --cached Project/src/Hardware/rtl_ver02*/*.vvp
  git rm --cached Project/src/Hardware/rtl_ver02/a.out
  git rm --cached Project/src/Hardware/synthesis/*.log Project/src/Hardware/synthesis/*.jou
  git rm --cached Notes/*.aux Notes/*.log
  ```

### Bước 4: Tạo commit dọn dẹp và đẩy mã siêu tốc lên GitHub
* Thực hiện:
  ```bash
  git add .gitignore
  git commit -m "build: optimize .gitignore and untrack simulation waveforms and vivado cache"
  git push origin DuyTuan321026
  ```
* **Kết quả thu được**:
  * Dung lượng gói đẩy giảm từ **~2.4 GB xuống còn vài megabyte**.
  * Quá trình `git push` hoàn thành trong **vài giây** thay vì bị treo hàng chục phút.
  * Không bao giờ bị dính lỗi `git add .` nhầm file rác hoặc file mật khẩu `ssh_kv260_Long_GiaThoai.txt` trong các lần làm việc tiếp theo.

---

## 5. Trạng Thái & Lời Kêu Gọi Phê Duyệt

Báo cáo này đã được hoàn thành đầy đủ để bạn thẩm định. Vui lòng xem xét:
1. Bạn có muốn điều chỉnh thêm bớt đuôi tệp nào trong danh sách `.gitignore` ở Mục 3 không?
2. Nếu bạn đồng ý với kế hoạch trên, hãy gửi phản hồi xác nhận để tôi tiến hành thực thi chuẩn xác và an toàn từng bước cho bạn!
