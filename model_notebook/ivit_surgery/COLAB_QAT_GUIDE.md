# Google Colab Execution Guide: Phase 3 Quantization-Aware Training (QAT)
## MobileViT-XX-Small with I-ViT for Xilinx Kria KV260 FPGA Deployment

---

## 📌 Overview

This guide provides step-by-step instructions for running **Phase 3: Quantization-Aware Training (QAT)** on a single Google Colab GPU (T4, V100, or A100). The goal is to fine-tune the surgered `apple/mobilevit-xx-small` architecture and recover the accuracy drop caused by integer approximations (**Shiftmax**, **ShiftGELU**, and **IntLayerNorm**) back toward the 69% Top-1 ImageNet baseline.

An interactive, pre-configured Colab notebook is available at:
* [`MobileViT_IViT_QAT_Colab.ipynb`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/MobileViT_IViT_QAT_Colab.ipynb) (in `model_notebook/`)
* [`ivit_surgery/MobileViT_IViT_QAT_Colab.ipynb`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/ivit_surgery/MobileViT_IViT_QAT_Colab.ipynb) (in `ivit_surgery/`)

---

## 🛠️ Step 1: Configure Colab GPU Runtime

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the provided notebook [`MobileViT_IViT_QAT_Colab.ipynb`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/MobileViT_IViT_QAT_Colab.ipynb) or open a new notebook.
3. Switch runtime to GPU:
   * Click **Runtime** in the top menu -> **Change runtime type**.
   * Under **Hardware accelerator**, select **T4 GPU** (free tier) or **A100 / V100** (Colab Pro).
4. Verify GPU allocation:
   ```python
   !nvidia-smi
   import torch
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"Device Name: {torch.cuda.get_device_name(0)}")
   ```

---

## 💾 Step 2: Mount Google Drive (Persistent Storage)

> [!IMPORTANT]
> Colab instances are ephemeral. If your session disconnects or times out, all files in `/content` are deleted. Always mount Google Drive so checkpoints (`mobilevit_ivit_qat_best.pth`) are automatically saved to persistent storage.

Run this cell:
```python
from google.colab import drive
import os

drive.mount('/content/drive')

# Create dedicated directory for QAT weights and logs
DRIVE_SAVE_DIR = '/content/drive/MyDrive/MobileViT_IViT_QAT'
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
print(f"Checkpoints will sync to: {DRIVE_SAVE_DIR}")
```

---

## 📦 Step 3: Clone Repositories & Install Dependencies

Run the following cell to install requirements and clone the I-ViT and Capstone repositories into Colab:

```bash
# 1. Install required packages
!pip install -q transformers timm tqdm torchvision matplotlib

import os, sys

# 2. Clone I-ViT quantization framework
if not os.path.exists('/content/I-ViT'):
    !git clone https://github.com/zkkli/I-ViT.git /content/I-ViT

# 3. Clone Capstone Project repository (if not uploading directly)
if not os.path.exists('/content/Capstone_project'):
    !git clone https://github.com/Planterian/Capstone_project.git /content/Capstone_project

# 4. Set environment and sys.path
os.environ["IVIT_ROOT"] = "/content/I-ViT"
if "/content/I-ViT" not in sys.path:
    sys.path.insert(0, "/content/I-ViT")

NOTEBOOK_DIR = "/content/Capstone_project/model_notebook"
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

%cd /content/Capstone_project/model_notebook
```

### Calibrated Checkpoint Verification:
Ensure `mobilevit_ivit_calibrated.pth` (7.37 MB) is available:
```python
CALIBRATED_CKPT = "/content/Capstone_project/model_notebook/ivit_surgery/mobilevit_ivit_calibrated.pth"
if not os.path.exists(CALIBRATED_CKPT):
    # Option: copy from Google Drive if you backed it up there
    !cp /content/drive/MyDrive/mobilevit_ivit_calibrated.pth {CALIBRATED_CKPT}

assert os.path.exists(CALIBRATED_CKPT), "Calibrated checkpoint not found!"
print("✓ Calibrated checkpoint verified.")
```

---

## 🖼️ Step 4: Dataset Setup (Choose 1 of 3 Options)

### Option A: Kaggle ImageNet-Mini (~3.6 GB) — Recommended for Colab
ImageNet-Mini contains all 1,000 ImageNet classes with 34,745 training images and 3,923 validation images. It downloads in ~2 minutes and fits easily on Colab's local disk.

```bash
# Method 1: If you uploaded kaggle.json to /content/
!mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Method 2: Or enter credentials directly via Python environment:
# import os, getpass
# os.environ['KAGGLE_USERNAME'] = input("Kaggle Username: ")
# os.environ['KAGGLE_KEY'] = getpass.getpass("Kaggle API Key: ")

# Download and extract directly to Colab local SSD (/content/dataset)
!pip install -q kaggle
!kaggle datasets download -d ifigotin/imagenetmini-1000 -p /content/dataset --unzip

# Dataset path:
# Train: /content/dataset/imagenet-mini/train
# Val:   /content/dataset/imagenet-mini/val
```

> [!TIP]
> **Performance Tip:** Always unpack image datasets onto Colab's local SSD (`/content/dataset`) rather than reading directly from Google Drive (`/content/drive/`). Direct Drive reads create network I/O bottlenecks that drastically slow down PyTorch DataLoader workers.

---

### Option B: Full ImageNet-1K from Google Drive
If you already have ImageNet-1K stored in Google Drive:
```bash
# Unpack train and val tarballs to local SSD for maximum I/O speed:
!mkdir -p /content/imagenet
!tar -xf /content/drive/MyDrive/imagenet_val.tar -C /content/imagenet/val
!tar -xf /content/drive/MyDrive/imagenet_train.tar -C /content/imagenet/train
```

---

### Option C: Synthetic Smoke-Test Mode (Zero Download)
If you want to quickly test the training loop, backpropagation, and checkpointing without downloading any dataset:
* Simply set `DATA_DIR = None`.
* The pipeline automatically activates `SyntheticQATDataset` to run a smoke test in under 60 seconds.

---

## ⚙️ Step 5: Hyperparameters & Configuration

The recommended hyperparameters for single-GPU QAT fine-tuning on Colab:

| Parameter | Recommended Value | Rationale |
|---|---|---|
| **Epochs** | `40` | Sufficient for integer boundary adaptation and convergence |
| **Batch Size** | `64` | Fits easily in 16GB T4 VRAM |
| **Transformer LR** | `2e-4` (or `1e-4`) | Standard fine-tuning rate for integer layers |
| **CNN Stem LR** | `1e-6` | Microscopic rate to preserve inverted residual features |
| **CNN BatchNorm** | **Strictly `eval()`** | **CỰC KỲ QUAN TRỌNG:** Giữ nguyên running stats đã huấn luyện của Apple, chống nát phân phối trên tập mini! |
| **Transforms** | **BGR `[0, 1]`** | **CỰC KỲ QUAN TRỌNG:** Apple MobileViT nhận ảnh BGR tỉ lệ [0, 1], TUYỆT ĐỐI KHÔNG dùng ImageNet Normalize! |
| **Classifier Head LR** | `2e-4` | Aligned with Transformer blocks |
| **Warmup Epochs** | `3` | Linear warmup to stabilize integer bounds |
| **Label Smoothing** | `0.0` | Cung cấp tín hiệu gradient nhãn thực rõ ràng nhất |
| **Gradient Clipping** | `1.0` | Cho phép STE gradient cập nhật thông thoáng |
| **DataLoader Workers** | `2` | Optimal for Colab 2-core virtual CPUs (tránh cảnh báo quá tải workers) |
| **Fast PTQ Calib** | `True` | Tự động chạy định cỡ 15 giây trên 1,024 ảnh thực BGR nếu có dataset |

---

## 🏃 Step 6: Execution Options

### Option 1: Run via the Interactive Colab Notebook (Recommended)
Open [`MobileViT_IViT_QAT_Colab.ipynb`](file:///c:/Users/tuan2/Desktop/Capstone_Project/model_notebook/MobileViT_IViT_QAT_Colab.ipynb) and run cells sequentially:
1. **Cell 7**: Builds the surgered model, adapts buffer shapes, and calls `enable_qat_mode(model)`.
2. **Cell 9**: Runs baseline evaluation on the pre-trained calibrated model to establish the starting point.
3. **Cell 10**: Runs the QAT training loop with live `tqdm` progress bars and saves the best model.
4. **Cell 11**: Plots loss and accuracy recovery curves with `matplotlib`.
5. **Cell 12**: Inspects dyadic multipliers and bit-shifts for Phase 4 FPGA export.

---

### Option 2: Run via CLI Command in Colab Terminal
In a Colab code cell:

```bash
!python -m ivit_surgery.mobilevit_ivit_qat \
    --calibrated-ckpt /content/Capstone_project/model_notebook/ivit_surgery/mobilevit_ivit_calibrated.pth \
    --data-dir /content/dataset/imagenet-mini \
    --epochs 30 \
    --batch-size 64 \
    --transformer-lr 1e-4 \
    --cnn-lr 1e-6 \
    --warmup-epochs 3 \
    --device cuda \
    --num-workers 4
```

To completely freeze the CNN stem instead of using micro-LR:
```bash
    --freeze-cnn
```

---

## 🛡️ Step 7: Colab Stability & Pro-Tips

### 1. Prevent Browser Inactivity Disconnections
Open the browser developer console (**F12** or **Ctrl+Shift+I** -> **Console**) and paste:
```javascript
function ClickConnect(){
    console.log("Keeping Colab connection alive...");
    document.querySelector("colab-connect-button")?.click();
}
setInterval(ClickConnect, 60000);
```

### 2. GPU Memory Cleanup
If you encounter `CUDA out of memory`:
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

### 3. Automatic Checkpoint Sync to Google Drive
The provided training code automatically syncs `mobilevit_ivit_qat_best.pth` to `/content/drive/MyDrive/MobileViT_IViT_QAT/` whenever a new best validation accuracy is achieved. You can verify it directly in Google Drive.

---

## 🎯 Verification Checklist for Phase 4 Transition

After QAT completes, ensure you have:
1. [x] **`mobilevit_ivit_qat_best.pth`** saved with model weights, optimizer state, and validation accuracy.
2. [x] **Observers remained locked** throughout training (`QuantAct.running_stat == False`).
3. [x] **Top-1 Accuracy Recovered** toward the 69% baseline.
4. [x] **Dyadic Scaling Factors ($M, E$)** verified for all layers.

You are now ready to proceed to **Phase 4: Hardware Weight & Dyadic Scale Export for the Kria KV260 Systolic Array**!
