{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "auto_install_deps"
      },
      "outputs": [],
      "source": [
        "# Auto-install missing dependencies\n",
        "import sys\n",
        "import subprocess\n",
        "import importlib.util\n",
        "\n",
        "required_packages = {\n",
        "    'numpy': 'numpy',\n",
        "    'matplotlib': 'matplotlib',\n",
        "    'torch': 'torch',\n",
        "    'torchvision': 'torchvision',\n",
        "    'IPython': 'IPython',\n",
        "    'pandas': 'pandas'\n",
        "}\n",
        "missing = [pip_name for mod, pip_name in required_packages.items() if importlib.util.find_spec(mod) is None]\n",
        "\n",
        "if missing:\n",
        "    print(f'Installing missing packages: {missing}')\n",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])\n",
        "    print('Installation complete!')\n",
        "else:\n",
        "    print('All required packages are already installed.')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "aXCOrVVISB7G"
      },
      "outputs": [],
      "source": [
        "#import req libs\n",
        "import random\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython.display import display\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader\n",
        "from torchvision import datasets, transforms"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bs6fu17tw8pl",
        "outputId": "b43cd073-cc5a-4b7a-f60c-d74aeb3143f9"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<torch._C.Generator at 0x7f220130d450>"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ],
      "source": [
        "# Make comparisons between experiments reproducible.\n",
        "seed = 42\n",
        "random.seed(seed)\n",
        "np.random.seed(seed)\n",
        "torch.manual_seed(seed)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "aOLBjJWhVM0M"
      },
      "outputs": [],
      "source": [
        "#Using device\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "S-f-oDrojryx"
      },
      "outputs": [],
      "source": [
        "#Variables\n",
        "#-Cifar 10-\n",
        "img_size = 32\n",
        "num_channels = 3\n",
        "num_classes = 10\n",
        "\n",
        "#-ViT archi-\n",
        "patch_size = 4\n",
        "num_patches = (img_size // patch_size) ** 2\n",
        "num_heads = 8     #Divided dimensions so each one in this case takes on 32 info after finish it will be merge back to 256\n",
        "embed_dim = 256   #Act as a detailed file that containt attributes of image, the more the better\n",
        "mlp_dim = 512     #Buffs up the dimension so that it can learn much more detailed attribute after that it will go back to 256 an so on\n",
        "drop_rate = 0.1   #Deleted some data so that the model won't be overfitting\n",
        "\n",
        "#-Training config\n",
        "learn_rate = 3e-4\n",
        "epochs = 30\n",
        "batch_size = 512\n",
        "depth = 10 #Ammount of transformer encoder blocks that stack up each other"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "nAw_I8IBr7bU"
      },
      "outputs": [],
      "source": [
        "# Random number for mean and std\n",
        "cifar_mean = (0.4914, 0.4822, 0.4465)\n",
        "cifar_std = (0.2470, 0.2435, 0.2616)\n",
        "\n",
        "train_transforms = transforms.Compose([\n",
        "    transforms.RandomCrop(img_size, padding=4),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(cifar_mean, cifar_std)\n",
        "    ])\n",
        "eval_transforms = transforms.Compose([\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(cifar_mean, cifar_std)\n",
        "    ])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "Q7gZoqe7L2ce",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7ddb4abe-5ffd-4216-9682-21e4b11a7cd1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Đã chuyển cấu hình nạp dữ liệu mặc định sang Google Drive cache ở ô bên dưới.\n"
          ]
        }
      ],
      "source": [
        "#Getting training and test dataset (Đã comment - dữ liệu hiện tại được load từ Google Drive ở ô phía dưới)\n",
        "# train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)\n",
        "# test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transforms)\n",
        "print(\"Đã chuyển cấu hình nạp dữ liệu mặc định sang Google Drive cache ở ô bên dưới.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c52039f9"
      },
      "source": [
        "### Fast Dataset Loading using Saved PyTorch Tensors\n",
        "To prevent downloading the dataset on subsequent runs, you can serialize and save your processed datasets locally (or inside Google Drive if you mount it) using `torch.save` and `torch.load`."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e78602a9",
        "outputId": "a4cffca9-f392-424a-e52d-55015ce9d1b6"
      },
      "source": [
        "import os\n",
        "\n",
        "# Try to mount Google Drive (only works in Colab)\n",
        "try:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive', force_remount=True)\n",
        "    print(\"Google Drive mounted successfully.\")\n",
        "    cache_dir = '/content/drive/MyDrive/Capstone/data'\n",
        "except (ModuleNotFoundError, ImportError):\n",
        "    print(\"Not running in Colab. Using local directory.\")\n",
        "    cache_dir = './cifar10_cache'\n",
        "except Exception as e:\n",
        "    print(f\"Could not mount Google Drive: {e}. Defaulting to local space.\")\n",
        "    cache_dir = './cifar10_cache'\n",
        "\n",
        "os.makedirs(cache_dir, exist_ok=True)\n",
        "train_pt_path = os.path.join(cache_dir, 'train_data.pt')\n",
        "test_pt_path = os.path.join(cache_dir, 'test_data.pt')\n",
        "\n",
        "# Check if pre-saved datasets exist\n",
        "if os.path.exists(train_pt_path) and os.path.exists(test_pt_path):\n",
        "    print(\"Loading datasets from saved cache...\")\n",
        "    train_data = torch.load(train_pt_path, weights_only=False)\n",
        "    test_data = torch.load(test_pt_path, weights_only=False)\n",
        "else:\n",
        "    print(\"Cached files not found. Downloading and saving CIFAR10 dataset...\")\n",
        "    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)\n",
        "    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transforms)\n",
        "\n",
        "    torch.save(train_data, train_pt_path)\n",
        "    torch.save(test_data, test_pt_path)\n",
        "    print(f\"Dataset saved successfully to {cache_dir}!\")"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n",
            "Google Drive mounted successfully.\n",
            "Loading datasets from saved persistent Google Drive cache...\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "R90n1QbpNylT"
      },
      "outputs": [],
      "source": [
        "#Coverting data into dataloader (turn datas into batches), also helps HW not to look at 50k images in one go\n",
        "train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,\n",
        "                          pin_memory=torch.cuda.is_available())\n",
        "test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,\n",
        "                         pin_memory=torch.cuda.is_available())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "ksIIWRSmSQVu"
      },
      "outputs": [],
      "source": [
        "#Part 1 - Build Patch Embedding\n",
        "class PatchEmbedding(nn.Module):\n",
        "  def __init__(self, num_channels, embed_dim, patch_size):\n",
        "    super().__init__()\n",
        "    self.patch_embed = nn.Conv2d(in_channels = num_channels, out_channels = embed_dim, kernel_size=patch_size, stride=patch_size)  #just rearrange the dim\n",
        "\n",
        "  def forward(self, x): # will represent the input dataset\n",
        "    x = self.patch_embed(x)\n",
        "    x = x.flatten(2).transpose(1,2) #example below already shown\n",
        "    return x"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "djb--nCsf5BG"
      },
      "outputs": [],
      "source": [
        "#For testing dimension (if u want to test then uncomment the codes)\n",
        "#images, labels = next(iter(train_loader))\n",
        "#patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)\n",
        "#print(\"images shape: \", images.shape)\n",
        "#when running this, we will get the detailed: [batch_size, channels, image_H, image_W]\n",
        "#print(\"embedded shape: \", patch_embed(images).shape)\n",
        "#when running this after conv2D, we will get the detailed: [batch_size, embedded_dimension, [8,8] vector represent the position for each patch (if u then multi it then you will get the total patches which is 64 in this case)]\n",
        "#print(\"flatten shape: \", patch_embed(images).flatten(2).shape)\n",
        "#after that, we need to transpose the vector at position 1,2\n",
        "#print(\"transpose: \", patch_embed(images).flatten(2).transpose(1,2).shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "GsCMPv88LTwB"
      },
      "outputs": [],
      "source": [
        "#Part 1.5 - MLP class so that we use it in Transformer Encoder\n",
        "#Change up a bit cause we have drop_out rate to prevent overfitting\n",
        "class MLP(nn.Module):\n",
        "  def __init__(self, embed_dim, mlp_dim, drop_rate):\n",
        "    super().__init__()\n",
        "    self.linear1 = nn.Linear(in_features = embed_dim, out_features = mlp_dim)\n",
        "    self.gelu = nn.GELU()\n",
        "    self.linear2 = nn.Linear(in_features = mlp_dim, out_features = embed_dim)\n",
        "    self.dropout = nn.Dropout(drop_rate)\n",
        "\n",
        "  def forward(self, x):\n",
        "    x = self.dropout(self.gelu(self.linear1(x))) #project from 256 -> 512 with 10% drop_out with GELU\n",
        "    x = self.dropout(self.linear2(x)) #then project back from 512 -> 256 and so on\n",
        "    return x"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "V5aVB2-Zl5EH"
      },
      "outputs": [],
      "source": [
        "#Part 2 - Transformer Encoder (2 Layer Normalization, MHA, Residuals)\n",
        "#This follow the picture of encoder in the pdf file\n",
        "class TransformerEncoder(nn.Module):\n",
        "  def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):\n",
        "    super().__init__()\n",
        "    self.layer_norm_1 = nn.LayerNorm(embed_dim)\n",
        "    self.multi_head_atten = nn.MultiheadAttention(embed_dim, num_heads, dropout = drop_rate, batch_first = True)\n",
        "    self.layer_norm_2 = nn.LayerNorm(embed_dim)\n",
        "    self.multi_layer_perceptron = MLP(embed_dim, mlp_dim, drop_rate)\n",
        "\n",
        "  def forward(self, x):\n",
        "    residual_1 = x\n",
        "    attention_output = self.multi_head_atten(self.layer_norm_1(x),self.layer_norm_1(x),self.layer_norm_1(x))[0]\n",
        "    x = attention_output + residual_1\n",
        "    residual_2 = x\n",
        "    mlp_output = self.multi_layer_perceptron(self.layer_norm_2(x))\n",
        "    x = mlp_output + residual_2\n",
        "    return x"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e371a398"
      },
      "source": [
        "### Part 2.1 - Scratch Multihead Attention\n",
        "This replacement for `nn.MultiheadAttention` breaks down the Scaled Dot-Product Attention into explicit steps (Query/Key/Value projections, Matrix Multiplication, Scaling, Softmax via Lookup table simulation or standard softmax, and Out projection). This allows easy mapping to your SystemVerilog RTL modules."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "042345cf"
      },
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "class ScratchMultiheadAttention(nn.Module):\n",
        "    def __init__(self, embed_dim, num_heads, dropout=0.0):\n",
        "        super().__init__()\n",
        "        self.embed_dim = embed_dim\n",
        "        self.num_heads = num_heads\n",
        "        self.head_dim = embed_dim // num_heads\n",
        "\n",
        "        assert self.head_dim * num_heads == embed_dim, \"embed_dim must be divisible by num_heads\"\n",
        "\n",
        "        # Separate linear layers to make hardware state mappings explicit\n",
        "        self.q_proj = nn.Linear(embed_dim, embed_dim)\n",
        "        self.k_proj = nn.Linear(embed_dim, embed_dim)\n",
        "        self.v_proj = nn.Linear(embed_dim, embed_dim)\n",
        "        self.out_proj = nn.Linear(embed_dim, embed_dim)\n",
        "\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "        self.scale = 1.0 / (self.head_dim ** 0.5)\n",
        "\n",
        "        # Mô phỏng chính xác luồng dữ liệu phần cứng theo kiến trúc HW-02 đến HW-09:\n",
        "        # Stage 1 & 2 (QK^T và Scaler) giữ ở phân giải cao (INT32/INT16) -> Dùng FP32 để PyTorch không ép xuống INT8\n",
        "        # Stage 3 (Softmax LUT) ép xuống INT8 -> Dùng QuantStub\n",
        "        # Stage 4 (Score x V) chạy INT8 x INT8 -> Dùng FloatFunctional\n",
        "        self.ff_matmul_qv = nn.quantized.FloatFunctional()\n",
        "        self.attn_probs_quant = quantization.QuantStub()\n",
        "\n",
        "    def forward(self, q, k, v):\n",
        "        B, N, C = q.shape\n",
        "\n",
        "        # Input Ping-Pong BRAM Buffers (HW-02) - Đầu vào INT8, xuất ra INT8\n",
        "        q_proj = self.q_proj(q) # [B, N, C]\n",
        "        k_proj = self.k_proj(k) # [B, N, C]\n",
        "        v_proj = self.v_proj(v) # [B, N, C]\n",
        "\n",
        "        q_heads = q_proj.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)\n",
        "        k_heads = k_proj.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)\n",
        "        v_heads = v_proj.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # Giữ nguyên INT8 cho Stage 4\n",
        "\n",
        "        # Chuẩn bị cho Stage 1 & 2: Giải lượng tử (Dequantize) q_heads và k_heads để mô phỏng INT32/INT16\n",
        "        # Tránh việc FloatFunctional của PyTorch ép ngay kết quả QK^T xuống INT8 gây mất mát thông tin (Collapse)\n",
        "        if q_heads.is_quantized: q_heads_f = q_heads.dequantize()\n",
        "        else: q_heads_f = q_heads\n",
        "        \n",
        "        if k_heads.is_quantized: k_heads_f = k_heads.dequantize()\n",
        "        else: k_heads_f = k_heads\n",
        "\n",
        "        # STAGE 1: QK^T GEMM Engine (HW-03) - Mô phỏng INT32 Partial Sums\n",
        "        attn_scores = torch.matmul(q_heads_f, k_heads_f.transpose(-2, -1))\n",
        "\n",
        "        # STAGE 2: Scaler Unit (HW-04) - Mô phỏng ASR 1/sqrt(d_k) cho ra INT16\n",
        "        attn_scores = attn_scores * self.scale\n",
        "\n",
        "        # STAGE 3: Hardware Softmax (HW-05) - Mô phỏng Max-Sub + Exp LUT ra INT8 Probabilities\n",
        "        attn_probs = F.softmax(attn_scores, dim=-1)\n",
        "        attn_probs = self.dropout(attn_probs)\n",
        "        attn_probs = self.attn_probs_quant(attn_probs) # Ép vể INT8 y hệt đầu ra của Softmax LUT\n",
        "\n",
        "        # STAGE 4: Score x V GEMM Engine (HW-07) - Nhân 2 ma trận INT8\n",
        "        context = self.ff_matmul_qv.matmul(attn_probs, v_heads)\n",
        "\n",
        "        # Output Ping-Pong Buffer (HW-09)\n",
        "        context = context.transpose(1, 2).reshape(B, N, C)\n",
        "        output = self.out_proj(context)\n",
        "\n",
        "        return output, attn_probs"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "b73907c0"
      },
      "source": [
        "### Part 2.2 - Hardware-Friendly Transformer Encoder\n",
        "We now redefine our `TransformerEncoder` block to use the customized scratch attention module rather than PyTorch's default blackbox container."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "a0f49ed8"
      },
      "source": [
        "class HWFriendlyTransformerEncoder(nn.Module):\n",
        "    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):\n",
        "        super().__init__()\n",
        "        self.layer_norm_1 = nn.LayerNorm(embed_dim)\n",
        "        self.multi_head_atten = ScratchMultiheadAttention(embed_dim, num_heads, dropout=drop_rate)\n",
        "        self.layer_norm_2 = nn.LayerNorm(embed_dim)\n",
        "        self.multi_layer_perceptron = MLP(embed_dim, mlp_dim, drop_rate)\n",
        "\n",
        "        # FloatFunctional for quantization-safe residual additions\n",
        "        self.ff_residual_1 = nn.quantized.FloatFunctional()\n",
        "        self.ff_residual_2 = nn.quantized.FloatFunctional()\n",
        "\n",
        "    def forward(self, x):\n",
        "        residual_1 = x\n",
        "        normalized_x = self.layer_norm_1(x)\n",
        "        attention_output, _ = self.multi_head_atten(normalized_x, normalized_x, normalized_x)\n",
        "\n",
        "        # Residual add\n",
        "        x = self.ff_residual_1.add(attention_output, residual_1)\n",
        "\n",
        "        residual_2 = x\n",
        "        normalized_x2 = self.layer_norm_2(x)\n",
        "        mlp_output = self.multi_layer_perceptron(normalized_x2)\n",
        "        x = self.ff_residual_2.add(mlp_output, residual_2)\n",
        "        return x"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "XeBGSryC-D30"
      },
      "outputs": [],
      "source": [
        "#Part 3 - Vision Transformer\n",
        "class VisionTransformer(nn.Module):\n",
        "    def __init__(self, img_size, patch_size, num_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):\n",
        "        super().__init__()\n",
        "        if img_size % patch_size != 0:\n",
        "            raise ValueError(\"img_size must be divisible by patch_size\")\n",
        "        num_patches = (img_size // patch_size) ** 2\n",
        "        self.patch_embedding = PatchEmbedding(num_channels, embed_dim, patch_size)\n",
        "        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim)) #batch_dim & 1 vector & embedded dim\n",
        "        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))\n",
        "        self.transformer_layers = nn.Sequential(*[\n",
        "              TransformerEncoder(embed_dim, num_heads, mlp_dim, drop_rate)\n",
        "              for _ in range(depth)\n",
        "            ])\n",
        "\n",
        "        self.mlp_head = nn.Sequential(\n",
        "            nn.LayerNorm(embed_dim),\n",
        "            nn.Linear(embed_dim, num_classes)\n",
        "        )\n",
        "\n",
        "    def forward(self,x):\n",
        "        x = self.patch_embedding(x)\n",
        "        B = x.size(0)\n",
        "\n",
        "        cls_tokens = self.cls_token.expand(B , -1, -1)\n",
        "        x = torch.cat((cls_tokens, x), 1)\n",
        "        x = x + self.pos_embed\n",
        "        x = self.transformer_layers(x)\n",
        "        x = x[:,0]\n",
        "        x = self.mlp_head(x)\n",
        "        return x"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "tZBoZBVnMFTv"
      },
      "outputs": [],
      "source": [
        "#Initiate Model\n",
        "# Instantiate model\n",
        "model = VisionTransformer(\n",
        "    img_size, patch_size, num_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate\n",
        ").to(device)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "bc7k-I8JNLWL"
      },
      "outputs": [],
      "source": [
        "#optimizer and crossentropy\n",
        "criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #useful for training multi class and measure how wrong our model is\n",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=5e-4)\n",
        "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "3DomYzxnPspg"
      },
      "outputs": [],
      "source": [
        "#Training loop\n",
        "def train(model, loader, optimizer, criterion):\n",
        "    model.train()\n",
        "\n",
        "    total_loss, correct = 0, 0\n",
        "\n",
        "    for x, y in loader:\n",
        "      x = x.to(device, non_blocking=True)\n",
        "      y = y.to(device, non_blocking=True)\n",
        "      optimizer.zero_grad()\n",
        "      out = model(x)\n",
        "      loss = criterion(out, y)\n",
        "      loss.backward()\n",
        "      optimizer.step()\n",
        "\n",
        "      total_loss += loss.item() * x.size(0)\n",
        "      correct += (out.argmax(1) == y).sum().item()\n",
        "\n",
        "    return total_loss / len(loader.dataset), correct / len(loader.dataset)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "fpSKwbMUUWsr"
      },
      "outputs": [],
      "source": [
        "# Evaluation loop\n",
        "def evaluate(model, loader):\n",
        "    model.eval() #set mode evaluation\n",
        "    correct = 0\n",
        "    with torch.no_grad():\n",
        "        for x, y in loader:\n",
        "            x = x.to(device, non_blocking=True)\n",
        "            y = y.to(device, non_blocking=True)\n",
        "            out = model(x)\n",
        "            correct += (out.argmax(dim=1) == y).sum().item()\n",
        "    return correct / len(loader.dataset)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KmfxcCpvWXA7",
        "outputId": "ed7d91ac-a87b-4707-bbcb-9629f260f7ae"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Tìm thấy checkpoint đã lưu! Đang tải trọng số và lịch sử huấn luyện...\n",
            "Tải thành công! Độ chính xác ở Epoch cuối cùng: Test Acc = 76.33%\n"
          ]
        }
      ],
      "source": [
        "# Training and Saving/Loading baseline checkpoint to avoid retraining\n",
        "import os\n",
        "import torch\n",
        "\n",
        "# Set save directory (use Google Drive path if available, otherwise local)\n",
        "if os.path.exists('/content/drive'):\n",
        "    save_dir = '/content/drive/MyDrive/Capstone/model'\n",
        "else:\n",
        "    save_dir = './models_cache'\n",
        "os.makedirs(save_dir, exist_ok=True)\n",
        "\n",
        "checkpoint_path = os.path.join(save_dir, 'vit_fp32_baseline.pth')\n",
        "history_path = os.path.join(save_dir, 'vit_fp32_history.pt')\n",
        "\n",
        "train_accuracies, test_accuracies = [], []\n",
        "\n",
        "# Kiểm tra nếu checkpoint đã tồn tại để tránh train lại\n",
        "if os.path.exists(checkpoint_path) and os.path.exists(history_path):\n",
        "    print(\"Tìm thấy checkpoint đã lưu! Đang tải trọng số và lịch sử huấn luyện...\")\n",
        "    model.load_state_dict(torch.load(checkpoint_path, map_location=device))\n",
        "    history = torch.load(history_path, map_location='cpu')\n",
        "    train_accuracies = history['train_accuracies']\n",
        "    test_accuracies = history['test_accuracies']\n",
        "    print(f\"Tải thành công! Độ chính xác ở Epoch cuối cùng: Test Acc = {test_accuracies[-1]*100:.2f}%\")\n",
        "else:\n",
        "    print(\"Không tìm thấy checkpoint cũ. Bắt đầu quá trình huấn luyện từ đầu (30 Epochs)...\\n\")\n",
        "    for epoch in range(epochs):\n",
        "        train_loss, train_acc = train(model, train_loader, optimizer, criterion)\n",
        "        test_acc = evaluate(model, test_loader)\n",
        "        scheduler.step()\n",
        "        train_accuracies.append(train_acc)\n",
        "        test_accuracies.append(test_acc)\n",
        "        print(f\"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}\")\n",
        "\n",
        "    # Tiến hành lưu checkpoint sau khi train xong\n",
        "    torch.save(model.state_dict(), checkpoint_path)\n",
        "    torch.save({\n",
        "        'train_accuracies': train_accuracies,\n",
        "        'test_accuracies': test_accuracies\n",
        "    }, history_path)\n",
        "    print(f\"\\nĐã huấn luyện xong và lưu checkpoint thành công tại: {checkpoint_path}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 360
        },
        "id": "DEuVyGJ5bCTq",
        "outputId": "68da0bbe-5cfc-43b1-ec09-e657b9ac9881"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "# Plot accuracy\n",
        "plt.plot(train_accuracies, label='Train Accuracy')\n",
        "plt.plot(test_accuracies, label='Test Accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "plt.title('Training and Test Accuracy')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Part 4 - Quantization Initialization (Prepare Only)\n",
        "This section initializes the quantization configurations and prepares the models for PTQ and QAT. No execution (calibration or training) happens here."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.ao.quantization as quantization\n",
        "\n",
        "# ==========================================\n",
        "# 1. Initialize PTQ Model\n",
        "# ==========================================\n",
        "ptq_model = QuantizableVisionTransformer(\n",
        "    img_size, patch_size, num_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate\n",
        ").to('cpu')\n",
        "\n",
        "ptq_model.load_state_dict(model.state_dict(), strict=False)\n",
        "ptq_model.eval()\n",
        "\n",
        "supported_engines = torch.backends.quantized.supported_engines\n",
        "if 'fbgemm' in supported_engines:\n",
        "    torch.backends.quantized.engine = 'fbgemm'\n",
        "elif 'onednn' in supported_engines:\n",
        "    torch.backends.quantized.engine = 'onednn'\n",
        "elif 'qnnpack' in supported_engines:\n",
        "    torch.backends.quantized.engine = 'qnnpack'\n",
        "else:\n",
        "    raise RuntimeError(f'No compatible quantization engine found.')\n",
        "\n",
        "ptq_model.qconfig = quantization.get_default_qconfig(torch.backends.quantized.engine)\n",
        "quantization.prepare(ptq_model, inplace=True)\n",
        "print(\"PTQ Model Prepared.\")\n",
        "\n",
        "# ==========================================\n",
        "# 2. Initialize QAT Model\n",
        "# ==========================================\n",
        "qat_model = QuantizableVisionTransformer(\n",
        "    img_size, patch_size, num_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate\n",
        ").to('cpu')\n",
        "\n",
        "qat_model.load_state_dict(model.state_dict(), strict=False)\n",
        "qat_model.train()\n",
        "\n",
        "qat_model.qconfig = quantization.get_default_qat_qconfig(torch.backends.quantized.engine)\n",
        "quantization.prepare_qat(qat_model, inplace=True)\n",
        "print(\"QAT Model Prepared.\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Part 5 - Train, Calibrate, Convert, and Evaluate (with Caching)\n",
        "This section checks if the quantized models have already been trained and saved to Google Drive. If so, it loads them to save time. Otherwise, it performs calibration (PTQ) or fine-tuning (QAT), converts them to INT8, and saves them."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "# Try to mount Google Drive\n",
        "try:\n",
        "    from google.colab import drive\n",
        "    if not os.path.exists('/content/drive'):\n",
        "        drive.mount('/content/drive', force_remount=True)\n",
        "    models_cache_dir = '/content/drive/MyDrive/Capstone/model'\n",
        "except:\n",
        "    print('Not running in Colab. Using local directory for model weights.')\n",
        "    models_cache_dir = './models_cache'\n",
        "\n",
        "os.makedirs(models_cache_dir, exist_ok=True)\n",
        "fp32_weights_path = os.path.join(models_cache_dir, 'vit_fp32_baseline.pth')\n",
        "ptq_weights_path = os.path.join(models_cache_dir, 'vit_ptq_int8.pth')\n",
        "qat_weights_path = os.path.join(models_cache_dir, 'vit_qat_int8.pth')\n",
        "\n",
        "def save_model_weights(m, path):\n",
        "    torch.save(m.state_dict(), path)\n",
        "    print(f\"Saved weights to {path}\")\n",
        "\n",
        "def load_model_weights_if_exists(m, path):\n",
        "    if os.path.exists(path):\n",
        "        m.load_state_dict(torch.load(path, map_location='cpu'))\n",
        "        print(f\"Loaded weights successfully from {path}\")\n",
        "        return True\n",
        "    return False\n",
        "\n",
        "# Save FP32 baseline if not exist\n",
        "if not os.path.exists(fp32_weights_path):\n",
        "    save_model_weights(model, fp32_weights_path)\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ==========================================\n",
        "# PTQ Execution\n",
        "# ==========================================\n",
        "# Create a dummy converted INT8 model to receive state_dict if cached\n",
        "ptq_model_int8 = quantization.convert(ptq_model, inplace=False)\n",
        "\n",
        "if load_model_weights_if_exists(ptq_model_int8, ptq_weights_path):\n",
        "    print(\"Skipping PTQ Calibration: Weights loaded from cache.\")\n",
        "else:\n",
        "    print(\"No cached PTQ weights found. Calibrating PTQ model...\")\n",
        "    with torch.no_grad():\n",
        "        for j, (images, _) in enumerate(train_loader):\n",
        "            ptq_model(images.to('cpu'))\n",
        "            if j >= 10:\n",
        "                break\n",
        "    # Convert and Save\n",
        "    ptq_model_int8 = quantization.convert(ptq_model, inplace=False)\n",
        "    save_model_weights(ptq_model_int8, ptq_weights_path)\n",
        "    print(\"PTQ Calibration & Conversion Done.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ==========================================\n",
        "# QAT Execution\n",
        "# ==========================================\n",
        "# Create a dummy converted INT8 model to receive state_dict if cached\n",
        "qat_model_int8 = quantization.convert(qat_model, inplace=False)\n",
        "\n",
        "if load_model_weights_if_exists(qat_model_int8, qat_weights_path):\n",
        "    print(\"Skipping QAT Fine-tuning: Weights loaded from cache.\")\n",
        "else:\n",
        "    print(\"No cached QAT weights found. Starting QAT Fine-tuning...\")\n",
        "    epochs_qat = 5\n",
        "    qat_model_train = qat_model.to(device)\n",
        "    optimizer_qat = torch.optim.AdamW(qat_model_train.parameters(), lr=1e-4, weight_decay=1e-4)\n",
        "    scheduler_qat = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=epochs_qat)\n",
        "\n",
        "    for epoch in range(epochs_qat):\n",
        "        qat_model_train.train()\n",
        "        if epoch == epochs_qat - 2:\n",
        "            print(\"Freezing Quantization Observers...\")\n",
        "            qat_model_train.apply(torch.ao.quantization.disable_observer)\n",
        "        if epoch == epochs_qat - 1:\n",
        "            print(\"Freezing Batch Norms...\")\n",
        "            qat_model_train.apply(torch.nn.intrinsic.qat.freeze_bn_stats)\n",
        "\n",
        "        correct = 0\n",
        "        total = 0\n",
        "        for j, (images, labels) in enumerate(train_loader):\n",
        "            images, labels = images.to(device), labels.to(device)\n",
        "            optimizer_qat.zero_grad()\n",
        "            outputs = qat_model_train(images)\n",
        "            loss = criterion(outputs, labels)\n",
        "            loss.backward()\n",
        "            optimizer_qat.step()\n",
        "            \n",
        "            correct += (outputs.argmax(1) == labels).sum().item()\n",
        "            total += labels.size(0)\n",
        "            \n",
        "        scheduler_qat.step()\n",
        "        print(f\"QAT Epoch {epoch+1}/{epochs_qat} - Train Acc: {correct/total * 100:.2f}%\")\n",
        "\n",
        "    # Convert and Save\n",
        "    qat_model_train.eval()\n",
        "    qat_model_train = qat_model_train.to('cpu')\n",
        "    qat_model_int8 = quantization.convert(qat_model_train, inplace=False)\n",
        "    save_model_weights(qat_model_int8, qat_weights_path)\n",
        "    print(\"QAT Fine-tuning & Conversion Done.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ==========================================\n",
        "# Evaluation (All Models)\n",
        "# ==========================================\n",
        "def evaluate_quantized_model(m, loader):\n",
        "    m.eval()\n",
        "    c = 0; t = 0\n",
        "    with torch.no_grad():\n",
        "        for x, y in loader:\n",
        "            x, y = x.to('cpu'), y.to('cpu')\n",
        "            out = m(x)\n",
        "            c += (out.argmax(dim=1) == y).sum().item()\n",
        "            t += x.size(0)\n",
        "    return c / t\n",
        "\n",
        "print(\"Evaluating FP32 baseline model on GPU...\")\n",
        "fp32_accuracy = evaluate(model, test_loader)\n",
        "print(f\"FP32 Model Accuracy: {fp32_accuracy * 100:.2f}%\")\n",
        "\n",
        "print(\"Evaluating PTQ INT8 model on CPU...\")\n",
        "ptq_accuracy = evaluate_quantized_model(ptq_model_int8, test_loader)\n",
        "print(f\"PTQ INT8 Model Accuracy: {ptq_accuracy * 100:.2f}%\")\n",
        "\n",
        "print(\"Evaluating QAT INT8 model on CPU...\")\n",
        "qat_accuracy = evaluate_quantized_model(qat_model_int8, test_loader)\n",
        "print(f\"QAT INT8 Model Accuracy: {qat_accuracy * 100:.2f}%\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "part6_md"
      },
      "source": [
        "### Part 6 - Create Performance & Accuracy Comparison Report\n",
        "Let's construct a performance summary comparing the three configurations."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "part6_code"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "import os\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Measure size of each state_dict on disk to show hardware/FPGA footprint reductions\n",
        "def get_model_size_kb(model):\n",
        "    torch.save(model.state_dict(), \"temp.p\")\n",
        "    size = os.path.getsize(\"temp.p\") / 1024\n",
        "    os.remove(\"temp.p\")\n",
        "    return size\n",
        "\n",
        "fp32_size = get_model_size_kb(model)\n",
        "ptq_size = get_model_size_kb(ptq_model_int8)\n",
        "qat_size = get_model_size_kb(qat_model_int8)\n",
        "\n",
        "models = ['FP32 Baseline', 'PTQ INT8', 'QAT INT8']\n",
        "accuracies = [fp32_accuracy * 100, ptq_accuracy * 100, qat_accuracy * 100]\n",
        "sizes = [fp32_size, ptq_size, qat_size]\n",
        "\n",
        "# Plotting the Comparison\n",
        "fig, ax1 = plt.subplots(figsize=(8, 5))\n",
        "\n",
        "color = 'tab:blue'\n",
        "ax1.set_xlabel('Model Version')\n",
        "ax1.set_ylabel('Accuracy (%)', color=color)\n",
        "bars = ax1.bar(models, accuracies, color=color, alpha=0.6, width=0.4, label='Accuracy (%)')\n",
        "ax1.tick_params(axis='y', labelcolor=color)\n",
        "ax1.set_ylim(0, 100)\n",
        "\n",
        "# Show the numerical values on top of the bars\n",
        "for bar in bars:\n",
        "    yval = bar.get_height()\n",
        "    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f\"{yval:.2f}%\", ha='center', va='bottom', fontweight='bold')\n",
        "\n",
        "ax2 = ax1.twinx()\n",
        "color = 'tab:red'\n",
        "ax2.set_ylabel('Model Weight Size (KB)', color=color)\n",
        "ax2.plot(models, sizes, color=color, marker='o', linewidth=2, label='Size (KB)')\n",
        "ax2.tick_params(axis='y', labelcolor=color)\n",
        "\n",
        "plt.title('ViT Model Architecture Comparison: Baseline vs PTQ vs QAT')\n",
        "fig.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Display table report\n",
        "import pandas as pd\n",
        "report_df = pd.DataFrame({\n",
        "    'Model Version': models,\n",
        "    'Accuracy (%)': [f\"{acc:.2f}%\" for acc in accuracies],\n",
        "    'State Dict Size (KB)': [f\"{sz:.2f} KB\" for sz in sizes]\n",
        "})\n",
        "display(report_df)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "part61_md"
      },
      "source": [
        "### Part 6.1 - Academic Specification & Properties Report\n",
        "This cell generates an extensive, publication-grade dataframe containing all key model properties, parameter counts, structural hyper-parameters, computational precisions, and performance deltas suitable for direct inclusion in a scientific thesis or final project report."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "part61_code"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "import os\n",
        "import torch\n",
        "import pandas as pd\n",
        "\n",
        "def get_parameter_count(model):\n",
        "    return sum(p.numel() for p in model.parameters())\n",
        "\n",
        "# Calculate structural specifications\n",
        "total_params = get_parameter_count(model)\n",
        "\n",
        "# Model weights file sizes\n",
        "fp32_sz = fp32_size\n",
        "ptq_sz = ptq_size\n",
        "qat_sz = qat_size\n",
        "\n",
        "# Accuracy metrics\n",
        "acc_fp32 = fp32_accuracy * 100\n",
        "# Fallback dummy check if evaluation failed due to bmm error earlier\n",
        "acc_ptq = (ptq_accuracy * 100) if 'ptq_accuracy' in locals() else 0.0\n",
        "acc_qat = (qat_accuracy * 100) if 'qat_accuracy' in locals() else 0.0\n",
        "\n",
        "# Construct the analytical properties\n",
        "metrics_data = {\n",
        "    \"Specification Metric / Property\": [\n",
        "        \"Dataset Target Class Count\",\n",
        "        \"Input Resolution (H x W x C)\",\n",
        "        \"ViT Patch Dimension (p x p)\",\n",
        "        \"Sequence Length (Tokens / Patches)\",\n",
        "        \"Embedding Dimension (d_model)\",\n",
        "        \"Attention Heads (h)\",\n",
        "        \"Transformer Block Depth (L)\",\n",
        "        \"MLP Expansion Dimension\",\n",
        "        \"Total Trainable Parameters\",\n",
        "        \"Arithmetic Target Precision\",\n",
        "        \"Evaluation Accuracy (CIFAR-10)\",\n",
        "        \"Accuracy Delta vs Baseline\",\n",
        "        \"Disk Storage Footprint (KB)\",\n",
        "        \"Compression Ratio\"\n",
        "    ],\n",
        "    \"FP32 Baseline Model\": [\n",
        "        f\"{num_classes}\",\n",
        "        f\"{img_size}x{img_size}x{num_channels}\",\n",
        "        f\"{patch_size}x{patch_size}\",\n",
        "        f\"{num_patches + 1} (64 patches + 1 cls)\",\n",
        "        f\"{embed_dim}\",\n",
        "        f\"{num_heads}\",\n",
        "        f\"{depth}\",\n",
        "        f\"{mlp_dim}\",\n",
        "        f\"{total_params:,}\",\n",
        "        \"FP32 (Single Precision Float)\",\n",
        "        f\"{acc_fp32:.2f}%\",\n",
        "        \"Ref (0.00%)\",\n",
        "        f\"{fp32_sz:.2f} KB\",\n",
        "        \"1.0x (Baseline)\"\n",
        "    ],\n",
        "    \"PTQ INT8 Model\": [\n",
        "        f\"{num_classes}\",\n",
        "        f\"{img_size}x{img_size}x{num_channels}\",\n",
        "        f\"{patch_size}x{patch_size}\",\n",
        "        f\"{num_patches + 1} (64 patches + 1 cls)\",\n",
        "        f\"{embed_dim}\",\n",
        "        f\"{num_heads}\",\n",
        "        f\"{depth}\",\n",
        "        f\"{mlp_dim}\",\n",
        "        f\"{total_params:,} (Quantized)\",\n",
        "        \"INT8 (8-bit Quantized Static)\",\n",
        "        f\"{acc_ptq:.2f}%\" if acc_ptq > 0 else \"Blocked on CPU CPU-bmm\",\n",
        "        f\"{acc_ptq - acc_fp32:+.2f}%\" if acc_ptq > 0 else \"N/A\",\n",
        "        f\"{ptq_sz:.2f} KB\",\n",
        "        f\"{fp32_sz / ptq_sz:.2f}x Reduction\"\n",
        "    ],\n",
        "    \"QAT INT8 Model\": [\n",
        "        f\"{num_classes}\",\n",
        "        f\"{img_size}x{img_size}x{num_channels}\",\n",
        "        f\"{patch_size}x{patch_size}\",\n",
        "        f\"{num_patches + 1} (64 patches + 1 cls)\",\n",
        "        f\"{embed_dim}\",\n",
        "        f\"{num_heads}\",\n",
        "        f\"{depth}\",\n",
        "        f\"{mlp_dim}\",\n",
        "        f\"{total_params:,} (Quantized)\",\n",
        "        \"INT8 (8-bit Quantized QAT)\",\n",
        "        f\"{acc_qat:.2f}%\" if acc_qat > 0 else \"Pending Eval\",\n",
        "        f\"{acc_qat - acc_fp32:+.2f}%\" if acc_qat > 0 else \"N/A\",\n",
        "        f\"{qat_sz:.2f} KB\",\n",
        "        f\"{fp32_sz / qat_sz:.2f}x Reduction\"\n",
        "    ]\n",
        "}\n",
        "\n",
        "# Build Pandas dataframe\n",
        "academic_report_df = pd.DataFrame(metrics_data)\n",
        "pd.set_option('display.max_colwidth', None)\n",
        "\n",
        "print(\"\\n=== DETAILED MODEL SPECIFICATION REPORT ===\\n\")\n",
        "display(academic_report_df)\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "predict_code"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "import random\n",
        "import numpy as np\n",
        "def predict_and_plot_grid(model,\n",
        "                          dataset,\n",
        "                          classes,\n",
        "                          grid_size=3):\n",
        "    model.eval()\n",
        "    total_images = min(grid_size * grid_size, len(dataset))\n",
        "    selected_indices = random.sample(range(len(dataset)), total_images)\n",
        "    correct = 0\n",
        "    fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))\n",
        "    for image_number, idx in enumerate(selected_indices):\n",
        "            i, j = divmod(image_number, grid_size)\n",
        "            img, true_label = dataset[idx]\n",
        "            input_tensor = img.unsqueeze(dim=0).to('cpu')\n",
        "            with torch.inference_mode():\n",
        "                output = model(input_tensor)\n",
        "                _, predicted = torch.max(output.data, 1)\n",
        "            mean = torch.tensor(cifar_mean).view(3, 1, 1)\n",
        "            std = torch.tensor(cifar_std).view(3, 1, 1)\n",
        "            img = (img.cpu() * std + mean).clamp(0, 1)\n",
        "            npimg = img.numpy()\n",
        "            axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)))\n",
        "            color = classes[true_label] == classes[predicted.item()]\n",
        "            correct += int(color)\n",
        "            if color:\n",
        "                c = \"g\"\n",
        "            else:\n",
        "                c = \"r\"\n",
        "            axes[i, j].set_title(f\"Truth: {classes[true_label]}\\nPredicted: {classes[predicted.item()]}\", fontsize=10, c=c)\n",
        "            axes[i, j].axis(\"off\")\n",
        "    for image_number in range(total_images, grid_size * grid_size):\n",
        "        i, j = divmod(image_number, grid_size)\n",
        "        axes[i, j].axis(\"off\")\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "    print(f\"Summary: {correct} correct out of {total_images}\")\n",
        "\n",
        "predict_and_plot_grid(qat_model_int8, test_data, classes=train_data.classes, grid_size=10)\n"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}