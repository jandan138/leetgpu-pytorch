# LeetGPU-PyTorch

这是一个用于学习 PyTorch 和解决 GPU 加速计算问题的项目。

## 目录结构

- **00_pytorch_basics**: 包含 PyTorch 基础知识的学习代码和笔记。
- **01_leetgpu_problems**: 包含不同难度的 GPU 计算问题。
  - **easy**: 简单问题
  - **medium**: 中等问题
  - **hard**: 困难问题
- **data**: 存放数据集和临时数据（已在 .gitignore 中忽略）。
- **utils**: 包含通用的工具函数和辅助代码。

## 如何开始

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖和虚拟环境，相比传统 `pip + venv` 速度更快、配置更清晰。

1. **克隆仓库**

   ```bash
   git clone <repository_url>
   cd leetgpu-pytorch
   ```

2. **安装 uv**

   ```powershell
   # Windows（PowerShell）
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   ```bash
   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   > 也可通过 pip 安装：`pip install uv`

3. **创建虚拟环境并安装依赖（一步完成）**

   ```bash
   uv sync
   ```

   > uv 会自动读取 `pyproject.toml`，创建 `.venv` 目录，并从 PyTorch 官方 CUDA 源拉取 GPU 版本。
   > 首次运行需下载约 2.5GB，请确保网络畅通。

4. **激活虚拟环境**

   ```powershell
   # Windows
   .\.venv\Scripts\activate
   ```

   ```bash
   # Linux/macOS
   source .venv/bin/activate
   ```

5. **运行代码**

   ```bash
   # 验证 GPU 是否可用
   python 00_pytorch_basics/03_check_gpu.py

   # 启动 Jupyter Notebook 进行交互式学习
   jupyter notebook
   ```

<details>
<summary>不想用 uv？备用方案（传统 pip）</summary>

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# GPU 版 PyTorch 需单独指定源
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install jupyter matplotlib numpy
```

</details>

## 学习路线

建议从 `00_pytorch_basics` 开始，掌握 PyTorch 的基本张量操作和自动求导机制，然后逐步尝试 `01_leetgpu_problems` 中的问题。

## 文档与教程

我们提供了详细的入门教程，帮助您理解代码背后的概念：

1. [PyTorch 基础 01：张量 (Tensor)](docs/tutorials/01_tensor_basics.md) - 对应 `00_pytorch_basics/01_tensors.py`
2. [PyTorch 基础 02：自动微分 (Autograd)](docs/tutorials/02_autograd_mechanics.md) - 对应 `00_pytorch_basics/02_autograd.py`
3. [PyTorch 基础 03：使用 GPU 加速](docs/tutorials/03_gpu_acceleration.md) - 对应 `00_pytorch_basics/03_check_gpu.py`
4. [Taichi vs PyTorch](docs/tutorials/concept_taichi_vs_pytorch.md) - 了解两者区别
5. [OpenAI Triton](docs/tutorials/concept_triton_overview.md) - 了解 GPU 编程新宠
