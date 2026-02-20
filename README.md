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

1. **克隆仓库**

   ```bash
   git clone <repository_url>
   cd leetgpu-pytorch
   ```

2. **创建虚拟环境 (推荐)**

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **运行代码**

   你可以直接运行 Python 脚本或启动 Jupyter Notebook 进行交互式学习。

   ```bash
   jupyter notebook
   ```

## 学习路线

建议从 `00_pytorch_basics` 开始，掌握 PyTorch 的基本张量操作和自动求导机制，然后逐步尝试 `01_leetgpu_problems` 中的问题。

## 文档与教程

我们提供了详细的入门教程，帮助您理解代码背后的概念：

1. [PyTorch 基础 01：张量 (Tensor)](docs/tutorials/01_tensor_basics.md) - 对应 `00_pytorch_basics/01_tensors.py`
2. [PyTorch 基础 02：自动微分 (Autograd)](docs/tutorials/02_autograd_mechanics.md) - 对应 `00_pytorch_basics/02_autograd.py`
3. [PyTorch 基础 03：使用 GPU 加速](docs/tutorials/03_gpu_acceleration.md) - 对应 `00_pytorch_basics/03_check_gpu.py`
