# PyTorch 基础教程 03：使用 GPU 加速

对应代码：[`00_pytorch_basics/03_check_gpu.py`](../../00_pytorch_basics/03_check_gpu.py)

## 为什么要用 GPU？
深度学习模型通常包含大量的矩阵运算（如矩阵乘法、卷积）。相比 CPU，GPU（图形处理器）拥有数千个核心，非常适合处理这种大规模并行计算任务。使用 GPU 可以将训练速度提升数倍甚至数十倍。

## 1. 检查 CUDA 环境
CUDA 是 NVIDIA 提供的并行计算平台。PyTorch 通过 CUDA 来利用 GPU。

首先，我们需要检查当前环境是否可以使用 GPU：

```python
import torch

# 检查 CUDA 是否可用
is_cuda_available = torch.cuda.is_available()

if is_cuda_available:
    print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU 不可用，将使用 CPU")
```

## 2. 设备对象 (Device Object)
PyTorch 使用 `torch.device` 来管理计算设备。

```python
# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")
```

## 3. 将张量移动到 GPU
默认情况下，创建的张量存储在 CPU 上。我们需要手动将它们移动到 GPU。

```python
# 创建一个 CPU 张量
x = torch.tensor([1, 2, 3])

# 移动到 GPU
if torch.cuda.is_available():
    x_gpu = x.to(device)
    print(f"x_gpu 存储在: {x_gpu.device}")
    
    # 在 GPU 上进行计算
    y_gpu = x_gpu + x_gpu
    print(f"计算结果也在 GPU 上: {y_gpu.device}")

    # 移回 CPU（如果需要打印或转为 NumPy）
    x_cpu = x_gpu.cpu()
```

## 常见问题
1. **RuntimeError: Expected all tensors to be on the same device**
   这是新手最常遇到的错误。它意味着你试图让一个 CPU 张量和一个 GPU 张量进行运算。解决方法是确保所有参与运算的张量都在同一个设备上（通常都 `.to(device)`）。

2. **如何安装 GPU 版本的 PyTorch？**

   **推荐（uv）**：项目已在 `pyproject.toml` 中配置好 PyTorch CUDA 源，直接运行：
   ```bash
   uv sync
   ```

   **备用（pip）**：需要手动指定安装源：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   > 也可访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 根据你的 CUDA 版本选择对应命令。

## 总结
现在你已经学会了：
1. 创建和操作张量
2. 计算梯度
3. 使用 GPU 加速

这是 PyTorch 的三大基石。接下来，你可以开始尝试构建简单的神经网络了！
