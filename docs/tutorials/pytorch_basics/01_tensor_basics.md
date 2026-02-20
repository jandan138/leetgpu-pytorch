# PyTorch 基础教程 01：张量 (Tensor)

对应代码：[`00_pytorch_basics/01_tensors.py`](../../00_pytorch_basics/01_tensors.py)

## 什么是张量？
张量（Tensor）是 PyTorch 中最核心的数据结构。你可以把它简单理解为一个**多维数组**。
- 0维张量：标量（Scalar），如 `1.0`
- 1维张量：向量（Vector），如 `[1, 2, 3]`
- 2维张量：矩阵（Matrix），如 `[[1, 2], [3, 4]]`
- N维张量...

它和 NumPy 的 `ndarray` 非常相似，但有一个关键区别：**张量可以在 GPU 上运行，从而加速计算**。

## 1. 创建张量
我们可以通过多种方式创建张量：

### 1.1 从列表创建
```python
import torch

data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)
print(tensor)
```

### 1.2 创建特定值的张量
- **全 0 张量**：`torch.zeros(rows, cols)`
- **全 1 张量**：`torch.ones(rows, cols)`
- **随机张量**：`torch.rand(rows, cols)` (生成 0 到 1 之间的随机数)

## 2. 张量的属性
每个张量都有三个重要属性：
1. **形状 (Shape)**: `tensor.shape`，描述张量的维度大小。
2. **数据类型 (Dtype)**: `tensor.dtype`，如 `torch.float32`, `torch.int64`。
3. **存储设备 (Device)**: `tensor.device`，数据是在 CPU 还是 GPU 上。

## 3. 基本运算
PyTorch 支持丰富的数学运算，操作方式与 NumPy 类似：

### 加法
```python
t1 = torch.tensor([1, 2])
t2 = torch.tensor([3, 4])
result = t1 + t2  # 结果: [4, 6]
```

### 逐元素乘法
```python
result = t1 * t2  # 结果: [1*3, 2*4] -> [3, 8]
```

### 矩阵乘法
使用 `@` 符号或 `torch.matmul`：
```python
# 假设 t1 是 2x2, t2 是 2x2
result = t1 @ t2
```

## 下一步
掌握了张量之后，我们就可以开始学习如何让 PyTorch 自动帮我们计算梯度了，请阅读 [02_autograd_mechanics.md](./02_autograd_mechanics.md)。
