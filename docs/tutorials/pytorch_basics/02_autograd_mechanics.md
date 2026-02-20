# PyTorch 基础教程 02：自动微分 (Autograd)

对应代码：[`00_pytorch_basics/02_autograd.py`](../../00_pytorch_basics/02_autograd.py)

## 什么是自动微分？
在深度学习中，我们需要通过**反向传播**（Backpropagation）来计算损失函数相对于模型参数的梯度。PyTorch 提供了一个强大的自动微分引擎 `torch.autograd`，它可以自动为你计算这些梯度。

## 核心概念

### 1. `requires_grad=True`
当你创建一个张量时，如果需要对其求导，必须设置 `requires_grad=True`。这告诉 PyTorch：“请追踪在这个张量上发生的所有运算”。

```python
x = torch.tensor([2.0], requires_grad=True)
```

### 2. 计算图 (Computation Graph)
PyTorch 会在后台构建一个动态计算图。例如，如果你定义了 $y = x^2 + 2x + 1$：
1. `x` 是叶子节点。
2. `x^2`、`2x` 是中间节点。
3. `y` 是根节点。

### 3. `.backward()`
当你调用 `y.backward()` 时，PyTorch 会从 `y` 开始，根据链式法则自动计算所有 `requires_grad=True` 的张量的梯度。

### 4. `.grad`
计算完成后，梯度会存储在张量的 `.grad` 属性中。

## 示例解析
让我们看一个具体的例子：
$$y = x^2 + 2x + 1$$
我们要计算当 $x=2$ 时，$y$ 对 $x$ 的导数 $\frac{dy}{dx}$。

数学推导：
$$ \frac{dy}{dx} = 2x + 2 $$
当 $x=2$ 时：
$$ \frac{dy}{dx} = 2(2) + 2 = 6 $$

代码实现：
```python
import torch

# 1. 定义变量 x，启用梯度追踪
x = torch.tensor([2.0], requires_grad=True)

# 2. 定义函数 y
y = x**2 + 2*x + 1

# 3. 反向传播
y.backward()

# 4. 查看梯度
print(f"dy/dx at x=2 is: {x.grad}")  # 输出应为 6.0
```

## 注意事项
- 只有浮点类型的张量才能计算梯度（如 `float32`, `float64`）。
- 每次调用 `.backward()` 时，梯度会**累加**到 `.grad` 中。所以在训练循环中，每次更新参数前都需要手动清零梯度 (`optimizer.zero_grad()`)。

## 下一步
了解了梯度计算，我们来看看如何利用 GPU 加速计算，请阅读 [03_gpu_acceleration.md](./03_gpu_acceleration.md)。
