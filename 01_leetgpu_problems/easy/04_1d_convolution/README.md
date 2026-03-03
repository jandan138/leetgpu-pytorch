# 4. 一维卷积 (1D Convolution)

## 题目描述

给定一个输入信号数组 `x`（长度为 `N`）和一个卷积核 `w`（长度为 `K`），计算一维卷积结果 `y`，定义为：

```
y[i] = sum_{j=0}^{K-1} x[i + j] * w[j]，对 i = 0, 1, ..., N-K
```

输出长度为 `N - K + 1`，使用 **valid 模式**（不对输入补零）。

## 实现要求

- `solve` 函数签名必须保持不变。
- 最终结果必须存储在预分配的张量 `y` 中，不返回任何值。

## 示例

**输入**:
```
x = [1, 2, 3, 4, 5]   (N=5)
w = [1, 0, -1]         (K=3)
```

**输出**:
```
y = [1*1 + 2*0 + 3*(-1),
     2*1 + 3*0 + 4*(-1),
     3*1 + 4*0 + 5*(-1)]
  = [-2, -2, -2]         (长度 = N-K+1 = 3)
```

## 约束条件

- `1 <= K <= N <= 1,000,000`
- 输入和卷积核均为 32 位浮点数（`float32`）
- 性能测试基准：`N = 1,000,000`，`K = 128`

## 函数签名

```python
def solve(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, N: int, K: int) -> None:
    ...
```

## 解题思路

### 方法 1：PyTorch 实现

PyTorch 提供了 `torch.nn.functional.conv1d` 函数，可以直接计算一维卷积。
需要注意的是，`conv1d` 期望输入形状为 `(batch, channels, length)`，因此需要在调用前对 `x` 和 `w` 进行 `unsqueeze` 升维，并在写入结果前对输出进行 `squeeze` 降维。

`padding=0` 对应 valid 模式，与题目定义一致。

代码见：`solution_pytorch.py`

### 方法 2：Triton 实现

Triton 实现采用**分块并行**策略：

1. **Grid 划分**：将输出数组 `y` 按 `BLOCK_SIZE` 分块，每个 Program（线程块）负责计算一段连续的输出元素。
2. **Program 身份**：每个 Program 通过 `tl.program_id(0)` 获取自己的块编号，据此计算出负责的输出索引范围。
3. **内层循环**：对于每个输出元素 `y[i]`，在 Program 内部循环遍历卷积核的 `K` 个权重，累加 `x[i+j] * w[j]` 的乘积。
4. **边界保护**：使用掩码（mask）确保不越界访问，尤其是在最后一个不完整块中。

代码见：`solution_triton.py`

## 关键概念

- **valid 卷积**：输出长度 = `N - K + 1`，不对输入进行零填充
- **tl.program_id**：获取当前 Triton Program 在 Grid 中的编号，用于确定每个线程块负责的数据范围
- **tl.arange**：在 Triton 中生成连续整数向量，用于批量计算偏移量
- **掩码（mask）**：`tl.load` / `tl.store` 的 `mask` 参数，防止越界内存访问
- **BLOCK_SIZE**：`tl.constexpr` 类型的编译期常量，决定每个 Program 处理的元素数量

## 参考资料

- [PyTorch `torch.nn.functional.conv1d` 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton 文档](https://triton-lang.org/)
