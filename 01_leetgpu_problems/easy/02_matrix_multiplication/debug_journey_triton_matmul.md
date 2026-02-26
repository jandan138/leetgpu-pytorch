# Triton 矩阵乘法踩坑记：一个参数引发的血案

> **摘要**：本文记录了一次在实现 Triton 矩阵乘法时遇到的诡异 Bug。从最初的“结果完全不对”，到怀疑边界、怀疑类型溢出，最后发现竟然是 Python 端参数传反了。这是一个关于“如何系统性 Debug”的真实故事。

---

## 0. 案发现场

当我们兴致勃勃地写完 `solution_triton.py` 的朴素矩阵乘法实现，并运行 `tests.py` 进行验证时，现实给了我们当头一棒：

```text
Running Matrix Multiplication Test...
1. Running PyTorch implementation (M=128, N=256, K=128)...
2. Running Triton implementation (M=128, N=256, K=128)...
❌ Correctness Check Failed!
Max difference: 82.94956970214844
```

最大误差达到了 **82.95**！对于矩阵乘法来说，这显然不是浮点误差，而是逻辑错误。

---

## 1. 第一阶段：怀疑边界问题（无用功）

**直觉反应**：Triton 是分块处理的，是不是我们在处理边缘的时候，没有正确屏蔽掉越界的元素？

在我们的代码中，Grid 是 `(M, K)`。如果 M 或 K 不是 Block Size 的整数倍（虽然这里 M, N, K 都是 128/256，应该是整除的），或者逻辑写错了，可能会导致越界读写。

**尝试修复**：我们给 `tl.store` 和 `tl.load` 加上了极其严格的边界检查。

```python
# 修改前
tl.store(offs_c, accumulator)

# 修改后
if pid_m < M and pid_k < K:
    tl.store(offs_c, accumulator)
```

以及在读取时：

```python
# 修改前
val_a = tl.load(offs_a)

# 修改后
if pid_m < M and n < N:
    val_a = tl.load(offs_a)
else:
    val_a = 0.0
```

**结果**：`Max difference` 依然是 82.95，纹丝不动。说明问题不在边界上。

---

## 2. 第二阶段：怀疑数据类型溢出（无用功）

**再次思考**：我们在计算内存地址时用了乘法：`pid_m * stride_am`。如果矩阵很大，这个乘积会不会超过 32 位整数的范围？

虽然现在的 M, N, K 只有几百，不太可能溢出，但为了排除隐患，我们决定强制使用 64 位整数计算指针。

**尝试修复**：

```python
# 修改前
offs_a = a_ptr + (pid_m * stride_am + n * stride_an)

# 修改后
offs_a = a_ptr + (pid_m * stride_am + n * stride_an).to(tl.int64)
```

**结果**：`Max difference` 还是 82.95。心态有点崩了。

---

## 3. 第三阶段：缩小问题范围（关键转折）

既然 128x256 的大矩阵看不出规律，我们决定**把矩阵缩小到 4x4**，并且打印出具体的数值，看看它到底算出了个什么鬼东西。

**修改测试代码**：
```python
M, N, K = 4, 4, 4
A = torch.randn(M, N, ...)
...
print("C_triton:\n", C_triton)
print("C_pytorch:\n", C_pytorch)
```

**输出结果**（简化版）：
```text
C_pytorch:
 [[-2.82, -2.13, ...],
  [-1.49, -1.12, ...], ...]

C_triton:
 [[-2.48, -2.12, ...],
  [-0.03, -0.44, ...], ...]
```

我们发现：
1.  结果不是全 0，说明 Kernel 确实跑起来了，也算出了数。
2.  有些值很接近，有些值差得离谱。这说明**计算公式大体是对的，但是取数取错了**。

---

## 4. 第四阶段：真相大白（参数传递错误）

我们开始死盯着 `solution_triton.py` 的代码，一行一行地看。

**Kernel 定义**：
```python
def matrix_multiplication_kernel(
    ...,
    stride_am, stride_an,  # A 的步长
    stride_bk, stride_bn,  # B 的步长 <--- 注意这里！我们定义的是 (K步长, N步长)
    ...
):
```
在 Kernel 里，我们的逻辑是：
*   `stride_bk` 应该是 **K 维度**（列）的步长。
*   `stride_bn` 应该是 **N 维度**（行）的步长。

**Solve 函数调用**：
```python
def solve(a, b, c, ...):
    stride_am, stride_an = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)  # <--- 凶手找到了！！！
```

**致命错误解析**：
对于矩阵 B (N x K)：
*   `b.stride(0)` 是 **行步长**（N 维度），在 Kernel 里应该对应 `stride_bn`。
*   `b.stride(1)` 是 **列步长**（K 维度），在 Kernel 里应该对应 `stride_bk`。

但是我们写成了：
`stride_bk = b.stride(0)` (把行步长给了列步长变量)
`stride_bn = b.stride(1)` (把列步长给了行步长变量)

**通俗解释**：
这就好比你告诉 GPU：“嘿，你要往下走一行（N变化），请跨过 `stride_bn` 个格子”。
但实际上你传给 `stride_bn` 的是列步长（通常是1）。
于是 GPU 本该跨过整整一行去拿数据，结果它只跨了一个格子。它拿到了错误的数据，自然算出了错误的结果。

**最终修复**：
```python
# 正确的顺序
stride_bn, stride_bk = b.stride(0), b.stride(1)
```
或者修改 Kernel 参数名为 `stride_bn, stride_bk` 以匹配调用顺序。我们选择了前者。

---

## 5. 总结与教训

1.  **参数对应是硬伤**：在 Python 和 Triton Kernel 之间传递参数时，没有类型检查，没有名字匹配，纯靠**位置**对应。一旦顺序搞反，编译器不会报错，只有结果会出错。
2.  **步长（Stride）最容易晕**：特别是当维度名字（M, N, K）和 Tensor 维度索引（0, 1）混合在一起时。建议明确注释 `stride(0)` 对应哪个物理维度。
3.  **Debug 心法**：
    *   不要瞎猜（比如盲目加边界检查）。
    *   **缩小问题规模**（4x4 矩阵）是看清数据的照妖镜。
    *   **打印中间结果**往往能直接暴露逻辑错误（比如取数错位）。

这次调试虽然花了不少时间，但让我们对 Triton 的内存寻址机制有了更刻骨铭心的理解。
