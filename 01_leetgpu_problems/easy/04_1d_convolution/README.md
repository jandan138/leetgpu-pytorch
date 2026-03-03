# 04. 一维卷积（1D Convolution）

## 目录

- [题目描述](#题目描述)
- [示例](#示例)
- [约束条件](#约束条件)
- [函数签名](#函数签名)
- [解题思路总览](#解题思路总览)
- [PyTorch 实现解析](#pytorch-实现解析)
- [Triton 实现解析](#triton-实现解析)
- [性能分析](#性能分析)
- [运行与测试](#运行与测试)
- [深入理解](#深入理解)

---

## 题目描述

给定一个输入信号数组 `x`（长度为 `N`）和一个卷积核 `w`（长度为 `K`），计算一维 **valid 模式**卷积的输出 `y`。

数学定义：

```text
y[i] = sum_{j=0}^{K-1} x[i + j] * w[j]

其中 i = 0, 1, ..., N - K
```

**Valid 模式**：不对输入信号进行任何零填充（zero-padding），输出长度严格等于 `N - K + 1`。这与信号处理中"滑动点积"的定义完全一致：将卷积核 `w` 沿信号 `x` 从左到右滑动，每个位置做一次点积，得到一个输出值。

> 注意：本题公式与信号处理教材中的"相关（cross-correlation）"完全相同（无翻转）。PyTorch 的 `F.conv1d` 同样不翻转卷积核，因此可以直接对应使用。

---

## 示例

**输入**：

```text
x = [1.0, 2.0, 3.0, 4.0, 5.0]   (N = 5)
w = [1.0, 0.0, -1.0]              (K = 3)
```

**计算过程**（滑动窗口）：

```text
i = 0: y[0] = x[0]*w[0] + x[1]*w[1] + x[2]*w[2]
             = 1.0*1.0 + 2.0*0.0 + 3.0*(-1.0)
             = 1.0 + 0.0 - 3.0 = -2.0

i = 1: y[1] = x[1]*w[0] + x[2]*w[1] + x[3]*w[2]
             = 2.0*1.0 + 3.0*0.0 + 4.0*(-1.0)
             = 2.0 + 0.0 - 4.0 = -2.0

i = 2: y[2] = x[2]*w[0] + x[3]*w[1] + x[4]*w[2]
             = 3.0*1.0 + 4.0*0.0 + 5.0*(-1.0)
             = 3.0 + 0.0 - 5.0 = -2.0
```

**输出**：

```text
y = [-2.0, -2.0, -2.0]   (长度 = N - K + 1 = 3)
```

---

## 约束条件

| 参数 | 范围 |
| :--- | :--- |
| 输入长度 N | `1 <= N <= 1,000,000` |
| 核长度 K | `1 <= K <= N` |
| 数据类型 | `float32` |
| 存储设备 | CUDA GPU |
| 性能基准 | `N = 25,000,000`，`K = 64` |

---

## 函数签名

```python
def solve(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, N: int, K: int) -> None:
    ...
```

**参数说明**：

| 参数 | 形状 | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| `x` | `(N,)` | `float32` | 输入信号，位于 CUDA 设备 |
| `w` | `(K,)` | `float32` | 卷积核，位于 CUDA 设备 |
| `y` | `(N-K+1,)` | `float32` | **预分配的**输出张量，函数直接写入，不返回值 |
| `N` | — | `int` | 输入信号长度 |
| `K` | — | `int` | 卷积核长度 |

---

## 解题思路总览

一维卷积在本质上是一个**滑动窗口点积**问题。每个输出元素 `y[i]` 独立地依赖于 `x[i:i+K]` 这段连续区间，与其他输出元素没有数据依赖，因此天然适合 GPU 并行化。

| 实现方法 | 核心思路 | 适合场景 |
| :--- | :--- | :--- |
| `F.conv1d`（推荐） | 调用 cuDNN 高度优化的实现 | 生产环境，追求最佳性能 |
| `unfold + @` | 将问题转换为矩阵-向量乘法 | 理解卷积本质，中间规模数据 |
| Triton kernel | 手写 GPU kernel，控制并行粒度 | 学习 GPU 编程，定制化场景 |

---

## PyTorch 实现解析

完整代码见 [`solution_pytorch.py`](solution_pytorch.py)。

### 方法一：`F.conv1d`（最推荐）

```python
import torch.nn.functional as F

def solve(x, w, y, N, K):
    x_3d = x.view(1, 1, N)    # (batch=1, in_channels=1, length=N)
    w_3d = w.view(1, 1, K)    # (out_channels=1, in_channels=1, kernel_size=K)
    y.copy_(F.conv1d(x_3d, w_3d, padding=0).view(-1))
```

**关键点解析**：

1. **形状变换**：`F.conv1d` 期望三维输入，形状为 `(batch, channels, length)`。一维信号 `x` 和核 `w` 都需要通过 `view` 升维，计算完毕后再用 `view(-1)` 降回一维。

2. **`padding=0`**：对应 valid 模式，即不补零，输出长度 = `N - K + 1`。

3. **不翻转卷积核**：PyTorch 的 `conv1d` 实际上实现的是**互相关（cross-correlation）**，不翻转卷积核 `w`，与本题公式 `y[i] = Σ x[i+j]*w[j]` 完全一致。

4. **写回预分配张量**：使用 `y.copy_(...)` 将结果就地写入预分配的输出张量 `y`，符合 `solve` 函数的约定（不返回值）。

5. **底层实现**：`F.conv1d` 在 GPU 上由 **cuDNN** 加速，内部会根据输入规模自动选择最优算法（直接卷积、FFT 卷积等）。

### 方法二：`unfold + @`（用于理解本质）

```python
def solve(x, w, y, N, K):
    x_windows = x.unfold(0, K, 1)  # (N-K+1, K)：每行是一个长度为 K 的滑动窗口
    y.copy_(x_windows @ w)          # 矩阵-向量乘法，等价于逐窗口点积
```

`unfold(dimension=0, size=K, step=1)` 将一维张量展开为滑动窗口矩阵，形状为 `(N-K+1, K)`，第 `i` 行即 `x[i:i+K]`。矩阵与向量 `w` 的乘法恰好是所有窗口与卷积核的点积，计算结果与公式完全等价。

> 注意：`unfold` 会产生一个 `(N-K+1, K)` 的中间张量，当 `N` 和 `K` 都很大时内存开销可观。生产环境推荐使用 `F.conv1d`。

---

## Triton 实现解析

完整代码见 [`solution_triton.py`](solution_triton.py)。

### 整体设计思路

Triton 实现的核心问题是：**如何把输出数组的 `N-K+1` 个元素分配给 GPU 上的多个 Program 并行计算？**

答案是**一维分块（1D tiling）**：将输出数组按 `BLOCK_SIZE` 均匀切分，每个 Program 负责计算一段连续的输出元素。

```text
输出数组 y（长度 = output_len = N-K+1）：

┌──────────┬──────────┬──────────┬────┐
│  pid=0   │  pid=1   │  pid=2   │ …  │
│ y[0..B-1]│y[B..2B-1]│y[2B..3B-1│    │
└──────────┴──────────┴──────────┴────┘
   BLOCK_SIZE 个元素    (最后一块可能不满)

Grid 大小 = ceil(output_len / BLOCK_SIZE)
```

### 内核函数解析

```python
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, y_ptr,    # 三个张量的指针
    N, K, output_len,        # 运行时标量参数
    BLOCK_SIZE: tl.constexpr # 编译期常量：每个 Program 处理的元素数
):
    # ── 第一步：确定"我"负责哪些输出元素 ────────────────────────
    pid = tl.program_id(axis=0)           # 当前 Program 的编号
    block_start = pid * BLOCK_SIZE        # 本块在 y 中的起始索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # offsets 是一个长度为 BLOCK_SIZE 的向量：
    # [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]

    # ── 第二步：边界掩码（保护最后一个不完整块）──────────────────
    mask = offsets < output_len
    # 对于最后一个块，offsets 中超出 output_len 的位置
    # 将被 mask 过滤，不做加载也不做存储

    # ── 第三步：累加卷积（内核维度上的串行循环）──────────────────
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for j in tl.range(0, K):
        # 加载 x[offsets + j]：本块所有输出位置对应的第 j 个输入元素
        # 形状：(BLOCK_SIZE,)，一次向量化加载
        x_vals = tl.load(x_ptr + offsets + j, mask=mask, other=0.0)
        # 加载标量权重 w[j]
        w_j = tl.load(w_ptr + j)
        # 向量化乘加：acc[k] += x[offsets[k] + j] * w[j]
        acc += x_vals * w_j

    # ── 第四步：将结果写回 y ─────────────────────────────────────
    tl.store(y_ptr + offsets, acc, mask=mask)
```

### Grid 和 Program 的数量

```python
def solve(x, w, y, N, K):
    output_len = N - K + 1
    BLOCK_SIZE = 1024

    # Grid：一维，共 ceil(output_len / BLOCK_SIZE) 个 Program
    grid = (triton.cdiv(output_len, BLOCK_SIZE),)

    conv1d_kernel[grid](
        x, w, y,
        N, K, output_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

以 `N=1,000,000`、`K=128` 为例：
- `output_len = 999,873`
- `BLOCK_SIZE = 1024`
- Grid 大小 = `ceil(999,873 / 1024) = 977` 个 Program

977 个 Program 在 GPU 上并发执行，每个 Program 独立计算 1024 个输出元素，各自运行一个长度为 128 的 `for j` 循环。

### 关键 Triton 语法说明

| 语法 | 含义 |
| :--- | :--- |
| `tl.program_id(axis=0)` | 获取当前 Program 在 Grid 第 0 维的编号（0-indexed） |
| `tl.arange(0, BLOCK_SIZE)` | 生成 `[0, 1, ..., BLOCK_SIZE-1]` 的整数向量（`BLOCK_SIZE` 必须是 `constexpr`） |
| `tl.zeros((BLOCK_SIZE,), dtype=tl.float32)` | 创建全零向量作为累加器 |
| `tl.load(ptr + offsets, mask=mask, other=0.0)` | 向量化加载，`mask` 为 `False` 的位置用 `other` 填充 |
| `tl.store(ptr + offsets, val, mask=mask)` | 向量化存储，`mask` 为 `False` 的位置不写入 |
| `tl.range(0, K)` | 在 Triton kernel 内部做循环（编译器可选择展开） |
| `BLOCK_SIZE: tl.constexpr` | 编译期常量，决定向量寄存器大小，不同值触发不同编译版本 |

---

## 性能分析

### 计算与访存特征

一维卷积是一个**访存密集型（memory-bound）**操作：

- **计算量**：`output_len × K` 次乘加，约 `(N-K+1) × K` 次浮点运算
- **访存量**（理论下界）：读取 `x`（`N` 个 float）+ 读取 `w`（`K` 个 float）+ 写入 `y`（`N-K+1` 个 float）

当 `K` 较小（如 64、128）时，算术强度（FLOP/Byte）偏低，瓶颈在显存带宽而非算力。

### Triton 实现的访存行为

本实现中，每个 Program 处理 `BLOCK_SIZE` 个输出元素，内部循环 `K` 次：

```text
每次循环：加载 BLOCK_SIZE 个 x 元素（连续地址，合并访问）
          加载 1 个 w 元素（所有 Program 共享，会被 L2 缓存命中）
总访存：output_len × K × 4 字节（x 的加载）
      + K × 4 字节（w 的加载，被缓存复用）
      + output_len × 4 字节（y 的写入）
```

由于对 `x` 的访问模式在每次循环中偏移了 `j`（跨相邻窗口），不同 Program 之间的访问地址连续，**满足合并访问（coalesced access）条件**，可以充分利用显存带宽。

### 与 `F.conv1d`（cuDNN）的性能差距

在大多数情况下，`F.conv1d` 的性能优于本 Triton 实现，主要原因如下：

| 差距来源 | 说明 |
| :--- | :--- |
| **cuDNN 算法选择** | cuDNN 会根据 `N`、`K` 自动选择直接卷积、Winograd 算法或 FFT 卷积，本 Triton 实现固定使用直接卷积 |
| **共享内存（Shared Memory）** | cuDNN 实现利用 SM 的 Shared Memory 缓存滑动窗口数据，减少重复的全局内存访问；本实现每次从全局内存加载 |
| **向量化加载宽度** | cuDNN 使用 128-bit（float4）向量化加载；本实现视编译器优化程度而定 |
| **内核参数自动调优** | cuDNN 针对各种 GPU 型号预编译了最优参数；本 Triton 实现 `BLOCK_SIZE=1024` 固定，未自动调优 |

> 进阶优化思路：可以考虑将 `w` 预加载进 Triton 的 Shared Memory（通过 `tl.make_block_ptr` 和 `tl.prefetch`），或使用 `@triton.autotune` 搜索最优 `BLOCK_SIZE`，缩小与 cuDNN 的性能差距。

### 性能基准参考

在 `N = 25,000,000`、`K = 64` 的场景下运行 `tests.py`，可以观察两种实现的耗时对比：

```bash
cd 01_leetgpu_problems/easy/04_1d_convolution
python tests.py
```

---

## 运行与测试

### 运行测试

```bash
cd 01_leetgpu_problems/easy/04_1d_convolution
python tests.py
```

测试文件会执行：
1. **正确性验证**：三组规模（`N=128`、`N=1024`、`N=98432`）下，PyTorch 和 Triton 实现与参考结果（`F.conv1d`）的误差比较（`torch.allclose`，`atol=rtol=1e-2`）
2. **性能基准**：`N=25,000,000`、`K=64` 下，各实现 10 次运行的平均耗时（使用 `torch.cuda.synchronize()` 确保计时准确）

> 注意：Triton 第一次运行会触发 JIT 编译（通常需要 0.5~2 秒），测试文件已包含 warmup 轮次以排除编译开销的干扰。

### Triton 可用性

测试文件使用 `try/except ImportError` 优雅处理 Triton 未安装的情况：

```python
try:
    import solution_triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```

若 Triton 未安装，PyTorch 测试仍会正常运行，Triton 测试会被跳过并打印提示。

> 注意：Triton 目前在 Windows 上没有官方 wheel，建议在 Linux 或 WSL2 环境下使用。

---

## 深入理解

本题涉及以下进阶概念，详见对应深度文档：

- [`deep_dive_triton_conv1d.md`](deep_dive_triton_conv1d.md) — 深入讲解 1D 卷积的 GPU 并行化思路、Triton SPMD 模型、内存合并访问、边界 masking 原理，以及与 cuDNN 的性能差距分析

---

## 参考资料

- [PyTorch `torch.nn.functional.conv1d` 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html)
- [Triton 官方入门教程：向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton 语言参考](https://triton-lang.org/main/python-api/triton.language.html)
- [NVIDIA cuDNN 开发者文档](https://docs.nvidia.com/deeplearning/cudnn/developer/index.html)
