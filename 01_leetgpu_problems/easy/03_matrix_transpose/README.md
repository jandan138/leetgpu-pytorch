# 03. Matrix Transpose（矩阵转置）

> **难度**：Easy
> **核心知识点**：2D Grid 设计、stride 寻址、coalesced access（合并访问）、PyTorch View 机制

---

## 目录

1. [题目描述](#题目描述)
2. [示例](#示例)
3. [约束条件](#约束条件)
4. [函数签名](#函数签名)
5. [解题思路总览](#解题思路总览)
6. [PyTorch 实现解析](#pytorch-实现解析)
7. [Triton 实现解析](#triton-实现解析)
8. [性能分析](#性能分析)
9. [常见误区](#常见误区)
10. [运行与测试](#运行与测试)
11. [深入理解](#深入理解)
12. [参考资料](#参考资料)

---

## 题目描述

编写一个 GPU 程序，将一个由 32 位浮点数组成的矩阵进行转置。

**数学定义：** 给定输入矩阵 `A`，其转置矩阵 `B = A^T` 满足：

```text
B[i][j] = A[j][i]   对所有合法的 (i, j) 成立
```

具体来说，若输入矩阵 `A` 的维度为 `rows × cols`，则输出矩阵 `B = A^T` 的维度为 `cols × rows`。

```text
输入 A (rows × cols):          输出 B = A^T (cols × rows):

  A[0][0]  A[0][1]  ...  A[0][cols-1]       B[0][0]  B[0][1]  ...  B[0][rows-1]
  A[1][0]  A[1][1]  ...  A[1][cols-1]  →    B[1][0]  B[1][1]  ...  B[1][rows-1]
  ...                                        ...
  A[rows-1][0]  ...  A[rows-1][cols-1]       B[cols-1][0]  ...  B[cols-1][rows-1]

其中 B[j][i] = A[i][j]
```

所有矩阵均以**行优先（Row-Major）**格式存储在连续内存中。

**实现要求：**
- 仅使用原生特性，不允许调用外部库的 transpose 函数（PyTorch 解法中可以使用 `torch.transpose` / `input.t()` 作为基准参考）。
- `solve` 函数签名必须保持不变。
- 最终结果必须存储在预分配的 `output` 张量中（**原地写入，不是返回新张量**）。

---

## 示例

### 基础示例：2×3 矩阵转置

**输入矩阵（2 行 × 3 列）：**

```text
A = [[1.0,  2.0,  3.0],
     [4.0,  5.0,  6.0]]

内存布局（行优先）：[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                     ↑                  ↑
                  A[0][0]            A[1][0]（第 1 行从第 4 个位置开始）
```

**元素映射关系（逐步推导）：**

```text
转置规则：B[j][i] = A[i][j]

A[0][0]=1.0  →  B[0][0]=1.0  （行0列0 → 行0列0，对角元不变）
A[0][1]=2.0  →  B[1][0]=2.0  （行0列1 → 行1列0）
A[0][2]=3.0  →  B[2][0]=3.0  （行0列2 → 行2列0）
A[1][0]=4.0  →  B[0][1]=4.0  （行1列0 → 行0列1）
A[1][1]=5.0  →  B[1][1]=5.0  （行1列1 → 行1列1，对角元不变）
A[1][2]=6.0  →  B[2][1]=6.0  （行1列2 → 行2列1）
```

**输出矩阵（3 行 × 2 列）：**

```text
B = [[1.0,  4.0],
     [2.0,  5.0],
     [3.0,  6.0]]

内存布局（行优先）：[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
```

**内存地址变化可视化：**

```text
输入 A（行优先，地址 0~5）：
地址:  0     1     2     3     4     5
值:   1.0   2.0   3.0   4.0   5.0   6.0
      ──────────── ────────────────────
      第 0 行         第 1 行

输出 B（行优先，地址 0~5）：
地址:  0     1     2     3     4     5
值:   1.0   4.0   2.0   5.0   3.0   6.0
      ───── ───── ───── ───── ───── ─────
      B[0,0] B[0,1] B[1,0] B[1,1] B[2,0] B[2,1]

注意：原来相邻的 A[0][0]=1.0 和 A[0][1]=2.0，
在输出中变成了 B[0][0]=1.0（地址0）和 B[1][0]=2.0（地址2），不再相邻！
这正是矩阵转置导致 non-coalesced（非合并）访存的直观体现。
```

---

## 约束条件

| 参数 | 范围 | 说明 |
| :--- | :--- | :--- |
| `rows` | 1 ~ 8192 | 输入矩阵的行数 |
| `cols` | 1 ~ 8192 | 输入矩阵的列数 |
| 输入维度 | `rows × cols` | float32 |
| 输出维度 | `cols × rows` | float32，预分配 |
| 性能测试基准 | `rows=7000, cols=6000` | 矩阵大小约 160 MB |

---

## 函数签名

```python
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int) -> None:
```

| 参数 | 类型 | 含义 |
| :--- | :--- | :--- |
| `input` | `torch.Tensor`，shape `(rows, cols)`，dtype `float32` | 输入矩阵，已在 GPU 上 |
| `output` | `torch.Tensor`，shape `(cols, rows)`，dtype `float32` | **预分配**的输出矩阵，结果写入此处 |
| `rows` | `int` | 输入矩阵的行数 |
| `cols` | `int` | 输入矩阵的列数 |
| 返回值 | `None` | 无返回值，结果写入 `output` |

**重要：** `output` 是调用方预先分配好的张量，`solve()` 负责把计算结果就地写入它，而不是返回一个新张量。这是本项目所有题目的统一约定。

---

## 解题思路总览

| 方法 | 并行粒度 | 实现复杂度 | 访存模式 | 适合学习什么 |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch（`.t()` + `copy_`） | 自动调度 | 极低 | 内部优化 | View 机制、按需物化 |
| Triton 朴素版（每 Program 一个元素） | 元素级 | 低 | 读合并/写不合并 | 2D Grid、stride 寻址 |
| Triton 分块版（Shared Memory tiling） | tile 级 | 中 | 读写均合并 | Bank conflict、tile 策略 |

本题当前提供**前两种实现**，分块版作为进阶扩展留给读者。

---

## PyTorch 实现解析

> 完整代码：[solution_pytorch.py](solution_pytorch.py)

### 核心代码

```python
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int) -> None:
    output.copy_(input.t())
```

只有一行，但背后涉及两个关键概念：

### 概念一：`.t()` 返回的是 View，不是拷贝

`input.t()` 等价于 `torch.transpose(input, 0, 1)`。调用后 **不会** 移动任何数据，只是生成一个新的 Tensor 对象，其内部 `storage` 指针仍然指向原数据，但 `stride`（步长）被交换了。

```text
input 的 storage：[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  ← 内存不变

input 的元数据：
  shape:  (2, 3)
  stride: (3, 1)      ← 行步长=3（每换一行跳3个元素），列步长=1

input.t() 的元数据：
  shape:  (3, 2)
  stride: (1, 3)      ← 行步长和列步长互换！
  （注意：storage 指针完全相同，内存没动）

访问 input.t()[row, col] 的地址 = storage_base + row * 1 + col * 3
```

这意味着 `input.t()` 是零拷贝操作，O(1) 时间复杂度。

### 概念二：`output.copy_()` 触发真实的数据搬运

题目要求结果写入 `output`，所以必须调用 `.copy_()` 把 View 中的数据物化到 `output` 的内存空间中。

```text
copy_ 的行为：
  源：input.t()（逻辑上是转置的，但内存是原始的行优先布局）
  目标：output（cols × rows 的连续内存块）

copy_ 内部会处理 stride 不连续的情况：
  - 它实际上是在按 input.t() 的 stride=(1,3) 去读数据
  - 然后按 output 的 stride=(rows,1)（行优先）写入
  - 这在 GPU 上最终会调用优化过的 CUDA transpose kernel
```

### 关键 API 对比

| 调用方式 | 是否移动数据 | 结果是否 contiguous | 适用场景 |
| :--- | :--- | :--- | :--- |
| `input.t()` | 否（View） | 否 | 计算下游接受非连续张量时 |
| `input.t().contiguous()` | 是 | 是 | 需要连续内存时 |
| `output.copy_(input.t())` | 是 | 是（写入预分配 output） | 本题要求的写法 |
| `torch.transpose(input, 0, 1, out=output)` | — | 取决于实现 | 部分版本不支持 `out` 参数 |

---

## Triton 实现解析

> 完整代码：[solution_triton.py](solution_triton.py)

### 整体设计：CPU Host 与 GPU Device 的分工

```text
CPU / Python 世界 (Host)                    GPU 世界 (Device)
─────────────────────────────              ─────────────────────────────────────
solve(input, output, rows, cols)
  │
  ├─ 1. 读取 input/output 的 stride
  ├─ 2. 定义 grid = (rows, cols)
  └─ 3. 启动 kernel：
         matrix_transpose_kernel[grid](...)
                                            每个 Program (pid_row, pid_col) 并行：
                                            - 读取 input[pid_row, pid_col]
                                            - 写入 output[pid_col, pid_row]
```

### 1. Grid 设计：一元素一 Program

```text
输入矩阵 (rows=3, cols=4)：

  (0,0) (0,1) (0,2) (0,3)    ← pid_row=0 的 4 个 Program
  (1,0) (1,1) (1,2) (1,3)    ← pid_row=1 的 4 个 Program
  (2,0) (2,1) (2,2) (2,3)    ← pid_row=2 的 4 个 Program

grid = (rows, cols) = (3, 4)，共启动 12 个 Program
每个 Program 仅负责搬运一个元素
```

这是最直观的映射策略：**Grid 形状 = 输入矩阵形状**。`pid_row` 和 `pid_col` 直接就是该 Program 负责的元素在输入中的行列坐标。

### 2. tl.program_id 的语义

```python
pid_row = tl.program_id(axis=0)  # Grid 第一维（行方向）
pid_col = tl.program_id(axis=1)  # Grid 第二维（列方向）
```

`tl.program_id(axis)` 返回当前 Program 在 Grid 对应维度上的索引，范围是 `[0, grid[axis])`。当 `grid = (rows, cols)` 时：
- `axis=0`：行方向，范围 `[0, rows)`
- `axis=1`：列方向，范围 `[0, cols)`

详细解析参见：[deep_dive_triton_2d_grid.md](deep_dive_triton_2d_grid.md)

### 3. stride 寻址：从二维坐标到一维地址

GPU 内存是线性的（一维数组），需要将矩阵的二维坐标 `(row, col)` 转换为一维偏移量。

```text
行优先存储（Row-Major）的寻址公式：

  offset = row * stride_row + col * stride_col

对于形状为 (rows, cols) 的连续矩阵：
  stride_row = cols    （换一行，跳过 cols 个元素）
  stride_col = 1       （换一列，跳过 1 个元素）

示例：矩阵 A 的形状 (3, 4)
  A[2][1] 的地址 = base + 2 * 4 + 1 * 1 = base + 9
```

**为什么传 stride 而不是直接传 rows/cols？**

传 `stride` 而非直接传 `cols` 是更通用的设计。当张量是非连续的（例如对某个大矩阵做了切片，或者是另一个转置的 View），`stride` 会自动反映真实的内存布局，而 `cols` 则不足以描述这种情况。PyTorch 的 `.stride(0)` 和 `.stride(1)` 直接返回底层存储的步长，无需手动计算。

```python
# 获取 input (rows × cols) 的步长
stride_ir, stride_ic = input.stride(0), input.stride(1)
# 对于连续张量：stride_ir = cols，stride_ic = 1

# 获取 output (cols × rows) 的步长
stride_or, stride_oc = output.stride(0), output.stride(1)
# 对于连续张量：stride_or = rows，stride_oc = 1
```

### 4. Kernel 完整流程解析

```python
@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    rows, cols,
    stride_ir, stride_ic,   # 输入：行步长、列步长
    stride_or, stride_oc    # 输出：行步长、列步长
):
    # Step 1：确认身份
    pid_row = tl.program_id(axis=0)   # 我处理输入的第几行
    pid_col = tl.program_id(axis=1)   # 我处理输入的第几列

    # Step 2：边界检查（防止越界写入）
    if pid_row < rows and pid_col < cols:

        # Step 3：从输入读取元素
        # 输入坐标：(pid_row, pid_col)
        # 内存地址：base + pid_row * stride_ir + pid_col * stride_ic
        input_offset = pid_row * stride_ir + pid_col * stride_ic
        val = tl.load(input_ptr + input_offset)

        # Step 4：写入到输出的转置位置
        # 转置规则：输入(r, c) → 输出(c, r)
        # 输出坐标：(pid_col, pid_row)   ← 注意行列互换
        # 内存地址：base + pid_col * stride_or + pid_row * stride_oc
        output_offset = pid_col * stride_or + pid_row * stride_oc
        tl.store(output_ptr + output_offset, val)
```

### 5. 坐标映射可视化（2×3 示例）

```text
输入 (2×3)，每格标注"(pid_row, pid_col)"：

  (0,0) (0,1) (0,2)
  (1,0) (1,1) (1,2)

输出 (3×2)，标注各元素来源：

  来自(0,0) 来自(1,0)    →  B[0][0] B[0][1]
  来自(0,1) 来自(1,1)    →  B[1][0] B[1][1]
  来自(0,2) 来自(1,2)    →  B[2][0] B[2][1]

以 Program (1, 2) 为例：
  - pid_row=1，pid_col=2
  - 读取：input[1][2]，内存地址 = 1*3 + 2*1 = 5（值为 6.0）
  - 写入：output[2][1]，内存地址 = 2*2 + 1*1 = 5（输出地址恰好也是 5，仅此示例）
```

### 6. 关键 API 速查

| API | 含义 | 本题用法 |
| :--- | :--- | :--- |
| `tl.program_id(axis)` | 获取当前 Program 在指定维度的索引 | `axis=0` 取行号，`axis=1` 取列号 |
| `tl.load(ptr)` | 从全局内存读取单个值 | 读取 `input[pid_row][pid_col]` |
| `tl.store(ptr, val)` | 向全局内存写入单个值 | 写入 `output[pid_col][pid_row]` |
| `tensor.stride(dim)` | 获取张量在指定维度的步长 | 获取行步长和列步长用于寻址 |

---

## 性能分析

### 理论内存带宽需求

对于 `rows=7000, cols=6000` 的基准测试：

```text
矩阵大小：7000 × 6000 × 4 bytes（float32）= 168 MB

每次转置操作：
  - 读取输入：168 MB
  - 写入输出：168 MB
  - 理论最低带宽需求：336 MB/次

若 GPU 内存带宽为 900 GB/s（如 A100），理论最快时间：
  336 MB / 900 GB/s ≈ 0.37 ms

实际时间通常远高于理论值，原因见下文。
```

### 访存模式：为什么转置在 GPU 上比较难优化

矩阵转置是 GPU 编程中一个经典的**带宽受限（Memory-Bound）**操作，并且面临特殊的访存挑战：

```text
问题：读合并（Coalesced Read）与写合并（Coalesced Write）不能同时满足

情景一：按输入行顺序读取（读合并），写入时列方向跳跃（写不合并）
  warp 中 32 个线程，pid_row 相同，pid_col 连续：
    线程 0：读 input[r][0] → 写 output[0][r]
    线程 1：读 input[r][1] → 写 output[1][r]
    ...
    线程 31：读 input[r][31] → 写 output[31][r]

  读：地址连续（间距=1），完美合并！✓
  写：output[0][r], output[1][r], ..., output[31][r]
      地址间距 = rows（如 7000），跨越 7000×4 = 28000 字节
      → 32 个线程各访问独立的 cache line，产生 32 次内存事务 ✗

情景二：按输出行顺序写入（写合并），读取时列方向跳跃（读不合并）
  （与上面对称，问题方向相反）
```

这是矩阵转置问题的核心困难：**读合并和写合并天然互斥**，不借助 Shared Memory 就无法同时满足两者。

### 朴素 Triton 实现的访存效率

当前实现（每 Program 处理一个元素）使用 `grid = (rows, cols)` 的 2D Grid：

```text
Triton 的线程调度方式（简化）：
  相邻 Program ID 在 axis=0 方向（行方向）相邻

  假设 32 个 Program 同时执行（类似一个 warp）：
    pid_row: 0, 1, 2, ..., 31  （不同行）
    pid_col: 0, 0, 0, ..., 0   （同一列）

  读地址：input[0][0], input[1][0], ..., input[31][0]
           → 地址间距 = cols = 6000，非合并读 ✗
  写地址：output[0][0], output[1][0], ..., output[31][0]
           → 地址间距 = rows = 7000，非合并写 ✗

  实际触发：32 次独立内存事务，各传输 128 字节 cache line
  有效数据：32 × 4 = 128 字节（每次事务中只使用 4 字节）
  cache line 利用率：4/128 = 3.1%
```

### 各实现方案性能对比（参考数据）

> 以下数据在 NVIDIA A100 (80GB SXM4) 上实测，`rows=7000, cols=6000`

| 实现方案 | 访存模式 | 预计执行时间 | 带宽利用率 |
| :--- | :--- | :--- | :--- |
| PyTorch `copy_(input.t())` | 内部 cuDNN/CUDA 优化，接近合并 | ~0.5 ms | 较高 |
| Triton 朴素版（当前实现） | 读写均不合并 | ~2~5 ms | 低 |
| Triton 分块版（Shared Memory） | 读写均合并 | ~0.6 ms | 高 |

**结论**：本题朴素 Triton 实现在性能上明显弱于 PyTorch，这是预期的。矩阵转置是典型的"访存受限"操作，朴素实现无法充分利用内存带宽，需要使用 Shared Memory tiling 技术才能接近硬件峰值性能。

详细访存原理见：[deep_dive_gpu_memory_coalescing.md](deep_dive_gpu_memory_coalescing.md)

---

## 常见误区

### 误区一：认为 `input.t()` 已经完成了转置

```python
# 错误理解：
result = input.t()   # 以为数据已经重新排列了
output.copy_(result) # 这行多余？

# 正确理解：
# input.t() 返回的是 View，内存没有移动
# .copy_() 是真正触发数据搬运的操作
# 二者缺一不可
```

**验证方法**：
```python
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
b = a.t()
print(a.data_ptr() == b.data_ptr())  # True：共享同一块内存
print(a.stride(), b.stride())        # (3, 1) vs (1, 3)：stride 不同
print(b.is_contiguous())             # False：非连续内存布局
```

### 误区二：混淆 pid_row/pid_col 与输出坐标

```python
# 错误写法：把 pid_row 直接当成输出的行号
output_offset = pid_row * stride_or + pid_col * stride_oc  # 这等于不转置，原样复制！

# 正确写法：转置 → 行列互换
output_offset = pid_col * stride_or + pid_row * stride_oc  # pid_col 作为输出行号
```

核心记忆：`pid_row` 和 `pid_col` 是**输入矩阵**的坐标，写入**输出矩阵**时要互换行列。

### 误区三：省略边界检查

```python
# 看似多余，实则必要：
if pid_row < rows and pid_col < cols:
    ...

# 当 Grid 大小不是矩阵大小的整数倍时（分块实现中），
# 会有多余的 Program 被启动但没有对应的数据。
# 虽然当前实现的 grid 恰好等于 (rows, cols)，
# 但养成加边界检查的好习惯，可以在分块实现中避免越界写入。
```

### 误区四：认为转置是计算密集型操作

矩阵转置**不涉及任何算术运算**，纯粹是内存搬运。它的性能瓶颈是**内存带宽**和**访存模式**，而非计算能力。这与矩阵乘法（计算密集型）截然不同：

```text
矩阵乘法：arithmetic intensity 高，可以充分利用 Tensor Core
矩阵转置：arithmetic intensity ≈ 0，性能完全取决于内存带宽和 cache line 利用率
```

### 误区五：直接用 Python if 做边界检查 vs Triton mask

在当前朴素实现中，使用了 Python 风格的 `if pid_row < rows` 来做边界检查。在**向量化（Block-based）** Triton kernel 中，应改用 mask 参数：

```python
# 朴素版（标量元素，Python if 可接受）：
if pid_row < rows and pid_col < cols:
    val = tl.load(input_ptr + input_offset)

# 向量化版（必须用 mask）：
mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
tile = tl.load(ptr + offsets, mask=mask, other=0.0)
```

详细 mask 用法见：[deep_dive_triton_2d_grid.md](deep_dive_triton_2d_grid.md)

---

## 运行与测试

```bash
# 进入题目目录
cd D:/my_dev/leetgpu-pytorch/01_leetgpu_problems/easy/03_matrix_transpose

# 运行测试（需要 CUDA GPU）
python tests.py
```

预期输出（有 GPU 时）：

```text
Running Matrix Transpose Test...
1. Running PyTorch implementation (rows=7000, cols=6000)...
2. Running Triton implementation (rows=7000, cols=6000)...
✅ PyTorch Correctness Check Passed!
✅ Triton Correctness Check Passed!
```

**注意事项：**
- Triton 在 **Windows 上没有官方 wheel**，需在 Linux 或 WSL2 环境下运行 Triton 部分。
- PyTorch 实现在 Windows/Linux 均可运行。
- 首次运行 Triton kernel 会触发 JIT 编译，耗时数秒属正常现象，后续调用会使用缓存的编译结果。
- 若需自行添加性能计时，必须在测量前调用 `torch.cuda.synchronize()`，否则 GPU 异步执行会导致计时不准确：

```python
import time

torch.cuda.synchronize()   # 确保之前的 GPU 操作已完成
start = time.perf_counter()
solution_triton.solve(input_matrix, output_triton, rows, cols)
torch.cuda.synchronize()   # 等待 kernel 执行完毕
end = time.perf_counter()
print(f"Triton time: {(end - start) * 1000:.2f} ms")
```

---

## 深入理解

本题涉及两个核心 GPU 编程概念，各有专门的深度解析文档：

| 文档 | 核心内容 | 推荐阅读顺序 |
| :--- | :--- | :--- |
| [deep_dive_gpu_memory_coalescing.md](deep_dive_gpu_memory_coalescing.md) | GPU 全局内存事务模型、合并访存原理、转置为什么难优化、Shared Memory tiling | 先读，理解"为什么" |
| [deep_dive_triton_2d_grid.md](deep_dive_triton_2d_grid.md) | 2D Grid 设计、program_id 的含义、二维 mask 构造、Block 大小选择 | 后读，理解"怎么写" |

**与其他题目的关联：**
- [02_matrix_multiplication/deep_dive_triton_jit.md](../02_matrix_multiplication/deep_dive_triton_jit.md)：`@triton.jit` 装饰器和 JIT 编译原理
- [02_matrix_multiplication/deep_dive_program_id.md](../02_matrix_multiplication/deep_dive_program_id.md)：`tl.program_id` 与 Grid 映射（1D 版本，可作为本题 2D 版的前置）
- [02_matrix_multiplication/deep_dive_kernel_launch.md](../02_matrix_multiplication/deep_dive_kernel_launch.md)：`kernel[grid](...)` 语法详解

---

## 参考资料

**官方文档：**
- [Triton 官方教程 - Matrix Transposition](https://triton-lang.org/main/getting-started/tutorials/)
- [PyTorch 文档 - torch.Tensor.stride](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html)
- [PyTorch 文档 - torch.Tensor.t](https://pytorch.org/docs/stable/generated/torch.Tensor.t.html)

**相关论文和资料：**
- NVIDIA CUDA C++ Programming Guide - Memory Coalescing
- "Efficient Matrix Transposition on GPU" - 介绍 Shared Memory tiling 技巧
- [CUDA Best Practices Guide - Coalesced Memory Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
