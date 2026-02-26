# Matrix Transpose (矩阵转置)

## 题目描述

编写一个 GPU 程序，将一个由 32 位浮点数组成的矩阵进行转置。矩阵转置是指将矩阵的行和列互换。给定一个维度为 `rows x cols` 的矩阵，其转置矩阵的维度将变为 `cols x rows`。所有矩阵均以行优先 (Row-Major) 格式存储。

## 实现要求

*   仅使用原生特性（不允许使用外部库的 matmul/transpose 函数，但在 PyTorch 解法中可以使用 `torch.transpose` 作为基准）。
*   `solve` 函数签名必须保持不变。
*   最终结果必须存储在矩阵 `output` 中。

## 示例

**输入**: 2x3 矩阵
```
[[1, 2, 3],
 [4, 5, 6]]
```

**输出**: 3x2 矩阵
```
[[1, 4],
 [2, 5],
 [3, 6]]
```

## 约束条件

*   1 <= rows, cols <= 8192
*   输入矩阵维度: `rows x cols`
*   输出矩阵维度: `cols x rows`
*   性能测试基准: `cols = 6000`, `rows = 7000`

## 解题思路

### 方法 1：PyTorch 原生实现

PyTorch 提供了 `torch.transpose`（或别名 `.t()`）函数。需要注意的是，PyTorch 的转置操作通常返回的是原张量的一个**视图 (View)**，也就是它只是修改了步长 (Stride) 而没有实际移动内存中的数据。为了满足题目要求“将结果存储在 `output` 矩阵中”，我们需要使用 `.copy_()` 将数据实际复制过去。

代码见：`solution_pytorch.py`

### 方法 2：Triton 朴素实现

这个实现展示了如何使用 Triton 编写一个最基础的、逐元素的矩阵转置 Kernel。我们将矩阵中的每一个元素映射到 Grid 中的一个线程 (Program Instance)。

#### 1. 核心概念：坐标映射 (Coordinate Mapping)

矩阵转置的核心就是交换行索引和列索引：
- 如果一个元素在 **输入** 矩阵中的位置是 `(r, c)`。
- 那么它在 **输出** 矩阵中的位置就应该是 `(c, r)`。

#### 2. Grid 结构（人海战术）

我们启动一个二维的 Grid，其形状与 **输入** 矩阵的维度一致。
- `grid = (rows, cols)`
- 每个线程负责搬运 **一个元素**。

线程的 ID `(pid_row, pid_col)` 告诉我们当前线程负责哪个元素：
- `pid_row`: 输入矩阵中的行号。
- `pid_col`: 输入矩阵中的列号。

#### 3. 内存寻址（按图索骥）

由于 GPU 内存是线性的（一维数组），我们需要将二维坐标 `(row, col)` 转换为一维的内存偏移量 (Offset)。

**输入矩阵 (rows x cols)**:
- 行优先存储。
- 要找到 `(pid_row, pid_col)` 处的元素：
  `Offset = pid_row * stride_input_row + pid_col * stride_input_col`

**输出矩阵 (cols x rows)**:
- 同样是行优先存储，但逻辑维度互换了。
- 原来在输入中 `(pid_row, pid_col)` 的元素，现在属于输出矩阵的 `(pid_col, pid_row)` 位置。
- 要找到输出内存中对应的位置：
  `Offset = pid_col * stride_output_row + pid_row * stride_output_col`

#### 4. Kernel 逻辑图解

```python
# 1. 确认身份
pid_row = tl.program_id(0)
pid_col = tl.program_id(1)

# 2. 从输入读取
input_offset = pid_row * stride_ir + pid_col * stride_ic
val = tl.load(input_ptr + input_offset)

# 3. 写入到输出（注意行列互换）
output_offset = pid_col * stride_or + pid_row * stride_oc
tl.store(output_ptr + output_offset, val)
```

#### 5. 可视化过程

**输入 (2x3)**:
```
(0,0) (0,1) (0,2)  --> 第 0 行
(1,0) (1,1) (1,2)  --> 第 1 行
```

**输出 (3x2)**:
```
(0,0) (1,0)  --> 输出第 0 行 (数据来自输入的第 0 列)
(0,1) (1,1)  --> 输出第 1 行 (数据来自输入的第 1 列)
(0,2) (1,2)  --> 输出第 2 行 (数据来自输入的第 2 列)
```

例如，线程 `(0, 1)`：
- 读取输入位置 `(0, 1)` 的值。
- 写入输出位置 `(1, 0)`。

线程 `(1, 2)`：
- 读取输入位置 `(1, 2)` 的值。
- 写入输出位置 `(2, 1)`。

虽然这个朴素实现是正确的，但它对于 GPU 的显存访问（特别是写操作）并不友好（非合并访存）。更高效的实现通常会使用 Shared Memory 进行分块 (Tiling)，以保证读写都能合并访存。
