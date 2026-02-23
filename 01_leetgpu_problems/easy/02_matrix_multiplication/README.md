# 02. Matrix Multiplication (矩阵乘法)

## 题目描述

编写一个 GPU 程序，执行两个 32 位浮点数矩阵的乘法。给定一个维度为 $M \times N$ 的矩阵 $A$ 和一个维度为 $N \times K$ 的矩阵 $B$，计算乘积矩阵 $C = A \times B$，其维度为 $M \times K$。所有矩阵都以行优先格式存储。

### 实现要求

*   仅使用原生特性（不允许使用外部库的 matmul 函数，但在 PyTorch 解法中可以使用 `torch.mm` 或 `torch.matmul` 作为基准）。
*   `solve` 函数签名必须保持不变。
*   最终结果必须存储在矩阵 `C` 中。

### 示例 1

```
Input:
Matrix A (2 x 2):
[[1.0, 2.0],
 [3.0, 4.0]]

Matrix B (2 x 2):
[[5.0, 6.0],
 [7.0, 8.0]]

Output:
Matrix C (2 x 2):
[[19.0, 22.0],
 [43.0, 50.0]]
```

### 约束条件

*   $1 \le M, N, K \le 8192$
*   性能测试时 $M=8192, N=6144, K=4096$

## 解题思路

### 方法 1：PyTorch 原生实现

使用 `torch.mm` (Matrix Multiplication) 或 `@` 运算符。为了满足 In-place 要求，建议使用 `torch.mm(A, B, out=C)` 或 `C.copy_(A @ B)`。

### 方法 2：Triton Kernel 实现

这是 GPU 编程的经典案例。我们需要利用 **Shared Memory (SRAM)** 来减少对 Global Memory 的访问。

**核心思想 (Tiling / 分块)**：
1.  将矩阵 $C$ 划分为 $BLOCK\_SIZE\_M \times BLOCK\_SIZE\_N$ 的小块。
2.  每个 Triton Program 负责计算 $C$ 的一个小块。
3.  在计算过程中，沿着 $K$ 维度进行迭代（每次步进 $BLOCK\_SIZE\_K$）。
4.  在每一步迭代中：
    *   将 $A$ 的一小块 ($BLOCK\_SIZE\_M \times BLOCK\_SIZE\_K$) 加载到 SRAM。
    *   将 $B$ 的一小块 ($BLOCK\_SIZE\_K \times BLOCK\_SIZE\_N$) 加载到 SRAM。
    *   计算这两个小块的乘积，并累加到寄存器（Accumulator）中。
5.  循环结束后，将累加器的结果写回矩阵 $C$。

这种方法通过在 SRAM 中复用数据，极大地降低了显存带宽压力。
