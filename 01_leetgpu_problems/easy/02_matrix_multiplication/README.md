# 02. Matrix Multiplication (矩阵乘法) - Naive Version

## 题目描述

编写一个 GPU 程序，执行两个 32 位浮点数矩阵的乘法。给定一个维度为 $M \times N$ 的矩阵 $A$ 和一个维度为 $N \times K$ 的矩阵 $B$，计算乘积矩阵 $C = A \times B$，其维度为 $M \times K$。所有矩阵都以行优先格式存储。

### 实现要求

*   仅使用原生特性（不允许使用外部库的 matmul 函数，但在 PyTorch 解法中可以使用 `torch.mm` 或 `torch.matmul` 作为基准）。
*   `solve` 函数签名必须保持不变。
*   最终结果必须存储在矩阵 `C` 中。

## 解题思路

### 方法 1：PyTorch 原生实现 (High-Level)

参考 [README_PYTORCH.md](README_PYTORCH.md)（这里简略带过，重点讲 Triton）。
推荐写法：`torch.mm(A, B, out=C)`。

### 方法 2：Triton Naive 实现 (Low-Level 入门)

这是一个**教学级**的实现，不包含任何复杂的性能优化（如分块、Shared Memory），旨在帮你理解 Triton 的基本语法和 Grid 映射逻辑。

#### 1. 核心心法：每人算一个格子

想象我们要计算 $C = A \times B$，其中 $C$ 的大小是 $M \times K$。
*   **Grid 设定**：我们启动 $M \times K$ 个 Triton Program（线程）。
*   **分工**：
    *   线程 $(0, 0)$ 负责计算 $C[0, 0]$
    *   线程 $(m, k)$ 负责计算 $C[m, k]$
    *   ...
*   **计算逻辑**：每个线程独立完成一个**点积 (Dot Product)** 运算。
    *   它需要去矩阵 $A$ 里取第 $m$ 行。
    *   去矩阵 $B$ 里取第 $k$ 列。
    *   把这两排数字对应相乘并累加。

#### 2. 关键概念：步长 (Stride)

在 GPU 显存（以及 C/C++ 内存）中，数据是**一维线性存储**的。并没有真正的“二维矩阵”。
为了用一维数组表示二维矩阵，我们引入了 **Stride (步长)** 的概念。

*   **问题**：如何找到矩阵 $A[m, n]$ 在内存中的绝对地址？
*   **公式**：`Address = Base_Ptr + m * Stride_Row + n * Stride_Col`
*   **图解**：
    假设 $A$ 是 $2 \times 3$ 的矩阵，行优先存储。
    ```text
    内存索引:  0      1      2      3      4      5
    数值:     A[0,0] A[0,1] A[0,2] A[1,0] A[1,1] A[1,2]
    ```
    *   要从 $A[0,0]$ 跳到下一行 $A[1,0]$，需要跳过 3 个元素。所以 `stride_am = 3`。
    *   要从 $A[0,0]$ 跳到下一列 $A[0,1]$，需要跳过 1 个元素。所以 `stride_an = 1`。

#### 3. 代码逐行详解

```python
@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,  # A 的步长
    stride_bk, stride_bn,  # B 的步长
    stride_cm, stride_ck   # C 的步长
):
    # 1. 身份确认：我是负责哪个格子的工人？
    # 我们启动的 grid 是 (M, K)
    # axis=0 是第一维 (M)，axis=1 是第二维 (K)
    pid_m = tl.program_id(axis=0)  # 我负责第 pid_m 行
    pid_k = tl.program_id(axis=1)  # 我负责第 pid_k 列
    
    # 2. 准备累加器
    # C[pid_m, pid_k] 的值初始为 0
    accumulator = 0.0
    
    # 3. 循环计算点积
    # C[m, k] = Sum( A[m, n] * B[n, k] ) for n in 0..N
    for n in range(0, N):
        # --- 定位 A[pid_m, n] ---
        # 也就是：A 的第 pid_m 行，第 n 列
        offs_a = a_ptr + (pid_m * stride_am + n * stride_an)
        
        # --- 定位 B[n, pid_k] ---
        # 也就是：B 的第 n 行，第 pid_k 列
        offs_b = b_ptr + (n * stride_bn + pid_k * stride_bk)
        
        # --- 读取并累加 ---
        # tl.load 从 Global Memory 读数据到寄存器
        val_a = tl.load(offs_a)
        val_b = tl.load(offs_b)
        accumulator += val_a * val_b
        
    # 4. 写回结果
    # 算出最终结果后，写回 C[pid_m, pid_k]
    offs_c = c_ptr + (pid_m * stride_cm + pid_k * stride_ck)
    tl.store(offs_c, accumulator)
```

#### 4. 性能分析 (为什么这个版本慢？)

虽然这个版本逻辑简单，但它犯了 GPU 编程的大忌：**重复读取 Global Memory**。

*   **算一下**：
    *   计算 $C[0, 0]$ 需要读 $A$ 的第 0 行。
    *   计算 $C[0, 1]$ **也需要读** $A$ 的第 0 行。
    *   ...
    *   计算 $C[0, K-1]$ **全都要读** $A$ 的第 0 行。
*   **后果**：$A$ 的第 0 行被从显存里重复读取了 $K$ 次！
*   **改进方向**：这就是为什么我们需要 **Tiling (分块)** 和 **Shared Memory**。我们可以把 $A$ 的第 0 行读一次放到“公共桌子”（Shared Memory）上，让大家一起用，从而减少几百倍的显存读取。这将在进阶版本中介绍。

## 运行与测试

可以直接运行 `tests.py` 进行验证。由于这是朴素版本，对于大矩阵（如 4096 x 4096），它会比 PyTorch 慢很多，这是符合预期的。

```bash
python tests.py
```
