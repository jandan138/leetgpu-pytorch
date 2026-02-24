# 02. Matrix Multiplication (矩阵乘法) - Naive Version

## 题目描述

编写一个 GPU 程序，执行两个 32 位浮点数矩阵的乘法。给定一个维度为 `M x N` 的矩阵 `A` 和一个维度为 `N x K` 的矩阵 `B`，计算乘积矩阵 `C = A x B`，其维度为 `M x K`。所有矩阵都以行优先格式存储。

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

#### 1. 核心心法：每人算一个格子（小学生排队比喻）

想象我们要计算 `C = A x B`。假设 `A` 和 `B` 都是 `2 x 2` 的矩阵（为了简单）：

在很多 Markdown “网页渲染器”里，`$$...$$` 这种 LaTeX 公式块可能**不会渲染**（会显示一堆反斜杠和花括号）。为了保证任何地方都能看懂，我先给出**纯文本版**：

```text
A = [[A00, A01],
     [A10, A11]]

B = [[B00, B01],
     [B10, B11]]

C = A x B = [[C00, C01],
             [C10, C11]]
```

<details>
<summary>可选：LaTeX 公式版（在支持 MathJax/KaTeX 的渲染器中会更美观）</summary>

$$
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{bmatrix}
\times
\begin{bmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{bmatrix}
=
\begin{bmatrix}
C_{00} & C_{01} \\
C_{10} & C_{11}
\end{bmatrix}
$$

</details>

我们的目标是算出 `C` 里面那 4 个格子的值。

*   **Triton 的“人海战术” (Grid)**：
    *   既然有 4 个格子要算，我们就雇 **4 个工人**（Program/Thread）！
    *   **Grid = (2, 2)**：启动 2 行 2 列，共 4 个工人。
    *   每个工人都有自己的编号 `(pid_m, pid_k)`：
        *   **工人 (0, 0)**：负责算 `C00`（也就是 `C[0,0]`）
        *   **工人 (0, 1)**：负责算 `C01`（也就是 `C[0,1]`）
        *   **工人 (1, 0)**：负责算 `C10`（也就是 `C[1,0]`）
        *   **工人 (1, 1)**：负责算 `C11`（也就是 `C[1,1]`）

*   **工人怎么干活？（以工人 (0, 0) 为例）**
    *   **任务**：算出 `C00`（`C[0,0]`）。
    *   **公式（点积）**：`C00 = A00*B00 + A01*B10`（也就是 A 的第 0 行 与 B 的第 0 列逐项相乘再相加）
    *   代码里的 `for n in range(0, N):` 就是在做这个“对应相乘再累加”的过程。
        *   **第 1 轮 (n=0)**：去 A 里拿 `A00`，去 B 里拿 `B00`，算 `A00*B00`，记在小本本上。
        *   **第 2 轮 (n=1)**：去 A 里拿 `A01`，去 B 里拿 `B10`，算 `A01*B10`，加到小本本上。

#### 2. 关键概念：步长 (Stride)

电脑内存是一条长长的纸带，**不是**二维表格。
矩阵 `A` 在内存里是这样排队的：
`[A00, A01, A10, A11]`

*   **问题**：工人 (1, 0) 想要找 `A10`（第 1 行第 0 列），他怎么知道在纸带的第几个位置？
*   **Stride (步长)**：
    *   **Stride_AM (行步长)**：每换一行，要跳过几个数？
        *   A 有 2 列，所以换行要跳过 2 个数。`stride_am = 2`。
    *   **Stride_AN (列步长)**：每换一列，要跳过几个数？
        *   就在隔壁，所以跳过 1 个数。`stride_an = 1`。
*   **寻址公式**：`地址 = pid_m * stride_am + n * stride_an`
*   **验证**：工人 (1, 0) 想找 `A10` (第 1 行，第 0 列)：
    `地址 = 1 * 2 + 0 * 1 = 2`。内存里第 0、1、**2** 号位置，正好就是 `A10`！

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
    # C[pid_m, pid_k] 的值初始为 0 (我的小本本)
    accumulator = 0.0
    
    # 3. 循环计算点积
    # C[m, k] = Sum( A[m, n] * B[n, k] ) for n in 0..N
    for n in range(0, N):
        # --- 定位 A[pid_m, n] ---
        # 也就是：A 的第 pid_m 行，第 n 列
        # 寻址公式：基地址 + 行偏移 + 列偏移
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
    # 寻址公式同上
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
