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

### 方法 1：PyTorch 原生实现 (High-Level 深度解析)

在 PyTorch 中，矩阵乘法有多种写法，但效率和原理大不相同。我们以 $C = A \times B$ 为例，假设 $A, B, C$ 都在 GPU 上。

#### 1. 常见写法对比

*   **写法 A (不推荐)：`C = torch.matmul(A, B)` 或 `C = A @ B`**
    *   **动作**：PyTorch 会在后台悄悄申请一块**新显存**（临时 Tensor），把计算结果存进去。
    *   **后果**：然后把变量名 `C` 的标签贴到这块新显存上。原来的 `C` 所指向的显存空间（如果外面有人引用）就被抛弃了，而且也没有被利用起来。
    *   **比喻**：你要把水倒进桶里。这种写法是**“买个新桶装水，然后把旧桶扔了”**。

*   **写法 B (推荐)：`C.copy_(A @ B)`**
    *   **动作**：先申请一块新显存存 `A @ B` 的结果，然后把结果**搬运（Copy）** 到 `C` 的显存里。
    *   **优点**：符合 Python 直觉，确实修改了 `C` 的内容。
    *   **缺点**：中间产生了一个临时 Tensor，多了一次显存分配和一次数据搬运。
    *   **比喻**：**“买个新盆接水，再倒进旧桶里”**。

*   **写法 C (最强)：`torch.mm(A, B, out=C)`**
    *   **动作**：直接调用底层的 cuBLAS 库，告诉它：“结果直接写到 `C` 的地址里去！”
    *   **优点**：**零显存开销 (Zero Memory Overhead)**。速度最快，显存占用最小。
    *   **比喻**：**“直接拿旧桶接水”**。

#### 2. 代码实现

```python
def solve(A, B, C, M, N, K):
    # 方式 1：In-place Copy (语义清晰)
    # C.copy_(torch.mm(A, B))
    
    # 方式 2：Direct Output (极致性能)
    # 注意：torch.mm 只能做 2D 矩阵乘法，而 torch.matmul 可以做高维。
    # 在本题中（都是 2D），torch.mm 是最纯粹的选择。
    torch.mm(A, B, out=C)
```

---

### ❓ 附录：什么是 `__pycache__` 文件夹？

你在目录下可能会看到一个叫 `__pycache__` 的文件夹，它是 Python 的**加速缓存**。

*   **它是什么？**
    当你运行 `import solution_pytorch` 时，Python 解释器不仅会读取源代码，还会把它编译成一种中间格式叫 **Bytecode (字节码)**。这些字节码文件通常以 `.pyc` 结尾。
    
*   **有什么用？**
    **为了下次启动更快！**
    下次你再运行程序时，Python 会检查：
    1.  `solution_pytorch.py` 改动过吗？
    2.  如果没有，直接加载 `__pycache__` 里的 `.pyc` 文件（跳过编译步骤）。
    3.  如果改过，重新编译并更新缓存。

*   **需要管它吗？**
    **完全不需要！** 它是自动生成的。你可以在 `.gitignore` 里忽略它，或者随时删掉它（Python 会自动重建）。千万不要手动去修改里面的文件。

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
