# GPU 内存层级 3：Shared Memory (SRAM) —— 革命性的车间料框

这是“GPU 内存层级四部曲”的第三篇。我们终于进入了 SM (Streaming Multiprocessor) 内部，来到了 GPU 优化的**绝对核心**区域：**Shared Memory**。

## 1. 它是谁？
Shared Memory 是位于每个 SM 内部的一块极小（通常 48KB - 164KB）但极快（带宽 10TB/s+）的存储器。

*   **比喻**：这是放在**工人手边的临时料框**。
*   **特点**：
    *   **可编程**：不同于 L1/L2 Cache 是硬件自动管理的，Shared Memory 完全由程序员（或 Triton 编译器）控制。你放什么、什么时候放、什么时候拿，全看代码。
    *   **共享**：同一个 Block 内的所有线程都可以看到这里面的数据（方便线程间通信）。
    *   **极速**：比 Global Memory 快 100 倍以上。

## 2. 核心作用：数据复用 (Data Reuse)

### 2.1 为什么我们需要 Tiling (分块)？

让我们深入看一看矩阵乘法 `C = A * B` 的计算过程。假设矩阵大小都是 `N * N`。

#### 笨办法 (Naive Implementation)
每个线程负责计算 `C` 中的一个元素 `C[i, j]`。
公式是：
> C[i, j] = Sum(A[i, k] * B[k, j]) for k in 0...N-1

*   **动作**：为了算这就**这一个点**，线程必须从 Global Memory 读取 `A` 的整整一行 (`N` 个数) 和 `B` 的整整一列 (`N` 个数)。
*   **总读取量**：`C` 共有 `N^2` 个点。所以总读取次数 = `N^2 * 2N = 2N^3`。
*   **问题**：`A` 中的同一个元素（比如 `A[0, 0]`）会被 `B` 的第 0 列的所有元素用到。但在笨办法里，每次用到它都要去遥远的 Global Memory 读一次。**没有复用！**

---

### 2.2 聪明办法：Tiling (分块) + Shared Memory

我们把大矩阵切成很多小块 (Tile)，比如 `32 * 32` 的小方块。

#### 核心流程
1.  **搬运 (Load)**：一个 Block 的线程合作，把 `A` 的一个小块和 `B` 的一个小块，从 Global Memory **搬运** 到 Shared Memory。
    *   注意：这里每个线程只负责搬运一两个数，大家一起搬，很快就搬完了。
2.  **计算 (Compute)**：所有线程都**只从 Shared Memory** 读取数据来计算。
    *   因为 Shared Memory 就在 SM 内部，读它就像读寄存器一样快。
    *   **复用**：`A` 的这一小块数据，被 Block 内的所有线程反复读取了 `BLOCK_SIZE` 次，但只从 Global Memory 读了一次！
3.  **循环**：算完这一个小块，再搬运下一个小块，直到算完整个 `N`。

#### 代码逻辑 (伪代码)

```python
# 假设 Block 大小是 32x32
# 我们要计算 C[i, j]

accum = 0
# 外层循环：按块移动 (k 每次跳 32 步)
for k in range(0, N, 32):
    # 1. 把 A 的小块 (32x32) 和 B 的小块 (32x32) 加载到 Shared Memory
    # 这步是所有线程并行做的，每人搬一点
    s_a[thread_y][thread_x] = A[i][k + thread_x]
    s_b[thread_y][thread_x] = B[k + thread_y][j]
    
    # !! 必须同步 !! 等大家把数据都搬进 Shared Memory
    __syncthreads()
    
    # 2. 在 Shared Memory 上进行点积计算
    # 这一步非常快，因为读的是 SRAM
    for m in range(32):
        accum += s_a[thread_y][m] * s_b[m][thread_x]
        
    # !! 必须同步 !! 等大家算完，准备搬下一块
    __syncthreads()

C[i, j] = accum
```

---

### 2.3 效果对比：数学证明

*   **笨办法读取次数**：`2 * N^3`
*   **Tiling 读取次数**：
    *   每个 `C` 的 Tile (`32 * 32`) 需要计算 `N` 次（分 `N/32` 个阶段）。
    *   每个阶段只需要读 `A` 的 Tile (`32 * 32`) 和 `B` 的 Tile (`32 * 32`）。
    *   总读取量 `≈ 2 * N^3 / 32`。
*   **结论**：显存访问量减少了 **32 倍**！如果 Block 大小是 128，就减少 **128 倍**！这就是为什么 Tiling 是 GPU 优化的**第一定律**。

---

## 3. 潜在陷阱：Bank Conflict (存储体冲突)

Shared Memory 虽然快，但它被分成了 32 个 **Bank (存储体)**（就像料框里有 32 个格子）。

*   **理想情况**：32 个线程（Warp）分别去访问 32 个不同的 Bank。大家互不干扰，速度全开。
*   **冲突情况**：32 个线程同时去访问**同一个 Bank** 的不同地址。
    *   比喻：32 个工人同时把手伸进同一个格子里拿东西。
    *   后果：**串行化 (Serialization)**。大家得排队，速度慢 32 倍。

### Triton 代码示例

```python
import triton
import triton.language as tl

@triton.jit
def tiled_matmul(a_ptr, b_ptr, c_ptr, ...):
    # 1. 在 Shared Memory 中分配空间
    # Triton 会自动把这些变量放在 SRAM 里
    a_tile = tl.load(a_ptr + offsets...)
    b_tile = tl.load(b_ptr + offsets...)
    
    # 2. 计算 (Compute)
    # 这里的 a_tile 和 b_tile 就在 Shared Memory 里
    # 线程们疯狂地反复读取它们，完全不消耗显存带宽
    accumulator += tl.dot(a_tile, b_tile)
    
    # ...
```

## 4. 优化建议

1.  **能用 Shared Memory 就用**：任何需要被多次读取的数据，都应该先搬到 Shared Memory。
2.  **注意 Bank Conflict**：如果你在写 CUDA C++，需要小心设计数组索引（比如加 padding）。好消息是，**Triton 编译器通常会自动帮你处理 Bank Conflict**，这是 Triton 的一大卖点。
3.  **Occupancy 权衡**：Shared Memory 是有限的（比如 100KB）。如果你的 Kernel 每个 Block 都要用 50KB，那一个 SM 只能跑 2 个 Block。有时候为了提高并行度（Occupancy），需要省着点用 Shared Memory。

**下一篇预告**：Shared Memory 还要大家分着用，有没有什么是完全属于线程自己的私房钱？请看终章 **Registers**。
