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

为什么矩阵乘法 (MatMul) 必须要用 Shared Memory？

假设计算 $C = A \times B$。如果不分块：
*   为了算 $C$ 的一个点，需要读 $A$ 的一行和 $B$ 的一列。
*   Global Memory 的读取次数是 $2 \times N^3$。这简直是灾难。

**Tiling (分块) 优化**：
1.  把 $A$ 和 $B$ 切成小块（Tile），比如 $128 \times 128$。
2.  把这一小块从 Global Memory **搬运** 到 Shared Memory。
3.  Block 内的线程反复读取 Shared Memory 里的这一小块数据，计算出 $C$ 的一部分。
4.  Global Memory 的读取次数降低到了 $2 \times N^3 / \text{BLOCK\\_SIZE}$。如果块大小是 128，**我们就减少了 99% 的显存访问！**

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
