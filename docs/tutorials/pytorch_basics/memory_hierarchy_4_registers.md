# GPU 内存层级 4：Registers (寄存器) —— 极速的私有领地

这是“GPU 内存层级四部曲”的最终章。我们来到了最底层、最快、也是最珍贵的资源：**Registers (寄存器)**。

## 1. 它是谁？
寄存器是位于 SM 内部、分配给**每个线程私有**的存储空间。它是 GPU 上速度最快的存储，没有之一。

*   **比喻**：这是**工人的双手**。你要干活，必须把东西拿在手上。
*   **特点**：
    *   **私有**：线程 A 不能看线程 B 的寄存器。
    *   **极速**：0 延迟。
    *   **编译器控制**：你声明的局部变量（如 `float val = ...`），通常都会被放在寄存器里。

## 2. 核心问题：Register Pressure (寄存器压力)

虽然每个 SM 有巨大的寄存器堆（比如 A100 有 256KB），但别忘了，SM 上要跑几千个线程！

**数学题**：
*   假设 SM 有 65536 个寄存器。
*   如果你写的 Kernel 很复杂，每个线程需要用 255 个寄存器（最大值）。
*   那么一个 SM 最多只能跑 $65536 / 255 \approx 256$ 个线程。
*   如果一个 Block 有 512 个线程，那么 **SM 连一个 Block 都跑不起来！** (Kernel Launch Failed)

这就是 **Register Pressure**。用的寄存器越多，SM 能同时跑的线程数（Occupancy）就越少，GPU 的潜能就发挥不出来。

### 2.1 Register Spill (溢出) —— 性能噩梦
如果你用的寄存器实在太多，连所有的物理寄存器都装不下了，编译器会被迫把多出来的变量**扔到 Local Memory** 里。
*   **Local Memory**：虽然名字叫 Local，但它其实是 **Global Memory (显存)** 的一部分！
*   **后果**：本来想访问极速的寄存器，结果变成了访问极慢的显存。性能直接崩盘。

---

## 3. 代码示例：循环展开 (Loop Unrolling)

循环展开是增加寄存器压力的常见原因。

### 未展开 (寄存器用得少)
```python
# 每次循环复用同一个 'val' 寄存器
for i in range(4):
    val = load(ptr + i)
    sum += val
```

### 展开后 (寄存器用得多)
```python
# 编译器可能会把它优化成这样：
# 需要 4 个寄存器来存 val0...val3
val0 = load(ptr + 0)
val1 = load(ptr + 1)
val2 = load(ptr + 2)
val3 = load(ptr + 3)
sum = val0 + val1 + val2 + val3
```
*   **好处**：增加了指令级并行度 (ILP)，流水线更满。
*   **坏处**：占用了更多寄存器。如果导致 Occupancy 下降，可能得不偿失。

## 4. 优化建议

1.  **保持 Kernel 简洁**：不要在一个 Kernel 里塞太多逻辑，变量越少越好。
2.  **控制 Launch Bounds**：在 CUDA/Triton 中，你可以显式告诉编译器“我这个 Kernel 最多只用多少个寄存器”，迫使编译器去优化（或者 Spill）。
3.  **Triton 的优势**：Triton 编译器非常擅长做 **Register Allocation (寄存器分配)** 和 **Live Range Analysis (活跃区间分析)**，它会自动复用不再使用的寄存器，通常比手写 CUDA 更省心。

---

## 5. 四部曲总结：全景图

| 层级 | 比喻 | 速度 | 容量 | 作用 | 优化关键词 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Memory** | **中心仓库** | 🐢 慢 | 🐘 巨大 | 存放所有数据 | **Coalesced Access (合并访问)** |
| **L2 Cache** | **门口分拣区** | 🚗 中 | 📦 中等 | 缓存 Global | **Locality (局部性)** |
| **Shared Memory** | **车间料框** | 🚀 快 | 👜 小 | Block 内共享/复用 | **Tiling (分块) / Bank Conflict** |
| **Registers** | **工人的手** | ⚡️ 极速 | 🤏 极小 | 线程私有计算 | **Occupancy (占用率)** |

理解了这四个层级，您就掌握了 GPU 性能优化的 **90% 的秘密**。所有的优化技巧（Tiling, Vectorization, Pipeline），本质上都是为了**让数据尽可能久地留在下面两层（寄存器和 Shared Memory），而尽可能少地去访问上面两层**。
