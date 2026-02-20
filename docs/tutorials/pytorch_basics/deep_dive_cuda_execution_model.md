# 深度解析：CUDA 执行模型——线程是如何被调度的？

如果你想真正理解 GPU 为什么快，不仅要懂**内存（仓库与车间）**，还要懂**执行模型（工人的组织方式）**。

## 1. 核心概念：Grid, Block, Thread

在 CUDA 编程中，当你启动一个 Kernel（比如矩阵加法）时，你会生成成千上万个线程。为了管理这些线程，CUDA 引入了一个三层级的组织结构：

### 1.1 Thread (线程) —— 最小的工人
*   **定义**：执行指令的最小单位。每个线程都有自己的寄存器（私有工具箱）。
*   **动作**：每个线程执行相同的代码（Kernel），但处理不同的数据（通过线程 ID 区分）。

### 1.2 Block (线程块) —— 生产小组
*   **定义**：一组线程的集合。一个 Block 通常包含 32 到 1024 个线程。
*   **关键特性**：
    *   **共享内存 (Shared Memory)**：同一个 Block 内的线程可以共享这一小块极快的内存（车间料框）。
    *   **同步 (Synchronization)**：同一个 Block 内的线程可以通过 `__syncthreads()` 等待彼此（小组开会）。
    *   **调度单位**：Block 是被分配到 SM (Streaming Multiprocessor, 生产车间) 的最小单位。一个 SM 可以同时跑多个 Block。

### 1.3 Grid (网格) —— 整个工厂
*   **定义**：所有 Block 的集合。对应一次 Kernel 启动。
*   **动作**：当你调用 `kernel<<<GridDim, BlockDim>>>(...)` 时，就是启动了一个 Grid。

---

## 2. 硬件调度之谜：Warp (线程束)

这是很多初学者的知识盲区。你以为 GPU 是以 Thread 为单位调度的吗？**错！**

### 2.1 什么是 Warp？
GPU 硬件（SM）实际上是以 **32 个线程** 为一组进行调度的，这一组线程叫 **Warp**（线程束）。

*   **SIMT 架构 (Single Instruction, Multiple Threads)**：
    *   在一个 Warp 里，这 32 个线程在同一时刻，必须执行**同一条指令**。
    *   想象一下：32 个工人排成一排，工头喊“拿扳手”，大家一起拿扳手；工头喊“拧螺丝”，大家一起拧螺丝。

### 2.2 Warp Divergence (线程束分歧) —— 为什么 GPU 怕 `if-else`？

因为 Warp 里的 32 个线程必须共进退，如果你写了这样的代码：

```cpp
if (threadIdx.x % 2 == 0) {
    // 偶数线程做这件事
    func_A();
} else {
    // 奇数线程做那件事
    func_B();
}
```

**发生了什么？**
1.  **执行 `func_A` 时**：奇数线程必须**暂停等待**（Masked out），只有偶数线程在干活。硬件利用率瞬间减半！
2.  **执行 `func_B` 时**：偶数线程必须**暂停等待**，只有奇数线程在干活。硬件利用率再次减半！
3.  **合并**：直到 `if-else` 结束，32 个线程才重新汇合。

**结论**：在 GPU 编程中，尽量避免让同一个 Warp 里的线程走不同的分支。这叫 **"Branch Divergence"**。

---

## 3. 总结：GPU 的组织架构

| 软件概念 (Software) | 硬件对应 (Hardware) | 特性 |
| :--- | :--- | :--- |
| **Thread** | **CUDA Core (SP)** | 私有寄存器，极快 |
| **Block** | **SM (Streaming Multiprocessor)** | 共享 Shared Memory，Block 间无法直接通信 |
| **Grid** | **Device (GPU)** | 共享 Global Memory，延迟高 |
| **Warp** | **调度单元 (Scheduler)** | 32 线程同进退，怕 `if-else` |

理解了这些，你就明白了为什么 Triton 和 CUDA 代码里总是充满了 `BLOCK_SIZE` 这样的参数——那是为了凑齐 Warp，榨干 SM 的每一个周期。
