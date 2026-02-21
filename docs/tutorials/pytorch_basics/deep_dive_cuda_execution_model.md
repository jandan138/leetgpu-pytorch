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

## 2. 硬件核心：SM (Streaming Multiprocessor) —— 车间主任

如果说 Block 是软件上的“生产小组”，那么 **SM** 就是硬件上的“生产车间”。

### 2.1 映射关系
*   **Grid (软件)** -> **Device (硬件)**：一个 Kernel 在整个 GPU 上跑。
*   **Block (软件)** -> **SM (硬件)**：一个 Block 会被分配给一个 SM 执行。
    *   **关键点**：一个 Block 一旦分配给某个 SM，它就会在这个 SM 上一直跑到结束，**不能中途换 SM**。
    *   **并行**：一个 SM 可以同时跑多个 Block（只要资源够用）。
*   **Thread (软件)** -> **CUDA Core (硬件)**：每个线程在具体的计算单元上执行。

### 2.2 SM 内部有什么？
想象 SM 是一个设备齐全的车间：
1.  **CUDA Cores (工人)**：真正干活的地方。包括 FP32 Core, INT32 Core, 和专门做矩阵乘法的 Tensor Core。
2.  **Shared Memory / L1 Cache (料框)**：这就是我们之前提到的“极速缓存”。Block 内的所有线程都共享这块宝贵的 100KB 左右的内存。
3.  **Register File (工具箱堆)**：每个 SM 有巨大的 **寄存器堆 (Register File)**（比如 256KB）。
    *   **分配机制**：这些寄存器会被**动态瓜分**给正在运行的所有线程。
    *   **私有性**：虽然物理上大家都在一个大堆里，但分给你之后，就只有你能用，别的线程看不见。这就是所谓的“私有寄存器”。
4.  **Warp Scheduler (工头)**：负责指挥 32 个线程（Warp）什么时候执行什么指令。

### 2.3 资源限制 (Occupancy)
为什么不能把 Block 设置得无穷大？或者为什么一个 SM 不能同时跑一万个 Block？
**因为资源（Shared Memory 和 Registers）是有限的！**

*   如果你的 Kernel 写的不好，每个线程用了太多寄存器，或者每个 Block 用了太多 Shared Memory。
*   SM 就只能同时跑很少的 Block。
*   结果就是：**SM 大部分时间在空转，GPU 利用率低**。这就是 CUDA 优化的核心难点之一。

---

## 3. 硬件调度之谜：Warp (线程束)

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

## 4. 总结：GPU 的组织架构

| 软件概念 (Software) | 硬件对应 (Hardware) | 特性 |
| :--- | :--- | :--- |
| **Thread** | **CUDA Core (SP)** | **私有寄存器 (Register File)**。物理上在 SM 的大堆里，逻辑上归该线程独占，极快。 |
| **Block** | **SM (Streaming Multiprocessor)** | 共享 Shared Memory，Block 间无法直接通信 |
| **Grid** | **Device (GPU)** | 共享 Global Memory，延迟高 |
| **Warp** | **调度单元 (Scheduler)** | 32 线程同进退，怕 `if-else` |

理解了这些，你就明白了为什么 Triton 和 CUDA 代码里总是充满了 `BLOCK_SIZE` 这样的参数——那是为了凑齐 Warp，榨干 SM 的每一个周期。
