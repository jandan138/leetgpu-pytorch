# DMA 与异步计算：释放 CPU 的潜能

在上一篇文档中，我们解释了为什么数据必须放在 Pinned Memory 里。这一篇，我们将深入探讨数据是如何通过 **DMA (Direct Memory Access)** 被“搬运”的，以及如何利用这一机制实现 CPU 和 GPU 的**并行工作**。

## 1. 什么是 DMA (直接内存访问)？

在早期的计算机中，如果要把数据从硬盘读到内存，CPU 必须亲自干活：读一个字节，写一个字节。这期间 CPU 完全被占用了，不能干别的。

**DMA 控制器** 的出现改变了一切。它就像是 CPU 的 **“搬运工”**。

### 1.1 工作流程
1.  **CPU 下达指令**：CPU 告诉 DMA 控制器：“嘿，把 Pinned RAM 地址 0x1000 开始的 1GB 数据，搬到 GPU 显存地址 0x2000。”
2.  **CPU 甩手掌柜**：CPU 发完指令后，立刻**返回**（Return），去执行下一行代码（比如准备下一个 Batch 的数据，或者做一些 CPU 上的预处理）。
3.  **DMA 搬运**：DMA 控制器接管 PCIe 总线，开始搬运数据。这个过程**完全不消耗 CPU 的算力**。
4.  **搬运完成**：DMA 给 CPU 发一个中断信号：“老板，活干完了。”

---

## 2. 异步传输 (Asynchronous Transfer)

在 PyTorch 中，默认的 `.to('cuda')` 是**同步**的。也就是说，CPU 会傻傻地等待 DMA 搬运完成，才执行下一行代码。这显然浪费了 CPU 的时间。

### 2.1 开启异步
```python
# 假设 host_tensor 已经在 Pinned Memory 里了
# non_blocking=True 告诉 PyTorch：不要等，发完指令就走！
gpu_tensor = host_tensor.to('cuda', non_blocking=True)

# 下面这行代码会立刻执行，哪怕数据还在 PCIe 上跑
do_something_on_cpu()
```

### 2.2 为什么要用 non_blocking=True？
这可以让 CPU 和 GPU **重叠工作 (Overlap)**。

*   **同步模式 (Serial)**：
    ```
    时间轴: |----CPU数据准备----|----PCIe传输----|----GPU计算----|
    ```
    总耗时 = T(CPU) + T(Copy) + T(GPU)

*   **异步模式 (Overlapped)**：
    ```
    GPU/DMA:                |----PCIe传输----|----GPU计算----|
    CPU:    |----CPU数据准备----|----准备下一个Batch----|
    ```
    如果 CPU 准备下一个 Batch 的时间小于 GPU 计算的时间，那么 CPU 的耗时就被**完全掩盖**了！

---

## 3. CUDA Streams (流)：多条流水线

光有异步传输还不够。如果 GPU 正在忙着计算上一个 Batch，DMA 能不能同时把下一个 Batch 的数据搬进去？

答案是：**能，但需要多条流水线（Streams）。**

### 3.1 默认流 (Default Stream)
默认情况下，PyTorch 的所有 GPU 操作都在**默认流**中排队。
*   任务 1：搬运 Batch A
*   任务 2：计算 Batch A
*   任务 3：搬运 Batch B
*   任务 4：计算 Batch B

即使 CPU 提前发出了任务 3 的指令，因为任务 2 还没做完，GPU 的 Copy 引擎（Copy Engine）也会闲着。

### 3.2 多流并行
如果我们创建两个流：
*   **Stream 1**: 处理 Batch A
*   **Stream 2**: 处理 Batch B

GPU 硬件有独立的 **Copy Engine**（负责搬运）和 **Compute Engine**（负责计算）。它们可以同时工作！

```
Copy Engine:    |--Copy A--|            |--Copy B--|
Compute Engine:            |--Calc A--|            |--Calc B--|
```
变成了：
```
Copy Engine:    |--Copy A--|--Copy B--|
Compute Engine:            |--Calc A--|--Calc B--|
```
**Copy B 和 Calc A 实现了重叠！** 这种技术叫 **"Hiding Latency" (掩盖延迟)**。

### 3.3 深入底层：GPU 多流是怎么实现的？
你可能会问：CPU 异步是因为有 DMA 控制器单独干活，那 GPU 凭什么能一边算一边搬？它内部也有多个“大脑”吗？

**是的！GPU 内部有独立的硬件引擎和调度器。**

#### 1. 独立的硬件引擎 (Independent Engines)
GPU 芯片上不仅有成千上万个计算核心（CUDA Cores / SMs），还有专门负责搬运数据的电路：
*   **Compute Engine (CE)**：负责执行 Kernel（矩阵乘法、卷积等）。
*   **Copy Engine (DMA)**：专门负责 PCIe 数据传输。
    *   现代高端 GPU（如 A100）通常有 **2 个 Copy Engines**：一个负责上传（Host -> Device），一个负责下载（Device -> Host）。这意味着它可以**同时**读和写！
    *   即使是消费级 GPU（如 RTX 3090/4090），通常也至少有一个独立的 Copy Engine。

#### 2. 硬件调度器 (Hardware Scheduler)
在 GPU 的最前端，有一个非常聪明的硬件单元，通常被称为 **GigaThread Engine** 或 **Front-End Scheduler**。它的工作流程如下：

1.  **指令队列**：CPU 通过驱动程序把指令塞进不同的 Stream 队列里。
    *   Stream 1 队列：`[Copy A, Calc A]`
    *   Stream 2 队列：`[Copy B, Calc B]`
2.  **依赖分析**：调度器会检查指令之间的依赖关系。Stream 1 和 Stream 2 是独立的，互不干扰。
3.  **分发指令**：
    *   调度器看到 `Copy A`，发现 Copy Engine 空闲，发给 Copy Engine 执行。
    *   当 `Copy A` 做完，开始做 `Calc A`（发给 Compute Engine）。
    *   **关键点**：此时调度器看到 Stream 2 的 `Copy B` 已经在排队了。
    *   它发现 **Copy Engine 闲着**（因为 `Calc A` 正在用 Compute Engine，没用 Copy Engine）。
    *   于是，调度器立刻把 `Copy B` 发给 Copy Engine。

**结果**：Compute Engine 在算 `Calc A`，Copy Engine 在搬 `Copy B`。两者互不干扰，完美并行！

#### 3. 代码实例：如何使用 Stream？
光说不练假把式，我们来看看在 PyTorch 中怎么写代码来实现上述的“多流并行”。

```python
import torch
import time

# 0. 准备数据 (Pinned Memory)
N = 10000000
# 必须使用 pin_memory=True，否则无法进行异步传输
data_a = torch.randn(N, pin_memory=True)
data_b = torch.randn(N, pin_memory=True)

# 1. 创建两个 Stream (两条流水线)
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 2. 预热 GPU (Warm-up)
torch.zeros(1).cuda()

start = time.time()

# 3. 在 Stream 1 中处理 Batch A
with torch.cuda.stream(stream1):
    # 异步搬运：CPU 发指令后立刻走，Copy Engine 开始搬 A
    input_a = data_a.to("cuda", non_blocking=True)
    # 计算任务：Compute Engine 负责算 A
    output_a = input_a * 2 

# 4. 在 Stream 2 中处理 Batch B
with torch.cuda.stream(stream2):
    # 异步搬运：此时 Copy Engine 可能还在搬 A，也可能搬完了。
    # 关键点：如果 Compute Engine 正在算 A，Copy Engine 闲着，它就会立刻开始搬 B！
    input_b = data_b.to("cuda", non_blocking=True)
    # 计算任务：等 B 搬完，且 Compute Engine 空闲时执行
    output_b = input_b * 2

# 5. 同步 (等待所有流完成)
torch.cuda.synchronize()

print(f"总耗时: {time.time() - start:.4f}秒")
```

**代码解析：**
*   `torch.cuda.Stream()`：创建了独立的命令队列。
*   `with torch.cuda.stream(s)`：在这个上下文里发出的所有 CUDA 命令（搬运、计算），都会进入流 `s` 的队列。
*   **并行时刻**：当 `stream1` 在做乘法计算（占用 Compute Engine）时，`stream2` 的搬运命令（占用 Copy Engine）可以同时执行。

#### 通俗比喻：工厂流水线
*   **CPU**：总指挥，负责下订单。
*   **Stream**：订单列表。
*   **Copy Engine**：**进货卡车**（专门负责运原料）。
*   **Compute Engine**：**生产车间**（专门负责加工）。
*   **Hardware Scheduler**：**工厂经理**。

**单流情况**：
经理死板地按顺序执行：卡车运货 -> 车间加工 -> 卡车运货 -> 车间加工。车间加工时，卡车司机在抽烟；卡车运货时，车间工人在聊天。

**多流情况**：
经理发现卡车闲着，马上让卡车去运下一批货（Stream 2），虽然现在的车间还在加工上一批货（Stream 1）。这样，卡车和车间都在满负荷运转！

---

## 4. 进阶：能同时跑多少个 Stream？
这是一个非常好的问题。**Stream 的数量限制**和**真正并发执行的 Kernel 数量**是两个概念。

### 4.1 软件层面：无限 (逻辑上的)
在代码里，你可以创建成千上万个 `torch.cuda.Stream()`。只要 CPU 内存够用，你可以一直 `new` 下去。
*   但这只是在软件层面创建了无数个“订单列表”。
*   并不代表你有无数个“工厂”来同时处理它们。

### 4.2 硬件层面：有限 (物理上的)
真正决定能同时跑多少任务的，是 **GPU 的硬件资源**。

#### 限制 1：硬件调度器的队列 (Hyper-Q)
早期的 GPU（如 Kepler 架构之前）很笨，硬件上只能同时管理 1 个或很少的几个流。即使你创建了 100 个 Stream，硬件也只能串行执行。
现代 GPU（Ampere, Hopper 等）非常强大，硬件调度器（CWD - Compute Work Distributor）可以同时跟踪和分发 **128 个甚至更多** 的并发 Stream。

#### 限制 2：核心资源 (SM & CUDA Cores) —— 最关键的瓶颈！
这是决定并发数量的根本原因。
想象一个 **大型自助餐厅 (GPU)**，里面有 **100 张桌子 (SMs)**。

*   **场景 A：大胃王 (大模型/大矩阵)**
    *   Stream 1 来了一个大任务（比如 ResNet50 的一次卷积），它太大了，一口气占用了 **100 张桌子**。
    *   此时，即使 Stream 2 也来了任务，调度器发现**没桌子了**。
    *   **结果**：Stream 2 必须排队等待，直到 Stream 1 吃完离开。**并发数 = 1**。

*   **场景 B：小鸟胃 (小向量加法)**
    *   Stream 1 来了一个小任务（比如向量加法），只占用了 **10 张桌子**。
    *   调度器发现还有 90 张空桌子！
    *   于是，Stream 2 的任务（占用 10 张）、Stream 3 的任务（占用 10 张）... 都可以同时坐下吃。
    *   **结果**：可以有 **10 个任务并发执行**。

### 4.3 总结：和什么硬件有关？
1.  **SM (Streaming Multiprocessor) 的数量**：桌子越多，能容纳的并发任务（特别是小任务）就越多。
2.  **Copy Engines 的数量**：决定了能同时进行多少个“搬运”任务（通常是 1-2 个）。
3.  **任务本身的“体积” (Occupancy)**：如果你的任务把 GPU 塞满了，Stream 再多也只能串行。

**一句话总结**：
Stream 就像 **车道**，而 GPU 核心就像 **马路宽度**。你可以画 100 条车道线，但如果马路只有 10 米宽，而第一辆车就是个 10 米宽的巨型卡车，那后面的车还是得等它开过去才能走。

---

## 5. 总结

1.  **DMA** 是 CPU 的外包搬运工，它让数据传输不消耗 CPU 算力。
2.  **`non_blocking=True`** 允许 CPU 在 DMA 搬运时去干别的事（准备下一个 Batch）。
3.  **CUDA Streams** 允许 GPU 的 Copy 引擎和 Compute Engine 同时工作，掩盖掉昂贵的 PCIe 传输时间。

这就是为什么深度学习框架能跑得这么快的秘诀——**永远不要让任何硬件（CPU, GPU, PCIe）闲着**。
