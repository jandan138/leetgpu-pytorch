# GPU 内存层级 2：L2 Cache —— 全局中转站

这是“GPU 内存层级四部曲”的第二篇。在 Global Memory 和 SM 之间，隔着一层非常重要的缓冲：**L2 Cache**。

## 1. 它是谁？
L2 Cache 是 GPU 芯片内部的一块高速缓存，大小通常在几 MB 到几百 MB 之间（例如 A100 是 40MB，RTX 4090 是 72MB）。

*   **比喻**：这是**工厂门口的快速分拣区**。
*   **特点**：
    *   比 Global Memory 快得多（带宽大约高 5 倍）。
    *   所有 SM 共享这块缓存。
    *   如果数据在这里找到了（Hit），就不用去那个遥远的市中心仓库（Global Memory）了。

## 2. 核心机制：Hit (命中) vs Miss (未命中)

当 CUDA Core 想要读取某个地址的数据时：
1.  **先查 L1 Cache** (SM 内部，下一篇讲)。
2.  **再查 L2 Cache**。
3.  **最后查 Global Memory**。

### 2.1 为什么它很重要？
在深度学习中，很多算子（比如卷积）会**反复读取同一个输入数据**。
*   如果没有 Cache：每次都要去 Global Memory 读，带宽立刻被打爆。
*   有了 Cache：第一次读慢一点（Miss），后面几次读直接从 L2 拿（Hit），速度起飞。

### 2.2 缓存行 (Cache Line)
L2 Cache 不是按字节管理的，而是按 **128 字节的 Cache Line** 管理的。
这就解释了上一篇提到的 **Coalesced Access**：
*   如果 32 个线程读的是连续的 128 字节，正好填满一个 Cache Line，完美利用。
*   如果 32 个线程读的是分散的，可能会导致加载 32 个 Cache Line，但每个 Line 里只用到了 4 个字节，浪费了 97% 的带宽。

---

## 3. 进阶特性：L2 Persistence (持久化)

在 CUDA 11 (Ampere 架构) 之后，NVIDIA 引入了一个新特性：**L2 Cache Persistence**。

通常，Cache 是自动管理的（LRU 策略，最近没用的数据会被踢出去）。但在某些场景下，程序员知道：“这块数据我等会儿还要用很多次，千万别把它踢出去！”

### 代码示例 (CUDA/Triton 概念)

```python
# 伪代码：控制 L2 缓存行为
# 这是一个高级优化，Triton 可能会自动处理

# 标记这块显存为 "Persisting" (持久化)
# 告诉 GPU：这块数据是 VIP，尽量留在 L2 Cache 里，别把位置让给别人
cuda.set_access_policy_window(ptr, size, access_policy=cuda.AccessPolicy.Persist)

# 读取数据
data = load(ptr)

# ... 大量计算 ...

# 再次读取数据 -> 此时极大概率还在 L2 Cache 里，秒读！
data_again = load(ptr)
```

## 4. 优化建议

1.  **时空局部性 (Locality)**：写代码时，尽量让需要反复使用的数据在**短时间内**被密集访问，这样能提高 Cache 命中率。
2.  **避免 "Cache Thrashing" (缓存抖动)**：如果你的工作集（Working Set）太大，超过了 L2 Cache 的大小（比如 72MB），数据就会不断地进进出出，Cache 也就失效了。这时候需要通过 **Tiling (分块)** 技术，把任务切小，保证每一小块任务的数据都能塞进 L2。

**下一篇预告**：L2 Cache 还是大家共享的，有没有什么是 SM 私有的、完全由程序员控制的“超级缓存”？请看下一篇 **Shared Memory**。
