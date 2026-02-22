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
# 我们要计算 C[i, j] (大矩阵中的某一个点)
# thread_y, thread_x 是当前线程在 Block 内的坐标 (0~31)

accum = 0

# 外层循环：k 代表“阶段”，每次跳 32 步 (k=0, 32, 64, ...)
# 就像我们要搬完一座金山，每次只能搬一车 (32宽)
for k in range(0, N, 32):
    
    # --- 阶段 1: 协同搬运 (Load) ---
    # 每个线程负责搬运 A 和 B 的一小部分到 Shared Memory (s_a, s_b)
    # 比如：线程(0,0) 搬运 A[i][k] 和 B[k][j]
    # 注意：这里读取的是 Global Memory (慢)
    s_a[thread_y][thread_x] = A[i][k + thread_x]
    s_b[thread_y][thread_x] = B[k + thread_y][j]
    
    # !! 第一次同步 (Barrier) !! 
    # 就像接力赛交棒，或者是拼图。
    # 必须等 Block 里所有 1024 个线程都把自己负责的那块“砖”搬到了 Shared Memory。
    # 否则，如果线程 A 还没搬完，线程 B 就开始算，B 就会读到垃圾数据！
    __syncthreads()
    
    # --- 阶段 2: 高速计算 (Compute) ---
    # 现在数据都在 Shared Memory (s_a, s_b) 里了，速度极快！
    # 我们要计算当前这 32x32 小块的点积贡献
    for m in range(32):
        # 这一步是在 SRAM 上跑，比 Global Memory 快 100 倍
        accum += s_a[thread_y][m] * s_b[m][thread_x]
        
    # !! 第二次同步 (Barrier) !! 
    # 必须等所有人都算完了当前这一车数据。
    # 否则，如果有线程跑得快，进入了下一轮循环(k+32)，
    # 它会覆盖掉 Shared Memory 里的旧数据，导致还没算完的线程读错数据！
    __syncthreads()

# 最后把累加结果写回 Global Memory
C[i, j] = accum
```

#### 对比：如果不使用 Shared Memory (Global Memory 直读)

如果不分块，直接让每个线程去 Global Memory 读数据，代码会变成这样：

```python
# 笨办法：没有 Tiling，没有 Shared Memory
# 线程 (thread_y, thread_x) 负责计算 C[i, j]

accum = 0

# 也要遍历 k (0 到 N-1)
for k in range(N):
    # 灾难现场！！
    # 每次循环，都要去遥远的 Global Memory 读两个数！
    # 1. 读 A[i][k]
    val_a = load_from_global(A, i, k) 
    # 2. 读 B[k][j]
    val_b = load_from_global(B, k, j)
    
    accum += val_a * val_b

C[i, j] = accum
```

**为什么这是灾难？**
1.  **没有团购**：每个线程都自己去跑一趟市场买一根葱。1024 个线程就要跑 1024 趟。
2.  **重复劳动**：
    *   线程 A 需要 `A[0][0]`。
    *   线程 B 也需要 `A[0][0]`。
    *   结果：显存控制器会收到两次读 `A[0][0]` 的请求。虽然 L1/L2 Cache 能缓解一点，但对于大矩阵，Cache 很快就被挤爆了，大部分请求还是要回 Global Memory。

**Tiling 的本质**：
*   **团购**：大家列个单子，派几个人开辆大车去市场，一次把所有需要的葱都买回来。
*   **共享**：买回来放在桌上，谁要用谁拿，不用再跑市场了。

### 2.3 核心疑问解答

**Q: 难道每个线程都去搬就能更快？是不是每个线程都有一条“传输线”？**

**A: 这是一个非常棒的直觉问题！答案是：不是线多了，是车坐满了，而且跑的趟数少了。**

1.  **只有一条大马路 (总线宽度)**
    *   GPU 的显存总线（Memory Bus）非常宽（比如 384-bit）。就像一条超宽的高速公路。
    *   显存控制器一次“发车”最少也要拉 **128 字节**（32 个 float）的数据。哪怕你只想要 **4 字节**（1 个 float），大巴车也得跑一趟，空载率极高。

2.  **合并访问 (Coalescing) —— 把车坐满**
    *   在 Tiling 的搬运阶段，32 个线程（一个 Warp）同时请求 32 个连续的地址。
    *   显存控制器非常聪明，它把这 32 个小请求合并成 **1 个大请求**。
    *   **结果**：大巴车一次拉满，利用率 100%。

3.  **数据复用 (Reuse) —— 少跑几趟**
    *   如果不搬到 Shared Memory：Thread 0 要读 `A[0][0]`，Thread 1 也要读 `A[0][0]`... 这一块数据要被从显存拉 N 次。
    *   如果搬到 Shared Memory：大家合伙把这一块数据拉来 **1 次**，放在手边。之后的一万次访问都是读手边的 Shared Memory，不需要再占用显存总线了。

**总结**：快的秘诀 = **带宽跑满 (Coalescing)** + **流量减少 (Reuse)**。

**Q: 也就是说，分块后才能判断这一块地址是连续的？如果不分块，整个矩阵太大，这块地址本身就是不连续的？**

**A: 地址一直都在那里，没变。变的是大家怎么去拿。**

1.  **矩阵的“物理真相”**
    *   在内存中，矩阵通常是**行优先存储**的。这意味着：
        *   第 0 行：`[0,0], [0,1], [0,2] ...` 是**连续**的。
        *   但是第 0 行和第 1 行之间，可能隔了很远（取决于矩阵宽度 N）。

2.  **不分块时 (各算各的)**
    *   线程 0 算 `C[0,0]`，需要读 `A[0][k]`。
    *   线程 1 算 `C[1,0]`，需要读 `A[1][k]`。
    *   ...
    *   **结果**：虽然大家都在读第 k 列，但 Thread 0 读的是第 0 行，Thread 1 读的是第 1 行。
    *   **内存访问模式**：`地址0, 地址N, 地址2N, ...`。这叫**跨步访问 (Strided Access)**。
    *   **显存控制器崩溃**：这是**极度不连续**的！必须发 32 辆车，每辆车去不同的行拉货。

3.  **分块搬运时 (合作凑单)**
    *   Thread 0~31 **商量好了**：“咱们这次先别管自己算什么，先合伙把 A 的**第 0 行前 32 个数**搬进来”。
    *   Thread 0 搬 `A[0][0]`。
    *   Thread 1 搬 `A[0][1]`。
    *   ...
    *   **结果**：大家请求的地址是 `地址0, 地址1, 地址2...`。
    *   **显存控制器狂喜**：这是**完美连续**的！发一辆大车就能全拉走。

**一句话总结**：分块不仅是为了复用，更是为了让 32 个线程能**“凑”**出一段连续的内存请求，让大巴车不空跑。

**Q: 可是 Thread 1 最终还是要算 `C[1][0]`，还是要用 `A[1][k]` 啊？先搬第 0 行对它有什么好处？**

**A: 这就是“我为人人，人人为我”！让我们用代码视角还原全过程。**

为了计算 `C`，我们必须把 `A` 和 `B` 搬到 Shared Memory。
假设 Block 是 32x32，`k` 是当前阶段的偏移量。

#### 1. 搬运阶段 (Load) - 核心是“连续”
在这一步，线程的目标只有一个：**最高效地把数据从 Global 搬到 Shared**。大家根本不关心自己计算要用什么数，只关心**怎么搬最快**。

*   **任务分配**：Thread `(ty, tx)` 负责搬运 `A` 的 `(ty, tx)` 位置的元素。
*   **Thread 0 (0, 0)**: 搬运 `A[0][k]` -> 存入 `s_a[0][0]`
*   **Thread 1 (0, 1)**: 搬运 `A[0][k+1]` -> 存入 `s_a[0][1]`
*   ...
*   **Thread 32 (1, 0)**: 搬运 `A[1][k]` -> 存入 `s_a[1][0]`

**注意看！**
*   Thread 1 (0, 1) 搬的是 `A[0][k+1]`。
*   但 Thread 1 负责计算的是 `C[0][1]` (假设对应)，计算公式里需要的是 `A[0][:]` 和 `B[:][1]`。
*   在这个例子里，Thread 1 搬的 `A[0][k+1]` 刚好是它自己**计算会用到**的数（撞大运了）。
*   **但是！** 如果 Thread 1 负责搬运 `B` 呢？
    *   Thread 1 (0, 1) 搬运 `B[0][k+1]` (假设 B 也是行优先)。
    *   但 Thread 1 计算需要的是 `B[:][1]` (第 1 列)。
    *   它搬运的 `B[0][k+1]` 是第 `k+1` 列的数，**根本不是它自己要用的！**
    *   **即便如此，它还是要搬！** 因为它搬这个位置，能保证和 Thread 0, Thread 2... 凑成连续访问。

#### 2. 计算阶段 (Compute) - 核心是“随机访问”
现在，`s_a` 和 `s_b` (Shared Memory) 已经被填满了。数据都在桌上了。
这时候，线程才开始做它**真正想做的事**：计算 `C[ty][tx]`。

*   **Thread 1 (0, 1) 的心路历程**：
    *   我要算 `C[0][1]`。
    *   公式：`Sum(A[0][m] * B[m][1])`。
    *   **m=0 时**：
        *   我要 `A[0][0]` -> 去 `s_a[0][0]` 拿。谁搬的？**Thread 0 搬的**。谢谢 Thread 0！
        *   我要 `B[0][1]` -> 去 `s_b[0][1]` 拿。谁搬的？**Thread 1 自己搬的**。
    *   **m=1 时**：
        *   我要 `A[0][1]` -> 去 `s_a[0][1]` 拿。谁搬的？**Thread 1 自己搬的**。
        *   我要 `B[1][1]` -> 去 `s_b[1][1]` 拿。谁搬的？**Thread 33 (1, 1) 搬的**。谢谢 Thread 33！

**结论**：
*   在 Load 阶段，大家是为了**显存带宽**而工作，各司其职搬运连续数据块。
*   在 Compute 阶段，大家是为了**自己的计算任务**而工作，去 Shared Memory 里随意拿别人搬好的数据。
*   这就是为什么说：**Load 阶段的线程索引** 和 **Compute 阶段的线程索引** 逻辑上是解耦的。

### 2.5 常见疑问深入

**Q1: 有没有可能我需要的元素是在别的 Block 被搬的？如何保证数据安全？**

*   **绝对不可能跨 Block 访问 Shared Memory**：
    *   Shared Memory 是**私有**于每个 Block 的。Block A 里的线程，**物理上**根本看不见 Block B 的 Shared Memory。
    *   这就像**每个班级都有自己的黑板**。一班的学生只能看一班的黑板，不可能去二班的黑板上看题。
*   **如何保证数据都在？**
    *   **算法设计**：必须在设计算法时保证**闭环**。
    *   在矩阵乘法中，计算 `C` 的一个小块 (32x32) 所需要的所有数据 (`A` 的对应行和 `B` 的对应列)，必须**完全由当前 Block 自己**负责搬运进来。
    *   所以我们在 `for k` 循环里，每次都把当前需要的 `A` 和 `B` 的那一部分搬进**自己的** Shared Memory。自给自足，不求别人。

**Q2: 这个代码里怎么显式地搬到 Shared Memory？**

在伪代码中我们写的是 `s_a[...] = ...`，在真实编程中，写法略有不同：

**CUDA C++ 写法**：
你需要用 `__shared__` 关键字声明变量。
```cpp
__global__ void matmul_kernel(...) {
    // 1. 声明 Shared Memory (静态分配)
    // 这块内存住在 SM 内部，所有线程可见
    __shared__ float s_a[32][32];
    __shared__ float s_b[32][32];

    // 2. 搬运 (从 Global 读，写入 Shared)
    // 这里的 s_a 就是显式的 Shared Memory 变量
    s_a[threadIdx.y][threadIdx.x] = A[...]; 
    s_b[threadIdx.y][threadIdx.x] = B[...];

    __syncthreads();
    // ...
}
```

**Triton Python 写法 (完整矩阵乘法)**：
Triton 的强大之处在于它把复杂的 Shared Memory 管理（声明、搬运、同步、防 Bank Conflict）都封装在了简洁的 `tl.load` 和 `tl.dot` 背后。

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # 指针参数：指向 Global Memory 中的大矩阵
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长 (Stride)：告诉 Triton 如何在内存中跳转
    # 比如 stride_am=1 表示 A 在 M 维度上是连续的
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 元编程参数 (Meta-parameters)：编译时确定的常量
    # 这些参数决定了分块的大小，必须是 2 的幂次
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # 1. 确定当前 Block 的位置 (Program ID)
    # 就像每个班级有班号，每个 Block 也有自己的 ID
    # pid 决定了当前 Block 负责计算 C 矩阵的哪一块 (32x32)
    pid = tl.program_id(axis=0)
    
    # 1.1 计算当前 Block 在 M (行) 和 N (列) 方向上的网格坐标
    # 假设 Grid 是二维的，我们需要把 1D 的 pid 拆分成 2D 坐标
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 2. 生成指针偏移量 (Load 阶段的准备)
    # -----------------------------------------------------------
    # 就像我们之前说的，Thread 0-31 准备好去搬运第一块连续数据
    # 这里生成的是初始的指针位置
    
    # offs_am: 生成 [0, 1, ..., BLOCK_SIZE_M-1] 的序列
    # 加上 pid_m * BLOCK_SIZE_M 后，就是当前 Block 负责的 M 维度范围
    # % M 是为了防止越界 (虽然通常矩阵大小是 Block 的倍数)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    
    # offs_bn: 同理，生成 N 维度的偏移量
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # offs_k: K 维度的偏移量 (从 0 开始，每次由循环推进)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算 A 和 B 在 Global Memory 中的真实物理指针
    # 指针 = 基地址 + (行索引 * 行步长) + (列索引 * 列步长)
    # None 是为了进行广播 (Broadcasting)，生成 2D 网格指针
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. 初始化累加器 (Compute 阶段的寄存器)
    # 在寄存器中申请一块全 0 的空间，用于存放 C 的部分和
    # 形状是 [BLOCK_SIZE_M, BLOCK_SIZE_N]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. 主循环：按块搬运 + 计算 (Tiling)
    # K 维度也被切成了很多块，我们要一块一块地遍历
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        # --- 自动 Shared Memory 管理 ---
        # 当你调用 tl.load 时，Triton 编译器会：
        # 1. 生成从 Global Memory 读取数据的指令 (Coalesced Access)
        # 2. 在 SM 内部自动分配 Shared Memory 空间
        # 3. 把数据存入 Shared Memory (可能会自动处理 Padding 以避免 Bank Conflict)
        # 4. 这里的 a_ptrs 和 b_ptrs 是指向当前 K 块的指针
        a_tile = tl.load(a_ptrs)
        b_tile = tl.load(b_ptrs)

        # --- 计算 ---
        # tl.dot 会生成 Tensor Core 指令或者高效率的 FMA 指令
        # 它的输入数据 (a_tile, b_tile) 实际上是来自 Shared Memory
        # 计算结果累加到寄存器 accumulator 中
        accumulator += tl.dot(a_tile, b_tile)

        # --- 指针推进 ---
        # 准备搬运下一块
        # A 指针向右移动 BLOCK_SIZE_K 步
        # B 指针向下移动 BLOCK_SIZE_K 步
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. 写回结果
    # 所有的 K 都算完了，accumulator 里存的就是最终的 C[i, j]
    # 重新计算 C 的写入位置指针
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    
    # 还可以加上 mask=... 来处理边界情况
    tl.store(c_ptrs, accumulator)
```

**深度解读 Triton 的魔法：**
1.  **隐式声明**：你找不到 `__shared__` 关键字。`a_tile` 在逻辑上是一个张量，Triton 编译器负责决定把它放在哪里（通常是 Shared Memory，如果是 FP16 甚至可能直接在寄存器）。
2.  **自动同步**：你找不到 `__syncthreads()`。Triton 编译器会分析数据依赖，自动在 `load` 和 `dot` 之间插入必要的同步指令。
3.  **自动流水线 (Pipeline)**：更厉害的是，Triton 还能自动把 `load` 和 `dot` 并行起来（Async Copy），你完全不需要手写复杂的流水线代码。

---

### 2.6 效果对比：数学证明

*   **笨办法读取次数**：`2 * N^3`
*   **Tiling 读取次数**：
    *   每个 `C` 的 Tile (`32 * 32`) 需要计算 `N` 次（分 `N/32` 个阶段）。
    *   每个阶段只需要读 `A` 的 Tile (`32 * 32`) 和 `B` 的 Tile (`32 * 32`）。
    *   总读取量 `≈ 2 * N^3 / 32`。
*   **结论**：显存访问量减少了 **32 倍**！如果 Block 大小是 128，就减少 **128 倍**！这就是为什么 Tiling 是 GPU 优化的**第一定律**。

---

## 3. 潜在陷阱：Bank Conflict (存储体冲突)

Shared Memory 虽然快，但它内部并不是“一块完整的大平地”，而是被切分成了 **32 个竖条**，这些竖条叫做 **Banks (存储体)**。

*   **物理结构**：Bank 0, Bank 1, ..., Bank 31。
*   **地址映射**：
    *   地址 0 -> Bank 0
    *   地址 4 (字节) -> Bank 1
    *   ...
    *   地址 124 -> Bank 31
    *   地址 128 -> **回到 Bank 0** (循环)

### 3.1 什么是冲突？

想象有 32 个取款窗口 (Banks)。

*   **理想情况 (No Conflict)**：
    *   32 个线程 (Warp) 同时去取钱。
    *   Thread 0 去窗口 0，Thread 1 去窗口 1... Thread 31 去窗口 31。
    *   **结果**：所有窗口同时服务，**1 个周期**全部办完。

*   **冲突情况 (Conflict)**：
    *   Thread 0 去窗口 0。
    *   Thread 1 也去窗口 0 (比如它访问地址 128)。
    *   **结果**：窗口 0 同一时间只能服务一个人。Thread 1 必须**排队**等 Thread 0 办完。
    *   这就是 **串行化 (Serialization)**。如果 32 个线程都去同一个窗口，速度就会慢 **32 倍**！

### 3.2 为什么会发生冲突？

通常是因为你的**步长 (Stride)** 设置得不好。

*   **步长 = 1 (连续访问)**：Thread `i` 访问 `i`。Bank `i % 32`。互不相同，**无冲突**。
*   **步长 = 32 (按列访问)**：
    *   Thread 0 访问 0 (Bank 0)。
    *   Thread 1 访问 32 (Bank 0)。
    *   Thread 2 访问 64 (Bank 0)。
    *   **大灾难**：所有 32 个线程都撞到了 Bank 0！这就是最经典的 Bank Conflict。

### 3.3 解决方案：Padding (填充)

为了避免步长 32 的冲突，我们可以在每一行后面多加一个无用的空位 (Padding)。
*   原来：每行 32 个数。
*   现在：每行 **33** 个数。

**效果**：
*   Thread 0 访问 `[0, 0]` -> 地址 0 -> Bank 0。
*   Thread 1 访问 `[1, 0]` -> 地址 33 -> **Bank 1** (33 % 32 = 1)。
*   **完美错开**！大家又去了不同的窗口。

### 3.4 Triton 代码示例

Triton 的一大优势是它通常会自动处理这些复杂的 Bank Conflict 优化，让你可以专注于算法逻辑。

```python
import triton
import triton.language as tl

@triton.jit
def tiled_matmul(a_ptr, b_ptr, c_ptr, ...):
    # 1. 在 Shared Memory 中分配空间
    # 当你写 tl.load 时，Triton 编译器会在幕后分析：
    # "哦，这个 Block 需要反复读这块数据，我把它放到 Shared Memory 里吧"
    # "并且，我会自动安排好数据的布局（比如加 Padding），防止 Bank Conflict"
    a_tile = tl.load(a_ptr + offsets...)
    b_tile = tl.load(b_ptr + offsets...)
    
    # 2. 计算 (Compute)
    # 这里的 a_tile 和 b_tile 就在 Shared Memory 里
    # 线程们疯狂地反复读取它们，完全不消耗显存带宽
    accumulator += tl.dot(a_tile, b_tile)
```

## 4. 优化建议

1.  **能用 Shared Memory 就用**：任何需要被多次读取的数据，都应该先搬到 Shared Memory。
2.  **注意 Bank Conflict**：如果你在写 CUDA C++，需要小心设计数组索引（比如加 padding）。好消息是，**Triton 编译器通常会自动帮你处理 Bank Conflict**，这是 Triton 的一大卖点。
3.  **Occupancy 权衡**：Shared Memory 是有限的（比如 100KB）。如果你的 Kernel 每个 Block 都要用 50KB，那一个 SM 只能跑 2 个 Block。有时候为了提高并行度（Occupancy），需要省着点用 Shared Memory。

**下一篇预告**：Shared Memory 还要大家分着用，有没有什么是完全属于线程自己的私房钱？请看终章 **Registers**。
