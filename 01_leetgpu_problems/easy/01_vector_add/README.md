# 01. Vector Addition (向量加法)

## 题目描述

编写一个 GPU 程序，执行两个包含 32 位浮点数的向量的逐元素加法。程序应接受两个等长的输入向量，并生成一个包含它们和的输出向量。

### 实现要求

*   不允许使用外部库（在 LeetGPU 平台上，但在本地我们可以使用 PyTorch/Triton）。
*   `solve` 函数签名必须保持不变。
*   最终结果必须存储在向量 `C` 中。

### 示例 1

```
Input: A = [1.0, 2.0, 3.0, 4.0]
       B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
```

### 示例 2

```
Input: A = [1.5, 1.5, 1.5]
       B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]
```

### 约束条件

*   输入向量 `A` 和 `B` 具有相同的长度。
*   $1 \le N \le 100,000,000$
*   性能测试时 $N = 25,000,000$

## 解题思路

### 方法 1：PyTorch 原生实现 (High-Level)

在 PyTorch 中，最直观的写法是 `C = A + B`，但这会创建一个新的 Tensor，而不是在原地修改 `C`。为了满足题目要求（结果存储在 `C` 中）且提高效率，我们需要使用 In-place 操作。

**代码实现：**

```python
def solve_pytorch(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    # 方法 1：使用 copy_ (推荐，语义清晰)
    # C.copy_(A + B)
    
    # 方法 2：使用 out 参数 (最高效，零显存开销)
    torch.add(A, B, out=C)
```

**为什么 `C = A + B` 不行？（C++ 指针视角的深度解析）**

在 Python 中，变量名本质上是**指针**（引用）。

*   **`C = A + B` (Rebinding / 修改指针指向)**
    
    这相当于 C++ 中的：
    ```cpp
    // 假设 A, B, C 最初是指针
    float* A = 0x1000;
    float* B = 0x2000;
    float* C = 0x3000; // 外部传入的显存地址

    void solve(float* A, float* B, float* C) {
        // Python: C = A + B
        // 1. 分配新内存存结果
        float* temp = malloc(N * sizeof(float)); 
        // 2. 计算
        add(A, B, temp);
        // 3. 让局部变量 C 指向新内存
        C = temp; 
        // 此时 C 变成了 0x4000，但原来的 0x3000 没有任何变化！
        // 函数结束，temp 内存泄露（或被回收），外部的 C 还是指向 0x3000（空数据）。
    }
    ```

*   **`C[:] = A + B` 或 `C.copy_()` (In-place Write / 修改指针指向的内存)**

    这相当于 C++ 中的 `memcpy` 或解引用赋值：
    ```cpp
    void solve(float* A, float* B, float* C) {
        // Python: C[:] = ... 或 C.copy_(...)
        // 1. 分配新内存存结果 (A+B 会产生临时 Tensor)
        float* temp = malloc(N * sizeof(float));
        add(A, B, temp);
        
        // 2. 将 temp 的数据**复制**到 C 指向的地址
        memcpy(C, temp, N * sizeof(float)); 
        // 也就是： *C = *temp;
        
        // 结果写进了 0x3000，外部可见！
    }
    ```

*   **`torch.add(A, B, out=C)` (Zero-Copy / 极致优化)**

    这相当于直接把指针传给 Kernel：
    ```cpp
    void solve(float* A, float* B, float* C) {
        // 直接告诉 GPU：把结果写到地址 0x3000
        launch_kernel_add(A, B, C);
        // 没有 malloc temp，没有 memcpy。
    }
    ```

### 方法 2：Triton Kernel 实现 (Low-Level 深度解析)

Triton 是一个让你能用 Python 语法写出 GPU 高性能代码的“魔法编译器”。
对于初学者，我们把整个过程想象成一个**“大型施工队”**的协作过程。

#### 1. 核心心法：SPMD (单程序多数据)

想象你要处理一个长度 $N = 1000$ 的向量加法。
*   **CPU 做法**：派**一个人**（一个线程），从第 1 个数加到第 1000 个数。
*   **GPU 做法**：派**一千个人**（一千个线程），每个人只负责加**一个数**。
*   **Triton 做法**：派**几个人**（几个 Program/Block），每个人负责加**一小块数据**（比如每人负责 1024 个数）。

在 Triton 中，你写的那个 `vector_add_kernel` 函数，就是给**每一个工人**下达的“施工图纸”。
所有工人都拿着**同一份图纸**（Single Program），但是每个人负责处理**不同的数据块**（Multiple Data）。

#### 2. 代码详解

我们直接看你提供的代码框架，逐行拆解：

```python
import torch
import triton
import triton.language as tl

# --- Device Code (设备端代码：给工人看的图纸) ---
# @triton.jit 告诉编译器：这不是普通 Python 函数，请把它编译成 GPU 机器码！
@triton.jit 
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,      # 三个大数组在显存里的“起始地址”（门牌号）
    n_elements,               # 数组总长度 N
    BLOCK_SIZE: tl.constexpr  # 每个工人负责搬多少块砖 (常量)
): 
    # 1. 身份确认：我是第几个工人？
    # tl.program_id(0) 返回当前程序的 ID。
    # 比如启动了 10 个程序，这个 ID 就是 0, 1, 2 ... 9
    pid = tl.program_id(axis=0)
    
    # 2. 划定地盘：我负责处理哪一段数据？
    # 假设 BLOCK_SIZE = 1024
    # 工人 0 负责：0 ~ 1023
    # 工人 1 负责：1024 ~ 2047
    # ...
    # block_start 就是当前工人负责区域的“起始偏移量”
    block_start = pid * BLOCK_SIZE
    
    # 3. 生成索引：具体到每一个元素的地址偏移
    # tl.arange(0, BLOCK_SIZE) 生成一个序列：[0, 1, 2, ..., 1023]
    # offsets 变成：[start+0, start+1, ..., start+1023]
    # 这就是当前工人要处理的 1024 个数的“绝对索引”
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 4. 边界检查 (Masking)：防止越界
    # 如果 N = 1050，BLOCK_SIZE = 1024。
    # 工人 0 负责 0~1023 (全都在 N 以内)，mask 全是 True。
    # 工人 1 负责 1024~2047。但只有 1024~1049 是有效的。
    # 1050~2047 都是越界的！读了会报错或读到垃圾数据。
    # 所以我们需要一个 mask (掩码)，标记哪些索引是合法的。
    mask = offsets < n_elements
    
    # 5. 搬运数据 (Load)：从显存读到寄存器
    # load 指令：去 a_ptr + offsets 的地址抓数据。
    # mask=mask：只抓合法的，越界的不要动（Triton 会自动补 0 或安全处理）。
    # 此时 x 和 y 是位于 GPU 极速寄存器里的数据块。
    x = tl.load(a_ptr + offsets, mask=mask)
    y = tl.load(b_ptr + offsets, mask=mask)
    
    # 6. 计算 (Compute)
    # 这一步是在 GPU 核心里完成的，速度极快。
    output = x + y
    
    # 7. 写回数据 (Store)：把结果写回显存
    # 把 output 写到 c_ptr + offsets 的位置。
    # 同样要加 mask，防止写坏了别人的内存。
    tl.store(c_ptr + offsets, output, mask=mask)

# --- Host Code (主机端代码：包工头) ---
# a, b, c are tensors on the GPU 
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int): 
    # 1. 设定分块大小
    # 这是一个超参数，通常设为 128, 256, 512, 1024 等。
    # 越大可能吞吐越高，但由于寄存器限制，太大了也跑不动。1024 是个不错的默认值。
    BLOCK_SIZE = 1024 
    
    # 2. 计算需要雇多少个工人 (Grid Size)
    # 比如 N = 2500, BLOCK_SIZE = 1024
    # 工人 0: 0~1023
    # 工人 1: 1024~2047
    # 工人 2: 2048~3071 (处理剩下的 2498~2500)
    # 所以需要 ceil(2500 / 1024) = 3 个工人。
    # triton.cdiv 就是 ceil division (向上取整除法)。
    grid = (triton.cdiv(N, BLOCK_SIZE),) 
    
    # 3. 启动 Kernel (发号施令)
    # vector_add_kernel[grid]：指定启动网格大小
    # (a, b, c, N, BLOCK_SIZE)：传入参数
    # 注意：BLOCK_SIZE 必须作为关键字参数传给 kernel，因为它在 kernel 里被标记为 tl.constexpr
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
```

#### 3. 关键概念图解

**Grid (网格) 与 Block (块)**

```text
N = 2500, BLOCK_SIZE = 1024

+-----------------------+   +-----------------------+   +-----------------------+
| Program ID (pid) = 0  |   | Program ID (pid) = 1  |   | Program ID (pid) = 2  |
| 负责: 0 ~ 1023        |   | 负责: 1024 ~ 2047     |   | 负责: 2048 ~ 3071     |
+-----------------------+   +-----------------------+   +-----------+-----------+
            |                           |                           |
            v                           v                           v
      [计算满载]                  [计算满载]             [mask: 前 52 个有效]
                                                        [后面 972 个被 mask 掉]
```

**Mask (掩码) 的作用**

如果没有 mask，`pid=2` 的工人在读取 `offset=2500` 的位置时，就会读取到数组 `A` 之外的内存。
*   运气好：读到 0 或者乱码。
*   运气不好：**Segmentation Fault (非法内存访问)**，导致程序崩溃。
`mask` 就像一个筛子，只允许 `True` 位置的数据通过 `load` 和 `store` 接口。

#### 4. 为什么 `BLOCK_SIZE` 要写成 `tl.constexpr`?

`tl.constexpr` 意思是 **常量表达式 (Constant Expression)**。
这告诉 Triton 编译器：“嘿，这个 `BLOCK_SIZE` 在编译的时候就已经确定了，它不会变！”

*   **好处**：Triton 可以根据这个已知的常数，对代码进行疯狂优化（比如循环展开、寄存器分配）。
*   **代价**：如果你下次调用 kernel 时换了一个 `BLOCK_SIZE`（比如从 1024 换成 512），Triton 就必须**重新编译**整个 kernel。
*   **用法**：在 host 端调用时，必须以 `key=value` 的形式传入（如 `BLOCK_SIZE=1024`），不能作为普通位置参数。

### 运行与测试

可以直接运行 `solution.py` 进行验证和性能测试：

```bash
python solution.py
```
