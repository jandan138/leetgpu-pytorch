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

### 方法 2：Triton Kernel 实现 (Low-Level)

Triton 允许我们编写类似 CUDA 的内核，但使用 Python 语法。

**核心思想：**
1.  **分块 (Tiling)**：将大向量切分成很多小块（比如每块 1024 个元素），每个 Triton Program（类似 CUDA Block）处理一块。
2.  **Grid 计算**：根据向量长度 $N$ 和块大小 `BLOCK_SIZE` 计算需要启动多少个 Program。
3.  **Masking (掩码)**：处理边界情况。当 $N$ 不是 `BLOCK_SIZE` 的倍数时，最后一个 Block 会有多余的线程，需要用 `mask` 防止越界访问。

**代码逻辑：**

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements

# 加载数据 (Load)
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)

# 计算 (Compute)
output = x + y

# 写回数据 (Store)
tl.store(output_ptr + offsets, output, mask=mask)
```

## 运行与测试

可以直接运行 `solution.py` 进行验证和性能测试：

```bash
python solution.py
```
