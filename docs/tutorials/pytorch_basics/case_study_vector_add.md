# 实战分析：从 Python 到 CUDA 的性能飞跃 —— 以向量加法为例

理论讲了这么多，代码到底差在哪里？我们通过一个最简单的 **Vector Add (向量加法)** 案例，一步步拆解性能是怎么被“榨”出来的。

## 1. 任务定义
计算两个长度为 $N$ 的向量 $A$ 和 $B$ 的和：
$$ C[i] = A[i] + B[i] $$
其中 $N = 10^7$ (一千万)。

---

## 2. 第一层：纯 Python 循环 (极慢)
```python
def add_python(a, b):
    c = [0.0] * len(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return c
```
*   **性能瓶颈**：
    *   **GIL (全局解释器锁)**：Python 解释器在任何时刻只能执行一个字节码。
    *   **动态类型检查**：每次 `a[i] + b[i]`，Python 都要检查 `a[i]` 是不是数字，能不能相加。
    *   **无法利用 SIMD**：CPU 有 AVX 指令集可以一次加 8 个数，但 Python 只能一个个加。

---

## 3. 第二层：PyTorch CPU (快)
```python
import torch
c = a + b  # a, b 是 CPU 上的 Tensor
```
*   **发生了什么**：
    *   PyTorch 底层调用了 C++ 写的 `ATen` 库。
    *   **SIMD (Single Instruction, Multiple Data)**：利用 CPU 的 AVX2/AVX-512 指令集，一次加 8 个 float32。
    *   **多线程 (OpenMP)**：利用多核 CPU 并行计算。
*   **瓶颈**：**内存带宽 (Memory Bandwidth)**。CPU 内存带宽（约 50GB/s）远低于 GPU。

---

## 4. 第三层：PyTorch GPU (极快)
```python
a_gpu = a.cuda()
b_gpu = b.cuda()
c_gpu = a_gpu + b_gpu
```
*   **发生了什么**：
    *   **Kernel Launch**：PyTorch 启动了一个 CUDA Kernel。
    *   **海量并行**：GPU 上几千个 CUDA Core 同时开工。
*   **瓶颈**：
    *   **Kernel Launch Overhead**：启动 Kernel 本身需要几微秒。对于小数组，可能比计算还慢。
    *   **Global Memory Bandwidth**：虽然 GPU 显存带宽很高（1TB/s），但如果只是简单的 $A+B$，算力远过剩于带宽，此时是 **Memory Bound**。

---

## 5. 第四层：手写 Triton/CUDA (榨干硬件)
为了极致优化，我们需要自己写 Kernel。

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 计算这一块数据的索引
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 2. 批量加载 (Coalesced Load) -> 利用 Shared Memory 缓存
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 3. 计算 (Compute) -> 利用寄存器
    output = x + y
    
    # 4. 批量存储 (Coalesced Store)
    tl.store(output_ptr + offsets, output, mask=mask)
```
*   **核心优化点**：
    *   **Tiling (分块)**：把 $N$ 个元素切成很多小块（比如 `BLOCK_SIZE=1024`）。
    *   **Coalesced Access (合并访问)**：一个 Warp 的 32 个线程，一次性读连续的 128 字节，充分利用显存带宽。
    *   **Shared Memory / Cache**：数据一旦读进来，就放在离核心最近的地方。

---

## 6. 总结：性能阶梯

| 实现方式 | 耗时 (假设) | 瓶颈 | 评价 |
| :--- | :--- | :--- | :--- |
| **Python 循环** | 10 s | 解释器开销 | 玩具 |
| **PyTorch CPU** | 0.1 s | CPU 算力/带宽 | 生产可用 |
| **PyTorch GPU** | 0.005 s | Kernel 启动开销 | 深度学习标准 |
| **Triton/CUDA** | 0.004 s | 显存带宽极限 | **算子开发专家** |

当你深入到 Triton 这一层，你就不再是写 Python 代码，而是在**指挥硬件的数据洪流**。
