# 02. Matrix Multiplication (Triton Naive Implementation Plan)

用户要求按照指定的 Naive（朴素）框架实现矩阵乘法的 Triton 版本，并提供一份像 `01_vector_add` 那样极其详细的文档。

**用户提供的框架（Naive Kernel）特点：**

* **Grid**: `(M, K)` —— 每个 Program 处理 $C$ 矩阵中的**一个元素**。

* **Stride**: 手动传递了 stride 参数。

* **逻辑**: 既然每个 Program 处理一个元素，那么 Kernel 内部就是一个简单的 for 循环（从 0 到 N-1），计算点积。

**注意**：这与我之前生成的 `solution_triton.py`（使用了 Block Tiling 优化）完全不同。用户的这个版本是**教学用的入门版**，虽然性能不如 Tiling 版，但逻辑更直观，非常适合作为教程的第一步。

## 1. 任务拆解

### 1.1 更新 `solution_triton.py 这个文件要写超详细逐行的注释`

* 将现有的 Tiling 版本替换为用户指定的 Naive 版本。

* 实现 `matrix_multiplication_kernel`：

  * 获取 `pid_m` 和 `pid_k` (通过 `tl.program_id`)。

  * 初始化累加器 `acc = 0.0`。

  * 循环 `n` 从 0 到 `N`：

    * 计算 A 的地址：`a_ptr + pid_m * stride_am + n * stride_an`

    * 计算 B 的地址：`b_ptr + n * stride_bn + pid_k * stride_bk`

    * 加载并累加：`acc += load(a_ptr) * load(b_ptr)`

  * 计算 C 的地址并写回：`c_ptr + pid_m * stride_cm + pid_k * stride_ck`

### 1.2 更新 `README.md`

* 需要重写 Triton 部分，适配这个 Naive 版本的逻辑。

* **文档结构规划**：

  * **核心比喻**：从“包工头分砖块”（Tiling）退回到“每人算一个格子”（Naive）。

  * **代码逐行解析**：

    * 解释 `grid = (M, K)` 的含义（启动 $M \times K$ 个线程）。

    * 解释 `stride`（步长）的概念：为什么二维数组在内存里是一维的？如何通过 stride 找到 `(i, j)`？

    * 解释点积循环逻辑。

  * **性能分析（预告）**：为什么这种写法很慢？（引出 Global Memory 读写过多的问题，为以后讲 Tiling 做铺垫）。

### 1.3 验证

* 确保 `tests.py` 依然能跑通（逻辑正确性不变，只是速度变慢）。

## 2. 详细步骤

1. **修改代码**：重写 `d:\my_dev\leetgpu-pytorch\01_leetgpu_problems\easy\02_matrix_multiplication\solution_triton.py`。
2. **修改文档**：重写 `d:\my_dev\leetgpu-pytorch\01_leetgpu_problems\easy\02_matrix_multiplication\README.md`，添加大量注释和图解。
3. **Git 提交**：提交更改。

