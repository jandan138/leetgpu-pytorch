import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton Naive Matrix Multiplication (朴素矩阵乘法)
# -----------------------------------------------------------------------------
# 这个版本是教学用的“最简”实现，没有使用分块 (Tiling) 和 Shared Memory。
# 它的性能远不如 cuBLAS，但逻辑最直观。
#
# 核心思想：
# 启动 M * K 个 Triton Program。
# 每个 Program 负责计算 C[m, k] 这一个元素的值。
# 计算公式：C[m, k] = sum(A[m, n] * B[n, k] for n in 0..N)
# -----------------------------------------------------------------------------

@triton.jit
def matrix_multiplication_kernel(
    # 1. 矩阵指针 (Pointers)
    a_ptr, b_ptr, c_ptr,
    # 2. 矩阵维度 (Dimensions)
    M, N, K,
    # 3. 步长 (Strides)
    # 告诉程序：在内存中，如果要移动一行或者一列，需要跳过多少个元素。
    # stride_am: A 的行步长 (通常等于 N)
    # stride_an: A 的列步长 (通常等于 1)
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_ck
):
    # --- 1. 确定当前线程负责的坐标 (m, k) ---
    # 我们启动的 Grid 是 (M, K)
    # axis=0 对应 M 维度，axis=1 对应 K 维度
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    # --- 2. 初始化累加器 ---
    # 我们要计算 C[pid_m, pid_k]，它是一个标量 (Scalar)。
    # 用一个 32 位浮点数寄存器来存它。
    # 只有当 pid_m < M 且 pid_k < K 时，才需要计算
    if pid_m >= M or pid_k >= K:
        return

    accumulator = 0.0
    
    # --- 3. 计算点积 (Dot Product) ---
    # C[m, k] = A 的第 m 行 与 B 的第 k 列 的点积
    # 也就是遍历 n 从 0 到 N-1
    for n in range(0, N):
        # 3.1 定位 A[pid_m, n] 在内存中的地址
        # 指针 = 基地址 + (行索引 * 行步长) + (列索引 * 列步长)
        # 注意：这里的 stride_am 和 stride_an 实际上是 PyTorch tensor 的 stride
        # 对于行优先矩阵，stride(0) 是行步长（N），stride(1) 是列步长（1）
        # 使用 tl.cast 确保索引计算使用 64 位整数，防止溢出
        offs_a = a_ptr + (pid_m * stride_am + n * stride_an).to(tl.int64)
        
        # 3.2 定位 B[n, pid_k] 在内存中的地址
        # 注意：这里 B 的行索引是 n，列索引是 pid_k
        offs_b = b_ptr + (n * stride_bn + pid_k * stride_bk).to(tl.int64)
        
        # 3.3 读取数据 (Load)
        # 从 Global Memory 读一个数到寄存器
        # 需要处理边界情况：如果当前行或列超出矩阵范围，则加载0
        # 这里的类型转换非常重要，确保指针计算是在 int64 下进行的
        # 虽然 Triton 通常会自动处理，但在某些情况下明确类型更安全
        if pid_m < M and n < N:
            val_a = tl.load(offs_a)
        else:
            val_a = 0.0
            
        if n < N and pid_k < K:
            val_b = tl.load(offs_b)
        else:
            val_b = 0.0
        
        # 3.4 累加 (Accumulate)
        accumulator += val_a * val_b
        
    # --- 4. 写回结果 (Store) ---
    # 算完了，把 accumulator 里的值写回 C[pid_m, pid_k]
    # 需要处理边界情况：如果 pid_m >= M 或 pid_k >= K，则不写入
    offs_c = c_ptr + (pid_m * stride_cm + pid_k * stride_ck)
    if pid_m < M and pid_k < K:
        tl.store(offs_c, accumulator)


# a, b, c are tensors on the GPU 
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    # 1. 准备步长参数 (Strides)
    # Tensor.stride() 返回的是 (行步长, 列步长)
    # 例如：对于一个 (2, 3) 的矩阵，如果它是行优先存储的：
    # stride(0) = 3 (下一行要跳 3 个数)
    # stride(1) = 1 (下一列要跳 1 个数)
    stride_am, stride_an = a.stride(0), a.stride(1)
    stride_bn, stride_bk = b.stride(0), b.stride(1)
    stride_cm, stride_ck = c.stride(0), c.stride(1)
    
    # 2. 定义 Grid (启动多少个线程)
    # 我们需要 M * K 个线程，每个线程算一个 C 的元素。
    # Grid 可以是 1D, 2D, 或 3D 的。这里用 2D 最直观。
    grid = (M, K)
    
    # 3. 启动 Kernel
    matrix_multiplication_kernel[grid](
        # 指针
        a, b, c,
        # 维度
        M, N, K,
        # 步长
        stride_am, stride_an,
        stride_bk, stride_bn,
        stride_cm, stride_ck
    )
