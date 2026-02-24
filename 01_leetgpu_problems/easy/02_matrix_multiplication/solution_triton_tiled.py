import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton Tiled Matrix Multiplication (分块矩阵乘法)
# -----------------------------------------------------------------------------
# 与朴素版（solution_triton.py）的核心区别：
#
# 朴素版：
#   每个 Program 只用 1 个 thread，计算 C 里的 1 个标量元素。
#   每次 tl.load 读 1 个 float（4 字节），内存利用率极低。
#
# 分块版（本文件）：
#   每个 Program 用 BM×BK 个 thread，计算 C 里的一个 BM×BK 的 tile。
#   每次 tl.load 读 BM×BN 或 BN×BK 个 float（整块加载），
#   再用 tl.dot 做 tile 级矩阵乘，充分利用 GPU 的向量化能力。
#
# 核心思想（分块算法）：
#   C[m:m+BM, k:k+BK]
#     = sum over n_start in range(0, N, BN):
#           A[m:m+BM, n:n+BN] @ B[n:n+BN, k:k+BK]
#
# 可配置的三个 tile 大小（均需是 2 的幂且 ≥ 16）：
#   BM：C tile 的行数 = 每个 Program 负责 C 的多少行
#   BK：C tile 的列数 = 每个 Program 负责 C 的多少列
#   BN：沿 N 维度的内层循环步长（A tile 列数 = B tile 行数）
# -----------------------------------------------------------------------------

@triton.jit
def matmul_tiled_kernel(
    # 矩阵指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度：A(M×N), B(N×K), C(M×K)
    M, N, K,
    # 步长：stride_xm = x.stride(0)（行步长），stride_xn = x.stride(1)（列步长）
    stride_am, stride_an,   # A 的行步长、列步长
    stride_bn, stride_bk,   # B 的行步长（沿 N）、列步长（沿 K）
    stride_cm, stride_ck,   # C 的行步长、列步长
    # tile 大小（编译时常量）
    BM: tl.constexpr,       # C tile 的行数
    BN: tl.constexpr,       # 内层循环步长（A tile 列数 = B tile 行数）
    BK: tl.constexpr,       # C tile 的列数
):
    """
    分块矩阵乘法 Kernel。

    Grid = (ceil(M/BM), ceil(K/BK))
    每个 Program 计算 C 的一个 BM×BK 的 tile。
    每个 Program 内有 BM×BK 个 thread（通过 tl.arange 的外积分布）。
    """

    # ── 1. 确定当前 Program 负责哪个 tile ─────────────────────────────────
    pid_m = tl.program_id(axis=0)   # tile 行编号
    pid_k = tl.program_id(axis=1)   # tile 列编号

    # tile 在 C 矩阵中的起始行、列
    m_start = pid_m * BM
    k_start = pid_k * BK

    # ── 2. 生成 tile 内的行列偏移 ──────────────────────────────────────────
    # offs_m: shape [BM]，当前 tile 覆盖的 C 行绝对索引
    # offs_k: shape [BK]，当前 tile 覆盖的 C 列绝对索引
    offs_m = m_start + tl.arange(0, BM)   # [BM]
    offs_k = k_start + tl.arange(0, BK)   # [BK]

    # ── 3. 初始化累加器 ────────────────────────────────────────────────────
    # acc 是 BM×BK 的矩阵，存在 BM×BK 个 thread 各自的寄存器里。
    # 朴素版：1 个 float 寄存器；分块版：BM×BK 个 float 寄存器（分散在各 thread）
    acc = tl.zeros((BM, BK), dtype=tl.float32)

    # ── 4. 内层循环：沿 N 维度分块累加 ────────────────────────────────────
    # C[m:m+BM, k:k+BK] = Σ A[m:m+BM, n:n+BN] @ B[n:n+BN, k:k+BK]
    for n_start in range(0, N, BN):
        offs_n = n_start + tl.arange(0, BN)   # [BN]

        # ── 4a. 加载 A 的 BM×BN tile ──────────────────────────────────────
        # a_ptrs[i, j] = a_ptr + offs_m[i] * stride_am + offs_n[j] * stride_an
        # 这是 BM×BN 的指针矩阵，对应 A[offs_m, offs_n]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        # mask 处理边界（当 M、N 不是 BM、BN 的整数倍时）
        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)  # shape: [BM, BN]

        # ── 4b. 加载 B 的 BN×BK tile ──────────────────────────────────────
        # b_ptrs[i, j] = b_ptr + offs_n[i] * stride_bn + offs_k[j] * stride_bk
        # 对应 B[offs_n, offs_k]
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)  # shape: [BN, BK]

        # ── 4c. tile 级矩阵乘加 ────────────────────────────────────────────
        # [BM, BN] × [BN, BK] → [BM, BK]
        # tl.dot 会尽可能使用 Tensor Core（需要 BN ≥ 16）
        # allow_tf32=False：使用严格 FP32，保证精度与朴素版一致
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)

    # ── 5. 将结果 tile 写回 C ──────────────────────────────────────────────
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    mask_c = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=mask_c)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    # ── 超参数（tile 大小）──────────────────────────────────────────────────
    # BM、BK：决定每个 Program 处理多大的输出 tile
    # BN：内层循环步长，越大每次加载越多数据，但寄存器压力也越大
    # 这三个值都必须是 2 的幂且 ≥ 16（tl.dot 的最低要求）
    BM = 32
    BN = 32
    BK = 32

    # ── Grid 计算 ────────────────────────────────────────────────────────────
    # 沿 M 方向需要 ceil(M/BM) 个 Program
    # 沿 K 方向需要 ceil(K/BK) 个 Program
    grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))

    # ── 启动 Kernel ──────────────────────────────────────────────────────────
    matmul_tiled_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),   # stride_am, stride_an
        b.stride(0), b.stride(1),   # stride_bn（B 行步长）, stride_bk（B 列步长）
        c.stride(0), c.stride(1),   # stride_cm, stride_ck
        BM=BM, BN=BN, BK=BK,
    )
