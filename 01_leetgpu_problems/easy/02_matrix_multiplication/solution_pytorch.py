import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    """
    Matrix Multiplication using PyTorch.
    The result is stored in C.
    """
    # -------------------------------------------------------------------------
    # 方法 1：In-place copy (推荐)
    # 说明：先计算 A @ B 产生一个临时张量，然后将内容复制到 C 中。
    # 优点：代码清晰，符合 Python 直觉。
    # 缺点：会产生一个临时的 (A @ B) 显存占用，如果 Tensor 极大可能会 OOM。
    # 是否走 GPU：是（前提是 A, B 在 GPU 上）。
    # -------------------------------------------------------------------------
    # C.copy_(A @ B)

    # -------------------------------------------------------------------------
    # 方法 2：torch.mm with out=C (最高效)
    # 说明：直接将乘法结果写入 C 的内存地址。
    # 优点：零显存开销（Zero Memory Overhead），不产生临时 Tensor。
    # 是否走 GPU：是。
    # 代码：
    torch.mm(A, B, out=C)
    # -------------------------------------------------------------------------
