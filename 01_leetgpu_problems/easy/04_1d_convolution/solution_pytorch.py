import torch
import torch.nn.functional as F


def solve(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, N: int, K: int) -> None:
    """Compute a 1D valid convolution of signal x with kernel w, writing results into y.

    Valid mode: output length = N - K + 1, no zero-padding applied.
    Formula: y[i] = sum_{j=0}^{K-1} x[i+j] * w[j], for i = 0, ..., N-K

    Args:
        x: Input signal tensor of shape (N,), dtype float32, on CUDA.
        w: Convolution kernel tensor of shape (K,), dtype float32, on CUDA.
        y: Pre-allocated output tensor of shape (N - K + 1,), dtype float32, on CUDA.
        N: Length of the input signal.
        K: Length of the convolution kernel.
    """
    # -------------------------------------------------------------------------
    # 方法 1：F.conv1d（推荐，最惯用）
    #
    # F.conv1d 期望输入形状为 (batch, in_channels, length)，因此需要先将
    # 1-D 的 x 和 w 通过 view 升维为 3-D，调用后再用 view(-1) 降回 1-D。
    #
    # padding=0 对应 valid 模式——与题目定义完全一致。
    #
    # 注意：conv1d 的 weight 形状为 (out_channels, in_channels/groups, kernel_size)，
    # 此处为 (1, 1, K)。PyTorch 的卷积定义与信号处理惯例相同（滑动点积），
    # 不会翻转卷积核，因此与题目公式 y[i] = Σ x[i+j]*w[j] 完全一致。
    # -------------------------------------------------------------------------
    x_3d = x.view(1, 1, N)    # (1, 1, N)
    w_3d = w.view(1, 1, K)    # (1, 1, K)  — weight: (out_ch, in_ch, kernel_size)
    y.copy_(F.conv1d(x_3d, w_3d, padding=0).view(-1))

    # -------------------------------------------------------------------------
    # 方法 2：unfold + 矩阵向量乘法
    #
    # torch.Tensor.unfold(dimension, size, step) 将输入"展开"为滑动窗口矩阵，
    # 形状为 (N-K+1, K)，每行是 x[i:i+K]。
    # 然后与卷积核 w (shape K) 做矩阵向量乘法，等价于逐窗口求点积。
    #
    # 优点：纯张量运算，便于理解卷积的本质（滑动窗口 + 点积）。
    # 缺点：unfold 产生一个 (N-K+1, K) 的中间张量，内存占用比 conv1d 高。
    #
    # x_windows = x.unfold(0, K, 1)     # (N-K+1, K)
    # y.copy_(x_windows @ w)             # (N-K+1,)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 方法 3：纯 Python 循环（仅用于教学对比，不适合大规模数据）
    #
    # 逐元素实现公式 y[i] = Σ_{j=0}^{K-1} x[i+j] * w[j]。
    # 在 CPU 上容易理解，但在 GPU 上因为没有并行化而极慢。
    #
    # for i in range(N - K + 1):
    #     y[i] = (x[i : i + K] * w).sum()
    # -------------------------------------------------------------------------
