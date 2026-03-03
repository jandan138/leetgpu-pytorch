import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def conv1d_kernel(
        # Pointers to tensors
        x_ptr,      # pointer to input signal x, shape (N,)
        w_ptr,      # pointer to convolution kernel w, shape (K,)
        y_ptr,      # pointer to output tensor y, shape (N-K+1,)
        # Scalar dimensions
        N,          # length of input signal (runtime int)
        K,          # length of convolution kernel (runtime int)
        output_len, # N - K + 1, used for boundary masking
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,  # number of output elements each program handles
    ):
        """Triton kernel for 1D valid convolution.

        Each program instance handles a contiguous block of BLOCK_SIZE output elements.
        For each output index i in its block, the program accumulates:
            y[i] = sum_{j=0}^{K-1} x[i + j] * w[j]

        Grid: 1D, ceil(output_len / BLOCK_SIZE) programs along axis 0.
        """
        # 1. Determine which output indices this program is responsible for.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        # offsets[k] = index of the k-th output element this program handles
        offsets = block_start + tl.arange(0, BLOCK_SIZE)  # shape: (BLOCK_SIZE,)

        # Mask to guard the last (potentially incomplete) block.
        mask = offsets < output_len

        # 2. Accumulate the dot product over the kernel dimension.
        #    acc[k] = sum_{j=0}^{K-1} x[offsets[k] + j] * w[j]
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for j in tl.range(0, K):
            # Load K elements from x starting at x[offsets + j].
            # The mask ensures we don't read past x when offsets are out of range.
            x_vals = tl.load(x_ptr + offsets + j, mask=mask, other=0.0)
            # Load the scalar kernel weight w[j].
            w_j = tl.load(w_ptr + j)
            # Fused multiply-add: accumulate contribution of kernel tap j.
            acc += x_vals * w_j

        # 3. Write the accumulated results to y.
        tl.store(y_ptr + offsets, acc, mask=mask)


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
        output_len = N - K + 1
        BLOCK_SIZE = 1024

        # 1D grid: one program per block of BLOCK_SIZE output elements.
        grid = (triton.cdiv(output_len, BLOCK_SIZE),)

        conv1d_kernel[grid](
            x, w, y,
            N, K, output_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )

else:
    def solve(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, N: int, K: int) -> None:
        raise RuntimeError("Triton is not installed.")
