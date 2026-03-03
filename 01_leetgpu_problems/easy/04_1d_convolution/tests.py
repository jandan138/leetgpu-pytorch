import torch
import torch.nn.functional as F
import time
import solution_pytorch

try:
    import solution_triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def reference_conv1d(x: torch.Tensor, w: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Reference 1D valid convolution via F.conv1d.

    F.conv1d with padding=0 implements the same formula:
        y[i] = sum_{j=0}^{K-1} x[i+j] * w[j]
    No kernel flip is applied, which matches the problem definition exactly.
    """
    x_3d = x.view(1, 1, N)   # (batch=1, in_ch=1, length=N)
    w_3d = w.view(1, 1, K)   # (out_ch=1, in_ch=1, kernel_size=K)
    return F.conv1d(x_3d, w_3d, padding=0).view(-1)  # (N-K+1,)


def test_1d_convolution():
    print("Running 1D Convolution Test...")

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("Warning: CUDA not available. Skipping GPU tests.")
        return

    if not HAS_TRITON:
        print("Warning: Triton not found, skipping Triton tests")

    # ------------------------------------------------------------------
    # Correctness checks — three (N, K) pairs covering edge cases
    # ------------------------------------------------------------------
    test_cases = [
        (128,   7,  "small  (N=128,   K=7)"),
        (1024,  16, "medium (N=1024,  K=16)"),
        (98432, 32, "large  (N=98432, K=32)"),
    ]

    for N, K, label in test_cases:
        x = torch.rand(N, device=device, dtype=torch.float32)
        w = torch.rand(K, device=device, dtype=torch.float32)
        output_len = N - K + 1

        expected = reference_conv1d(x, w, N, K)

        # --- PyTorch ---
        y_pytorch = torch.zeros(output_len, device=device, dtype=torch.float32)
        solution_pytorch.solve(x, w, y_pytorch, N, K)

        if torch.allclose(y_pytorch, expected, atol=1e-2, rtol=1e-2):
            print(f"1. PyTorch correctness PASSED  [{label}]")
        else:
            max_diff = (y_pytorch - expected).abs().max().item()
            print(f"1. PyTorch correctness FAILED  [{label}]  max_diff={max_diff:.6f}")

        # --- Triton ---
        if HAS_TRITON:
            y_triton = torch.zeros(output_len, device=device, dtype=torch.float32)
            try:
                solution_triton.solve(x, w, y_triton, N, K)
                if torch.allclose(y_triton, expected, atol=1e-2, rtol=1e-2):
                    print(f"2. Triton  correctness PASSED  [{label}]")
                else:
                    max_diff = (y_triton - expected).abs().max().item()
                    print(f"2. Triton  correctness FAILED  [{label}]  max_diff={max_diff:.6f}")
            except Exception as e:
                print(f"2. Triton  execution  FAILED  [{label}]: {e}")

    # ------------------------------------------------------------------
    # Benchmark — large signal to stress-test throughput
    # ------------------------------------------------------------------
    N_bench = 25_000_000
    K_bench = 64
    output_len_bench = N_bench - K_bench + 1

    print(f"\n--- Benchmark (N={N_bench:,}, K={K_bench}) ---")
    try:
        x_b = torch.rand(N_bench, device=device, dtype=torch.float32)
        w_b = torch.rand(K_bench, device=device, dtype=torch.float32)
        y_b = torch.zeros(output_len_bench, device=device, dtype=torch.float32)

        # Warmup
        solution_pytorch.solve(x_b, w_b, y_b, N_bench, K_bench)
        if HAS_TRITON:
            solution_triton.solve(x_b, w_b, y_b, N_bench, K_bench)
        torch.cuda.synchronize()

        # Time PyTorch
        start = time.time()
        for _ in range(10):
            solution_pytorch.solve(x_b, w_b, y_b, N_bench, K_bench)
        torch.cuda.synchronize()
        print(f"PyTorch Time: {(time.time() - start) / 10 * 1000:.2f} ms")

        # Time Triton
        if HAS_TRITON:
            start = time.time()
            for _ in range(10):
                solution_triton.solve(x_b, w_b, y_b, N_bench, K_bench)
            torch.cuda.synchronize()
            print(f"Triton Time:  {(time.time() - start) / 10 * 1000:.2f} ms")

    except RuntimeError as e:
        print(f"Benchmark failed (possibly OOM): {e}")


if __name__ == "__main__":
    test_1d_convolution()
