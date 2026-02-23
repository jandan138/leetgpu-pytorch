import torch
import time
import solution_pytorch
try:
    import solution_triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def test_matmul():
    print("Running Matrix Multiplication Test...")
    
    # 1. Setup
    torch.manual_seed(0)
    # Use smaller size for correctness check
    M, N, K = 128, 256, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: CUDA not available. Skipping GPU tests.")
        return

    if not HAS_TRITON:
        print("Warning: Triton not found, skipping Triton tests")

    # Create random matrices
    A = torch.randn(M, N, device=device)
    B = torch.randn(N, K, device=device)
    C_pytorch = torch.zeros(M, K, device=device)
    C_triton = torch.zeros(M, K, device=device)
    
    # 2. Run PyTorch
    print(f"1. Running PyTorch implementation (M={M}, N={N}, K={K})...")
    solution_pytorch.solve(A, B, C_pytorch, M, N, K)
    
    # 3. Run Triton
    if HAS_TRITON:
        print(f"2. Running Triton implementation (M={M}, N={N}, K={K})...")
        try:
            solution_triton.solve(A, B, C_triton, M, N, K)
            
            # Verify
            if torch.allclose(C_pytorch, C_triton, atol=1e-2, rtol=1e-2):
                print("✅ Correctness Check Passed!")
            else:
                print("❌ Correctness Check Failed!")
                diff = torch.abs(C_pytorch - C_triton).max()
                print(f"Max difference: {diff}")
        except Exception as e:
            print(f"❌ Triton execution failed: {e}")
    
    # 4. Benchmark
    print("\n--- Benchmark (M=4096, N=4096, K=4096) ---")
    M, N, K = 4096, 4096, 4096
    try:
        A = torch.randn(M, N, device=device)
        B = torch.randn(N, K, device=device)
        C = torch.zeros(M, K, device=device)
        
        # Warmup
        solution_pytorch.solve(A, B, C, M, N, K)
        if HAS_TRITON:
            solution_triton.solve(A, B, C, M, N, K)
        torch.cuda.synchronize()
        
        # Measure PyTorch
        start = time.time()
        for _ in range(5):
            solution_pytorch.solve(A, B, C, M, N, K)
        torch.cuda.synchronize()
        print(f"PyTorch Time: {(time.time() - start)/5 * 1000:.2f} ms")
        
        # Measure Triton
        if HAS_TRITON:
            start = time.time()
            for _ in range(5):
                solution_triton.solve(A, B, C, M, N, K)
            torch.cuda.synchronize()
            print(f"Triton Time:  {(time.time() - start)/5 * 1000:.2f} ms")
    except RuntimeError as e:
        print(f"Benchmark failed (possibly OOM): {e}")

if __name__ == "__main__":
    test_matmul()
