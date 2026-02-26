import torch
import solution_triton_tiled

def test_tiled_matmul():
    print("Running Tiled Matrix Multiplication Test...")
    
    # 1. Setup
    M, N, K = 4096, 4096, 4096
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: Triton not found or GPU not available, skipping Triton tests")
        return

    torch.manual_seed(0)
    
    # Create random matrices
    # Triton expects contiguous memory for optimal performance, 
    # but our kernel should handle strides correctly now.
    A = torch.randn(M, N, device=device)
    B = torch.randn(N, K, device=device)
    C_triton = torch.zeros(M, K, device=device)
    
    # 2. Run PyTorch (Reference)
    print(f"1. Running PyTorch implementation (M={M}, N={N}, K={K})...")
    # Warmup
    torch.mm(A, B)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    C_pytorch = torch.mm(A, B)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event)
    
    # 3. Run Triton Tiled Implementation
    print(f"2. Running Triton Tiled implementation (M={M}, N={N}, K={K})...")
    # Warmup
    solution_triton_tiled.solve(A, B, C_triton, M, N, K)
    
    start_event.record()
    solution_triton_tiled.solve(A, B, C_triton, M, N, K)
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event)
    
    # 4. Verify Correctness
    try:
        if torch.allclose(C_pytorch, C_triton, atol=1e-2, rtol=1e-2):
            print("✅ Correctness Check Passed!")
        else:
            print("❌ Correctness Check Failed!")
            diff = torch.abs(C_pytorch - C_triton).max()
            print(f"Max difference: {diff}")
    except Exception as e:
        print(f"❌ Triton execution failed: {e}")
        
    print(f"\n--- Benchmark (M={M}, N={N}, K={K}) ---")
    print(f"PyTorch Time: {pytorch_time:.2f} ms")
    print(f"Triton Time:  {triton_time:.2f} ms")

if __name__ == "__main__":
    test_tiled_matmul()
