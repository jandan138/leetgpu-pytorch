import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not found. Skipping Triton tests.")

# -----------------------------------------------------------------------------
# 1. PyTorch Solution
# -----------------------------------------------------------------------------

def solve_pytorch(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    """
    Vector Addition using PyTorch.
    The result is stored in C.
    """
    # -------------------------------------------------------------------------
    # 方法 1：In-place copy (推荐)
    # 说明：先计算 A + B 产生一个临时张量，然后将内容复制到 C 中。
    # 优点：代码清晰，符合 Python 直觉。
    # 缺点：会产生一个临时的 (A+B) 显存占用，如果 Tensor 极大可能会 OOM。
    # 是否走 GPU：是（前提是 A, B 在 GPU 上）。
    # -------------------------------------------------------------------------
    C.copy_(A + B)
    
    # -------------------------------------------------------------------------
    # 方法 2：torch.add with out=C (最高效)
    # 说明：直接将加法结果写入 C 的内存地址。
    # 优点：零显存开销（Zero Memory Overhead），不产生临时 Tensor。
    # 是否走 GPU：是。
    # 代码：
    # torch.add(A, B, out=C)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 方法 3：In-place 加法 (C += ...)
    # 说明：先将 A 复制给 C，然后把 B 加到 C 上。
    # 警告：题目要求 C = A + B。如果 C 初始值不是 A，这种写法是错的。
    # 除非写法是：C.copy_(A); C.add_(B)
    # 是否走 GPU：是。
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # 方法 4：Kernel 融合 (JIT / torch.compile)
    # 说明：如果是复杂的 (A + B) * D，PyTorch 2.0+ 会自动融合算子。
    # 对于简单的 A + B，效果和方法 2 差不多。
    # ------------------------------------------------------------------------- 

# -----------------------------------------------------------------------------
# 2. Triton Solution
# -----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def add_kernel(
        x_ptr,  # Pointer to first input vector
        y_ptr,  # Pointer to second input vector
        output_ptr, # Pointer to output vector
        n_elements, # Size of the vector
        BLOCK_SIZE: tl.constexpr, # Number of elements each program should process
    ):
        # There are multiple 'programs' processing different data. We identify which program
        # we are here:
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        
        # This program will process inputs that are offset from the initial data.
        # For instance, if you had a vector of length 256 and block_size of 64, the programs
        # would each access the elements [0:64, 64:128, 128:192, 192:256].
        # Note that offsets is a list of pointers:
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Create a mask to guard memory access. Not all blocks have the same size.
        mask = offsets < n_elements
        
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        
        # Write x + y back to DRAM
        tl.store(output_ptr + offsets, output, mask=mask)

    def solve_triton(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
        """
        Vector Addition using Triton.
        The result is stored in C.
        """
        # 1. Define the grid.
        # The grid is a tuple of (number_of_programs_x, number_of_programs_y, number_of_programs_z).
        # Here we use a 1D grid.
        # We need enough programs to cover all N elements.
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
        
        # 2. Launch the kernel.
        add_kernel[grid](
            A, B, C,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
else:
    def solve_triton(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
        print("Triton not installed. Skipping Triton execution.")
        pass

# -----------------------------------------------------------------------------
# 3. Verification & Benchmark
# -----------------------------------------------------------------------------

def test_vector_add():
    print(f"Running Vector Addition Test...")
    
    # Setup
    torch.manual_seed(0)
    size = 98432 # Arbitrary size, not a multiple of 1024 to test masking
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: CUDA not available. Triton requires CUDA. Skipping Triton test.")
        return

    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output_torch = torch.empty_like(x)
    output_triton = torch.empty_like(x)
    
    # Run PyTorch
    print(f"1. Running PyTorch implementation...")
    solve_pytorch(x, y, output_torch, size)
    
    # Run Triton
    print(f"2. Running Triton implementation...")
    solve_triton(x, y, output_triton, size)
    
    # Verify
    if torch.allclose(output_torch, output_triton):
        print("✅ Correctness Check Passed: PyTorch and Triton results match!")
    else:
        print("❌ Correctness Check Failed!")
        print(f"Max difference: {torch.max(torch.abs(output_torch - output_triton))}")
        
    # Simple Benchmark
    print("\n--- Simple Benchmark (Size: 25M elements) ---")
    size_bench = 25_000_000
    x_bench = torch.rand(size_bench, device=device)
    y_bench = torch.rand(size_bench, device=device)
    out_bench = torch.empty_like(x_bench)
    
    # Warmup
    solve_pytorch(x_bench, y_bench, out_bench, size_bench)
    solve_triton(x_bench, y_bench, out_bench, size_bench)
    torch.cuda.synchronize()
    
    import time
    
    # Measure PyTorch
    start = time.time()
    for _ in range(10):
        solve_pytorch(x_bench, y_bench, out_bench, size_bench)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10 * 1000 # ms
    print(f"PyTorch Time: {pytorch_time:.3f} ms")
    
    # Measure Triton
    start = time.time()
    for _ in range(10):
        solve_triton(x_bench, y_bench, out_bench, size_bench)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 10 * 1000 # ms
    print(f"Triton Time:  {triton_time:.3f} ms")

if __name__ == "__main__":
    test_vector_add()
