import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    rows, cols,
    stride_ir, stride_ic,  # input strides (row, col)
    stride_or, stride_oc   # output strides (row, col)
):
    """
    Naive kernel for matrix transpose.
    Each program instance handles one element (row, col) of the matrix.
    grid = (rows, cols)
    """
    # 1. Identify which element this thread is responsible for
    # We launched a grid of (rows, cols)
    pid_row = tl.program_id(axis=0)  # Corresponds to row index of input
    pid_col = tl.program_id(axis=1)  # Corresponds to col index of input
    
    # 2. Boundary check (optional if grid is exact, but good practice)
    # Since we launch exactly rows * cols threads, we might not strictly need this if grid matches
    # But if we use blocks later, this is crucial. For naive 1-thread-per-element:
    if pid_row < rows and pid_col < cols:
        
        # 3. Calculate address in Input Matrix
        # Input is (rows x cols), we want element at (pid_row, pid_col)
        # Address = base + row_idx * stride_row + col_idx * stride_col
        input_offset = pid_row * stride_ir + pid_col * stride_ic
        val = tl.load(input_ptr + input_offset)
        
        # 4. Calculate address in Output Matrix
        # Output is (cols x rows) - transpose dimensions
        # The element at Input(r, c) should go to Output(c, r)
        # So for the output matrix:
        #   - Its 'row' index is `pid_col` (from input's column)
        #   - Its 'col' index is `pid_row` (from input's row)
        output_offset = pid_col * stride_or + pid_row * stride_oc
        
        # 5. Store the value
        tl.store(output_ptr + output_offset, val)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # Strides for input matrix (rows x cols)
    # For row-major layout: stride_row = cols, stride_col = 1
    # However, it's safer to get them from the tensor itself in case it's non-contiguous
    stride_ir, stride_ic = input.stride(0), input.stride(1)
    
    # Strides for output matrix (cols x rows)
    # For row-major layout: stride_row = rows, stride_col = 1
    stride_or, stride_oc = output.stride(0), output.stride(1)

    # Launch configuration
    # We use a simple 2D grid where each program handles one element
    # grid = (rows, cols)
    grid = (rows, cols)
    
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    )
