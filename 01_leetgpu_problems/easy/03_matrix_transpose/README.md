# Matrix Transpose

## Problem Description
Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix of dimensions `rows x cols`, the transpose will have dimensions `cols x rows`. All matrices are stored in row-major format.

## Implementation Requirements
- Use only native features (external libraries are not permitted).
- The `solve` function signature must remain unchanged.
- The final result must be stored in the matrix `output`.

## Example
**Input**: 2x3 matrix
```
[[1, 2, 3],
 [4, 5, 6]]
```

**Output**: 3x2 matrix
```
[[1, 4],
 [2, 5],
 [3, 6]]
```

## Constraints
- 1 <= rows, cols <= 8192
- Input matrix dimensions: `rows x cols`
- Output matrix dimensions: `cols x rows`
- Performance is measured with `cols = 6000`, `rows = 7000`

## Solutions

### Method 1: PyTorch Native
The solution is implemented in PyTorch using `torch.transpose` (or `.t()` alias). Since `torch.transpose` returns a view of the original tensor with swapped dimensions, we use `.copy_()` to copy the data into the `output` tensor, ensuring the result is stored in the correct memory location and format.

### Method 2: Triton Naive Implementation

This implementation demonstrates a basic, element-wise matrix transpose kernel using Triton. It maps each element of the matrix to a single thread (program instance) in the grid.

#### 1. Concept: Coordinate Mapping

The core idea of matrix transpose is swapping the row and column indices.
- If an element is at `(r, c)` in the **Input** matrix.
- It should be placed at `(c, r)` in the **Output** matrix.

#### 2. Grid Structure (The "Workforce")

We launch a 2D grid of threads (program instances) that matches the dimensions of the **Input** matrix.
- `grid = (rows, cols)`
- Each thread is responsible for moving **one single element**.

The thread ID `(pid_row, pid_col)` tells us which element this thread is responsible for:
- `pid_row`: The row index in the input matrix.
- `pid_col`: The column index in the input matrix.

#### 3. Memory Addressing (The "Map")

Since GPU memory is linear (1D), we need to convert 2D coordinates `(row, col)` into 1D memory offsets.

**Input Matrix (rows x cols)**:
- Stored in Row-Major order.
- To find element at `(pid_row, pid_col)`:
  `Offset = pid_row * stride_input_row + pid_col * stride_input_col`

**Output Matrix (cols x rows)**:
- Also stored in Row-Major order, but its logical dimensions are swapped.
- The element that was at `(pid_row, pid_col)` in Input belongs at `(pid_col, pid_row)` in Output.
- To find the location for `(pid_col, pid_row)` in the Output memory:
  `Offset = pid_col * stride_output_row + pid_row * stride_output_col`

#### 4. The Kernel Logic

```python
# 1. Identify identity
pid_row = tl.program_id(0)
pid_col = tl.program_id(1)

# 2. Read from Input
input_offset = pid_row * stride_ir + pid_col * stride_ic
val = tl.load(input_ptr + input_offset)

# 3. Write to Output (swapping indices)
output_offset = pid_col * stride_or + pid_row * stride_oc
tl.store(output_ptr + output_offset, val)
```

#### 5. Visualization

**Input (2x3)**:
```
(0,0) (0,1) (0,2)  --> Row 0
(1,0) (1,1) (1,2)  --> Row 1
```

**Output (3x2)**:
```
(0,0) (1,0)  --> Output Row 0 (Input Col 0)
(0,1) (1,1)  --> Output Row 1 (Input Col 1)
(0,2) (1,2)  --> Output Row 2 (Input Col 2)
```

Thread `(0, 1)` reads from Input `(0, 1)` and writes to Output `(1, 0)`.
Thread `(1, 2)` reads from Input `(1, 2)` and writes to Output `(2, 1)`.

This naive implementation is correct but not optimized for memory coalescing (especially for writes), which is critical for performance in transpose operations. Future versions can use shared memory tiling to improve this.
