import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    rows, cols,
    stride_ir, stride_ic,  # 输入矩阵的步长 (行, 列)
    stride_or, stride_oc   # 输出矩阵的步长 (行, 列)
):
    """
    Triton 朴素版矩阵转置 Kernel。
    核心思想：每个 Program (线程) 负责搬运矩阵中的**一个元素**。
    Grid 形状为 (rows, cols)，对应输入矩阵的每个坐标。
    """
    # 1. 确认身份：我是谁？
    # 我们启动的 grid 是 (rows, cols) 的二维网格。
    # axis=0 对应输入矩阵的“行索引”，axis=1 对应输入矩阵的“列索引”。
    pid_row = tl.program_id(axis=0)  # 我负责输入矩阵的第几行
    pid_col = tl.program_id(axis=1)  # 我负责输入矩阵的第几列
    
    # 2. 边界检查 (Boundary Check)
    # 虽然我们启动的 grid 正好是 (rows, cols)，但在使用分块（Block）时，
    # 线程数可能会多于元素数，所以加上边界检查是个好习惯。
    # 这里的任务是：只有当我在矩阵范围内时，才干活。
    if pid_row < rows and pid_col < cols:
        
        # 3. 计算输入矩阵中的地址 (Read Address)
        # 输入矩阵是 (rows x cols) 的，行优先存储。
        # 我要读取的元素坐标是 (pid_row, pid_col)。
        # 内存地址 = 基地址 + 行号 * 行步长 + 列号 * 列步长
        input_offset = pid_row * stride_ir + pid_col * stride_ic
        val = tl.load(input_ptr + input_offset)
        
        # 4. 计算输出矩阵中的地址 (Write Address)
        # 输出矩阵是 (cols x rows) 的，也是行优先存储。
        # 矩阵转置的定义是：输入中 (r, c) 的元素，要放到输出中 (c, r) 的位置。
        # 所以对于输出矩阵：
        #   - 它的“逻辑行号”是我的 `pid_col` (来自输入的列)
        #   - 它的“逻辑列号”是我的 `pid_row` (来自输入的行)
        # 内存地址 = 基地址 + 逻辑行号 * 输出行步长 + 逻辑列号 * 输出列步长
        output_offset = pid_col * stride_or + pid_row * stride_oc
        
        # 5. 写入数据 (Store)
        tl.store(output_ptr + output_offset, val)

# input, output 都是 GPU 上的 Tensor
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # 获取输入矩阵的步长 (rows x cols)
    # 对于行优先布局：stride(0) 是行步长 (=cols), stride(1) 是列步长 (=1)
    stride_ir, stride_ic = input.stride(0), input.stride(1)
    
    # 获取输出矩阵的步长 (cols x rows)
    # 注意输出矩阵的形状变了，所以它的行步长通常等于 rows
    stride_or, stride_oc = output.stride(0), output.stride(1)

    # 定义 Grid (启动多少个线程)
    # 我们使用最简单的映射策略：
    # 每一个元素对应一个线程。输入矩阵有 rows * cols 个元素。
    # 所以我们启动一个 (rows, cols) 的二维 Grid。
    grid = (rows, cols)
    
    # 启动 Kernel
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    )
