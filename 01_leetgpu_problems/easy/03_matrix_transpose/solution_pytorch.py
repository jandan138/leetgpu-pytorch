import torch

# input, output 都是 GPU 上的 Tensor
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    """
    计算输入矩阵的转置。
    
    参数:
        input: 输入张量，维度为 (rows, cols)
        output: 输出张量，维度为 (cols, rows)
        rows: 输入矩阵的行数
        cols: 输入矩阵的列数
    """
    # PyTorch 的 transpose 函数 (或 .t()) 返回的是原张量的一个"视图" (View)，
    # 也就是它并没有真的在内存里搬运数据，只是改变了读取数据的方式（步长）。
    #
    # 但是题目要求我们把最终结果存储在 `output` 矩阵里，这就意味着我们需要
    # 把数据真真切切地从 input 搬运到 output 的内存空间里。
    # 
    # output.copy_(...) 会执行这个搬运操作。
    # 当我们把 input.t() 复制给 output 时，PyTorch 会自动处理内存布局，
    # 确保 output 里的数据是按照行优先 (Row-Major) 格式存储的。
    output.copy_(input.t())
