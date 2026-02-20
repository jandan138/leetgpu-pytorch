import torch

# 1. Autograd 简介 (Autograd Introduction)
print("--- 1. Autograd 简介 ---")
# Autograd 是 PyTorch 的自动微分引擎，用于自动计算梯度。
# 在深度学习中，我们需要计算损失函数相对于模型参数的梯度，以便通过梯度下降来更新参数。
# 设置 requires_grad=True 告诉 PyTorch 追踪对该张量的所有操作。

# 创建一个张量 x，我们需要计算关于 x 的梯度
x = torch.tensor(2.0, requires_grad=True)
print(f"输入张量 x: {x}")
print(f"x.requires_grad: {x.requires_grad}")

# 2. 前向传播 (Forward Pass)
print("\n--- 2. 前向传播 ---")
# 定义函数 y = x^2 + 2x + 1
# 当 x = 2 时, y = 2^2 + 2*2 + 1 = 4 + 4 + 1 = 9
y = x**2 + 2*x + 1

print(f"计算结果 y: {y}")
# y 是通过运算得到的，所以它有一个 grad_fn 属性，指向创建该张量的函数
print(f"y.grad_fn: {y.grad_fn}")

# 3. 反向传播 (Backward Pass)
print("\n--- 3. 反向传播 ---")
# 调用 .backward() 来计算梯度
# 这会计算 dy/dx 并将结果存储在 x.grad 中
y.backward()

print("已执行 y.backward()")

# 4. 获取梯度 (Accessing Gradients)
print("\n--- 4. 获取梯度 ---")
# dy/dx = d(x^2 + 2x + 1)/dx = 2x + 2
# 当 x = 2 时, dy/dx = 2*2 + 2 = 6
print(f"x 的梯度 (x.grad): {x.grad}")

# 验证计算是否正确
expected_grad = 2 * 2.0 + 2
print(f"预期梯度 (2x + 2): {expected_grad}")

if x.grad == expected_grad:
    print("梯度计算正确！")
else:
    print("梯度计算错误！")

# 注意：如果不清空梯度，再次调用 backward() 会累加梯度
# 在训练循环中，通常需要使用 optimizer.zero_grad() 清空梯度
