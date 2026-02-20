import torch

# 1. 张量创建 (Tensor Creation)
print("--- 1. 张量创建 ---")

# 从列表创建张量
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"从列表创建:\n{x_data}")

# 创建全 0 张量
shape = (2, 3)
x_zeros = torch.zeros(shape)
print(f"全 0 张量:\n{x_zeros}")

# 创建全 1 张量
x_ones = torch.ones(shape)
print(f"全 1 张量:\n{x_ones}")

# 创建随机张量 (0-1 之间的随机数)
x_rand = torch.rand(shape)
print(f"随机张量:\n{x_rand}")

# 2. 张量形状与属性 (Shape & Attributes)
print("\n--- 2. 张量形状与属性 ---")
tensor = torch.rand(3, 4)
print(f"形状 (Shape): {tensor.shape}")
print(f"数据类型 (Datatype): {tensor.dtype}")
print(f"存储设备 (Device): {tensor.device}")

# 3. 基本运算 (Basic Math)
print("\n--- 3. 基本运算 ---")
t1 = torch.ones(2, 2)
t2 = torch.ones(2, 2) * 2 # 广播机制，每个元素乘以 2

print(f"t1:\n{t1}")
print(f"t2:\n{t2}")

# 加法
t3 = t1 + t2
print(f"加法 (t1 + t2):\n{t3}")

# 乘法 (逐元素相乘)
t4 = t1 * t2
print(f"逐元素乘法 (t1 * t2):\n{t4}")

# 矩阵乘法
# 为了演示矩阵乘法，我们需要兼容的形状
t5 = torch.matmul(t1, t2) 
print(f"矩阵乘法 (t1 @ t2):\n{t5}")

# 4. 移动到 GPU (Moving to GPU)
print("\n--- 4. 移动到 GPU ---")
# 检查是否有可用的 GPU (CUDA)
if torch.cuda.is_available():
    device = "cuda"
    print("GPU 可用！正在将张量移动到 GPU...")
    # 将张量移动到 GPU
    tensor_gpu = tensor.to(device)
    print(f"移动后的张量存储在: {tensor_gpu.device}")
    
    # 注意：如果要在 CPU 上使用 numpy()，需要先转回 CPU
    # print(tensor_gpu.numpy()) # 这会报错
    print(f"转回 CPU: {tensor_gpu.cpu().device}")
else:
    print("GPU 不可用，张量将保留在 CPU 上。")
