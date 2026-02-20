import torch

print("--- GPU 检查工具 ---")

# 检查 CUDA 是否可用
is_cuda_available = torch.cuda.is_available()

if is_cuda_available:
    print("✅ CUDA (GPU) 可用！")
    
    # 获取 GPU 设备数量
    device_count = torch.cuda.device_count()
    print(f"检测到的 GPU 数量: {device_count}")
    
    # 打印每个 GPU 的名称
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
        
    print(f"当前默认设备: {torch.cuda.current_device()}")
    
else:
    print("❌ CUDA (GPU) 不可用。")
    print("不用担心！您仍然可以使用 CPU 运行 PyTorch 代码。")
    print("如果您有 NVIDIA 显卡，请检查是否安装了正确的驱动程序和 CUDA Toolkit。")
    print("如果您没有 NVIDIA 显卡，PyTorch 将默认在 CPU 上运行。")

# 打印 PyTorch 版本
print(f"\nPyTorch 版本: {torch.__version__}")
