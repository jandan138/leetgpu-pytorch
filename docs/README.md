# LeetGPU-PyTorch 文档与教程

欢迎来到 LeetGPU-PyTorch 的学习中心。我们为您准备了两类文档：一类帮助您从零开始掌握 PyTorch，另一类带您领略最前沿的 GPU 编程生态。

## 📚 第一部分：PyTorch 快速入门 (PyTorch Basics)
适合 PyTorch 初学者，手把手带您入门。

1.  **[PyTorch 基础 01：张量 (Tensor)](tutorials/pytorch_basics/01_tensor_basics.md)**
    *   什么是张量？如何创建和操作它？
    *   对应代码：`00_pytorch_basics/01_tensors.py`
2.  **[PyTorch 基础 02：自动微分 (Autograd)](tutorials/pytorch_basics/02_autograd_mechanics.md)**
    *   深度学习的核心：梯度是如何自动计算的？
    *   对应代码：`00_pytorch_basics/02_autograd.py`
3.  **[PyTorch 基础 03：使用 GPU 加速](tutorials/pytorch_basics/03_gpu_acceleration.md)**
    *   如何检测 CUDA 环境并将计算移动到 GPU 上？
    *   对应代码：`00_pytorch_basics/03_check_gpu.py`

---

## 🚀 第二部分：GPU 编程生态全景 (GPU Ecosystem)
适合对高性能计算感兴趣的进阶读者。这里我们将探讨除了 PyTorch 之外，还有哪些强大的工具（Taichi, Triton, Warp）以及它们之间的关系。

1.  **[Taichi vs PyTorch](tutorials/gpu_ecosystem/01_taichi_vs_pytorch.md)**
    *   **核心问题**：既然 PyTorch 也能用 GPU，为什么还需要 Taichi？
    *   **结论**：PyTorch 适合搭积木（深度学习），Taichi 适合造积木（物理仿真）。
2.  **[OpenAI Triton](tutorials/gpu_ecosystem/02_triton_overview.md)**
    *   **核心问题**：Triton 是什么？为什么它是现在的版本之子？
    *   **结论**：它是 Python 版的 CUDA，专门用来加速 AI 算子，也是 PyTorch 2.0 的默认后端。
3.  **[Triton vs Taichi](tutorials/gpu_ecosystem/03_triton_vs_taichi.md)**
    *   **核心问题**：Triton 能完全替代 Taichi 吗？
    *   **结论**：不能。Triton 擅长稠密矩阵（AI），Taichi 擅长稀疏粒子（物理）。
4.  **[NVIDIA Warp](tutorials/gpu_ecosystem/04_nvidia_warp_intro.md)**
    *   **核心问题**：NVIDIA 有没有 Taichi 的替代品？
    *   **结论**：有，Warp。它是 NVIDIA 官方推出的物理仿真框架，支持稀疏体积（NanoVDB）。
5.  **[Warp vs Triton](tutorials/gpu_ecosystem/05_warp_vs_triton.md)**
    *   **核心问题**：它们是竞争关系吗？
    *   **结论**：不是。它们是兄弟，一个负责修内功（加速神经网络），一个负责练外功（模拟物理世界）。
