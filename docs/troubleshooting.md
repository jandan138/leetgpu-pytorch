# 常见问题排查 (Troubleshooting)

本文档记录了在配置和运行 LeetGPU-PyTorch 项目时可能遇到的常见问题及其解决方案。

## 1. Triton 运行时错误：找不到 C 编译器

### 现象 (Symptom)
在运行包含 Triton 内核的代码（如 `02_matrix_multiplication/tests.py`）时，程序抛出以下错误：

```text
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

或者在某些环境中：

```text
RuntimeError: Triton Error [CUDA]: Failed to compile kernel.
```

### 原因 (Cause)
Triton 是一个即时编译 (JIT) 编译器，它在运行时需要调用系统的 C 编译器（通常是 `gcc` 或 `clang`）来处理部分内核代码或链接过程。如果您的运行环境（如某些精简版的 Docker 容器或云开发环境）没有预装标准的构建工具链，Triton 就会因为找不到编译器而失败。

### 诊断步骤 (Diagnosis)
在终端中运行以下命令检查 `gcc` 是否存在：

```bash
which gcc
# 如果没有任何输出，或者提示 "gcc: command not found"

gcc --version
# 如果提示 "command not found"
```

如果上述命令无法正确输出版本号，说明系统中缺失 C 编译器。

### 解决方案 (Solution)

根据您的操作系统类型，安装相应的构建工具包。

#### Ubuntu / Debian
安装 `build-essential` 软件包，它包含了 `gcc`, `g++`, `make` 等标准工具。

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

#### CentOS / RHEL
```bash
sudo yum install -y gcc gcc-c++ make
```

#### 验证 (Verification)
安装完成后，再次检查版本：

```bash
gcc --version
# 输出示例: gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
```

此时再次运行 Triton 代码，该错误应当消失。
