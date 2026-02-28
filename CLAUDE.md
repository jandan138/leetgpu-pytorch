# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational GPU programming project for learning PyTorch and Triton kernel development. Combines PyTorch fundamentals, LeetCode-style GPU challenges, and deep-dive documentation. Documentation is written in Chinese; code uses English identifiers.

## Setup & Dependencies

- **Package manager**: `uv` (not pip). Run `uv sync` to install everything.
- **Python**: >=3.10
- **GPU**: NVIDIA with CUDA 12.4 (driver >= 525.x). PyTorch 2.6.0+cu124.
- **Virtual env**: `.venv/` managed by uv. VS Code configured to use `.venv/Scripts/python.exe`.
- **Verify GPU**: `python 00_pytorch_basics/03_check_gpu.py`

## Running Code & Tests

```bash
# Run a basics script
python 00_pytorch_basics/01_tensors.py

# Run a problem's test (must cd into the problem directory)
cd 01_leetgpu_problems/easy/02_matrix_multiplication
python tests.py

# Run a specific solution directly
python 01_leetgpu_problems/easy/01_vector_add/solution.py
```

There is no project-wide test runner, linter, or build system. Each problem has its own `tests.py` that runs correctness checks (`torch.allclose`) and performance benchmarks.

## Architecture

### Problem Structure (`01_leetgpu_problems/`)

Problems are organized by difficulty (`easy/`, `medium/`, `hard/`). Each problem directory contains:
- `README.md` — problem description, solution walkthrough, deep-dive explanations
- `solution_pytorch.py` — baseline PyTorch implementation
- `solution_triton.py` — Triton kernel implementation (may have variants like `solution_triton_tiled.py`)
- `tests.py` — correctness + benchmark harness
- `deep_dive_*.md` — supplementary docs on specific concepts

Every solution module exposes a `solve(A, B, C, ...)` function that writes results into a pre-allocated output tensor `C`.

### Other Directories

- `00_pytorch_basics/` — standalone learning scripts (tensors, autograd, GPU checks)
- `docs/tutorials/` — tutorial markdown files in two tracks: `pytorch_basics/` and `gpu_ecosystem/`
- `utils/` — reserved for shared utilities (currently empty)

## Conventions

- **Triton is optional**: test files use `try/except ImportError` to gracefully skip Triton tests when not installed.
- **Solutions use type hints**: `def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int)`
- **Benchmarks sync CUDA**: always call `torch.cuda.synchronize()` before timing measurements.
- **Commit messages**: use conventional commits style (`feat:`, `docs:`, `fix:`).
