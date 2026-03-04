# AGENTS.md

This file provides guidance to Codex-style agents working in this repository.

## Project Overview

Educational GPU programming project for learning PyTorch and Triton kernel development.

- Documentation is primarily written in Chinese.
- Code identifiers and code comments should use English.
- The repository combines:
  - PyTorch basics tutorials
  - LeetGPU-style practice problems
  - Deep-dive docs on GPU concepts

## Primary Orchestration System

This repo now uses the `agent_team/` system as the primary multi-agent workflow.

Canonical location:

- `agent_team/`

Core assets:

- `agent_team/agents.yaml`
- `agent_team/agents/<agent_id>/profile.md`
- `agent_team/agents/<agent_id>/memory.md`
- `agent_team/templates/`
- `agent_team/scripts/`
- `agent_team/references/wait-and-stuck-policy.md`

When user intent is to "use agent team", follow `agent_team/README.md` and execute the run lifecycle below.

## Agent Team Run Lifecycle

1. Enable Codex multi-agent capability once:
   - Linux/macOS: `codex features enable multi_agent`
   - Windows PowerShell: `codex.cmd features enable multi_agent`
2. Run preflight checks:
   - Linux/macOS: `bash agent_team/scripts/codex_multi_agent_preflight.sh`
   - Windows PowerShell: `powershell -ExecutionPolicy Bypass -File agent_team/scripts/codex_multi_agent_preflight.ps1`
3. Initialize a run:
   - Linux/macOS: `bash agent_team/scripts/init_run.sh <run_id>`
   - Windows PowerShell: `powershell -ExecutionPolicy Bypass -File agent_team/scripts/init_run.ps1 -RunId <run_id>`
4. Prepare/edit isolated worktrees for editable roles:
   - `bash agent_team/scripts/setup_run_worktrees.sh <run_id> [base_ref]`
5. Delegate work with:
   - `/agent` thread switching
   - `agent_team/templates/codex_spawn_prompt_template.md`
6. Keep run metadata auditable:
   - maintain `runs/<run_id>/threads/registry.md`
   - maintain `runs/<run_id>/worktrees/registry.md`
7. Validate run integrity:
   - Linux/macOS: `bash agent_team/scripts/check_run_logs.sh <run_id>`
   - Windows PowerShell: `powershell -ExecutionPolicy Bypass -File agent_team/scripts/check_run_logs.ps1 -RunId <run_id>`
8. Merge run memory deltas into long-term memory:
   - `bash agent_team/scripts/update_agent_memory.sh <run_id>`

## Cross-Platform Notes (Windows/Linux)

- Linux/macOS should use the `*.sh` scripts under `agent_team/scripts/`.
- Native Windows should prefer `*.ps1` scripts when provided.
- Current PowerShell-native coverage:
  - `agent_team/scripts/codex_multi_agent_preflight.ps1`
  - `agent_team/scripts/init_run.ps1`
  - `agent_team/scripts/check_run_logs.ps1`
- For scripts that are still Bash-only, use Git Bash or WSL on Windows until PowerShell parity is added.
- On Windows, prefer `codex.cmd` over `codex` in PowerShell sessions to avoid execution-policy issues with `codex.ps1`.

## Governance Rules (Agent Team)

- Every key conclusion must include evidence path (command output, file path, or report path).
- Every test claim must include a reproducible command or artifact path.
- Thread-to-agent mapping must remain explicit in `threads/registry.md`.
- Editable roles must only edit code in their assigned worktree from `worktrees/registry.md`.
- For roles with `can_edit_code=false`:
  1. Write handoff facts first.
  2. `doc-writer` produces formal log entry.
  3. Log must include both `Source Agent` and `Log Writer`.

## Wait and Stuck Policy

Default policy: slow-wait, no hard timeout by default.

- Use heartbeat monitoring:
  - `bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45`
- Restart only after human confirmation of "stuck":
  - update registry status to `stuck-confirmed`
  - then run `bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>`
- Detailed policy:
  - `agent_team/references/wait-and-stuck-policy.md`

## Setup and Dependencies

- Package manager: `uv` (prefer `uv sync`, avoid ad-hoc `pip install`)
- Python: `>=3.10`
- GPU target: NVIDIA CUDA 12.4 environment
- Typical venv path: `.venv/`

Useful checks:

```bash
uv sync
python 00_pytorch_basics/03_check_gpu.py
```

## How to Run Code

Examples:

```bash
# Run a basics script
python 00_pytorch_basics/01_tensors.py

# Run one problem test (from its own directory)
cd 01_leetgpu_problems/easy/02_matrix_multiplication
python tests.py

# Run a specific solution module directly
python 01_leetgpu_problems/easy/01_vector_add/solution.py
```

There is no single global test runner. Each problem directory owns its own `tests.py`.

## Repository Structure

- `00_pytorch_basics/`: standalone learning scripts
- `01_leetgpu_problems/`: problems grouped by `easy/`, `medium/`, `hard/`
- `docs/tutorials/`: tutorial markdown
- `agent_team/`: codex-native multi-agent governance and workflow system
- `utils/`: shared utilities (currently minimal)

Typical problem directory layout:

- `README.md`: problem statement and explanation
- `solution_pytorch.py`: baseline implementation
- `solution_triton.py`: Triton kernel implementation
- `tests.py`: correctness checks + benchmark harness
- `deep_dive_*.md`: optional deep notes

## Core Engineering Conventions

- Prefer correctness first, then optimization.
- Keep solution APIs stable and explicit.
- Use type hints on solution functions.
- Write outputs into pre-allocated output tensors rather than returning new tensors when the problem contract requires it.
- Keep benchmark-only logic in `tests.py`, not inside `solve()`.
- Use `torch.cuda.synchronize()` around timing windows in benchmarks.
- Keep Triton optional:
  - guard imports with `try/except ImportError`
  - provide clear fallback behavior when Triton is unavailable

## Workflow Rules for Problem Changes

When implementing or changing a problem:

1. Read the target problem `README.md` first.
2. Check neighboring solved problems for style consistency.
3. Preserve existing `solve(...)` signatures unless explicitly asked to change them.
4. Update tests when behavior or signatures change.
5. Run the local problem `tests.py` whenever feasible.

When writing `solution_pytorch.py`:

- Import only what is needed (usually just `torch`).
- Keep implementation concise and educational.
- Avoid debug prints in library-style solution files.

When writing `solution_triton.py`:

- Use masked `tl.load`/`tl.store` for boundaries.
- Choose grid/block decomposition that matches tensor shape and operation semantics.
- Keep fallback path explicit when Triton is not installed.

When writing `tests.py`:

- Compare outputs with `torch.allclose(...)` using tolerances appropriate for GPU precision (often fp16).
- Include warmup before benchmarking.
- Synchronize CUDA before and after timed loops.
- Handle missing Triton gracefully.

## Commit and Change Style

- Prefer focused, minimal diffs.
- Keep docs aligned with code behavior.
- Conventional commit style is preferred (`feat:`, `fix:`, `docs:`, etc.).

## Relationship to Legacy Claude Config

The existing `.claude/` configuration can remain as reference material, but `agent_team/` is the primary orchestration workflow for Codex-native multi-agent operations in this repository.
