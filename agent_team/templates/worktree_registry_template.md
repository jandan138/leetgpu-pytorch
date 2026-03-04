# Worktree Registry

- Run ID: <run_id>
- Maintainer: orchestrator
- Purpose: isolate editable agents into dedicated git worktrees.

## Status Legend
- `planned`: worktree row prepared, not created
- `ready`: worktree exists and agent can start coding
- `blocked`: creation failed or dependency issue
- `done`: agent work completed
- `cleaned`: worktree removed

## Worktrees
| agent_id | can_edit_code | branch | worktree_path | status | notes |
| --- | --- | --- | --- | --- | --- |
