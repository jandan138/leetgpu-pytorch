# Codex Multi-Agent Spawn Prompt Template

Use this in the main Codex thread after `init_run.sh`.

## Example Prompt (Chinese)
```text
采用agent team。run_id=<run_id>。

请按以下规则进入 Codex multi-agent 执行：
1) 基于 agent_team/agents.yaml 的角色分工 spawn 子 agent（只为本任务需要的角色）。
2) 每个子 agent 必须在 runs/<run_id>/logs/<agent_id>.md 持续记录研究、改动、测试、证据。
3) 每个子 agent 必须维护 runs/<run_id>/memory/<agent_id>.delta.md。
4) 可编辑角色必须在专属 worktree 工作：
   - 先执行 bash agent_team/scripts/setup_run_worktrees.sh <run_id>
   - 每个可编辑 agent 只能在 runs/<run_id>/worktrees/registry.md 指定的 branch/path 下修改代码
5) can_edit_code=false 的角色不要直接改代码，先写 handoff，再由 doc-writer 代写日志。
6) 把线程映射写入 runs/<run_id>/threads/registry.md（包含 thread id、状态、负责人、时间）。
7) 等待策略：主代理默认慢慢等，不设硬超时。
   - 子代理慢是允许的，除非被确认卡死
   - 每个子代理必须周期性更新 last_heartbeat_at（建议每10分钟）
   - 主代理周期巡检：
     - bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45
   - 仅当“卡死已人工确认（stuck-confirmed）”才可重启同角色线程：
     - bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>
8) 任务完成后运行：
   - bash agent_team/scripts/check_run_logs.sh <run_id>
   - bash agent_team/scripts/update_agent_memory.sh <run_id>
9) 输出最终汇总：变更、测试结果、风险、未完成项。
```

## Example Prompt (English)
```text
Use agent team mode with run_id=<run_id>.

Spawn Codex sub-agents according to agent_team/agents.yaml (only needed roles).
Requirements:
- Each sub-agent writes to runs/<run_id>/logs/<agent_id>.md.
- Each sub-agent updates runs/<run_id>/memory/<agent_id>.delta.md.
- For editable roles, run `bash agent_team/scripts/setup_run_worktrees.sh <run_id>` and only edit in assigned branch/path from `runs/<run_id>/worktrees/registry.md`.
- Non-edit roles (can_edit_code=false) must produce handoff; doc-writer records delegated logs.
- Record all thread mappings in runs/<run_id>/threads/registry.md.
- Waiting policy: no hard timeout while sub-agents are alive. Slow progress is acceptable.
- Each sub-agent should update `last_heartbeat_at` periodically (recommended every 10 min).
- Main orchestrator should monitor with:
  - `bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45`
- Restart is allowed only after manual stuck confirmation (`stuck-confirmed`):
  - `bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>`
- At the end run:
  - bash agent_team/scripts/check_run_logs.sh <run_id>
  - bash agent_team/scripts/update_agent_memory.sh <run_id>
- Return a final summary: changes, tests, risks, open items.
```
