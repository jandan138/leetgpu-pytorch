# Agent Team System (Codex-Native Integrated)

## 目标
把两套能力融合为一套工作流：
- **Codex 官方多代理**：spawn sub-agent、`/agent` 切线程、并行执行与上下文隔离
- **仓库内治理系统**：角色规范、日志审计、handoff 追责、memory 沉淀

一句话：`/agent` 负责“并行执行”，`agent_team/` 负责“可审计交付”。

## 你只需怎么说
在主线程直接说：
- `采用agent team，目标是...`
- `用agent team做...`

主 agent 将按本系统自动落地 run（初始化目录、分工、日志、校验、memory 合并）。

## Codex 官方能力接入方式
1. 启用多代理能力（一次性）  
   `codex features enable multi_agent`  
   然后重启 Codex/TUI。
2. 运行预检脚本  
   `bash agent_team/scripts/codex_multi_agent_preflight.sh`
3. 初始化 run  
   `bash agent_team/scripts/init_run.sh <run_id>`
4. 使用 spawn 模板在主线程下达并行指令  
   `agent_team/templates/codex_spawn_prompt_template.md`
5. 创建可编辑角色专属 worktree（强隔离）  
   `bash agent_team/scripts/setup_run_worktrees.sh <run_id>`
6. 用 `/agent` 查看/切换子线程；同时更新 `runs/<run_id>/threads/registry.md`

## 目录结构
```text
agent_team/
  agents.yaml
  agents/
    <agent_id>/
      profile.md
      memory.md
  templates/
    agent_profile_template.md
    agent_memory_template.md
    agent_log_template.md
    handoff_to_doc_template.md
    memory_delta_template.md
    thread_registry_template.md
    codex_spawn_prompt_template.md
  references/
    wait-and-stuck-policy.md
  scripts/
    codex_multi_agent_preflight.sh
    bootstrap_agents.sh
    init_run.sh
    setup_run_worktrees.sh
    teardown_run_worktrees.sh
    monitor_subagents.sh
    restart_stuck_subagent.sh
    check_run_logs.sh
    update_agent_memory.sh
  runs/
    <run_id>/
      logs/
      handoff/
      memory/
      threads/
        registry.md
      worktrees/
        registry.md
      artifacts/
```

## Agent 契约
每个 agent 有两份常驻文档：
- `agents/<id>/profile.md`：标准化职责与规则（类似 agent spec）
- `agents/<id>/memory.md`：长期经验记忆

每次 run 有增量记忆：
- `runs/<run_id>/memory/<id>.delta.md`

## 无编辑权限角色规则
若 `can_edit_code=false`：
1. 必须先写 handoff（事实源）
2. `doc-writer` 代写正式日志
3. 日志必须标注：
   - `Source Agent`
   - `Log Writer`

## 命令
- 初始化/补齐全部 agent profile+memory  
  `bash agent_team/scripts/bootstrap_agents.sh`
- 初始化某次 run  
  `bash agent_team/scripts/init_run.sh <run_id>`
- 创建该 run 的并行隔离 worktrees（仅可编辑角色）  
  `bash agent_team/scripts/setup_run_worktrees.sh <run_id> [base_ref]`
- 自定义 worktree 根目录（可选）  
  `AGENT_TEAM_WORKTREE_ROOT=/tmp/agent_team_worktrees/<repo>/<run_id> bash agent_team/scripts/setup_run_worktrees.sh <run_id>`
- 清理该 run 的 worktrees  
  `bash agent_team/scripts/teardown_run_worktrees.sh <run_id> [--delete-branches]`
- 巡检子代理心跳（默认慢等待，不设硬超时）  
  `bash agent_team/scripts/monitor_subagents.sh <run_id> [--interval-min 10] [--stuck-min 45]`
- 子代理确认卡死后重启同角色线程  
  `bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>`
- 校验 run 完整性（日志 + memory delta + delegation + thread registry + worktree registry）  
  `bash agent_team/scripts/check_run_logs.sh <run_id>`
- 合并 run 增量记忆到长期 memory  
  `bash agent_team/scripts/update_agent_memory.sh <run_id>`

## Memory 策略
默认策略：**按 run 归档，再合并长期记忆**。
- Run-time：`runs/<run_id>/memory/*.delta.md`
- Long-term：`agents/<id>/memory.md`

## 治理约束
- 任何关键结论都要有证据路径（命令输出、文件、报告）。
- 任何测试结论都要给出可复现命令或产物路径。
- 线程与角色必须在 `threads/registry.md` 中一一映射，避免“谁做了什么”不可追溯。
- 可编辑角色必须在 `worktrees/registry.md` 指定的分支与路径里改代码，避免并行写冲突。

## 等待与卡死策略（重要）
- 默认策略：**主代理慢慢等，不设硬超时**。子代理慢并不等于失败。
- 仅在“疑似卡死 -> 人工确认卡死”后才允许重启线程。
- 心跳建议：每个活跃子代理建议每 10 分钟更新 `last_heartbeat_at`。
- 巡检建议：主代理周期运行  
  `bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45`
- 人工确认卡死时，更新 `threads/registry.md` 该行为：
  - `status = stuck-confirmed`
  - `stuck_candidate = confirmed`
  - `notes` 记录确认原因与时间
- 处置建议：确认卡死后，默认“重启同角色线程”  
  `bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>`
- 详细规范见：`agent_team/references/wait-and-stuck-policy.md`
