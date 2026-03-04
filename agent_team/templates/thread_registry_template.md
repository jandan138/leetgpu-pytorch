# Codex Thread Registry

- Run ID: <run_id>
- Maintainer: orchestrator
- Purpose: map Codex `/agent` threads to repository agents.

## Status Legend
- `planned`: agent planned but not spawned
- `active`: thread is running
- `waiting-subagent`: main orchestrator is waiting on this sub-agent
- `stuck-suspected`: heartbeat is stale and requires manual confirmation
- `stuck-confirmed`: manually confirmed as stuck
- `restarted`: agent thread was restarted after stuck confirmation
- `blocked`: waiting dependency
- `done`: completed and handed off

## Threads
| agent_id | codex_thread_id | status | owner | started_at | last_heartbeat_at | heartbeat_interval_min | stuck_candidate | escalation_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
