# Wait-and-Stuck Policy

## Core Rule
Main orchestrator should wait for sub-agents by default.  
Slow progress is acceptable. Do not terminate early due to slowness alone.

## Decision Policy
1. Waiting strategy: no hard timeout.
2. Detection strategy: monitor heartbeat and raise `stuck-suspected`.
3. Confirmation strategy: manual confirmation required before restart.
4. Recovery strategy: restart same role thread after `stuck-confirmed`.

## Heartbeat Convention
- Active or waiting sub-agent should update `last_heartbeat_at` periodically.
- Recommended interval: 10 minutes.

## Suggested Monitoring
```bash
bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45
```

`stuck-suspected` means warning only. It is not auto-kill or auto-restart.

## Restart Procedure
After manual confirmation (`stuck-confirmed`), run:
```bash
bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>
```

The restart script should:
- set status back to `active`
- update `codex_thread_id`
- reset `stuck_candidate` to `no`
- increment `escalation_count`
- append restart notes for audit trail
