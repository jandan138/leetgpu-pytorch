#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "== Codex Multi-Agent Preflight =="

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex command not found in PATH."
  echo "Install/enable Codex CLI first."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git command not found in PATH."
  exit 1
fi

echo "Codex binary: $(command -v codex)"
echo "Codex version:"
codex --version || true

echo
echo "Checking feature flags..."
FEATURES_OUT="$(codex features list 2>/dev/null || true)"
if [[ -z "${FEATURES_OUT}" ]]; then
  echo "WARN: cannot read 'codex features list'."
  echo "Try manually:"
  echo "  codex features list"
  echo "  codex features enable multi_agent"
  exit 2
fi

echo "${FEATURES_OUT}"
echo

if echo "${FEATURES_OUT}" | rg -n "^multi_agent[[:space:]].*[[:space:]]true$" >/dev/null 2>&1; then
  echo "OK: multi_agent is enabled."
else
  echo "WARN: multi_agent is NOT enabled."
  echo "Enable it with:"
  echo "  codex features enable multi_agent"
  echo "Then restart codex."
fi

echo
echo "Checking git worktree support..."
if git worktree list >/dev/null 2>&1; then
  echo "OK: git worktree is available."
else
  echo "WARN: git worktree command failed. Worktree isolation may not work."
fi

echo
echo "Recommended next steps:"
echo "1) bash agent_team/scripts/bootstrap_agents.sh"
echo "2) bash agent_team/scripts/init_run.sh <run_id>"
echo "3) bash agent_team/scripts/setup_run_worktrees.sh <run_id>"
echo "4) Use template: agent_team/templates/codex_spawn_prompt_template.md"
echo "5) In TUI use /agent to inspect/switch spawned sub-agent threads."
echo "6) During long runs, monitor heartbeats:"
echo "   bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45"
