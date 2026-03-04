#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "== Codex Multi-Agent Preflight =="

CODEX_CMD=""
if command -v codex >/dev/null 2>&1; then
  CODEX_CMD="codex"
elif command -v codex.cmd >/dev/null 2>&1; then
  CODEX_CMD="codex.cmd"
fi

if [[ -z "${CODEX_CMD}" ]]; then
  echo "ERROR: codex/codex.cmd command not found in PATH."
  echo "Install/enable Codex CLI first."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git command not found in PATH."
  exit 1
fi

if ! command -v awk >/dev/null 2>&1; then
  echo "ERROR: awk command not found in PATH."
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "WARN: rg (ripgrep) not found. Current preflight can continue without it."
  echo "      Other scripts may still benefit from rg for faster scans."
fi

CODEX_BIN="$(command -v "${CODEX_CMD}" || true)"
echo "Codex command: ${CODEX_CMD}"
echo "Codex binary: ${CODEX_BIN}"
echo "Codex version:"
"${CODEX_CMD}" --version || true

if [[ "${CODEX_BIN}" == /mnt/c/* ]] && ! command -v node >/dev/null 2>&1; then
  echo "WARN: codex points to a Windows shim under /mnt/c but node is missing in this Linux environment."
  echo "      Install node in Linux PATH or run preflight from native Windows PowerShell."
fi

echo
echo "Checking feature flags..."
FEATURES_OUT="$("${CODEX_CMD}" features list 2>/dev/null || true)"
if [[ -z "${FEATURES_OUT}" ]]; then
  echo "WARN: cannot read '${CODEX_CMD} features list'."
  echo "Try manually:"
  echo "  ${CODEX_CMD} features list"
  echo "  ${CODEX_CMD} features enable multi_agent"
  exit 2
fi

echo "${FEATURES_OUT}"
echo

if echo "${FEATURES_OUT}" | awk '$1=="multi_agent" && $NF=="true"{found=1} END{exit(found?0:1)}'; then
  echo "OK: multi_agent is enabled."
else
  echo "WARN: multi_agent is NOT enabled."
  echo "Enable it with:"
  echo "  ${CODEX_CMD} features enable multi_agent"
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
echo "1) Linux/macOS: bash agent_team/scripts/bootstrap_agents.sh"
echo "   Windows: powershell -ExecutionPolicy Bypass -File agent_team/scripts/bootstrap_agents.ps1"
echo "2) Linux/macOS: bash agent_team/scripts/init_run.sh <run_id>"
echo "   Windows: powershell -ExecutionPolicy Bypass -File agent_team/scripts/init_run.ps1 -RunId <run_id>"
echo "3) Linux/macOS: bash agent_team/scripts/setup_run_worktrees.sh <run_id>"
echo "   Windows: powershell -ExecutionPolicy Bypass -File agent_team/scripts/setup_run_worktrees.ps1 -RunId <run_id>"
echo "4) Use template: agent_team/templates/codex_spawn_prompt_template.md"
echo "5) In TUI use /agent to inspect/switch spawned sub-agent threads."
echo "6) During long runs, monitor heartbeats:"
echo "   bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45"
echo "   powershell -ExecutionPolicy Bypass -File agent_team/scripts/monitor_subagents.ps1 -RunId <run_id> -IntervalMin 10 -StuckMin 45"
