#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash agent_team/scripts/teardown_run_worktrees.sh <run_id> [--delete-branches]"
  exit 1
fi

RUN_ID="$1"
DELETE_BRANCHES="false"
if [[ "${2:-}" == "--delete-branches" ]]; then
  DELETE_BRANCHES="true"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${TEAM_DIR}/runs/${RUN_ID}"
WORKTREE_REGISTRY="${RUN_DIR}/worktrees/registry.md"

if [[ ! -f "${WORKTREE_REGISTRY}" ]]; then
  echo "ERROR: missing ${WORKTREE_REGISTRY}"
  exit 2
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "ERROR: current directory is not inside a git repository"
  exit 3
fi

while IFS=$'\t' read -r AGENT_ID CAN_EDIT BRANCH PATH_OUT STATUS; do
  if [[ "${CAN_EDIT}" != "true" ]]; then
    continue
  fi

  if [[ -z "${PATH_OUT}" ]]; then
    continue
  fi

  if [[ -d "${PATH_OUT}" ]]; then
    if git worktree remove --force "${PATH_OUT}" >/tmp/agent_team_worktree_rm.log 2>&1; then
      echo "REMOVED worktree: ${PATH_OUT}"
    else
      echo "WARN: failed removing ${PATH_OUT}"
      cat /tmp/agent_team_worktree_rm.log
    fi
  fi

  if [[ "${DELETE_BRANCHES}" == "true" && -n "${BRANCH}" ]]; then
    if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
      if git branch -D "${BRANCH}" >/tmp/agent_team_worktree_branch_rm.log 2>&1; then
        echo "DELETED branch: ${BRANCH}"
      else
        echo "WARN: failed deleting branch ${BRANCH}"
        cat /tmp/agent_team_worktree_branch_rm.log
      fi
    fi
  fi
done < <(
  awk -F'|' '
    /^\|/ {
      agent=$2; can_edit=$3; branch=$4; path=$5; status=$6;
      gsub(/^[ \t]+|[ \t]+$/, "", agent);
      gsub(/^[ \t]+|[ \t]+$/, "", can_edit);
      gsub(/^[ \t]+|[ \t]+$/, "", branch);
      gsub(/^[ \t]+|[ \t]+$/, "", path);
      gsub(/^[ \t]+|[ \t]+$/, "", status);
      if (agent != "" && agent != "agent_id" && agent != "---") {
        print agent "\t" can_edit "\t" branch "\t" path "\t" status;
      }
    }
  ' "${WORKTREE_REGISTRY}"
)

rm -f /tmp/agent_team_worktree_rm.log /tmp/agent_team_worktree_branch_rm.log
echo "Teardown complete for run: ${RUN_ID}"
