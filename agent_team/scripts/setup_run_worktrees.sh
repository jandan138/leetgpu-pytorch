#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash agent_team/scripts/setup_run_worktrees.sh <run_id> [base_ref]"
  exit 1
fi

RUN_ID="$1"
BASE_REF="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTS_YAML="${TEAM_DIR}/agents.yaml"
RUN_DIR="${TEAM_DIR}/runs/${RUN_ID}"
WORKTREE_DIR="${RUN_DIR}/worktrees"
WORKTREE_REGISTRY="${WORKTREE_DIR}/registry.md"

if [[ ! -f "${AGENTS_YAML}" ]]; then
  echo "ERROR: missing ${AGENTS_YAML}"
  exit 2
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}"
  echo "Run init first: bash agent_team/scripts/init_run.sh ${RUN_ID}"
  exit 2
fi

if [[ ! -f "${WORKTREE_REGISTRY}" ]]; then
  echo "ERROR: worktree registry not found: ${WORKTREE_REGISTRY}"
  exit 2
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "ERROR: current directory is not inside a git repository"
  exit 3
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "${REPO_ROOT}")"
SAFE_RUN_ID="$(printf '%s' "${RUN_ID}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-*//; s/-*$//')"
if [[ -z "${SAFE_RUN_ID}" ]]; then
  SAFE_RUN_ID="run"
fi

DEFAULT_ROOT="/tmp/agent_team_worktrees/${REPO_NAME}/${SAFE_RUN_ID}"
ROOT_DIR="${AGENT_TEAM_WORKTREE_ROOT:-${DEFAULT_ROOT}}"
mkdir -p "${ROOT_DIR}"

if [[ -z "${BASE_REF}" ]]; then
  BASE_REF="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "${BASE_REF}" == "HEAD" ]]; then
    BASE_REF="HEAD"
  fi
fi

TMP_REG="$(mktemp)"
{
  echo "# Worktree Registry"
  echo
  echo "- Run ID: ${RUN_ID}"
  echo "- Maintainer: orchestrator"
  echo "- Purpose: isolate editable agents into dedicated git worktrees."
  echo
  echo "## Status Legend"
  echo "- \`planned\`: worktree row prepared, not created"
  echo "- \`ready\`: worktree exists and agent can start coding"
  echo "- \`blocked\`: creation failed or dependency issue"
  echo "- \`done\`: agent work completed"
  echo "- \`cleaned\`: worktree removed"
  echo
  echo "## Worktrees"
  echo "| agent_id | can_edit_code | branch | worktree_path | status | notes |"
  echo "| --- | --- | --- | --- | --- | --- |"
} > "${TMP_REG}"

while IFS=$'\t' read -r AGENT_ID CAN_EDIT; do
  BRANCH=""
  PATH_OUT=""
  STATUS="planned"
  NOTES=""

  if [[ "${CAN_EDIT}" == "true" ]]; then
    BRANCH="run/${SAFE_RUN_ID}/${AGENT_ID}"
    PATH_OUT="${ROOT_DIR}/${AGENT_ID}"

    if [[ -d "${PATH_OUT}" && ! -f "${PATH_OUT}/.git" && ! -d "${PATH_OUT}/.git" ]]; then
      STATUS="blocked"
      NOTES="path exists but is not a git worktree"
    else
      if [[ -f "${PATH_OUT}/.git" || -d "${PATH_OUT}/.git" ]]; then
        if git -C "${PATH_OUT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
          STATUS="ready"
          NOTES="existing worktree reused"
        else
          STATUS="blocked"
          NOTES="existing path not usable as worktree"
        fi
      else
        if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
          if git worktree add "${PATH_OUT}" "${BRANCH}" >/tmp/agent_team_worktree_add.log 2>&1; then
            STATUS="ready"
            NOTES="attached existing branch"
          else
            STATUS="blocked"
            NOTES="$(tr '\n' ';' </tmp/agent_team_worktree_add.log | sed 's/|/\\|/g' | cut -c1-240)"
          fi
        else
          if git worktree add -b "${BRANCH}" "${PATH_OUT}" "${BASE_REF}" >/tmp/agent_team_worktree_add.log 2>&1; then
            STATUS="ready"
            NOTES="created branch from ${BASE_REF}"
          else
            STATUS="blocked"
            NOTES="$(tr '\n' ';' </tmp/agent_team_worktree_add.log | sed 's/|/\\|/g' | cut -c1-240)"
          fi
        fi
      fi
    fi
  fi

  printf "| %s | %s | %s | %s | %s | %s |\n" \
    "${AGENT_ID}" "${CAN_EDIT}" "${BRANCH}" "${PATH_OUT}" "${STATUS}" "${NOTES}" >> "${TMP_REG}"
done < <(
  awk '
    /- id:/ {id=$3; order[++n]=id}
    /can_edit_code:/ {edit[id]=$2}
    END {
      for (i=1; i<=n; i++) {
        id=order[i]
        print id "\t" edit[id]
      }
    }
  ' "${AGENTS_YAML}"
)

mv "${TMP_REG}" "${WORKTREE_REGISTRY}"
rm -f /tmp/agent_team_worktree_add.log

echo "Worktree setup complete for run: ${RUN_ID}"
echo "Registry: ${WORKTREE_REGISTRY}"
echo "Root: ${ROOT_DIR}"
