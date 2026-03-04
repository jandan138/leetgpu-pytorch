#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash agent_team/scripts/check_run_logs.sh <run_id>"
  exit 1
fi

RUN_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTS_YAML="${TEAM_DIR}/agents.yaml"
RUN_DIR="${TEAM_DIR}/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
HANDOFF_DIR="${RUN_DIR}/handoff"
MEMORY_DIR="${RUN_DIR}/memory"
THREAD_DIR="${RUN_DIR}/threads"
THREAD_REGISTRY="${THREAD_DIR}/registry.md"
WORKTREE_DIR="${RUN_DIR}/worktrees"
WORKTREE_REGISTRY="${WORKTREE_DIR}/registry.md"

if [[ ! -f "${AGENTS_YAML}" ]]; then
  echo "ERROR: missing ${AGENTS_YAML}"
  exit 2
fi

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "ERROR: log directory not found: ${LOG_DIR}"
  exit 2
fi

if [[ ! -d "${MEMORY_DIR}" ]]; then
  echo "ERROR: memory directory not found: ${MEMORY_DIR}"
  exit 2
fi

if [[ ! -f "${THREAD_REGISTRY}" ]]; then
  echo "ERROR: thread registry not found: ${THREAD_REGISTRY}"
  exit 2
fi

SEARCH_BIN="rg"
if ! command -v rg >/dev/null 2>&1; then
  SEARCH_BIN="grep"
  echo "WARN: rg not found. Falling back to grep for literal checks."
fi

contains_literal() {
  local literal="$1"
  local file="$2"
  if [[ "${SEARCH_BIN}" == "rg" ]]; then
    rg -n --fixed-strings "${literal}" "${file}" >/dev/null 2>&1
  else
    grep -n -F "${literal}" "${file}" >/dev/null 2>&1
  fi
}

if ! contains_literal "last_heartbeat_at" "${THREAD_REGISTRY}"; then
  echo "ERROR: thread registry header is outdated, missing last_heartbeat_at"
  exit 2
fi

if ! contains_literal "heartbeat_interval_min" "${THREAD_REGISTRY}"; then
  echo "ERROR: thread registry header is outdated, missing heartbeat_interval_min"
  exit 2
fi

if ! contains_literal "stuck_candidate" "${THREAD_REGISTRY}"; then
  echo "ERROR: thread registry header is outdated, missing stuck_candidate"
  exit 2
fi

if [[ ! -f "${WORKTREE_REGISTRY}" ]]; then
  echo "ERROR: worktree registry not found: ${WORKTREE_REGISTRY}"
  exit 2
fi

declare -a AGENTS

declare -A CAN_EDIT_MAP

declare -A DOC_DELEGATE_MAP

while IFS=$'\t' read -r AGENT_ID CAN_EDIT DOC_DELEGATE; do
  AGENTS+=("${AGENT_ID}")
  CAN_EDIT_MAP["${AGENT_ID}"]="${CAN_EDIT}"
  DOC_DELEGATE_MAP["${AGENT_ID}"]="${DOC_DELEGATE}"
done < <(
  awk '
    /- id:/ {id=$3; order[++n]=id}
    /can_edit_code:/ {edit[id]=$2}
    /doc_delegate:/ {doc[id]=$2}
    END {
      for (i=1; i<=n; i++) {
        id=order[i]
        print id "\t" edit[id] "\t" doc[id]
      }
    }
  ' "${AGENTS_YAML}"
)

LOG_HEADERS=(
  "## 1) Task & Inputs"
  "## 2) Research"
  "## 3) Code Changes"
  "## 4) Test Execution"
  "## 5) Risks & Open Items"
  "## 6) Handoff"
  "## 7) Evidence"
)

MEMORY_HEADERS=(
  "## 1) Delta Summary"
  "## 2) New Stable Preferences"
  "## 3) Pitfalls Learned"
  "## 4) Reusable Patterns"
  "## 5) Evidence Links"
)

FAILED=0

for AGENT in "${AGENTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${AGENT}.md"
  MEMORY_FILE="${MEMORY_DIR}/${AGENT}.delta.md"

  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "MISSING: ${LOG_FILE}"
    FAILED=1
  else
    for HEADER in "${LOG_HEADERS[@]}"; do
      if ! contains_literal "${HEADER}" "${LOG_FILE}"; then
        echo "INVALID: ${LOG_FILE} missing header: ${HEADER}"
        FAILED=1
      fi
    done
  fi

  if [[ ! -f "${MEMORY_FILE}" ]]; then
    echo "MISSING: ${MEMORY_FILE}"
    FAILED=1
  else
    for HEADER in "${MEMORY_HEADERS[@]}"; do
      if ! contains_literal "${HEADER}" "${MEMORY_FILE}"; then
        echo "INVALID: ${MEMORY_FILE} missing header: ${HEADER}"
        FAILED=1
      fi
    done
  fi

  if [[ "${CAN_EDIT_MAP[${AGENT}]}" == "false" ]]; then
    HANDOFF_FILE="${HANDOFF_DIR}/${AGENT}_to_${DOC_DELEGATE_MAP[${AGENT}]}.md"
    HAS_HANDOFF=0
    HAS_DELEGATE_MARK=0

    if [[ -f "${HANDOFF_FILE}" ]]; then
      HAS_HANDOFF=1
    fi

    if [[ -f "${LOG_FILE}" ]] && contains_literal "Log Writer: ${DOC_DELEGATE_MAP[${AGENT}]}" "${LOG_FILE}"; then
      if contains_literal "Source Agent: ${AGENT}" "${LOG_FILE}"; then
        HAS_DELEGATE_MARK=1
      fi
    fi

    if [[ "${HAS_HANDOFF}" -eq 0 && "${HAS_DELEGATE_MARK}" -eq 0 ]]; then
      echo "INVALID: ${AGENT} requires handoff or delegated log marker"
      FAILED=1
    fi
  fi
done

for AGENT in "${AGENTS[@]}"; do
  if ! contains_literal "| ${AGENT} |" "${THREAD_REGISTRY}"; then
    echo "INVALID: ${THREAD_REGISTRY} missing row for ${AGENT}"
    FAILED=1
  fi
done

for AGENT in "${AGENTS[@]}"; do
  THREAD_DATA="$(
    awk -F'|' -v a="${AGENT}" '
      /^\|/ {
        agent=$2; status=$4; started=$6; last_hb=$7; interval=$8; stuck=$9; escalation=$10; notes=$11;
        gsub(/^[ \t]+|[ \t]+$/, "", agent);
        gsub(/^[ \t]+|[ \t]+$/, "", status);
        gsub(/^[ \t]+|[ \t]+$/, "", started);
        gsub(/^[ \t]+|[ \t]+$/, "", last_hb);
        gsub(/^[ \t]+|[ \t]+$/, "", interval);
        gsub(/^[ \t]+|[ \t]+$/, "", stuck);
        gsub(/^[ \t]+|[ \t]+$/, "", escalation);
        gsub(/^[ \t]+|[ \t]+$/, "", notes);
        if (agent==a) {
          print status "\t" started "\t" last_hb "\t" interval "\t" stuck "\t" escalation "\t" notes;
          exit;
        }
      }
    ' "${THREAD_REGISTRY}"
  )"

  THREAD_STATUS="$(echo "${THREAD_DATA}" | cut -f1)"
  THREAD_STARTED="$(echo "${THREAD_DATA}" | cut -f2)"
  THREAD_LAST_HB="$(echo "${THREAD_DATA}" | cut -f3)"
  THREAD_INTERVAL="$(echo "${THREAD_DATA}" | cut -f4)"
  THREAD_STUCK="$(echo "${THREAD_DATA}" | cut -f5)"
  THREAD_ESCALATION="$(echo "${THREAD_DATA}" | cut -f6)"
  THREAD_NOTES="$(echo "${THREAD_DATA}" | cut -f7)"

  if [[ -z "${THREAD_STATUS}" ]]; then
    echo "INVALID: cannot parse thread row for ${AGENT}"
    FAILED=1
    continue
  fi

  if [[ "${THREAD_STATUS}" == "active" || "${THREAD_STATUS}" == "waiting-subagent" ]]; then
    if [[ -z "${THREAD_LAST_HB}" ]]; then
      echo "INVALID: ${AGENT} is ${THREAD_STATUS} but last_heartbeat_at is empty"
      FAILED=1
    fi
  fi

  if [[ -n "${THREAD_INTERVAL}" ]] && ! [[ "${THREAD_INTERVAL}" =~ ^[0-9]+$ ]]; then
    echo "INVALID: ${AGENT} heartbeat_interval_min should be an integer"
    FAILED=1
  fi

  if [[ -n "${THREAD_ESCALATION}" ]] && ! [[ "${THREAD_ESCALATION}" =~ ^[0-9]+$ ]]; then
    echo "INVALID: ${AGENT} escalation_count should be an integer"
    FAILED=1
  fi

  if [[ "${THREAD_STUCK}" != "no" && "${THREAD_STUCK}" != "yes-confirm-required" && "${THREAD_STUCK}" != "confirmed" && -n "${THREAD_STUCK}" ]]; then
    echo "INVALID: ${AGENT} has unsupported stuck_candidate value: ${THREAD_STUCK}"
    FAILED=1
  fi

  if [[ "${THREAD_STATUS}" == "stuck-confirmed" ]]; then
    if [[ "${THREAD_STUCK}" != "confirmed" ]]; then
      echo "INVALID: ${AGENT} status is stuck-confirmed but stuck_candidate is not confirmed"
      FAILED=1
    fi
    if [[ -z "${THREAD_NOTES}" ]]; then
      echo "INVALID: ${AGENT} status is stuck-confirmed but notes are empty"
      FAILED=1
    fi
  fi
done

for AGENT in "${AGENTS[@]}"; do
  if ! contains_literal "| ${AGENT} |" "${WORKTREE_REGISTRY}"; then
    echo "INVALID: ${WORKTREE_REGISTRY} missing row for ${AGENT}"
    FAILED=1
    continue
  fi

  if [[ "${CAN_EDIT_MAP[${AGENT}]}" == "true" ]]; then
    ROW_DATA="$(
      awk -F'|' -v a="${AGENT}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $2);
          if ($2==a) {
            branch=$4; path=$5; status=$6;
            gsub(/^[ \t]+|[ \t]+$/, "", branch);
            gsub(/^[ \t]+|[ \t]+$/, "", path);
            gsub(/^[ \t]+|[ \t]+$/, "", status);
            print branch "\t" path "\t" status;
            exit;
          }
        }
      ' "${WORKTREE_REGISTRY}"
    )"

    BRANCH="$(echo "${ROW_DATA}" | cut -f1)"
    WORKTREE_PATH="$(echo "${ROW_DATA}" | cut -f2)"
    WORKTREE_STATUS="$(echo "${ROW_DATA}" | cut -f3)"

    if [[ -z "${BRANCH}" || -z "${WORKTREE_PATH}" ]]; then
      echo "INVALID: ${WORKTREE_REGISTRY} editable agent ${AGENT} missing branch/path"
      FAILED=1
    fi

    if [[ "${WORKTREE_STATUS}" == "planned" ]]; then
      echo "INVALID: ${WORKTREE_REGISTRY} editable agent ${AGENT} still planned (run setup_run_worktrees.sh)"
      FAILED=1
    fi
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "Run ${RUN_ID}: FAILED checks."
  exit 3
fi

echo "Run ${RUN_ID}: logs, memory deltas, delegation, thread registry, and worktree checks passed."
