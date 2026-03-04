#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash agent_team/scripts/init_run.sh <run_id>"
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
WORKTREE_DIR="${RUN_DIR}/worktrees"
ARTIFACT_DIR="${RUN_DIR}/artifacts"

if [[ ! -f "${AGENTS_YAML}" ]]; then
  echo "ERROR: missing ${AGENTS_YAML}"
  exit 2
fi

mkdir -p "${LOG_DIR}" "${HANDOFF_DIR}" "${MEMORY_DIR}" "${THREAD_DIR}" "${WORKTREE_DIR}" "${ARTIFACT_DIR}"

declare -a AGENTS

declare -A CAN_EDIT_MAP

declare -A CAN_RUN_MAP

declare -A DOC_DELEGATE_MAP

while IFS=$'\t' read -r AGENT_ID CAN_EDIT CAN_RUN DOC_DELEGATE; do
  AGENTS+=("${AGENT_ID}")
  CAN_EDIT_MAP["${AGENT_ID}"]="${CAN_EDIT}"
  CAN_RUN_MAP["${AGENT_ID}"]="${CAN_RUN}"
  DOC_DELEGATE_MAP["${AGENT_ID}"]="${DOC_DELEGATE}"
done < <(
  awk '
    /- id:/ {id=$3; order[++n]=id}
    /can_edit_code:/ {edit[id]=$2}
    /can_run_tests:/ {run[id]=$2}
    /doc_delegate:/ {doc[id]=$2}
    END {
      for (i=1; i<=n; i++) {
        id=order[i]
        print id "\t" edit[id] "\t" run[id] "\t" doc[id]
      }
    }
  ' "${AGENTS_YAML}"
)

for AGENT in "${AGENTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${AGENT}.md"
  MEMORY_FILE="${MEMORY_DIR}/${AGENT}.delta.md"

  LOG_WRITER="${AGENT}"
  if [[ "${CAN_EDIT_MAP[${AGENT}]}" == "false" ]]; then
    LOG_WRITER="${DOC_DELEGATE_MAP[${AGENT}]}"
  fi

  if [[ ! -f "${LOG_FILE}" ]]; then
    cp "${TEAM_DIR}/templates/agent_log_template.md" "${LOG_FILE}"
    sed -i "s/<agent_id>/${AGENT}/g" "${LOG_FILE}"
    sed -i "s/<run_id>/${RUN_ID}/g" "${LOG_FILE}"
    sed -i "s/<log_writer>/${LOG_WRITER}/g" "${LOG_FILE}"
    sed -i "s/<can_edit_code>/${CAN_EDIT_MAP[${AGENT}]}/g" "${LOG_FILE}"
    sed -i "s/<can_run_tests>/${CAN_RUN_MAP[${AGENT}]}/g" "${LOG_FILE}"
  fi

  if [[ ! -f "${MEMORY_FILE}" ]]; then
    cp "${TEAM_DIR}/templates/memory_delta_template.md" "${MEMORY_FILE}"
    sed -i "s/<agent_id>/${AGENT}/g" "${MEMORY_FILE}"
    sed -i "s/<run_id>/${RUN_ID}/g" "${MEMORY_FILE}"
    if [[ "${CAN_EDIT_MAP[${AGENT}]}" == "false" ]]; then
      sed -i "s/- Recorder: ${AGENT}/- Recorder: ${DOC_DELEGATE_MAP[${AGENT}]}/" "${MEMORY_FILE}"
    fi
  fi

  if [[ "${CAN_EDIT_MAP[${AGENT}]}" == "false" ]]; then
    HANDOFF_FILE="${HANDOFF_DIR}/${AGENT}_to_${DOC_DELEGATE_MAP[${AGENT}]}.md"
    if [[ ! -f "${HANDOFF_FILE}" ]]; then
      cp "${TEAM_DIR}/templates/handoff_to_doc_template.md" "${HANDOFF_FILE}"
      sed -i "s/<run_id>/${RUN_ID}/g" "${HANDOFF_FILE}"
      sed -i "s/<agent_id>/${AGENT}/g" "${HANDOFF_FILE}"
      sed -i "s/Target Agent: doc-writer/Target Agent: ${DOC_DELEGATE_MAP[${AGENT}]}/" "${HANDOFF_FILE}"
    fi
  fi
done

THREAD_REGISTRY="${THREAD_DIR}/registry.md"
if [[ ! -f "${THREAD_REGISTRY}" ]]; then
  cp "${TEAM_DIR}/templates/thread_registry_template.md" "${THREAD_REGISTRY}"
  sed -i "s/<run_id>/${RUN_ID}/g" "${THREAD_REGISTRY}"
  for AGENT in "${AGENTS[@]}"; do
    printf "| %s |  | planned | %s |  |  | 10 | no | 0 |  |\n" "${AGENT}" "${AGENT}" >> "${THREAD_REGISTRY}"
  done
fi

WORKTREE_REGISTRY="${WORKTREE_DIR}/registry.md"
if [[ ! -f "${WORKTREE_REGISTRY}" ]]; then
  cp "${TEAM_DIR}/templates/worktree_registry_template.md" "${WORKTREE_REGISTRY}"
  sed -i "s/<run_id>/${RUN_ID}/g" "${WORKTREE_REGISTRY}"
  for AGENT in "${AGENTS[@]}"; do
    printf "| %s | %s |  |  | planned |  |\n" "${AGENT}" "${CAN_EDIT_MAP[${AGENT}]}" >> "${WORKTREE_REGISTRY}"
  done
fi

echo "Initialized agent-team run at: ${RUN_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Handoff notes: ${HANDOFF_DIR}"
echo "Memory deltas: ${MEMORY_DIR}"
echo "Thread registry: ${THREAD_REGISTRY}"
echo "Worktree registry: ${WORKTREE_REGISTRY}"
echo "Artifacts: ${ARTIFACT_DIR}"
