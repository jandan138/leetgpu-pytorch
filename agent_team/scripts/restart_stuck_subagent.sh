#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>"
  exit 1
fi

RUN_ID="$1"
AGENT_ID="$2"
NEW_THREAD_ID="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
THREAD_REGISTRY="${TEAM_DIR}/runs/${RUN_ID}/threads/registry.md"

if [[ ! -f "${THREAD_REGISTRY}" ]]; then
  echo "ERROR: thread registry not found: ${THREAD_REGISTRY}"
  exit 2
fi

FOUND_ROW=0
TMP_PREFIX="$(mktemp)"
TMP_ROWS="$(mktemp)"
NOW_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

awk '
  {print}
  /^\| ---/ {exit}
' "${THREAD_REGISTRY}" > "${TMP_PREFIX}"

while IFS=$'\t' read -r AGENT THREAD STATUS OWNER STARTED LAST_HB INTERVAL STUCK ESCALATION NOTES; do
  NEW_AGENT="${AGENT}"
  NEW_THREAD="${THREAD}"
  NEW_STATUS="${STATUS}"
  NEW_OWNER="${OWNER}"
  NEW_STARTED="${STARTED}"
  NEW_LAST_HB="${LAST_HB}"
  NEW_INTERVAL="${INTERVAL}"
  NEW_STUCK="${STUCK}"
  NEW_ESCALATION="${ESCALATION}"
  NEW_NOTES="${NOTES}"

  if [[ "${AGENT}" == "${AGENT_ID}" ]]; then
    FOUND_ROW=1
    if [[ "${STATUS}" != "stuck-confirmed" && "${STUCK}" != "confirmed" ]]; then
      echo "ERROR: ${AGENT_ID} is not stuck-confirmed. Current status=${STATUS}, stuck_candidate=${STUCK}"
      rm -f "${TMP_PREFIX}" "${TMP_ROWS}"
      exit 3
    fi

    OLD_THREAD="${THREAD}"
    if ! [[ "${NEW_ESCALATION}" =~ ^[0-9]+$ ]]; then
      NEW_ESCALATION="0"
    fi
    NEW_ESCALATION="$((NEW_ESCALATION + 1))"
    NEW_THREAD="${NEW_THREAD_ID}"
    NEW_STATUS="active"
    NEW_STARTED="${NOW_UTC}"
    NEW_LAST_HB="${NOW_UTC}"
    if ! [[ "${NEW_INTERVAL}" =~ ^[0-9]+$ ]]; then
      NEW_INTERVAL="10"
    fi
    NEW_STUCK="no"
    RESTART_NOTE="[${NOW_UTC}] restart: old_thread=${OLD_THREAD}, new_thread=${NEW_THREAD_ID}, escalation=${NEW_ESCALATION}"
    if [[ -z "${NEW_NOTES}" ]]; then
      NEW_NOTES="${RESTART_NOTE}"
    else
      NEW_NOTES="${NEW_NOTES}; ${RESTART_NOTE}"
    fi
  fi

  SAFE_NOTES="${NEW_NOTES//|/\\/|}"
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "${NEW_AGENT}" "${NEW_THREAD}" "${NEW_STATUS}" "${NEW_OWNER}" "${NEW_STARTED}" "${NEW_LAST_HB}" "${NEW_INTERVAL}" "${NEW_STUCK}" "${NEW_ESCALATION}" "${SAFE_NOTES}" >> "${TMP_ROWS}"
done < <(
  awk -F'|' '
    /^\|/ {
      agent=$2; thread=$3; status=$4; owner=$5; started=$6; last_hb=$7; interval=$8; stuck=$9; escalation=$10; notes=$11;
      gsub(/^[ \t]+|[ \t]+$/, "", agent);
      gsub(/^[ \t]+|[ \t]+$/, "", thread);
      gsub(/^[ \t]+|[ \t]+$/, "", status);
      gsub(/^[ \t]+|[ \t]+$/, "", owner);
      gsub(/^[ \t]+|[ \t]+$/, "", started);
      gsub(/^[ \t]+|[ \t]+$/, "", last_hb);
      gsub(/^[ \t]+|[ \t]+$/, "", interval);
      gsub(/^[ \t]+|[ \t]+$/, "", stuck);
      gsub(/^[ \t]+|[ \t]+$/, "", escalation);
      gsub(/^[ \t]+|[ \t]+$/, "", notes);
      if (agent != "" && agent != "agent_id" && agent != "---") {
        print agent "\t" thread "\t" status "\t" owner "\t" started "\t" last_hb "\t" interval "\t" stuck "\t" escalation "\t" notes;
      }
    }
  ' "${THREAD_REGISTRY}"
)

if [[ "${FOUND_ROW}" -eq 0 ]]; then
  echo "ERROR: agent ${AGENT_ID} not found in ${THREAD_REGISTRY}"
  rm -f "${TMP_PREFIX}" "${TMP_ROWS}"
  exit 2
fi

cat "${TMP_PREFIX}" > "${THREAD_REGISTRY}"
cat "${TMP_ROWS}" >> "${THREAD_REGISTRY}"
rm -f "${TMP_PREFIX}" "${TMP_ROWS}"

echo "Restarted ${AGENT_ID} with new thread ${NEW_THREAD_ID} (status=active, stuck_candidate=no)."
