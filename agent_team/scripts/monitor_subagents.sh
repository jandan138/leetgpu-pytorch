#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
  echo "Usage: bash agent_team/scripts/monitor_subagents.sh <run_id> [--interval-min N] [--stuck-min N]"
  exit 1
fi

RUN_ID="$1"
SHIFT_ARGS=("${@:2}")
DEFAULT_INTERVAL_MIN=10
STUCK_MIN=45

IDX=0
while [[ ${IDX} -lt ${#SHIFT_ARGS[@]} ]]; do
  ARG="${SHIFT_ARGS[${IDX}]}"
  case "${ARG}" in
    --interval-min)
      IDX=$((IDX + 1))
      DEFAULT_INTERVAL_MIN="${SHIFT_ARGS[${IDX}]:-}"
      ;;
    --stuck-min)
      IDX=$((IDX + 1))
      STUCK_MIN="${SHIFT_ARGS[${IDX}]:-}"
      ;;
    *)
      echo "ERROR: unknown option ${ARG}"
      exit 2
      ;;
  esac
  IDX=$((IDX + 1))
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
THREAD_REGISTRY="${TEAM_DIR}/runs/${RUN_ID}/threads/registry.md"

if [[ ! -f "${THREAD_REGISTRY}" ]]; then
  echo "ERROR: thread registry not found: ${THREAD_REGISTRY}"
  exit 2
fi

if ! [[ "${DEFAULT_INTERVAL_MIN}" =~ ^[0-9]+$ ]] || ! [[ "${STUCK_MIN}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --interval-min and --stuck-min must be integers"
  exit 2
fi

if [[ "${STUCK_MIN}" -lt "${DEFAULT_INTERVAL_MIN}" ]]; then
  echo "ERROR: --stuck-min should be >= --interval-min"
  exit 2
fi

TMP_PREFIX="$(mktemp)"
awk '
  {print}
  /^\| ---/ {exit}
' "${THREAD_REGISTRY}" > "${TMP_PREFIX}"

TMP_ROWS="$(mktemp)"
CHANGED=0
NOW_EPOCH="$(date -u +%s)"
NOW_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

while IFS=$'\t' read -r AGENT THREAD STATUS OWNER STARTED LAST_HB INTERVAL STUCK ESCALATION NOTES; do
  NEW_STATUS="${STATUS}"
  NEW_STUCK="${STUCK}"
  NEW_NOTES="${NOTES}"

  USE_INTERVAL="${INTERVAL}"
  if ! [[ "${USE_INTERVAL}" =~ ^[0-9]+$ ]]; then
    USE_INTERVAL="${DEFAULT_INTERVAL_MIN}"
  fi

  REF_TS="${LAST_HB}"
  if [[ -z "${REF_TS}" ]]; then
    REF_TS="${STARTED}"
  fi

  if [[ "${STATUS}" == "active" || "${STATUS}" == "waiting-subagent" ]]; then
    if [[ -n "${REF_TS}" ]]; then
      REF_EPOCH="$(date -u -d "${REF_TS}" +%s 2>/dev/null || true)"
      if [[ -n "${REF_EPOCH}" ]]; then
        ELAPSED_MIN=$(( (NOW_EPOCH - REF_EPOCH) / 60 ))

        if [[ "${ELAPSED_MIN}" -ge "${STUCK_MIN}" ]]; then
          NEW_STATUS="stuck-suspected"
          NEW_STUCK="yes-confirm-required"
          ALERT_NOTE="[${NOW_UTC}] monitor: heartbeat stale ${ELAPSED_MIN}m >= ${STUCK_MIN}m; manual confirm required"
          if [[ -z "${NEW_NOTES}" ]]; then
            NEW_NOTES="${ALERT_NOTE}"
          else
            NEW_NOTES="${NEW_NOTES}; ${ALERT_NOTE}"
          fi
          CHANGED=1
          echo "SUSPECTED_STUCK ${AGENT}: stale ${ELAPSED_MIN}m, set status=stuck-suspected"
        elif [[ "${ELAPSED_MIN}" -ge "${USE_INTERVAL}" ]]; then
          echo "LATE_HEARTBEAT ${AGENT}: stale ${ELAPSED_MIN}m (interval ${USE_INTERVAL}m), continue waiting"
        else
          echo "HEALTHY ${AGENT}: stale ${ELAPSED_MIN}m"
        fi
      else
        NEW_STATUS="stuck-suspected"
        NEW_STUCK="yes-confirm-required"
        ALERT_NOTE="[${NOW_UTC}] monitor: cannot parse heartbeat time '${REF_TS}', manual confirm required"
        if [[ -z "${NEW_NOTES}" ]]; then
          NEW_NOTES="${ALERT_NOTE}"
        else
          NEW_NOTES="${NEW_NOTES}; ${ALERT_NOTE}"
        fi
        CHANGED=1
        echo "SUSPECTED_STUCK ${AGENT}: unparseable heartbeat timestamp"
      fi
    else
      NEW_STATUS="stuck-suspected"
      NEW_STUCK="yes-confirm-required"
      ALERT_NOTE="[${NOW_UTC}] monitor: missing started_at/last_heartbeat_at, manual confirm required"
      if [[ -z "${NEW_NOTES}" ]]; then
        NEW_NOTES="${ALERT_NOTE}"
      else
        NEW_NOTES="${NEW_NOTES}; ${ALERT_NOTE}"
      fi
      CHANGED=1
      echo "SUSPECTED_STUCK ${AGENT}: no heartbeat timestamp"
    fi
  fi

  SAFE_NOTES="${NEW_NOTES//|/\\/|}"
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "${AGENT}" "${THREAD}" "${NEW_STATUS}" "${OWNER}" "${STARTED}" "${LAST_HB}" "${USE_INTERVAL}" "${NEW_STUCK}" "${ESCALATION}" "${SAFE_NOTES}" >> "${TMP_ROWS}"
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

cat "${TMP_PREFIX}" > "${THREAD_REGISTRY}"
cat "${TMP_ROWS}" >> "${THREAD_REGISTRY}"
rm -f "${TMP_PREFIX}" "${TMP_ROWS}"

if [[ "${CHANGED}" -eq 1 ]]; then
  echo "Updated ${THREAD_REGISTRY} with stuck-suspected candidates."
else
  echo "No stuck-suspected updates. Continue waiting for sub-agents."
fi
