#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash agent_team/scripts/update_agent_memory.sh <run_id>"
  exit 1
fi

RUN_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DELTA_DIR="${TEAM_DIR}/runs/${RUN_ID}/memory"
AGENT_ROOT="${TEAM_DIR}/agents"
STAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ ! -d "${DELTA_DIR}" ]]; then
  echo "ERROR: missing delta dir: ${DELTA_DIR}"
  exit 2
fi

FOUND=0
for DELTA_FILE in "${DELTA_DIR}"/*.delta.md; do
  if [[ ! -f "${DELTA_FILE}" ]]; then
    continue
  fi

  FOUND=1
  BASE_NAME="$(basename "${DELTA_FILE}")"
  AGENT_ID="${BASE_NAME%.delta.md}"
  MEMORY_FILE="${AGENT_ROOT}/${AGENT_ID}/memory.md"

  if [[ ! -f "${MEMORY_FILE}" ]]; then
    echo "WARN: missing memory file for ${AGENT_ID}, skip"
    continue
  fi

  MARKER="- Run: ${RUN_ID}"
  if rg -n --fixed-strings "${MARKER}" "${MEMORY_FILE}" >/dev/null 2>&1; then
    echo "SKIP: ${AGENT_ID} already merged for run ${RUN_ID}"
    continue
  fi

  {
    echo
    echo "- Run: ${RUN_ID}"
    echo "  - Summary: merged delta at ${STAMP}"
    echo "  - Source: runs/${RUN_ID}/memory/${AGENT_ID}.delta.md"
    echo "  - Notes:"
    sed 's/^/    /' "${DELTA_FILE}"
  } >> "${MEMORY_FILE}"

  echo "UPDATED: ${MEMORY_FILE}"
done

if [[ "${FOUND}" -eq 0 ]]; then
  echo "ERROR: no delta files found in ${DELTA_DIR}"
  exit 3
fi

echo "Memory merge complete for run ${RUN_ID}."
