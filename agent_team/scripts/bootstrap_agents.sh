#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTS_YAML="${TEAM_DIR}/agents.yaml"
AGENT_ROOT="${TEAM_DIR}/agents"

if [[ ! -f "${AGENTS_YAML}" ]]; then
  echo "ERROR: missing ${AGENTS_YAML}"
  exit 1
fi

mkdir -p "${AGENT_ROOT}"

mission_for() {
  case "$1" in
    orchestrator) echo "Drive end-to-end delivery by decomposing goals, assigning work, and controlling decision quality." ;;
    product-owner) echo "Translate user intent into stable requirements and acceptance criteria." ;;
    researcher) echo "Produce reliable technical findings and options with evidence." ;;
    architect) echo "Design interfaces and boundaries that reduce long-term complexity and risk." ;;
    backend-coder) echo "Implement robust backend logic with clear compatibility behavior." ;;
    ml-coder) echo "Implement model/training changes with reproducible experiment settings." ;;
    data-engineer) echo "Maintain trustworthy data pipelines and schema compatibility." ;;
    test-engineer) echo "Build and execute tests that protect critical behaviors from regressions." ;;
    qa-reviewer) echo "Validate delivered behavior against acceptance criteria and edge cases." ;;
    security-reviewer) echo "Assess security exposure and ensure safe handling of secrets and dependencies." ;;
    performance-engineer) echo "Measure and improve performance using repeatable benchmarks." ;;
    release-manager) echo "Control release readiness, rollback safety, and final go/no-go decisions." ;;
    doc-writer) echo "Maintain high-quality, traceable technical documentation and delegated logs." ;;
    *) echo "Deliver assigned responsibilities with evidence-backed decisions." ;;
  esac
}

capabilities_for() {
  case "$1" in
    orchestrator) cat <<'EOC'
- Scope decomposition and sequencing across multiple agents.
- Dependency and blocker tracking with explicit handoff ownership.
- Enforcement of run-level completeness and acceptance gates.
EOC
      ;;
    product-owner) cat <<'EOC'
- Clarify target outcomes, constraints, and priority tradeoffs.
- Maintain stable done-definition for implementation and review.
- Resolve requirement ambiguity before execution drift occurs.
EOC
      ;;
    researcher) cat <<'EOC'
- Perform repository and config investigations with command evidence.
- Compare implementation options and recommend default choices.
- Surface technical risks early with concrete impact descriptions.
EOC
      ;;
    architect) cat <<'EOC'
- Define module boundaries, contracts, and compatibility strategy.
- Review cross-cutting concerns across code, config, and workflows.
- Keep implementation decision-complete for coding agents.
EOC
      ;;
    backend-coder) cat <<'EOC'
- Deliver backend logic changes with predictable behavior.
- Preserve compatibility or document explicit migration needs.
- Provide implementation evidence and targeted test output.
EOC
      ;;
    ml-coder) cat <<'EOC'
- Implement model/training/inference path changes.
- Record experiment configs, seeds, and result interpretation.
- Document metric impact and known limitations.
EOC
      ;;
    data-engineer) cat <<'EOC'
- Update data ingestion, transformation, and feature pipelines.
- Track schema changes and backfill/compat requirements.
- Ensure dataset lineage and reproducibility notes are complete.
EOC
      ;;
    test-engineer) cat <<'EOC'
- Add or update tests to cover changed behaviors.
- Execute regression suites and capture failing-to-passing transitions.
- Report residual risk when full coverage is not feasible.
EOC
      ;;
    qa-reviewer) cat <<'EOC'
- Perform scenario-based validation from user perspective.
- Verify edge cases and expected/actual behavior deltas.
- Issue quality verdict with explicit blocker/non-blocker labels.
EOC
      ;;
    security-reviewer) cat <<'EOC'
- Review secrets handling, permission boundaries, and dependency risk.
- Identify unsafe defaults and recommend mitigations.
- Record security assumptions and unresolved exposure.
EOC
      ;;
    performance-engineer) cat <<'EOC'
- Define benchmark methodology and baseline.
- Compare before/after latency, throughput, and resource usage.
- Explain bottlenecks and prioritized optimization actions.
EOC
      ;;
    release-manager) cat <<'EOC'
- Validate release checklist and rollback preparedness.
- Confirm evidence completeness across all agent logs.
- Publish final release recommendation with risk summary.
EOC
      ;;
    doc-writer) cat <<'EOC'
- Standardize logs, handoffs, and final summaries.
- Write delegated documentation for non-edit agents.
- Maintain traceability between claims and evidence paths.
EOC
      ;;
    *) cat <<'EOC'
- Execute assigned responsibilities.
- Keep records reproducible.
- Escalate blockers promptly.
EOC
      ;;
  esac
}

non_goals_for() {
  case "$1" in
    orchestrator|product-owner|researcher|qa-reviewer|security-reviewer|release-manager)
      cat <<'EON'
- Direct code implementation unless explicitly approved.
- Modifying acceptance criteria without traceable rationale.
EON
      ;;
    doc-writer)
      cat <<'EON'
- Altering technical decisions made by source agents.
- Omitting delegation trace fields in rewritten logs.
EON
      ;;
    *)
      cat <<'EON'
- Expanding scope without orchestrator approval.
- Shipping changes without test evidence.
EON
      ;;
  esac
}

while IFS=$'\t' read -r AGENT_ID ROLE CAN_EDIT CAN_RUN DOC_DELEGATE; do
  AGENT_DIR="${AGENT_ROOT}/${AGENT_ID}"
  PROFILE_FILE="${AGENT_DIR}/profile.md"
  MEMORY_FILE="${AGENT_DIR}/memory.md"
  mkdir -p "${AGENT_DIR}"

  if [[ ! -f "${PROFILE_FILE}" ]]; then
    {
      echo "# Agent Profile: ${AGENT_ID}"
      echo
      echo "## Identity"
      echo "- Agent ID: ${AGENT_ID}"
      echo "- Role: ${ROLE}"
      echo "- Permissions:"
      echo "  - can_edit_code: ${CAN_EDIT}"
      echo "  - can_run_tests: ${CAN_RUN}"
      echo "- Doc Delegate: ${DOC_DELEGATE}"
      echo
      echo "## Mission"
      echo "- $(mission_for "${AGENT_ID}")"
      echo
      echo "## Capabilities"
      capabilities_for "${AGENT_ID}"
      echo
      echo "## Non-Goals"
      non_goals_for "${AGENT_ID}"
      echo
      echo "## Inputs"
      echo "- Assigned task scope and acceptance criteria."
      echo "- Current repository state, configs, and relevant logs."
      echo
      echo "## Outputs"
      echo "- Agent log updates under runs/<run_id>/logs/${AGENT_ID}.md"
      echo "- Evidence paths and reproducible command outputs."
      echo
      echo "## Decision Rules"
      echo "- Prefer verifiable evidence over assumptions."
      echo "- Escalate blockers early with concrete alternatives."
      echo "- Keep changes within agreed scope and contracts."
      echo
      echo "## Escalation Rules"
      echo "- Escalate when dependencies are blocked for more than one execution cycle."
      echo "- Escalate when requirements conflict with safety, security, or release constraints."
      echo
      echo "## Collaboration"
      echo "- Upstream: orchestrator assignment + product-owner constraints."
      echo "- Downstream: handoff to dependent agents with evidence links."
      echo
      echo "## Evidence Requirements"
      echo "- List exact commands/files supporting each key claim."
      echo "- Attach test outcomes or explicit reasons when tests are skipped."
      echo
      echo "## Definition of Done"
      echo "- Required sections in agent log are complete and factual."
      echo "- Decisions, risks, and next owner are explicitly recorded."
      echo "- Evidence is sufficient for independent verification."
    } > "${PROFILE_FILE}"
  fi

  if [[ ! -f "${MEMORY_FILE}" ]]; then
    cp "${TEAM_DIR}/templates/agent_memory_template.md" "${MEMORY_FILE}"
    sed -i "s/<agent_id>/${AGENT_ID}/g" "${MEMORY_FILE}"
    sed -i "s/<run_id_or_none>/none/g" "${MEMORY_FILE}"
  fi
done < <(
  awk '
    /- id:/ {id=$3; order[++n]=id}
    /^\s*role:/ {
      line=$0
      sub(/^[[:space:]]*role:[[:space:]]*/, "", line)
      role[id]=line
    }
    /can_edit_code:/ {edit[id]=$2}
    /can_run_tests:/ {run[id]=$2}
    /doc_delegate:/ {doc[id]=$2}
    END {
      for (i=1; i<=n; i++) {
        id=order[i]
        print id "\t" role[id] "\t" edit[id] "\t" run[id] "\t" doc[id]
      }
    }
  ' "${AGENTS_YAML}"
)

echo "Bootstrapped agent profiles/memories in: ${AGENT_ROOT}"
