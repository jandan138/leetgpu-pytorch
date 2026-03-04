#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-Utf8Text {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Get-AgentDefinitions {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AgentsYamlPath
    )

    $defaultDocDelegate = "doc-writer"
    $agents = @()
    $current = $null

    foreach ($rawLine in (Get-Content -Path $AgentsYamlPath -Encoding utf8)) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        if ($line -match "^default_doc_delegate:\s*(\S+)\s*$") {
            $defaultDocDelegate = $Matches[1]
            continue
        }

        if ($line -match "^- id:\s*(\S+)\s*$") {
            if ($null -ne $current) {
                $agents += [PSCustomObject]$current
            }

            $current = [ordered]@{
                id            = $Matches[1]
                role          = ""
                can_edit_code = "false"
                can_run_tests = "false"
                doc_delegate  = ""
            }
            continue
        }

        if ($null -eq $current) {
            continue
        }

        if ($line -match "^role:\s*(.+)$") {
            $current.role = $Matches[1].Trim()
            continue
        }

        if ($line -match "^can_edit_code:\s*(\S+)\s*$") {
            $current.can_edit_code = $Matches[1].ToLowerInvariant()
            continue
        }

        if ($line -match "^can_run_tests:\s*(\S+)\s*$") {
            $current.can_run_tests = $Matches[1].ToLowerInvariant()
            continue
        }

        if ($line -match "^doc_delegate:\s*(\S+)\s*$") {
            $current.doc_delegate = $Matches[1]
            continue
        }
    }

    if ($null -ne $current) {
        $agents += [PSCustomObject]$current
    }

    if ($agents.Count -eq 0) {
        throw "No agents parsed from $AgentsYamlPath"
    }

    foreach ($agent in $agents) {
        if ([string]::IsNullOrWhiteSpace($agent.doc_delegate)) {
            $agent.doc_delegate = $defaultDocDelegate
        }

        if ([string]::IsNullOrWhiteSpace($agent.role)) {
            $agent.role = "Unspecified role"
        }
    }

    return $agents
}

function Get-Mission {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AgentId
    )

    switch ($AgentId) {
        "orchestrator" { return "Drive end-to-end delivery by decomposing goals, assigning work, and controlling decision quality." }
        "product-owner" { return "Translate user intent into stable requirements and acceptance criteria." }
        "researcher" { return "Produce reliable technical findings and options with evidence." }
        "architect" { return "Design interfaces and boundaries that reduce long-term complexity and risk." }
        "backend-coder" { return "Implement robust backend logic with clear compatibility behavior." }
        "ml-coder" { return "Implement model/training changes with reproducible experiment settings." }
        "data-engineer" { return "Maintain trustworthy data pipelines and schema compatibility." }
        "test-engineer" { return "Build and execute tests that protect critical behaviors from regressions." }
        "qa-reviewer" { return "Validate delivered behavior against acceptance criteria and edge cases." }
        "security-reviewer" { return "Assess security exposure and ensure safe handling of secrets and dependencies." }
        "performance-engineer" { return "Measure and improve performance using repeatable benchmarks." }
        "release-manager" { return "Control release readiness, rollback safety, and final go/no-go decisions." }
        "doc-writer" { return "Maintain high-quality, traceable technical documentation and delegated logs." }
        default { return "Deliver assigned responsibilities with evidence-backed decisions." }
    }
}

function Get-Capabilities {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AgentId
    )

    switch ($AgentId) {
        "orchestrator" {
            return @(
                "- Scope decomposition and sequencing across multiple agents.",
                "- Dependency and blocker tracking with explicit handoff ownership.",
                "- Enforcement of run-level completeness and acceptance gates."
            )
        }
        "product-owner" {
            return @(
                "- Clarify target outcomes, constraints, and priority tradeoffs.",
                "- Maintain stable done-definition for implementation and review.",
                "- Resolve requirement ambiguity before execution drift occurs."
            )
        }
        "researcher" {
            return @(
                "- Perform repository and config investigations with command evidence.",
                "- Compare implementation options and recommend default choices.",
                "- Surface technical risks early with concrete impact descriptions."
            )
        }
        "architect" {
            return @(
                "- Define module boundaries, contracts, and compatibility strategy.",
                "- Review cross-cutting concerns across code, config, and workflows.",
                "- Keep implementation decision-complete for coding agents."
            )
        }
        "backend-coder" {
            return @(
                "- Deliver backend logic changes with predictable behavior.",
                "- Preserve compatibility or document explicit migration needs.",
                "- Provide implementation evidence and targeted test output."
            )
        }
        "ml-coder" {
            return @(
                "- Implement model/training/inference path changes.",
                "- Record experiment configs, seeds, and result interpretation.",
                "- Document metric impact and known limitations."
            )
        }
        "data-engineer" {
            return @(
                "- Update data ingestion, transformation, and feature pipelines.",
                "- Track schema changes and backfill/compat requirements.",
                "- Ensure dataset lineage and reproducibility notes are complete."
            )
        }
        "test-engineer" {
            return @(
                "- Add or update tests to cover changed behaviors.",
                "- Execute regression suites and capture failing-to-passing transitions.",
                "- Report residual risk when full coverage is not feasible."
            )
        }
        "qa-reviewer" {
            return @(
                "- Perform scenario-based validation from user perspective.",
                "- Verify edge cases and expected/actual behavior deltas.",
                "- Issue quality verdict with explicit blocker/non-blocker labels."
            )
        }
        "security-reviewer" {
            return @(
                "- Review secrets handling, permission boundaries, and dependency risk.",
                "- Identify unsafe defaults and recommend mitigations.",
                "- Record security assumptions and unresolved exposure."
            )
        }
        "performance-engineer" {
            return @(
                "- Define benchmark methodology and baseline.",
                "- Compare before/after latency, throughput, and resource usage.",
                "- Explain bottlenecks and prioritized optimization actions."
            )
        }
        "release-manager" {
            return @(
                "- Validate release checklist and rollback preparedness.",
                "- Confirm evidence completeness across all agent logs.",
                "- Publish final release recommendation with risk summary."
            )
        }
        "doc-writer" {
            return @(
                "- Standardize logs, handoffs, and final summaries.",
                "- Write delegated documentation for non-edit agents.",
                "- Maintain traceability between claims and evidence paths."
            )
        }
        default {
            return @(
                "- Execute assigned responsibilities.",
                "- Keep records reproducible.",
                "- Escalate blockers promptly."
            )
        }
    }
}

function Get-NonGoals {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AgentId
    )

    switch ($AgentId) {
        "orchestrator" { return @("- Direct code implementation unless explicitly approved.", "- Modifying acceptance criteria without traceable rationale.") }
        "product-owner" { return @("- Direct code implementation unless explicitly approved.", "- Modifying acceptance criteria without traceable rationale.") }
        "researcher" { return @("- Direct code implementation unless explicitly approved.", "- Modifying acceptance criteria without traceable rationale.") }
        "qa-reviewer" { return @("- Direct code implementation unless explicitly approved.", "- Modifying acceptance criteria without traceable rationale.") }
        "security-reviewer" { return @("- Direct code implementation unless explicitly approved.", "- Modifying acceptance criteria without traceable rationale.") }
        "release-manager" { return @("- Direct code implementation unless explicitly approved.", "- Modifying acceptance criteria without traceable rationale.") }
        "doc-writer" { return @("- Altering technical decisions made by source agents.", "- Omitting delegation trace fields in rewritten logs.") }
        default { return @("- Expanding scope without orchestrator approval.", "- Shipping changes without test evidence.") }
    }
}

function Build-ProfileContent {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Agent
    )

    $agentId = $Agent.id
    $builder = New-Object System.Text.StringBuilder

    [void]$builder.AppendLine("# Agent Profile: $agentId")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Identity")
    [void]$builder.AppendLine("- Agent ID: $agentId")
    [void]$builder.AppendLine("- Role: $($Agent.role)")
    [void]$builder.AppendLine("- Permissions:")
    [void]$builder.AppendLine("  - can_edit_code: $($Agent.can_edit_code)")
    [void]$builder.AppendLine("  - can_run_tests: $($Agent.can_run_tests)")
    [void]$builder.AppendLine("- Doc Delegate: $($Agent.doc_delegate)")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Mission")
    [void]$builder.AppendLine("- $(Get-Mission -AgentId $agentId)")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Capabilities")
    foreach ($line in (Get-Capabilities -AgentId $agentId)) {
        [void]$builder.AppendLine($line)
    }
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Non-Goals")
    foreach ($line in (Get-NonGoals -AgentId $agentId)) {
        [void]$builder.AppendLine($line)
    }
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Inputs")
    [void]$builder.AppendLine("- Assigned task scope and acceptance criteria.")
    [void]$builder.AppendLine("- Current repository state, configs, and relevant logs.")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Outputs")
    [void]$builder.AppendLine("- Agent log updates under runs/<run_id>/logs/$agentId.md")
    [void]$builder.AppendLine("- Evidence paths and reproducible command outputs.")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Decision Rules")
    [void]$builder.AppendLine("- Prefer verifiable evidence over assumptions.")
    [void]$builder.AppendLine("- Escalate blockers early with concrete alternatives.")
    [void]$builder.AppendLine("- Keep changes within agreed scope and contracts.")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Escalation Rules")
    [void]$builder.AppendLine("- Escalate when dependencies are blocked for more than one execution cycle.")
    [void]$builder.AppendLine("- Escalate when requirements conflict with safety, security, or release constraints.")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Collaboration")
    [void]$builder.AppendLine("- Upstream: orchestrator assignment + product-owner constraints.")
    [void]$builder.AppendLine("- Downstream: handoff to dependent agents with evidence links.")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Evidence Requirements")
    [void]$builder.AppendLine("- List exact commands/files supporting each key claim.")
    [void]$builder.AppendLine("- Attach test outcomes or explicit reasons when tests are skipped.")
    [void]$builder.AppendLine("")
    [void]$builder.AppendLine("## Definition of Done")
    [void]$builder.AppendLine("- Required sections in agent log are complete and factual.")
    [void]$builder.AppendLine("- Decisions, risks, and next owner are explicitly recorded.")
    [void]$builder.AppendLine("- Evidence is sufficient for independent verification.")

    return $builder.ToString()
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$agentsYaml = Join-Path -Path $teamDir -ChildPath "agents.yaml"
$agentRoot = Join-Path -Path $teamDir -ChildPath "agents"

if (-not (Test-Path -LiteralPath $agentsYaml)) {
    Write-Host "ERROR: missing $agentsYaml"
    exit 1
}

$null = New-Item -ItemType Directory -Force -Path $agentRoot
$agents = Get-AgentDefinitions -AgentsYamlPath $agentsYaml
$memoryTemplatePath = Join-Path -Path $teamDir -ChildPath "templates/agent_memory_template.md"

foreach ($agent in $agents) {
    $agentDir = Join-Path -Path $agentRoot -ChildPath $agent.id
    $profilePath = Join-Path -Path $agentDir -ChildPath "profile.md"
    $memoryPath = Join-Path -Path $agentDir -ChildPath "memory.md"

    $null = New-Item -ItemType Directory -Force -Path $agentDir

    if (-not (Test-Path -LiteralPath $profilePath)) {
        $profileContent = Build-ProfileContent -Agent $agent
        Write-Utf8NoBom -Path $profilePath -Content $profileContent
    }

    if (-not (Test-Path -LiteralPath $memoryPath)) {
        $memoryContent = Read-Utf8Text -Path $memoryTemplatePath
        $memoryContent = $memoryContent.Replace("<agent_id>", $agent.id)
        $memoryContent = $memoryContent.Replace("<run_id_or_none>", "none")
        Write-Utf8NoBom -Path $memoryPath -Content $memoryContent
    }
}

Write-Host "Bootstrapped agent profiles/memories in: $agentRoot"
