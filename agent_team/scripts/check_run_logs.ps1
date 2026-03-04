#!/usr/bin/env pwsh
param(
    [string]$RunId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/check_run_logs.ps1 -RunId <run_id>"
    exit 1
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
                can_edit_code = "false"
                can_run_tests = "false"
                doc_delegate  = ""
            }
            continue
        }

        if ($null -eq $current) {
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
    }

    return $agents
}

function Test-ContainsLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Literal
    )

    return [bool](Select-String -Path $Path -SimpleMatch -Pattern $Literal -Quiet)
}

function Parse-MarkdownRowCells {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Line
    )

    $trimmed = $Line.Trim()
    if ($trimmed -notmatch "^\|(.+)\|$") {
        return $null
    }

    $cells = @()
    foreach ($cell in ($Matches[1] -split "\|")) {
        $cells += $cell.Trim()
    }

    return $cells
}

function Get-MarkdownRowByAgent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$AgentId,
        [Parameter(Mandatory = $true)]
        [int]$MinimumCellCount
    )

    foreach ($line in (Get-Content -Path $Path -Encoding utf8)) {
        $cells = Parse-MarkdownRowCells -Line $line
        if ($null -eq $cells) {
            continue
        }

        if ($cells.Count -lt $MinimumCellCount) {
            continue
        }

        $rowAgent = $cells[0]
        if ([string]::IsNullOrWhiteSpace($rowAgent) -or $rowAgent -eq "agent_id" -or $rowAgent -eq "---") {
            continue
        }

        if ($rowAgent -eq $AgentId) {
            return $cells
        }
    }

    return $null
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$agentsYaml = Join-Path -Path $teamDir -ChildPath "agents.yaml"
$runDir = Join-Path -Path $teamDir -ChildPath ("runs/{0}" -f $RunId)
$logDir = Join-Path -Path $runDir -ChildPath "logs"
$handoffDir = Join-Path -Path $runDir -ChildPath "handoff"
$memoryDir = Join-Path -Path $runDir -ChildPath "memory"
$threadRegistry = Join-Path -Path $runDir -ChildPath "threads/registry.md"
$worktreeRegistry = Join-Path -Path $runDir -ChildPath "worktrees/registry.md"

if (-not (Test-Path -LiteralPath $agentsYaml)) {
    Write-Host "ERROR: missing $agentsYaml"
    exit 2
}

if (-not (Test-Path -LiteralPath $logDir)) {
    Write-Host "ERROR: log directory not found: $logDir"
    exit 2
}

if (-not (Test-Path -LiteralPath $memoryDir)) {
    Write-Host "ERROR: memory directory not found: $memoryDir"
    exit 2
}

if (-not (Test-Path -LiteralPath $threadRegistry)) {
    Write-Host "ERROR: thread registry not found: $threadRegistry"
    exit 2
}

if (-not (Test-ContainsLiteral -Path $threadRegistry -Literal "last_heartbeat_at")) {
    Write-Host "ERROR: thread registry header is outdated, missing last_heartbeat_at"
    exit 2
}

if (-not (Test-ContainsLiteral -Path $threadRegistry -Literal "heartbeat_interval_min")) {
    Write-Host "ERROR: thread registry header is outdated, missing heartbeat_interval_min"
    exit 2
}

if (-not (Test-ContainsLiteral -Path $threadRegistry -Literal "stuck_candidate")) {
    Write-Host "ERROR: thread registry header is outdated, missing stuck_candidate"
    exit 2
}

if (-not (Test-Path -LiteralPath $worktreeRegistry)) {
    Write-Host "ERROR: worktree registry not found: $worktreeRegistry"
    exit 2
}

$agents = Get-AgentDefinitions -AgentsYamlPath $agentsYaml

$logHeaders = @(
    "## 1) Task & Inputs",
    "## 2) Research",
    "## 3) Code Changes",
    "## 4) Test Execution",
    "## 5) Risks & Open Items",
    "## 6) Handoff",
    "## 7) Evidence"
)

$memoryHeaders = @(
    "## 1) Delta Summary",
    "## 2) New Stable Preferences",
    "## 3) Pitfalls Learned",
    "## 4) Reusable Patterns",
    "## 5) Evidence Links"
)

$failed = $false

foreach ($agent in $agents) {
    $agentId = $agent.id
    $logFile = Join-Path -Path $logDir -ChildPath ("{0}.md" -f $agentId)
    $memoryFile = Join-Path -Path $memoryDir -ChildPath ("{0}.delta.md" -f $agentId)

    if (-not (Test-Path -LiteralPath $logFile)) {
        Write-Host "MISSING: $logFile"
        $failed = $true
    } else {
        foreach ($header in $logHeaders) {
            if (-not (Test-ContainsLiteral -Path $logFile -Literal $header)) {
                Write-Host "INVALID: $logFile missing header: $header"
                $failed = $true
            }
        }
    }

    if (-not (Test-Path -LiteralPath $memoryFile)) {
        Write-Host "MISSING: $memoryFile"
        $failed = $true
    } else {
        foreach ($header in $memoryHeaders) {
            if (-not (Test-ContainsLiteral -Path $memoryFile -Literal $header)) {
                Write-Host "INVALID: $memoryFile missing header: $header"
                $failed = $true
            }
        }
    }

    if ($agent.can_edit_code -eq "false") {
        $handoffFile = Join-Path -Path $handoffDir -ChildPath ("{0}_to_{1}.md" -f $agentId, $agent.doc_delegate)
        $hasHandoff = Test-Path -LiteralPath $handoffFile
        $hasDelegateMark = $false

        if (Test-Path -LiteralPath $logFile) {
            $hasDelegateMark = (Test-ContainsLiteral -Path $logFile -Literal ("Log Writer: {0}" -f $agent.doc_delegate)) -and
                (Test-ContainsLiteral -Path $logFile -Literal ("Source Agent: {0}" -f $agentId))
        }

        if ((-not $hasHandoff) -and (-not $hasDelegateMark)) {
            Write-Host "INVALID: $agentId requires handoff or delegated log marker"
            $failed = $true
        }
    }
}

foreach ($agent in $agents) {
    $agentId = $agent.id
    $threadRow = Get-MarkdownRowByAgent -Path $threadRegistry -AgentId $agentId -MinimumCellCount 10
    if ($null -eq $threadRow) {
        Write-Host "INVALID: $threadRegistry missing or unparsable row for $agentId"
        $failed = $true
        continue
    }

    $status = $threadRow[2]
    $lastHeartbeatAt = $threadRow[5]
    $heartbeatIntervalMin = $threadRow[6]
    $stuckCandidate = $threadRow[7]
    $escalationCount = $threadRow[8]
    $notes = $threadRow[9]

    if (($status -eq "active" -or $status -eq "waiting-subagent") -and [string]::IsNullOrWhiteSpace($lastHeartbeatAt)) {
        Write-Host "INVALID: $agentId is $status but last_heartbeat_at is empty"
        $failed = $true
    }

    if (-not [string]::IsNullOrWhiteSpace($heartbeatIntervalMin) -and ($heartbeatIntervalMin -notmatch "^\d+$")) {
        Write-Host "INVALID: $agentId heartbeat_interval_min should be an integer"
        $failed = $true
    }

    if (-not [string]::IsNullOrWhiteSpace($escalationCount) -and ($escalationCount -notmatch "^\d+$")) {
        Write-Host "INVALID: $agentId escalation_count should be an integer"
        $failed = $true
    }

    if (-not [string]::IsNullOrWhiteSpace($stuckCandidate)) {
        if (@("no", "yes-confirm-required", "confirmed") -notcontains $stuckCandidate) {
            Write-Host "INVALID: $agentId has unsupported stuck_candidate value: $stuckCandidate"
            $failed = $true
        }
    }

    if ($status -eq "stuck-confirmed") {
        if ($stuckCandidate -ne "confirmed") {
            Write-Host "INVALID: $agentId status is stuck-confirmed but stuck_candidate is not confirmed"
            $failed = $true
        }

        if ([string]::IsNullOrWhiteSpace($notes)) {
            Write-Host "INVALID: $agentId status is stuck-confirmed but notes are empty"
            $failed = $true
        }
    }
}

foreach ($agent in $agents) {
    $agentId = $agent.id
    $worktreeRow = Get-MarkdownRowByAgent -Path $worktreeRegistry -AgentId $agentId -MinimumCellCount 6
    if ($null -eq $worktreeRow) {
        Write-Host "INVALID: $worktreeRegistry missing or unparsable row for $agentId"
        $failed = $true
        continue
    }

    if ($agent.can_edit_code -eq "true") {
        $branch = $worktreeRow[2]
        $worktreePath = $worktreeRow[3]
        $worktreeStatus = $worktreeRow[4]

        if ([string]::IsNullOrWhiteSpace($branch) -or [string]::IsNullOrWhiteSpace($worktreePath)) {
            Write-Host "INVALID: $worktreeRegistry editable agent $agentId missing branch/path"
            $failed = $true
        }

        if ($worktreeStatus -eq "planned") {
            Write-Host "INVALID: $worktreeRegistry editable agent $agentId still planned (run setup_run_worktrees.sh)"
            $failed = $true
        }
    }
}

if ($failed) {
    Write-Host "Run ${RunId}: FAILED checks."
    exit 3
}

Write-Host "Run ${RunId}: logs, memory deltas, delegation, thread registry, and worktree checks passed."
