#!/usr/bin/env pwsh
param(
    [string]$RunId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/init_run.ps1 -RunId <run_id>"
    exit 1
}

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

function New-FileFromTemplate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemplatePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [hashtable]$Tokens
    )

    $content = Read-Utf8Text -Path $TemplatePath
    foreach ($token in $Tokens.GetEnumerator()) {
        $content = $content.Replace([string]$token.Key, [string]$token.Value)
    }

    Write-Utf8NoBom -Path $TargetPath -Content $content
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

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$agentsYaml = Join-Path -Path $teamDir -ChildPath "agents.yaml"
$runDir = Join-Path -Path $teamDir -ChildPath ("runs/{0}" -f $RunId)
$logDir = Join-Path -Path $runDir -ChildPath "logs"
$handoffDir = Join-Path -Path $runDir -ChildPath "handoff"
$memoryDir = Join-Path -Path $runDir -ChildPath "memory"
$threadDir = Join-Path -Path $runDir -ChildPath "threads"
$worktreeDir = Join-Path -Path $runDir -ChildPath "worktrees"
$artifactDir = Join-Path -Path $runDir -ChildPath "artifacts"

if (-not (Test-Path -LiteralPath $agentsYaml)) {
    Write-Host "ERROR: missing $agentsYaml"
    exit 2
}

$null = New-Item -ItemType Directory -Force -Path $logDir, $handoffDir, $memoryDir, $threadDir, $worktreeDir, $artifactDir

$agents = Get-AgentDefinitions -AgentsYamlPath $agentsYaml

$logTemplate = Join-Path -Path $teamDir -ChildPath "templates/agent_log_template.md"
$memoryTemplate = Join-Path -Path $teamDir -ChildPath "templates/memory_delta_template.md"
$handoffTemplate = Join-Path -Path $teamDir -ChildPath "templates/handoff_to_doc_template.md"
$threadTemplate = Join-Path -Path $teamDir -ChildPath "templates/thread_registry_template.md"
$worktreeTemplate = Join-Path -Path $teamDir -ChildPath "templates/worktree_registry_template.md"

foreach ($agent in $agents) {
    $agentId = $agent.id
    $canEditCode = $agent.can_edit_code
    $canRunTests = $agent.can_run_tests
    $docDelegate = $agent.doc_delegate
    $logWriter = $agentId
    if ($canEditCode -eq "false") {
        $logWriter = $docDelegate
    }

    $logFile = Join-Path -Path $logDir -ChildPath ("{0}.md" -f $agentId)
    $memoryFile = Join-Path -Path $memoryDir -ChildPath ("{0}.delta.md" -f $agentId)

    if (-not (Test-Path -LiteralPath $logFile)) {
        New-FileFromTemplate -TemplatePath $logTemplate -TargetPath $logFile -Tokens @{
            "<agent_id>"      = $agentId
            "<run_id>"        = $RunId
            "<log_writer>"    = $logWriter
            "<can_edit_code>" = $canEditCode
            "<can_run_tests>" = $canRunTests
        }
    }

    if (-not (Test-Path -LiteralPath $memoryFile)) {
        New-FileFromTemplate -TemplatePath $memoryTemplate -TargetPath $memoryFile -Tokens @{
            "<agent_id>" = $agentId
            "<run_id>"   = $RunId
        }

        if ($canEditCode -eq "false") {
            $memoryContent = Read-Utf8Text -Path $memoryFile
            $memoryContent = [System.Text.RegularExpressions.Regex]::Replace(
                $memoryContent,
                "^- Recorder: .*$",
                ("- Recorder: {0}" -f $docDelegate),
                [System.Text.RegularExpressions.RegexOptions]::Multiline
            )
            Write-Utf8NoBom -Path $memoryFile -Content $memoryContent
        }
    }

    if ($canEditCode -eq "false") {
        $handoffFile = Join-Path -Path $handoffDir -ChildPath ("{0}_to_{1}.md" -f $agentId, $docDelegate)
        if (-not (Test-Path -LiteralPath $handoffFile)) {
            New-FileFromTemplate -TemplatePath $handoffTemplate -TargetPath $handoffFile -Tokens @{
                "<run_id>"   = $RunId
                "<agent_id>" = $agentId
            }

            $handoffContent = Read-Utf8Text -Path $handoffFile
            $handoffContent = $handoffContent.Replace(
                "Target Agent: doc-writer",
                ("Target Agent: {0}" -f $docDelegate)
            )
            Write-Utf8NoBom -Path $handoffFile -Content $handoffContent
        }
    }
}

$threadRegistry = Join-Path -Path $threadDir -ChildPath "registry.md"
if (-not (Test-Path -LiteralPath $threadRegistry)) {
    New-FileFromTemplate -TemplatePath $threadTemplate -TargetPath $threadRegistry -Tokens @{
        "<run_id>" = $RunId
    }

    $threadRows = New-Object System.Text.StringBuilder
    foreach ($agent in $agents) {
        [void]$threadRows.AppendLine(
            ("| {0} |  | planned | {0} |  |  | 10 | no | 0 |  |" -f $agent.id)
        )
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($threadRegistry, $threadRows.ToString(), $encoding)
}

$worktreeRegistry = Join-Path -Path $worktreeDir -ChildPath "registry.md"
if (-not (Test-Path -LiteralPath $worktreeRegistry)) {
    New-FileFromTemplate -TemplatePath $worktreeTemplate -TargetPath $worktreeRegistry -Tokens @{
        "<run_id>" = $RunId
    }

    $worktreeRows = New-Object System.Text.StringBuilder
    foreach ($agent in $agents) {
        [void]$worktreeRows.AppendLine(
            ("| {0} | {1} |  |  | planned |  |" -f $agent.id, $agent.can_edit_code)
        )
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($worktreeRegistry, $worktreeRows.ToString(), $encoding)
}

Write-Host "Initialized agent-team run at: $runDir"
Write-Host "Logs: $logDir"
Write-Host "Handoff notes: $handoffDir"
Write-Host "Memory deltas: $memoryDir"
Write-Host "Thread registry: $threadRegistry"
Write-Host "Worktree registry: $worktreeRegistry"
Write-Host "Artifacts: $artifactDir"
