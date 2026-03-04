#!/usr/bin/env pwsh
param(
    [string]$RunId,
    [string]$BaseRef = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/setup_run_worktrees.ps1 -RunId <run_id> [-BaseRef <base_ref>]"
    exit 1
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

function Get-AgentEditDefinitions {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AgentsYamlPath
    )

    $agents = @()
    $current = $null

    foreach ($rawLine in (Get-Content -Path $AgentsYamlPath -Encoding utf8)) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        if ($line -match "^- id:\s*(\S+)\s*$") {
            if ($null -ne $current) {
                $agents += [PSCustomObject]$current
            }

            $current = [ordered]@{
                id            = $Matches[1]
                can_edit_code = "false"
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
    }

    if ($null -ne $current) {
        $agents += [PSCustomObject]$current
    }

    if ($agents.Count -eq 0) {
        throw "No agent definitions parsed from $AgentsYamlPath"
    }

    return $agents
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $tempOut = [System.IO.Path]::GetTempFileName()
    $tempErr = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process -FilePath "git" -ArgumentList $Args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
        $exitCode = $proc.ExitCode
        $text = ""
        if (Test-Path -LiteralPath $tempOut) {
            $text = [System.IO.File]::ReadAllText($tempOut, [System.Text.Encoding]::UTF8)
        }
        if (Test-Path -LiteralPath $tempErr) {
            $stderr = [System.IO.File]::ReadAllText($tempErr, [System.Text.Encoding]::UTF8)
            if (-not [string]::IsNullOrWhiteSpace($stderr)) {
                if ([string]::IsNullOrWhiteSpace($text)) {
                    $text = $stderr
                } else {
                    $text = "$text`n$stderr"
                }
            }
        }
        $text = $text.TrimEnd()

        return [PSCustomObject]@{
            ExitCode = $exitCode
            Output   = $text
        }
    }
    finally {
        Remove-Item -LiteralPath $tempOut, $tempErr -Force -ErrorAction SilentlyContinue
    }
}

function Sanitize-Note {
    param(
        [string]$Text
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $singleLine = $Text -replace "`r?`n", ";"
    $singleLine = $singleLine.Replace("|", "\|")
    if ($singleLine.Length -gt 240) {
        return $singleLine.Substring(0, 240)
    }

    return $singleLine
}

function Get-SafeRunId {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $safe = [System.Text.RegularExpressions.Regex]::Replace($Value, "[^A-Za-z0-9._-]+", "-")
    $safe = $safe.Trim("-")
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return "run"
    }

    return $safe
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$agentsYaml = Join-Path -Path $teamDir -ChildPath "agents.yaml"
$runDir = Join-Path -Path $teamDir -ChildPath ("runs/{0}" -f $RunId)
$worktreeDir = Join-Path -Path $runDir -ChildPath "worktrees"
$worktreeRegistry = Join-Path -Path $worktreeDir -ChildPath "registry.md"

if (-not (Test-Path -LiteralPath $agentsYaml)) {
    Write-Host "ERROR: missing $agentsYaml"
    exit 2
}

if (-not (Test-Path -LiteralPath $runDir)) {
    Write-Host "ERROR: run directory not found: $runDir"
    Write-Host "Run init first: powershell -ExecutionPolicy Bypass -File agent_team/scripts/init_run.ps1 -RunId $RunId"
    exit 2
}

if (-not (Test-Path -LiteralPath $worktreeRegistry)) {
    Write-Host "ERROR: worktree registry not found: $worktreeRegistry"
    exit 2
}

$repoCheck = Invoke-Git -Args @("rev-parse", "--show-toplevel")
if ($repoCheck.ExitCode -ne 0) {
    Write-Host "ERROR: current directory is not inside a git repository"
    exit 3
}

$repoRoot = $repoCheck.Output.Trim()
$repoName = Split-Path -Leaf $repoRoot
$safeRunId = Get-SafeRunId -Value $RunId

$defaultRoot = ""
if ($env:OS -eq "Windows_NT") {
    $defaultRoot = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("agent_team_worktrees\{0}\{1}" -f $repoName, $safeRunId)
} else {
    $defaultRoot = "/tmp/agent_team_worktrees/$repoName/$safeRunId"
}

$rootDir = if ([string]::IsNullOrWhiteSpace($env:AGENT_TEAM_WORKTREE_ROOT)) { $defaultRoot } else { $env:AGENT_TEAM_WORKTREE_ROOT }
$null = New-Item -ItemType Directory -Force -Path $rootDir

$resolvedBaseRef = $BaseRef
if ([string]::IsNullOrWhiteSpace($resolvedBaseRef)) {
    $headRes = Invoke-Git -Args @("rev-parse", "--abbrev-ref", "HEAD")
    if ($headRes.ExitCode -ne 0) {
        Write-Host "ERROR: failed to determine current git branch"
        Write-Host $headRes.Output
        exit 3
    }

    $resolvedBaseRef = $headRes.Output.Trim()
    if ($resolvedBaseRef -eq "HEAD") {
        $resolvedBaseRef = "HEAD"
    }
}

$agents = Get-AgentEditDefinitions -AgentsYamlPath $agentsYaml
$rows = New-Object System.Collections.Generic.List[string]

Push-Location $repoRoot
try {
    foreach ($agent in $agents) {
        $agentId = $agent.id
        $canEdit = $agent.can_edit_code
        $branch = ""
        $worktreePath = ""
        $status = "planned"
        $notes = ""

        if ($canEdit -eq "true") {
            $branch = "run/$safeRunId/$agentId"
            $worktreePath = Join-Path -Path $rootDir -ChildPath $agentId

            $hasGitMarker = (Test-Path -LiteralPath (Join-Path -Path $worktreePath -ChildPath ".git"))
            if ((Test-Path -LiteralPath $worktreePath) -and (-not $hasGitMarker)) {
                $status = "blocked"
                $notes = "path exists but is not a git worktree"
            } else {
                if ($hasGitMarker) {
                    $probe = Invoke-Git -Args @("-C", $worktreePath, "rev-parse", "--is-inside-work-tree")
                    if ($probe.ExitCode -eq 0) {
                        $status = "ready"
                        $notes = "existing worktree reused"
                    } else {
                        $status = "blocked"
                        $notes = "existing path not usable as worktree"
                    }
                } else {
                    $branchExists = (Invoke-Git -Args @("show-ref", "--verify", "--quiet", "refs/heads/$branch")).ExitCode -eq 0
                    if ($branchExists) {
                        $res = Invoke-Git -Args @("worktree", "add", $worktreePath, $branch)
                        if ($res.ExitCode -eq 0) {
                            $status = "ready"
                            $notes = "attached existing branch"
                        } else {
                            $status = "blocked"
                            $notes = Sanitize-Note -Text $res.Output
                        }
                    } else {
                        $res = Invoke-Git -Args @("worktree", "add", "-b", $branch, $worktreePath, $resolvedBaseRef)
                        if ($res.ExitCode -eq 0) {
                            $status = "ready"
                            $notes = "created branch from $resolvedBaseRef"
                        } else {
                            $status = "blocked"
                            $notes = Sanitize-Note -Text $res.Output
                        }
                    }
                }
            }
        }

        $rows.Add("| $agentId | $canEdit | $branch | $worktreePath | $status | $notes |")
    }
}
finally {
    Pop-Location
}

$registryBuilder = New-Object System.Text.StringBuilder
[void]$registryBuilder.AppendLine("# Worktree Registry")
[void]$registryBuilder.AppendLine("")
[void]$registryBuilder.AppendLine("- Run ID: $RunId")
[void]$registryBuilder.AppendLine("- Maintainer: orchestrator")
[void]$registryBuilder.AppendLine("- Purpose: isolate editable agents into dedicated git worktrees.")
[void]$registryBuilder.AppendLine("")
[void]$registryBuilder.AppendLine("## Status Legend")
[void]$registryBuilder.AppendLine("- ``planned``: worktree row prepared, not created")
[void]$registryBuilder.AppendLine("- ``ready``: worktree exists and agent can start coding")
[void]$registryBuilder.AppendLine("- ``blocked``: creation failed or dependency issue")
[void]$registryBuilder.AppendLine("- ``done``: agent work completed")
[void]$registryBuilder.AppendLine("- ``cleaned``: worktree removed")
[void]$registryBuilder.AppendLine("")
[void]$registryBuilder.AppendLine("## Worktrees")
[void]$registryBuilder.AppendLine("| agent_id | can_edit_code | branch | worktree_path | status | notes |")
[void]$registryBuilder.AppendLine("| --- | --- | --- | --- | --- | --- |")
foreach ($line in $rows) {
    [void]$registryBuilder.AppendLine($line)
}

Write-Utf8NoBom -Path $worktreeRegistry -Content $registryBuilder.ToString()

Write-Host "Worktree setup complete for run: $RunId"
Write-Host "Registry: $worktreeRegistry"
Write-Host "Root: $rootDir"
