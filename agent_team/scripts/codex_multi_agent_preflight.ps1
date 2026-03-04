#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CommandName {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        $cmd = Get-Command -Name $candidate -ErrorAction SilentlyContinue
        if ($null -ne $cmd) {
            return $candidate
        }
    }

    return $null
}

function Test-IsWindows {
    return $env:OS -eq "Windows_NT"
}

Write-Host "== Codex Multi-Agent Preflight =="

$codexCandidates = @("codex", "codex.cmd")
if (Test-IsWindows) {
    $codexCandidates = @("codex.cmd", "codex")
}

$codexCommand = Resolve-CommandName -Candidates $codexCandidates
if ([string]::IsNullOrWhiteSpace($codexCommand)) {
    Write-Host "ERROR: codex/codex.cmd command not found in PATH."
    Write-Host "Install/enable Codex CLI first."
    exit 1
}

if ($null -eq (Get-Command -Name "git" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: git command not found in PATH."
    exit 1
}

$codexBinary = (Get-Command -Name $codexCommand -ErrorAction Stop).Source
Write-Host "Codex command: $codexCommand"
Write-Host "Codex binary: $codexBinary"
Write-Host "Codex version:"
try {
    & $codexCommand --version
} catch {
    Write-Host "WARN: failed to query codex version."
    Write-Host "      $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Checking feature flags..."
$featuresOut = ""
try {
    $featuresOut = (& $codexCommand features list 2>$null | Out-String).Trim()
} catch {
    $featuresOut = ""
}

if ([string]::IsNullOrWhiteSpace($featuresOut)) {
    Write-Host "WARN: cannot read '$codexCommand features list'."
    Write-Host "Try manually:"
    Write-Host "  $codexCommand features list"
    Write-Host "  $codexCommand features enable multi_agent"
    exit 2
}

Write-Host $featuresOut
Write-Host ""

if ($featuresOut -match "(?m)^multi_agent\s+\S+\s+true\s*$") {
    Write-Host "OK: multi_agent is enabled."
} else {
    Write-Host "WARN: multi_agent is NOT enabled."
    Write-Host "Enable it with:"
    Write-Host "  $codexCommand features enable multi_agent"
    Write-Host "Then restart codex."
}

Write-Host ""
Write-Host "Checking git worktree support..."
$worktreeOk = $true
try {
    & git worktree list *> $null
    if ($LASTEXITCODE -ne 0) {
        $worktreeOk = $false
    }
} catch {
    $worktreeOk = $false
}

if ($worktreeOk) {
    Write-Host "OK: git worktree is available."
} else {
    Write-Host "WARN: git worktree command failed. Worktree isolation may not work."
}

Write-Host ""
Write-Host "Recommended next steps:"
Write-Host "1) Linux/macOS: bash agent_team/scripts/bootstrap_agents.sh"
Write-Host "2) Linux/macOS: bash agent_team/scripts/init_run.sh <run_id>"
Write-Host "3) Windows: powershell -ExecutionPolicy Bypass -File agent_team/scripts/init_run.ps1 -RunId <run_id>"
Write-Host "4) Use template: agent_team/templates/codex_spawn_prompt_template.md"
Write-Host "5) In TUI use /agent to inspect/switch spawned sub-agent threads."
Write-Host "6) During long runs, monitor heartbeats:"
Write-Host "   bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45"
