#!/usr/bin/env pwsh
param(
    [string]$RunId,
    [switch]$DeleteBranches
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/teardown_run_worktrees.ps1 -RunId <run_id> [-DeleteBranches]"
    exit 1
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

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$runDir = Join-Path -Path $teamDir -ChildPath ("runs/{0}" -f $RunId)
$worktreeRegistry = Join-Path -Path $runDir -ChildPath "worktrees/registry.md"

if (-not (Test-Path -LiteralPath $worktreeRegistry)) {
    Write-Host "ERROR: missing $worktreeRegistry"
    exit 2
}

$repoCheck = Invoke-Git -Args @("rev-parse", "--show-toplevel")
if ($repoCheck.ExitCode -ne 0) {
    Write-Host "ERROR: current directory is not inside a git repository"
    exit 3
}

$rows = Get-Content -Path $worktreeRegistry -Encoding utf8
foreach ($line in $rows) {
    $cells = Parse-MarkdownRowCells -Line $line
    if ($null -eq $cells -or $cells.Count -lt 6) {
        continue
    }

    $agentId = $cells[0]
    $canEdit = $cells[1]
    $branch = $cells[2]
    $worktreePath = $cells[3]

    if ([string]::IsNullOrWhiteSpace($agentId) -or $agentId -eq "agent_id" -or $agentId -eq "---") {
        continue
    }

    if ($canEdit -ne "true") {
        continue
    }

    if ([string]::IsNullOrWhiteSpace($worktreePath)) {
        continue
    }

    if (Test-Path -LiteralPath $worktreePath) {
        $rmRes = Invoke-Git -Args @("worktree", "remove", "--force", $worktreePath)
        if ($rmRes.ExitCode -eq 0) {
            Write-Host "REMOVED worktree: $worktreePath"
        } else {
            Write-Host "WARN: failed removing $worktreePath"
            if (-not [string]::IsNullOrWhiteSpace($rmRes.Output)) {
                Write-Host $rmRes.Output
            }
        }
    }

    if ($DeleteBranches -and -not [string]::IsNullOrWhiteSpace($branch)) {
        $branchExists = (Invoke-Git -Args @("show-ref", "--verify", "--quiet", "refs/heads/$branch")).ExitCode -eq 0
        if ($branchExists) {
            $branchRes = Invoke-Git -Args @("branch", "-D", $branch)
            if ($branchRes.ExitCode -eq 0) {
                Write-Host "DELETED branch: $branch"
            } else {
                Write-Host "WARN: failed deleting branch $branch"
                if (-not [string]::IsNullOrWhiteSpace($branchRes.Output)) {
                    Write-Host $branchRes.Output
                }
            }
        }
    }
}

Write-Host "Teardown complete for run: $RunId"
