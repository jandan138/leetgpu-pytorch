#!/usr/bin/env pwsh
param(
    [string]$RunId,
    [string]$AgentId,
    [string]$NewThreadId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId) -or [string]::IsNullOrWhiteSpace($AgentId) -or [string]::IsNullOrWhiteSpace($NewThreadId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/restart_stuck_subagent.ps1 -RunId <run_id> -AgentId <agent_id> -NewThreadId <new_thread_id>"
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

function Try-Parse-Int {
    param(
        [AllowEmptyString()]
        [string]$Value
    )

    $number = 0
    if ([int]::TryParse($Value, [ref]$number)) {
        return $number
    }

    return $null
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$threadRegistry = Join-Path -Path $teamDir -ChildPath ("runs/{0}/threads/registry.md" -f $RunId)

if (-not (Test-Path -LiteralPath $threadRegistry)) {
    Write-Host "ERROR: thread registry not found: $threadRegistry"
    exit 2
}

$lines = Get-Content -Path $threadRegistry -Encoding utf8
$separatorIndex = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "^\| ---") {
        $separatorIndex = $i
        break
    }
}

if ($separatorIndex -lt 0) {
    Write-Host "ERROR: invalid thread registry format, missing markdown separator row"
    exit 2
}

$prefixLines = $lines[0..$separatorIndex]
$newRowLines = New-Object System.Collections.Generic.List[string]
$foundRow = $false
$nowStamp = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")

foreach ($line in $lines) {
    $cells = Parse-MarkdownRowCells -Line $line
    if ($null -eq $cells -or $cells.Count -lt 10) {
        continue
    }

    $agent = $cells[0]
    if ([string]::IsNullOrWhiteSpace($agent) -or $agent -eq "agent_id" -or $agent -eq "---") {
        continue
    }

    $thread = $cells[1]
    $status = $cells[2]
    $owner = $cells[3]
    $startedAt = $cells[4]
    $lastHeartbeatAt = $cells[5]
    $intervalRaw = $cells[6]
    $stuckCandidate = $cells[7]
    $escalationRaw = $cells[8]
    $notes = $cells[9]

    if ($agent -eq $AgentId) {
        $foundRow = $true
        if ($status -ne "stuck-confirmed" -and $stuckCandidate -ne "confirmed") {
            Write-Host "ERROR: $AgentId is not stuck-confirmed. Current status=$status, stuck_candidate=$stuckCandidate"
            exit 3
        }

        $oldThread = $thread
        $escalation = Try-Parse-Int -Value $escalationRaw
        if ($null -eq $escalation) {
            $escalation = 0
        }
        $escalation += 1

        $thread = $NewThreadId
        $status = "active"
        $startedAt = $nowStamp
        $lastHeartbeatAt = $nowStamp
        $interval = Try-Parse-Int -Value $intervalRaw
        if ($null -eq $interval) {
            $interval = 10
        }
        $intervalRaw = "$interval"
        $stuckCandidate = "no"
        $escalationRaw = "$escalation"

        $restartNote = "[$nowStamp] restart: old_thread=$oldThread, new_thread=$NewThreadId, escalation=$escalation"
        if ([string]::IsNullOrWhiteSpace($notes)) {
            $notes = $restartNote
        } else {
            $notes = "$notes; $restartNote"
        }
    }

    $safeNotes = $notes.Replace("|", "\|")
    $newRowLines.Add("| $agent | $thread | $status | $owner | $startedAt | $lastHeartbeatAt | $intervalRaw | $stuckCandidate | $escalationRaw | $safeNotes |")
}

if (-not $foundRow) {
    Write-Host "ERROR: agent $AgentId not found in $threadRegistry"
    exit 2
}

$builder = New-Object System.Text.StringBuilder
foreach ($line in $prefixLines) {
    [void]$builder.AppendLine($line)
}
foreach ($line in $newRowLines) {
    [void]$builder.AppendLine($line)
}

Write-Utf8NoBom -Path $threadRegistry -Content $builder.ToString()
Write-Host "Restarted $AgentId with new thread $NewThreadId (status=active, stuck_candidate=no)."
