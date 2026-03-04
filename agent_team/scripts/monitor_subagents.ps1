#!/usr/bin/env pwsh
param(
    [string]$RunId,
    [int]$IntervalMin = 10,
    [int]$StuckMin = 45
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/monitor_subagents.ps1 -RunId <run_id> [-IntervalMin N] [-StuckMin N]"
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

function Try-Parse-Utc {
    param(
        [AllowEmptyString()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    $timestamp = [DateTimeOffset]::MinValue
    if ([DateTimeOffset]::TryParse($Value, [ref]$timestamp)) {
        return $timestamp.ToUniversalTime()
    }

    return $null
}

if ($IntervalMin -lt 0 -or $StuckMin -lt 0) {
    Write-Host "ERROR: -IntervalMin and -StuckMin must be non-negative integers"
    exit 2
}

if ($StuckMin -lt $IntervalMin) {
    Write-Host "ERROR: -StuckMin should be >= -IntervalMin"
    exit 2
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
$changed = $false
$nowUtc = [DateTimeOffset]::UtcNow
$nowStamp = $nowUtc.ToString("yyyy-MM-ddTHH:mm:ssZ")

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
    $escalationCount = $cells[8]
    $notes = $cells[9]

    $newStatus = $status
    $newStuck = $stuckCandidate
    $newNotes = $notes

    $intervalParsed = Try-Parse-Int -Value $intervalRaw
    $useInterval = if ($null -eq $intervalParsed) { $IntervalMin } else { $intervalParsed }

    $referenceTs = if ([string]::IsNullOrWhiteSpace($lastHeartbeatAt)) { $startedAt } else { $lastHeartbeatAt }

    if ($status -eq "active" -or $status -eq "waiting-subagent") {
        if (-not [string]::IsNullOrWhiteSpace($referenceTs)) {
            $refUtc = Try-Parse-Utc -Value $referenceTs
            if ($null -ne $refUtc) {
                $elapsedMin = [int][Math]::Floor(($nowUtc - $refUtc).TotalMinutes)

                if ($elapsedMin -ge $StuckMin) {
                    $newStatus = "stuck-suspected"
                    $newStuck = "yes-confirm-required"
                    $alertNote = "[$nowStamp] monitor: heartbeat stale ${elapsedMin}m >= ${StuckMin}m; manual confirm required"
                    if ([string]::IsNullOrWhiteSpace($newNotes)) {
                        $newNotes = $alertNote
                    } else {
                        $newNotes = "$newNotes; $alertNote"
                    }
                    $changed = $true
                    Write-Host "SUSPECTED_STUCK $agent`: stale ${elapsedMin}m, set status=stuck-suspected"
                } elseif ($elapsedMin -ge $useInterval) {
                    Write-Host "LATE_HEARTBEAT $agent`: stale ${elapsedMin}m (interval ${useInterval}m), continue waiting"
                } else {
                    Write-Host "HEALTHY $agent`: stale ${elapsedMin}m"
                }
            } else {
                $newStatus = "stuck-suspected"
                $newStuck = "yes-confirm-required"
                $alertNote = "[$nowStamp] monitor: cannot parse heartbeat time '$referenceTs', manual confirm required"
                if ([string]::IsNullOrWhiteSpace($newNotes)) {
                    $newNotes = $alertNote
                } else {
                    $newNotes = "$newNotes; $alertNote"
                }
                $changed = $true
                Write-Host "SUSPECTED_STUCK $agent`: unparseable heartbeat timestamp"
            }
        } else {
            $newStatus = "stuck-suspected"
            $newStuck = "yes-confirm-required"
            $alertNote = "[$nowStamp] monitor: missing started_at/last_heartbeat_at, manual confirm required"
            if ([string]::IsNullOrWhiteSpace($newNotes)) {
                $newNotes = $alertNote
            } else {
                $newNotes = "$newNotes; $alertNote"
            }
            $changed = $true
            Write-Host "SUSPECTED_STUCK $agent`: no heartbeat timestamp"
        }
    }

    $safeNotes = $newNotes.Replace("|", "\|")
    $newRowLines.Add("| $agent | $thread | $newStatus | $owner | $startedAt | $lastHeartbeatAt | $useInterval | $newStuck | $escalationCount | $safeNotes |")
}

$contentBuilder = New-Object System.Text.StringBuilder
foreach ($line in $prefixLines) {
    [void]$contentBuilder.AppendLine($line)
}
foreach ($line in $newRowLines) {
    [void]$contentBuilder.AppendLine($line)
}

Write-Utf8NoBom -Path $threadRegistry -Content $contentBuilder.ToString()

if ($changed) {
    Write-Host "Updated $threadRegistry with stuck-suspected candidates."
} else {
    Write-Host "No stuck-suspected updates. Continue waiting for sub-agents."
}
