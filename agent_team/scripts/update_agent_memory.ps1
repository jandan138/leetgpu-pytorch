#!/usr/bin/env pwsh
param(
    [string]$RunId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File agent_team/scripts/update_agent_memory.ps1 -RunId <run_id>"
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

function Test-ContainsLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Literal
    )

    return [bool](Select-String -Path $Path -SimpleMatch -Pattern $Literal -Quiet)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$teamDir = Split-Path -Parent $scriptDir
$deltaDir = Join-Path -Path $teamDir -ChildPath ("runs/{0}/memory" -f $RunId)
$agentRoot = Join-Path -Path $teamDir -ChildPath "agents"
$stamp = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")

if (-not (Test-Path -LiteralPath $deltaDir)) {
    Write-Host "ERROR: missing delta dir: $deltaDir"
    exit 2
}

$deltaFiles = Get-ChildItem -Path $deltaDir -Filter "*.delta.md" -File -ErrorAction SilentlyContinue
if ($deltaFiles.Count -eq 0) {
    Write-Host "ERROR: no delta files found in $deltaDir"
    exit 3
}

foreach ($deltaFile in $deltaFiles) {
    $agentId = $deltaFile.BaseName -replace "\.delta$", ""
    $memoryFile = Join-Path -Path $agentRoot -ChildPath ("{0}/memory.md" -f $agentId)

    if (-not (Test-Path -LiteralPath $memoryFile)) {
        Write-Host "WARN: missing memory file for $agentId, skip"
        continue
    }

    $marker = "- Run: $RunId"
    if (Test-ContainsLiteral -Path $memoryFile -Literal $marker) {
        Write-Host "SKIP: $agentId already merged for run $RunId"
        continue
    }

    $deltaContent = Read-Utf8Text -Path $deltaFile.FullName
    $indentedDelta = ($deltaContent -split "`r?`n" | ForEach-Object { "    $_" }) -join [Environment]::NewLine

    $append = @(
        "",
        "- Run: $RunId",
        "  - Summary: merged delta at $stamp",
        "  - Source: runs/$RunId/memory/$agentId.delta.md",
        "  - Notes:",
        $indentedDelta
    ) -join [Environment]::NewLine

    $memoryContent = Read-Utf8Text -Path $memoryFile
    $newContent = "$memoryContent$append"
    Write-Utf8NoBom -Path $memoryFile -Content $newContent

    Write-Host "UPDATED: $memoryFile"
}

Write-Host "Memory merge complete for run $RunId."
