[CmdletBinding()]
param(
    [string]$LogFile,
    [string]$CombatDir,
    [string]$MacroDir,
    [int]$Tail = 80
)

$ErrorActionPreference = "Continue"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($LogFile)) {
    $LogFile = Join-Path $Root "RL_Datasets\rl_monitor.log"
}
if ([string]::IsNullOrWhiteSpace($CombatDir)) {
    $CombatDir = Join-Path $Root "RL_Datasets\Combat"
}
if ([string]::IsNullOrWhiteSpace($MacroDir)) {
    $MacroDir = Join-Path $Root "RL_Datasets\Macro"
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogFile -Parent) | Out-Null
if (-not (Test-Path -LiteralPath $LogFile)) {
    New-Item -ItemType File -Force -Path $LogFile | Out-Null
}

function Get-DirStats {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) {
        return @{ Count = 0; SizeMB = 0 }
    }
    $files = Get-ChildItem -LiteralPath $Dir -Filter "*.jsonl" -File -ErrorAction SilentlyContinue
    $size = ($files | Measure-Object -Property Length -Sum).Sum
    if (-not $size) {
        $size = 0
    }
    return @{
        Count = @($files).Count
        SizeMB = [math]::Round($size / 1MB, 2)
    }
}

function Write-LogLine {
    param([string]$Line)
    if ([string]::IsNullOrWhiteSpace($Line)) {
        return
    }

    if ($Line -match "\[ERROR\]|error|exception") {
        Write-Host $Line -ForegroundColor Red
    } elseif ($Line -match "battle_end.*win") {
        Write-Host $Line -ForegroundColor Green
    } elseif ($Line -match "battle_end.*lose") {
        Write-Host $Line -ForegroundColor Red
    } elseif ($Line -match "battle_start|turn_start") {
        Write-Host $Line -ForegroundColor Cyan
    } elseif ($Line -match "turn_end|end_turn") {
        Write-Host $Line -ForegroundColor DarkYellow
    } elseif ($Line -match "COMBAT|play_card|use_potion") {
        Write-Host $Line -ForegroundColor White
    } elseif ($Line -match "MACRO|map_node|choose_card|buy_item") {
        Write-Host $Line -ForegroundColor Magenta
    } elseif ($Line -match "\[INIT\]|\[NEW FILE\]") {
        Write-Host $Line -ForegroundColor Green
    } else {
        Write-Host $Line -ForegroundColor DarkGray
    }
}

Clear-Host
$combat = Get-DirStats $CombatDir
$macro = Get-DirStats $MacroDir

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  STS2 AI runtime log monitor" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Log file: $LogFile" -ForegroundColor Gray
Write-Host "Combat:   $($combat.Count) files, $($combat.SizeMB) MB" -ForegroundColor Gray
Write-Host "Macro:    $($macro.Count) files, $($macro.SizeMB) MB" -ForegroundColor Gray
Write-Host ""
Write-Host "Waiting for new lines. Close this window to stop watching." -ForegroundColor Gray
Write-Host ""

try {
    Get-Content -LiteralPath $LogFile -Tail $Tail -Wait -Encoding UTF8 | ForEach-Object {
        Write-LogLine $_
    }
} catch {
    Write-Host ("Log monitor stopped: " + $_.Exception.Message) -ForegroundColor Red
}
