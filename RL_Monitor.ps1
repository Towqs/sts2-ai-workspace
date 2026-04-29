[CmdletBinding()]
param(
    [string]$LogFile,
    [string]$CombatDir,
    [string]$MacroDir,
    [int]$Tail = 80
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Watcher = Join-Path $Root "tools\watch_rl_logs.ps1"

if (-not (Test-Path -LiteralPath $Watcher)) {
    throw "Log watcher not found: $Watcher"
}

& $Watcher -LogFile $LogFile -CombatDir $CombatDir -MacroDir $MacroDir -Tail $Tail
