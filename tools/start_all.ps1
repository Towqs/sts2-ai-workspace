[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoMonitor,
    [switch]$NoAgents,
    [switch]$NoAi,
    [switch]$NoLlm,
    [switch]$ForceAi,
    [switch]$ForceLlm,
    [switch]$DryRun,
    [int]$PanelPort = 8765,
    [int]$WaitSeconds = 30
)

$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PanelUrl = "http://127.0.0.1:$PanelPort"
$ControlPanel = Join-Path $Root "AI_Training\control_panel.py"
$LogDir = Join-Path $Root "RuntimeLogs"
$ControlLog = Join-Path $LogDir "control_panel.log"
$WatchScript = Join-Path $Root "tools\watch_rl_logs.ps1"
$RlLog = Join-Path $Root "RL_Datasets\rl_monitor.log"
$CombatDir = Join-Path $Root "RL_Datasets\Combat"
$MacroDir = Join-Path $Root "RL_Datasets\Macro"

if ($env:STS2_AI_PYTHON) {
    $PythonExe = $env:STS2_AI_PYTHON
} else {
    $PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = "python"
}

function Quote-PSLiteral {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Start-ConsoleWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $titleLiteral = Quote-PSLiteral $Title
    $bootstrap = @"
`$Host.UI.RawUI.WindowTitle = $titleLiteral
try {
$Command
} catch {
    Write-Host ("ERROR: " + `$_.Exception.Message) -ForegroundColor Red
}
"@

    if ($DryRun) {
        Write-Host "[dry-run] Would start window: $Title"
        Write-Host $Command
        return
    }

    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($bootstrap))
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-NoExit",
        "-EncodedCommand",
        $encoded
    ) | Out-Null
}

function Test-PanelReady {
    try {
        $null = Invoke-WebRequest -Uri "$PanelUrl/api/status" -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Wait-PanelReady {
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PanelReady) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-PanelStatus {
    try {
        return Invoke-RestMethod -Uri "$PanelUrl/api/status" -TimeoutSec 10
    } catch {
        Write-Warning ("Cannot read panel status: " + $_.Exception.Message)
        return $null
    }
}

function Invoke-PanelPost {
    param([string]$Path)
    try {
        return Invoke-RestMethod -Uri "$PanelUrl$Path" -Method Post -Body "{}" -ContentType "application/json" -TimeoutSec 10
    } catch {
        Write-Warning ("POST $Path failed: " + $_.Exception.Message)
        return $null
    }
}

function Test-LlmConfigured {
    param($Status)
    if (-not $Status -or -not $Status.llm -or -not $Status.llm.config) {
        return $false
    }
    $config = $Status.llm.config
    return [bool]$config.has_api_key -and -not [string]::IsNullOrWhiteSpace([string]$config.model)
}

function Test-CombatModelReady {
    param($Status)
    if (-not $Status -or -not $Status.models -or -not $Status.models.combat) {
        return $false
    }
    return [bool]$Status.models.combat.ready
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $RlLog -Parent) | Out-Null
if (-not (Test-Path -LiteralPath $RlLog)) {
    New-Item -ItemType File -Force -Path $RlLog | Out-Null
}

if (-not (Test-Path -LiteralPath $ControlPanel)) {
    throw "Control panel not found: $ControlPanel"
}
if (-not (Test-Path -LiteralPath $WatchScript)) {
    throw "Log watcher not found: $WatchScript"
}

Write-Host "STS2 AI launcher"
Write-Host "Workspace: $Root"
Write-Host "Panel:    $PanelUrl"
Write-Host ""

$rootQ = Quote-PSLiteral $Root
$pythonQ = Quote-PSLiteral $PythonExe
$panelQ = Quote-PSLiteral $ControlPanel
$controlLogQ = Quote-PSLiteral $ControlLog

if (Test-PanelReady) {
    Write-Host "Control panel is already running."
} else {
    $controlCommand = @"
Set-Location -LiteralPath $rootQ
`$env:PYTHONIOENCODING = 'utf-8'
& $pythonQ $panelQ 2>&1 | Tee-Object -FilePath $controlLogQ -Append
"@
    Write-Host "Starting control panel..."
    Start-ConsoleWindow -Title "STS2 AI Control Panel" -Command $controlCommand
}

if (-not $DryRun) {
    if (Wait-PanelReady) {
        Write-Host "Control panel ready."
    } else {
        throw "Control panel did not become ready within $WaitSeconds seconds."
    }
}

if (-not $NoMonitor) {
    $watchQ = Quote-PSLiteral $WatchScript
    $rlLogQ = Quote-PSLiteral $RlLog
    $combatQ = Quote-PSLiteral $CombatDir
    $macroQ = Quote-PSLiteral $MacroDir
    $monitorCommand = @"
Set-Location -LiteralPath $rootQ
& $watchQ -LogFile $rlLogQ -CombatDir $combatQ -MacroDir $macroQ
"@
    Write-Host "Starting RL log monitor..."
    Start-ConsoleWindow -Title "STS2 RL Log Monitor" -Command $monitorCommand
}

if (-not $NoBrowser -and -not $DryRun) {
    Write-Host "Opening browser..."
    Start-Process $PanelUrl | Out-Null
}

if (-not $NoAgents -and -not $DryRun) {
    $status = Get-PanelStatus

    if (-not $NoAi) {
        if ((Test-CombatModelReady $status) -or $ForceAi) {
            $aiResult = Invoke-PanelPost "/api/ai/start"
            if ($aiResult) {
                Write-Host ("AI:  " + $aiResult.message)
            }
        } else {
            Write-Warning "AI not started: combat model is missing. Run training or use -ForceAi."
        }
    }

    if (-not $NoLlm) {
        if ((Test-LlmConfigured $status) -or $ForceLlm) {
            $llmResult = Invoke-PanelPost "/api/llm/start"
            if ($llmResult) {
                Write-Host ("LLM: " + $llmResult.message)
            }
        } else {
            Write-Warning "LLM not started: model name or API key is missing. Configure it in the panel, or use -ForceLlm."
        }
    }
}

Write-Host ""
Write-Host "Done. Keep the control panel and log monitor windows open while playing."
