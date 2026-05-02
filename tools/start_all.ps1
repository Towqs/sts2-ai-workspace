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

function Quote-PSLiteral {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Test-PythonExe {
    param([string]$Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $false
    }
    if ((Split-Path -Leaf $Candidate) -ne $Candidate -and -not (Test-Path -LiteralPath $Candidate)) {
        return $false
    }
    try {
        $output = & $Candidate --version 2>&1
        return ($LASTEXITCODE -eq 0 -and ($output -join "`n") -match "Python")
    } catch {
        return $false
    }
}

function Get-VenvBasePython {
    $cfg = Join-Path $Root ".venv\pyvenv.cfg"
    if (-not (Test-Path -LiteralPath $cfg)) {
        return ""
    }
    try {
        foreach ($line in Get-Content -LiteralPath $cfg -Encoding UTF8) {
            if ($line -match "^\s*executable\s*=\s*(.+?)\s*$") {
                return $Matches[1]
            }
        }
    } catch {
    }
    return ""
}

function Resolve-PythonExe {
    $candidates = New-Object System.Collections.Generic.List[string]

    if ($env:STS2_AI_PYTHON) {
        $candidates.Add($env:STS2_AI_PYTHON)
    }
    $candidates.Add((Join-Path $Root ".venv\Scripts\python.exe"))
    $venvBase = Get-VenvBasePython
    if ($venvBase) {
        $candidates.Add($venvBase)
    }
    if ($env:USERPROFILE) {
        $candidates.Add((Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"))
    }

    foreach ($name in @("python.exe", "python", "py.exe", "py")) {
        try {
            $cmd = Get-Command $name -ErrorAction Stop
            if ($cmd.Source) {
                $candidates.Add($cmd.Source)
            }
        } catch {
        }
    }

    $seen = @{}
    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $key = $candidate.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        if (Test-PythonExe $candidate) {
            return $candidate
        }
    }

    throw "No runnable Python found. Rebuild .venv or set STS2_AI_PYTHON to a working python.exe."
}

function Should-UseVenvSitePackages {
    param([string]$PythonExe)
    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    $venvBase = Get-VenvBasePython
    foreach ($candidate in @($venvPython, $venvBase)) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and
            $PythonExe.Equals($candidate, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
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
        $null = Invoke-WebRequest -Uri "$PanelUrl/api/ping" -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Set-PanelPort {
    param([int]$Port)
    $script:PanelPort = $Port
    $script:PanelUrl = "http://127.0.0.1:$Port"
}

function Test-PanelPortFree {
    param([int]$Port)
    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $listener = [System.Net.Sockets.TcpListener]::new($address, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Find-FreePanelPort {
    param(
        [int]$StartPort,
        [int]$Limit = 20
    )
    for ($port = $StartPort; $port -lt ($StartPort + $Limit); $port++) {
        if (Test-PanelPortFree $port) {
            return $port
        }
    }
    throw "No free local panel port found from $StartPort to $($StartPort + $Limit - 1)."
}

function Test-PanelUsesCurrentUi {
    try {
        $response = Invoke-WebRequest -Uri "$PanelUrl/" -UseBasicParsing -TimeoutSec 4
        $html = [string]$response.Content
        return ($html.Contains("renderTrainingComposition") -and $html.Contains("trainCompositionMain"))
    } catch {
        return $false
    }
}

function Get-PanelPortPid {
    try {
        $lines = netstat -ano | Select-String (":$PanelPort\s")
        foreach ($line in $lines) {
            $text = [string]$line
            if ($text -match "LISTENING\s+(\d+)\s*$") {
                return $Matches[1]
            }
        }
    } catch {
    }
    return ""
}

function Test-CommandLineProcess {
    param([string[]]$Needles)
    if (-not $Needles -or $Needles.Count -eq 0) {
        return $false
    }

    try {
        $needleList = @($Needles | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($needleList.Count -eq 0) {
            return $false
        }

        $currentPid = $PID
        $processes = Get-CimInstance Win32_Process -ErrorAction Stop
        foreach ($process in $processes) {
            if ($process.ProcessId -eq $currentPid) {
                continue
            }
            $commandLine = [string]$process.CommandLine
            if ([string]::IsNullOrWhiteSpace($commandLine)) {
                continue
            }
            $allMatched = $true
            foreach ($needle in $needleList) {
                if ($commandLine.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
                    $allMatched = $false
                    break
                }
            }
            if ($allMatched) {
                return $true
            }
        }
    } catch {
        Write-Warning ("Cannot inspect running processes: " + $_.Exception.Message)
    }
    return $false
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

$PythonExe = Resolve-PythonExe
$VenvSitePackages = Join-Path $Root ".venv\Lib\site-packages"
$UseVenvSitePackages = (Should-UseVenvSitePackages $PythonExe) -and (Test-Path -LiteralPath $VenvSitePackages)
Write-Host "Python:   $PythonExe"
if ($UseVenvSitePackages) {
    Write-Host "Packages: $VenvSitePackages"
}
Write-Host ""

$rootQ = Quote-PSLiteral $Root
$pythonQ = Quote-PSLiteral $PythonExe
$venvSiteQ = Quote-PSLiteral $VenvSitePackages
$panelQ = Quote-PSLiteral $ControlPanel
$controlLogQ = Quote-PSLiteral $ControlLog
$venvPathCommand = if ($UseVenvSitePackages) { "`$env:PYTHONPATH = $venvSiteQ" } else { "" }

if (Test-PanelReady) {
    if (Test-PanelUsesCurrentUi) {
        Write-Host "Control panel is already running."
    } else {
        $panelPid = Get-PanelPortPid
        $pidHint = if ($panelPid) { " PID $panelPid" } else { "" }
        $oldUrl = $PanelUrl
        $fallbackPort = Find-FreePanelPort -StartPort ($PanelPort + 1)
        Set-PanelPort $fallbackPort
        Write-Warning "A stale control panel$pidHint is already using $oldUrl. Starting a fresh panel on $PanelUrl instead."
    }
}

if (-not (Test-PanelReady)) {
    $controlCommand = @"
Set-Location -LiteralPath $rootQ
`$env:PYTHONIOENCODING = 'utf-8'
`$env:PYTHONUTF8 = '1'
`$env:STS2_AI_PYTHON = $pythonQ
`$env:STS2_AI_PANEL_PORT = '$PanelPort'
$venvPathCommand
& $pythonQ -X utf8 $panelQ 2>&1 | Tee-Object -FilePath $controlLogQ -Append
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
    if (Test-CommandLineProcess @($WatchScript, $RlLog)) {
        Write-Host "RL log monitor is already running."
    } else {
        Write-Host "Starting RL log monitor..."
        Start-ConsoleWindow -Title "STS2 RL Log Monitor" -Command $monitorCommand
    }
}

if (-not $NoBrowser -and -not $DryRun) {
    Write-Host "Opening browser..."
    Start-Process $PanelUrl | Out-Null
}

if (-not $NoAgents -and -not $DryRun) {
    $status = Get-PanelStatus

    if (-not $NoAi) {
        $agentReady = [bool]($status -and $status.python_runtime -and $status.python_runtime.agent_ready)
        if (-not $agentReady -and -not $ForceAi) {
            $missing = @($status.python_runtime.missing) -join ", "
            Write-Warning "AI not started: Python dependencies are missing ($missing). Rebuild .venv or use -ForceAi."
        } elseif ((Test-CombatModelReady $status) -or $ForceAi) {
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
