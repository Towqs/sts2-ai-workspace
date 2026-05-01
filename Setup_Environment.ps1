[CmdletBinding()]
param(
    [string]$GameDir
)

$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

$Root = $PSScriptRoot
Write-Host "=== STS2 AI Setup ===" -ForegroundColor Cyan
Write-Host "Workspace: $Root"

# 1. Configure Python environment
Write-Host "`n[1/3] Configuring Python virtual environment..." -ForegroundColor Yellow
$PythonExe = "python.exe"
try {
    $null = & $PythonExe --version 2>&1
} catch {
    Write-Host "ERROR: python was not found. Install Python 3.10+ and enable Add to PATH." -ForegroundColor Red
    pause
    exit 1
}

$VenvPath = Join-Path $Root ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment (.venv)..."
    & $PythonExe -m venv "$VenvPath"
} else {
    Write-Host "Virtual environment already exists."
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "Installing/updating dependencies (requirements.txt)..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

# 2. Locate game directory
Write-Host "`n[2/3] Locating Slay the Spire 2 installation..." -ForegroundColor Yellow

function Get-GameDir {
    if ($GameDir) { return $GameDir }
    
    $steamPath = (Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -Name "SteamPath" -ErrorAction SilentlyContinue).SteamPath
    if ($steamPath) {
        $commonPath = "steamapps\common\Slay the Spire 2"
        $candidate = Join-Path $steamPath $commonPath
        if (Test-Path "$candidate\data_sts2_windows_x86_64") { return $candidate }
        
        $vdf = Join-Path $steamPath "steamapps\libraryfolders.vdf"
        if (Test-Path $vdf) {
            $content = Get-Content $vdf
            $paths = $content | Select-String -Pattern '"path"\s+"(.+?)"' | ForEach-Object { $_.Matches.Groups[1].Value -replace '\\\\', '\' }
            foreach ($p in $paths) {
                $candidate = Join-Path $p $commonPath
                if (Test-Path "$candidate\data_sts2_windows_x86_64") { return $candidate }
            }
        }
    }
    return $null
}

function Resolve-ModSourceDir {
    $directCandidates = @(
        (Join-Path $Root "STS2MCP"),
        (Join-Path $Root "TrainingScripts\STS2MCP")
    )
    foreach ($candidate in $directCandidates) {
        if (Test-Path (Join-Path $candidate "build.ps1")) {
            return $candidate
        }
    }

    $found = Get-ChildItem -Path $Root -Directory -Recurse -Filter "STS2MCP" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "build.ps1") } |
        Select-Object -First 1
    if ($found) {
        return $found.FullName
    }
    return $null
}

$ResolvedGameDir = Get-GameDir
if (-not $ResolvedGameDir) {
    Write-Host "Could not auto-detect the game directory. Enter the game root path:" -ForegroundColor Magenta
    Write-Host "(example: D:\SteamLibrary\steamapps\common\Slay the Spire 2)"
    $ResolvedGameDir = Read-Host "Path"
}

if (-not (Test-Path "$ResolvedGameDir\data_sts2_windows_x86_64")) {
    Write-Host "ERROR: Invalid game directory. Could not find game files under '$ResolvedGameDir'." -ForegroundColor Red
    pause
    exit 1
}
Write-Host "Game directory: $ResolvedGameDir"

# 3. Install STS2_MCP game mod
Write-Host "`n[3/3] Installing game mod (STS2_MCP)..." -ForegroundColor Yellow

$ModSourceDir = Resolve-ModSourceDir
if (-not $ModSourceDir) {
    Write-Host "ERROR: Could not find STS2MCP source directory in this workspace." -ForegroundColor Red
    pause
    exit 1
}
$BuildScript = Join-Path $ModSourceDir "build.ps1"
$DllSource = Join-Path $ModSourceDir "out\STS2_MCP\STS2_MCP.dll"
$JsonSource = Join-Path $ModSourceDir "mod_manifest.json"

if (-not (Test-Path $DllSource)) {
    Write-Host "STS2_MCP.dll was not found. Attempting to build it..." -ForegroundColor Yellow
    if (-not (Test-Path $BuildScript)) {
        Write-Host "ERROR: Mod build script not found: $BuildScript" -ForegroundColor Red
        pause
        exit 1
    }
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: dotnet was not found. Install .NET 9 SDK, then run this script again." -ForegroundColor Red
        Write-Host "Download: https://dotnet.microsoft.com/download/dotnet/9.0" -ForegroundColor Gray
        pause
        exit 1
    }

    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$BuildScript" -GameDir "$ResolvedGameDir"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $DllSource)) {
        Write-Host "ERROR: STS2_MCP build failed. Cannot install the mod." -ForegroundColor Red
        pause
        exit 1
    }
}

if (-not (Test-Path $JsonSource)) {
    Write-Host "ERROR: Mod manifest not found: $JsonSource" -ForegroundColor Red
    pause
    exit 1
}

$ModsDir = Join-Path $ResolvedGameDir "mods"
if (-not (Test-Path $ModsDir)) {
    New-Item -ItemType Directory -Path $ModsDir -Force | Out-Null
}

Copy-Item -Path $DllSource -Destination (Join-Path $ModsDir "STS2_MCP.dll") -Force
Copy-Item -Path $JsonSource -Destination (Join-Path $ModsDir "STS2_MCP.json") -Force
Write-Host "Mod installed to the game mods directory:" -ForegroundColor Green
$InstalledDll = Join-Path $ModsDir "STS2_MCP.dll"
$InstalledJson = Join-Path $ModsDir "STS2_MCP.json"
Write-Host "  $InstalledDll"
Write-Host "  $InstalledJson"

$OldNestedModDest = Join-Path $ResolvedGameDir "mods\STS2_MCP"
if (Test-Path (Join-Path $OldNestedModDest "STS2_MCP.dll")) {
    Write-Host "Note: Found an old nested install directory: $OldNestedModDest" -ForegroundColor DarkYellow
    Write-Host "The game should load STS2_MCP.dll and STS2_MCP.json from the mods root." -ForegroundColor DarkYellow
}

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "1. Start the game and enable 'STS2 MCP' in Settings -> Mods."
Write-Host "2. Open http://localhost:15526/ after entering the game to verify the Mod API."
Write-Host "3. Run 'start_all.bat' to start the control panel and AI."
Write-Host ""
pause
