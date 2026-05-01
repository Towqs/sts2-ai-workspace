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
Write-Host "=== STS2 AI 环境一键配置工具 ===" -ForegroundColor Cyan
Write-Host "项目路径: $Root"

# 1. 检查并配置 Python 环境
Write-Host "`n[1/3] 正在配置 Python 虚拟环境..." -ForegroundColor Yellow
$PythonExe = "python.exe"
try {
    $null = & $PythonExe --version 2>&1
} catch {
    Write-Host "错误: 系统未找到 python。请安装 Python 3.10+ 并勾选 'Add to PATH'。" -ForegroundColor Red
    pause
    exit 1
}

$VenvPath = Join-Path $Root ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "正在创建虚拟环境 (.venv)..."
    & $PythonExe -m venv "$VenvPath"
} else {
    Write-Host "虚拟环境已存在。"
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "正在安装/更新依赖 (requirements.txt)..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

# 2. 自动定位游戏目录
Write-Host "`n[2/3] 正在定位《杀戮尖塔2》安装目录..." -ForegroundColor Yellow

function Get-GameDir {
    if ($GameDir) { return $GameDir }
    
    # 尝试从注册表读取 Steam 路径
    $steamPath = (Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -Name "SteamPath" -ErrorAction SilentlyContinue).SteamPath
    if ($steamPath) {
        $commonPath = "steamapps\common\Slay the Spire 2"
        # 检查主库
        $candidate = Join-Path $steamPath $commonPath
        if (Test-Path "$candidate\data_sts2_windows_x86_64") { return $candidate }
        
        # 检查其他库 (libraryfolders.vdf)
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

$ResolvedGameDir = Get-GameDir
if (-not $ResolvedGameDir) {
    Write-Host "未能自动找到游戏目录，请手动输入游戏根目录路径:" -ForegroundColor Magenta
    Write-Host "(例如: D:\SteamLibrary\steamapps\common\Slay the Spire 2)"
    $ResolvedGameDir = Read-Host "路径"
}

if (-not (Test-Path "$ResolvedGameDir\data_sts2_windows_x86_64")) {
    Write-Host "错误: 路径无效，未在 '$ResolvedGameDir' 找到游戏文件。" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "已找到游戏目录: $ResolvedGameDir"

# 3. 安装 Mod 插件
Write-Host "`n[3/3] 正在安装游戏 Mod 插件 (STS2_MCP)..." -ForegroundColor Yellow

$ModDest = Join-Path $ResolvedGameDir "mods\STS2_MCP"
if (-not (Test-Path $ModDest)) {
    New-Item -ItemType Directory -Path $ModDest -Force | Out-Null
}

$DllSource = Join-Path $Root "训练脚本\STS2MCP\out\STS2_MCP\STS2_MCP.dll"
$JsonSource = Join-Path $Root "训练脚本\STS2MCP\mod_manifest.json"

if (-not (Test-Path $DllSource)) {
    Write-Host "警告: 未在项目中找到编译好的 STS2_MCP.dll。" -ForegroundColor Red
    Write-Host "请确保你已经运行过编译脚本，或者将 DLL 放在: $DllSource" -ForegroundColor Gray
} else {
    Copy-Item -Path $DllSource -Destination (Join-Path $ModDest "STS2_MCP.dll") -Force
    Copy-Item -Path $JsonSource -Destination (Join-Path $ModDest "STS2_MCP.json") -Force
    Write-Host "Mod 插件已成功安装到: $ModDest" -ForegroundColor Green
}

Write-Host "`n=== 配置完成！ ===" -ForegroundColor Green
Write-Host "1. 启动游戏并在 Mod 管理器中启用 'STS2 MCP'。"
Write-Host "2. 运行 '一键启动全部.bat' 开始使用。"
Write-Host ""
pause
