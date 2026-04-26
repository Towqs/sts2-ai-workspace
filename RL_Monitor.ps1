# RL 实时监控脚本 v2 — 显示训练数据采集状态
# 用法: 在 PowerShell 里直接运行此脚本

param(
    [string]$LogFile = "D:\2024 fa fan\XJ12615\STS2_AI_Workspace\RL_Datasets\rl_monitor.log",
    [string]$CombatDir = "D:\2024 fa fan\XJ12615\STS2_AI_Workspace\RL_Datasets\Combat",
    [string]$MacroDir = "D:\2024 fa fan\XJ12615\STS2_AI_Workspace\RL_Datasets\Macro"
)

# 如果日志不存在就建一个空文件
if (-not (Test-Path $LogFile)) {
    New-Item -Path $LogFile -ItemType File -Force | Out-Null
}

function Get-LogStats {
    param([string]$Dir)
    if (Test-Path $Dir) {
        $files = Get-ChildItem $Dir -Filter "*.jsonl" -ErrorAction SilentlyContinue
        $count = 0
        $size = 0
        foreach ($f in $files) {
            $count++
            $size += $f.Length
        }
        return @{ Count = $count; Size = $size; SizeMB = [math]::Round($size / 1MB, 2) }
    }
    return @{ Count = 0; Size = 0; SizeMB = 0 }
}

function Write-ColorLine {
    param([string]$line)
    if ($line -match "\[INIT\]") {
        Write-Host $line -ForegroundColor Green
    }
    elseif ($line -match "battle_start") {
        Write-Host $line -ForegroundColor Magenta
    }
    elseif ($line -match "battle_end.*win") {
        Write-Host $line -ForegroundColor Green
    }
    elseif ($line -match "battle_end.*lose") {
        Write-Host $line -ForegroundColor Red
    }
    elseif ($line -match "turn_start") {
        Write-Host $line -ForegroundColor Cyan
    }
    elseif ($line -match "turn_end") {
        Write-Host $line -ForegroundColor DarkYellow
    }
    elseif ($line -match "COMBAT.*action=play_card") {
        Write-Host $line -ForegroundColor White
    }
    elseif ($line -match "COMBAT.*action=use_potion") {
        Write-Host $line -ForegroundColor Yellow
    }
    elseif ($line -match "COMBAT.*action=end_turn") {
        Write-Host $line -ForegroundColor DarkYellow
    }
    elseif ($line -match "COMBAT") {
        Write-Host $line -ForegroundColor Cyan
    }
    elseif ($line -match "MACRO.*map_node") {
        Write-Host $line -ForegroundColor Magenta
    }
    elseif ($line -match "MACRO.*choose_card") {
        Write-Host $line -ForegroundColor Yellow
    }
    elseif ($line -match "MACRO.*buy_item") {
        Write-Host $line -ForegroundColor DarkCyan
    }
    elseif ($line -match "MACRO") {
        Write-Host $line -ForegroundColor Blue
    }
    elseif ($line -match "\[ERROR\]") {
        Write-Host $line -ForegroundColor Red
    }
    elseif ($line -match "\[NEW FILE\]") {
        Write-Host $line -ForegroundColor Green
    }
    else {
        Write-Host $line -ForegroundColor DarkGray
    }
}

Clear-Host
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  STS2 AI 训练数据 实时监控 v2" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  监控文件: $LogFile" -ForegroundColor Gray
Write-Host ""
Write-Host "  数据统计:" -ForegroundColor Gray

# 持续读取新内容
$lastLength = (Get-Item $LogFile).Length
$combatWins = 0
$combatLosses = 0
$lastStatsUpdate = Get-Date

while ($true) {
    Start-Sleep -Milliseconds 300
    $currentLength = (Get-Item $LogFile).Length

    # 每 5 秒更新统计
    if ((Get-Date) - $lastStatsUpdate -gt [TimeSpan]::FromSeconds(5)) {
        $lastStatsUpdate = Get-Date
        $combatStats = Get-LogStats $CombatDir
        $macroStats = Get-LogStats $MacroDir

        # 移动光标到开头更新统计
        $statsLine = "  ├─ Combat: $($combatStats.Count) 文件, $($combatStats.SizeMB) MB"
        $macroLine = "  └─ Macro:  $($macroStats.Count) 文件, $($macroStats.SizeMB) MB"
        Write-Host "`r$statsLine" -ForegroundColor Gray
        Write-Host "`r$macroLine" -ForegroundColor Gray
        Write-Host ""
    }

    if ($currentLength -gt $lastLength) {
        $stream = [System.IO.File]::Open($LogFile, 'Open', 'Read', 'ReadWrite')
        $stream.Seek($lastLength, 'Begin') | Out-Null
        $reader = New-Object System.IO.StreamReader($stream)
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ($line.Trim() -ne "") {
                Write-ColorLine $line

                # 统计战斗结果
                if ($line -match "battle_end result=win") { $combatWins++ }
                if ($line -match "battle_end result=lose") { $combatLosses++ }
            }
        }
        $reader.Close()
        $stream.Close()
        $lastLength = $currentLength

        # 显示实时统计
        if ($combatWins + $combatLosses -gt 0) {
            $winRate = [math]::Round($combatWins / ($combatWins + $combatLosses) * 100, 1)
            Write-Host "    战绩: $combatWins 胜 / $combatLosses 败 (胜率: $winRate%)" -ForegroundColor Gray
        }
    }
}
