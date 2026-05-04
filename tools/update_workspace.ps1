[CmdletBinding()]
param(
    [switch]$SkipDependencySync,
    [switch]$DryRun,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

function Finish-Script {
    param([int]$Code = 0)
    if (-not $NoPause) {
        Write-Host ""
        pause
    }
    exit $Code
}

function Fail-Script {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
    Finish-Script 1
}

function Test-CommandExists {
    param([string]$Name)
    try {
        $null = Get-Command $Name -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Invoke-Git {
    param(
        [string[]]$GitArgs,
        [switch]$AllowFailure,
        [switch]$WriteOperation
    )

    $commandText = "git " + ($GitArgs -join " ")
    if ($DryRun -and $WriteOperation) {
        Write-Host "[dry-run] $commandText" -ForegroundColor DarkGray
        return @()
    }

    $output = & git @GitArgs 2>&1
    $exitCode = $LASTEXITCODE
    if (-not $AllowFailure -and $exitCode -ne 0) {
        $joined = ($output | Out-String).Trim()
        if ($joined) {
            throw "Command failed ($exitCode): $commandText`n$joined"
        }
        throw "Command failed ($exitCode): $commandText"
    }
    return @($output)
}

function Test-GitDiffQuiet {
    param([string[]]$DiffArgs)
    if ($DryRun) {
        return $false
    }
    & git diff @DiffArgs --quiet --ignore-submodules --
    return ($LASTEXITCODE -eq 0)
}

function Resolve-PythonExe {
    $candidates = @(
        (Join-Path $Root ".venv\Scripts\python.exe"),
        "python.exe",
        "python"
    )

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        try {
            $output = & $candidate --version 2>&1
            if ($LASTEXITCODE -eq 0 -and (($output -join "`n") -match "Python")) {
                return $candidate
            }
        } catch {
        }
    }
    return ""
}

function Sync-Dependencies {
    $pythonExe = Resolve-PythonExe
    if (-not $pythonExe) {
        Write-Warning "No runnable Python found. Skipping dependency sync. If needed, run 一键安装环境与Mod.bat afterwards."
        return $false
    }

    $requirementsPath = Join-Path $Root "requirements.txt"
    if (-not (Test-Path -LiteralPath $requirementsPath)) {
        Write-Warning "requirements.txt not found. Skipping dependency sync."
        return $false
    }

    if ($DryRun) {
        Write-Host "[dry-run] $pythonExe -m pip install -r $requirementsPath" -ForegroundColor DarkGray
        return $true
    }

    Write-Host "Syncing Python dependencies..." -ForegroundColor Yellow
    & $pythonExe -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency sync failed. Please run 一键安装环境与Mod.bat to rebuild the environment."
    }
    return $true
}

Write-Host "=== STS2 AI Workspace Updater ===" -ForegroundColor Cyan
Write-Host "Workspace: $Root"
if ($DryRun) {
    Write-Host "Mode: dry-run (no changes will be written)" -ForegroundColor DarkYellow
}

if (-not (Test-Path -LiteralPath (Join-Path $Root ".git"))) {
    Fail-Script "This folder is not a Git repository."
}
if (-not (Test-CommandExists "git")) {
    Fail-Script "git was not found. Install Git first, then run this updater again."
}

$mergeHead = Join-Path $Root ".git\MERGE_HEAD"
$rebaseMerge = Join-Path $Root ".git\rebase-merge"
$rebaseApply = Join-Path $Root ".git\rebase-apply"
if ((Test-Path -LiteralPath $mergeHead) -or (Test-Path -LiteralPath $rebaseMerge) -or (Test-Path -LiteralPath $rebaseApply)) {
    Fail-Script "Git is already in the middle of a merge or rebase. Finish that first, then run the updater again."
}

$branch = ((Invoke-Git -GitArgs @("branch", "--show-current")) | Select-Object -First 1).ToString().Trim()
if (-not $branch) {
    Fail-Script "Detached HEAD detected. Please switch back to a branch before updating."
}

$originUrl = ((Invoke-Git -GitArgs @("remote", "get-url", "origin")) | Select-Object -First 1).ToString().Trim()
if (-not $originUrl) {
    Fail-Script "Remote 'origin' is missing. Cannot auto-update this workspace."
}

$headBefore = ((Invoke-Git -GitArgs @("rev-parse", "HEAD")) | Select-Object -First 1).ToString().Trim()
$headBeforeShort = ((Invoke-Git -GitArgs @("rev-parse", "--short", "HEAD")) | Select-Object -First 1).ToString().Trim()

Write-Host ""
Write-Host "Branch: $branch"
Write-Host "Remote: $originUrl"
Write-Host "Current commit: $headBeforeShort"

$untracked = @(Invoke-Git -GitArgs @("ls-files", "--others", "--exclude-standard") -AllowFailure) |
    ForEach-Object { $_.ToString().Trim() } |
    Where-Object { $_ }
if ($untracked.Count -gt 0) {
    Write-Warning "Untracked local files will be kept as-is. If remote adds files with the same names, update may fail."
    $preview = $untracked | Select-Object -First 8
    foreach ($item in $preview) {
        Write-Host "  $item" -ForegroundColor DarkYellow
    }
    if ($untracked.Count -gt $preview.Count) {
        Write-Host "  ... and $($untracked.Count - $preview.Count) more" -ForegroundColor DarkYellow
    }
}

Write-Host ""
Write-Host "Fetching latest changes..." -ForegroundColor Yellow
Invoke-Git -GitArgs @("fetch", "origin", $branch, "--prune") -WriteOperation

$remoteRef = "origin/$branch"
$ahead = 0
$behind = 0
$countsText = ((Invoke-Git -GitArgs @("rev-list", "--left-right", "--count", "HEAD...$remoteRef") -AllowFailure) | Select-Object -First 1)
if ($countsText) {
    $parts = @($countsText.ToString().Trim() -split "\s+") | Where-Object { $_ -ne "" }
    if ($parts.Count -ge 2) {
        $ahead = [int]$parts[0]
        $behind = [int]$parts[1]
    }
}

if (-not $DryRun) {
    if ($ahead -gt 0 -and $behind -gt 0) {
        Fail-Script "Local branch and remote branch have both changed. To keep this updater safe for new users, it will stop here. Please handle this branch manually."
    }
    if ($behind -eq 0) {
        if ($ahead -gt 0) {
            Write-Host "Local branch is ahead of origin by $ahead commit(s). Nothing was pulled." -ForegroundColor Green
        } else {
            Write-Host "Already up to date." -ForegroundColor Green
        }
        Finish-Script 0
    }
}

$stashCreated = $false
$stashLabel = ""
if (-not $DryRun) {
    $hasUnstaged = -not (Test-GitDiffQuiet @())
    $hasStaged = -not (Test-GitDiffQuiet @("--cached"))
    if ($hasUnstaged -or $hasStaged) {
        $stashLabel = "auto-update-backup-" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Write-Host ""
        Write-Host "Local tracked changes detected. Creating a safety stash first..." -ForegroundColor Yellow
        Invoke-Git -GitArgs @("stash", "push", "--message", $stashLabel) -WriteOperation
        $stashCreated = $true
    }
} else {
    Write-Host "[dry-run] would create a safety stash if tracked changes exist" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Pulling latest version..." -ForegroundColor Yellow
Invoke-Git -GitArgs @("pull", "--ff-only", "origin", $branch) -WriteOperation

$headAfter = $headBefore
$headAfterShort = $headBeforeShort
$changedFiles = @()
if (-not $DryRun) {
    $headAfter = ((Invoke-Git -GitArgs @("rev-parse", "HEAD")) | Select-Object -First 1).ToString().Trim()
    $headAfterShort = ((Invoke-Git -GitArgs @("rev-parse", "--short", "HEAD")) | Select-Object -First 1).ToString().Trim()
    $changedFiles = @(Invoke-Git -GitArgs @("diff", "--name-only", "$headBefore..$headAfter")) |
        ForEach-Object { $_.ToString().Trim() } |
        Where-Object { $_ }
} else {
    Write-Host "[dry-run] would compare changed files between the old and new commit" -ForegroundColor DarkGray
}

$syncedDeps = $false
if (-not $SkipDependencySync) {
    if ($DryRun -or ($changedFiles -contains "requirements.txt")) {
        $syncedDeps = Sync-Dependencies
    } else {
        Write-Host "requirements.txt did not change. Skipping dependency sync."
    }
} else {
    Write-Host "Dependency sync skipped by option."
}

Write-Host ""
Write-Host "Update complete." -ForegroundColor Green
Write-Host "Commit: $headBeforeShort -> $headAfterShort"
if (-not $DryRun -and $changedFiles.Count -gt 0) {
    Write-Host "Changed files:"
    foreach ($item in ($changedFiles | Select-Object -First 12)) {
        Write-Host "  $item"
    }
    if ($changedFiles.Count -gt 12) {
        Write-Host "  ... and $($changedFiles.Count - 12) more"
    }
}
if ($stashCreated) {
    Write-Host ""
    Write-Host "Your previous tracked changes were saved to Git stash:" -ForegroundColor Yellow
    Write-Host "  $stashLabel"
    Write-Host "If you want them back later, run:" -ForegroundColor Yellow
    Write-Host "  git stash list"
    Write-Host "  git stash pop"
}
if ($syncedDeps) {
    Write-Host ""
    Write-Host "Python dependencies were synced." -ForegroundColor Green
}

Finish-Script 0
