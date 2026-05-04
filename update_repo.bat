@echo off
chcp 65001 >nul
title STS2 AI Workspace Updater
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\update_workspace.ps1" %*

echo.
echo Updater finished. You can close this window.
pause >nul
