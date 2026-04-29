@echo off
chcp 65001 >nul
title STS2 AI One Click Launcher
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_all.ps1" %*

echo.
echo Launcher finished. You can close this window.
pause >nul
