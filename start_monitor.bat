@echo off
chcp 65001 >nul
title STS2 AI Log Monitor
cd /d "%~dp0"

powershell.exe -ExecutionPolicy Bypass -NoProfile -NoExit -File "%~dp0tools\watch_rl_logs.ps1" %*
