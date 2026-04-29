@echo off
chcp 65001 >nul
title RL Monitor
cd /d "%~dp0"
powershell.exe -ExecutionPolicy Bypass -NoProfile -NoExit -File "%~dp0tools\watch_rl_logs.ps1"
pause
