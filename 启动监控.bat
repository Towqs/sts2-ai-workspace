@echo off
chcp 65001 >nul
title RL Monitor
echo ================================
echo  RL Monitor - Waiting for game...
echo ================================
powershell.exe -ExecutionPolicy Bypass -NoProfile -NoExit -Command "Get-Content 'D:\2024 fa fan\XJ12615\STS2_AI_Workspace\RL_Datasets\rl_monitor.log' -Wait -Tail 0"
pause
