@echo off
chcp 65001 >nul
title STS2 AI 一键安装工具
cd /d "%~dp0"

echo 正在启动安装脚本...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup_Environment.ps1"

exit
