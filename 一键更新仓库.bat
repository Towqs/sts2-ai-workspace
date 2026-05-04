@echo off
chcp 65001 >nul
title STS2 AI 一键更新工具
cd /d "%~dp0"

call "%~dp0update_repo.bat" %*
