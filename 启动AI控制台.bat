@echo off
chcp 65001 >nul
title STS2 AI Control Panel
cd /d "%~dp0"
echo Starting STS2 AI Control Panel...
echo Open http://127.0.0.1:8765 in your browser.
.\.venv\Scripts\python.exe AI_Training\control_panel.py
pause
