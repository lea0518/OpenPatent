@echo off
REM 一键启动 OpenPatent：切到项目根目录 -> 激活 conda 环境 -> 启动 Web UI
cd /d %~dp0
call conda activate OpenPatent
python src/web_ui.py
pause
