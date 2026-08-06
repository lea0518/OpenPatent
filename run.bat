@echo off
REM One-click launch OpenPatent: cd to project root -> activate conda env -> start Web UI
cd /d %~dp0
call conda activate OpenPatent

REM Kill any old process still holding port 7860 (in case last run was not closed)
echo Checking port 7860 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
    echo Killing old process PID=%%p
    taskkill /F /PID %%p
)

python src/web_ui.py
pause
