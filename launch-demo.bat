@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import dual_arm_assembly" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m dual_arm_assembly demo
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or demo failed. See the message above and README.md.
pause
exit /b 1
