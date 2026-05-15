@echo off
REM ============================================================================
REM LegalMind -- Windows launcher.
REM
REM Kills any orphan uvicorn/python processes holding port 8000, then
REM bootstraps the venv and launches the server fresh.
REM ============================================================================
setlocal
cd /d "%~dp0"

REM --- Kill orphans holding port 8000 -----------------------------------------
echo [LegalMind] Cleaning up orphan processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [LegalMind]   Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

REM --- Locate Python -----------------------------------------------------------
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
    goto :run
)
where python >nul 2>&1
if %errorlevel%==0 (
    set "PY=python"
    goto :run
)

echo.
echo [LegalMind] Python 3.11+ is required but was not found on PATH.
echo             Install from https://www.python.org/downloads/
echo             (tick "Add Python to PATH" during install)
echo.
pause
exit /b 1

:run
echo [LegalMind] Starting...
%PY% "scripts\bootstrap.py" %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [LegalMind] Launcher exited with code %RC%.
    echo.
    pause
)
endlocal & exit /b %RC%
