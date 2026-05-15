@echo off
REM ============================================================================
REM LegalMind — Windows launcher
REM
REM Uses the local .venv (llama-cpp-python CUDA build copied from TRACE).
REM Double-click or pin to taskbar to launch.  Ctrl+C to stop the server.
REM
REM Port: 7860   URL: http://localhost:7860
REM ============================================================================
setlocal
cd /d "%~dp0"

set "PORT=7860"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHONPATH=%~dp0"

echo.
echo  ========================================
echo   PSL LegalMind  ^|  Document Intelligence
echo  ========================================
echo.

REM ── Verify the venv Python exists ─────────────────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo [LegalMind] ERROR: Local venv not found at:
    echo             %VENV_PYTHON%
    echo.
    echo             Run: python -m venv .venv  then pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ── Kill any process already holding port 7860 ────────────────────────────
echo [LegalMind] Clearing port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo [LegalMind]   Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

REM ── Open browser after 4 seconds (fire and forget) ────────────────────────
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:%PORT%"

REM ── Start the server (foreground — Ctrl+C to stop) ────────────────────────
echo [LegalMind] Starting on http://localhost:%PORT%
echo [LegalMind] Press Ctrl+C to stop.
echo.

"%VENV_PYTHON%" -m uvicorn webapp.main:app --host 0.0.0.0 --port %PORT% --reload

echo.
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo [LegalMind] Server exited with code %RC%.
    pause
)
endlocal & exit /b %RC%
