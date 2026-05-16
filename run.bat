@echo off
title LegalMind Server
echo.
echo ============================================================
echo   LegalMind - AI-Powered Legal Document Intelligence
echo ============================================================
echo.

:: Activate virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:: Set environment
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: Add CUDA to PATH if available
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin" (
    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
)

:: Auto-seed sample data if no processed documents exist
if not exist "data\processed" mkdir data\processed
dir /B "data\processed\*.json" 2>nul | findstr /C:".json" >nul
if errorlevel 1 (
    echo [SETUP] First run detected - seeding sample data...
    .venv\Scripts\python scripts\seed_sample_data.py 2>nul
    if not errorlevel 1 (
        echo [OK] Sample documents populated
    ) else (
        echo [WARN] Sample data seed had issues - continuing anyway
    )
)

:: Check model availability
echo.
echo   Model Configuration:
.venv\Scripts\python -c "from config.paths import check_model_availability; info=check_model_availability(); print(f'    Directory: {info[\"model_dir\"]}'); print(f'    Available: {info[\"dir_exists\"]}'); [print(f'      {k}: {\"READY\" if v[\"exists\"] else \"MISSING\"} ({v[\"size_gb\"]}GB)') for k,v in info.get('models',{}).items()]" 2>nul
echo.

:: Start server
echo   Starting LegalMind at http://localhost:8000
echo   Press Ctrl+C to stop
echo.

.venv\Scripts\python -m uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir code --reload-dir webapp --reload-dir config
