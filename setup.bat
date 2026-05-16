@echo off
title LegalMind - Setup
echo.
echo ============================================================
echo   LegalMind - AI-Powered Legal Document Intelligence
echo   One-Click Setup
echo ============================================================
echo.

:: Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo [OK] Python %PYVER% detected

:: Create virtual environment if not exists
if not exist ".venv" (
    echo.
    echo [STEP 1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

:: Activate and install dependencies
echo.
echo [STEP 2/4] Installing dependencies...
call .venv\Scripts\activate.bat

pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt 2>&1 | findstr /V "already satisfied"
if errorlevel 1 (
    echo [WARN] Some packages may have failed. Continuing...
)
echo [OK] Dependencies installed

:: Verify llama-cpp-python with CUDA support
echo.
echo [STEP 3/4] Checking llama-cpp-python (CUDA)...

:: First check if already installed and working
.venv\Scripts\python -c "from llama_cpp import Llama; print('[OK] llama-cpp-python with CUDA already installed')" 2>nul
if not errorlevel 1 goto :llama_done

:: Not installed — try copying from TRACE venv (same Python 3.11, pre-compiled CUDA build)
echo [INFO] llama-cpp-python not found. Attempting to copy from TRACE venv...
if exist "F:\Research_Paper_Projects\TRACE\.venv\Lib\site-packages\llama_cpp" (
    xcopy /E /I /Y "F:\Research_Paper_Projects\TRACE\.venv\Lib\site-packages\llama_cpp" ".venv\Lib\site-packages\llama_cpp" >nul 2>&1
    xcopy /E /I /Y "F:\Research_Paper_Projects\TRACE\.venv\Lib\site-packages\llama_cpp_python-0.3.20.dist-info" ".venv\Lib\site-packages\llama_cpp_python-0.3.20.dist-info" >nul 2>&1
    .venv\Scripts\python -c "from llama_cpp import Llama; print('[OK] llama-cpp-python copied from TRACE venv')" 2>nul
    if not errorlevel 1 goto :llama_done
)

:: Fallback: compile from source with CUDA
echo [INFO] Compiling llama-cpp-python from source (this takes several minutes)...
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    set CMAKE_ARGS=-DGGML_CUDA=on
    pip install --force-reinstall llama-cpp-python==0.3.20 2>&1 | findstr /V "already satisfied"
) else (
    echo [WARN] CUDA not detected - installing CPU-only version
    pip install llama-cpp-python==0.3.20 2>&1 | findstr /V "already satisfied"
)

:llama_done

:: Check model directory
echo.
echo [STEP 4/4] Checking model configuration...

:: Check for LEGALMIND_MODEL_DIR env var first
if defined LEGALMIND_MODEL_DIR (
    echo [OK] Model directory from environment: %LEGALMIND_MODEL_DIR%
    set MODEL_DIR=%LEGALMIND_MODEL_DIR%
) else (
    :: Read from models.yaml
    set MODEL_DIR=F:\Research_Paper_Projects\LLMs
    echo [INFO] Model directory: %MODEL_DIR%
)

if exist "%MODEL_DIR%" (
    echo [OK] Model directory found
    dir /B "%MODEL_DIR%\*.gguf" 2>nul | findstr /C:".gguf" >nul
    if errorlevel 1 (
        echo [WARN] No .gguf model files found in %MODEL_DIR%
        echo        The system will run in degraded mode without LLMs.
        echo        Download models and place them in the model directory.
    ) else (
        echo [OK] GGUF model files detected
    )
) else (
    echo [WARN] Model directory not found: %MODEL_DIR%
    echo        Set LEGALMIND_MODEL_DIR environment variable to your model path.
    echo        Or edit config/models.yaml to set model_cache_dir.
    echo        The system will still work for text extraction and retrieval.
)

:: Create data directories
echo.
.venv\Scripts\python -c "from config.paths import ensure_dirs; ensure_dirs(); print('[OK] Data directories created')"

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To start the server, run:  run.bat
echo   Or manually:  .venv\Scripts\python -m uvicorn webapp.main:app --reload
echo.
echo   Web UI will be at:  http://localhost:8000
echo ============================================================
echo.
pause
