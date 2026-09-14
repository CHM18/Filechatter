@echo off
setlocal EnableExtensions

REM Filechatter one-click Windows setup + launch script
REM - Ensures Python is available
REM - Creates venv if missing (prefers existing .\.venv or legacy names)
REM - Sets PowerShell execution policy (CurrentUser RemoteSigned)
REM - Activates venv, installs requirements, starts rag_server.py

cd /d "%~dp0"

set "SCRIPT_DIR=%~dp0"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"
set "REQ_DEV_FILE=%SCRIPT_DIR%requirements-dev.txt"
set "SERVER_FILE=%SCRIPT_DIR%rag_server.py"

set "VENV_DIR=%SCRIPT_DIR%.venv_filechatter"
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "VENV_DIR=%SCRIPT_DIR%.venv"
) else if exist "%SCRIPT_DIR%.venv_filechatter\Scripts\python.exe" (
    set "VENV_DIR=%SCRIPT_DIR%.venv_filechatter"
) else if exist "%SCRIPT_DIR%filechatter\Scripts\python.exe" (
    set "VENV_DIR=%SCRIPT_DIR%filechatter"
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "VENV_DIR=%SCRIPT_DIR%venv"
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE_BAT=%VENV_DIR%\Scripts\activate.bat"

echo.
echo == Filechatter setup and launch ==
echo Project: %SCRIPT_DIR%
echo Virtualenv: %VENV_DIR%

if not exist "%SERVER_FILE%" (
    echo [ERROR] rag_server.py not found in this folder.
    exit /b 1
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PY_LAUNCHER=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "PY_LAUNCHER=python"
    ) else (
        echo [ERROR] Python is not installed or not in PATH.
        echo Install Python first, then rerun this script.
        echo Example: winget install -e --id Python.Python.3.12
        exit /b 1
    )
)

if not exist "%VENV_PY%" (
    echo.
    echo [INFO] Creating virtual environment at "%VENV_DIR%" ...
    %PY_LAUNCHER% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

echo.
echo [INFO] Setting PowerShell execution policy (CurrentUser RemoteSigned)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force -ErrorAction Stop; Write-Host 'Execution policy set.' } catch { Write-Host 'Could not set execution policy:' $_.Exception.Message }"

if not exist "%VENV_ACTIVATE_BAT%" (
    echo [ERROR] Could not find activate.bat at "%VENV_ACTIVATE_BAT%".
    exit /b 1
)

echo.
echo [INFO] Activating virtual environment...
call "%VENV_ACTIVATE_BAT%"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo.
echo [INFO] Checking installed dependencies ...
python -m pip check >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies from requirements.txt ...
    python -m pip install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        exit /b 1
    )
    if exist "%REQ_DEV_FILE%" (
        echo [INFO] Installing development dependencies from requirements-dev.txt ...
        python -m pip install -r "%REQ_DEV_FILE%"
        if errorlevel 1 (
            echo [ERROR] Development dependency installation failed.
            exit /b 1
        )
    )
) else (
    echo [INFO] Dependencies already satisfied; skipping installation.
)

echo.
echo [INFO] Starting rag_server.py ...
python "%SERVER_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [INFO] rag_server.py exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
