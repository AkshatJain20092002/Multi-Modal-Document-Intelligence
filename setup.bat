@echo off
setlocal
cd /d "%~dp0"

echo == Multi-Modal Document Intelligence -- setup ==

set VENV_DIR=venv

if not exist "%VENV_DIR%" (
    echo Creating virtual environment in .\%VENV_DIR% ...
    python -m venv "%VENV_DIR%"
) else (
    echo Virtual environment already exists at .\%VENV_DIR%, reusing it.
)

set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe
set PIP_BIN=%VENV_DIR%\Scripts\pip.exe

echo Upgrading pip...
"%PYTHON_BIN%" -m pip install --upgrade pip

echo Installing dependencies from requirements.txt ...
"%PIP_BIN%" install -r requirements.txt

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo Created .env from .env.example -- edit it and set OPENAI_API_KEY before running the pipeline.
    ) else (
        echo WARNING: .env.example not found -- create .env manually with OPENAI_API_KEY set.
    )
) else (
    echo .env already exists, leaving it as-is.
)

echo.
echo Setup complete.
echo Activate the environment yourself before running scripts directly:
echo   %VENV_DIR%\Scripts\activate.bat
echo Or just use run_question_bank.bat, which doesn't require activation.

endlocal
