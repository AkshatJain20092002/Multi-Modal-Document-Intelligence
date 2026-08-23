@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: %~nx0 ^<input-file-or-folder^> [doc-id]
    echo   e.g. %~nx0 sample_data\task_page-0016.jpg
    echo        %~nx0 sample_data my_full_document
    exit /b 1
)

set VENV_DIR=venv
set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe

if not exist "%PYTHON_BIN%" (
    echo No virtual environment found at .\%VENV_DIR% -- run setup.bat first.
    exit /b 1
)

rem Avoids a Windows console crash on Unicode characters that show up in real OCR'd text.
set PYTHONIOENCODING=utf-8

set INPUT_PATH=%~1
set DOC_ID_ARG=
if not "%~2"=="" set DOC_ID_ARG=--doc-id "%~2"

"%PYTHON_BIN%" scripts\run_full_pipeline.py --input "%INPUT_PATH%" %DOC_ID_ARG%

endlocal
