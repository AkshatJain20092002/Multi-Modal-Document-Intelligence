#!/usr/bin/env bash
# One-time environment setup: creates a virtualenv, installs dependencies,
# and prepares .env. Works via Git Bash on Windows or a real shell on Linux.
set -e

VENV_DIR="venv"

echo "== Multi-Modal Document Intelligence -- setup =="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR ..."
    python -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at ./$VENV_DIR, reusing it."
fi

# venv layout differs by OS: Windows uses Scripts/, Linux/macOS uses bin/
if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
    PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
    PIP_BIN="$VENV_DIR/Scripts/pip.exe"
    ACTIVATE_HINT="source $VENV_DIR/Scripts/activate"
else
    PYTHON_BIN="$VENV_DIR/bin/python"
    PIP_BIN="$VENV_DIR/bin/pip"
    ACTIVATE_HINT="source $VENV_DIR/bin/activate"
fi

echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt ..."
"$PIP_BIN" install -r requirements.txt

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example -- edit it and set OPENAI_API_KEY before running the pipeline."
    else
        echo "WARNING: .env.example not found -- create .env manually with OPENAI_API_KEY set."
    fi
else
    echo ".env already exists, leaving it as-is."
fi

echo
echo "Setup complete."
echo "Activate the environment yourself before running scripts directly:"
echo "  $ACTIVATE_HINT"
echo "Or just use ./run_question_bank.sh, which doesn't require activation."
