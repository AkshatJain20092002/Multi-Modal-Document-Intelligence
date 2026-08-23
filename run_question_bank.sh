#!/usr/bin/env bash
# Runs the full pipeline (extract -> markdown -> chunk+embed -> detect ->
# answer -> validate -> escalate) and produces an Excel Question Bank.
#
# Usage:
#   ./run_question_bank.sh <input-file-or-folder> [doc-id]
#   ./run_question_bank.sh sample_data/task_page-0016.jpg
#   ./run_question_bank.sh sample_data my_full_document
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input-file-or-folder> [doc-id]"
    echo "  e.g. $0 sample_data/task_page-0016.jpg"
    echo "       $0 sample_data my_full_document"
    exit 1
fi

VENV_DIR="venv"
if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
    PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
elif [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    echo "No virtual environment found at ./$VENV_DIR -- run ./setup.sh first."
    exit 1
fi

# Avoids a Windows console crash on Unicode characters that show up in real OCR'd text.
export PYTHONIOENCODING=utf-8

INPUT_PATH="$1"
DOC_ID_ARGS=()
if [ -n "$2" ]; then
    DOC_ID_ARGS=(--doc-id "$2")
fi

"$PYTHON_BIN" scripts/run_full_pipeline.py --input "$INPUT_PATH" "${DOC_ID_ARGS[@]}"
