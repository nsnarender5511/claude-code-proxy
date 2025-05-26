#!/bin/bash
set -e

# Ensure poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found in PATH"
    exit 1
fi

# Get poetry environment's Python path
VENV_PATH=$(poetry env info -p 2>/dev/null || true)
if [[ -z "$VENV_PATH" ]]; then
    echo "❌ Failed to get poetry virtual environment path"
    exit 1
fi

PYTHON="$VENV_PATH/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "❌ Python not found at $PYTHON"
    exit 1
fi

# Run the server
exec "$PYTHON" server.py
