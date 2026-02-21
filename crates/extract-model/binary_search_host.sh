#!/bin/bash
# Binary search wrapper that runs on the host (no container)
# This is useful for debugging and when you have Python/ONNX installed locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use venv if it exists
if [ -d "/tmp/onnx-venv" ]; then
    PYTHON="/tmp/onnx-venv/bin/python3"
else
    PYTHON="python3"
fi

cd "$PROJECT_ROOT"

exec "$PYTHON" "$SCRIPT_DIR/binary_search.py" "$@"
