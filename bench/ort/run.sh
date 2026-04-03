#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

podman build -t ort-bench "$REPO_ROOT/bench/ort/"

TARGET="${1:?Usage: $0 <cpu|cuda> [extra args...]}"
shift

podman run --rm \
    --device nvidia.com/gpu=all \
    -v "$REPO_ROOT/models:/workspace/models:ro" \
    ort-bench \
    python3 benchmark.py "$TARGET" "$@"
