#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${FLUIDV_PYTHON:-python}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
case "$PYTHON_BIN" in
  */*)
    PYTHON_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
    PYTHON_ENV_ROOT="$(cd "$PYTHON_DIR/.." && pwd)"
    export PATH="$PYTHON_DIR:$PATH"
    if [ -d "$PYTHON_ENV_ROOT/lib" ]; then
      export LD_LIBRARY_PATH="$PYTHON_ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
    fi
    ;;
esac

"$PYTHON_BIN" model/infer.py "$@"
