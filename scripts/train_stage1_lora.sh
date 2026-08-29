#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${FLUIDV_PYTHON:-python}"
export PYTHONPATH="$PWD:$PWD/llamafactory/src:${PYTHONPATH:-}"
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
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

"$PYTHON_BIN" -m llamafactory.cli train config/train_openpangu_vl_nothink_lora.yaml
