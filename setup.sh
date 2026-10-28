#!/usr/bin/env bash
# One-time setup: Python environment, packages, and the model weights (~5.4 GB download).
# Run from the project folder:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    echo "==> creating .venv with uv"
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
else
    PY=$(command -v python3.12 || command -v python3 || true)
    if [ -z "$PY" ]; then
        echo "Python 3.10+ is needed. Install it from https://www.python.org/downloads/ (or install uv) and re-run." >&2
        exit 1
    fi
    echo "==> creating .venv with $PY"
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

echo "==> downloading LiquidAI/LFM2.5-2.6B into hf_cache/ (~5.4 GB, one time)"
HF_HOME="$PWD/hf_cache" .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('LiquidAI/LFM2.5-2.6B')
"

echo "==> checking the model loads"
.venv/bin/python inspect_model.py | tail -3

chmod +x start.sh explorer.command play.command 2>/dev/null || true
echo
echo "Done. Start the explorer with:  ./start.sh   (Mac: or double-click explorer.command)"
