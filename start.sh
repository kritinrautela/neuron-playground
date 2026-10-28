#!/usr/bin/env bash
# Start the neuron explorer; it opens http://127.0.0.1:8765 in your browser. Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Run ./setup.sh first." >&2
    exit 1
fi
if [ ! -f index/index.pt ]; then
    echo "Building the neuron index (~20 min, one time only)..."
    .venv/bin/python build_index.py
fi
.venv/bin/python explorer.py
