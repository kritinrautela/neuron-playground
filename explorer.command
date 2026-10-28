#!/bin/zsh
# Double-click to start the neuron explorer; it opens in your browser. Close this window to stop it.
cd "${0:A:h}"
source .venv/bin/activate
clear
if [[ ! -f index/index.pt ]]; then
    echo "First run: building the neuron index (~20 min, one time only)..."
    python build_index.py || exit 1
fi
python explorer.py
