#!/usr/bin/env bash
# Builds the benchmark venv. No arguments.
#
#   benchmarks/start.sh
#
# torch installs with --torch-backend=auto, so a Linux box with a GPU driver gets CUDA torch and
# everything else gets the CPU build. Nothing to pass.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"

echo ">> bench-env"
uv venv benchmarks/.venv-bench --python 3.13 --clear
# shellcheck disable=SC1091
source benchmarks/.venv-bench/bin/activate
uv pip install torch --torch-backend=auto
uv pip install -e "$REPO" numba safetensors
# pytest is not optional here: the validation suite runs in this venv, against these package
# versions, and it is what says the harness measures correctly before a benchmark is believed.
uv pip install causal-learn "gcastle==1.0.4" "lingam==1.12.2" psutil pandas pyarrow pytest
python -c "import andrey, causallearn, castle, lingam, pyarrow, torch; import importlib.metadata as m; print('bench-env OK causal-learn', m.version('causal-learn'), 'cuda', torch.cuda.is_available())"
deactivate

echo ">> done"
