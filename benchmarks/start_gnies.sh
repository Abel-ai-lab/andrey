#!/usr/bin/env bash
# Builds the GnIES reference environment. No arguments.
#
#   benchmarks/start_gnies.sh
#   export ANDREY_BENCH_GNIES_PYTHON=$PWD/benchmarks/.venv-gnies/bin/python
#
# Separate from .venv-bench so the pins stay exact and the PyPI module `ges`, which gnies imports,
# stays out of the bench environment. gnies 0.3.3 needs only NumPy and ges; nothing else goes in,
# since its fits run `gnies_worker.py`, which imports numpy and gnies alone.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-gnies"

echo ">> gnies-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" "gnies==0.3.3" "ges==1.1.1" "numpy==2.5.4"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import numpy as np
import gnies

data = [np.random.default_rng(k).normal(size=(100, 3)) for k in range(2)]
score, icpdag, targets = gnies.fit(data)
print(f"gnies-env OK gnies {m.version('gnies')} ges {m.version('ges')} numpy {np.__version__}")
PY

echo ">> done"
