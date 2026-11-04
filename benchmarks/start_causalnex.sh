#!/usr/bin/env bash
# Builds the environment CausalNex's DYNOTEARS fits run in. No arguments.
#
#   benchmarks/start_causalnex.sh
#   export ANDREY_BENCH_CAUSALNEX_PYTHON=$PWD/benchmarks/.venv-causalnex/bin/python
#
# CausalNex 0.12.2 pins Python below 3.11, NumPy below 1.24, and pandas below 2, none of which
# .venv-bench can hold, so its fits run here. CausalNex imports PyTorch, which DYNOTEARS does not
# use, so the CPU build is installed. The child runs `python -m andrey_bench._worker`, which imports
# numpy and the adapter module; andrey stays in bench-env, where the parent scores the results. The
# final check fits DYNOTEARS on a small series.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-causalnex"

echo ">> causalnex-env"
uv venv "$VENV" --python 3.10 --clear
uv pip install --python "$VENV/bin/python" --torch-backend=cpu \
    "causalnex==0.12.2" "numpy==1.23.5" "pandas==1.5.3" "torch==2.1.2"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import numpy as np
import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic

rng = np.random.default_rng(0)
x = rng.standard_normal((300, 3))
for t in range(1, 300):
    x[t, 1] += 0.5 * x[t - 1, 0]
model = from_pandas_dynamic(pd.DataFrame(x, columns=["a", "b", "c"]), p=1, w_threshold=0.1)
assert ("a_lag1", "b_lag0") in model.edges
print(f"causalnex-env OK causalnex {m.version('causalnex')} numpy {np.__version__}")
PY

echo ">> done"
