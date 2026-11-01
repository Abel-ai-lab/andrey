#!/usr/bin/env bash
# Builds the environment lingam's fits, and causal-learn's CAM-UV, run in. No arguments.
#
#   benchmarks/start_lingam.sh
#   export ANDREY_BENCH_LINGAM_PYTHON=$PWD/benchmarks/.venv-lingam/bin/python
#
# Separate from .venv-bench because lingam 1.13.0 pins scipy 1.13.1 or older and pygam, which lingam
# imports, pins scipy below 1.17, which bench-env (scipy 1.18 on Python 3.13) cannot satisfy; scipy
# 1.13.1 installs on Python 3.12. causal-learn's CAM-UV fits its smooth functions with pygam too,
# so causal-learn is installed here beside lingam.
#
# Nothing else goes in: the child runs `python -m andrey_bench._worker`, which imports numpy and the
# adapter module and nothing more. andrey, pandas and pyarrow stay in bench-env, where the parent
# scores the results. The final check fits each estimator the adapters call on a small sample.

set -euo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$BENCH/.venv-lingam"

echo ">> lingam-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" "lingam==1.13.0" "pygam==0.12.0" "causal-learn==0.1.4.8"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import lingam
import numpy as np
from causallearn.search.FCMBased.lingam import CAMUV

rng = np.random.default_rng(0)
groups = [rng.uniform(-1, 1, (200, 3)) for _ in range(2)]
assert len(lingam.MultiGroupDirectLiNGAM().fit(groups).adjacency_matrices_) == 2
x = rng.uniform(-2, 2, 300)
pair = np.column_stack([x, np.sin(2 * x) + 0.3 * rng.uniform(-1, 1, 300)])
assert lingam.CAMUV(alpha=0.01, num_explanatory_vals=3).fit(pair).adjacency_matrix_[1, 0] == 1
assert CAMUV.execute(pair, 0.01, 3)[0][1] == {0}
print(
    f"lingam-env OK lingam {m.version('lingam')} pygam {m.version('pygam')} "
    f"causal-learn {m.version('causal-learn')} numpy {np.__version__} scipy {m.version('scipy')}"
)
PY

echo ">> done"
