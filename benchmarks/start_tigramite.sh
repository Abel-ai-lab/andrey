#!/usr/bin/env bash
# Builds the tigramite measurement environment. No arguments.
#
#   benchmarks/start_tigramite.sh
#   export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
#
# Separate from .venv-bench so tigramite (GPL-3.0) and the dependencies its CMIknn and GPDC tests
# need (scikit-learn for the Gaussian processes, dcor for the distance correlation) stay out of
# bench-env. The child runs `python -m andrey_bench._worker`, which imports numpy and the adapter
# module; andrey, pandas and pyarrow stay in bench-env, where the parent scores the results.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-tigramite"

echo ">> tigramite-env"
uv venv "$VENV" --python 3.13 --clear
uv pip install --python "$VENV/bin/python" "tigramite==5.2.10.1" scikit-learn dcor numba pytest

"$VENV/bin/python" - <<'PY'
import numpy as np
import tigramite
from tigramite import data_processing
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.pcmci import PCMCI

rng = np.random.default_rng(0)
x = rng.standard_normal(150)
data = np.column_stack([x, np.abs(x) + 0.3 * rng.standard_normal(150), rng.standard_normal(150)])
cmiknn = CMIknn(knn=0.1, sig_samples=100, sig_blocklength=1, workers=1)
for test in (cmiknn, GPDC(significance="analytic")):
    pcmci = PCMCI(dataframe=data_processing.DataFrame(data), cond_ind_test=test, verbosity=0)
    graph = pcmci.run_pcalg_non_timeseries_data(pc_alpha=0.05, contemp_collider_rule="none")["graph"]
    assert graph.shape == (3, 3, 1), graph.shape
print(f"tigramite-env OK tigramite {tigramite.__version__} numpy {np.__version__}")
PY

echo ">> done"
