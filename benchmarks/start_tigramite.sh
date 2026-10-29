#!/usr/bin/env bash
# Builds the tigramite measurement environment. No arguments.
#
#   benchmarks/start_tigramite.sh
#   export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
#
# Separate from .venv-bench: tigramite is GPL-licensed, so it is installed only here, its fits run
# here, and bench-env compares their outputs. tigramite is pinned to the release the PCMCI lane's
# agreement checks were made against.
#
# joblib is installed although tigramite lists it only under an extra: `from tigramite.pcmci import
# PCMCI` fails with ModuleNotFoundError without it. scikit-learn (the Gaussian processes) and dcor
# (the distance correlation) are the dependencies of its GPDC test, numba of its CMIknn test. The
# final checks run PCMCI with ParCorr on a planted lag-1 link, and PC with CMIknn and GPDC.
#
# Nothing else goes in: the child runs `python -m andrey_bench._worker`, which imports numpy and the
# adapter module and nothing more. andrey, pandas and pyarrow stay in bench-env, where the parent
# scores the results.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-tigramite"

echo ">> tigramite-env"
uv venv "$VENV" --python 3.13 --clear
uv pip install --python "$VENV/bin/python" "tigramite==5.2.10.1" numpy scipy joblib \
    scikit-learn dcor numba

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import numpy as np
from tigramite import data_processing
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

rng = np.random.default_rng(0)
data = rng.standard_normal((300, 3))
data[1:, 1] += 0.8 * data[:-1, 0]
pcmci = PCMCI(
    dataframe=data_processing.DataFrame(data),
    cond_ind_test=ParCorr(significance="analytic"),
    verbosity=0,
)
results = pcmci.run_pcmci(tau_min=1, tau_max=2, pc_alpha=0.2, alpha_level=0.05, fdr_method="fdr_bh")
assert results["graph"].shape == (3, 3, 3), results["graph"].shape
assert results["graph"][0, 1, 1] == "-->", "the planted link X0(t-1) -> X1(t) was not found"

x = rng.standard_normal(150)
data = np.column_stack([x, np.abs(x) + 0.3 * rng.standard_normal(150), rng.standard_normal(150)])
cmiknn = CMIknn(knn=0.1, sig_samples=100, sig_blocklength=1, workers=1)
for test in (cmiknn, GPDC(significance="analytic")):
    pcmci = PCMCI(dataframe=data_processing.DataFrame(data), cond_ind_test=test, verbosity=0)
    graph = pcmci.run_pcalg_non_timeseries_data(pc_alpha=0.05, contemp_collider_rule="none")["graph"]
    assert graph.shape == (3, 3, 1), graph.shape
print(
    f"tigramite-env OK tigramite {m.version('tigramite')} numpy {np.__version__} "
    f"scipy {m.version('scipy')}"
)
PY

echo ">> done"
