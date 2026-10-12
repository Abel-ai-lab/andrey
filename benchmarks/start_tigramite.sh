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
# PCMCI` fails with ModuleNotFoundError without it. The final check runs PCMCI with ParCorr on a
# planted lag-1 link.
#
# Nothing else goes in: the child runs `python -m andrey_bench._worker`, which imports numpy and the
# adapter module and nothing more. andrey, pandas and pyarrow stay in bench-env, where the parent
# scores the results.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-tigramite"

echo ">> tigramite-env"
uv venv "$VENV" --python 3.13 --clear
uv pip install --python "$VENV/bin/python" "tigramite==5.2.10.1" numpy scipy joblib

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import numpy as np
from tigramite import data_processing
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
results = pcmci.run_pcmci(tau_min=1, tau_max=2, pc_alpha=0.2, alpha_level=0.05, fdr_method="none")
assert results["graph"].shape == (3, 3, 3), results["graph"].shape
assert results["graph"][0, 1, 1] == "-->", "the planted link X0(t-1) -> X1(t) was not found"
print(
    f"tigramite-env OK tigramite {m.version('tigramite')} numpy {np.__version__} "
    f"scipy {m.version('scipy')}"
)
PY

echo ">> done"
