#!/usr/bin/env bash
# Builds the environment lingam's VARMALiNGAM and LongitudinalLiNGAM fits run in. No arguments.
#
#   benchmarks/start_lingam_ts.sh
#   export ANDREY_BENCH_LINGAM_TS_PYTHON=$PWD/benchmarks/.venv-lingam-ts/bin/python
#
# lingam 1.13.0, the release that ships both estimators, installs on Python 3.12 but not on the
# Python of .venv-bench, so its fits run here. The child runs `python -m andrey_bench._worker`,
# which imports numpy and the adapter module; andrey stays in bench-env, where the parent scores the
# results. The final check fits both estimators on a small series.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-lingam-ts"

echo ">> lingam-ts-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" "lingam==1.13.0" numpy scipy

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import lingam
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, (300, 3))
x[1:, 1] += 0.6 * x[:-1, 0]
model = lingam.VARMALiNGAM(order=(1, 1), criterion=None, prune=True).fit(x)
assert len(model.adjacency_matrices_) == 2
panel = [rng.uniform(-1, 1, (200, 3)) for _ in range(3)]
assert lingam.LongitudinalLiNGAM(n_lags=1).fit(panel).adjacency_matrices_.shape == (3, 2, 3, 3)
print(f"lingam-ts-env OK lingam {m.version('lingam')} numpy {np.__version__}")
PY

echo ">> done"
