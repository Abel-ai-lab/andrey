#!/usr/bin/env bash
# Builds the XGES measurement environment. No arguments.
#
#   benchmarks/start_xges.sh
#   export ANDREY_BENCH_XGES_PYTHON=$PWD/benchmarks/.venv-xges/bin/python
#
# Separate from .venv-bench because xges pins numpy<2 and numba<0.60, and bench-env is numpy 2 on
# Python 3.13. Python 3.12 is the last release numba<0.60 builds for.
#
# scipy is installed although xges does not declare it: xges/bic_scorer_fast.py imports
# scipy.special.gammaln, and XGES catches the resulting ImportError and falls back to BICScorer
# with only a log warning. The final check verifies that use_fast_numba=True selects BICScorerFast.
#
# Nothing else goes in: the child runs `python -m andrey_bench._worker`, which imports numpy and the
# adapter module and nothing more. andrey, torch, pandas and pyarrow stay in bench-env, where the
# parent scores the results.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-xges"

echo ">> xges-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" "xges==0.1.6" scipy pytest

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import numpy as np
import numba
import xges
from xges.bic_scorer_fast import BICScorerFast

data = np.random.default_rng(0).normal(size=(200, 5))
model = xges.XGES(alpha=1.0)
model.fit(data, extended_search=True, use_fast_numba=True, verbose=0)
assert isinstance(model.scorer, BICScorerFast), f"fell back to {type(model.scorer).__name__}"
print(
    f"xges-env OK xges {xges.__version__} numpy {np.__version__} numba {numba.__version__} "
    f"scipy {m.version('scipy')} | fast scorer {type(model.scorer).__name__}"
)
PY

echo ">> done"
