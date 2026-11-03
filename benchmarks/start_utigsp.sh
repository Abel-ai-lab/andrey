#!/usr/bin/env bash
# Builds the UT-IGSP reference environment. No arguments.
#
#   benchmarks/start_utigsp.sh
#   export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
#
# The reference is graphical-model-learning 0.1a8 with conditional-independence 0.1a6 and
# graphical-models 0.1a21, the authors' library; requirements-utigsp.txt pins all 73 packages of
# the environment it was measured in. Some of them (pgmpy, xgboost, the NCCL wheel) are
# dependencies the reference never imports; they stay here, outside Andrey's environment. A child
# in this environment imports NumPy and the reference packages and nothing more; andrey stays in
# bench-env, where the parent scores the results.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-utigsp"
LOCK="$REPO/benchmarks/requirements-utigsp.txt"

echo ">> utigsp-env"
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" -r "$LOCK"
sha256sum "$LOCK"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
import random

import graphical_model_learning as gml
import graphical_models as gm
import numpy as np
from conditional_independence import (
    MemoizedCI_Tester,
    MemoizedInvarianceTester,
    gauss_invariance_suffstat,
    gauss_invariance_test,
    partial_correlation_suffstat,
    partial_correlation_test,
)

random.seed(0)
np.random.seed(0)
dag = gm.rand.rand_weights(gm.rand.directed_erdos(5, exp_nbrs=1.5))
obs = dag.sample(2000)
ivs = [dag.sample_interventional({k: gm.ShiftIntervention(2)}, 2000) for k in (0, 2)]
ci = MemoizedCI_Tester(partial_correlation_test, partial_correlation_suffstat(obs), alpha=1e-5)
inv = MemoizedInvarianceTester(
    gauss_invariance_test, gauss_invariance_suffstat(obs, ivs), alpha=1e-5
)
found, targets = gml.unknown_target_igsp(
    [{"known_interventions": [0]}, {"known_interventions": []}], set(range(5)), ci, inv
)
igsp = gml.igsp([{"interventions": [0]}, {"interventions": [2]}], set(range(5)), ci, inv)
print(
    f"utigsp-env OK graphical-model-learning {m.version('graphical-model-learning')} "
    f"conditional-independence {m.version('conditional-independence')} graphical-models "
    f"{m.version('graphical-models')} numpy {np.__version__} | UT-IGSP {len(found.arcs)} arcs, "
    f"targets {[sorted(int(v) for v in t) for t in targets]}; IGSP {len(igsp.arcs)} arcs"
)
PY

echo ">> done"
