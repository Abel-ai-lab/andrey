#!/usr/bin/env bash
# Builds the UT-IGSP reference environment. No arguments.
#
#   benchmarks/start_utigsp.sh
#   export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
#   export ANDREY_BENCH_GML_DEF42_PYTHON=$PWD/benchmarks/.venv-gml-def42/bin/python
#
# The reference is graphical-model-learning 0.1a8 with conditional-independence 0.1a6 and
# graphical-models 0.1a21, the authors' library; requirements-utigsp.txt pins all 73 packages of
# the environment it was measured in. Some of them (pgmpy, xgboost, the NCCL wheel) are
# dependencies the reference never imports; they stay here, outside Andrey's environment. A child
# in this environment imports NumPy and the reference packages and nothing more; andrey stays in
# bench-env, where the parent scores the results. The check fits UT-IGSP, IGSP, the lane's JCI-GSP
# copy (andrey_bench/adapters/gml_interventional.py), and UT-IGSP with the kernel tests.
#
# The IGSP lane also builds .venv-gml-def42: the same packages, with
# patches/gml-igsp-def42.patch applied to the installed graphical-model-learning, so that IGSP's
# I-contradicting test follows Definition 4.2 of Yang, Katcoff, and Uhler (2018) when a setting has
# two or more targets (comparisons/igsp.md).

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
VENV="$REPO/benchmarks/.venv-utigsp"
LOCK="$REPO/benchmarks/requirements-utigsp.txt"

echo ">> utigsp-env"
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" -r "$LOCK"
sha256sum "$LOCK"

"$VENV/bin/python" - "$REPO/benchmarks/andrey_bench/adapters" <<'PY'
import importlib.metadata as m
import random
import sys

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

sys.path.insert(0, sys.argv[1])
from gml_interventional import run_fit  # noqa: E402

X = np.vstack([obs, *ivs])
env = np.repeat([0, 1, 2], 2000)
known = [[0], []]
fit = {"alpha": 1e-5, "nruns": 5, "tester": "gauss"}
jci = run_fit({**fit, "algorithm": "jci_gsp"}, X, env, known)
rows = np.concatenate([np.flatnonzero(env == k)[:200] for k in range(3)])
kci = run_fit({**fit, "algorithm": "ut_igsp", "tester": "kci"}, X[rows], env[rows], known)
print(
    f"utigsp-env OK graphical-model-learning {m.version('graphical-model-learning')} "
    f"conditional-independence {m.version('conditional-independence')} graphical-models "
    f"{m.version('graphical-models')} numpy {np.__version__} | UT-IGSP {len(found.arcs)} arcs, "
    f"targets {[sorted(int(v) for v in t) for t in targets]}; IGSP {len(igsp.arcs)} arcs; "
    f"JCI-GSP {len(jci['arcs'])} arcs, targets {jci['targets']}; UT-IGSP with kernel tests "
    f"{len(kci['arcs'])} arcs"
)
PY

echo ">> gml-def42-env"
DEF42="$REPO/benchmarks/.venv-gml-def42"
uv venv "$DEF42" --python 3.11 --clear
# Copies, not links, so the patch reaches neither uv's cache nor .venv-utigsp.
uv pip install --python "$DEF42/bin/python" --link-mode copy -r "$LOCK"
SITE=$("$DEF42/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
patch -p1 -N -s -d "$SITE" < "$REPO/benchmarks/patches/gml-igsp-def42.patch"
# The package exports a function named gsp, which hides the module of that name.
"$DEF42/bin/python" -c 'import importlib, inspect; \
g = importlib.import_module("graphical_model_learning.algorithms.dag.gsp"); \
assert "j_not_i" in inspect.getsource(g.igsp); print("gml-def42-env OK: Definition 4.2 patch applied")'

echo ">> done"
