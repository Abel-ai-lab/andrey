#!/usr/bin/env bash
# Fetches MissDAG's code, the reference for `missing="em"`, and builds the environment it runs in.
#
#   benchmarks/start_missdag.sh
#   export ANDREY_BENCH_MISSDAG=$PWD/benchmarks/.missdag/MissDAG-<commit>
#   export ANDREY_BENCH_MISSDAG_PYTHON=$PWD/benchmarks/.venv-missdag/bin/python
#
# The code accompanies Gao, Ng, Gong, Shen, Huang, Liu, Zhang, and Bondell (2022). Its repository
# has no license file, so the benchmark runs it unmodified as a comparator: the adapter imports the
# files from the directory this script prints and copies none of them. The archive of the pinned
# commit is checked against its SHA-256 and extracted under `$ANDREY_BENCH_MISSDAG_DIR` (default
# `benchmarks/.missdag`).
#
# The linear-Gaussian path (`miss_methods/miss_dag_gaussian.py`, `miss_methods/em_utils.py`,
# `dag_methods/notears/notears.py`, and `postprocess` in `utils/utils.py`) needs NumPy, SciPy,
# pandas, and NetworkX, pinned here to the versions of the repository's `environment.yml` on the
# oldest Python their wheels support. torch and TensorFlow serve only the nonlinear paths and
# `utils.utils.set_seed`, so they stay out (see `andrey_bench/adapters/missdag.py`).

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_MISSDAG_DIR:-$REPO/benchmarks/.missdag}"
VENV="$REPO/benchmarks/.venv-missdag"
COMMIT="3c1eeed8f6245670614d2f2938ee3c3b5a21f17b"
URL="https://codeload.github.com/ErdunGAO/MissDAG/tar.gz/$COMMIT"
SHA256="a72d1173339d1f15a4bea8325ae1bba1371a65ff388cd1796cb9710da568646f"
CODE="$DEST/MissDAG-$COMMIT"

mkdir -p "$DEST"
if [[ ! -f "$CODE/miss_methods/miss_dag_gaussian.py" ]]; then
    curl -fsSL -o "$DEST/missdag.tar.gz" "$URL"
    echo "$SHA256  $DEST/missdag.tar.gz" | sha256sum --check --status \
        || { echo "missdag.tar.gz does not match $SHA256" >&2; exit 1; }
    tar -xzf "$DEST/missdag.tar.gz" -C "$DEST"
    echo "$SHA256" > "$CODE/ARCHIVE_SHA256"
fi

echo ">> missdag-env"
uv venv "$VENV" --python 3.10 --clear
uv pip install --python "$VENV/bin/python" \
    "numpy==1.21.5" "scipy==1.7.3" "pandas==1.3.5" "networkx==2.6.3"

ANDREY_BENCH_MISSDAG="$CODE" PYTHONPATH="$REPO/benchmarks" "$VENV/bin/python" - <<'PY'
import importlib.metadata as m

import numpy as np

from andrey_bench.adapters.missdag import MissDagGaussian

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 4)) @ np.triu(rng.uniform(0.5, 2.0, size=(4, 4)), 1) + rng.normal(size=(100, 4))
X[rng.random(X.shape) < 0.1] = np.nan
solution = MissDagGaussian()
solution.setup()
adj = solution.fit(X, {**solution.params(), "em_iter": 2})
print(
    f"missdag-env OK edges {int(adj.sum())} numpy {np.__version__} scipy {m.version('scipy')} "
    f"pandas {m.version('pandas')} networkx {m.version('networkx')}"
)
PY
printf 'export ANDREY_BENCH_MISSDAG=%q\n' "$CODE"
printf 'export ANDREY_BENCH_MISSDAG_PYTHON=%q\n' "$VENV/bin/python"
