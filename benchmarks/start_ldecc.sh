#!/usr/bin/env bash
# Fetches the authors' LDECC code and builds the environment it runs in, the reference of the
# adjustment lane's LDECC solution.
#
#   benchmarks/start_ldecc.sh    # then: adjustment_cpu.py, with the exports this script prints
#
# LDECC is acmi-lab/local-causal-discovery by Gupta, Childers, and Lipton at the commit below, a
# repository with no license file: the benchmark runs it unmodified as a measurement reference and
# redistributes none of it. The script downloads the archive of the pinned commit, checks its
# SHA-256, and extracts it under `$ANDREY_BENCH_LDECC_DIR` (default `benchmarks/.ldecc`); the
# benchmark imports the code from there and copies none of it. requirements-ldecc.txt pins every
# package of the environment, `.venv-ldecc`: the code imports NumPy, pandas, SciPy, scikit-learn,
# matplotlib, networkx, and causal-learn's `CIT` for Fisher's z test, and nothing else, so the
# packages causal-learn lists but never imports on that path (graphviz, pydot, statsmodels, tqdm)
# are left out. The environment imports none of Andrey; the driver scores the records in
# bench-env. Export what the script prints before running adjustment_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_LDECC_DIR:-$REPO/benchmarks/.ldecc}"
VENV="$REPO/benchmarks/.venv-ldecc"
LOCK="$REPO/benchmarks/requirements-ldecc.txt"
COMMIT="605d8d1e82d0cd63dce717a5618baa0b5fa153fb"
URL="https://codeload.github.com/acmi-lab/local-causal-discovery/tar.gz/$COMMIT"
SHA256="1c80b8c175dca44bd2ac1f86eaf22e46205ad79e62e56f628725a818d189c5b8"
CODE="$DEST/local-causal-discovery-$COMMIT"

echo ">> LDECC $COMMIT"
mkdir -p "$DEST"
if [[ ! -f "$CODE/causal_discovery/ldecc.py" ]]; then
    curl -fsSL --retry 3 -o "$DEST/ldecc.tar.gz" "$URL"
    echo "$SHA256  $DEST/ldecc.tar.gz" | sha256sum --check --status \
        || { echo "ldecc.tar.gz does not match $SHA256" >&2; exit 1; }
    tar -xzf "$DEST/ldecc.tar.gz" -C "$DEST"
    echo "$SHA256" > "$CODE/ARCHIVE_SHA256"
fi

echo ">> ldecc-env"
uv venv "$VENV" --python 3.10 --clear
uv pip install --python "$VENV/bin/python" --no-deps -r "$LOCK"
sha256sum "$LOCK"

# One fit on a collider with an outcome below it: a -> x <- b and x -> y, the treatment x. The
# treatment's neighbors are fully oriented, so the one valid parent set is {a, b}.
ANDREY_BENCH_LDECC="$CODE" "$VENV/bin/python" -I - "$REPO/benchmarks" <<'PY'
import importlib.metadata as m
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.gupta_ldecc import GuptaLDECC
from andrey_bench.adjustment_lane import unpack_effects

rng = np.random.default_rng(0)
n = 4000
a, b = rng.standard_normal(n), rng.standard_normal(n)
x = 0.9 * a + 0.9 * b + rng.standard_normal(n)
y = 0.9 * x + rng.standard_normal(n)
adapter = GuptaLDECC()
adapter.setup()
data = np.column_stack([a, x, b, y])
native = adapter.fit(data, {**adapter.params(), "treatment": 1, "outcome": 3})
[record], tests = unpack_effects(native)
assert record["adjustment_sets"] == [[0, 2]] and record["identifiable"] is True, record
versions = " ".join(f"{p} {m.version(p)}" for p in ("numpy", "scipy", "pandas", "causal-learn"))
print(f"ldecc-env OK {versions} | {tests} tests | sets {record['adjustment_sets']}")
PY

printf 'export ANDREY_BENCH_LDECC_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_LDECC=%q\n' "$CODE"
echo ">> done"
