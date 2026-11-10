#!/usr/bin/env bash
# Fetches the authors' NOTEARS code and builds the environment of the NOTEARS and DAGMA references.
#
#   benchmarks/start_notears.sh    # NOTEARS, DAGMA, and the sortability controls of notears_cpu.py
#
# NOTEARS is xunzheng/notears (Apache-2.0) at the commit below, which has no package release. The
# script downloads `notears/linear.py` and the license, checks each file's SHA-256, and writes them
# under `$ANDREY_BENCH_NOTEARS_DIR` (default `benchmarks/.notears`); the benchmark loads the file
# from there and copies none of it. DAGMA is the authors' package `dagma` 1.1.1 (Apache-2.0), and
# the sortability controls are CausalDisco 0.3.0 (BSD-3-Clause). requirements-notears.txt pins every
# package of the environment, `.venv-notears`, and installs without dependency resolution: `dagma`
# declares PyTorch, which its linear model does not import, so PyTorch stays out. The environment
# imports none of Andrey; the driver scores the results in bench-env. Export what it prints before
# running notears_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_NOTEARS_DIR:-$REPO/benchmarks/.notears}"
VENV="$REPO/benchmarks/.venv-notears"
LOCK="$REPO/benchmarks/requirements-notears.txt"
COMMIT="4a9ab19fe502e392503da331773c5223d82d3666"
URL="https://raw.githubusercontent.com/xunzheng/notears/$COMMIT"

echo ">> NOTEARS $COMMIT"
mkdir -p "$DEST"
while read -r sha source file; do
    if [[ ! -f "$DEST/$file" ]]; then
        curl -fsSL -o "$DEST/$file.part" "$URL/$source"
        mv "$DEST/$file.part" "$DEST/$file"
    fi
    echo "$sha  $DEST/$file" | sha256sum --check --status \
        || { echo "$file does not match $sha" >&2; exit 1; }
done <<'FILES'
39644acd5b5d5ebc5fb5be5783365137eaa3b309df73bc0a75bad1942a0a426f notears/linear.py linear.py
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 LICENSE LICENSE
FILES
echo "$COMMIT" > "$DEST/COMMIT"

echo ">> notears-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" --no-deps -r "$LOCK"
sha256sum "$LOCK"

# One fit of each reference and control on a small chain.
ANDREY_BENCH_NOTEARS="$DEST" "$VENV/bin/python" - "$REPO/benchmarks" <<'PY'
import importlib.metadata as m
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.bello_dagma import BelloDAGMA
from andrey_bench.adapters.causaldisco_r2sort import (
    CausalDiscoR2SortnRegress,
    CausalDiscoVarSortnRegress,
)
from andrey_bench.adapters.zheng_notears import ZhengNOTEARS

rng = np.random.default_rng(0)
x = rng.standard_normal((500, 3))
x[:, 1] += 1.5 * x[:, 0]
x[:, 2] += 1.5 * x[:, 1]
edges = {}
for adapter in (
    ZhengNOTEARS(),
    BelloDAGMA(),
    CausalDiscoR2SortnRegress(),
    CausalDiscoVarSortnRegress(),
):
    if hasattr(adapter, "setup"):
        adapter.setup()
    adjacency = adapter.fit(x, adapter.params())
    edges[adapter.name] = sorted((int(i), int(j)) for i, j in zip(*np.nonzero(adjacency)))
assert edges["zheng.notears"] == edges["bello.dagma"] == [(0, 1), (1, 2)], edges
print(
    f"notears-env OK numpy {np.__version__} scipy {m.version('scipy')} dagma "
    f"{m.version('dagma')} CausalDisco {m.version('causaldisco')} | {edges}"
)
PY

printf 'export ANDREY_BENCH_NOTEARS_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_NOTEARS=%q\n' "$DEST"
echo ">> done"
