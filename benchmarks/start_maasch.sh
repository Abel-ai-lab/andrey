#!/usr/bin/env bash
# Fetches the authors' LDP and LD3 code and builds the one environment both run in, the references
# of the adjustment lane's LDP and LD3 solutions.
#
#   benchmarks/start_maasch.sh    # then: adjustment_cpu.py, with the exports this script prints
#
# LDP is jmaasch/ldp (Maasch et al., 2024) and LD3 is jmaasch/LD3 (Maasch et al., 2025), each at
# the commit below, repositories with no license file: the benchmark runs them unmodified as
# measurement references and redistributes none of them. The script downloads the archive of each
# pinned commit, checks its SHA-256, and extracts both under `$ANDREY_BENCH_MAASCH_DIR` (default
# `benchmarks/.maasch`); the benchmark imports the code from there and copies none of it. The two
# modules, `ldp.py` and `ld3.py`, import the same packages (NumPy, pandas, SciPy, scikit-learn,
# matplotlib, seaborn, networkx, and causal-learn's `CIT`), so requirements-maasch.txt pins one
# environment, `.venv-maasch`, for both; it leaves out the packages causal-learn lists but the
# `CIT` import never reaches (graphviz, pydot, statsmodels, tqdm) and the ones only the authors'
# helper script and notebooks import (IPython, pydot). The environment imports none of Andrey; the
# driver scores the records in bench-env. Export what the script prints before running
# adjustment_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_MAASCH_DIR:-$REPO/benchmarks/.maasch}"
VENV="$REPO/benchmarks/.venv-maasch"
LOCK="$REPO/benchmarks/requirements-maasch.txt"
LDP_COMMIT="838f6e26b669ab049b797f0e2eda1f31faa316fa"
LDP_SHA256="e4a6e16f5a2b8c13fb71d3d8b68af0fcf425a716f05dfb97936447472346ac76"
LD3_COMMIT="74b91cb9ae13e05d26c23095410ee908616256e8"
LD3_SHA256="cfb6f24e2f05864ef48bcf0b3471b67df840276fa9b000b8d5118bc7824ebaf7"
LDP_CODE="$DEST/ldp-$LDP_COMMIT"
LD3_CODE="$DEST/LD3-$LD3_COMMIT"

mkdir -p "$DEST"
fetch() {  # repository, commit, sha256, extracted directory, a file the directory must hold
    if [[ ! -f "$4/$5" ]]; then
        curl -fsSL --retry 3 -o "$DEST/$1.tar.gz" "https://codeload.github.com/jmaasch/$1/tar.gz/$2"
        echo "$3  $DEST/$1.tar.gz" | sha256sum --check --status \
            || { echo "$1.tar.gz does not match $3" >&2; exit 1; }
        tar -xzf "$DEST/$1.tar.gz" -C "$DEST"
        echo "$3" > "$4/ARCHIVE_SHA256"
    fi
}
echo ">> LDP $LDP_COMMIT"
fetch ldp "$LDP_COMMIT" "$LDP_SHA256" "$LDP_CODE" ldp.py
echo ">> LD3 $LD3_COMMIT"
fetch LD3 "$LD3_COMMIT" "$LD3_SHA256" "$LD3_CODE" ld3.py

echo ">> maasch-env"
uv venv "$VENV" --python 3.10 --clear
uv pip install --python "$VENV/bin/python" --no-deps -r "$LOCK"
sha256sum "$LOCK"

# One fit of each on a graph with a confounder z1 (z1 -> x, z1 -> y), a cause of the outcome z4, an
# instrument z5, an isolated variable z8, and the effect x -> y: LDP finds the valid set {z1}; LD3
# finds the direct-effect criterion and the outcome's parents {z1, z4}.
ANDREY_BENCH_LDP="$LDP_CODE" ANDREY_BENCH_LD3="$LD3_CODE" \
    "$VENV/bin/python" -I - "$REPO/benchmarks" <<'PY'
import importlib.metadata as m
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.maasch_ldp import MaaschLD3, MaaschLDP
from andrey_bench.adjustment_lane import unpack_effects

rng = np.random.default_rng(0)
n = 4000
z1, z4, z5, z8 = (rng.standard_normal(n) for _ in range(4))
x = 0.9 * z1 + 0.9 * z5 + rng.standard_normal(n)
y = 0.9 * z1 + 0.9 * z4 + 0.9 * x + rng.standard_normal(n)
data = np.column_stack([z1, x, y, z4, z5, z8])
found = {}
for adapter in (MaaschLDP(), MaaschLD3()):
    adapter.setup()
    native = adapter.fit(data, {**adapter.params(), "treatment": 1, "outcome": 2})
    [record], tests = unpack_effects(native)
    found[adapter.name] = (record, tests)
ldp, ld3 = found["ldp.authors"][0], found["ld3.authors"][0]
assert ldp["identifiable"] is True and ldp["adjustment_sets"] == [[0]], ldp
assert ld3["nonzero"] is True and ld3["adjustment_sets"] == [[0, 3]], ld3
versions = " ".join(f"{p} {m.version(p)}" for p in ("numpy", "scipy", "pandas", "causal-learn"))
tests = {name: count for name, (_, count) in found.items()}
sets = f"LDP {ldp['adjustment_sets']} LD3 {ld3['adjustment_sets']}"
print(f"maasch-env OK {versions} | tests {tests} | {sets}")
PY

printf 'export ANDREY_BENCH_MAASCH_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_LDP=%q\n' "$LDP_CODE"
printf 'export ANDREY_BENCH_LD3=%q\n' "$LD3_CODE"
echo ">> done"
