#!/usr/bin/env bash
# Fetches the authors' SNAP and LOAD code and builds the one environment both run in, the
# references of the adjustment lane's pair solutions.
#
#   benchmarks/start_schubert.sh    # then: adjustment_cpu.py, with the exports this script prints
#
# SNAP (Matyasch/snap) and LOAD (Matyasch/load) are Schubert, Claassen, and Magliacane's code at
# the commits below, both under the BSD 2-clause license and both NumPy only. The benchmark runs
# them unmodified as measurement references and copies none of them: the script downloads each
# commit as an archive, checks the archive's SHA-256, and extracts it under
# `$ANDREY_BENCH_SCHUBERT_DIR` (default `benchmarks/.schubert`); the adapters import the code from
# there. requirements-schubert.txt pins every package of the environment, `.venv-schubert`; the
# environment imports none of Andrey, and the driver reads the answers in bench-env. Export what the
# script prints before running adjustment_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_SCHUBERT_DIR:-$REPO/benchmarks/.schubert}"
VENV="$REPO/benchmarks/.venv-schubert"
LOCK="$REPO/benchmarks/requirements-schubert.txt"
SNAP_COMMIT="432ab5c969048817ff6ea87d5733a26319503059"
SNAP_SHA256="f34162d95d8d53d65734d679041b94756b5916c99e4fab3a66460f9e53ef8190"
LOAD_COMMIT="7febfc68d477b82b07e52aa50e454339f6e0ebcd"
LOAD_SHA256="c5eb4d623aff5b1122b5430678c356613f16532007ebfd9efa3cd1b95bf60ee4"

# fetch NAME COMMIT SHA256: the archive of the commit, checked and extracted to $DEST/NAME-COMMIT.
fetch() {
    local name="$1" commit="$2" sha="$3"
    local url="https://codeload.github.com/Matyasch/$name/tar.gz/$commit"
    local code="$DEST/$name-$commit"
    echo ">> $name $commit"
    if [[ ! -d "$code" ]]; then
        # HTTP/2 to codeload ended in a stream error once; HTTP/1.1 is the fallback.
        curl -fsSL --retry 3 -o "$DEST/$name.tar.gz" "$url" \
            || curl -fsSL --http1.1 -o "$DEST/$name.tar.gz" "$url"
        echo "$sha  $DEST/$name.tar.gz" | sha256sum --check --status \
            || { echo "$name.tar.gz does not match $sha" >&2; exit 1; }
        tar -xzf "$DEST/$name.tar.gz" -C "$DEST"
        echo "$sha" > "$code/ARCHIVE_SHA256"
    fi
}

mkdir -p "$DEST"
fetch snap "$SNAP_COMMIT" "$SNAP_SHA256"
fetch load "$LOAD_COMMIT" "$LOAD_SHA256"
SNAP_CODE="$DEST/snap-$SNAP_COMMIT"
LOAD_CODE="$DEST/load-$LOAD_COMMIT"

echo ">> schubert-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" -r "$LOCK"
sha256sum "$LOCK"

# One fit of each reference on a small planted case: d -> x <- c, c -> y, x -> m -> y, with the
# columns (d, c, x, m, y). The v-structure at x lets the class identify the effect of x on y, whose
# optimal adjustment set is {c}.
ANDREY_BENCH_SNAP="$SNAP_CODE" ANDREY_BENCH_LOAD="$LOAD_CODE" "$VENV/bin/python" - "$REPO/benchmarks" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.schubert_load import SchubertLOAD
from andrey_bench.adapters.schubert_snap import SchubertSNAP
from andrey_bench.adjustment_lane import VARIABLES_PARAM, unpack_effects
from andrey_bench.local_lane import unpack_answer

rng = np.random.default_rng(0)
n = 4000
d, c = rng.standard_normal(n), rng.standard_normal(n)
x = 0.9 * d + 0.9 * c + rng.standard_normal(n)
m_ = 0.9 * x + rng.standard_normal(n)
y = 0.9 * m_ + 0.9 * c + rng.standard_normal(n)
data = np.column_stack([d, c, x, m_, y])
pair = {VARIABLES_PARAM: [2, 4]}

snap = SchubertSNAP()
marks, tests = unpack_answer(snap.fit(data, {**snap.params(), **pair}))
edges = [(i, j) for i in range(5) for j in range(5) if marks[i, j] == 1 and marks[j, i] == 2]
lines = [(i, j) for i in range(5) for j in range(i) if marks[i, j] == 1 and marks[j, i] == 1]
print(f"snap {snap.package_version} | {tests} tests | directed {edges} undirected {lines}")

load = SchubertLOAD()
records, tests = unpack_effects(load.fit(data, {**load.params(), **pair}))
for r in records:
    print(
        f"load {load.package_version} | {tests} tests | {r['treatment']}->{r['outcome']} "
        f"nonzero {r['nonzero']} identifiable {r['identifiable']} {r['set_kind']} {r['adjustment_sets']}"
    )
print(f"schubert-env OK numpy {np.__version__} python {sys.version.split()[0]}")
PY

printf 'export ANDREY_BENCH_SCHUBERT_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_SNAP=%q\n' "$SNAP_CODE"
printf 'export ANDREY_BENCH_LOAD=%q\n' "$LOAD_CODE"
echo ">> done"
