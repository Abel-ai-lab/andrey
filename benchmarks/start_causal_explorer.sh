#!/usr/bin/env bash
# Fetches Causal Explorer, the authors' MATLAB toolbox of HITON-PC, HITON-MB, and MMMB, the reference
# of those rows of the set-learner lane.
#
#   benchmarks/start_causal_explorer.sh    # then: blanket_cpu.py, with the export this script prints
#
# Causal Explorer 1.5 is mensxmachina/CausalExplorer_1.5 at the commit below: MATLAB P-code under a
# non-commercial single-user license that grants no right to copy or redistribute it. The script
# clones the public repository at that commit into `$ANDREY_BENCH_CAUSAL_EXPLORER` (default
# `benchmarks/.causal-explorer`, which git ignores); the lane calls the toolbox from there through
# MATLAB, and this repository holds none of it. A fit needs `matlab` on the PATH (the P-files load
# in MATLAB R2021a to R2026a); with it there, the script ends with one call on a small dataset.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_CAUSAL_EXPLORER:-$REPO/benchmarks/.causal-explorer}"
COMMIT="205a31f65577339ffb9a465ecc9fef0d4b04ca76"
URL="https://github.com/mensxmachina/CausalExplorer_1.5.git"

echo ">> Causal Explorer $COMMIT"
if [[ ! -d "$DEST/.git" ]]; then
    git clone --quiet --no-checkout "$URL" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || { echo "not at $COMMIT" >&2; exit 1; }
[[ -f "$DEST/causal_explorer/Causal_Explorer.p" ]] || { echo "no Causal_Explorer.p" >&2; exit 1; }

if command -v matlab >/dev/null 2>&1; then
    # One fit on a collider, a -> c <- b, and c -> d: HITON-MB's blanket of a is {b, c}.
    ANDREY_BENCH_CAUSAL_EXPLORER="$DEST" "${ANDREY_BENCH_PYTHON:-python}" - "$REPO/benchmarks" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.causal_explorer_blanket import CausalExplorerBlanket
from andrey_bench.blanket_lane import unpack_answer

rng = np.random.default_rng(0)
a, b = rng.integers(0, 2, (2, 3000))
c = (a | b) ^ (rng.random(3000) < 0.1)
d = c ^ (rng.random(3000) < 0.1)
adapter = CausalExplorerBlanket("HITON_MB", indep_test="gsq")
answer, _, _ = unpack_answer(adapter.fit(np.column_stack([a, b, c, d]), {"targets": [0]}))
print(f"causal-explorer OK | HITON_MB blanket of a: {np.flatnonzero(answer[0]).tolist()}")
PY
else
    echo ">> matlab is not on the PATH: the toolbox is in place, and the lane leaves its arm out"
fi

printf 'export ANDREY_BENCH_CAUSAL_EXPLORER=%q\n' "$DEST"
echo ">> done"
