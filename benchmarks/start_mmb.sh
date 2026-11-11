#!/usr/bin/env bash
# Fetches the authors' MMB-by-MMB code and builds the environment it runs in, the reference of the
# MMB-by-MMB lane.
#
#   benchmarks/start_mmb.sh    # then: mmb_by_mmb_cpu.py, with the exports this script prints
#
# MMB-by-MMB is fengxie009/MMB-by-MMB at the commit below, a repository with no license file: the
# benchmark runs it unmodified as a measurement reference and redistributes none of it. The script
# downloads the files a fit imports one by one, checks each file's SHA-256, and writes them under
# `$ANDREY_BENCH_MMB_DIR` (default `benchmarks/.mmb`); the benchmark imports them from there and
# copies none of them. requirements-mmb.txt pins every package of the environment, `.venv-mmb`: the
# code imports NumPy and SciPy and nothing else. The environment imports none of Andrey; the driver
# scores the results in bench-env. Export what the script prints before running mmb_by_mmb_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_MMB_DIR:-$REPO/benchmarks/.mmb}"
VENV="$REPO/benchmarks/.venv-mmb"
LOCK="$REPO/benchmarks/requirements-mmb.txt"
COMMIT="2c63c890726fbe6f46e68850c4de52873810927f"
URL="https://raw.githubusercontent.com/fengxie009/MMB-by-MMB/$COMMIT"

echo ">> MMB-by-MMB $COMMIT"
mkdir -p "$DEST/Utils"
while read -r sha file; do
    if [[ ! -f "$DEST/$file" ]]; then
        curl -fsSL -o "$DEST/$file.part" "$URL/$file"
        mv "$DEST/$file.part" "$DEST/$file"
    fi
    echo "$sha  $DEST/$file" | sha256sum --check --status \
        || { echo "$file does not match $sha" >&2; exit 1; }
done <<'FILES'
e883c3a7f564a9927a5c665266d5564eaed4840091f682fc5fb0fe9cea3175f5 MMB_by_MMB.py
13e712e3e83d920037dddb9197a9d0c3ffbf76236bd16186f5d39da699a2f0aa Utils/CI_test.py
40fb29b16a0fadf06ade85e274a2d189916e4b6759294131327dbd5ff6769864 Utils/MMB_TC_Z.py
d54893829a521c98bfb1e8c0c18f6ccd618da90d82cea9873a5a2ea8250e2bd4 Utils/subsets1.py
595794d1069d8e647e9e1f68ef8f8acf658ff9378b960c00cc1048746c491a38 Utils/zhang_orient.py
FILES
echo "$COMMIT" > "$DEST/COMMIT"

echo ">> mmb-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" -r "$LOCK"
sha256sum "$LOCK"

# One fit on a small chain with a hidden common cause: x0 <- h -> x1 -> x2 and the target x1.
ANDREY_BENCH_MMB="$DEST" "$VENV/bin/python" - "$REPO/benchmarks" <<'PY'
import importlib.metadata as m
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.xie_mmb_by_mmb import XieMMBbyMMB
from andrey_bench.local_lane import unpack_answer

rng = np.random.default_rng(0)
n = 2000
h = rng.standard_normal(n)
x0 = 0.8 * h + rng.standard_normal(n)
x1 = 0.8 * h + rng.standard_normal(n)
x2 = 0.9 * x1 + rng.standard_normal(n)
adapter = XieMMBbyMMB()
marks, tests = unpack_answer(adapter.fit(np.column_stack([x0, x1, x2]), {**adapter.params(), "target": 1}))
print(f"mmb-env OK numpy {np.__version__} scipy {m.version('scipy')} | {tests} tests | marks at the target {marks[1].tolist()} {marks[:, 1].tolist()}")
PY

printf 'export ANDREY_BENCH_MMB_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_MMB=%q\n' "$DEST"
echo ">> done"
