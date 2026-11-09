#!/usr/bin/env bash
# Fetches the authors' LOCI code and builds the environment it runs in, the reference of the LOCI lane.
#
#   benchmarks/start_loci.sh    # then: loci_pairs.py, with the exports this script prints
#
# LOCI is aleximmer/loci at the commit below, a repository with no license file: the benchmark runs
# it unmodified as a measurement reference and redistributes none of it. The script downloads the
# files a fit imports one by one, checks each file's SHA-256, and writes them under
# `$ANDREY_BENCH_LOCI_DIR` (default `benchmarks/.loci`); the benchmark imports them from there and
# copies none of them. requirements-loci.txt pins every package of the environment, `.venv-loci`:
# the code imports torch, scikit-learn, matplotlib, and tueplots at the top of its modules, and its
# own requirements ask for NumPy 1. The environment imports none of Andrey; the driver scores the
# results in bench-env. Export what it prints before running loci_pairs.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_LOCI_DIR:-$REPO/benchmarks/.loci}"
VENV="$REPO/benchmarks/.venv-loci"
LOCK="$REPO/benchmarks/requirements-loci.txt"
COMMIT="da61b8cdf95cbbc00fc42d351f89e64410f535f4"
URL="https://raw.githubusercontent.com/aleximmer/loci/$COMMIT"

echo ">> LOCI $COMMIT"
mkdir -p "$DEST/causa"
while read -r sha file; do
    if [[ ! -f "$DEST/$file" ]]; then
        curl -fsSL -o "$DEST/$file.part" "$URL/$file"
        mv "$DEST/$file.part" "$DEST/$file"
    fi
    echo "$sha  $DEST/$file" | sha256sum --check --status \
        || { echo "$file does not match $sha" >&2; exit 1; }
done <<'FILES'
c831b20dde5af1ebbe9e207bd493ee509651c462e806c1611cd6ee06eaf60ddd causa/__init__.py
1e4691e5e7c92469fe91a8ee48753a28807baaa347009b857f7aff332c14ab2c causa/loci.py
f63fade89ccb011d7d1e148391d21accf63a434b2d7532e9f1fd7bf38454ae07 causa/hsic.py
c986c7ea4ab3e14e4a692e86c1718e75d6fab1f659fd352c0e737f62f7049c2e causa/het_ridge.py
adbda5149955844dbd06823b6043b1e09fcd6c07496cbbaa3bec20206b46195f causa/ml.py
91c8323fd74fced5a7a2670d3e0c99fa2bed49591b702f728190a85570ceb155 causa/utils.py
FILES
echo "$COMMIT" > "$DEST/COMMIT"

echo ">> loci-env"
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r "$LOCK"
sha256sum "$LOCK"

# One fit of each criterion on a small pair.
MPLBACKEND=Agg ANDREY_BENCH_LOCI="$DEST" "$VENV/bin/python" - "$REPO/benchmarks" <<'PY'
import importlib.metadata as m
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.immer_loci import ImmerLOCI, ImmerLOCILikelihood

rng = np.random.default_rng(0)
x = rng.standard_normal(300)
pair = np.column_stack([x, np.sin(2 * x) + x + (0.6 + 0.5 * np.tanh(x)) * rng.standard_normal(300)])
edges = {}
for adapter in (ImmerLOCI(), ImmerLOCILikelihood()):
    marks = adapter.fit(pair, adapter.params())
    edges[adapter.name] = [(int(i), int(j)) for i, j in zip(*np.nonzero(marks == 1))]
print(
    f"loci-env OK numpy {np.__version__} torch {m.version('torch')} scikit-learn "
    f"{m.version('scikit-learn')} | {edges}"
)
PY

printf 'export ANDREY_BENCH_LOCI_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_LOCI=%q\n' "$DEST"
echo ">> done"
