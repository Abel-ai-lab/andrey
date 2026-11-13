#!/usr/bin/env bash
# Fetches the authors' LocPC-CDE code and builds the environment it runs in, the reference of the
# adjustment lane's LocPC-CDE solution.
#
#   benchmarks/start_ciphod.sh    # then: adjustment_cpu.py, with the exports this script prints
#
# LocPC-CDE is in pyCIPHOD (CIPHOD/pyCIPHOD) by Assaad and coauthors at the commit below, under the
# MIT license. The benchmark runs it unmodified as a measurement reference and copies none of it:
# the script downloads the archive of the pinned commit, checks its SHA-256, and extracts it under
# `$ANDREY_BENCH_CIPHOD_DIR` (default `benchmarks/.ciphod`); the adapter imports the package from
# the `src` directory there, so nothing is installed. requirements-ciphod.txt pins every package of
# the environment, `.venv-ciphod`; the environment imports none of Andrey, and the driver reads the
# answers in bench-env. The code iterates sets of variable names, so its graph depends on the string
# hash seed: export what the script prints, `PYTHONHASHSEED` included, before running
# adjustment_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_CIPHOD_DIR:-$REPO/benchmarks/.ciphod}"
VENV="$REPO/benchmarks/.venv-ciphod"
LOCK="$REPO/benchmarks/requirements-ciphod.txt"
COMMIT="37a62c9f2052a793243a81b1a0b728eb33fbace1"
URL="https://codeload.github.com/CIPHOD/pyCIPHOD/tar.gz/$COMMIT"
SHA256="e1519c6c73dc18b15b1d54b6ad7eafebd4c9f12a49132998fea799faf76370ed"
CODE="$DEST/pyCIPHOD-$COMMIT"

echo ">> pyCIPHOD $COMMIT"
mkdir -p "$DEST"
if [[ ! -d "$CODE" ]]; then
    # HTTP/2 to codeload ended in a stream error once; HTTP/1.1 is the fallback.
    curl -fsSL --retry 3 -o "$DEST/pyCIPHOD.tar.gz" "$URL" \
        || curl -fsSL --http1.1 -o "$DEST/pyCIPHOD.tar.gz" "$URL"
    echo "$SHA256  $DEST/pyCIPHOD.tar.gz" | sha256sum --check --status \
        || { echo "pyCIPHOD.tar.gz does not match $SHA256" >&2; exit 1; }
    tar -xzf "$DEST/pyCIPHOD.tar.gz" -C "$DEST"
    echo "$SHA256" > "$CODE/ARCHIVE_SHA256"
fi

echo ">> ciphod-env"
uv venv "$VENV" --python 3.12 --clear
uv pip install --python "$VENV/bin/python" -r "$LOCK"
sha256sum "$LOCK"

# One fit on a small planted case: a -> x -> y <- b with c unrelated, columns (a, x, y, b, c). The
# collider at y orients both edges into it, so the effect of x on y is identified and the
# outcome's other parent, b, is the adjustment set; a is not adjacent to y.
export PYTHONHASHSEED=0
ANDREY_BENCH_CIPHOD="$CODE" "$VENV/bin/python" -P - "$REPO/benchmarks" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.ciphod_locpc import CiphodLocPCCDE

rng = np.random.default_rng(0)
n = 4000
a, b, c = rng.standard_normal((3, n))
x = 0.9 * a + rng.standard_normal(n)
y = 0.9 * x + 0.9 * b + rng.standard_normal(n)
data = np.column_stack([a, x, y, b, c])

adapter = CiphodLocPCCDE()
for pair in ((1, 2), (0, 2)):
    params = {**adapter.params(), "treatment": pair[0], "outcome": pair[1]}
    (record,), tests = adapter.records(adapter.fit(data, params), params)
    print(
        f"{adapter.package_version} | {tests} tests | {record['treatment']}->{record['outcome']} "
        f"{record['reason']}: nonzero {record['nonzero']} identifiable {record['identifiable']} "
        f"{record['set_kind']} {record['adjustment_sets']}"
    )
print(f"ciphod-env OK numpy {np.__version__} python {sys.version.split()[0]}")
PY

printf 'export ANDREY_BENCH_CIPHOD_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_CIPHOD=%q\n' "$CODE"
echo 'export PYTHONHASHSEED=0'
echo ">> done"
