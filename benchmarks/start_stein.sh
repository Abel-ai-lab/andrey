#!/usr/bin/env bash
# Fetches the authors' SCORE code and builds the environment of the Stein-score screen's references.
#
#   benchmarks/start_r.sh        # first: the R lane, whose module supplies Rscript
#   benchmarks/start_stein.sh    # then: SCORE, NoGAM, and R2-SortnRegress, read by stein_cpu.py
#
# SCORE is paulrolland1307/SCORE (Apache-2.0) at the commit below. The script downloads its files
# one by one, checks each file's SHA-256, and writes them under `$ANDREY_BENCH_SCORE_DIR` (default
# `benchmarks/.score`); the benchmark imports them from there and copies none of them. NoGAM is the
# first author's code in dodiscover (MIT), which has no release with it, installed at a pinned
# commit; R2-SortnRegress is CausalDisco 0.3.0 (BSD-3-Clause). requirements-stein.txt pins every
# package of the environment, `.venv-stein`: the SCORE code imports GPy, the Causal Discovery
# Toolbox, igraph, matplotlib, and torch at the top of its modules, and both it and dodiscover need
# NumPy 1. The environment imports none of Andrey; the driver scores the results in bench-env.
# SCORE prunes with `mgcv` through Rscript, so the script installs `mgcv` into the R library.
# Export what it prints before running stein_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"
DEST="${ANDREY_BENCH_SCORE_DIR:-$REPO/benchmarks/.score}"
VENV="$REPO/benchmarks/.venv-stein"
LOCK="$REPO/benchmarks/requirements-stein.txt"
COMMIT="c2c2b8e0645b1a2a3970b5a16710ee776c4ed801"
URL="https://raw.githubusercontent.com/paulrolland1307/SCORE/$COMMIT"

echo ">> SCORE $COMMIT"
mkdir -p "$DEST"
while read -r sha file; do
    if [[ ! -f "$DEST/$file" ]]; then
        curl -fsSL -o "$DEST/$file.part" "$URL/$file"
        mv "$DEST/$file.part" "$DEST/$file"
    fi
    echo "$sha  $DEST/$file" | sha256sum --check --status \
        || { echo "$file does not match $sha" >&2; exit 1; }
done <<'FILES'
0338e8379c7e1a7a83ca0b949311971de58fb29c146e5aaab03242e18aa2cf81 stein.py
f615131446ccf9b445a0639d331803937225a7de8f95095d18e8cf9beca62190 utils.py
4b4e5f8f97924cc9c8f194188074c1773d44bdd784de08dd72d73c26c64bec81 cam_pruning.R
c0166382a09d1c8ec38867a4f9b8ecc98a3f16a1bd47c19fe25f351922abb497 pruning.R
9b11cc0459c275124c0a44e2e713404b94f58c634ee5f4e0e67175febee9c6fc selGam.R
7298868bec828a126b2406a93baa26ccb6feb2b849f1fe4a649e3a4e9076c4c6 train_gam.R
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 LICENSE
FILES
echo "$COMMIT" > "$DEST/COMMIT"

echo ">> stein-env"
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r "$LOCK"
sha256sum "$LOCK"

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS_USER")
    .libPaths(lib)
    options(repos = c(CRAN = "https://cloud.r-project.org"))
    if (length(find.package("mgcv", quiet = TRUE)) == 0L) install.packages("mgcv", lib = lib)
    cat("mgcv", as.character(packageVersion("mgcv")), "\n")
'

# One fit of each reference on a small chain, SCORE's pruning through R included.
ANDREY_BENCH_SCORE="$DEST" "$VENV/bin/python" - "$REPO/benchmarks" <<'PY'
import importlib.metadata as m
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.causaldisco_r2sort import CausalDiscoR2SortnRegress
from andrey_bench.adapters.dodiscover_nogam import DodiscoverNoGAM
from andrey_bench.adapters.fcm_warm import tiny_table
from andrey_bench.adapters.rolland_score import RollandSCORE

x = tiny_table()
edges = {}
for adapter in (RollandSCORE(), DodiscoverNoGAM(), CausalDiscoR2SortnRegress()):
    adjacency = adapter.fit(x, adapter.params())
    edges[adapter.name] = sorted((int(i), int(j)) for i, j in zip(*np.nonzero(adjacency)))
print(
    f"stein-env OK numpy {np.__version__} torch {m.version('torch')} dodiscover "
    f"{m.version('dodiscover')} CausalDisco {m.version('causaldisco')} | {edges}"
)
PY

printf 'export ANDREY_BENCH_STEIN_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_SCORE=%q\n' "$DEST"
echo ">> done"
