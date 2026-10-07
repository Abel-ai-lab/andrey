#!/usr/bin/env bash
# Installs the R package library, then `rpy2`, in the benchmark environment.
#
#   benchmarks/start.sh          # first: the Python bench venv
#   benchmarks/start_r.sh        # then: R packages + rpy2
#
# `$ANDREY_BENCH_R_MODULE` names the modules that supply R: an R configured with `--enable-R-shlib`,
# whose `libR.so` `rpy2` embeds in the benchmark worker, and the compiler it was built with. Python
# has already loaded the system `libstdc++` by then, so `libR.so` finds the newer one it needs only
# through the compiler module.
#
# The package library sits beside the venv, one per checkout like `.venv-bench` itself. Point
# `$ANDREY_BENCH_R_LIBS` at a shared directory to build it once for checkouts using the same R.
#
# Every benchmark process needs the same two settings this script uses. Print them with
#
#   benchmarks/start_r.sh --env
#
# and export them in the shell, job script, or driver that runs an R solution.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"

if [[ "${1:-}" == "--env" ]]; then
    printf 'export ANDREY_BENCH_R_MODULE=%q\n' "$R_MODULE"
    printf 'export ANDREY_BENCH_R_LIBS=%q\n' "$R_LIBS"
    echo "module load $R_MODULE"
    printf 'export R_LIBS_USER=%q\n' "$R_LIBS"
    exit 0
fi

echo ">> R modules: $R_MODULE"
# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
command -v Rscript >/dev/null || { echo "Rscript not on PATH after 'module load $R_MODULE'" >&2; exit 1; }
[[ -f "$R_HOME/lib/libR.so" ]] || { echo "$R_HOME/lib/libR.so missing; rpy2 needs an R built --enable-R-shlib" >&2; exit 1; }

mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
echo ">> R library: $R_LIBS"

# `pcalg` uses two Bioconductor graph packages, installed by `BiocManager`; the rest come from CRAN.
# `update = FALSE` prevents rebuilding packages used in earlier campaign measurements.
Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS_USER")
    .libPaths(lib)
    options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))
    need <- function(p) length(find.package(p, quiet = TRUE)) == 0L
    if (need("BiocManager")) install.packages("BiocManager", lib = lib)
    bioc <- Filter(need, c("graph", "RBGL"))
    if (length(bioc)) BiocManager::install(bioc, lib = lib, ask = FALSE, update = FALSE)
    cran <- Filter(need, c("pcalg", "bnlearn"))
    if (length(cran)) install.packages(cran, lib = lib)
    for (p in c("pcalg", "bnlearn")) {
        cat(p, as.character(packageVersion(p)), "\n")
    }
'

echo ">> rpy2"
# `rpy2` compiles against the loaded R, so the module must stay loaded for installation and every
# process that imports it.
source "$REPO/benchmarks/.venv-bench/bin/activate"
uv pip install rpy2
python -c "
import importlib.metadata as metadata
import rpy2.robjects as ro
print('rpy2', metadata.version('rpy2'), ro.r('R.version.string')[0])
"
deactivate

echo ">> done"
