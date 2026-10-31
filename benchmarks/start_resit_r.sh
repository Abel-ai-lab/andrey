#!/usr/bin/env bash
# Fetches the authors' R code for RESIT, the reference for Andrey's RESIT, and installs `kernlab`.
#
#   benchmarks/start_r.sh          # first: the R lane
#   benchmarks/start_resit_r.sh    # then: the authors' code, read from disk by the benchmark
#
# The code is supplementary material to Peters, Mooij, Janzing, and Scholkopf (2014), released
# under a two-clause BSD license. The benchmark `source()`s it from the directory this script
# prints and copies none of it. The archive's SHA-256 is checked. The script writes the extracted
# `codeANM/` under `$ANDREY_BENCH_RESIT_R_DIR` (default `benchmarks/.resit-r`); export what it
# prints before running the RESIT lane.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"
DEST="${ANDREY_BENCH_RESIT_R_DIR:-$REPO/benchmarks/.resit-r}"
URL="http://people.tuebingen.mpg.de/jpeters/onlineCodeANM.zip"
SHA256="221cf4510ffc2175de138dd64a71253006c8b294b55c6efa72766cc154598795"

mkdir -p "$DEST"
if [[ ! -f "$DEST/codeANM/code/inferDAG/ICML.R" ]]; then
    curl -fsSL -o "$DEST/onlineCodeANM.zip" "$URL"
    echo "$SHA256  $DEST/onlineCodeANM.zip" | sha256sum --check --status \
        || { echo "onlineCodeANM.zip does not match $SHA256" >&2; exit 1; }
    unzip -q -o "$DEST/onlineCodeANM.zip" -d "$DEST"
    echo "$SHA256" > "$DEST/codeANM/ARCHIVE_SHA256"
fi

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
    need <- Filter(function(p) length(find.package(p, quiet = TRUE)) == 0L, c("kernlab", "mgcv"))
    if (length(need)) install.packages(need, lib = lib)
    for (p in c("kernlab", "mgcv")) cat(p, as.character(packageVersion(p)), "\n")
'
printf 'export ANDREY_BENCH_RESIT_R=%q\n' "$DEST/codeANM/code"
