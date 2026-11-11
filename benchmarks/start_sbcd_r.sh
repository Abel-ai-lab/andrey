#!/usr/bin/env bash
# Fetches the authors' R code for Y-structures under selection bias, the reference for
# `y_structures`, and installs the R packages it loads.
#
#   benchmarks/start_r.sh          # first: the R lane, with pcalg and RBGL
#   benchmarks/start_sbcd_r.sh     # then: the authors' code, read from disk by the benchmark
#
# The code is sbcd by Versteeg, Mooij, and Zhang, at the commit below, under the BSD 3-clause
# license. The benchmark runs it unmodified as a comparator: it `source()`s the files from the
# directory this script prints and copies none of them. The archive of the pinned commit is checked
# against its SHA-256. The script writes the extracted repository under `$ANDREY_BENCH_SBCD_R_DIR`
# (default `benchmarks/.sbcd-r`) and adds the packages the repository lists beside pcalg and RBGL
# (expm, PRROC, InvariantCausalPrediction) to the R lane's library; export what it prints before
# running `y_structures_cpu.py`.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"
DEST="${ANDREY_BENCH_SBCD_R_DIR:-$REPO/benchmarks/.sbcd-r}"
COMMIT="4bae8afb885382b7045cf2d67c25a88fd1c013e8"
URL="https://codeload.github.com/philipversteeg/sbcd/tar.gz/$COMMIT"
SHA256="1cec9a74ee55ba97a43d634c824770700f94b40c4b01995c9e1dd28c52d49a06"
CODE="$DEST/sbcd-$COMMIT"

mkdir -p "$DEST"
if [[ ! -f "$CODE/R/wrapYStructures.R" ]]; then
    curl -fsSL -o "$DEST/sbcd.tar.gz" "$URL"
    echo "$SHA256  $DEST/sbcd.tar.gz" | sha256sum --check --status \
        || { echo "sbcd.tar.gz does not match $SHA256" >&2; exit 1; }
    tar -xzf "$DEST/sbcd.tar.gz" -C "$DEST"
    echo "$SHA256" > "$CODE/ARCHIVE_SHA256"
fi

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
# pcalg and RBGL come from `start_r.sh`; a library without them stops here instead of building them.
Rscript --vanilla -e "
    lib <- Sys.getenv('R_LIBS_USER')
    .libPaths(lib)
    options(repos = c(CRAN = 'https://cloud.r-project.org'))
    for (p in c('pcalg', 'RBGL'))
        if (length(find.package(p, quiet = TRUE)) == 0L) stop(p, ' is missing: run start_r.sh first')
    pkgs <- c('expm', 'PRROC', 'InvariantCausalPrediction')
    need <- Filter(function(p) length(find.package(p, quiet = TRUE)) == 0L, pkgs)
    if (length(need)) install.packages(need, lib = lib)
    for (p in c('pcalg', 'RBGL', pkgs)) cat(p, as.character(packageVersion(p)), '\n')
    # The files simulate.R sources, from the repository's own directory as it does.
    setwd('$CODE')
    for (f in c('R/lib.R', 'R/wrapYStructures.R', 'R/wrapLCD.R', 'R/wrapICP.R')) source(f)
    cat('sbcd sources OK\n')
"
printf 'export ANDREY_BENCH_SBCD_R=%q\n' "$CODE"
