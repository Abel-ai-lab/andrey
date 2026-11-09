#!/usr/bin/env bash
# Fetches the authors' R code for MVPC, the reference for `pc(missing="mvpc")`, and installs the R
# packages it loads.
#
#   benchmarks/start_r.sh          # first: the R lane, with pcalg
#   benchmarks/start_mvpc_r.sh     # then: the authors' code, read from disk by the benchmark
#
# The code accompanies Tu, Zhang, Ackermann, Mohan, Kjellstrom, and Zhang (2019). Its repository has
# no license file, so the benchmark runs it unmodified as a comparator: it `source()`s the files
# from the directory this script prints and copies none of them. The archive of the pinned commit
# is checked against its SHA-256. The script writes the extracted repository under
# `$ANDREY_BENCH_MVPC_R_DIR` (default `benchmarks/.mvpc-r`); export what it prints before running
# the missing-value lane.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"
DEST="${ANDREY_BENCH_MVPC_R_DIR:-$REPO/benchmarks/.mvpc-r}"
COMMIT="e95160c00987f540e47af776bbd7b6a1d9c25d5b"
URL="https://codeload.github.com/TURuibo/MVPC/tar.gz/$COMMIT"
SHA256="3780c273f233b2bad88270417bfeaf18d4d62cb2005af1469280377fcea44ec7"
CODE="$DEST/MVPC-$COMMIT"
# The packages `src/MissingValuePC.R` loads beside pcalg; R.matlab is loaded only by a function the
# benchmark never calls.
PACKAGES='c("mvtnorm", "weights", "ks", "e1071", "DescTools", "mipfp")'

mkdir -p "$DEST"
if [[ ! -f "$CODE/src/MissingValuePC.R" ]]; then
    curl -fsSL -o "$DEST/mvpc.tar.gz" "$URL"
    echo "$SHA256  $DEST/mvpc.tar.gz" | sha256sum --check --status \
        || { echo "mvpc.tar.gz does not match $SHA256" >&2; exit 1; }
    tar -xzf "$DEST/mvpc.tar.gz" -C "$DEST"
    echo "$SHA256" > "$CODE/ARCHIVE_SHA256"
fi

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
Rscript --vanilla -e "
    lib <- Sys.getenv('R_LIBS_USER')
    .libPaths(lib)
    options(repos = c(CRAN = 'https://cloud.r-project.org'))
    pkgs <- $PACKAGES
    need <- Filter(function(p) length(find.package(p, quiet = TRUE)) == 0L, pkgs)
    if (length(need)) install.packages(need, lib = lib)
    for (p in c('pcalg', pkgs)) cat(p, as.character(packageVersion(p)), '\n')
    for (f in c('CITest.R', 'MissingValuePC.R', 'SyntheticDataGeneration.R'))
        source(file.path('$CODE', 'src', f))
    cat('MVPC sources OK\n')
"
printf 'export ANDREY_BENCH_MVPC_R=%q\n' "$CODE/src"
