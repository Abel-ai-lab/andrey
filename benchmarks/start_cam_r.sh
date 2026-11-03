#!/usr/bin/env bash
# Installs CAM 1.0, the reference for Andrey's CAM, into the R lane's package library.
#
#   benchmarks/start_r.sh        # first: the R lane
#   benchmarks/start_cam_r.sh    # then: CAM 1.0 and the packages it depends on
#
# CAM was archived on CRAN, so the source comes from the archive and its SHA-256 is checked. The
# library is the one `start_r.sh` uses (`$ANDREY_BENCH_R_LIBS`, `$ANDREY_BENCH_R_MODULE`).

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"
URL="https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz"
SHA256="74957cf793568723b21105a03543316da9b3fa7c9d7dc1d5fba63224cdcfb51e"

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/CAM_1.0.tar.gz" "$URL"
echo "$SHA256  $tmp/CAM_1.0.tar.gz" | sha256sum --check --status \
    || { echo "CAM_1.0.tar.gz does not match $SHA256" >&2; exit 1; }

Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS_USER")
    .libPaths(lib)
    options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))
    need <- function(p) length(find.package(p, quiet = TRUE)) == 0L
    deps <- Filter(need, c("glmnet", "mboost", "mgcv"))
    if (length(deps)) install.packages(deps, lib = lib)
    install.packages(file.path(commandArgs(TRUE)[1], "CAM_1.0.tar.gz"), lib = lib, repos = NULL,
                     type = "source")
    for (p in c("CAM", "glmnet", "mboost", "mgcv")) cat(p, as.character(packageVersion(p)), "\n")
' "$tmp"
