#!/usr/bin/env bash
# Fetches tetrad-lib 8.0.0, whose EM covariance policy is the reference for `missing="em"` on
# `ges`, `boss`, and `grasp`.
#
#   benchmarks/start_tetrad_missing.sh
#   export ANDREY_BENCH_TETRAD_MISSING_JAR=$PWD/benchmarks/.tetrad-missing/tetrad-lib-8.0.0-shaded.jar
#
# 8.0.0 is the first Tetrad release with `MissingDataPolicy`; the other Tetrad lanes stay on the
# jars their pages name. The shaded jar comes from Maven Central, is checked against its SHA-256,
# and is kept under `$ANDREY_BENCH_TETRAD_MISSING_DIR` (default `benchmarks/.tetrad-missing`),
# which git ignores. The adapter (`andrey_bench/adapters/tetrad_missing.py`) runs
# `TetradMissingDriver.java` from source against it, so `java` needs the `jdk.compiler` module, as a
# JDK 21 or later has; `ANDREY_BENCH_JAVA` names `java` when it is not on `PATH`.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_TETRAD_MISSING_DIR:-$REPO/benchmarks/.tetrad-missing}"
VERSION="8.0.0"
JAR="$DEST/tetrad-lib-$VERSION-shaded.jar"
URL="https://repo1.maven.org/maven2/io/github/cmu-phil/tetrad-lib/$VERSION/tetrad-lib-$VERSION-shaded.jar"
SHA256="956caf7390ff0a8c47a263e3abd00e64a66e388fccdd538b5fc59776ee86554b"

mkdir -p "$DEST"
if [[ ! -f "$JAR" ]]; then
    curl -fsSL -o "$JAR.part" "$URL"
    mv "$JAR.part" "$JAR"
fi
echo "$SHA256  $JAR" | sha256sum --check --status \
    || { echo "$JAR does not match $SHA256" >&2; exit 1; }

JAVA="${ANDREY_BENCH_JAVA:-java}"
"$JAVA" --list-modules | grep -q '^jdk\.compiler@' \
    || { echo "$JAVA has no jdk.compiler module; set ANDREY_BENCH_JAVA to a JDK's java" >&2; exit 1; }
echo ">> tetrad-lib $VERSION OK, $("$JAVA" -version 2>&1 | head -n 1)"
printf 'export ANDREY_BENCH_TETRAD_MISSING_JAR=%q\n' "$JAR"
