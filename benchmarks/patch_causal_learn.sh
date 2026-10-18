#!/usr/bin/env bash
# Applies benchmarks/patches/causal-learn-gin-column-index.patch to the causal-learn installed in a
# venv. Already applied: nothing to do. Any other mismatch fails.
#
#   benchmarks/patch_causal_learn.sh <venv python>

set -euo pipefail

PY=${1:?usage: patch_causal_learn.sh <venv python>}
PATCH="$(cd "$(dirname "$0")" && pwd)/patches/causal-learn-gin-column-index.patch"
SITE=$("$PY" -c 'import causallearn, os; print(os.path.dirname(os.path.dirname(causallearn.__file__)))')

if patch -p1 -R --dry-run -s -f -d "$SITE" < "$PATCH" > /dev/null; then
  echo "causal-learn GIN column-index fix already applied"
else
  patch -p1 -N -s -d "$SITE" < "$PATCH"
  echo "causal-learn GIN column-index fix applied"
fi
