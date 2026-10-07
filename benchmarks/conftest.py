"""Put the harness root on ``sys.path`` so ``andrey_bench`` imports from any working directory.

``andrey_bench`` is not installed — ``[tool.uv] package = false``, a runnable harness rather
than a distributable package — so it resolves by path. Here rather than in the tests, so
``pytest benchmarks/tests`` works from the repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

#: Set on a machine that is supposed to have R, so a missing one fails instead of skipping.
REQUIRE_R_ENV = "ANDREY_BENCH_REQUIRE_R"


def r_available() -> bool:
    """Test whether `rpy2` can load R by importing it."""
    try:
        import rpy2.robjects  # noqa: F401
    except Exception:
        return False
    return True


def needs_r() -> pytest.MarkDecorator:
    """Skip tests without R unless `$ANDREY_BENCH_REQUIRE_R` is set.

    R tests skip on import errors in CI. Setting the variable on a benchmark node makes a missing
    R module fail the tests before a campaign, instead of passing a suite that skipped R checks.
    """
    skip = not r_available() and not os.environ.get(REQUIRE_R_ENV)
    return pytest.mark.skipif(skip, reason="no R / rpy2 in this environment")
