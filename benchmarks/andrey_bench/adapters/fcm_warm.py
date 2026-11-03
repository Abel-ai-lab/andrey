"""A tiny additive-noise table that warms a solution inside `setup`, outside fit timing.

The first call of a method loads lazily imported modules and fills caches. At ``--warmup 0`` that
cost would land in the first timed fit, and a slow filesystem makes it seconds. Each solution of
the functional-causal-model lanes fits this table in `setup`, so every one starts its timed fit
warm.
"""

from __future__ import annotations

import numpy as np


def tiny_table(n: int = 150, seed: int = 0) -> np.ndarray:
    """Four continuous columns with a nonlinear chain ``x0 -> x1 -> x2`` and an independent ``x3``."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = np.sin(2 * x0) + 0.4 * rng.normal(size=n)
    x2 = x1**2 + 0.4 * rng.normal(size=n)
    x3 = rng.normal(size=n)
    return np.column_stack([x0, x1, x2, x3])
