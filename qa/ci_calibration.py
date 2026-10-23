"""Measure how often each nonlinear conditional-independence test rejects, with and without a
dependence.

Run ``uv run python -m qa.ci_calibration``, or name the tests and rows:
``... ci_calibration kci rcot --rows 200``.
Each draw has columns ``x, y, z_1..z_k``: ``x = tanh(2 s) + e`` and ``y = s^2 + e'`` with ``s`` the
scaled sum of the ``z``, so ``x`` is independent of ``y`` given ``z`` (with ``k = 0``, ``x`` and
``y`` are independent noise). Adding one noise term to both makes them dependent. For each test,
row count, and ``k``, the tool prints the share of seeded draws rejected at ``alpha = 0.05``
without the shared term (the level, near 0.05 for a calibrated test) and with it (the power), and
the median seconds per query. It is a measurement with no passing threshold; the tests run at
their default settings, serially, with one BLAS thread.
"""

import argparse
import time

import numpy as np
from threadpoolctl import threadpool_limits

from andrey.core.ci import make_indep_test

ALPHA = 0.05
SHARED = 0.5
#: Test -> (row counts, null draws per cell). The permutation tests cost more per query.
GRID = {
    "kci": ((200, 500), 200),
    "rcot": ((200, 500, 2000), 200),
    "cmiknn": ((200, 500), 50),
    "gpdc": ((200, 500), 50),
}
CONDITIONING_SIZES = (0, 1, 2, 3)
POWER_DRAWS = 20


def draw(seed: int, n: int, k: int, shared: float) -> np.ndarray:
    """One draw with columns ``x, y, z_1..z_k``; ``shared`` scales the noise ``x`` and ``y``
    share."""
    rng = np.random.default_rng([seed, n, k])
    z = rng.standard_normal((n, k))
    s = z.sum(axis=1) / np.sqrt(k) if k else np.zeros(n)
    u = rng.standard_normal(n)
    x = np.tanh(2.0 * s) + 0.4 * rng.standard_normal(n) + shared * u
    y = s**2 + 0.4 * rng.standard_normal(n) + shared * u
    return np.column_stack([x, y, z])


def rejection_rate(test: str, n: int, k: int, draws: int, shared: float) -> tuple[float, float]:
    """The share of ``draws`` seeded draws rejected at ``ALPHA``, and the median seconds."""
    rejected, seconds = 0, []
    for seed in range(draws):
        data = draw(seed, n, k, shared)
        start = time.perf_counter()
        p = make_indep_test(test, data)(0, 1, tuple(range(2, 2 + k)))
        seconds.append(time.perf_counter() - start)
        rejected += p <= ALPHA
    return rejected / draws, float(np.median(seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tests", nargs="*", help=f"tests to measure (default: {' '.join(GRID)})")
    parser.add_argument("--rows", type=int, nargs="+", help="row counts (default: each test's)")
    args = parser.parse_args()
    if unknown := sorted(set(args.tests) - set(GRID)):
        parser.error(f"unknown tests {unknown}; the tests are {list(GRID)}")
    with threadpool_limits(limits=1):
        print("test     rows  |S|  level  (draws)  power  s/query")
        for test in args.tests or list(GRID):
            rows, draws = GRID[test]
            for n in args.rows or rows:
                for k in CONDITIONING_SIZES:
                    level, seconds = rejection_rate(test, n, k, draws, 0.0)
                    power, _ = rejection_rate(test, n, k, POWER_DRAWS, SHARED)
                    print(
                        f"{test:8s} {n:5d}  {k:3d}  {level:5.3f}  ({draws:4d})  {power:5.2f}  "
                        f"{seconds:7.3f}",
                        flush=True,
                    )
