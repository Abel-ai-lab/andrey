"""Count how often reordering the columns changes FCI's and GFCI's endpoint marks.

Run ``uv run python -m qa.column_order``. The grid holds linear-Gaussian Erdos-Renyi and
scale-free graphs with one hidden confounder and average degree 2: 6, 8, 10, and 12 total nodes,
500, 1200, and 5000 samples, seeds 0-9. Draws without a node that has two children are skipped.
Each model is refit on three seeded column orders and mapped back; a relabeling counts as changed
when any final endpoint mark differs from the fit in the original order. Fits use the default
settings on numpy, with one worker and one BLAS thread.
"""

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

import andrey
from andrey import data

ORDERS = 3


def models():
    """Yield the observed data and seed of every eligible model in the grid."""
    for graph in ("erdos_renyi", "scale_free"):
        for d in (6, 8, 10, 12):
            for n in (500, 1200, 5000):
                for seed in range(10):
                    try:
                        ds = data.sample_scm(
                            graph=graph, d=d, n=n, seed=seed, density=2.0, latents=1
                        )
                    except ValueError as exc:
                        if "children" not in str(exc):
                            raise
                        continue
                    yield ds.data, seed


def changes(method: str, x: np.ndarray, seed: int) -> int:
    """How many of the seeded column orders change the fit's marks after mapping back."""
    fit = getattr(andrey, method)
    marks = fit(x).structure.to_numpy()
    count = 0
    for k in range(ORDERS):
        order = np.random.default_rng(seed * 31 + k + 991).permutation(x.shape[1])
        back = np.argsort(order)
        count += bool((fit(x[:, order]).structure.to_numpy()[np.ix_(back, back)] != marks).any())
    return count


if __name__ == "__main__":
    warnings.simplefilter("ignore", andrey.ExperimentalWarning)  # GFCI is experimental
    with andrey.config(backend="numpy", num_workers=1), threadpool_limits(limits=1):
        grid = list(models())
        for method in ("fci", "gfci"):
            counts = [changes(method, x, seed) for x, seed in grid]
            print(
                f"{method}: {sum(counts)} of {ORDERS * len(grid)} relabelings change the marks "
                f"({sum(c > 0 for c in counts)} of {len(grid)} models)"
            )
