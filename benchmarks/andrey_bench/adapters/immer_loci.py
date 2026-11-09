"""LOCI adapter for the authors' code (Immer et al., 2023), run unmodified in its own environment.

The code is ``aleximmer/loci`` at commit :data:`COMMIT`, a repository with no license file: its
``causa`` package holds ``loci``. It is a measurement reference only: ``benchmarks/start_loci.sh``
downloads the files a fit imports, checks each one's SHA-256, and builds the environment; the
benchmark imports them from the directory ``$ANDREY_BENCH_LOCI`` names and copies none of them.

A fit standardizes each column to mean 0 and variance 1, as the paper does for its own methods
(Supplementary Material D) and the authors' experiments do with scikit-learn's ``StandardScaler``,
then calls ``loci(x, y, independence_test=..., neural_network=False)``: the paper's estimator with
spline feature maps, once per criterion, ``independence_test=True`` for LOCI-H and ``False`` for
LOCI-M. Everything else is at the code's defaults, which are fixed inside it:

- ``n_steps=None``, which the convex fit reads as at most 1,000 steps; the paper's Supplementary
  Material D says up to 100. The fit stops earlier once the mean objective changes by less than
  ``1e-6`` in a step.
- The features: scikit-learn's ``SplineTransformer(n_knots=25, degree=5)``, fit on the cause; the
  mean's natural parameter reads them, the precision's reads their absolute values.
- The fit: feasible generalized least squares with Gaussian priors of precision ``1e-5`` on both
  weight vectors, a closed-form step in the mean's weights alternating with ten L-BFGS steps (strong
  Wolfe line search) in the precision's, those weights clamped at 0 after each; the precision's
  linear predictor is clamped at ``1e-7``.
- LOCI-H: the biased HSIC statistic with Gaussian kernels, each bandwidth the median heuristic, of
  the cause against the residual divided by the fitted standard deviation, in each direction. The
  score is the backward statistic minus the forward one, so the code compares the two statistics,
  where the paper says it predicts from the p-values.
- LOCI-M: the score is the forward mean log-likelihood minus the backward one.

The edge follows the score's sign: ``0 -> 1`` when it is positive, ``1 -> 0`` when negative, and
no edge when it is 0 or not a number, as the paper counts a pair correct only when the score is
strictly on the right side. The fit draws no random numbers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Names the directory holding the downloaded ``causa`` package.
CODE_ENV = "ANDREY_BENCH_LOCI"

#: The pinned commit of ``aleximmer/loci``; ``start_loci.sh`` fetches it.
COMMIT = "da61b8cdf95cbbc00fc42d351f89e64410f535f4"

#: Optimizer steps of the untimed warm-up fit in ``setup``; at the default 1,000 it costs as much
#: as a timed fit.
WARMUP_STEPS = 5

#: The files a fit imports, relative to the code directory.
REQUIRED_FILES = (
    "causa/__init__.py",
    "causa/loci.py",
    "causa/hsic.py",
    "causa/het_ridge.py",
    "causa/ml.py",
    "causa/utils.py",
)

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


def code_dir() -> Path:
    """The directory of the authors' files, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' LOCI code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_loci.sh and export what it prints."
        )
    root = Path(raw)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks {', '.join(missing)}")
    return root


def _loci():
    """The authors' ``loci`` function, imported from :func:`code_dir`."""
    root = str(code_dir())
    if root not in sys.path:
        sys.path.insert(0, root)
    from causa.loci import loci  # ty: ignore[unresolved-import]  # the downloaded code, on sys.path

    return loci


def standardized(data: np.ndarray) -> np.ndarray:
    """Each column of `data` at mean 0 and variance 1, ``float64``, each column contiguous."""
    z = np.asarray(data, dtype=np.float64)
    return np.asfortranarray((z - z.mean(axis=0)) / z.std(axis=0))


def edge_marks(score: float) -> np.ndarray:
    """The two-node DAG's endpoint marks for a score: positive ``0 -> 1``, negative ``1 -> 0``."""
    marks = np.zeros((2, 2), dtype=np.int8)
    if score > 0:
        marks[0, 1], marks[1, 0] = _TAIL, _ARROW
    elif score < 0:
        marks[1, 0], marks[0, 1] = _TAIL, _ARROW
    return marks


class ImmerLOCI:
    """The authors' LOCI under the independence criterion (LOCI-H), spline features.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "immer.loci.independence"
    package: str = "loci"
    package_version: str = COMMIT[:8]
    backend: str = "torch-cpu"
    mode: str = "serial"
    algorithm: str = "loci"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_LOCI_PYTHON"
    independence_test: bool = True

    def setup(self) -> None:
        """Import the code and fit a tiny pair before timing, so first-call costs stay out.

        The warm-up fit stops after :data:`WARMUP_STEPS` optimizer steps, which run every code
        path of a fit; the timed fit runs the code's default.
        """
        from andrey_bench.adapters.fcm_warm import tiny_table

        z = standardized(tiny_table()[:, :2])
        _loci()(
            z[:, 0],
            z[:, 1],
            independence_test=self.independence_test,
            neural_network=False,
            n_steps=WARMUP_STEPS,
        )

    def params(self) -> dict[str, Any]:
        """The two arguments the adapter sets, the standardization, and the pinned commit."""
        return {
            "commit": COMMIT,
            "independence_test": self.independence_test,
            "neural_network": False,
            "standardize": True,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' LOCI; return the two-node DAG's endpoint marks. The only timed region."""
        loci = _loci()
        z = standardized(data)
        score = loci(
            z[:, 0],
            z[:, 1],
            independence_test=params["independence_test"],
            neural_network=params["neural_network"],
        )
        return edge_marks(float(score))

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class ImmerLOCILikelihood(ImmerLOCI):
    """The authors' LOCI under the likelihood criterion (LOCI-M), spline features."""

    name: str = "immer.loci.likelihood"
    independence_test: bool = False
