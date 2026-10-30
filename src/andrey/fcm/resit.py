"""RESIT: regression with subsequent independence test (Peters, Mooij, Janzing, Scholkopf 2014).

RESIT (Algorithm 1 of the paper) assumes an additive noise model, ``x_j = f_j(parents) + e_j``
with each noise independent of the variable's non-descendants, and runs two phases:

1. **Causal order.** Each remaining variable is regressed on the other remaining ones, and its
   residual is tested for independence of them jointly. The variable whose residual has the largest
   p-value is the next sink: it leaves the set, and the variables still in it are its possible
   parents. The causal order is the sinks in reverse.
2. **Edge removal.** Each variable's possible parents are tried one at a time: a parent goes when
   the residual of the fit on the other current parents is independent of all the variable's
   predecessors in the order, at a p-value above ``alpha``. With one parent left, the residual
   without it is the variable itself.

The settings are those of the paper's experiments: the regression is an additive model of thin
plate regression splines with ten basis functions per term and GCV smoothing,
:func:`andrey.fcm.gam.fit_gam`, which lowers the rank with fewer than 30 rows per term; the test is
the gamma HSIC test of :func:`andrey.core.independence.hsic_gamma_log_p`, with Gaussian kernels on
each side's joint squared distances, the bandwidth set by the median heuristic, and the null
moments of Gretton et al. (2007) in their product form; and ``alpha`` is ``0.05``.

The choices Algorithm 1 leaves open:

- The kernels read the columns centered and divided by their standard deviations, so neither a
  column's units nor its offset changes a test.
- The bandwidth is the median over every pair of rows with a positive distance.
- p-values are compared as their logarithms, computed directly so that a p-value below the
  smallest double keeps its place, rounded to ``_KEY_DP`` decimal places, as :func:`andrey.hc`
  compares its score changes; a tie goes to the earlier column.
- The parents are tried in column order, and :func:`andrey.resit` passes the columns in their rank
  order, so they are tried by rank. Each removal changes the fits after it, so another order can
  keep other parents.

The fit draws no random numbers.

References
----------
Peters, Mooij, Janzing, Scholkopf. "Causal discovery with continuous additive noise models."
JMLR 15:2009-2053, 2014.

Gretton, Fukumizu, Teo, Song, Scholkopf, Smola. "A kernel statistical test of independence."
NIPS 20, 2007.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from andrey.core.independence import gamma_gram, hsic_gamma_log_p
from andrey.core.units import unit_factors
from andrey.fcm.gam import BasisCache, fit_gam
from andrey.search._parallel import _KEY_DP

_log = logging.getLogger(__name__)

#: Significance level of the edge-removal tests.
ALPHA = 0.05
#: Basis functions per term of each regression.
K = 10


def _residual(bases: BasisCache, child: int, parents: list[int]) -> np.ndarray:
    """The residual of ``child``'s additive fit on ``parents``; with none, ``child`` centered."""
    if not parents:
        y = bases.data[:, child]
        return y - y.mean()
    return fit_gam(bases, child, parents, k=K).residuals


def _gram(side: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The gamma HSIC gram of ``side``, one column or several read jointly, its median over every
    row, with the centered gram's sum of squares, which the product-form variance reads, so the
    edge removal's tests against one predecessor gram sum it once."""
    centered, mu = gamma_gram(side, median_rows=None)
    return centered, mu, np.sum(centered * centered)


def _log_p(gram: tuple[np.ndarray, float, float], residual: np.ndarray) -> float:
    """The log HSIC p-value of ``residual`` against the side whose :func:`_gram` is ``gram``, the
    null variance in its product form.

    Raises
    ------
    FloatingPointError
        If the test's value is not finite, so that no choice rests on it.
    """
    n = residual.shape[0]
    log_p = hsic_gamma_log_p(gram, _gram(residual), n, variance="product")
    if not math.isfinite(log_p):
        raise FloatingPointError(
            "resit: an HSIC test gave a non-finite p-value; the kernels could not be computed "
            "from these data"
        )
    return log_p


def _key(log_p: float) -> float:
    """``log_p`` rounded to ``_KEY_DP`` decimal places."""
    return round(log_p, _KEY_DP)


def causal_order(
    bases: BasisCache,
    unit: np.ndarray,
    allowed: np.ndarray,
    precedence: np.ndarray | None = None,
) -> np.ndarray:
    """Find a causal order from the sink up, the first phase.

    Each step regresses every candidate on the remaining variables it may have as parents and tests
    its residual against all the other remaining ones; the candidate of largest p-value is the next
    sink. The log p-values are compared by :func:`_key`, and a tie goes to the earlier column.

    Parameters
    ----------
    bases : BasisCache
        Bases of the data's columns.
    unit : np.ndarray
        ``(n, d)`` columns the kernels read.
    allowed : np.ndarray
        ``(d, d)`` bool; ``allowed[i, j]`` lets ``i`` be a parent of ``j``.
    precedence : np.ndarray or None
        ``(d, d)`` bool; ``precedence[a, b]`` puts ``a`` before ``b``, so ``a`` is a candidate
        only once ``b`` has left.

    Returns
    -------
    np.ndarray
        The order, causes first.
    """
    remaining = list(range(unit.shape[1]))
    sinks: list[int] = []
    while len(remaining) > 1:
        candidates = [
            k for k in remaining if precedence is None or not precedence[k, remaining].any()
        ]
        keys = {}
        for k in candidates:
            others = [i for i in remaining if i != k]
            parents = [i for i in others if allowed[i, k]]
            log_p = _log_p(_gram(unit[:, others]), _residual(bases, k, parents))
            _log.debug("%d remaining: sink %d, log p-value %.9g", len(remaining), k, log_p)
            keys[k] = _key(log_p)
        sink = min(candidates, key=lambda k: (-keys[k], k))
        remaining.remove(sink)
        sinks.append(sink)
    return np.array([*remaining, *reversed(sinks)], dtype=np.int64)


def remove_edges(
    bases: BasisCache,
    unit: np.ndarray,
    order: np.ndarray,
    *,
    alpha: float = ALPHA,
    allowed: np.ndarray,
    required: np.ndarray,
) -> np.ndarray:
    """Remove the superfluous edges of the complete DAG of ``order``, the second phase.

    Each variable's possible parents are its allowed predecessors, tried in column order; a parent
    that is not ``required`` goes when the residual of the fit on the other current parents has a
    p-value above ``alpha`` against all the variable's predecessors.

    Returns
    -------
    np.ndarray
        ``(d, d)`` bool DAG; ``out[i, j]`` is the edge ``i -> j``.
    """
    d = unit.shape[1]
    dag = np.zeros((d, d), dtype=bool)
    for position in range(1, d):
        child = int(order[position])
        gram = _gram(unit[:, order[:position]])
        parents = sorted(int(i) for i in order[:position] if allowed[i, child])
        for parent in list(parents):
            if required[parent, child]:
                continue
            rest = [i for i in parents if i != parent]
            log_p = _log_p(gram, _residual(bases, child, rest))
            _log.debug("edge %d -> %d: log p-value without it %.9g", parent, child, log_p)
            if log_p > math.log(alpha):
                parents.remove(parent)
        dag[parents, child] = True
    return dag


@dataclass(frozen=True)
class RESITResult:
    """A RESIT fit: the DAG after the edge removal and the causal order, causes first."""

    dag: np.ndarray
    order: np.ndarray


def resit(
    X: np.ndarray,
    *,
    alpha: float = ALPHA,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> RESITResult:
    """Fit RESIT to the ``(n, d)`` matrix ``X``.

    A ``forbidden`` edge is in no regression and never in the result, a ``required`` one is never
    removed, and ``precedence`` goes to :func:`causal_order`. Ties go to the earlier column
    throughout, so a caller that orders the columns by rank breaks them by rank.
    """
    d = X.shape[1]
    allowed = ~np.eye(d, dtype=bool)
    if forbidden is not None:
        allowed &= ~forbidden
    required = np.zeros((d, d), dtype=bool) if required is None else required
    bases = BasisCache(X)
    unit = (X - X.mean(axis=0)) * unit_factors(X, standardize=True)
    order = causal_order(bases, unit, allowed, precedence)
    dag = remove_edges(bases, unit, order, alpha=alpha, allowed=allowed, required=required)
    return RESITResult(dag=dag, order=order)
