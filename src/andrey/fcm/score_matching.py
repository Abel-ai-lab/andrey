"""SCORE: a causal order from the Jacobian of the score (Rolland et al. 2022).

SCORE assumes a nonlinear additive noise model with Gaussian noise, ``x_j = f_j(parents) + e_j``.
Its Lemma 1: the diagonal entry ``j`` of the Jacobian of the score ``grad log p`` is constant over
the sample space exactly when ``j`` is a leaf, for a leaf has ``d^2 log p / d x_j^2 = -1 /
sigma_j^2``, its noise variance, and any other variable adds terms that change with its children's
links, each nonlinear in it. The method runs two phases:

1. **Causal order** (Algorithm 1). The diagonal of the score's Jacobian is estimated at every row
   of the remaining variables by :func:`andrey.fcm.stein.jacobian_diagonal`, and the variable
   whose entry has the smallest variance over the rows is the next leaf: it leaves the set, and
   the estimate, its kernel bandwidth included, is made again on the others. The causal order is
   the leaves in reverse.
2. **Pruning.** The complete DAG of the order is pruned by the smooth-term tests of CAM,
   :func:`andrey.fcm.pruning.prune_order`, at the cutoff ``0.001`` of the paper's experiments.

The choices Algorithm 1 leaves open:

- The estimator's ridge term and bandwidth, as :mod:`andrey.fcm.stein` records them.
- The variances are compared rounded to ``KEY_DP`` decimal places, as :func:`andrey.hc` compares
  its score changes; a tie goes to the earlier column.
- The estimator reads the columns' units, and :func:`andrey.score_matching` passes them centered
  and divided by their standard deviations.

The fit draws no random numbers.

References
----------
Rolland, Cevher, Kleindessner, Russell, Janzing, Scholkopf, Locatello. "Score matching enables
causal discovery of nonlinear additive noise models." ICML, PMLR 162:18741-18753, 2022.

Buhlmann, Peters, Ernest. "CAM: Causal additive models, high-dimensional order search and
penalized regression." Annals of Statistics 42(6):2526-2556, 2014.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from andrey.core.ties import tie_key
from andrey.core.units import unit_factors
from andrey.fcm.pruning import prune_order
from andrey.fcm.stein import jacobian_diagonal

_log = logging.getLogger(__name__)

#: Cutoff of the pruning tests.
ALPHA = 0.001
#: Basis functions per smooth term of the pruning fits.
K = 10


def leaf_order(
    unit: np.ndarray,
    criterion: Callable[[np.ndarray], np.ndarray],
    precedence: np.ndarray | None = None,
) -> np.ndarray:
    """Find a causal order from the leaf up.

    Each step calls ``criterion`` on the remaining columns, which returns one value per column, and
    takes the candidate of smallest value as the next leaf. The values are compared by
    :func:`~andrey.core.ties.tie_key`, and a tie goes to the earlier column.

    Parameters
    ----------
    unit : np.ndarray
        ``(n, d)`` columns the criterion reads.
    criterion : callable
        Maps the ``(n, m)`` remaining columns to ``m`` values; the smallest marks the leaf.
    precedence : np.ndarray or None
        ``(d, d)`` bool; ``precedence[a, b]`` puts ``a`` before ``b``, so ``a`` is a candidate
        only once ``b`` has left.

    Returns
    -------
    np.ndarray
        The order, causes first.
    """
    remaining = list(range(unit.shape[1]))
    leaves: list[int] = []
    while len(remaining) > 1:
        values = criterion(unit[:, remaining])
        candidates = [
            (tie_key(float(values[position])), k)
            for position, k in enumerate(remaining)
            if precedence is None or not precedence[k, remaining].any()
        ]
        value, leaf = min(candidates)
        _log.debug("%d remaining: leaf %d, criterion %.9g", len(remaining), leaf, value)
        remaining.remove(leaf)
        leaves.append(leaf)
    return np.array([*remaining, *reversed(leaves)], dtype=np.int64)


def jacobian_variances(unit: np.ndarray) -> np.ndarray:
    """The variance over the rows of each diagonal entry of the estimated Jacobian of the score."""
    return jacobian_diagonal(unit, method="score_matching").var(axis=0)


@dataclass(frozen=True)
class ScoreMatchingResult:
    """A SCORE fit: the DAG after the pruning and the causal order, causes first."""

    dag: np.ndarray
    order: np.ndarray


def score_matching(
    X: np.ndarray,
    *,
    alpha: float = ALPHA,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> ScoreMatchingResult:
    """Fit SCORE to the ``(n, d)`` matrix ``X``.

    ``precedence`` goes to :func:`leaf_order`; a ``forbidden`` edge is in no pruning fit and never
    in the result, and a ``required`` one is never removed. Ties go to the earlier column, so a
    caller that orders the columns by rank breaks them by rank.
    """
    unit = (X - X.mean(axis=0)) * unit_factors(X, standardize=True)
    order = leaf_order(unit, jacobian_variances, precedence)
    dag = prune_order(X, order, cutoff=alpha, k=K, forbidden=forbidden, required=required)
    return ScoreMatchingResult(dag=dag, order=order)
