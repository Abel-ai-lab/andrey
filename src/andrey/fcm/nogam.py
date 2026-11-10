"""NoGAM: a causal order from how well residuals predict the score (Montagna et al. 2023).

NoGAM assumes a nonlinear additive noise model, ``x_j = f_j(parents) + e_j``, with noise of any
distribution. For a leaf ``j`` the score entry ``d log p(x) / d x_j`` is a function of the leaf's
noise alone, and that noise is, in the limit, the residual of the regression of ``x_j`` on all the
other variables (the paper's Section 4): so the score entry of a leaf can be predicted from its
residual without error, and that of another variable cannot. The method runs two phases:

1. **Causal order** (Algorithm 1). On the remaining variables, the score is estimated at every row
   by :func:`andrey.fcm.stein.score`; each variable is regressed on all the others by kernel ridge
   regression, which gives its residual; and each score entry is regressed on its variable's
   residual the same way. The variable whose score entry has the smallest mean squared prediction
   error is the next leaf: it leaves the set, and the three estimates are made again on the
   others. The causal order is the leaves in reverse.
2. **Pruning.** The complete DAG of the order is pruned by the smooth-term tests of CAM,
   :func:`andrey.fcm.pruning.prune_order`, at the cutoff ``0.001`` of the paper's experiments.

The settings are those of the paper's experiments (its Section 6): kernel ridge regression with the
kernel ``exp(-gamma ||x - y||^2)`` at :data:`GAMMA`, ``0.1``, and the ridge term :data:`RIDGE`,
``0.01``, with no intercept, and predictions by K-fold cross-validation, each row predicted by the
model fit on the other folds.

The choices the paper leaves open:

- The folds: :data:`FOLDS`, 5, each a block of consecutive rows, the first ``n mod 5`` one row
  longer.
- The residuals are predicted across the same folds as the score entries. The paper states the
  cross-validation for the score entries.
- The score estimator's ridge term and bandwidth, as :mod:`andrey.fcm.stein` records them.
- The errors are compared rounded to ``KEY_DP`` decimal places, as :func:`andrey.hc` compares its
  score changes; a tie goes to the earlier column.
- The score estimator and the kernel ridge regressions read the columns' units, and
  :func:`andrey.nogam` passes the columns centered and divided by their standard deviations.

The fit draws no random numbers.

References
----------
Montagna, Noceti, Rosasco, Zhang, Locatello. "Causal discovery with score matching on additive
models with arbitrary noise." CLeaR, PMLR 213:726-751, 2023.

Buhlmann, Peters, Ernest. "CAM: Causal additive models, high-dimensional order search and
penalized regression." Annals of Statistics 42(6):2526-2556, 2014.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from andrey.core.units import unit_factors
from andrey.fcm.pruning import prune_order
from andrey.fcm.score_matching import leaf_order
from andrey.fcm.stein import score, squared_distances

#: Cutoff of the pruning tests.
ALPHA = 0.001
#: Basis functions per smooth term of the pruning fits.
K = 10
#: Ridge term of the kernel ridge regressions.
RIDGE = 0.01
#: Their kernel is ``exp(-GAMMA ||x - y||^2)``.
GAMMA = 0.1
#: Folds of the cross-validated predictions.
FOLDS = 5


def cross_predict(kernel: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predict ``y`` at every row by kernel ridge regression fit on the other folds.

    Parameters
    ----------
    kernel : np.ndarray
        ``(n, n)`` kernel matrix of the rows' predictors.
    y : np.ndarray
        ``(n,)`` response.

    Returns
    -------
    np.ndarray
        ``(n,)``; each entry is the prediction of the model fit without the entry's fold.
    """
    n = y.shape[0]
    predicted = np.empty(n)
    held_out = np.zeros(n, dtype=bool)
    for fold in np.array_split(np.arange(n), FOLDS):
        held_out[:] = False
        held_out[fold] = True
        train = np.flatnonzero(~held_out)
        system = kernel[np.ix_(train, train)]
        system[np.diag_indices(train.size)] += RIDGE
        weights = cho_solve(cho_factor(system, lower=True, overwrite_a=True), y[train])
        predicted[fold] = kernel[np.ix_(fold, train)] @ weights
    return predicted


def residuals(unit: np.ndarray) -> np.ndarray:
    """Each column's residual from its cross-validated regression on all the other columns."""
    total = squared_distances(unit)
    out = np.empty_like(unit)
    for j in range(unit.shape[1]):
        column = unit[:, j]
        others = np.maximum(total - (column[:, None] - column[None, :]) ** 2, 0.0)
        out[:, j] = column - cross_predict(np.exp(-GAMMA * others), column)
    return out


def prediction_errors(unit: np.ndarray) -> np.ndarray:
    """The mean squared error of each score entry predicted from its variable's residual."""
    estimate = score(unit, method="nogam")
    residual = residuals(unit)
    errors = np.empty(unit.shape[1])
    for j in range(unit.shape[1]):
        r = residual[:, j]
        kernel = np.exp(-GAMMA * (r[:, None] - r[None, :]) ** 2)
        errors[j] = np.mean((estimate[:, j] - cross_predict(kernel, estimate[:, j])) ** 2)
    return errors


@dataclass(frozen=True)
class NoGAMResult:
    """A NoGAM fit: the DAG after the pruning and the causal order, causes first."""

    dag: np.ndarray
    order: np.ndarray


def nogam(
    X: np.ndarray,
    *,
    alpha: float = ALPHA,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> NoGAMResult:
    """Fit NoGAM to the ``(n, d)`` matrix ``X``.

    ``precedence`` goes to :func:`~andrey.fcm.score_matching.leaf_order`; a ``forbidden`` edge is
    in no pruning fit and never in the result, and a ``required`` one is never removed. Ties go to
    the earlier column, so a caller that orders the columns by rank breaks them by rank.
    """
    unit = (X - X.mean(axis=0)) * unit_factors(X, standardize=True)
    order = leaf_order(unit, prediction_errors, precedence)
    dag = prune_order(X, order, cutoff=alpha, k=K, forbidden=forbidden, required=required)
    return NoGAMResult(dag=dag, order=order)
