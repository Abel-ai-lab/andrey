"""Location-scale noise model (LOCI) fit of one direction of a cause-effect pair.

Under ``effect = f(cause) + g(cause) noise`` with ``g > 0`` and the noise independent of the cause,
LOCI models ``effect | cause`` as a Gaussian in its natural parameters, ``eta_1 = Phi w_1`` and
``eta_2 = -Phi w_2`` with ``Phi`` a B-spline basis of the cause and ``w_2 >= 0``, so that
``eta_2 < 0`` (Immer et al. 2023, Eq. 3). The mean is ``mu = -eta_1 / (2 eta_2)`` and the variance
``sigma^2 = -1 / (2 eta_2)``. The estimate maximizes the log-likelihood plus the log density of a
Gaussian prior ``N(0, I / delta)`` on ``(w_1, w_2)`` (Supplementary Material A), which is strictly
concave (Lemma 1), so it has one maximizer.

The paper reaches it by a closed-form update of ``w_1`` and L-BFGS steps on ``w_2`` in turn. Here
each step is a projected Newton step on both weight vectors at once (Bertsekas 1982): the Newton
direction over the weights not held at the bound ``w_2 = 0``, then a backtracking search along
its projection onto ``w_2 >= 0``. Both reach the same maximizer, Newton in far fewer steps, as the
mean and the variance weights are strongly coupled. The fit stops once the mean objective changes
by less than ``1e-6`` in a step, or after 100 steps (Supplementary Material D).

References
----------
Immer, Schultheiss, Vogt, Scholkopf, Buhlmann, and Marx (2023). On the identifiability and
estimation of causal location-scale noise models. ICML, PMLR 202.

Eilers and Marx (1996). Flexible smoothing with B-splines and penalties. Statistical Science 11(2).

Bertsekas (1982). Projected Newton methods for optimization problems with simple constraints.
SIAM Journal on Control and Optimization 20(2).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import SplineTransformer

#: The B-spline basis of Supplementary Material D: 25 knots and order 5, which the authors' code
#: passes to scikit-learn's ``SplineTransformer`` as ``degree=5``. Its defaults place the knots
#: uniformly over the cause's range, extend each end spline as a constant, and keep every spline.
_KNOTS = 25
_DEGREE = 5
#: The prior precision ``delta`` on both weight vectors (Supplementary Material A).
_PRIOR_PRECISION = 1e-6
#: The most steps, and the change of the mean objective in a step below which the fit stops
#: (Supplementary Material D).
_MAX_STEPS = 100
_TOLERANCE = 1e-6
#: The largest distance from the bound at which a weight of ``w_2`` whose gradient points out of
#: the feasible set is held there (Bertsekas 1982, the epsilon-active set).
_ACTIVE_DISTANCE = 1e-3
#: The sufficient-decrease constant of the backtracking search, and the smallest step it tries.
_ARMIJO = 1e-4
_MIN_STEP = 1e-10

_HALF_LOG_TWO_PI = 0.5 * math.log(2 * math.pi)


def _log_likelihoods(eta_1: np.ndarray, neg_eta_2: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Each row's Gaussian log density of ``effect`` at natural parameters ``(eta_1, -neg_eta_2)``.

    ``log p = eta_1 y + eta_2 y^2 + eta_1^2 / (4 eta_2) + log sqrt(-2 eta_2) - log sqrt(2 pi)``
    (Immer et al. 2023, Eq. 2).
    """
    return (
        eta_1 * effect
        - neg_eta_2 * effect**2
        - eta_1**2 / (4 * neg_eta_2)
        + 0.5 * np.log(2 * neg_eta_2)
        - _HALF_LOG_TWO_PI
    )


def _objective(weights: np.ndarray, features: np.ndarray, effect: np.ndarray) -> float:
    """The mean over the rows of the negative log-likelihood plus ``delta / 2 |weights|^2``.

    ``weights`` stacks ``w_1`` and ``w_2``. Weights that give a row ``eta_2 >= 0``, outside the
    model, give ``inf``.
    """
    dim = features.shape[1]
    neg_eta_2 = features @ weights[dim:]
    if not neg_eta_2.min() > 0:
        return math.inf
    nll = -_log_likelihoods(features @ weights[:dim], neg_eta_2, effect).sum()
    return float((nll + 0.5 * _PRIOR_PRECISION * (weights @ weights)) / effect.shape[0])


def _gradient_hessian(
    weights: np.ndarray, features: np.ndarray, effect: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The gradient and Hessian of :func:`_objective` in ``(w_1, w_2)``.

    With ``q = -eta_2``, a row's negative log density has the derivatives
    ``d/d eta_1 = eta_1 / (2 q) - y``, ``d/dq = y^2 - eta_1^2 / (4 q^2) - 1 / (2 q)``,
    ``d2/d eta_1^2 = 1 / (2 q)``, ``d2/(d eta_1 dq) = -eta_1 / (2 q^2)``, and
    ``d2/dq^2 = eta_1^2 / (2 q^3) + 1 / (2 q^2)``; both natural parameters are linear in the weights
    through ``Phi``.
    """
    n, dim = features.shape
    eta_1 = features @ weights[:dim]
    q = features @ weights[dim:]
    slopes = (eta_1 / (2 * q) - effect, effect**2 - eta_1**2 / (4 * q**2) - 0.5 / q)
    gradient = np.concatenate([features.T @ s for s in slopes]) + _PRIOR_PRECISION * weights
    blocks = [
        features.T @ (features * c[:, None])
        for c in (0.5 / q, -eta_1 / (2 * q**2), eta_1**2 / (2 * q**3) + 0.5 / q**2)
    ]
    hessian = np.block([[blocks[0], blocks[1]], [blocks[1], blocks[2]]])
    hessian[np.diag_indices_from(hessian)] += _PRIOR_PRECISION
    return gradient / n, hessian / n


def _maximize(features: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """The weights ``(w_1, w_2)`` that minimize :func:`_objective`, by projected Newton steps.

    The start is ``w_1 = 0`` and ``w_2 = 1``, that of the authors' code, where every row has
    ``eta_2 < 0``; the backtracking search never leaves the model, so every iterate keeps it.
    """
    dim = features.shape[1]
    weights = np.concatenate([np.zeros(dim), np.ones(dim)])
    value = _objective(weights, features, effect)
    bounded = np.arange(2 * dim) >= dim
    for _ in range(_MAX_STEPS):
        gradient, hessian = _gradient_hessian(weights, features, effect)
        projected = weights - gradient
        projected[bounded] = np.maximum(projected[bounded], 0.0)
        distance = min(_ACTIVE_DISTANCE, float(np.linalg.norm(weights - projected)))
        free = ~(bounded & (weights <= distance) & (gradient > 0))
        direction = np.zeros_like(weights)
        direction[free] = -cho_solve(cho_factor(hessian[np.ix_(free, free)]), gradient[free])
        step = 1.0
        while step >= _MIN_STEP:
            trial = weights + step * direction
            trial[bounded] = np.maximum(trial[bounded], 0.0)
            trial_value = _objective(trial, features, effect)
            if trial_value <= value + _ARMIJO * float(gradient @ (trial - weights)):
                break
            step /= 2
        else:
            return weights
        change = value - trial_value
        weights, value = trial, trial_value
        if change < _TOLERANCE:
            break
    return weights


def fit_direction(cause: np.ndarray, effect: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit the location-scale model ``effect | cause`` and return its mean log-likelihood and the
    scaled residual ``(effect - mu(cause)) / sigma(cause)``.

    ``cause`` and ``effect`` are 1-D arrays of equal length, standardized by the caller. The mean
    log-likelihood is that of ``effect`` given ``cause`` over the rows, without the prior's term.
    """
    transformer = SplineTransformer(n_knots=_KNOTS, degree=_DEGREE)
    # B-splines are nonnegative, as the feature map of eta_2 must be; the absolute value removes the
    # sign of a zero or of a rounding below it.
    features = np.abs(transformer.fit_transform(cause.reshape(-1, 1)))
    dim = features.shape[1]
    weights = _maximize(features, effect)
    eta_1 = features @ weights[:dim]
    neg_eta_2 = features @ weights[dim:]
    log_likelihood = float(_log_likelihoods(eta_1, neg_eta_2, effect).mean())
    variance = 0.5 / neg_eta_2
    return log_likelihood, (effect - eta_1 * variance) / np.sqrt(variance)
