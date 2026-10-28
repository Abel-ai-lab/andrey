"""VARMA-LiNGAM, the engine behind :func:`andrey.varma_lingam`.

The algorithm, its settings, and its references are described on :func:`andrey.varma_lingam`.

The reduced-form VARMA(p, q) ``x(t) = sum_i Phi_i x(t-i) + e(t) + sum_j Theta_j e(t-j)`` is fit
by Gaussian maximum likelihood (a Kalman filter), and DirectLiNGAM on the innovations, with
least-squares pairwise slopes, gives the causal order and the instantaneous matrix ``B0``. The
structural matrices are ``psi[0] = B0``, ``psi[tau] = (I - B0) Phi_tau``, and
``omega[j] = (I - B0) Theta_j``, or ``(I - B0) Theta_j (I - B0)^-1`` with ``structural_ma=True``;
``[i, j]`` weights the influence of ``j`` on ``i``. With ``prune=True`` all structural matrices
are re-estimated on the causal order: adaptive Lasso selects each variable's regressors and
ordinary least squares refits their weights.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResultsWrapper

from andrey.core.knowledge import KnowledgeError
from andrey.core.warning_policy import warn_search_limit
from andrey.lingam.adjacency import estimate, predict_adaptive_lasso, predict_with_required
from andrey.lingam.direct import direct_lingam
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import LaggedConstraints

_CRITERIA = ("aic", "bic", "hqic")


def varma_lingam(
    X: np.ndarray,
    *,
    order: tuple[int, int] = (1, 1),
    criterion: str | None = None,
    prune: bool = False,
    max_iter: int = 100,
    structural_ma: bool = False,
    constraints: LaggedConstraints | None = None,
    name: Callable[[int], str] = str,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Fit VARMA-LiNGAM: return ``(causal_order, psis, omegas)``.

    ``X`` is a ``(n_samples, n)`` time series (row = time step). ``order`` is ``(p, q)``, the AR and
    MA lag counts; when ``criterion`` is one of ``"aic"``, ``"bic"``, ``"hqic"`` the best ``(p, q)``
    over ``0 <= p <= order[0]``, ``0 <= q <= order[1]`` (excluding ``(0, 0)``) is selected by that
    information criterion of the fitted model (``criterion=None`` uses ``order`` directly).

    ``causal_order`` is the instantaneous DirectLiNGAM ordering. ``psis`` has shape
    ``(1 + p, n, n)``: index 0 is the instantaneous ``B0`` and index ``tau`` in ``1..p`` is the
    structural lag-``tau`` autoregressive matrix. ``omegas`` has shape ``(q, n, n)``: index ``j`` is
    the structural moving-average matrix for innovation lag ``j + 1``. In both, entry ``[i, j]`` is
    the effect of source ``j`` on target ``i``.

    With ``prune=True`` the matrices are sparse: adaptive Lasso selects the regressors and least
    squares refits their weights; otherwise they are the dense algebraic transform of the
    reduced-form coefficients. ``structural_ma=True`` applies
    the paper-exact moving-average transform ``(I - B0) Theta_j (I - B0)^-1``, which recovers the
    true structural MA influence on the independent innovations; the default
    ``structural_ma=False`` uses ``(I - B0) Theta_j``.

    The fit runs on the columns sorted by :func:`~andrey.search.hc.tie_ranks`, so reordering the
    columns of ``X`` relabels the result and changes nothing else, unless two columns are identical.
    The likelihood fit, which can stop before it converges, then starts from and stops at the same
    point for every column order.

    With background knowledge ``constraints`` over the autoregressive lags ``0`` to ``order[0]``,
    DirectLiNGAM on the innovations honors the lag-0 part, and with ``prune=True`` each
    variable's regression drops its forbidden regressors and keeps its required ones unpenalized
    (:func:`_prune`). Raises :class:`~andrey.core.knowledge.KnowledgeError` when ``prune=False``
    and a lagged link is forbidden, since the dense matrices keep every lagged weight, or when the
    selected ``p`` is below the lag of a required link; ``name`` names a column in the message.
    """
    X = np.asarray(X, dtype=np.float64)
    by_rank = np.argsort(tie_ranks(X))  # the fit's column k is the data's column by_rank[k]
    back = np.argsort(by_rank)
    X = np.ascontiguousarray(X[:, by_rank])
    if constraints is not None:
        constraints = constraints.permuted(by_rank)
        link = constraints.lagged_forbidden()
        if not prune and link is not None:
            cause, effect = name(int(by_rank[link[0]])), name(int(by_rank[link[1]]))
            raise KnowledgeError(
                f"knowledge forbids the link {cause} -> {effect} at lag {link[2]}, but "
                "varma_lingam with prune=False keeps every lagged weight; use prune=True"
            )

    if criterion is None:
        p, q = int(order[0]), int(order[1])
        phis, thetas, residuals = _fit_varma(X, p, q, max_iter)
    else:
        p, q, phis, thetas, residuals = _select_order(X, order, criterion, max_iter)

    if constraints is None:
        causal_order, b0 = direct_lingam(residuals)
    else:
        beyond = np.argwhere(constraints.required[p + 1 :])
        if len(beyond):
            lag, i, j = beyond[0].tolist()
            raise KnowledgeError(
                f"knowledge requires the link {name(int(by_rank[i]))} -> {name(int(by_rank[j]))} "
                f"at lag {lag + p + 1}, but criterion={criterion!r} selected p={p}"
            )
        causal_order, b0 = direct_lingam(residuals, constraints=constraints.contemporaneous())

    if prune:
        psis, omegas = _prune(X, residuals, causal_order, p, q, constraints)
    else:
        psis, omegas = _structural_matrices(b0, phis, thetas, p, q, structural_ma)

    causal_order = [int(by_rank[i]) for i in causal_order]
    return causal_order, psis[:, back][:, :, back], omegas[:, back][:, :, back]


def _fit_varma(
    X: np.ndarray, p: int, q: int, max_iter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum-likelihood VARMA(p, q) fit; return ``(phis, thetas, residuals)``.

    ``phis`` has shape ``(p, n, n)`` (``phis[i]`` the lag-``i+1`` AR matrix), ``thetas`` shape
    ``(q, n, n)``, and ``residuals`` the one-step innovations ``e(t)`` (shape ``(n_samples, n)``).
    """
    n = X.shape[1]
    fitted = _fit_varmax(X, p, q, max_iter)

    phis = _coef_stack(fitted.coefficient_matrices_var, p, n)
    thetas = _coef_stack(fitted.coefficient_matrices_vma, q, n)
    residuals = np.asarray(fitted.resid, dtype=np.float64)
    return phis, thetas, residuals


def _fit_varmax(X: np.ndarray, p: int, q: int, max_iter: int) -> VARMAXResultsWrapper:
    """The state-space VARMA(p, q) fit with a constant trend, skipping the parameter covariance.

    statsmodels' own warnings are silenced; a fit that takes ``max_iter`` optimizer iterations
    without converging warns with :class:`~andrey.SearchLimitWarning`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VARMAX(X, order=(p, q), trend="c")
        fitted = model.fit(maxiter=max_iter, disp=False, cov_type="none")
    retvals = fitted.mle_retvals or {}
    if not retvals.get("converged", True) and retvals.get("iterations", 0) >= max_iter:
        warn_search_limit("varma_lingam", "max_iter", max_iter, "likelihood iterations")
    return fitted


def _coef_stack(matrices: np.ndarray | None, n_lags: int, n: int) -> np.ndarray:
    """Normalize a state-space coefficient stack to shape ``(n_lags, n, n)`` (empty if none)."""
    if n_lags == 0 or matrices is None:
        return np.zeros((n_lags, n, n), dtype=np.float64)
    return np.asarray(matrices, dtype=np.float64).reshape(n_lags, n, n)


def _structural_matrices(
    b0: np.ndarray, phis: np.ndarray, thetas: np.ndarray, p: int, q: int, structural_ma: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Dense structural matrices from ``B0`` and the reduced-form AR/MA coefficients.

    ``psis[0] = B0`` and ``psis[tau] = (I - B0) Phi_tau``. The moving-average blocks are
    ``omegas[j] = (I - B0) Theta_j (I - B0)^-1`` when ``structural_ma`` else ``(I - B0) Theta_j``.
    """
    n = b0.shape[0]
    transform = np.eye(n, dtype=np.float64) - b0
    psis = np.stack([b0, *(transform @ phis[i] for i in range(p))])
    if q == 0:
        return psis, np.zeros((0, n, n), dtype=np.float64)
    if structural_ma:
        inv = np.linalg.inv(transform)
        omegas = np.stack([transform @ thetas[j] @ inv for j in range(q)])
    else:
        omegas = np.stack([transform @ thetas[j] for j in range(q)])
    return psis, omegas


def _prune(
    X: np.ndarray,
    residuals: np.ndarray,
    causal_order: list[int],
    p: int,
    q: int,
    constraints: LaggedConstraints | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse structural matrices on the causal order: adaptive-Lasso selection, then OLS refit.

    Each variable ``x_i(t)`` is regressed on the contemporaneous values of its causal predecessors,
    all ``p`` autoregressive lags of every variable, and all ``q`` lags of the independent
    innovations ``n(t) = (I - B0) e(t)``, with ``B0`` the least-squares refit of the residuals on
    the causal order. Only time steps ``t >= max(p, q)`` are regression rows, so every lag refers
    to an earlier step of the series. Returns ``(psis, omegas)`` with ``psis`` of shape
    ``(1 + p, n, n)`` (index 0 the instantaneous block) and ``omegas`` of shape ``(q, n, n)``.

    With ``constraints``, the lag-0 refit and each regression leave out the contemporaneous and
    autoregressive regressors they forbid; the ones they require are not penalized
    (:func:`~andrey.lingam.adjacency.predict_with_required`). The moving-average regressors are
    not constrained.
    """
    n = X.shape[1]
    s = max(p, q)
    Xc = X - X.mean(axis=0)

    if constraints is None:
        b0 = estimate(residuals, list(causal_order), refit=True)
    else:
        lag0 = constraints.contemporaneous()
        b0 = estimate(
            residuals,
            list(causal_order),
            refit=True,
            forbidden=lag0.forbidden,
            required=lag0.required,
        )
    n_samples = X.shape[0]
    innov = np.zeros((n_samples, n), dtype=np.float64)
    innov[n_samples - residuals.shape[0] :] = residuals @ (np.eye(n) - b0).T

    rows = np.arange(s, n_samples)
    blocks = [Xc[rows]]  # contemporaneous block
    blocks += [Xc[rows - i] for i in range(1, p + 1)]  # AR lags
    blocks += [innov[rows - j] for j in range(1, q + 1)]  # MA innovation lags
    design = np.concatenate(blocks, axis=1)

    width = n * (1 + p + q)
    coefs = np.zeros((n, width), dtype=np.float64)
    order = list(causal_order)
    for pos, target in enumerate(order):
        later = set(order[pos:])  # self and causally-later contemporaneous columns
        predictors = [j for j in range(width) if not (j < n and j in later)]
        if constraints is None:
            coefs[target, predictors] = predict_adaptive_lasso(
                design, predictors, target, refit=True
            )
            continue

        def known(part: np.ndarray, column: int) -> bool:
            lag, source = divmod(column, n)
            return lag <= p and bool(part[lag, source, target])

        free = [j for j in predictors if not known(constraints.forbidden, j)]
        forced = [j for j in free if known(constraints.required, j)]
        free = [j for j in free if j not in forced]
        if forced:
            coefs[target, forced], coefs[target, free] = predict_with_required(
                design, forced, free, target
            )
        elif free:
            coefs[target, free] = predict_adaptive_lasso(design, free, target, refit=True)

    psis = np.stack([coefs[:, n * k : n * (k + 1)] for k in range(1 + p)])
    omegas = (
        np.stack([coefs[:, n * (1 + p + j) : n * (1 + p + j + 1)] for j in range(q)])
        if q
        else np.zeros((0, n, n), dtype=np.float64)
    )
    return psis, omegas


def _select_order(
    X: np.ndarray, order: tuple[int, int], criterion: str, max_iter: int
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Select ``(p, q)`` by an information criterion of the fitted VARMA over the ``order`` grid.

    Returns ``(p, q, phis, thetas, residuals)`` for the chosen order. Candidates are all
    ``0 <= p <= order[0]``, ``0 <= q <= order[1]`` except ``(0, 0)``; the criterion (``"aic"``,
    ``"bic"``, ``"hqic"``) is read from the fitted maximum-likelihood model.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA} or None, got {criterion!r}")
    best = None
    for p in range(0, order[0] + 1):
        for q in range(0, order[1] + 1):
            if p == 0 and q == 0:
                continue
            fitted = _fit_varmax(X, p, q, max_iter)
            value = float(getattr(fitted, criterion))
            if best is None or value < best[0]:
                best = (value, p, q, fitted)

    if best is None:
        raise ValueError(f"no (p, q) candidates in order grid {order}; need order != (0, 0)")
    _, p, q, fitted = best
    n = X.shape[1]
    phis = _coef_stack(fitted.coefficient_matrices_var, p, n)
    thetas = _coef_stack(fitted.coefficient_matrices_vma, q, n)
    residuals = np.asarray(fitted.resid, dtype=np.float64)
    return p, q, phis, thetas, residuals
