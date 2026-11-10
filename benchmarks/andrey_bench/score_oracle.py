"""SCORE's leaf rule on the exact Hessian of the true model: the oracle arm of the Stein-score screen.

SCORE (Rolland et al. 2022, Lemma 1) takes as the next leaf the variable whose diagonal entry of
the Hessian of ``log p`` varies least over the rows. Under Gaussian additive noise a leaf's entry
is constant, so the rule is exact; under other noise it need not be. SCORE estimates the Hessian
from the data. This arm evaluates the rule on the Hessian of the true density instead, so a
screen cell can tell an estimator's failure (oracle right, estimate wrong) from the setting's
(oracle wrong too). It reads the generator's parameters by design: a harness diagnostic, never a
method, and it is not timed.

The data are ``andrey.data``'s additive-noise model, ``x_j = sum_p w_pj g(x_p) + e_j`` with
``g(z) = z + sin(z)``, where the Hessian has a closed form. With ``r_j`` the residual of ``x_j`` and
``l_j`` the log density of ``e_j``,

    d^2 log p / dx_k^2 = l_k''(r_k)
        + sum over children j of k of [ l_j''(r_j) (w_kj g'(x_k))^2 - l_j'(r_j) w_kj g''(x_k) ].

Removing a leaf leaves the additive-noise model of the other variables, so the formula holds on
every remaining set that keeps each variable's parents. The rule runs along that path: at each step
it picks the remaining variable whose entry has the least variance over the rows (SCORE's rule, no
division by the mean, as the authors' ``SCORE`` runs it); a pick that is not a leaf of the
remaining variables is a wrong step, and the leaf with the least variance is removed in its place,
so the next step's set again keeps every parent. :func:`oracle_steps` returns the number of wrong
steps of ``d - 1``; none means the rule finds a correct order on the true Hessian.

A z-scored column ``x_k / c_k`` has the entry ``c_k^2`` times the raw one at the same row, so for
``standardize="standardized"`` the entries are scaled before the rule reads them.
"""

from __future__ import annotations

import numpy as np

import andrey.data as data
from andrey_bench import datasets
from andrey_bench.datasets import DatasetKey

_EULER_GAMMA = 0.5772156649015329
_GUMBEL_STD = np.pi / np.sqrt(6.0)


def _log_density_derivatives(noise: str, r: np.ndarray, sigma: float):
    """``(l', l'')`` at the residuals `r` of noise ``sigma z``, ``z`` of ``andrey.data``'s family.

    The families have mean 0 and variance 1: Gaussian; Laplace of scale ``1 / sqrt(2)``, whose
    ``l''`` is 0 away from 0; Gumbel of scale ``sqrt(6) / pi``, centered.
    """
    if noise == "gaussian":
        return -r / sigma**2, np.full_like(r, -1.0 / sigma**2)
    if noise == "laplace":
        b = sigma / np.sqrt(2.0)
        return -np.sign(r) / b, np.zeros_like(r)
    if noise == "gumbel":
        beta = sigma / _GUMBEL_STD
        u = np.exp(-(r + beta * _EULER_GAMMA) / beta)
        return (u - 1.0) / beta, -u / beta**2
    raise ValueError(f"no closed-form log density for {noise!r} noise")


def hessian_diagonal(
    x: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    sigmas: np.ndarray,
    noise: str,
    remaining: list[int],
) -> np.ndarray:
    """The diagonal of the Hessian of ``log p`` of the `remaining` variables at every row.

    `x` holds the raw values, ``edges[k] = (parent, child)`` with weight ``weights[k]``, and
    ``sigmas[j]`` is ``e_j``'s scale. `remaining` must keep every parent of its variables. Returns
    an ``(n, len(remaining))`` array, column ``i`` for ``remaining[i]``.
    """
    d = x.shape[1]
    w = np.zeros((d, d))
    w[edges[:, 0], edges[:, 1]] = weights
    residual = x - (x + np.sin(x)) @ w
    first, second = {}, {}
    for j in remaining:
        first[j], second[j] = _log_density_derivatives(noise, residual[:, j], float(sigmas[j]))
    out = np.empty((x.shape[0], len(remaining)))
    for i, k in enumerate(remaining):
        dg, d2g = 1.0 + np.cos(x[:, k]), -np.sin(x[:, k])
        entry = second[k].copy()
        for j in remaining:
            if w[k, j]:
                entry += second[j] * (w[k, j] * dg) ** 2 - first[j] * w[k, j] * d2g
        out[:, i] = entry
    return out


def oracle_steps(key: DatasetKey) -> dict[str, object]:
    """Run SCORE's rule on the exact Hessian of `key`'s model; see the module docstring.

    `key` is an ``"anm"`` key without hidden nodes, at raw scale or z-scored. The model is drawn
    again at raw scale from the key's own generator call, and checked against the dataset the
    solutions fit. Returns the dataset id and label, ``steps`` (``d - 1``), and ``wrong_steps``.
    """
    if key.functional != "anm" or key.latents or key.standardize not in ("raw", "standardized"):
        raise ValueError(f"{key.label}: the oracle takes additive-noise data, raw or z-scored")
    call = datasets.generation_kwargs(key)
    ds = data.sample_scm(**{**call, "scale": "raw"})  # ty: ignore[invalid-argument-type]
    x = np.asarray(ds.data, dtype=np.float64)
    fitted, _, _, _ = datasets.generate_one(key)
    seen = x if key.standardize == "raw" else (x - x.mean(axis=0)) / x.std(axis=0)
    if not np.allclose(seen, fitted, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{key.label}: the redrawn model does not give the fitted data")
    params = ds.params
    scale = np.ones(x.shape[1]) if key.standardize == "raw" else x.std(axis=0)
    children = {k: set(params.edges[params.edges[:, 0] == k, 1].tolist()) for k in range(key.d)}
    remaining = list(range(key.d))
    wrong = 0
    while len(remaining) > 1:
        entries = hessian_diagonal(
            x, params.edges, params.weights, params.noise_scales, params.noise, remaining
        )
        spread = np.var(entries * scale[remaining] ** 2, axis=0)
        leaves = [i for i, k in enumerate(remaining) if not children[k] & set(remaining)]
        pick = int(np.argmin(spread))
        if pick not in leaves:
            wrong += 1
            pick = min(leaves, key=lambda i: spread[i])
        remaining.pop(pick)
    return {
        "dataset_id": key.digest,
        "label": key.label,
        "noise": key.noise,
        "standardize": key.standardize,
        "d": key.d,
        "seed": key.seed,
        "steps": key.d - 1,
        "wrong_steps": wrong,
    }
