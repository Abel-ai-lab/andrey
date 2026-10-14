"""Scoring substrate for structure search: linear-Gaussian BIC, discrete BDeu, and mixed-data BICs.

The negative local BIC score (lower is better; proportional to ``-2 * BIC``) for a target node
given a parent set, evaluated from the sample covariance. Uses the linear-Gaussian BIC closed
form, branching on parent-set size for the scalar fast paths.

:func:`local_score_bic` is the pure per-node primitive; :class:`BICScore` precomputes the covariance
once and memoizes scores behind the :class:`Score` protocol that structure-search engines consume.
:class:`BDeuScore` scores category codes with the negative log BDeu marginal likelihood behind the
same protocol. :class:`DGScore` and :class:`CGScore` score mixed data, continuous and categorical
columns marked by a mask, with the BIC of the degenerate Gaussian and conditional Gaussian models
of :mod:`andrey.core.mixed`.

References
----------
Schwarz (1978). Estimating the Dimension of a Model. Annals of Statistics 6(2):461-464.
https://doi.org/10.1214/aos/1176344136

Heckerman, Geiger & Chickering (1995). Learning Bayesian networks: the combination of knowledge
and statistical data. Machine Learning 20(3):197-243. https://doi.org/10.1007/BF00994016

Andrews, Ramsey & Cooper (2018). Scoring Bayesian networks of mixed variables. International
Journal of Data Science and Analytics 6(1):3-18. https://doi.org/10.1007/s41060-017-0085-7

Andrews, Ramsey & Cooper (2019). Learning high-dimensional directed acyclic graphs with mixed
data-types. Proceedings of Machine Learning Research 104:4-21.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from . import categorical, stats
from .mixed import _LRU, ConditionalGaussian, DegenerateGaussian

#: The scores that read category codes, and those that read mixed data with a categorical mask.
DISCRETE_SCORES: tuple[str, ...] = ("local_score_BDeu",)
MIXED_SCORES: tuple[str, ...] = ("local_score_cond_gauss", "local_score_degen_gauss")


def local_score_bic(
    cov: np.ndarray,
    n: int,
    node: int,
    parents: Sequence[int],
    *,
    lambda_value: float = 1.0,
    log_n: float | None = None,
    cov_diag: np.ndarray | None = None,
) -> float:
    """Return the negative local BIC score for ``node`` given ``parents``.

    Evaluates ``n * log R(node | parents) + log_n * len(parents) * lambda_value``, where ``R`` is
    the linear-regression residual variance
    ``cov[node, node] - cov[node, parents] @ inv(cov[parents, parents]) @ cov[parents, node]``.
    Lower is better; the value is proportional to ``-2 * BIC``.

    Branches on ``len(parents)`` for scalar fast paths (0/1/2 parents); the general path uses
    ``np.linalg.solve`` with a ``pinv`` fallback on singular ``cov[parents, parents]``.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Sample covariance matrix (``np.cov(data, rowvar=False)``, ddof=1).
    n : int
        Sample count.
    node : int
        Target column index.
    parents : sequence of int
        Parent column indices.
    lambda_value : float
        Weight on the BIC complexity term.
    log_n : float, optional
        Precomputed ``log(n)``; computed from ``n`` when omitted.
    cov_diag : ndarray, optional
        Precomputed diagonal of ``cov``; computed from ``cov`` when omitted.
    """
    if log_n is None:
        log_n = float(np.log(n))
    if cov_diag is None:
        cov_diag = np.diag(cov)
    node = int(node)
    n_parents = len(parents)

    if n_parents == 0:
        return float(n * np.log(cov_diag[node]))

    if n_parents == 1:
        parent = int(parents[0])
        parent_var = float(cov_diag[parent])
        if parent_var != 0.0:
            cov_ip = float(cov[node, parent])
            residual = float(cov_diag[node]) - cov_ip * (cov_ip / parent_var)
            return float(n * np.log(residual) + log_n * lambda_value)
        yX = cov[node : node + 1, parent : parent + 1]
        XX = cov[parent : parent + 1, parent : parent + 1]
        beta = _solve(XX, yX.T)
        H = float(np.log(cov_diag[node] - (yX @ beta).item()))
        return float(n * H + log_n * lambda_value)

    if n_parents == 2:
        parent0 = int(parents[0])
        parent1 = int(parents[1])
        yX = np.array([[cov[node, parent0], cov[node, parent1]]])
        XX = np.array(
            [
                [cov[parent0, parent0], cov[parent0, parent1]],
                [cov[parent1, parent0], cov[parent1, parent1]],
            ]
        )
        beta = _solve(XX, yX.T)
        H = float(np.log(cov_diag[node] - (yX @ beta).item()))
        return float(n * H + log_n * 2 * lambda_value)

    pa = [int(p) for p in parents]
    yX = cov[node : node + 1, pa]
    XX = cov[pa][:, pa]
    beta = _solve(XX, yX.T)
    H = np.log(cov_diag[node] - (yX @ beta).item())
    return float(n * H + log_n * n_parents * lambda_value)


def penalty_weight(lambda_value: float) -> float:
    """Return the BIC penalty weight ``lambda_value`` as a float, checked to be finite and >= 0.

    Raises ``ValueError`` naming ``lambda_value`` otherwise. The searches check it under BDeu too,
    which ignores it, so a setting is valid or not whatever the score.
    """
    value = float(lambda_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"lambda_value must be finite and non-negative, got {value}")
    return value


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a @ x = b``, falling back to the pseudo-inverse when ``a`` is singular."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


@runtime_checkable
class Score(Protocol):
    """Contract for a scoring object: the negative local score of a node given its parents."""

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the negative local score for ``node`` given ``parents`` (lower is better)."""
        ...


class BICScore:
    """Linear-Gaussian BIC score over a fixed dataset.

    Precomputes the covariance, sample count, and ``log(n)``. Memoizes negative local scores.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; rows are observations, columns are variables.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term of the deviance the search minimizes.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={data.ndim}"
            )
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or Inf; cannot compute BIC scores")
        # np.cov of a single column is a 0-d variance; keep the (1, 1) matrix the score reads.
        cov = np.atleast_2d(stats.cov(data, rowvar=False))
        self._init(cov, data.shape[0], lambda_value, cache_size)

    @classmethod
    def from_cov(
        cls,
        cov: np.ndarray,
        n: int,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> BICScore:
        """Build a scorer from a precomputed covariance.

        Pooled workers use the parent's ``cov`` bytes, giving every process identical inputs to
        :func:`local_score_bic` and bit-identical scores on the same host BLAS.

        Parameters
        ----------
        cov : ndarray, shape (d, d)
            Sample covariance matrix; must be square and finite.
        n : int
            Sample count the BIC complexity penalty uses.
        lambda_value : float, default=1.0
            Weight on the BIC complexity term of the deviance the search minimizes.
        cache_size : int or None, default=1_000_000
            Memoization cache bound; ``None`` is unbounded.

        Raises
        ------
        ValueError
            If ``cov`` is not a square 2-D matrix, or holds ``NaN`` / ``inf``.
        """
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be a square 2-D matrix, got shape {cov.shape}")
        if not np.isfinite(cov).all():
            raise ValueError("cov contains NaN or Inf; cannot compute BIC scores")
        obj = cls.__new__(cls)
        obj._init(cov, int(n), lambda_value, cache_size)
        return obj

    def _init(self, cov: np.ndarray, n: int, lambda_value: float, cache_size: int | None) -> None:
        lambda_value = penalty_weight(lambda_value)
        self.cov = cov
        self.n = n
        self.lambda_value = lambda_value
        self.log_n = float(np.log(self.n))
        self._cov_diag = np.diag(self.cov)
        self._score_key = lru_cache(maxsize=cache_size)(self._score_key_impl)

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        return local_score_bic(
            self.cov,
            self.n,
            node,
            parents,
            lambda_value=self.lambda_value,
            log_n=self.log_n,
            cov_diag=self._cov_diag,
        )

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized negative local BIC score for ``node`` given ``parents``.

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = tuple(sorted(int(p) for p in parents))
        return self._score_key(int(node), key)

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the negative local BIC scores for many parent sets, in request order."""
        return [self.score(node, parents) for parents in parent_sets]


class BDeuScore:
    """Negative local BDeu score over category codes.

    The score of ``node`` given ``parents`` is minus the log of the structure prior times the BDeu
    marginal likelihood [1]_, so lower is better as for :class:`BICScore`. With ``r`` the node's
    categories, ``q`` the product of its parents' categories, ``N_ij`` the rows in parent
    configuration ``j``, and ``N_ijk`` those of them where the node takes value ``k``, the log
    marginal likelihood is

    ``sum_j [lgamma(a / q) - lgamma(N_ij + a / q)
    + sum_k (lgamma(N_ijk + a / (r q)) - lgamma(a / (r q)))]``

    with ``a`` the ``sample_prior`` (the equivalent sample size). A configuration or cell no row
    falls in adds zero, so only the occurring ones are summed. The structure prior gives each of
    the other ``d - 1`` variables probability ``p = structure_prior / (d - 1)`` of being a parent:
    ``k log p + (d - 1 - k) log(1 - p)`` for ``k`` parents. It depends only on the number of edges,
    so the score stays score-equivalent; it is left out when ``p`` is not below ``1``, which with
    the default happens only for two or fewer variables.

    ``__init__`` re-encodes each column to the codes ``0..r-1`` (see
    :func:`~andrey.core.categorical.encode`), each distinct value one category. Scores are memoized
    by ``(node, sorted parents)``.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Finite values; each distinct value of a column is one category.
    sample_prior : float, default=1.0
        The equivalent sample size ``a`` of the Dirichlet prior; positive.
    structure_prior : float, default=1.0
        The expected number of parents per variable under the structure prior; positive.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    References
    ----------
    .. [1] Heckerman, D., Geiger, D., and Chickering, D. M. (1995). Learning Bayesian networks:
       the combination of knowledge and statistical data. Machine Learning 20(3), 197-243.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        sample_prior: float = 1.0,
        structure_prior: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}"
            )
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError("data contains NaN or Inf; cannot compute BDeu scores")
        for name, value in (("sample_prior", sample_prior), ("structure_prior", structure_prior)):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        self.n: int = arr.shape[0]
        self.sample_prior = float(sample_prior)
        self.structure_prior = float(structure_prior)
        self._codes, self._cardinalities = categorical.encode(arr)
        others = arr.shape[1] - 1
        p = self.structure_prior / others if others > 0 else 1.0
        # Per-parent and per-non-parent log prior, or none when p is not a probability below 1.
        self._prior = (float(np.log(p)), float(np.log1p(-p)), others) if p < 1.0 else None
        self._score_key = lru_cache(maxsize=cache_size)(self._score_key_impl)

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        codes = self._codes
        r = int(self._cardinalities[node])
        q = float(np.prod([float(self._cardinalities[p]) for p in parents]))
        config, m = categorical.configurations(codes, self._cardinalities, parents)
        counts = np.bincount(config * r + codes[:, node], minlength=m * r).reshape(m, r)
        n_ij = counts.sum(axis=1)
        n_ijk = counts[counts > 0]
        a_j = self.sample_prior / q
        a_jk = a_j / r
        log_likelihood = (
            m * float(gammaln(a_j))
            - float(np.sum(gammaln(n_ij + a_j)))
            + float(np.sum(gammaln(n_ijk + a_jk)))
            - n_ijk.size * float(gammaln(a_jk))
        )
        log_prior = 0.0
        if self._prior is not None:
            log_p, log_not_p, others = self._prior
            k = len(parents)
            log_prior = k * log_p + (others - k) * log_not_p
        return -(log_likelihood + log_prior)

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized negative local BDeu score for ``node`` given ``parents``.

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = tuple(sorted(int(p) for p in parents))
        return self._score_key(int(node), key)

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the negative local BDeu scores for many parent sets, in request order."""
        return [self.score(node, parents) for parents in parent_sets]


class DGScore:
    """Degenerate Gaussian BIC score over mixed data.

    The columns are embedded as :class:`~andrey.core.mixed.DegenerateGaussian` describes, a
    categorical variable with ``k`` categories as the indicators of ``k - 1`` of them, and the
    embedded columns are scored as Gaussian [1]_. With ``Y`` the node's embedded columns, ``X`` its
    parents', and ``S`` the sample covariance of the embedded columns, the negative local score is

    ``n * log det S(Y | X) + lambda_value * log(n) * |Y| * |X|``

    where ``S(Y | X)`` is the residual covariance of ``Y`` regressed on ``X`` and ``|Y| * |X|``
    counts the regression's coefficients. Lower is better. It is :class:`BICScore` on the embedded
    columns, so on continuous columns only, the two scores are equal. Scores are memoized by
    ``(node, sorted parents)``.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Finite values; categorical columns hold their categories as numbers, each distinct value of
        a column one category.
    categorical : array-like of bool, shape (d,), optional
        Marks the categorical columns; ``None`` marks none.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    """

    def __init__(
        self,
        data: npt.ArrayLike,
        categorical: npt.ArrayLike | None = None,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        self._dg = DegenerateGaussian(data, categorical)
        self.n: int = self._dg.n
        self.lambda_value = penalty_weight(lambda_value)
        self.log_n = float(np.log(self.n))
        self._cov_diag = np.diag(self._dg.cov)
        self._score_key = lru_cache(maxsize=cache_size)(self._score_key_impl)

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        cov = self._dg.cov
        ys = self._dg.columns([node])
        xs = self._dg.columns(parents)
        if ys.size == 1:
            return local_score_bic(
                cov,
                self.n,
                int(ys[0]),
                xs.tolist(),
                lambda_value=self.lambda_value,
                log_n=self.log_n,
                cov_diag=self._cov_diag,
            )
        residual = cov[np.ix_(ys, ys)]
        if xs.size:
            cross = cov[np.ix_(xs, ys)]
            residual = residual - cross.T @ _solve(cov[np.ix_(xs, xs)], cross)
        sign, log_det = np.linalg.slogdet(residual)
        log_det = log_det if sign > 0 else -np.inf
        return float(self.n * log_det + self.log_n * self.lambda_value * ys.size * xs.size)

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized negative local DG score for ``node`` given ``parents``.

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = tuple(sorted(int(p) for p in parents))
        return self._score_key(int(node), key)

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the negative local DG scores for many parent sets, in request order."""
        return [self.score(node, parents) for parents in parent_sets]


class CGScore:
    """Conditional Gaussian BIC score over mixed data.

    The variables are fit as :class:`~andrey.core.mixed.ConditionalGaussian` describes: the rows
    split into the cells of the categorical variables' joint categories, the continuous variables
    Gaussian within each cell, with a covariance shrunk toward that of all rows. With ``l(A)`` the
    log-likelihood of the set ``A`` and ``k(A)`` its parameter count, a node is scored given its
    parents by the model of the node and its parents together divided by the model of the parents
    alone [1]_. With ``F`` the node and its
    parents and ``P`` the parents,

    ``-2 * (l(F) - l(P)) + lambda_value * log(n) * (k(F) - k(P))``

    Lower is better; the value is ``-2`` times a BIC. Each term models a set of variables jointly,
    so a categorical node with continuous parents is scored as well as a continuous one, and two
    graphs with the same equivalence class score the same. Scores are memoized by
    ``(node, sorted parents)``.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Finite values; categorical columns hold their categories as numbers, each distinct value of
        a column one category.
    categorical : array-like of bool, shape (d,), optional
        Marks the categorical columns; ``None`` marks none.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    Raises
    ------
    ValueError
        If a continuous column is constant.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    """

    def __init__(
        self,
        data: npt.ArrayLike,
        categorical: npt.ArrayLike | None = None,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        self._cg = ConditionalGaussian(data, categorical, cache_size=cache_size)
        self.n: int = self._cg.n
        self.lambda_value = penalty_weight(lambda_value)
        self.log_n = float(np.log(self.n))
        self._memo = _LRU(cache_size)

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized negative local CG score for ``node`` given ``parents``.

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        return self.score_many(node, [parents])[0]

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the negative local CG scores for many parent sets, in request order.

        The log-likelihoods the scores need are fit in one batch; each score equals the one
        :meth:`score` returns alone.
        """
        node = int(node)
        keys = [tuple(sorted(int(p) for p in parents)) for parents in parent_sets]
        found = {key: self._memo.get((node, key)) for key in keys}
        todo = [key for key, value in found.items() if value is None]
        if todo:
            cg = self._cg
            families = [(*key, node) for key in todo]
            loglik = cg.loglik_many(families + todo)
            for i, (family, key) in enumerate(zip(families, todo, strict=True)):
                dof = cg.dof(family) - cg.dof(key)
                value = -2.0 * (loglik[i] - loglik[len(todo) + i])
                value += self.lambda_value * self.log_n * dof
                found[key] = value
                self._memo.put((node, key), value)
        return [found[key] for key in keys]


#: Every local score class; the searches that difference full scores take any of them.
LocalScore = BICScore | BDeuScore | DGScore | CGScore


def make_score(
    score_func: str,
    data: npt.ArrayLike,
    *,
    lambda_value: float = 1.0,
    categorical: npt.ArrayLike | None = None,
) -> LocalScore:
    """Build the local score ``score_func`` names over ``data``.

    ``"local_score_BIC"`` and ``"local_score_BIC_from_cov"`` name :class:`BICScore`,
    ``"local_score_BDeu"`` :class:`BDeuScore` with its default priors, and
    ``"local_score_degen_gauss"`` and ``"local_score_cond_gauss"`` :class:`DGScore` and
    :class:`CGScore` over the ``categorical`` mask. BDeu has no penalty weight but still checks
    ``lambda_value``'s range. A search checks ``score_func`` against the names it takes before
    calling this. Raises :class:`NotImplementedError` for any other name.
    """
    if score_func in ("local_score_BIC", "local_score_BIC_from_cov"):
        return BICScore(np.asarray(data, dtype=np.float64), lambda_value=lambda_value)
    if score_func == "local_score_BDeu":
        penalty_weight(lambda_value)
        return BDeuScore(np.asarray(data))
    if score_func == "local_score_degen_gauss":
        return DGScore(data, categorical, lambda_value=lambda_value)
    if score_func == "local_score_cond_gauss":
        return CGScore(data, categorical, lambda_value=lambda_value)
    raise NotImplementedError(f"unknown score_func {score_func!r}")
