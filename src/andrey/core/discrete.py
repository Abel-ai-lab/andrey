"""Conditional-independence tests and local scores for discrete (categorical) data.

Each column is a categorical variable; its distinct values are recoded to the codes ``0 .. r - 1``
in sorted order, so any numeric labels work (``0/1``, ``1/2/3``, ``-1/+1``). Every query counts the
joint table of the variables it involves, one stratum per observed configuration of the
conditioning set or the parent set; configurations that never occur add nothing.

- :class:`ChiSquare` and :class:`GSquare` test ``x`` independent of ``y`` given ``S`` from the
  per-stratum ``x``-by-``y`` contingency tables: Pearson's chi-square or the likelihood-ratio G
  statistic summed over the strata, against a chi-square law whose degrees of freedom drop each
  stratum's empty rows and columns.
- :class:`BDeuScore` and :class:`DiscreteBICScore` score a node given its parents by the
  multinomial counts ``N_ijk`` (child state ``k`` under parent configuration ``j``) on the deviance
  scale of :class:`~andrey.core.score.BICScore`: lower is better, ``-2`` times the log score.

References
----------
Spirtes, Glymour and Scheines (2000). Causation, Prediction, and Search, 2nd ed. MIT Press.

Heckerman, Geiger and Chickering (1995). Learning Bayesian networks: the combination of knowledge
and statistical data. Machine Learning 20(3):197-243. https://doi.org/10.1007/BF00994016

Buntine (1991). Theory refinement on Bayesian networks. UAI, 52-60.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2


def encode_discrete(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(codes, levels)``: each column recoded to ``0 .. r - 1`` and its level count ``r``.

    ``codes`` is an ``int64`` ``(n_samples, n_features)`` matrix; a column's codes follow the sorted
    order of its distinct values. ``levels[j]`` is the number of distinct values in column ``j``.

    Raises
    ------
    ValueError
        If ``data`` is not a 2-D matrix, or holds ``NaN`` or ``inf``.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError("data contains NaN or Inf; cannot encode discrete columns")
    codes = np.empty(arr.shape, dtype=np.int64)
    levels = np.empty(arr.shape[1], dtype=np.int64)
    for j in range(arr.shape[1]):
        values, codes[:, j] = np.unique(arr[:, j], return_inverse=True)
        levels[j] = len(values)
    return codes, levels


def _strata(codes: np.ndarray, levels: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    """Return each row's configuration of ``columns`` as an index into the observed configurations.

    The index runs over the configurations that occur, in mixed-radix order; with no columns every
    row is in stratum 0.
    """
    n = codes.shape[0]
    if not columns:
        return np.zeros(n, dtype=np.int64)
    if len(columns) == 1:
        return codes[:, columns[0]]
    if float(np.prod(levels[list(columns)], dtype=np.float64)) < 2.0**62:
        flat = np.zeros(n, dtype=np.int64)
        for c in columns:
            flat *= levels[c]
            flat += codes[:, c]
        return np.unique(flat, return_inverse=True)[1]
    return np.unique(codes[:, list(columns)], axis=0, return_inverse=True)[1].reshape(-1)


class _ContingencyTest:
    """Shared counting for the chi-square and G-square tests.

    ``__init__`` recodes the data once (:func:`encode_discrete`). Each call counts the
    ``(stratum, x, y)`` table of the query and passes it to :meth:`_statistic`. Queries are
    canonicalized (``x``, ``y`` ordered, the conditioning set sorted unique) and nothing is
    memoized.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._codes, self._levels = encode_discrete(data)
        self._n = self._codes.shape[0]

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        A larger p-value means more independence. A query with no degrees of freedom left, where
        every stratum has one level of ``x`` or of ``y``, returns 1.
        """
        cond = sorted(set(int(c) for c in condition_set))
        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        assert x_int not in cond and y_int not in cond, "x, y cannot be in condition_set."
        counts = self._table(x_int, y_int, cond)
        rows = counts.sum(axis=2)  # (k, rx)
        cols = counts.sum(axis=1)  # (k, ry)
        total = rows.sum(axis=1)  # (k,)
        expected = rows[:, :, None] * cols[:, None, :] / total[:, None, None]
        dof = int(
            np.sum((np.count_nonzero(rows, axis=1) - 1) * (np.count_nonzero(cols, axis=1) - 1))
        )
        if dof <= 0:
            return 1.0
        return float(chi2.sf(self._statistic(counts, expected), dof))

    def _table(self, x: int, y: int, cond: list[int]) -> np.ndarray:
        """The ``(k, r_x, r_y)`` counts of ``x`` and ``y`` in each of the ``k`` observed strata."""
        rx, ry = int(self._levels[x]), int(self._levels[y])
        strata = _strata(self._codes, self._levels, cond)
        k = int(strata.max()) + 1
        flat = (strata * rx + self._codes[:, x]) * ry + self._codes[:, y]
        counts = np.bincount(flat, minlength=k * rx * ry).reshape(k, rx, ry)
        return counts[counts.sum(axis=(1, 2)) > 0].astype(np.float64)

    def _statistic(self, counts: np.ndarray, expected: np.ndarray) -> float:
        raise NotImplementedError


class ChiSquare(_ContingencyTest):
    """Pearson chi-square conditional-independence test for discrete data.

    The statistic sums ``(N - E)**2 / E`` over the cells of every stratum's ``x``-by-``y`` table,
    with ``E`` the count expected under independence within the stratum; cells with ``E == 0`` add
    nothing. Its null law is chi-square with ``sum_k (r_xk - 1) * (r_yk - 1)`` degrees of freedom,
    where ``r_xk`` and ``r_yk`` count the levels of ``x`` and ``y`` observed in stratum ``k``.
    """

    def _statistic(self, counts: np.ndarray, expected: np.ndarray) -> float:
        safe = np.where(expected > 0, expected, 1.0)
        return float(np.sum((counts - expected) ** 2 / safe))


class GSquare(_ContingencyTest):
    """Likelihood-ratio G-square conditional-independence test for discrete data.

    The statistic is ``2 * sum N * log(N / E)`` over the cells with ``N > 0``, twice the
    conditional mutual information of ``x`` and ``y`` given ``S`` times the sample size, against
    the same chi-square law as :class:`ChiSquare`.
    """

    def _statistic(self, counts: np.ndarray, expected: np.ndarray) -> float:
        seen = counts > 0
        return float(2.0 * np.sum(counts[seen] * np.log(counts[seen] / expected[seen])))


class DiscreteScore:
    """The shared counting and memoization of :class:`BDeuScore` and :class:`DiscreteBICScore`.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; each column is one categorical variable.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.
    """

    def __init__(self, data: np.ndarray, *, cache_size: int | None = 1_000_000) -> None:
        self._codes, self.levels = encode_discrete(data)
        self.n = int(self._codes.shape[0])
        self.log_n = float(np.log(self.n))
        self._score_key = lru_cache(maxsize=cache_size)(self._score_key_impl)

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized local score of ``node`` given ``parents`` (lower is better).

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = tuple(sorted(int(p) for p in parents))
        return self._score_key(int(node), key)

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the local scores for many parent sets, in request order."""
        return [self.score(node, parents) for parents in parent_sets]

    def _counts(self, node: int, parents: tuple[int, ...]) -> np.ndarray:
        """Return the ``(q_observed, r)`` counts ``N_ijk``: ``node``'s states per parent config."""
        r = int(self.levels[node])
        strata = _strata(self._codes, self.levels, list(parents))
        k = int(strata.max()) + 1
        flat = strata * r + self._codes[:, node]
        counts = np.bincount(flat, minlength=k * r).reshape(k, r)
        return counts[counts.sum(axis=1) > 0].astype(np.float64)

    def _q(self, parents: tuple[int, ...]) -> float:
        """The number of parent configurations, observed or not: the product of their levels."""
        return float(np.prod(self.levels[list(parents)], dtype=np.float64))

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        raise NotImplementedError


class BDeuScore(DiscreteScore):
    """BDeu score: the Bayesian-Dirichlet likelihood-equivalent uniform score, on a deviance scale.

    With ``r`` states of the node, ``q`` configurations of its parents, and equivalent sample size
    ``ess``, every Dirichlet parameter is ``ess / (q * r)``. The log marginal likelihood is

        log BDeu = sum_j [lgamma(ess / q) - lgamma(ess / q + N_ij)]
                   + sum_jk [lgamma(ess / (q r) + N_ijk) - lgamma(ess / (q r))],

    over the observed configurations ``j``; an unobserved one adds zero. The structure prior makes
    each of the other ``d - 1`` variables a parent independently with probability
    ``p = structure_prior / (d - 1)``, capped at 0.5, so a node with ``k`` parents adds
    ``k log p + (d - 1 - k) log(1 - p)``: it penalizes each parent, and the expected number of
    parents is ``structure_prior``. Below four variables the cap makes ``p`` 0.5, where the prior
    is the same for every parent set and so has no effect. The local score is ``-2`` times the sum
    of the two; lower is better.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; each column is one categorical variable.
    ess : float, default=1.0
        Equivalent sample size of the Dirichlet prior; must be positive.
    structure_prior : float, default=1.0
        Expected number of parents per node under the structure prior; must be positive.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        ess: float = 1.0,
        structure_prior: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        ess = float(ess)
        if not np.isfinite(ess) or ess <= 0.0:
            raise ValueError(f"ess must be finite and positive, got {ess}")
        structure_prior = float(structure_prior)
        if not np.isfinite(structure_prior) or structure_prior <= 0.0:
            raise ValueError(f"structure_prior must be finite and positive, got {structure_prior}")
        self.ess = ess
        self.structure_prior = structure_prior
        super().__init__(data, cache_size=cache_size)
        others = self.levels.size - 1
        p = min(structure_prior / others, 0.5) if others else 0.5
        self._log_p, self._log_not_p, self._others = float(np.log(p)), float(np.log1p(-p)), others

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        counts = self._counts(node, parents)
        a_j = self.ess / self._q(parents)
        a_jk = a_j / int(self.levels[node])
        n_j = counts.sum(axis=1)
        log_score = np.sum(gammaln(a_j) - gammaln(a_j + n_j)) + np.sum(
            gammaln(a_jk + counts) - gammaln(a_jk)
        )
        k = len(parents)
        log_prior = k * self._log_p + (self._others - k) * self._log_not_p
        return float(-2.0 * (log_score + log_prior))


class DiscreteBICScore(DiscreteScore):
    """Multinomial BIC score on the deviance scale of :class:`~andrey.core.score.BICScore`.

    The local score is ``-2 * sum_jk N_ijk * log(N_ijk / N_ij) + lambda_value * log(n) * q * (r -
    1)``: twice the negative maximized log-likelihood plus ``lambda_value`` times ``log(n)`` per
    free parameter of the node's conditional probability table. Lower is better; with
    ``lambda_value=1`` it is ``-2`` times the usual BIC.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; each column is one categorical variable.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.
    """

    def __init__(
        self, data: np.ndarray, *, lambda_value: float = 1.0, cache_size: int | None = 1_000_000
    ) -> None:
        lambda_value = float(lambda_value)
        if not np.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError(f"lambda_value must be finite and non-negative, got {lambda_value}")
        self.lambda_value = lambda_value
        super().__init__(data, cache_size=cache_size)

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        counts = self._counts(node, parents)
        n_j = counts.sum(axis=1, keepdims=True)
        seen = counts > 0
        log_lik = np.sum(counts[seen] * np.log((counts / n_j)[seen]))
        n_params = self._q(parents) * (int(self.levels[node]) - 1)
        return float(-2.0 * log_lik + self.lambda_value * self.log_n * n_params)
