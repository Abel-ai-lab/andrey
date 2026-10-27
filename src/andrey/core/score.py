"""Scoring substrate for structure search: linear-Gaussian BIC, discrete BDeu, and mixed-data BICs.

The negative local BIC score (lower is better; proportional to ``-2 * BIC``) for a target node
given a parent set, evaluated from the sample covariance. Uses the linear-Gaussian BIC closed
form, branching on parent-set size for the scalar fast paths. Its penalty can add the
extended-BIC term of :func:`ebic_penalty`, which grows with the number of parent sets a node can
choose from.

:func:`local_score_bic` is the pure per-node primitive; :class:`BICScore` precomputes the covariance
once and memoizes scores behind the :class:`Score` protocol that structure-search engines consume.
:class:`BDeuScore` scores category codes with ``-2`` times the log BDeu marginal likelihood behind
the same protocol, and :class:`CategoricalBICScore` with the multinomial BIC. :class:`DGScore` and
:class:`CGScore` score mixed data, continuous and categorical columns marked by a mask, with the BIC
of the degenerate Gaussian and conditional Gaussian models of :mod:`andrey.core.mixed`.

References
----------
Schwarz (1978). Estimating the Dimension of a Model. Annals of Statistics 6(2):461-464.
https://doi.org/10.1214/aos/1176344136

Chen & Chen (2008). Extended Bayesian information criteria for model selection with large model
spaces. Biometrika 95(3):759-771. https://doi.org/10.1093/biomet/asn034

Heckerman, Geiger & Chickering (1995). Learning Bayesian networks: the combination of knowledge
and statistical data. Machine Learning 20(3):197-243. https://doi.org/10.1007/BF00994016

Chickering & Heckerman (1997). Efficient approximations for the marginal likelihood of Bayesian
networks with hidden variables. Machine Learning 29(2-3):181-212.
https://doi.org/10.1023/A:1007469629108

Andrews, Ramsey & Cooper (2018). Scoring Bayesian networks of mixed variables. International
Journal of Data Science and Analytics 6(1):3-18. https://doi.org/10.1007/s41060-017-0085-7

Andrews, Ramsey & Cooper (2019). Learning high-dimensional directed acyclic graphs with mixed
data-types. Proceedings of Machine Learning Research 104:4-21.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.linalg import get_lapack_funcs
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln

from . import categorical, stats
from .mixed import _LRU, ConditionalGaussian, DegenerateGaussian

#: The two names of the linear-Gaussian BIC, which :func:`make_score` builds alike. A search that
#: takes this score takes both.
LINEAR_SCORES: tuple[str, ...] = ("local_score_BIC", "local_score_BIC_from_cov")
#: The scores that read category codes, and those that read mixed data with a categorical mask.
DISCRETE_SCORES: tuple[str, ...] = ("local_score_BDeu", "local_score_BIC_categorical")
MIXED_SCORES: tuple[str, ...] = ("local_score_cond_gauss", "local_score_degen_gauss")
#: Every score name :func:`make_score` builds: what a search with all three kinds of score accepts.
SCORES: tuple[str, ...] = (*LINEAR_SCORES, *DISCRETE_SCORES, *MIXED_SCORES)

# A residual variance at most this fraction of the node's variance is zero up to rounding.
ZERO_RESIDUAL = 1e-12


def local_score_bic(
    cov: np.ndarray,
    n: int,
    node: int,
    parents: Sequence[int],
    *,
    lambda_value: float = 1.0,
    ebic_gamma: float = 0.0,
    log_n: float | None = None,
    cov_diag: np.ndarray | None = None,
) -> float:
    """Return the negative local BIC score for ``node`` given ``parents``.

    Evaluates ``n * log R(node | parents) + log_n * len(parents) * lambda_value``, plus
    :func:`ebic_penalty` of ``len(parents)`` among the ``d`` columns of ``cov`` when
    ``ebic_gamma`` is not 0, where ``R`` is the linear-regression residual variance
    ``cov[node, node] - cov[node, parents] @ inv(cov[parents, parents]) @ cov[parents, node]``.
    Lower is better; the value is proportional to ``-2 * BIC``.

    The family is unavailable, and the score ``inf``, when ``parents`` has more than ``n - 2``
    members, which fit ``node`` exactly, or when ``R`` is at most :data:`ZERO_RESIDUAL` times
    ``cov[node, node]``, zero up to rounding. A search that minimizes the score never selects an
    unavailable family.

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
    ebic_gamma : float, default=0.0
        Weight on the extended-BIC term (:func:`ebic_penalty`); ``0.0`` is the standard BIC.
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
    if n_parents > n - 2:
        return math.inf
    variance = float(cov_diag[node])
    penalty = log_n * n_parents * lambda_value
    if ebic_gamma:
        penalty += ebic_penalty(n_parents, cov.shape[0], ebic_gamma)

    if n_parents == 0:
        return _from_residual(variance, variance, n, penalty)

    if n_parents == 1:
        parent = int(parents[0])
        parent_var = float(cov_diag[parent])
        if parent_var != 0.0:
            cov_ip = float(cov[node, parent])
            residual = variance - cov_ip * (cov_ip / parent_var)
            return _from_residual(residual, variance, n, penalty)
        yX = cov[node : node + 1, parent : parent + 1]
        XX = cov[parent : parent + 1, parent : parent + 1]
        beta = _solve(XX, yX.T)
        return _from_residual(variance - (yX @ beta).item(), variance, n, penalty)

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
        return _from_residual(variance - (yX @ beta).item(), variance, n, penalty)

    pa = [int(p) for p in parents]
    yX = cov[node : node + 1, pa]
    XX = cov[pa][:, pa]
    beta = _solve(XX, yX.T)
    return _from_residual(variance - (yX @ beta).item(), variance, n, penalty)


def _from_residual(residual: float, variance: float, n: int, penalty: float) -> float:
    """``n * log(residual) + penalty``, or ``inf`` when ``residual`` is zero up to rounding."""
    if not residual > ZERO_RESIDUAL * variance:
        return math.inf
    return float(n * np.log(residual) + penalty)


class CopiedColumnsError(ValueError):
    """Raised when data columns repeat other columns up to scale and shift.

    ``groups`` holds each group of copies as column positions, as :func:`copy_groups` returns
    them; the message is :func:`copies_message` of those groups, naming each position by
    ``column_name``.
    """

    groups: tuple[tuple[int, ...], ...] = ()
    column_name: Callable[[int], str] = str


def copy_groups(corr: np.ndarray, n: int) -> tuple[tuple[int, ...], ...]:
    """Return the groups of columns that repeat one another up to scale and shift.

    Two columns are copies when ``1 - corr**2`` is at most ``ZERO_RESIDUAL``: either one's residual
    variance given the other is zero up to rounding. A group is a connected set of copies, given as
    its positions in increasing order, so the first is the column to keep. Groups are ordered by
    that first column. With at most two rows ``n`` every pair of columns is a copy, so the result is
    empty.

    Parameters
    ----------
    corr : ndarray, shape (d, d)
        Correlation matrix of the data's columns.
    n : int
        Sample count.
    """
    if n <= 2 or corr.shape[0] < 2:
        return ()
    alike = np.abs(corr)
    np.fill_diagonal(alike, 0.0)
    copies = 1.0 - alike**2 <= ZERO_RESIDUAL
    if not copies.any():
        return ()
    _, component = connected_components(copies, directed=False)
    members: dict[int, list[int]] = {}
    for j, label in enumerate(component.tolist()):
        members.setdefault(label, []).append(j)
    return tuple(tuple(group) for group in members.values() if len(group) > 1)


def copies_message(groups: Sequence[Sequence[int]], name: Callable[[int], str] = str) -> str:
    """Return the error message for the groups of copies ``groups``, naming columns by ``name``.

    The message names every group, its first column the one to keep, and says what to do: keep one
    column per group and read its edges as the group's.
    """
    parts = []
    for kept, *copies in groups:
        names = [name(j) for j in copies]
        listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        parts.append(f"{listed} {'repeats' if len(names) == 1 else 'repeat'} {name(kept)}")
    return (
        f"data columns repeat other columns up to scale and shift: {'; '.join(parts)}. Copies of "
        "one variable cannot be told apart, so keep one column per group, drop the others before "
        "the fit, and read the kept column's edges as the group's"
    )


def copies_error(
    groups: Sequence[Sequence[int]], name: Callable[[int], str] = str
) -> CopiedColumnsError:
    """Return the :class:`CopiedColumnsError` for the groups of copies ``groups``.

    Its message is :func:`copies_message` of ``groups`` and ``name``; its ``groups`` and
    ``column_name`` hold them.
    """
    error = CopiedColumnsError(copies_message(groups, name))
    error.groups = tuple(tuple(group) for group in groups)
    error.column_name = name
    return error


def check_columns(cov: np.ndarray, n: int) -> None:
    """Raise ``ValueError`` when a column of the data behind ``cov`` makes BIC scoring meaningless.

    A column is rejected when it is constant; with more than two rows ``n``, when its residual
    variance given one other column is at most ``ZERO_RESIDUAL`` times its variance (an affine copy
    of that column); and with more rows than columns, when its residual variance given all other
    columns is at most ``ZERO_RESIDUAL`` times its variance (a linear function of other columns).
    Every column is a linear function of any ``n - 1`` others, so the last two checks apply only
    where the row count does not force such a fit. Copies raise one :class:`CopiedColumnsError`
    that names every group :func:`copy_groups` finds.

    The residual variance of column ``j`` given all others, relative to its variance, is
    ``1 / inv(corr)[j, j]`` for the correlation matrix ``corr``. It is at least the smallest
    eigenvalue of ``corr``, so data whose smallest eigenvalue exceeds ``ZERO_RESIDUAL`` pass
    without the per-column computation.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Sample covariance matrix.
    n : int
        Sample count.

    Raises
    ------
    CopiedColumnsError
        If a column is an affine copy of another column when ``n`` exceeds 2.
    ValueError
        If a column is constant, or a linear function of other columns when ``n`` exceeds the
        column count.
    """
    variance = np.diag(cov)
    constant = np.flatnonzero(~(variance > 0.0))
    if constant.size:
        raise ValueError(f"data column {int(constant[0])} is constant; drop it before the fit")
    d = cov.shape[0]
    if d < 2 or n <= 2:
        return
    sd = np.sqrt(variance)
    corr = cov / np.outer(sd, sd)
    groups = copy_groups(corr, n)
    if groups:
        raise copies_error(groups)
    if n <= d:
        return
    if np.linalg.eigvalsh(corr)[0] > ZERO_RESIDUAL:
        return
    values, vectors = np.linalg.eigh(corr)
    # diag(inv(corr)) from the eigendecomposition. Eigenvalues below the rounding of the largest
    # are zero up to rounding; the floor keeps them finite and off the columns they do not touch.
    floor = np.finfo(np.float64).eps * values[-1]
    residual = 1.0 / ((vectors**2) @ (1.0 / np.maximum(values, floor)))
    columns = np.flatnonzero(~(residual > ZERO_RESIDUAL)).tolist()
    if columns:
        raise ValueError(
            f"data columns {columns} are linearly dependent: one is a linear function of the "
            "others; drop one of them before the fit"
        )


def penalty_weight(lambda_value: float, name: str = "lambda_value") -> float:
    """Return the penalty weight ``lambda_value`` as a float, checked to be finite and >= 0.

    Raises ``ValueError`` naming the parameter ``name`` otherwise. The searches check
    ``lambda_value`` and ``ebic_gamma`` under every score, including those that ignore them, so a
    setting is valid or not whatever the score.
    """
    value = float(lambda_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


@lru_cache(maxsize=65_536)
def ebic_penalty(n_parents: int, n_variables: int, ebic_gamma: float) -> float:
    """Return the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``.

    A node picks its parents among the other ``n_variables - 1`` variables, so
    ``C(n_variables - 1, n_parents)`` counts its parent sets of ``n_parents`` members. Added to a
    family's BIC, the term gives the node-wise extended BIC of Chen and Chen (2008), which spreads
    a prior mass of ``C(n_variables - 1, n_parents) ** -ebic_gamma`` evenly over the parent sets of
    each size. With ``ebic_gamma`` 0 it is 0, the standard BIC. It depends on the parent count
    only, so equivalent DAGs, which give their nodes the same parent counts, score the same. Every
    score and score change of the linear-Gaussian BIC takes the term from here;
    :func:`ebic_step` is the change of one added parent. It is ``inf`` for ``n_parents`` of
    ``n_variables`` or more, which no node can have.
    """
    if n_parents >= n_variables:
        return math.inf
    log_count = (
        math.lgamma(n_variables) - math.lgamma(n_parents + 1) - math.lgamma(n_variables - n_parents)
    )
    return 2.0 * ebic_gamma * log_count


def ebic_step(n_parents: int, n_variables: int, ebic_gamma: float) -> float:
    """Return the change in :func:`ebic_penalty` when a node of ``n_parents`` parents gains one.

    It is ``2 * ebic_gamma * (log(n_variables - 1 - n_parents) - log(n_parents + 1))``, computed
    as the difference of the two terms, so a move's score change and the difference of its two
    family scores carry the same term. It falls below 0 past ``(n_variables - 2) / 2`` parents,
    where the count of parent sets shrinks.
    """
    return ebic_penalty(n_parents + 1, n_variables, ebic_gamma) - ebic_penalty(
        n_parents, n_variables, ebic_gamma
    )


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
    ebic_gamma : float, default=0.0
        Weight on the extended-BIC term :func:`ebic_penalty` over the ``d`` columns; ``0.0`` is the
        standard BIC.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, holds ``NaN`` or ``inf``, or has a column :func:`check_columns`
        rejects; or if ``lambda_value`` or ``ebic_gamma`` is negative or non-finite.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        lambda_value: float = 1.0,
        ebic_gamma: float = 0.0,
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
        check_columns(cov, data.shape[0])
        self._init(cov, data.shape[0], lambda_value, ebic_gamma, cache_size)

    @classmethod
    def from_cov(
        cls,
        cov: np.ndarray,
        n: int,
        *,
        lambda_value: float = 1.0,
        ebic_gamma: float = 0.0,
        cache_size: int | None = 1_000_000,
    ) -> BICScore:
        """Build a scorer from a precomputed covariance.

        Pooled workers use the parent's ``cov`` bytes, giving every process identical inputs to
        :func:`local_score_bic` and bit-identical scores on the same host BLAS. The columns are not
        checked with :func:`check_columns`.

        Parameters
        ----------
        cov : ndarray, shape (d, d)
            Sample covariance matrix; must be square and finite.
        n : int
            Sample count the BIC complexity penalty uses.
        lambda_value : float, default=1.0
            Weight on the BIC complexity term of the deviance the search minimizes.
        ebic_gamma : float, default=0.0
            Weight on the extended-BIC term :func:`ebic_penalty` over the columns of ``cov``;
            ``0.0`` is the standard BIC.
        cache_size : int or None, default=1_000_000
            Memoization cache bound; ``None`` is unbounded.

        Raises
        ------
        ValueError
            If ``cov`` is not a square 2-D matrix, or holds ``NaN`` / ``inf``; or if
            ``lambda_value`` or ``ebic_gamma`` is negative or non-finite.
        """
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be a square 2-D matrix, got shape {cov.shape}")
        if not np.isfinite(cov).all():
            raise ValueError("cov contains NaN or Inf; cannot compute BIC scores")
        obj = cls.__new__(cls)
        obj._init(cov, int(n), lambda_value, ebic_gamma, cache_size)
        return obj

    def _init(
        self,
        cov: np.ndarray,
        n: int,
        lambda_value: float,
        ebic_gamma: float,
        cache_size: int | None,
    ) -> None:
        lambda_value = penalty_weight(lambda_value)
        self.cov = cov
        self.n = n
        self.lambda_value = lambda_value
        self.ebic_gamma = penalty_weight(ebic_gamma, "ebic_gamma")
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
            ebic_gamma=self.ebic_gamma,
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


class _CountScore:
    """Memoized local scores computed from a node's counts in each configuration of its parents.

    A subclass sets ``_codes`` and ``_cardinalities`` (see :func:`~andrey.core.categorical.encode`)
    and ``_memo``, and scores one table in :meth:`_local`. :meth:`score_many` counts the tables of
    every parent set not yet memoized together (:func:`~andrey.core.categorical.table_counts`); a
    score is the same whether it is computed alone or in a batch.
    """

    _codes: np.ndarray
    _cardinalities: np.ndarray
    _memo: _LRU

    def _local(self, node: int, parents: tuple[int, ...], counts: np.ndarray) -> float:
        raise NotImplementedError

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized local score for ``node`` given ``parents``.

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = (int(node), tuple(sorted(int(p) for p in parents)))
        value = self._memo.get(key)
        if value is None:
            (counts,) = categorical.table_counts(self._codes, self._cardinalities, key[0], [key[1]])
            value = self._local(key[0], key[1], counts)
            self._memo.put(key, value)
        return value

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the local scores for many parent sets, in request order."""
        node = int(node)
        keys = [tuple(sorted(int(p) for p in parents)) for parents in parent_sets]
        found = {key: self._memo.get((node, key)) for key in keys}
        todo = [key for key, value in found.items() if value is None]
        if todo:
            tables = categorical.table_counts(self._codes, self._cardinalities, node, todo)
            for key, counts in zip(todo, tables, strict=True):
                found[key] = self._local(node, key, counts)
                self._memo.put((node, key), found[key])
        return [found[key] for key in keys]


class BDeuScore(_CountScore):
    """Local BDeu score over category codes, on the scale of :class:`BICScore`.

    The score of ``node`` given ``parents`` is ``-2`` times the log of the structure prior times the
    BDeu marginal likelihood [1]_, so lower is better and a difference of scores reads on the
    scale of :class:`BICScore`'s, ``-2`` times a log-likelihood. With ``r`` the node's
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
        self._memo = _LRU(cache_size)

    def _local(self, node: int, parents: tuple[int, ...], counts: np.ndarray) -> float:
        m, r = counts.shape
        q = float(np.prod([float(self._cardinalities[p]) for p in parents]))
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
        return -2.0 * (log_likelihood + log_prior)


class CategoricalBICScore(_CountScore):
    """Multinomial BIC score over category codes, on the scale of :class:`BICScore`.

    Each node is a categorical variable whose distribution in each configuration of its parents is
    fit by its counts [1]_ [2]_. With ``r`` the node's categories, ``q`` the product of its parents'
    categories, ``N_ij`` the rows in parent configuration ``j``, and ``N_ijk`` those of them where
    the node takes value ``k``, the local score is

    ``-2 * sum_jk N_ijk * log(N_ijk / N_ij) + lambda_value * log(n) * q * (r - 1)``

    ``-2`` times the maximized log-likelihood plus ``lambda_value * log(n)`` per free parameter of
    the node's conditional probability table. Lower is better; with ``lambda_value=1`` it is ``-2``
    times the BIC. A cell no row falls in adds nothing to the likelihood but still counts its
    parameters. Equivalent graphs score the same.

    ``__init__`` re-encodes each column to the codes ``0..r-1`` (see
    :func:`~andrey.core.categorical.encode`), each distinct value one category. Scores are memoized
    by ``(node, sorted parents)``.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Finite values; each distinct value of a column is one category.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    Notes
    -----
    **Limitations.** It counts ``q * (r - 1)`` parameters whether or not every parent configuration
    occurs, so with many parent categories and few rows the penalty outgrows the likelihood gain
    and the score prefers fewer parents than the data support. Its tests check it against the
    formula on fixed tables. Hill climbing was measured with it on simulated categorical networks
    of 10 to 40 variables with 2,000 rows (``andrey.hc``); the other searches that take it are not
    benchmarked on it yet. The numbers are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics 6(2),
       461-464.
    .. [2] Chickering, D. M., and Heckerman, D. (1997). Efficient approximations for the marginal
       likelihood of Bayesian networks with hidden variables. Machine Learning 29(2-3), 181-212.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}"
            )
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError("data contains NaN or Inf; cannot compute categorical BIC scores")
        self.n: int = arr.shape[0]
        self.lambda_value = penalty_weight(lambda_value)
        self.log_n = float(np.log(self.n))
        self._codes, self._cardinalities = categorical.encode(arr)
        self._memo = _LRU(cache_size)

    def _local(self, node: int, parents: tuple[int, ...], counts: np.ndarray) -> float:
        r = counts.shape[1]
        q = float(np.prod([float(self._cardinalities[p]) for p in parents]))
        n_ij = np.repeat(counts.sum(axis=1), r)[counts.ravel() > 0]
        n_ijk = counts[counts > 0]
        log_likelihood = float(np.sum(n_ijk * np.log(n_ijk / n_ij)))
        return -2.0 * log_likelihood + self.lambda_value * self.log_n * q * (r - 1)


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

    The parents' columns are collinear when one parent's categories determine another's, as in a
    network with deterministic conditional probabilities, and a solve on their singular covariance
    can return a wrong fit without raising. A pivoted Cholesky factorization of their correlation
    matrix then picks, one at a time, the column with the most residual variance given the columns
    already picked, and stops when none keeps more than :data:`ZERO_RESIDUAL` of its variance.
    ``Y`` is regressed on the picked columns, which fit it as all of them do, and ``|X|`` still
    counts every column.

    The family is unavailable, and the score ``inf``, when a combination of the node's columns
    keeps at most :data:`ZERO_RESIDUAL` of its variance given the parents, the rule
    :func:`local_score_bic` applies to a single column. A node its parents determine, such as a
    merge of a parent's categories, is one. A search that minimizes the score never selects an
    unavailable family.

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
        self._sd = np.sqrt(self._cov_diag)
        # When every eigenvalue of the embedded columns' correlation matrix exceeds ZERO_RESIDUAL,
        # so does each column's share of variance left given any other columns: no parents are
        # collinear.
        corr = self._dg.cov / np.outer(self._sd, self._sd)
        try:
            np.linalg.cholesky(corr - ZERO_RESIDUAL * np.eye(len(corr)))
            self._check_collinear = False
        except np.linalg.LinAlgError:
            self._check_collinear = True
        self._score_key = lru_cache(maxsize=cache_size)(self._score_key_impl)

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        cov = self._dg.cov
        ys = self._dg.columns([node])
        xs = self._dg.columns(parents)
        kept = xs
        if self._check_collinear and xs.size > 1:
            sd = self._sd[xs]
            corr = cov[np.ix_(xs, xs)] / np.outer(sd, sd)
            (pstrf,) = get_lapack_funcs(("pstrf",), (corr,))
            _, pivots, rank, _ = pstrf(corr, tol=ZERO_RESIDUAL, lower=1)
            if rank < xs.size:
                kept = xs[np.sort(pivots[:rank] - 1)]
        if ys.size == 1 and kept.size == xs.size:
            return local_score_bic(
                cov,
                self.n,
                int(ys[0]),
                xs.tolist(),
                lambda_value=self.lambda_value,
                log_n=self.log_n,
                cov_diag=self._cov_diag,
            )
        own = cov[np.ix_(ys, ys)]
        residual = own
        if kept.size:
            cross = cov[np.ix_(kept, ys)]
            residual = own - cross.T @ _solve(cov[np.ix_(kept, kept)], cross)
        try:
            # Positive definite when every combination of the node's columns keeps more than
            # ZERO_RESIDUAL of its variance.
            np.linalg.cholesky(residual - ZERO_RESIDUAL * own)
        except np.linalg.LinAlgError:
            return math.inf
        log_det = np.linalg.slogdet(residual)[1]
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
LocalScore = BICScore | BDeuScore | CategoricalBICScore | DGScore | CGScore


def make_score(
    score_func: str,
    data: npt.ArrayLike,
    *,
    lambda_value: float = 1.0,
    ebic_gamma: float = 0.0,
    categorical: npt.ArrayLike | None = None,
) -> LocalScore:
    """Build the local score ``score_func`` names over ``data``.

    ``"local_score_BIC"`` and ``"local_score_BIC_from_cov"`` name :class:`BICScore`,
    ``"local_score_BDeu"`` :class:`BDeuScore` with its default priors,
    ``"local_score_BIC_categorical"`` :class:`CategoricalBICScore`, and
    ``"local_score_degen_gauss"`` and ``"local_score_cond_gauss"`` :class:`DGScore` and
    :class:`CGScore` over the ``categorical`` mask. ``ebic_gamma`` weighs the extended-BIC term
    of the linear-Gaussian BIC; the other scores leave it out but still check its range, as BDeu
    does ``lambda_value``'s. A search checks ``score_func`` against the names it takes before
    calling this. Raises :class:`NotImplementedError` for any other name.
    """
    if score_func in LINEAR_SCORES:
        return BICScore(
            np.asarray(data, dtype=np.float64), lambda_value=lambda_value, ebic_gamma=ebic_gamma
        )
    penalty_weight(ebic_gamma, "ebic_gamma")
    if score_func == "local_score_BDeu":
        penalty_weight(lambda_value)
        return BDeuScore(np.asarray(data))
    if score_func == "local_score_BIC_categorical":
        return CategoricalBICScore(np.asarray(data), lambda_value=lambda_value)
    if score_func == "local_score_degen_gauss":
        return DGScore(data, categorical, lambda_value=lambda_value)
    if score_func == "local_score_cond_gauss":
        return CGScore(data, categorical, lambda_value=lambda_value)
    raise NotImplementedError(f"unknown score_func {score_func!r}")
