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

from collections import OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.special import chdtrc, gammaln

# A count over every possible configuration is one bincount with no sort; it is used while the
# table has at most this many cells per row (or _DENSE_MIN cells), so it never costs much more than
# reading the rows.
_DENSE_ROWS = 4
_DENSE_MIN = 1 << 16
# Cells counted by one bincount in a batch: bounds the batch's memory to a few tens of megabytes.
_BATCH_CELLS = 1 << 21
# Bytes of conditioning-set configurations a test keeps for reuse.
_STRATA_CACHE_BYTES = 64 << 20


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


def _strata(
    codes: np.ndarray, levels: np.ndarray, columns: Sequence[int], cells: int
) -> tuple[np.ndarray, int]:
    """Return ``(index, k)``: each row's configuration of ``columns`` as an index below ``k``.

    The index follows the mixed-radix order of the configurations, the last column varying
    fastest; with no columns every row is in stratum 0. ``cells`` is the number of table cells
    each stratum gets. While ``k * cells`` stays within a few times the row count, ``k`` is the
    number of possible configurations, so a count over the index is one ``bincount`` with no sort,
    and its empty strata are dropped afterward; beyond that, only the configurations that occur
    are numbered, in the same order. Either way the occurring strata come out in the same order.
    """
    n = codes.shape[0]
    if not columns:
        return np.zeros(n, dtype=np.int64), 1
    if len(columns) == 1:
        return codes[:, columns[0]], int(levels[columns[0]])
    q = float(np.prod(levels[list(columns)], dtype=np.float64))
    if q < 2.0**62:
        flat = np.zeros(n, dtype=np.int64)
        for c in columns:
            flat *= levels[c]
            flat += codes[:, c]
        if q * cells <= max(_DENSE_ROWS * n, _DENSE_MIN):
            return flat, int(q)
        inverse = np.unique(flat, return_inverse=True)[1]
    else:
        inverse = np.unique(codes[:, list(columns)], axis=0, return_inverse=True)[1].reshape(-1)
    return inverse, int(inverse.max()) + 1


class _ContingencyTest:
    """Shared counting for the chi-square and G-square tests.

    ``__init__`` recodes the data once (:func:`encode_discrete`). A query counts the
    ``(stratum, x, y)`` table and passes it to :meth:`_pvalues`. :meth:`batched_call` counts many
    conditioning sets of one pair with one ``bincount`` per table shape; a single query is a batch
    of one, and a set's p-value is the same whatever batch it is in. Queries are canonicalized
    (``x``, ``y`` ordered, the conditioning set sorted unique) and nothing is memoized.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._codes, self._levels = encode_discrete(data)
        self._n = self._codes.shape[0]
        self._level_list = [int(r) for r in self._levels]
        self._strata_cache: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        A larger p-value means more independence. A query with no degrees of freedom left, where
        every stratum has one level of ``x`` or of ``y``, returns 1.
        """
        return self.batched_call(x, y, [condition_set])[0]

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return the p-values for one ``(x, y)`` pair over many conditioning sets, in order.

        Each set is canonicalized as in :meth:`__call__`, and its p-value equals that call's.
        """
        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        levels = self._level_list
        rx, ry = levels[x_int], levels[y_int]
        cells = rx * ry
        xy = self._codes[:, x_int] * ry + self._codes[:, y_int]
        out: list[float] = []
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        sparse: list[tuple[int, list[int]]] = []
        limit = max(_DENSE_ROWS * self._n, _DENSE_MIN)
        for i, condition_set in enumerate(condition_sets):
            cond = sorted(set(int(c) for c in condition_set))
            assert x_int not in cond and y_int not in cond, "x, y cannot be in condition_set."
            out.append(1.0)
            q = 1
            for c in cond:
                q *= levels[c]
            if q * cells > limit:
                sparse.append((i, cond))
            else:
                groups.setdefault(q, []).append((i, cond))
        codes = self._codes
        for q, members in groups.items():
            step = max(1, _BATCH_CELLS // (q * cells))
            for start in range(0, len(members), step):
                part = members[start : start + step]
                # Row b holds set b's stratum index, then its cell index offset into its own block.
                flat = np.empty((len(part), self._n), dtype=np.int64)
                for b, (_, cond) in enumerate(part):
                    flat[b] = self._stratum(cond)
                flat += (np.arange(len(part), dtype=np.int64) * q)[:, None]
                flat *= cells
                flat += xy
                counts = np.bincount(flat.ravel(), minlength=len(part) * q * cells)
                pvalues = self._pvalues(counts.reshape(len(part), q, rx, ry).astype(np.float64))
                for (i, _), p in zip(part, pvalues):
                    out[i] = float(p)
        for i, cond in sparse:
            strata, k = _strata(codes, self._levels, cond, cells)
            counts = np.bincount(strata * cells + xy, minlength=k * cells)
            out[i] = float(self._pvalues(counts.reshape(1, k, rx, ry).astype(np.float64))[0])
        return out

    def _stratum(self, cond: list[int]) -> np.ndarray:
        """Return each row's mixed-radix configuration of ``cond``, cached.

        PC reuses a conditioning set across the pairs it separates; the cache keeps the most recent
        sets, up to a byte budget.
        """
        key = tuple(cond)
        cache = self._strata_cache
        row = cache.get(key)
        if row is not None:
            cache.move_to_end(key)
            return row
        row = np.zeros(self._n, dtype=np.int64)
        for c in cond:
            row *= self._level_list[c]
            row += self._codes[:, c]
        cache[key] = row
        if len(cache) * row.nbytes > _STRATA_CACHE_BYTES:
            cache.popitem(last=False)
        return row

    def _pvalues(self, counts: np.ndarray) -> np.ndarray:
        """P-values for a ``(m, k, r_x, r_y)`` stack of tables; an empty stratum adds nothing."""
        rows = counts.sum(axis=3)  # (m, k, rx)
        cols = counts.sum(axis=2)  # (m, k, ry)
        total = rows.sum(axis=2)  # (m, k)
        seen = total > 0
        expected = (
            rows[..., None] * cols[..., None, :] / np.where(seen, total, 1.0)[..., None, None]
        )
        dof = np.sum(((rows > 0).sum(axis=2) - 1) * ((cols > 0).sum(axis=2) - 1) * seen, axis=1)
        stat = self._statistic(counts, expected)
        with np.errstate(invalid="ignore"):
            return np.where(dof > 0, chdtrc(np.maximum(dof, 1), stat), 1.0)

    def _statistic(self, counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ChiSquare(_ContingencyTest):
    """Pearson chi-square conditional-independence test for discrete data.

    The statistic sums ``(N - E)**2 / E`` over the cells of every stratum's ``x``-by-``y`` table,
    with ``E`` the count expected under independence within the stratum; cells with ``E == 0`` add
    nothing. Its null law is chi-square with ``sum_k (r_xk - 1) * (r_yk - 1)`` degrees of freedom,
    where ``r_xk`` and ``r_yk`` count the levels of ``x`` and ``y`` observed in stratum ``k``.
    """

    def _statistic(self, counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
        safe = np.where(expected > 0, expected, 1.0)
        return np.sum(np.where(expected > 0, (counts - expected) ** 2 / safe, 0.0), axis=(1, 2, 3))


class GSquare(_ContingencyTest):
    """Likelihood-ratio G-square conditional-independence test for discrete data.

    The statistic is ``2 * sum N * log(N / E)`` over the cells with ``N > 0``, twice the
    conditional mutual information of ``x`` and ``y`` given ``S`` times the sample size, against
    the same chi-square law as :class:`ChiSquare`.
    """

    def _statistic(self, counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
        seen = counts > 0
        ratio = np.where(seen, counts / np.where(seen, expected, 1.0), 1.0)
        return 2.0 * np.sum(counts * np.log(ratio), axis=(1, 2, 3))


class DiscreteScore:
    """The shared counting and memoization of :class:`BDeuScore` and :class:`DiscreteBICScore`.

    Every local score comes from :meth:`_batch`, which counts the node's states under each parent
    set with one ``bincount`` per table shape and hands the stack to the subclass's
    :meth:`_log_scores`; a single score is a batch of one, and a score is the same whatever batch it
    is in. :meth:`add_scores` scores every one-parent extension of a parent set at once.

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
        self._level_list = [int(r) for r in self.levels]
        self._cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self._cache_size = cache_size
        self._dense_cells = max(_DENSE_ROWS * self.n, _DENSE_MIN)

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized local score of ``node`` given ``parents`` (lower is better).

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = (int(node), tuple(sorted(int(p) for p in parents)))
        value = self._cache.get(key)
        if value is None:
            value = self._batch(key[0], [key[1]])[0]
            self._remember(key, value)
        return value

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the local scores for many parent sets, in request order."""
        return [self.score(node, parents) for parents in parent_sets]

    def add_scores(self, node: int, base: Sequence[int], candidates: Sequence[int]) -> np.ndarray:
        """Return ``score(node, base + u)`` for each ``u`` in ``candidates``, in order.

        Equal to the :meth:`score` of each extended set, and memoized with them. The extended
        sets' configurations are built from the base's prefixes and suffixes in sorted-parent
        order, so each costs a few vector operations whatever the size of ``base``.
        """
        node = int(node)
        base_sorted = sorted(int(p) for p in base)
        out = np.empty(len(candidates))
        todo: list[tuple[int, int, tuple[int, ...]]] = []
        for i, u in enumerate(candidates):
            key = (node, tuple(sorted([*base_sorted, int(u)])))
            value = self._cache.get(key)
            if value is None:
                todo.append((i, int(u), key[1]))
            else:
                out[i] = value
        if todo:
            values = self._batch(node, [parents for _, _, parents in todo], base=base_sorted)
            for (i, _, parents), value in zip(todo, values):
                out[i] = value
                self._remember((node, parents), value)
        return out

    def _remember(self, key: tuple[int, tuple[int, ...]], value: float) -> None:
        if self._cache_size is not None and len(self._cache) >= self._cache_size:
            self._cache.clear()  # values are pure, so a full cache only costs recomputation
        self._cache[key] = value

    def _batch(
        self, node: int, parent_sets: list[tuple[int, ...]], base: list[int] | None = None
    ) -> list[float]:
        """Score ``node`` under each parent set. With ``base``, each set is ``base`` plus one."""
        levels, codes = self._level_list, self._codes
        r = levels[node]
        out = [0.0] * len(parent_sets)
        groups: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for i, parents in enumerate(parent_sets):
            q = 1
            for c in parents:
                q *= levels[c]
            if q * r <= self._dense_cells:
                groups.setdefault(q, []).append((i, parents))
            else:
                strata, k = _strata(codes, self.levels, list(parents), r)
                counts = np.bincount(strata * r + codes[:, node], minlength=k * r)
                out[i] = self._log_scores(
                    node, counts.reshape(1, k, r).astype(np.float64), [parents], q
                )[0]
        parts = _BaseParts(codes, levels, base) if base is not None else None
        for q, members in groups.items():
            step = max(1, _BATCH_CELLS // (q * r))
            for start in range(0, len(members), step):
                part = members[start : start + step]
                if base == []:  # one parent each: its codes are the configuration
                    flat = codes[:, [parents[0] for _, parents in part]].T.copy()
                else:
                    flat = np.empty((len(part), self.n), dtype=np.int64)
                    for b, (_, parents) in enumerate(part):
                        if parts is not None:
                            parts.extended(flat[b], parents)
                        else:
                            _radix(flat[b], codes, levels, parents)
                flat += (np.arange(len(part), dtype=np.int64) * q)[:, None]
                flat *= r
                flat += codes[:, node]
                counts = np.bincount(flat.ravel(), minlength=len(part) * q * r)
                scores = self._log_scores(
                    node,
                    counts.reshape(len(part), q, r).astype(np.float64),
                    [parents for _, parents in part],
                    q,
                )
                for (i, _), value in zip(part, scores):
                    out[i] = float(value)
        return out

    def _log_scores(
        self, node: int, counts: np.ndarray, parent_sets: list[tuple[int, ...]], q: int
    ) -> np.ndarray:
        """Local scores from a ``(m, k, r)`` stack of counts; ``q`` configurations per set."""
        raise NotImplementedError


def _radix(row: np.ndarray, codes: np.ndarray, levels: list[int], columns: Sequence[int]) -> None:
    """Write each row's mixed-radix configuration of ``columns`` into ``row``, in place."""
    row[:] = 0
    for c in columns:
        row *= levels[c]
        row += codes[:, c]


class _BaseParts:
    """A parent set's configuration split at every position, for its one-parent extensions.

    For the sorted base ``b_1 < ... < b_m``, ``prefix[i]`` is the mixed-radix configuration of
    ``b_1 .. b_i`` and ``suffix[i]`` that of ``b_(i+1) .. b_m``, with ``span[i]`` its number of
    configurations. Inserting ``u`` at position ``i`` of the sorted order gives the configuration
    ``(prefix[i] * r_u + codes_u) * span[i] + suffix[i]``, exactly the mixed radix of the sorted
    extended set.
    """

    def __init__(self, codes: np.ndarray, levels: list[int], base: list[int]) -> None:
        self._codes, self._levels, self._base = codes, levels, base
        self._members = set(base)
        n = codes.shape[0]
        self._prefix = [np.zeros(n, dtype=np.int64)]
        for c in base:
            self._prefix.append(self._prefix[-1] * levels[c] + codes[:, c])
        self._suffix = [np.zeros(n, dtype=np.int64)]
        self._span = [1]
        for c in reversed(base):
            self._suffix.append(codes[:, c] * self._span[-1] + self._suffix[-1])
            self._span.append(self._span[-1] * levels[c])
        self._suffix.reverse()
        self._span.reverse()

    def extended(self, row: np.ndarray, parents: tuple[int, ...]) -> None:
        """Write the configuration of ``parents`` (the base plus one variable) into ``row``."""
        u = next(c for c in parents if c not in self._members)
        i = sum(1 for c in self._base if c < u)
        np.multiply(self._prefix[i], self._levels[u], out=row)
        row += self._codes[:, u]
        row *= self._span[i]
        row += self._suffix[i]


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

    def _log_scores(
        self, node: int, counts: np.ndarray, parent_sets: list[tuple[int, ...]], q: int
    ) -> np.ndarray:
        a_j = self.ess / q
        a_jk = a_j / self._level_list[node]
        n_j = counts.sum(axis=2)
        log_score = np.sum(gammaln(a_j) - gammaln(a_j + n_j), axis=1) + np.sum(
            gammaln(a_jk + counts) - gammaln(a_jk), axis=(1, 2)
        )
        k = np.array([len(parents) for parents in parent_sets])
        log_prior = k * self._log_p + (self._others - k) * self._log_not_p
        return -2.0 * (log_score + log_prior)


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

    def _log_scores(
        self, node: int, counts: np.ndarray, parent_sets: list[tuple[int, ...]], q: int
    ) -> np.ndarray:
        n_j = counts.sum(axis=2, keepdims=True)
        seen = counts > 0
        ratio = np.where(seen, counts / np.where(seen, n_j, 1.0), 1.0)
        log_lik = np.sum(counts * np.log(ratio), axis=(1, 2))
        n_params = q * (self._level_list[node] - 1)
        return -2.0 * log_lik + self.lambda_value * self.log_n * n_params
