"""The nearest-neighbor conditional mutual information test (CMIknn), for nonlinear data.

Selected through ``indep_test="cmiknn"``. The test follows the :class:`~andrey.core.ci.CITest`
protocol: built once over the data and called as ``test(x, y, condition_set)``, returning a p-value
where a larger value means more independent.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator, Mapping
from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.special import digamma

from andrey.search.hc import tie_ranks

from . import backend, env
from .ci import _canonical_ranked, _checked_data, per_query
from .seeding import resolve_seed

__all__ = ["CMIknn"]

# The cache of CMIknn's shuffles holds about this many bytes.
_CACHE_BYTES = 2**28

# Smallest distance matrix (rows squared) at which `auto` runs a CMIknn query on a CUDA GPU.
# `qa/gpu_calibration_validate.py` on an A100-SXM4-80GB found its probe floor, 1,024 (32 rows); per
# query with 500 shuffles, the GPU was 8x the CPU's speed at 100 rows and 40x at 1,000.
_CMIKNN_GPU_THRESHOLDS: dict[str, int | None] = {"cuda": 1_024, "mps": None}

# Device memory one batch of shuffled estimates uses, about 20 bytes per distance-matrix entry.
_DEVICE_BATCH_BYTES = 2**30

# Rows of the distance matrices one step of the CPU estimate holds, so its temporaries stay small.
_BLOCK_ROWS = 64


def _partition_dtype(features: Mapping[str, bool]) -> type[np.integer]:
    """The integer type in which :func:`_cmi_given_yz` selects each row's ``k``-th distance, given
    the CPU features NumPy reports.

    NumPy selects 16-bit integers with SIMD only where the CPU has AVX512_ICL, and 32-bit integers
    with AVX2 or AVX-512. Without AVX512_ICL a 16-bit selection runs scalar, several times slower
    than the same values as 32-bit integers, so those CPUs widen. Where NumPy reports no AVX2, as
    on ARM, widening gains nothing. The values are integers below the rows, so the ``k``-th
    smallest is the same in either type.
    """
    return np.int16 if features.get("AVX512_ICL") or not features.get("AVX2") else np.int32


try:
    _PARTITION_DTYPE = _partition_dtype(
        importlib.import_module("numpy._core._multiarray_umath").__cpu_features__
    )
except (ImportError, AttributeError):  # NumPy moved its feature report: the type never 4x slower
    _PARTITION_DTYPE = np.int32


def _rank_columns(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank each column (0 to n - 1), ties ordered by a random key."""
    order = np.lexsort((rng.random(data.shape), data), axis=0)
    return np.argsort(order, axis=0)


def _distances(column: np.ndarray, dtype: type) -> np.ndarray:
    """All pairwise absolute differences of one ranked column, as an ``n x n`` integer matrix."""
    return np.abs(column[:, None] - column[None, :]).astype(dtype)


def _sorted_rows(d: np.ndarray) -> np.ndarray:
    """Each row of the ``n x n`` distance matrix ``d`` sorted and offset by ``n`` times its index,
    flattened: one sorted array, in which a binary search for ``i n + eps`` counts the entries of
    row ``i`` below ``eps``."""
    n = d.shape[0]
    dtype = np.int32 if n * n < 2**31 else np.int64
    return (np.sort(d, axis=1).astype(dtype) + np.arange(0, n * n, n, dtype=dtype)[:, None]).ravel()


def _yz_space(dyz: np.ndarray, dz: np.ndarray) -> tuple[np.ndarray, ...]:
    """The distances of the space of y and z, ``dyz = max(dy, dz)``, and of z, each with its
    :func:`_sorted_rows`, as :func:`_cmi_given_yz` reads them."""
    return dyz, dz, _sorted_rows(dyz), _sorted_rows(dz)


def _cmi_given_yz(x: np.ndarray, yz: tuple[np.ndarray, ...], k: int) -> float:
    """The nearest-neighbor CMI estimate from the ranked column ``x`` and the space ``yz`` of
    :func:`_yz_space`.

    The ``k``-th neighbor distance ``eps_i`` is taken in the joint space; each ``k_*,i`` counts the
    rows strictly closer than ``eps_i`` in a subspace, row ``i`` included. The joint distances and
    ``k_xz`` take ``_BLOCK_ROWS`` rows at a time, selecting ``eps`` as :func:`_partition_dtype`
    says; ``k_yz`` and ``k_z`` are binary searches in the sorted rows. The counts are integers,
    and the estimate is formed once from all of them.
    """
    dyz, dz, yz_rows, z_rows = yz
    n = x.shape[0]
    select = np.promote_types(x.dtype, _PARTITION_DTYPE)
    # A row of flags summed as bytes, in the narrowest type that holds its count, is several times
    # faster than `np.count_nonzero` along an axis.
    count = np.uint16 if n <= np.iinfo(np.uint16).max else np.intp
    eps = np.empty(n, dtype=x.dtype)
    k_xz = np.empty(n, dtype=np.intp)
    for start in range(0, n, _BLOCK_ROWS):
        rows = slice(start, start + _BLOCK_ROWS)
        dx = np.abs(x[rows, None] - x[None, :])
        joint = np.maximum(dx, dyz[rows]).astype(select, copy=False)
        joint.partition(k, axis=1)
        eps[rows] = joint[:, k]  # 0: row i itself
        closer = np.maximum(dx, dz[rows]) < eps[rows, None]
        k_xz[rows] = closer.view(np.uint8).sum(axis=1, dtype=count)
    offset = np.arange(0, n * n, n, dtype=yz_rows.dtype)
    k_yz = np.searchsorted(yz_rows, offset + eps) - offset
    k_z = np.searchsorted(z_rows, offset + eps) - offset
    return float(digamma(k) + np.mean(digamma(k_z) - digamma(k_xz) - digamma(k_yz)))


def _cmi_from_counts(
    k: int, k_xz: np.ndarray, k_yz: np.ndarray, k_z: np.ndarray
) -> Iterator[float]:
    """The estimate of :func:`_cmi_given_yz` for each row of the ``(b, n)`` neighbor counts."""
    for terms in digamma(k_z) - digamma(k_xz) - digamma(k_yz):
        yield float(digamma(k) + np.mean(terms))


def _counts_torch(dx, dyz, dz, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The neighbor counts of :func:`_cmi_given_yz` for a ``(b, n, n)`` stack of x distances, on
    the device of the torch tensors; the counts return as ``(b, n)`` arrays."""
    torch = backend.torch()
    eps = torch.maximum(dx, dyz).topk(k + 1, dim=-1, largest=False).values[..., k:]
    counts = (
        (torch.maximum(dx, dz) < eps).sum(dim=-1),
        (dyz < eps).sum(dim=-1),
        (dz < eps).sum(dim=-1),
    )
    return tuple(c.cpu().numpy() for c in counts)


def _nearest_in_z(dz: np.ndarray, k_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Each row's ``k_perm`` nearest rows in z, itself included; ties are broken at random."""
    # Distances are integers, so a uniform draw below 0.5 reorders only tied entries.
    keyed = dz + 0.5 * rng.random(dz.shape)
    return np.argpartition(keyed, k_perm - 1, axis=1)[:, :k_perm]


def _local_permutation(nearest: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A map of each row to one of its nearest rows in z, a repeat only when unavoidable.

    Each row's neighbors are shuffled; rows are visited in random order, and each takes the first
    of its neighbors no earlier row took, or its last neighbor when all are taken.
    """
    n, k_perm = nearest.shape
    neighbors = rng.permuted(nearest, axis=1)
    used = np.zeros(n, dtype=bool)
    perm = np.empty(n, dtype=np.intp)
    for i in rng.permutation(n):
        row = neighbors[i]
        m = 0
        j = row[0]
        while used[j] and m < k_perm - 1:
            m += 1
            j = row[m]
        perm[i] = j
        used[j] = True
    return perm


def _cmiknn_backend(n: int) -> str:
    """Where a CMIknn query on ``n`` rows runs: ``numpy``, or torch on ``cpu`` or ``cuda``."""
    manual = env.CMIKNN_GPU_THRESHOLD.read()
    return backend.select(
        ("numpy", "cpu", "cuda"),
        ("cuda", "numpy"),
        metric=n * n,
        thresholds=_CMIKNN_GPU_THRESHOLDS if manual is None else {"cuda": manual},
        amortized=True,
        op="cmiknn",
    )


class CMIknn:
    """Conditional mutual information test with a nearest-neighbor estimator (CMIknn).

    Each query estimates the conditional mutual information of ``x`` and ``y`` given the
    conditioning set from nearest-neighbor distances, then compares it with estimates on copies of
    the data where ``x`` is shuffled only among rows close in the conditioning set. That keeps
    the dependence of ``x`` on the conditioning set and removes any other.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is replaced by its ranks.
    k_cmi : float or int, default=0.1
        Nearest neighbors of the estimator: a fraction of the rows in ``(0, 1)``, or a count, an
        int from 1 to one less than the rows. More neighbors lower the estimate's variance and
        raise its bias toward zero, which costs power as the conditioning set grows.
    k_perm : int, default=5
        Nearest neighbors in the conditioning set among which ``x`` is shuffled.
    n_permutations : int, default=500
        Shuffled copies that form the null distribution.
    seed : int or None, default=None
        Seed for the tie-breaking and the shuffles. ``None`` uses ``ANDREY_SEED``, or ``0``
        when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has fewer than 2 rows, or contains NaN or infinity, ``k_cmi`` is
        neither a fraction in ``(0, 1)`` nor an int from 1 to one less than the rows, or
        ``k_perm`` or ``n_permutations`` is below 1.

    Notes
    -----
    **How it works.** Each column is replaced by its ranks, ties ordered at random, and distances
    are in the maximum norm. With ``k`` the neighbor count (``round(k_cmi n)`` for a fraction),
    ``eps_i`` the distance from row ``i`` to its ``k``-th nearest row in the space of ``x``, ``y``,
    and the conditioning set ``z``, and ``k_xz``, ``k_yz``, ``k_z`` the rows strictly closer than
    ``eps_i`` in each subspace (row ``i`` counted), the estimate is ``I = psi(k) + mean_i [psi(k_z)
    - psi(k_xz) - psi(k_yz)]``, ``psi`` the digamma function [1]_. Each null copy maps every row to
    one of its ``k_perm`` nearest rows in ``z``, avoiding a row already taken where it can. With
    ``b`` of the ``B`` copies estimating at least ``I``, the p-value is ``(b + 1) / (B + 1)``: the
    observed data count as one more draw, so the p-value is never zero [3]_. In the skeleton
    searches of PC, FCI, GFCI, and CD-NOD, a query's shuffles stop once ``p > alpha`` or ``p <=
    alpha`` is certain whatever the remaining shuffles give, so the search decides as with every
    shuffle; where the search orients colliders by p-value (``collider_order="p_value"``), only ``p
    <= alpha`` stops them, so every p-value it ranks is the full one. An empty conditioning set
    shuffles ``x`` over all rows, and the estimate becomes the mutual information estimate of ``x``
    and ``y`` [2]_. The default ``k_cmi``, a tenth of the rows, is the low end of the range [1]_
    recommends; a larger share smooths the estimate, which costs power as the conditioning set
    grows.

    :meth:`statistic` returns the estimate ``I`` itself, in nats, for use as a dependence measure.
    It is biased on finite samples, so compare estimates at the same rows and ``k``: on Gaussian
    pairs with 1,000 rows and ``k = 5`` it reads 0.01 to 0.05 above the true value, independent
    pairs included (0.03 to 0.09 with 200 rows), and with ``k`` a tenth of the rows it reads near
    zero for independent pairs and low for strong dependence, 0.06 below at a correlation of 0.9
    and 0.26 below given one more variable. The p-value compares ``I`` with shuffled copies, which
    share the bias.

    Of ``x`` and ``y``, the one with the lower :func:`~andrey.search.hc.tie_ranks` rank, a rank
    computed from the data, is shuffled. The tie-breaking is drawn once, from
    ``seed``, when the test is built, and each column takes its keys by that rank. The shuffles
    come from a generator seeded by ``seed`` and the ranks of the conditioning set, so every query
    with the same conditioning set uses the same shuffles, and the test object keeps them for later
    queries. A p-value does not depend on the order of the queries, of the arguments, or of the
    columns.

    The distances and neighbor counts run on a CUDA GPU under the ``cuda`` backend, or under
    ``auto`` from ``ANDREY_CMIKNN_GPU_THRESHOLD`` rows squared, a batch of shuffles at a time. The
    counts are integers, so the p-value equals the CPU's.

    **Limitations.** Measured in PC on simulated data from sparse random graphs (mean degree 2),
    each variable a weighted sum of ``z + sin(z)``, or of ``cos(z)``, over its parents plus Gaussian
    noise, with 20 datasets per size on 4 cores. At 5 variables with 500 to 2,000 rows, 10 with 500
    and 1,000, and 20 with 500, its median errors (SHD) stayed below the empty graph's at every
    size. The median fit took 47 seconds at 5 variables and 500 rows, and 12 to 14 minutes at 5
    variables with 2,000 rows, 10 with 1,000, and 20 with 500. Each query computes
    ``n_permutations + 1`` estimates, each over ``n x n`` distance matrices, so its time grows with
    the square of the number of rows times the permutations, and its memory with the square of the
    rows: hundreds of rows take seconds per query, and tens of thousands do not fit. The p-value
    moves with ``seed``, is a multiple of ``1 / (n_permutations + 1)``, and is at least that. The
    numbers are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

    References
    ----------
    .. [1] Runge, J. (2018). Conditional independence testing based on a nearest-neighbor
       estimator of conditional mutual information. AISTATS, PMLR 84, 938-947.
    .. [2] Kraskov, A., Stogbauer, H., and Grassberger, P. (2004). Estimating mutual information.
       Physical Review E 69, 066138.
    .. [3] Phipson, B., and Smyth, G. K. (2010). Permutation p-values should never be zero:
       calculating exact p-values when permutations are randomly drawn. Statistical Applications
       in Genetics and Molecular Biology 9(1), Article 39.
    """

    pin_blas = True  # read by pins_blas: the skeleton scans run it on one BLAS thread

    def __init__(
        self,
        data: np.ndarray,
        *,
        k_cmi: float = 0.1,
        k_perm: int = 5,
        n_permutations: int = 500,
        seed: int | None = None,
    ) -> None:
        count = isinstance(k_cmi, Integral) and not isinstance(k_cmi, bool)
        if not (k_cmi >= 1 if count else 0.0 < k_cmi < 1.0):
            raise ValueError(
                f"k_cmi must be a fraction of the rows in (0, 1) or an int count of at least 1, "
                f"got {k_cmi!r}"
            )
        if k_perm < 1 or n_permutations < 1:
            raise ValueError(
                f"k_perm and n_permutations must be at least 1, got {k_perm} and {n_permutations}"
            )
        arr = _checked_data(data, "CMIknn", min_rows=2)
        # The test holds data column j at index _tie_rank[j], which picks its tie-breaking keys
        # and seeds the shuffles of the conditioning sets that hold it.
        self._tie_rank = tie_ranks(arr).tolist()
        arr = arr[:, np.argsort(self._tie_rank)]
        self._seed = resolve_seed(seed)
        n = arr.shape[0]
        if count and k_cmi >= n:
            raise ValueError(f"k_cmi={k_cmi} neighbors needs more rows than {n}")
        self._n = n
        self._k = int(k_cmi) if count else min(max(1, round(k_cmi * n)), n - 1)
        self._k_perm = min(int(k_perm), n)
        self._n_permutations = int(n_permutations)
        self._dtype = np.int16 if n <= np.iinfo(np.int16).max else np.int32
        # Ranks in the distance dtype: a difference of two ranks fits it, so distances never widen.
        self._ranks = _rank_columns(arr, np.random.default_rng(self._seed)).astype(self._dtype)
        # The shuffles depend only on the conditioning set, so queries share them.
        entries = max(16, _CACHE_BYTES // (self._n_permutations * n * self._dtype().itemsize))
        self._permutations = lru_cache(entries)(self._draw_permutations)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the CMIknn p-value for ``x`` independent of ``y`` given ``condition_set``."""
        return self._bounded_pvalue(x, y, condition_set, None)

    def _bounded_pvalue(
        self,
        x: int,
        y: int,
        condition_set: Iterable[int],
        alpha: float | None,
        exact_above: bool = False,
    ) -> float:
        """The p-value, or with ``alpha`` given, a bound on it on the same side of ``alpha``.

        The skeleton scans pass their ``alpha``. With ``b`` of the shuffles so far estimating at
        least the observed value and ``r`` left, the p-value ``(b' + 1) / (B + 1)`` lies between
        ``(b + 1) / (B + 1)`` and ``(b + r + 1) / (B + 1)``, so the shuffles stop once the first
        exceeds ``alpha`` (that bound is returned) or the second does not (that one is). The
        division is the full p-value's and is monotone in the numerator, so ``p > alpha`` is
        decided as with every shuffle. With ``exact_above``, only the second stop applies, so a
        p-value above ``alpha`` is the full one: the scans that rank colliders by p-value pass it.
        """
        x_int, y_int, cond = _canonical_ranked(x, y, condition_set, self._tie_rank)
        cond = tuple(cond)
        dev = _cmiknn_backend(self._n)
        if dev == "numpy":
            hits = self._hits(x_int, y_int, cond)
        else:
            hits = self._hits_torch(x_int, y_int, cond, dev)
        total = self._n_permutations + 1
        exceed = 0
        for left, hit in zip(range(self._n_permutations, 0, -1), hits):
            if alpha is not None:
                if not exact_above and (exceed + 1) / total > alpha:
                    return (exceed + 1) / total
                if (exceed + left + 1) / total <= alpha:
                    return (exceed + left + 1) / total
            exceed += hit
        return (exceed + 1) / total

    def statistic(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the estimate of the conditional mutual information of ``x`` and ``y`` given
        ``condition_set``, in nats.

        The estimate the p-value compares with its shuffles, from the test's ranks (see Notes);
        with an empty ``condition_set``, the mutual information estimate. It does not depend on
        the order of the arguments.
        """
        x_int, y_int, cond = _canonical_ranked(x, y, condition_set, self._tie_rank)
        return _cmi_given_yz(self._ranks[:, x_int], self._yz_space(y_int, tuple(cond)), self._k)

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The estimates of :meth:`statistic` and the p-values of many queries, one per row.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` holds query ``i``'s estimate and :meth:`__call__`'s p-value.
        """
        return per_query(lambda a, b, s: (self.statistic(a, b, s), self(a, b, s)), x, y, conditions)

    def _hits(self, x: int, y: int, cond: tuple[int, ...]) -> Iterator[bool]:
        """Whether each shuffle's estimate reaches the observed one, computed as they are read."""
        ranks, k = self._ranks, self._k
        yz = self._yz_space(y, cond)
        observed = _cmi_given_yz(ranks[:, x], yz, k)
        for perm in self._permutations(cond):
            yield _cmi_given_yz(ranks[perm, x], yz, k) >= observed

    def _yz_space(self, y: int, cond: tuple[int, ...]) -> tuple[np.ndarray, ...]:
        """The :func:`_yz_space` of ``y`` and the conditioning set."""
        dz = self._cond_distances(cond)
        return _yz_space(np.maximum(_distances(self._ranks[:, y], self._dtype), dz), dz)

    def _hits_torch(self, x: int, y: int, cond: tuple[int, ...], dev: str) -> Iterator[bool]:
        """:meth:`_hits` with the distances and neighbor counts on ``dev``, a batch of shuffles at
        a time. The counts are integers and the estimates are formed from them as :meth:`_hits`
        forms them, so every estimate is the same."""
        torch = backend.torch()
        n, k = self._n, self._k
        ranks = torch.as_tensor(self._ranks[:, [x, y, *cond]], device=dev).to(torch.int32)

        def distances(column):
            return (column[..., :, None] - column[..., None, :]).abs()

        dz = torch.zeros((n, n), dtype=torch.int32, device=dev)
        for j in range(2, ranks.shape[1]):
            dz = torch.maximum(dz, distances(ranks[:, j]))
        dyz = torch.maximum(distances(ranks[:, 1]), dz)
        (observed,) = _cmi_from_counts(k, *_counts_torch(distances(ranks[None, :, 0]), dyz, dz, k))
        perms = torch.as_tensor(self._permutations(cond), device=dev).long()
        batch = max(1, _DEVICE_BATCH_BYTES // (20 * n * n))
        for start in range(0, perms.shape[0], batch):
            shuffled = ranks[:, 0][perms[start : start + batch]]
            for estimate in _cmi_from_counts(k, *_counts_torch(distances(shuffled), dyz, dz, k)):
                yield estimate >= observed

    def _cond_distances(self, cond: tuple[int, ...]) -> np.ndarray:
        """Max-norm distances between the rows in the conditioning set; zero when it is empty."""
        dz = np.zeros((self._n, self._n), dtype=self._dtype)
        for c in cond:
            np.maximum(dz, _distances(self._ranks[:, c], self._dtype), out=dz)
        return dz

    def _draw_permutations(self, cond: tuple[int, ...]) -> np.ndarray:
        """The ``n_permutations`` shuffles of the rows for ``cond``, one per row of the result,
        from a generator seeded by the seed and ``cond``."""
        rng = np.random.default_rng([self._seed, len(cond), *cond])
        n, dtype = self._n, self._dtype
        perms = np.empty((self._n_permutations, n), dtype=dtype)
        if not cond:
            for row in perms:
                row[:] = rng.permutation(n)
            return perms
        nearest = _nearest_in_z(self._cond_distances(cond), self._k_perm, rng)
        for row in perms:
            row[:] = _local_permutation(nearest, rng)
        return perms
