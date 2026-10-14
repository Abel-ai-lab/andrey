"""The nearest-neighbor conditional mutual information test (CMIknn), for nonlinear data.

Selected through ``indep_test="cmiknn"``. The test follows the :class:`~andrey.core.ci.CITest`
protocol: built once over the data and called as ``test(x, y, condition_set)``, returning a p-value
where a larger value means more independent.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.special import digamma

from .ci import _canonical, _checked_data
from .seeding import resolve_seed


def _rank_columns(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank each column (0 to n - 1) after a jitter far below its spread breaks ties at random."""
    spread = data.std(axis=0)
    spread[spread == 0] = 1.0
    jittered = data + 1e-9 * spread * rng.standard_normal(data.shape)
    return np.argsort(np.argsort(jittered, axis=0), axis=0)


def _distances(column: np.ndarray, dtype: type) -> np.ndarray:
    """All pairwise absolute differences of one ranked column, as an ``n x n`` integer matrix."""
    return np.abs(column[:, None] - column[None, :]).astype(dtype)


def _cmi(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray, k: int) -> float:
    """The nearest-neighbor CMI estimate from max-norm distance matrices of x, y, and z.

    The ``k``-th neighbor distance ``eps_i`` is taken in the joint space; each ``k_*,i`` counts the
    rows strictly closer than ``eps_i`` in a subspace, row ``i`` included.
    """
    dxz = np.maximum(dx, dz)
    dyz = np.maximum(dy, dz)
    joint = np.maximum(dxz, dy)
    eps = np.partition(joint, k, axis=1)[:, k][:, None]  # position 0 holds row i itself
    k_xz = np.count_nonzero(dxz < eps, axis=1)
    k_yz = np.count_nonzero(dyz < eps, axis=1)
    k_z = np.count_nonzero(dz < eps, axis=1)
    return float(digamma(k) + np.mean(digamma(k_z) - digamma(k_xz) - digamma(k_yz)))


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
    k_cmi : float, default=0.1
        Nearest neighbors of the estimator, as a fraction of the rows. More neighbors lower the
        estimate's variance and raise its bias toward zero, which costs power as the conditioning
        set grows.
    k_perm : int, default=5
        Nearest neighbors in the conditioning set among which ``x`` is shuffled.
    n_permutations : int, default=500
        Shuffled copies that form the null distribution.
    seed : int or None, default=None
        Seed for the tie-breaking jitter and the shuffles. ``None`` uses ``ANDREY_SEED``, or ``0``
        when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or contains NaN or infinity, ``k_cmi`` is not in ``(0, 1)``, or
        ``k_perm`` or ``n_permutations`` is below 1.

    Notes
    -----
    **How it works.** A tiny random jitter breaks ties, then each column is replaced by its ranks,
    and distances are in the maximum norm. With ``k = round(k_cmi n)``, ``eps_i`` the distance
    from row ``i`` to its ``k``-th nearest row in the space of ``x``, ``y``, and the conditioning
    set ``z``, and ``k_xz``, ``k_yz``, ``k_z`` the rows strictly closer than ``eps_i`` in each
    subspace (row ``i`` counted), the estimate is
    ``I = psi(k) + mean_i [psi(k_z) - psi(k_xz) - psi(k_yz)]``, ``psi`` the digamma function.
    Each null copy maps every row to one of its ``k_perm`` nearest rows in ``z``, avoiding a row
    already taken where it can; the p-value is the share of copies whose estimate is at least
    ``I``. An empty conditioning set shuffles ``x`` over all rows, and the estimate becomes the
    mutual information of ``x`` and ``y``.

    The jitter is drawn once, from ``seed``, when the test is built; the shuffles of a query come
    from a generator seeded by ``seed``, the two variables, and the conditioning set, so a p-value
    does not depend on the order of queries or of the arguments.

    **Limitations.** Not benchmarked yet. Each query computes ``n_permutations + 1`` estimates,
    each over ``n x n`` distance matrices, so its time grows with the square of the number of rows
    times the permutations, and its memory with the square of the rows: hundreds of rows take
    seconds per query, and tens of thousands do not fit. The p-value moves with ``seed`` and
    cannot fall between multiples of ``1 / n_permutations``. The unit tests run it on up to 200
    rows and two conditioning variables.

    References
    ----------
    .. [1] Runge, J. (2018). Conditional independence testing based on a nearest-neighbor
       estimator of conditional mutual information. AISTATS, PMLR 84, 938-947.
    .. [2] Kraskov, A., Stogbauer, H., and Grassberger, P. (2004). Estimating mutual information.
       Physical Review E 69, 066138.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        k_cmi: float = 0.1,
        k_perm: int = 5,
        n_permutations: int = 500,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < k_cmi < 1.0:
            raise ValueError(f"k_cmi must be a fraction of the rows in (0, 1), got {k_cmi}")
        if k_perm < 1 or n_permutations < 1:
            raise ValueError(
                f"k_perm and n_permutations must be at least 1, got {k_perm} and {n_permutations}"
            )
        arr = _checked_data(data, "CMIknn")
        self._seed = resolve_seed(seed)
        self._ranks = _rank_columns(arr, np.random.default_rng(self._seed))
        n = arr.shape[0]
        self._n = n
        self._k = min(max(1, round(k_cmi * n)), n - 1)
        self._k_perm = min(int(k_perm), n)
        self._n_permutations = int(n_permutations)
        self._dtype = np.int16 if n <= np.iinfo(np.int16).max else np.int32

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the CMIknn p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        rng = np.random.default_rng([self._seed, x_int, y_int, len(cond), *cond])
        ranks, dtype = self._ranks, self._dtype
        dx = _distances(ranks[:, x_int], dtype)
        dy = _distances(ranks[:, y_int], dtype)
        dz = np.zeros_like(dx)
        for c in cond:
            np.maximum(dz, _distances(ranks[:, c], dtype), out=dz)
        observed = _cmi(dx, dy, dz, self._k)
        nearest = _nearest_in_z(dz, self._k_perm, rng) if cond else None
        exceed = 0
        for _ in range(self._n_permutations):
            if nearest is None:
                perm = rng.permutation(self._n)
            else:
                perm = _local_permutation(nearest, rng)
            exceed += _cmi(dx[np.ix_(perm, perm)], dy, dz, self._k) >= observed
        return exceed / self._n_permutations
