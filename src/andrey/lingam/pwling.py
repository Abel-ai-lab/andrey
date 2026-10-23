"""The pairwise-likelihood (pwling) exogeneity measure for LiNGAM ordering.

Scores each candidate variable by how plausibly exogenous it is -- the Hyvarinen maximum-entropy
approximation of the pairwise likelihood ratios (Hyvarinen & Smith 2013) -- and peels the
most-exogenous variable one at a time. Entropy is evaluated in batch over a stack of residuals.
Shared by the LiNGAM ordering methods (DirectLiNGAM, MultiGroupDirectLiNGAM, the temporal LiNGAMs).
"""

from __future__ import annotations

import numpy as np

from andrey.core import stats
from andrey.search.hc import tie_ranks

# A score is ``-sum_j min(0, diff_j)^2``, so ``sqrt(-score)`` is the norm of the negative entropy
# differences, and rounding moves it by at most the norm of their rounding errors. Mathematically
# tied candidates measure < 1e-12 apart in this norm and distinct ones > 5e-10 apart.
_TIE_TOL = 1e-10
# Two columns are copies when one leaves at most this fraction of the other's variance
# unexplained (1 - corr^2): their pairwise residuals are then zero or rounding noise, whose
# entropy differs by CPU. Copies, exact or perturbed by 1e-12, measure < 2e-15 at every step;
# rank-transformed near-copies with |corr| up to 0.999999 measure > 7e-6, and scoring both columns
# of a pair gave orders that differed between CPUs only below 1e-14.
_COPY_TOL = 1e-12


def compute_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``a`` regressed on ``b``: ``a - (cov(a, b) / var(b)) * b``.

    Covariance and variance share ``ddof=0``, so the residual is uncorrelated with ``b`` in the
    sample, as the order search assumes.
    """
    return a - (np.cov(a, b, ddof=0)[0, 1] / np.var(b)) * b


def score_candidates(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Score how plausibly exogenous each column in ``U`` of ``X`` is (higher = more exogenous).

    Returns an array aligned with ``U``: ``score(i) = -sum_j min(0, diff_mi(i, j))^2`` with
    ``diff_mi(i, j) = (H(x_j) + H(r_{i|j})) - (H(x_i) + H(r_{j|i}))`` on standardized columns.
    Each ``H(r_{i|j})`` is computed once and serves both ``i`` and ``j``, batched over the residuals
    of one column on the others; :func:`pick` chooses the best, MultiGroup sums across groups
    first. Order-exactness needs the float64 entropy path -- on a float32 device (atol 1e-3) a
    near-tie could flip the pick, so pin ``ANDREY_DEVICE=numpy`` for exact order.
    """
    cols = np.asarray(U)
    if len(cols) < 2:
        return np.zeros(len(cols), dtype=np.float64)  # a lone candidate has no pair to score
    sub = X[:, cols]
    x_std = (sub - sub.mean(axis=0)) / sub.std(axis=0)  # (n, m) standardized candidate columns
    ent = stats.entropy(x_std)  # (m,) batched candidate entropy
    cov = np.cov(x_std, rowvar=False, ddof=0)  # (m, m), same ddof as var: least-squares slopes
    var = np.var(x_std, axis=0)  # (m,), ddof=0
    n, m = x_std.shape
    # h[i, j] = H(r_{i|j}), computed once per ordered pair: candidate i reads its H(r_{i|j}) terms
    # from row i and its H(r_{j|i}) terms from column i.
    h = np.zeros((m, m), dtype=np.float64)
    r, work = np.empty((n, m - 1), order="F"), np.empty((n, m - 1), order="F")
    for a in range(m):
        others = np.arange(m) != a
        np.multiply(cov[a, others] / var[others], x_std[:, others], out=r)
        np.subtract(x_std[:, a][:, None], r, out=r)  # residual of i on each j
        r /= np.std(r, axis=0)
        h[a, others] = stats.entropy(r, work=work)
    scores = np.empty(m, dtype=np.float64)
    for a in range(m):
        others = np.arange(m) != a
        diff = (ent[others] + h[a, others]) - (ent[a] + h[others, a])
        scores[a] = -np.sum(np.minimum(0.0, diff) ** 2)
    return scores


def pick(scores: np.ndarray, candidates: list[int], ranks: np.ndarray) -> int:
    """The candidate with the highest score; the lowest ``ranks`` entry among tied candidates.

    ``scores`` are aligned with ``candidates`` and ``<= 0`` or NaN; two tie when their
    ``sqrt(-score)`` values both lie within ``_TIE_TOL`` of the best one. A NaN score ranks highest,
    as under ``np.argmax``; all NaN candidates then tie. ``ranks`` is indexed by column, as
    :func:`~andrey.search.hc.tie_ranks` returns for the input data.
    """
    norm = np.sqrt(-np.asarray(scores, dtype=np.float64))
    nan = np.isnan(norm)
    tied = np.flatnonzero(nan) if nan.any() else np.flatnonzero(norm <= norm.min() + _TIE_TOL)
    return min((candidates[t] for t in tied), key=lambda c: ranks[c])


def copy_groups(groups: list[np.ndarray], U: list[int], ranks: np.ndarray) -> dict[int, list[int]]:
    """Each candidate in ``U`` that represents its copies, mapped to those copies.

    Two columns are copies when, in any of ``groups``, one leaves at most ``_COPY_TOL`` of the
    other's variance unexplained; a copy group is a connected set of such pairs. Its
    representative is the column that ranks first by ``ranks`` (as :func:`pick` uses), and its
    copies follow in ``ranks`` order. Representatives keep their order in ``U``; a column without
    copies represents itself.
    """
    k = len(U)
    linked = np.zeros((k, k), dtype=bool)
    for X in groups:
        cov = np.cov(X[:, U], rowvar=False, ddof=0).reshape(k, k)
        var = np.diag(cov)
        linked |= cov * cov.T >= (1.0 - _COPY_TOL) * np.outer(var, var)
    np.fill_diagonal(linked, False)
    if not linked.any():
        return {u: [] for u in U}
    copies: dict[int, list[int]] = {}
    seen = np.zeros(k, dtype=bool)
    for start in range(k):
        if seen[start]:
            continue
        component, frontier = [start], [start]
        seen[start] = True
        while frontier:
            new = np.flatnonzero(linked[frontier].any(axis=0) & ~seen)
            seen[new] = True
            component += new.tolist()
            frontier = new.tolist()
        members = sorted((U[a] for a in component), key=lambda c: ranks[c])
        copies[members[0]] = members[1:]
    return {u: copies[u] for u in U if u in copies}


def find_causal_order(X: np.ndarray, *, precedence: np.ndarray | None = None) -> list[int]:
    """Peel exogenous variables (:func:`pick` of :func:`score_candidates`) one at a time.

    Each pick is residualized out of the rest. Only one column of each copy group
    (:func:`copy_groups`) is scored; when it is picked, its copies follow it directly in the order.
    Ties and copy representatives are decided by :func:`~andrey.search.hc.tie_ranks` of the input
    columns, which are identical on every machine and keep their ranks when the columns are
    reordered; residualized columns differ in the last bits across machines. Single-dataset;
    MultiGroupDirectLiNGAM runs its own group loop (per-group residualization), reusing
    :func:`copy_groups`, :func:`score_candidates`, and :func:`pick` directly. With ``precedence``
    (``precedence[a, b]``: ``a`` comes before ``b``, acyclic), no copy group is formed, since
    placing one at once could put a column before one that must precede it: every remaining
    variable is scored, and the pick is the best one that no remaining variable must precede.
    """
    remaining = list(range(X.shape[1]))
    order: list[int] = []
    residualized = np.array(X, dtype=np.float64)  # copy + float64: the write-back must not truncate
    ranks = tie_ranks(residualized)
    while remaining:
        if precedence is None:
            copies = copy_groups([residualized], remaining, ranks)
        else:
            copies = {u: [] for u in remaining}
        candidates = list(copies)
        # A lone candidate needs no scoring (and has no pair to score).
        if len(candidates) == 1:
            m = candidates[0]
        elif precedence is None:
            m = pick(score_candidates(residualized, candidates), candidates, ranks)
        else:
            eligible = ~precedence[np.ix_(candidates, candidates)].any(axis=0)
            values = score_candidates(residualized, candidates)[eligible]
            m = pick(values, [c for c, ok in zip(candidates, eligible) if ok], ranks)
        placed = [m, *copies[m]]
        remaining = [u for u in remaining if u not in placed]
        for i in remaining:
            residualized[:, i] = compute_residual(residualized[:, i], residualized[:, m])
        order += placed
    return order
