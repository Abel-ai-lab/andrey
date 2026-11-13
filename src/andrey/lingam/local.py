"""Local ISA-LiNG, the engine behind :func:`andrey.local_isa_ling`.

For a target ``t``, the Markov blanket is the support of the target's row of the precision matrix,
read from a Lasso regression of ``t`` on every other column (Dai et al. 2024, App. A). On ``t`` and
its blanket, independent subspace analysis (ISA) gives a demixing matrix ``W`` and a partition of
its rows into irreducible subspaces: the sparse ICA of :mod:`andrey.lingam.sparse_ica`, then the
rows grouped by pairwise HSIC tests of their components (App. E.1). Entries of ``W`` below the
threshold read as zero. Algorithm 1 then assigns the rows to the variables in every admissible way
(Definition 4: each subspace's block of ``W`` is invertible), scales each block to the identity, and
reads ``B' = I - D W'``; the rows with a nonzero entry in the target's column stand for the target
and its children, and their rows of ``B'`` are the weighted edges into them (Theorem 3). Under the
acyclic option, Algorithm 3 (App. C) peels sink columns instead.

Every function here works on the columns it is given; :func:`local_isa_ling` standardizes them, so
the answer does not depend on their units, and returns standardized weights.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Lasso

from andrey.core.independence import gamma_gram, hsic_gamma_log_p
from andrey.lingam.sparse_ica import SparseICA, sparse_ica

#: The Lasso weight of the blanket's regression and the coefficient magnitude above which a
#: column joins the blanket (the authors' code).
BLANKET_PENALTY, BLANKET_COEF = 0.001, 0.01

#: An edge ``(source, child)`` with its weight.
Edges = dict[tuple[int, int], float]


@dataclass(frozen=True)
class LocalModel:
    """One model of the local answer: the weighted edges into the target and into its children.

    ``stable`` says whether some admissible assignment giving these edges has a convergent ``B'``
    (spectral radius below 1, Corollary 1), and ``cycle_strength`` is the least, over those
    assignments (over all of them when none is convergent), of ``sum_k sum |diag(B'^k)|`` for
    ``k = 2..m``. ``subspaces`` are the variables each subspace of rows stands for in the
    assignment that gives that least strength.
    """

    edges: Edges
    stable: bool
    cycle_strength: float
    subspaces: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LocalFit:
    """A fit: the blanket, the models in order (the first is the answer), and the ICA's run."""

    blanket: list[int]
    models: list[LocalModel]
    ica: SparseICA | None


def standardized(X: np.ndarray) -> np.ndarray:
    """Center each column and scale it to unit sample standard deviation."""
    Z = X - X.mean(axis=0)
    return Z / Z.std(axis=0, ddof=1)


def markov_blanket(Z: np.ndarray, t: int) -> list[int]:
    """The columns whose Lasso coefficient in the regression of ``t`` on the others exceeds
    ``BLANKET_COEF`` in magnitude, ascending."""
    others = np.delete(np.arange(Z.shape[1]), t)
    if others.size == 0:
        return []
    coef = Lasso(alpha=BLANKET_PENALTY).fit(Z[:, others], Z[:, t]).coef_
    return sorted(int(j) for j in others[np.abs(coef) > BLANKET_COEF])


def subspaces(components: np.ndarray, singles: list[int], alpha: float) -> list[list[int]]:
    """Group the rows of ``components`` (``m`` by ``n``) into independent subspaces.

    The rows in ``singles`` are subspaces of their own (Lemma 2). Every other pair of rows is
    tested with the gamma HSIC test, its bandwidth the median distance over the first 100 rows,
    unless a chain of dependent pairs already joins them; a p-value at most ``alpha`` joins a pair,
    and the subspaces are the connected groups, each listed ascending, in order of their first row.
    """
    m, n = components.shape
    group = list(range(m))

    def root(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    rest = [i for i in range(m) if i not in singles]
    grams = {i: gamma_gram(components[i]) for i in rest} if len(rest) > 1 else {}
    log_alpha = np.log(alpha)
    for i, j in itertools.combinations(rest, 2):
        a, b = root(i), root(j)
        if a != b and hsic_gamma_log_p(grams[i], grams[j], n) <= log_alpha:
            group[max(a, b)] = min(a, b)
    members: dict[int, list[int]] = {}
    for i in range(m):
        members.setdefault(root(i), []).append(i)
    return sorted(members.values())


def _full_rank(block: np.ndarray, tol: float) -> bool:
    """Whether every singular value of ``block`` exceeds ``tol``."""
    return bool(np.linalg.svd(block, compute_uv=False).min() > tol)


def admissible(W: np.ndarray, parts: list[list[int]], tol: float) -> list[list[tuple[int, ...]]]:
    """Every admissible assignment of the subspaces of rows to sets of columns (Definition 4).

    An assignment gives the ``k`` rows of each subspace ``k`` columns, each column to one subspace,
    such that the subspace's block of ``W`` has every singular value above ``tol``. Within a
    subspace the order of its columns does not matter (the block is scaled to the identity), so
    each assignment lists one ascending column tuple per subspace, in the order of ``parts``. The
    search fills the subspaces with the fewest candidate columns first and tries only columns on
    which some row of the subspace is nonzero.
    """
    m = W.shape[1]
    nonzero = W != 0
    reach = [np.flatnonzero(nonzero[rows].any(axis=0)) for rows in parts]
    order = sorted(range(len(parts)), key=lambda p: (len(reach[p]), p))
    found: list[list[tuple[int, ...]]] = []
    chosen: list[tuple[int, ...]] = [()] * len(parts)
    used = np.zeros(m, dtype=bool)

    def fill(k: int) -> None:
        if k == len(order):
            found.append(list(chosen))
            return
        p = order[k]
        rows = parts[p]
        free = [int(j) for j in reach[p] if not used[j]]
        for cols in itertools.combinations(free, len(rows)):
            if not _full_rank(W[np.ix_(rows, cols)], tol):
                continue
            chosen[p] = cols
            used[list(cols)] = True
            fill(k + 1)
            used[list(cols)] = False

    fill(0)
    return found


def _strength(B: np.ndarray) -> float:
    """``sum_k sum |diag(B^k)|`` for ``k = 2..m``: the weight of the cycles through ``B``."""
    total, power = 0.0, B
    for _ in range(len(B) - 1):
        power = power @ B
        total += float(np.abs(np.diagonal(power)).sum())
    return total


def algorithm1(W: np.ndarray, parts: list[list[int]], t: int, tol: float) -> list[LocalModel]:
    """Algorithm 1 on a thresholded demixing matrix: one model per distinct local answer.

    ``parts`` are the subspaces of rows, ``t`` the target's column. The answer of an admissible
    assignment holds, for each row ``i`` nonzero in column ``t`` (a subspace of its own) assigned to
    column ``j``, the edges ``k -> j`` of weight ``-W[i, k] / W[i, j]`` for each other ``k`` with
    ``W[i, k] != 0``. Assignments with the same answer form one model; the models are ordered
    stable first, then by least cycle strength, then by their edges.
    """
    m = W.shape[0]
    rows_t = [int(i) for i in np.flatnonzero(W[:, t])]
    where = {i: p for p, rows in enumerate(parts) for i in rows}
    found: dict[tuple, tuple[Edges, bool, float, tuple]] = {}
    for assignment in admissible(W, parts, tol):
        B = np.eye(m)
        for rows, cols in zip(parts, assignment, strict=True):
            scale = np.linalg.inv(W[np.ix_(rows, cols)])
            B[list(cols)] -= scale @ W[rows]
            B[np.ix_(cols, cols)] = 0.0  # the block scaled to the identity, without rounding
        edges: Edges = {}
        for i in rows_t:
            (j,) = assignment[where[i]]
            for k in np.flatnonzero(W[i]):
                if k != j:
                    edges[(int(k), j)] = float(-W[i, k] / W[i, j])
        key = tuple(sorted(edges.items()))
        stable = bool(np.abs(np.linalg.eigvals(B)).max() < 1)
        strength = _strength(B)
        spaces = tuple(sorted(tuple(cols) for cols in assignment))
        best = found.get(key)
        if best is None or (stable, -strength) > (best[1], -best[2]):
            found[key] = (edges, stable, strength, spaces)
    models = [LocalModel(e, s, c, sp) for e, s, c, sp in found.values()]
    return sorted(models, key=lambda md: (not md.stable, md.cycle_strength, sorted(md.edges)))


def algorithm3(W: np.ndarray, t: int) -> Edges | None:
    """Algorithm 3 (App. C): peel sink columns of a thresholded demixing matrix.

    A sink is a remaining column with exactly one nonzero entry in the remaining rows. Its row
    stands for it; when that row is nonzero in the target's column, the row gives the edges
    ``k -> j`` of weight ``-W[i, k] / W[i, j]`` from the remaining columns ``k``. The search stops
    after the target. Of several sinks, the one whose entry is largest in magnitude goes first.
    Returns ``None`` when no remaining column is a sink, which the method's acyclic assumption rules
    out.
    """
    cols = list(range(W.shape[1]))
    rows = list(range(W.shape[0]))
    edges: Edges = {}
    while cols:
        sub = W[np.ix_(rows, cols)] != 0
        sinks = [cols[c] for c in np.flatnonzero(sub.sum(axis=0) == 1)]
        if not sinks:
            return None
        j = max(sinks, key=lambda c: np.abs(W[rows, c]).max())
        i = rows[int(np.flatnonzero(W[rows, j])[0])]
        if W[i, t] != 0:
            for k in cols:
                if k != j and W[i, k] != 0:
                    edges[(k, j)] = float(-W[i, k] / W[i, j])
        if j == t:
            break
        cols.remove(j)
        rows.remove(i)
    return edges


def local_isa_ling(
    X: np.ndarray,
    t: int,
    *,
    acyclic: bool = False,
    w_threshold: float = 0.05,
    alpha: float = 0.05,
    blanket: list[int] | None = None,
) -> LocalFit:
    """Fit local ISA-LiNG for column ``t`` of ``X``; weights are those of standardized columns.

    ``blanket`` replaces the Lasso's blanket when given (a known blanket). Under ``acyclic``, a
    fit whose thresholded matrix has no sink falls back to Algorithm 1.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = standardized(X)
    mb = markov_blanket(Z, t) if blanket is None else sorted(blanket)
    if not mb:
        return LocalFit(mb, [LocalModel({}, True, 0.0, ((t,),))], None)
    local, lt = [t, *mb], 0  # the target first, as the authors' code orders the columns
    ica = sparse_ica(X[:, local])
    W = np.where(np.abs(ica.W) > w_threshold, ica.W, 0.0)
    back = np.asarray(local)
    if acyclic:
        edges = algorithm3(W, lt)
        if edges is not None:
            model = LocalModel(_names(edges, back), True, 0.0, ())
            return LocalFit(mb, [model], ica)
    singles = [int(i) for i in np.flatnonzero(W[:, lt])]
    parts = subspaces(W @ Z[:, local].T, singles, alpha)
    models = [
        LocalModel(
            _names(md.edges, back),
            md.stable,
            md.cycle_strength,
            tuple(sorted(tuple(sorted(int(back[c]) for c in cols)) for cols in md.subspaces)),
        )
        for md in algorithm1(W, parts, lt, w_threshold)
    ]
    return LocalFit(mb, models, ica)


def _names(edges: Edges, back: np.ndarray) -> Edges:
    """The edges with local positions turned back into column positions."""
    return {(int(back[k]), int(back[j])): w for (k, j), w in edges.items()}
