"""Causal additive models: neighbor preselection, a greedy order search, and pruning (CAM).

CAM (Buhlmann, Peters, Ernest 2014) models each variable as an additive function of its parents
plus Gaussian noise, ``x_j = sum_k f_jk(x_k) + e_j``, and runs three steps:

1. **Preliminary neighbor selection.** Each variable is regressed on all the others by
   componentwise L2 boosting (Buhlmann and Yu 2003; Buhlmann and Hothorn 2007) with one P-spline
   base learner per other variable (Eilers and Marx 1996). The possible parents of a variable are
   the ten others picked most often in 100 boosting iterations, among those picked at least three
   times (Section 5.1), ties going to the earlier column.
2. **Greedy order search.** Starting from the empty graph, the search adds the edge ``i -> j``
   whose additive-model fit lowers ``log`` of ``j``'s residual variance the most, among the
   possible parents that close no cycle, and refits only ``j``'s candidates after each addition,
   until no candidate is left (Section 5.2). Each fit is :func:`andrey.fcm.gam.fit_gam` with ten
   basis functions per parent.
3. **Pruning.** Each variable is fit on its parents in that DAG, and a parent stays when its smooth
   term's p-value is at most the cutoff, ``0.001`` in the paper (Section 5.3,
   :func:`andrey.fcm.pruning.prune`).

The boosting settings are the paper's: step size ``0.1``, and per base
learner a cubic B-spline basis on 20 equidistant interior knots over the variable's range (24
functions), a second-order difference penalty, and the penalty weight that gives the learner 4
degrees of freedom, ``tr(2S - S'S)`` for its smoother ``S``. Every step is deterministic and unit
free: the base learners and the smooth terms depend on a variable's values only through an affine
map to a fixed interval, and the gains compare residual variances as ratios.

References
----------
Buhlmann, Peters, Ernest. "CAM: Causal additive models, high-dimensional order search and
penalized regression." Annals of Statistics 42(6):2526-2556, 2014.

Buhlmann, Yu. "Boosting with the L2 loss: regression and classification." JASA 98:324-339, 2003.

Buhlmann, Hothorn. "Boosting algorithms: regularization, prediction and model fitting."
Statistical Science 22(4):477-505, 2007.

Eilers, Marx. "Flexible smoothing with B-splines and penalties." Statistical Science
11(2):89-121, 1996.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, sparse
from scipy.interpolate import BSpline

from andrey.fcm.gam import BasisCache, fit_gam
from andrey.fcm.pruning import prune
from andrey.search._parallel import _KEY_DP

_log = logging.getLogger(__name__)

#: Boosting iterations of the neighbor selection.
N_ITER = 100
#: Boosting step size.
STEP = 0.1
#: Fewest boosting picks of a possible parent, and the most possible parents per variable.
MIN_PICKS = 3
MAX_NEIGHBORS = 10
#: Basis functions per parent in the order search and the pruning.
K = 10
#: P-spline base learner: interior knots, degree, difference order, degrees of freedom.
N_KNOTS = 20
DEGREE = 3
DIFFERENCES = 2
DF = 4.0
#: Weight of the difference penalty added to ``X'X`` before the degrees-of-freedom search, which
#: keeps the factorization defined when ``X'X`` is singular.
_RIDGE = 1e-9
#: Largest penalty weight the degrees-of-freedom search tries.
_LAMBDA_MAX = 1e15


@dataclass(frozen=True)
class PSpline:
    """One variable's P-spline base learner: its sparse design and its smoother.

    Attributes
    ----------
    index, value : np.ndarray
        ``(n, DEGREE + 1)`` columns and values of each row's nonzero B-splines.
    inverse : np.ndarray
        ``(m, m)`` inverse ``P`` of ``X'X + lam D'D`` for the ``(n, m)`` design ``X`` and the
        penalty weight ``lam``, or the pseudo-inverse of ``X'X`` when ``lam`` is ``0``.
    explained : np.ndarray
        ``(m, m)`` matrix ``2P - P X'X P``: the fit to a response ``u`` lowers the residual sum of
        squares by ``b' explained b`` for ``b = X'u``.
    """

    index: np.ndarray
    value: np.ndarray
    inverse: np.ndarray
    explained: np.ndarray


def pspline(x: np.ndarray) -> PSpline:
    """Build the P-spline base learner of covariate ``x``.

    The knots are ``N_KNOTS`` equidistant interior points of ``[min x, max x]`` plus ``DEGREE``
    more at the same spacing beyond each end. The penalty weight solves
    ``tr(2S - S'S) = DF`` by Brent's method, with the eigenvalues of the penalty taken relative to
    ``X'X + 1e-9 D'D``; a design of rank at most ``DF`` is not penalized.
    """
    lo, hi = float(x.min()), float(x.max())
    inner = np.linspace(lo, hi, N_KNOTS + 2)[1:-1]
    dx = (hi - lo) / (N_KNOTS + 1)
    t = np.concatenate(
        [
            np.linspace(lo - DEGREE * dx, lo, DEGREE + 1),
            inner,
            np.linspace(hi, hi + DEGREE * dx, DEGREE + 1),
        ]
    )
    design = BSpline.design_matrix(x, t, DEGREE)
    m = design.shape[1]
    index = design.indices.reshape(-1, DEGREE + 1)
    value = design.data.reshape(-1, DEGREE + 1)
    dense = design.toarray()
    xtx = dense.T @ dense
    D = np.diff(np.eye(m), n=DIFFERENCES, axis=0)
    penalty = D.T @ D
    lam = _df_lambda(xtx, penalty, rank=int(np.linalg.matrix_rank(dense)))
    if lam == 0.0:
        inverse = np.linalg.pinv(xtx, hermitian=True)
    else:
        inverse = linalg.inv(xtx + lam * penalty)
    explained = 2.0 * inverse - inverse @ xtx @ inverse
    return PSpline(index=index, value=value, inverse=inverse, explained=explained)


def _df_lambda(xtx: np.ndarray, penalty: np.ndarray, *, rank: int) -> float:
    """The penalty weight that gives the smoother ``DF`` degrees of freedom, ``tr(2S - S'S)``."""
    if DF >= rank:
        return 0.0
    A = _positive_semidefinite(xtx + _RIDGE * penalty)
    root = linalg.solve_triangular(linalg.cholesky(A), np.eye(A.shape[0]))
    d = linalg.svdvals(root.T @ penalty @ root)
    if DF >= d.size:
        return 0.0

    def excess(lam: float) -> float:
        shrink = 1.0 / (1.0 + lam * d)
        return float(2.0 * shrink.sum() - (shrink * shrink).sum() - DF)

    if excess(_LAMBDA_MAX) > 0:
        return _LAMBDA_MAX
    return float(optimize.brentq(excess, 0.0, _LAMBDA_MAX, xtol=np.finfo(float).eps ** 0.5))


def _positive_semidefinite(A: np.ndarray) -> np.ndarray:
    """``A``, shrunk toward the identity until its smallest eigenvalue is at least about 0."""
    tol = np.finfo(float).eps ** 0.5
    low = float(np.linalg.eigvalsh(A)[0]) - tol
    if low < -1e-10:
        rho = 1.0 / (1.0 - low)
        return _positive_semidefinite(rho * A + (1.0 - rho) * np.eye(A.shape[0]))
    return A


def boosting_counts(X: np.ndarray, *, n_iter: int = N_ITER, step: float = STEP) -> np.ndarray:
    """Count how often componentwise L2 boosting picks each variable to fit each other one.

    For response ``j`` the fit starts at the mean of ``x_j``. Each iteration fits every other
    variable's base learner to the current residuals by penalized least squares, takes the one with
    the smallest residual sum of squares (the first in column order on an exact tie), and adds
    ``step`` times its fit. The residual sums of squares are compared through the amount each fit
    lowers them, a quadratic form in the ``X'u`` of each learner, so only the picked learner's fit
    is evaluated on the rows.

    Returns
    -------
    np.ndarray
        ``(d, d)`` integers; ``counts[i, j]`` is the number of the ``n_iter`` iterations for
        response ``j`` that picked variable ``i``.
    """
    n, d = X.shape
    counts = np.zeros((d, d), dtype=np.int64)
    if d < 2:
        return counts
    learners = [pspline(X[:, i]) for i in range(d)]
    m = learners[0].inverse.shape[0]
    index = np.stack([b.index for b in learners])
    value = np.stack([b.value for b in learners])
    explained = np.stack([b.explained for b in learners])
    rows = (np.arange(d)[:, None, None] * m + index).ravel()
    cols = np.broadcast_to(np.arange(n)[None, :, None], index.shape).ravel()
    transpose = sparse.csr_array((value.ravel(), (rows, cols)), shape=(d * m, n))
    for j in range(d):
        y = X[:, j]
        fit = np.full(n, y.mean())
        u = y - fit
        for _ in range(n_iter):
            b = (transpose @ u).reshape(d, m)
            drop = np.einsum("ik,ikl,il->i", b, explained, b)
            drop[j] = -np.inf
            pick = int(np.argmax(drop))
            counts[pick, j] += 1
            coef = learners[pick].inverse @ b[pick]
            fit = fit + step * np.sum(value[pick] * coef[index[pick]], axis=1)
            u = y - fit
    return counts


def neighbors(
    counts: np.ndarray, *, min_picks: int = MIN_PICKS, max_neighbors: int = MAX_NEIGHBORS
) -> np.ndarray:
    """The possible parents from the boosting counts: ``out[i, j]`` when ``i`` may cause ``j``.

    The possible parents of ``j`` are the ``max_neighbors`` variables picked most often among those
    picked at least ``min_picks`` times, ties going to the earlier column.
    """
    out = np.zeros(counts.shape, dtype=bool)
    for j in range(counts.shape[1]):
        picked = np.flatnonzero(counts[:, j] >= min_picks)
        most = picked[np.argsort(-counts[picked, j], kind="stable")]
        out[most[:max_neighbors], j] = True
    return out


def order_search(
    bases: BasisCache,
    candidates: np.ndarray,
    *,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
    max_parents: int | None = None,
    k: int = K,
) -> np.ndarray:
    """Greedily add the edges that most lower the sum of log residual variances.

    The search starts from the ``required`` edges, each variable's residual variance that of its
    fit on its required parents (its sample variance with none). The gain of ``i -> j`` is
    ``log(v_j / v_j')``, ``v_j`` the residual variance (denominator ``n - 1``) of ``j``'s current
    fit and ``v_j'`` that of its fit with ``i`` added. The search adds the edge of largest gain,
    whatever its sign. Gains are compared rounded to ``_KEY_DP`` decimal places, as
    :func:`andrey.hc` compares its score changes per row, so two gains that differ only in their
    rounding tie, and a tie goes to the smallest ``(i, j)``, the cause first; then it
    drops every candidate that would close a cycle or break ``precedence`` and refits ``j``'s
    remaining candidates. It stops when no candidate is left. A candidate whose model would have as
    many coefficients as rows is never added, and a node with ``max_parents`` parents takes no
    more.

    Parameters
    ----------
    bases : BasisCache
        Bases of the data's columns, each of rank at most its column's distinct values.
    candidates : np.ndarray
        ``(d, d)`` bool; ``candidates[i, j]`` allows ``i -> j``.
    required : np.ndarray or None
        ``(d, d)`` bool edges present from the start.
    precedence : np.ndarray or None
        ``(d, d)`` bool; ``precedence[a, b]`` puts ``a`` before ``b`` in every order of the result.
    max_parents : int or None
        The most parents of any node; ``None`` sets no cap.
    k : int
        Basis rank of each smooth term before :func:`andrey.fcm.gam.effective_k` adjusts it.

    Returns
    -------
    np.ndarray
        ``(d, d)`` bool DAG; ``out[i, j]`` is the edge ``i -> j``.

    Raises
    ------
    ValueError
        If the required edges give a node more than ``max_parents`` parents.
    """
    X = bases.data
    n, d = X.shape
    adj = np.zeros((d, d), dtype=bool) if required is None else required.copy()
    if max_parents is not None and adj.sum(axis=0).max(initial=0) > max_parents:
        raise ValueError(
            f"the required edges give a node more than max_parents={max_parents} parents"
        )
    path = np.eye(d, dtype=bool) | adj
    if precedence is not None:
        path |= precedence
    path = _closure(path)
    open_ = candidates & ~adj & ~path.T
    np.fill_diagonal(open_, False)
    if max_parents is not None:
        open_[:, adj.sum(axis=0) >= max_parents] = False
    variance = np.empty(d)
    for j in range(d):
        parents = np.flatnonzero(adj[:, j]).tolist()
        y = X[:, j]
        variance[j] = (
            fit_gam(bases, j, parents, k=k).residual_variance if parents else np.var(y, ddof=1)
        )
    gain = np.full((d, d), -np.inf)
    refit = np.full((d, d), np.nan)

    def refresh(j: int) -> None:
        parents = np.flatnonzero(adj[:, j]).tolist()
        for i in np.flatnonzero(open_[:, j]).tolist():
            with_i = [*parents, i]
            if bases.n_coef(with_i, k) >= n:
                open_[i, j] = False
                gain[i, j] = -np.inf
                continue
            refit[i, j] = fit_gam(bases, j, with_i, k=k).residual_variance
            gain[i, j] = math.log(variance[j] / refit[i, j])

    for j in range(d):
        refresh(j)
    while open_.any():
        # Monotone rounding puts every tie of the largest rounded gain within one grid step of it.
        best = gain[open_].max()
        near = np.argwhere(open_ & (gain >= best - 2 * 10.0**-_KEY_DP)).tolist()
        i, j = min(near, key=lambda e: (-round(float(gain[e[0], e[1]]), _KEY_DP), e[0], e[1]))
        adj[i, j] = True
        variance[j] = refit[i, j]
        before, after = path[:, i], path[j, :]
        path[np.ix_(before, after)] = True
        open_ &= ~path.T
        open_[i, j] = False
        if max_parents is not None and adj[:, j].sum() >= max_parents:
            open_[:, j] = False
        gain[~open_] = -np.inf
        _log.debug("edge %d -> %d, gain %.6g", i, j, best)
        refresh(j)
    return adj


def _closure(reach: np.ndarray) -> np.ndarray:
    """The transitive closure of the reflexive relation ``reach`` (Warshall)."""
    out = reach.copy()
    for k in range(out.shape[0]):
        out |= out[:, k : k + 1] & out[k : k + 1, :]
    return out


def topological_order(adj: np.ndarray) -> np.ndarray:
    """An order of the DAG ``adj``, causes first: each step takes the first ready node."""
    indegree = adj.sum(axis=0)
    placed = np.zeros(adj.shape[0], dtype=bool)
    order = []
    while len(order) < adj.shape[0]:
        ready = np.flatnonzero(~placed & (indegree == 0))
        v = int(ready[0])
        order.append(v)
        placed[v] = True
        indegree = indegree - adj[v]
    return np.array(order, dtype=np.int64)


@dataclass(frozen=True)
class CAMResult:
    """A CAM fit: the pruned DAG and a causal order of the order search's DAG."""

    dag: np.ndarray
    order: np.ndarray


def cam(
    X: np.ndarray,
    *,
    alpha: float = 0.001,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
    max_parents: int | None = None,
) -> CAMResult:
    """Fit CAM to the ``(n, d)`` matrix ``X``.

    The possible parents come from :func:`boosting_counts` and :func:`neighbors`, less the
    ``forbidden`` edges; the ``required`` edges are in the order search from the start and in the
    result whatever their p-values; ``precedence`` and ``max_parents`` go to :func:`order_search`.
    A parent stays when its p-value is at most ``alpha``. The order is a topological order of the
    searched edges and ``precedence`` together. Ties go to the earlier column throughout, so a
    caller that orders the columns by rank breaks them by rank.
    """
    d = X.shape[1]
    required = np.zeros((d, d), dtype=bool) if required is None else required
    candidates = neighbors(boosting_counts(X))
    if forbidden is not None:
        candidates &= ~forbidden
    bases = BasisCache(X)
    searched = order_search(
        bases, candidates, required=required, precedence=precedence, max_parents=max_parents
    )
    dag = prune(X, searched, cutoff=alpha, k=K, bases=bases).astype(bool) | required
    ordered = searched if precedence is None else searched | precedence
    return CAMResult(dag=dag, order=topological_order(ordered))
