"""Greedy noise-interventional equivalence search, the engine behind :func:`andrey.gnies`.

The data come from several environments that share one linear structural equation model and differ
only in the noise variances of an unknown set of variables, the targets ``I``.
:class:`NoiseInterventionScore` scores a variable outside ``I`` by the Gaussian likelihood of its
regression on its parents with one variance, over the rows of every environment centered by the
environment's own mean, and a variable in ``I`` with one coefficient vector and one variance per
environment. For a fixed ``I`` the GIES engine (:func:`~andrey.search.gies.search_class`) searches
the interventional equivalence classes of the family that intervenes on nothing and on each
target alone. The outer search adds the target that lowers the total score most, while one does,
then removes targets the same way.

Diagnostics go to the ``andrey.search.gnies`` logger at DEBUG level: a family a guard makes
unavailable, a target family with one residual degree of freedom in the smallest environment, an
alternating fit that reaches its iteration cap, the start of each inner search, and an outer step
whose two best candidates share a rounded total.

References
----------
Gamella, J. L., Taeb, A., Heinze-Deml, C., and Buhlmann, P. (2022). Characterization and greedy
learning of Gaussian structural causal models under unknown interventions. arXiv:2211.14897.

Hauser and Buhlmann (2012). Characterization and greedy learning of interventional Markov
equivalence classes of directed acyclic graphs. https://jmlr.org/papers/v13/hauser12a.html
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from andrey.core import GraphStructure, backend, score_delta
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.orient import to_structure
from andrey.core.score import ZERO_RESIDUAL, BICScore, _solve, check_columns, ebic_penalty
from andrey.core.ties import EPS_IMPROVE, stable_key, tie_ranks
from andrey.search.gies import search_class

logger = logging.getLogger(__name__)

#: The most iterations of a target's alternating fit.
MAX_ITER = 1000
#: The alternating fit stops once no coefficient moves by more than this share of ``1 + max |b|``.
STEP_TOL = 1e-15


def singleton_cut(targets: Iterable[int], d: int) -> np.ndarray:
    """``cut[a, b]`` for the family that intervenes on nothing and on each target alone.

    An edge with a target at one end has one orientation in every DAG of the class: ``cut[a, b]``
    is True for ``a != b`` when ``a`` or ``b`` is a target.
    """
    hit = np.zeros(d, dtype=bool)
    hit[sorted(targets)] = True
    cut = hit[:, None] | hit[None, :]
    np.fill_diagonal(cut, False)
    return cut


def alternating_fit(
    covs: np.ndarray, sizes: np.ndarray, cov: np.ndarray, node: int, parents: Sequence[int]
) -> tuple[float, np.ndarray] | None:
    """Fit ``node`` on ``parents`` with one coefficient vector and one variance per environment.

    ``covs`` holds each environment's covariance with divisor ``n_e``, ``sizes`` the ``n_e``, and
    ``cov`` the within-environment covariance, which gives the start, the regression over every
    environment. Each step sets ``w_e``, environment ``e``'s residual variance, at the current
    coefficients ``b``, then ``b`` to the least-squares fit weighted by ``n_e / w_e``; each half
    of a step minimizes ``L = sum_e n_e log w_e`` exactly. Returns ``(L, b)`` at the step with the
    lowest ``L``, the fit stopping at the first step that does not lower it, when no coefficient
    moves by more than :data:`STEP_TOL` times ``1 + max |b|``, or after :data:`MAX_ITER` steps.
    Returns None when a step leaves some environment a residual variance of at most
    ``ZERO_RESIDUAL`` times the node's variance there.
    """
    P = list(parents)
    n = sizes.astype(np.float64)
    Sjj = covs[:, node, node]
    floor = ZERO_RESIDUAL * Sjj
    Spp = covs[:, P][:, :, P]
    Spj = covs[:, P, node]
    b = _solve(cov[np.ix_(P, P)], cov[P, node]) if P else np.zeros(0)
    best: tuple[float, np.ndarray] | None = None
    last = not P
    for _ in range(MAX_ITER):
        w = Sjj - 2.0 * (Spj @ b) + np.einsum("kab,a,b->k", Spp, b, b)
        if not (w > floor).all():
            return None
        L = math.fsum((n * np.log(w)).tolist())
        if best is not None and not L < best[0]:
            break
        best = (L, b)
        if last:
            break
        weights = n / w
        b_new = _solve(
            (weights[:, None, None] * Spp).sum(axis=0), (weights[:, None] * Spj).sum(axis=0)
        )
        last = bool(np.max(np.abs(b_new - b)) <= STEP_TOL * (1.0 + np.max(np.abs(b))))
        b = b_new
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("target %d on %s: the alternating fit reached %d steps", node, P, MAX_ITER)
    return best


class NoiseInterventionScore:
    """Local scores under noise interventions, in deviance units (lower is better).

    Each environment's columns are centered by their own mean, and ``S_e`` is the environment's
    covariance with divisor ``n_e``; ``S``, the within-environment covariance, is their sum weighted
    by ``n_e / N``, with ``N`` every row. For node ``j`` with parents ``P``, ``c =
    penalty_discount``, and ``K`` environments:

    - outside the targets, ``N log r + c log(N) |P|`` plus the extended-BIC term
      :func:`~andrey.core.score.ebic_penalty` of ``|P|``, with ``r`` the residual variance of the
      regression of ``j`` on ``P`` under ``S`` (a :class:`~andrey.core.score.BICScore` on ``S``);
    - at a target, ``L + c log(N) (|P| + K - 1)`` plus the same extended-BIC term, with ``L = sum_e
      n_e log w_e(b)`` and ``w_e(b)`` the residual variance of environment ``e`` under one
      coefficient vector ``b``. ``b`` is the fixed point of the alternating fit started from the
      regression under ``S``: each step sets every ``w_e`` at the current ``b``, then ``b`` to the
      least-squares fit weighted by ``n_e / w_e``. The fit stops at the first step that does not
      lower ``L``, when no coefficient moves by more than :data:`STEP_TOL` times ``1 + max |b|``,
      or after :data:`MAX_ITER` steps, and keeps the lowest ``L``.

    The ``K - 1`` extra variances of a target are charged ``c log(N)`` each. The environments are
    held in an order their own rows set (row count, then the bytes of ``S_e``), so every sum over
    them is the same bits whichever environment got which code.

    A family is unavailable, its score ``inf``: at a target, when ``|P| > min_e n_e - 2``, where
    some coefficient vector fits an environment exactly and the likelihood has no maximum, or when
    the fit leaves some environment a residual variance of at most ``ZERO_RESIDUAL`` times the
    node's variance there; outside the targets, when ``|P| > N - K - 1``, the degrees of freedom
    of ``S``, or as :class:`~andrey.core.score.BICScore` rules.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        The rows of every environment.
    env_codes : ndarray of int, shape (n,)
        Each row's environment, ``0..K-1``.
    penalty_discount : float, default=1.0
        Weight on the ``log(N)`` charged per parameter.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of each node's parent count.
    column_name : callable or None
        Names a column position in the message of linearly dependent columns.

    Raises
    ------
    ValueError
        If there are fewer than two environments, an environment has fewer than two rows, or
        :func:`~andrey.core.score.check_columns` rejects a column of ``S``. Centering each
        environment leaves ``S`` a rank of at most ``N - K``, so the check counts ``N - K + 1``
        rows, as for a covariance with divisor ``n - 1``.
    """

    def __init__(
        self,
        data: np.ndarray,
        env_codes: np.ndarray,
        *,
        penalty_discount: float = 1.0,
        ebic_gamma: float = 1.0,
        column_name: Callable[[int], str] | None = None,
    ) -> None:
        X = np.asarray(data, dtype=np.float64)
        codes = np.asarray(env_codes, dtype=np.intp)
        self.d = X.shape[1]
        self.sizes = np.bincount(codes)
        if self.sizes.size < 2 or self.sizes.min() < 2:
            raise ValueError("noise-intervention scores need two or more environments of 2+ rows")
        self.K = int(self.sizes.size)
        self.N = int(self.sizes.sum())
        self.min_rows = int(self.sizes.min())
        means = np.stack([X[codes == k].mean(axis=0) for k in range(self.K)])
        self.centered = X - means[codes]
        scatter = np.stack(
            [self.centered[codes == k].T @ self.centered[codes == k] for k in range(self.K)]
        )
        covs = scatter / self.sizes[:, None, None]
        order = sorted(range(self.K), key=lambda k: (int(self.sizes[k]), covs[k].tobytes()))
        self.sizes = self.sizes[order]
        self.covs = covs[order]
        self.cov = scatter[order].sum(axis=0) / self.N
        check_columns(self.cov, self.N - self.K + 1, column_name)
        self.penalty_discount = float(penalty_discount)
        self.ebic_gamma = float(ebic_gamma)
        self.log_n = math.log(self.N)
        self.bic = BICScore.from_cov(
            self.cov,
            self.N,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            cache_size=None,
        )
        self._targets: dict[tuple[int, tuple[int, ...]], float] = {}

    def view(self, targets: Iterable[int]) -> TargetView:
        """The score with ``targets`` as the intervened variables, for the GIES engine."""
        return TargetView(self, frozenset(int(j) for j in targets))

    def other(self, node: int, parents: tuple[int, ...]) -> float:
        """The local score of ``node`` outside the targets; ``parents`` sorted."""
        if len(parents) > self.N - self.K - 1:
            _rejected(node, parents, "the within-environment covariance's degrees of freedom")
            return math.inf
        return self.bic.score(node, parents)

    def target(self, node: int, parents: tuple[int, ...]) -> float:
        """The memoized local score of ``node`` as a target; ``parents`` sorted."""
        key = (node, parents)
        if key not in self._targets:
            self._targets[key] = self._target(node, parents)
        return self._targets[key]

    def _target(self, j: int, P: tuple[int, ...]) -> float:
        m = len(P)
        if m > self.min_rows - 2:
            _rejected(j, P, "the rows of the smallest environment")
            return math.inf
        if m == self.min_rows - 2 and logger.isEnabledFor(logging.DEBUG):
            logger.debug("target %d with %d parents: one residual degree of freedom left", j, m)
        fit = alternating_fit(self.covs, self.sizes, self.cov, j, P)
        if fit is None:
            _rejected(j, P, "a residual variance of zero")
            return math.inf
        L = fit[0]
        penalty = self.log_n * (m + self.K - 1) * self.penalty_discount
        if self.ebic_gamma:
            penalty += ebic_penalty(m, self.d, self.ebic_gamma)
        return L + penalty


class TargetView:
    """:class:`NoiseInterventionScore` with one set of targets, as the GIES engine reads it.

    A local score depends only on whether its own node is a target, so every view of one
    :class:`NoiseInterventionScore` shares its memoized scores.
    """

    def __init__(self, score: NoiseInterventionScore, targets: frozenset[int]) -> None:
        self.base = score
        self.targets = targets

    def score(self, node: int, parents: Iterable[int]) -> float:
        """The local score of ``node`` given ``parents``; lower is better."""
        key = tuple(sorted(int(p) for p in parents))
        node = int(node)
        if node in self.targets:
            return self.base.target(node, key)
        return self.base.other(node, key)

    def add_deltas(self, node: int, parents: Iterable[int]) -> np.ndarray:
        """``score(node, parents + u) - score(node, parents)`` for every variable ``u``.

        Outside the targets, one rank-one update per ``u`` (:func:`~andrey.core.score_delta.
        add_deltas`); at a target, two full scores. ``inf`` for ``node`` and ``parents``.
        """
        base = sorted(int(p) for p in parents)
        node = int(node)
        base_score = self.base
        d = base_score.d
        if node not in self.targets:
            if len(base) + 1 > base_score.N - base_score.K - 1:
                return np.full(d, np.inf)
            return score_delta.add_deltas(base_score.bic, node, base)
        out = np.full(d, np.inf)
        old = base_score.target(node, tuple(base))
        taken = {node, *base}
        for u in range(d):
            if u not in taken:
                out[u] = base_score.target(node, tuple(sorted([*base, u]))) - old
        return out


def _rejected(node: int, parents: tuple[int, ...], guard: str) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("family of %d with %d parents unavailable: %s", node, len(parents), guard)


def gnies(
    data: np.ndarray,
    env_codes: np.ndarray,
    *,
    known: Iterable[int] = (),
    fixed: bool = False,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: int | None = None,
    constraints: Constraints | None = None,
    column_name: Callable[[int], str] | None = None,
) -> tuple[GraphStructure, float, list[int]]:
    """Learn the targets and the interventional essential graph by greedy search.

    Each set of targets ``I`` the search visits gets one inner search from the empty graph
    (:func:`~andrey.search.gies.search_class`) over the classes of the family that intervenes on
    nothing and on each member of ``I`` alone, on :meth:`NoiseInterventionScore.view` of ``I``;
    its total is that set's score. Starting from ``known``, the forward phase adds the variable
    whose set scores lowest while that lowers the total by more than ``1e-9``; the backward phase
    then removes a variable not in ``known`` the same way. Candidates are compared by their total
    divided by the variable count and rounded to nine decimal places, then by
    :func:`~andrey.core.ties.tie_ranks` of the per-environment centered data. A set whose required
    edges give a family the score makes unavailable totals ``inf`` and is never chosen. The fit
    runs on one BLAS thread, so the host's thread count does not change its bits.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        The rows of every environment.
    env_codes : ndarray of int, shape (n_samples,)
        Each row's environment, ``0..K-1`` with ``K >= 2``.
    known : iterable of int
        Variables that are targets in every set the search visits.
    fixed : bool
        Score only ``known``: one inner search, no outer search.
    penalty_discount, ebic_gamma : float
        The penalties of :class:`NoiseInterventionScore`.
    max_parents : int or None
        The cap on any DAG's parents in every inner search, as for :func:`~andrey.search.gies.gies`.
    constraints : Constraints or None
        Background knowledge in every inner search; the result then directs the edges it decides,
        and Meek's rules with R4 (:func:`~andrey.core.knowledge.mpdag`).
    column_name : callable or None
        Names a column position in the message of linearly dependent columns.

    Returns
    -------
    tuple[GraphStructure, float, list[int]]
        The interventional essential graph (``kind="cpdag"``), its total score (lower is better),
        and the targets, sorted.

    Raises
    ------
    ValueError
        If :class:`NoiseInterventionScore` rejects the data, or if no set of targets the search
        visits has a finite total: some family of the required edges is unavailable.
    """
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    known = frozenset(int(j) for j in known)
    with backend.one_blas_thread():
        score = NoiseInterventionScore(
            X,
            env_codes,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            column_name=column_name,
        )
        ranks = tie_ranks(score.centered, np.zeros(d, dtype=bool)).tolist()
    runs: dict[frozenset[int], tuple[np.ndarray, float]] = {}

    def inner(targets: frozenset[int]) -> tuple[np.ndarray, float]:
        if targets not in runs:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("inner search with targets %s", sorted(targets))
            cut = singleton_cut(targets, d)
            runs[targets] = search_class(score.view(targets), cut, ranks, constraints, max_parents)
        return runs[targets]

    current = known
    adj, total = inner(current)
    if not fixed:
        for adding in (True, False):
            while True:
                pool = sorted(set(range(d)) - current) if adding else sorted(current - known)
                if not pool:
                    break
                ranked = sorted((stable_key(inner(current ^ {j})[1], d), ranks[j], j) for j in pool)
                if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("outer step tie at key %r", ranked[0][0])
                step = current ^ {ranked[0][2]}
                if not runs[step][1] < total - EPS_IMPROVE:
                    break
                current = step
                adj, total = runs[current]
    if math.isinf(total):
        raise ValueError(
            "gnies: no graph with the required edges has a finite score under the targets "
            f"{sorted(current)}: the required edges give a variable more parents than its rows "
            "allow, or fit it exactly"
        )
    adj = adj.copy() if constraints is None else mpdag(adj, constraints)
    return to_structure(adj, kind="cpdag"), total, sorted(current)
