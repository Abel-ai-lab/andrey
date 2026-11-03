"""CAM-UV: causal additive models with unobserved variables (Maeda and Shimizu 2021).

CAM-UV assumes each variable, observed or not, is a sum of nonlinear functions of its direct causes
plus independent noise, ``v_i = sum_j f_ij(x_j) + sum_k f_ik(y_k) + n_i``, on a DAG. Where an
unobserved variable is a direct cause of an observed one, the effect that passes through it cannot
be regressed away: the paper calls a directed path ``x_j -> ... -> y_m -> x_i`` into an unobserved
direct cause an unobserved causal path, and two paths from a common ancestor that both end in an
unobserved direct cause an unobserved backdoor path. The relation of a pair joined by either cannot
be identified (Lemma 1); every other pair's can (Lemmas 2 and 3). The method tests every pair of
variables for marginal dependence, then runs three steps:

1. **Candidate parents** (Algorithm 1, lines 2-23). Over the sets ``K`` of ``t`` variables, ``t``
   from 2 to ``max_depth + 1``, each member ``x_i`` dependent on every other member is regressed on
   the others together with its candidates ``M_i``, and its residual is tested against the joint
   residuals of the others, each on its own candidates. The member of largest p-value, ``x_b``, is
   a sink of ``K`` when that p-value is above ``alpha`` and its residual on ``M_b`` alone is
   dependent on each other member's residual, every p-value below ``alpha`` (Lemma 4); the other
   members then join ``M_b``. A sweep that changes some ``M_i`` starts over from ``t = 2``; one
   that changes none moves to ``t + 1``.
2. **Parents** (Algorithm 1, lines 24-30). Each variable's candidates are tried one at a time: a
   candidate ``x_j`` goes when the residual of ``x_i`` on its other candidates is independent of
   the residual of ``x_j`` on ``M_j``, at a p-value above ``alpha`` (Lemma 2).
3. **Pairs with an unobserved path** (Algorithm 2). A dependent pair with no parent edge whose
   residuals, each on its parents, are dependent at a p-value below ``alpha`` is joined by an
   unobserved causal or backdoor path (Lemma 1).

The settings are those of the paper's experiments: ``alpha = 0.01`` and sets of at most 3
variables, ``max_depth = 2``; each regression is an additive model of thin plate regression
splines with ten basis functions per term and GCV smoothing (Wood 2004),
:func:`andrey.fcm.gam.fit_gam`, which lowers the rank with fewer than 30 rows per term; each test
is the gamma HSIC test of :func:`andrey.core.independence.hsic_gamma_log_p`, with Gaussian kernels
whose bandwidth is the median heuristic and the null moments of Gretton et al. (2007) in their
product form, as RESIT runs it. A side of several residuals takes one kernel on their joint
squared distances.

Where the steps here differ from the paper's:

- Every step reads the columns divided by their standard deviations, so the joint kernel of
  several residuals does not weigh each by its units; on standardized data the two are the same.
  The kernels read the residuals centered, so an offset cannot cancel the distances' digits.
- The paper does not state the kernels' bandwidth; here it is the median of the positive squared
  distances over every pair of rows.
- The two screens of marginal dependence come from the authors' implementation, not from the paper's
  algorithm, which tests every member of a set and every pair. Under the model they add no
  conclusion the lemmas do not support: two independent variables share no external effect
  (Assumption 1), so their residuals are independent too, a member with an independent other member
  fails line 13, and such a pair is never marked. With tests at ``alpha`` they skip most sink tests
  and the pairs whose residuals would look dependent by chance; a parent whose marginal dependence
  the test misses, such as a weak parent beside a strong one, is not found.
- A set that holds a pair already related, one in the other's candidates, is skipped: Lemma 4
  identifies a sink only with each candidate set outside ``K``. Without the skip, line 18 resets
  ``t`` whenever a known sink is found again, and the search need not end.
- p-values are compared as their logarithms, computed directly so that a p-value below the
  smallest double keeps its place, rounded to ``_KEY_DP`` decimal places, as :func:`andrey.hc`
  compares its score changes; a tie goes to the earlier column. The sets
  run in lexicographic order and the candidates are tried in column order; :func:`andrey.camuv`
  passes the columns in their rank order, so each order follows the ranks.
- The paper does not say what to do when the parents form a directed cycle, which its model rules
  out and failed tests can leave. Every parent edge on a directed cycle is dropped, except a
  required one, and step 3 tests the pair.

The fit draws no random numbers.

References
----------
Maeda, Shimizu. "Causal additive models with unobserved variables." UAI, PMLR 161:97-106, 2021.

Gretton, Fukumizu, Teo, Song, Scholkopf, Smola. "A kernel statistical test of independence."
NIPS 20, 2007.

Wood. "Stable and efficient multiple smoothing parameter estimation for generalized additive
models." JASA 99:673-686, 2004.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from andrey.core.independence import gamma_gram, hsic_gamma_log_p
from andrey.core.units import unit_factors
from andrey.fcm.gam import BasisCache, fit_gam
from andrey.search._parallel import _KEY_DP

_log = logging.getLogger(__name__)

#: Significance level of every test.
ALPHA = 0.01
#: The most other variables a sink test reads at once: sets of up to ``MAX_DEPTH + 1`` variables.
MAX_DEPTH = 2
#: Basis functions per term of each regression.
K = 10
#: Memory for the kept kernel matrices; each holds ``n_samples**2`` float64 entries, so with more
#: than about 11,600 rows none is kept.
_GRAM_BYTES = 2**30

#: A residual: a variable regressed on the sorted variables of the tuple.
Term = tuple[int, tuple[int, ...]]
#: One side of a test: the joint residuals of its terms.
Side = tuple[Term, ...]
#: One side's kernel: its centered gram, its raw kernel's off-diagonal mean, and the centered gram's
#: sum of squares.
Gram = tuple[np.ndarray, float, float]


def _key(log_p: float) -> float:
    """``log_p`` rounded to ``_KEY_DP`` decimal places."""
    return round(log_p, _KEY_DP)


def _running(variables: tuple[int, ...], candidates: list[set[int]]) -> Side:
    """The side of each variable's residual on its current candidates."""
    return tuple((v, tuple(sorted(candidates[v]))) for v in variables)


class _Tests:
    """The regressions and HSIC tests of one fit; the only place a test runs.

    A residual depends only on its variable and regressors, and a test only on its two sides, so the
    search's repeated sweeps look up the log p-values of the tests they repeat; a test's value is
    computed with its two sides in one order, so it does not depend on which step asks first. Each
    variable's residual on its current candidates is kept, and the kernel matrices of the most
    recently used sides of such residuals, as many as ``_GRAM_BYTES`` holds.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.bases = BasisCache(data)
        n = data.shape[0]
        self._residuals: dict[Term, np.ndarray] = {}
        self._grams: OrderedDict[Side, Gram] = OrderedDict()
        self._capacity = _GRAM_BYTES // (8 * n * n)
        self._log_p: dict[tuple[Side, Side], float] = {}

    def _fit(self, term: Term) -> np.ndarray:
        child, regressors = term
        if not regressors:
            y = self.bases.data[:, child]
            return y - y.mean()
        return fit_gam(self.bases, child, list(regressors), k=K).residuals

    def _residual(self, term: Term) -> np.ndarray:
        found = self._residuals.get(term)
        if found is None:
            found = self._residuals[term] = self._fit(term)
        return found

    @staticmethod
    def _gram(residuals: list[np.ndarray]) -> Gram:
        """The HSIC kernel of the residuals, centered and read jointly, its bandwidth's median over
        every row, with the centered gram's sum of squares, which the product-form variance reads.
        """
        side = np.column_stack(residuals)
        centered, mu = gamma_gram(side - side.mean(axis=0), median_rows=None)
        return centered, mu, float(np.sum(centered * centered))

    @staticmethod
    def _hsic(x: Gram, y: Gram) -> float:
        """The log HSIC p-value of two sides' kernels, the null variance in its product form.

        Raises
        ------
        FloatingPointError
            If the test's value is not finite, so that no choice rests on it.
        """
        log_p = hsic_gamma_log_p(x, y, x[0].shape[0], variance="product")
        if not math.isfinite(log_p):
            raise FloatingPointError(
                "camuv: an HSIC test gave a non-finite p-value; the kernels could not be computed "
                "from these data"
            )
        return log_p

    def _side_gram(self, side: Side) -> Gram:
        found = self._grams.pop(side, None)
        if found is None:
            found = self._gram([self._residual(term) for term in side])
            if self._capacity and len(self._grams) >= self._capacity:
                self._grams.popitem(last=False)
        if self._capacity:
            self._grams[side] = found
        return found

    def log_p(self, x: Side, y: Side) -> float:
        """The log HSIC p-value between two sides of residuals on current candidates."""
        key = (min(x, y), max(x, y))
        found = self._log_p.get(key)
        if found is None:
            found = self._log_p[key] = self._hsic(*(self._side_gram(side) for side in key))
        return found

    def log_p_fresh(self, child: int, regressors: set[int], y: Side) -> float:
        """The log HSIC p-value of ``child``'s residual on ``regressors`` against the side ``y``.

        The residual is fitted for this test alone, so neither it nor its kernel is kept.
        """
        x: Side = ((child, tuple(sorted(regressors))),)
        key = (min(x, y), max(x, y))
        found = self._log_p.get(key)
        if found is None:
            residual = self._residuals.get(x[0])
            fresh = self._gram([self._fit(x[0]) if residual is None else residual])
            grams = {x: fresh, y: self._side_gram(y)}
            found = self._log_p[key] = self._hsic(*(grams[side] for side in key))
        return found


def marginal_dependence(tests: _Tests, d: int, *, alpha: float) -> np.ndarray:
    """Test every pair of variables for dependence, the screen of the first and third steps.

    Returns
    -------
    np.ndarray
        ``(d, d)`` symmetric bool; ``out[i, j]``: the HSIC p-value of ``i`` and ``j`` is below
        ``alpha``.
    """
    dependent = np.zeros((d, d), dtype=bool)
    for i, j in combinations(range(d), 2):
        if tests.log_p(((i, ()),), ((j, ()),)) < math.log(alpha):
            dependent[i, j] = dependent[j, i] = True
    return dependent


def find_candidates(
    tests: _Tests,
    d: int,
    *,
    alpha: float,
    max_depth: int,
    dependent: np.ndarray,
    forbidden: np.ndarray,
    required: np.ndarray,
) -> list[set[int]]:
    """Collect each variable's candidate parents, the first step.

    A member is tried as the sink of a set only when it is ``dependent`` on every other member and
    no other member is ``forbidden`` as its parent; each variable's ``required`` parents are its
    candidates from the start.

    Returns
    -------
    list of set of int
        ``out[i]``: the candidate parents of variable ``i``.
    """
    candidates = [{int(p) for p in np.flatnonzero(required[:, j])} for j in range(d)]
    log_alpha = math.log(alpha)
    size = 2
    while size <= min(max_depth + 1, d):
        changed = False
        for members in combinations(range(d), size):
            if any(j in candidates[i] for i in members for j in members):
                continue
            scored = []
            for child in members:
                rest = tuple(j for j in members if j != child)
                if forbidden[list(rest), child].any() or not dependent[list(rest), child].all():
                    continue
                regressors = candidates[child] | set(rest)
                log_p = tests.log_p_fresh(child, regressors, _running(rest, candidates))
                scored.append((-_key(log_p), child, log_p, rest))
            if not scored:
                continue
            _, sink, log_p, rest = min(scored)
            if not log_p > log_alpha:
                continue
            own = _running((sink,), candidates)
            if all(tests.log_p(own, _running((j,), candidates)) < log_alpha for j in rest):
                _log.debug("sink %d of %s", sink, members)
                candidates[sink].update(rest)
                changed = True
        size = 2 if changed else size + 1
    return candidates


def find_parents(
    tests: _Tests, candidates: list[set[int]], *, alpha: float, required: np.ndarray
) -> list[set[int]]:
    """Remove the candidates that are not parents, the second step.

    Variables run in column order and each one's candidates in column order; a candidate that is
    not ``required`` goes when the residual without it has a p-value above ``alpha`` against the
    candidate's residual on its current candidates. ``candidates`` changes in place.
    """
    for child in range(len(candidates)):
        for parent in sorted(candidates[child]):
            if required[parent, child]:
                continue
            rest = candidates[child] - {parent}
            if tests.log_p_fresh(child, rest, _running((parent,), candidates)) > math.log(alpha):
                candidates[child] = rest
    return candidates


def drop_cycles(parents: list[set[int]], required: np.ndarray) -> np.ndarray:
    """Drop every parent edge on a directed cycle but the required ones; ``parents`` changes.

    An edge lies on a directed cycle exactly when its two variables share a strongly connected
    component.

    Returns
    -------
    np.ndarray
        ``(d, d)`` bool; ``out[i, j]``: the dropped edge ``i -> j``.
    """
    d = len(parents)
    edges = np.zeros((d, d), dtype=bool)
    for child, found in enumerate(parents):
        edges[sorted(found), child] = True
    _, component = connected_components(csr_matrix(edges), directed=True, connection="strong")
    dropped = edges & (component[:, None] == component[None, :]) & ~required
    for parent, child in zip(*np.nonzero(dropped), strict=True):
        parents[child].discard(int(parent))
    return dropped


def find_pairs(
    tests: _Tests,
    parents: list[set[int]],
    *,
    alpha: float,
    dependent: np.ndarray,
    forbidden: np.ndarray,
) -> np.ndarray:
    """Find the pairs joined by an unobserved path, the third step.

    A ``dependent`` pair with no parent edge, and not ``forbidden`` both ways, is one when the
    residuals of its variables on their parents are dependent at a p-value below ``alpha``.

    Returns
    -------
    np.ndarray
        ``(d, d)`` symmetric bool; ``out[i, j]``: ``i`` and ``j`` are such a pair.
    """
    d = len(parents)
    pairs = np.zeros((d, d), dtype=bool)
    for i, j in combinations(range(d), 2):
        if not dependent[i, j] or i in parents[j] or j in parents[i]:
            continue
        if forbidden[i, j] and forbidden[j, i]:
            continue
        if tests.log_p(_running((i,), parents), _running((j,), parents)) < math.log(alpha):
            pairs[i, j] = pairs[j, i] = True
    return pairs


@dataclass(frozen=True)
class CAMUVResult:
    """A CAM-UV fit: the parent edges, the pairs joined by an unobserved path, and the edges dropped
    from directed cycles.

    ``edges[i, j]`` is the edge ``i -> j``; ``pairs`` is symmetric and shares no pair with
    ``edges``; ``cycle_edges[i, j]`` is a dropped edge ``i -> j``.
    """

    edges: np.ndarray
    pairs: np.ndarray
    cycle_edges: np.ndarray


def camuv(
    X: np.ndarray,
    *,
    alpha: float = ALPHA,
    max_depth: int = MAX_DEPTH,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
) -> CAMUVResult:
    """Fit CAM-UV to the ``(n, d)`` matrix ``X``.

    ``forbidden[i, j]`` keeps ``i`` out of ``j``'s candidates; ``required[i, j]`` makes ``i`` a
    parent of ``j`` that no step removes. Ties go to the earlier column throughout, so a caller that
    orders the columns by rank breaks them by rank.
    """
    d = X.shape[1]
    forbidden = np.zeros((d, d), dtype=bool) if forbidden is None else forbidden
    required = np.zeros((d, d), dtype=bool) if required is None else required
    tests = _Tests(X * unit_factors(X, standardize=True))
    dependent = marginal_dependence(tests, d, alpha=alpha)
    candidates = find_candidates(
        tests,
        d,
        alpha=alpha,
        max_depth=max_depth,
        dependent=dependent,
        forbidden=forbidden,
        required=required,
    )
    parents = find_parents(tests, candidates, alpha=alpha, required=required)
    cycle_edges = drop_cycles(parents, required)
    pairs = find_pairs(tests, parents, alpha=alpha, dependent=dependent, forbidden=forbidden)
    edges = np.zeros((d, d), dtype=bool)
    for child, found in enumerate(parents):
        edges[sorted(found), child] = True
    return CAMUVResult(edges=edges, pairs=pairs, cycle_edges=cycle_edges)
