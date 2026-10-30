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
is the HSIC test with Gaussian kernels whose bandwidth is the median heuristic and whose null is
a gamma law with the moments of Gretton et al. (2007),
:func:`andrey.core.independence.hsic_moments_from_grams`. A side of several residuals takes one
kernel on their joint squared distances.

Where the steps here differ from the paper's:

- Every step reads the columns divided by their standard deviations, so the joint kernel of
  several residuals does not weigh each by its units. On standardized data the two are the same.
- The paper does not state the kernels' bandwidth; here it is the median of the positive squared
  distances over every pair of rows.
- The two screens of marginal dependence are the authors' code's; the paper's algorithms test
  every member of a set and every pair. Under the model they add no conclusion the lemmas do not
  support: two independent variables share no external effect (Assumption 1), so their residuals
  are independent too, a member with an independent other member fails line 13, and such a pair is
  never marked. With tests at ``alpha`` they skip most sink tests and the pairs whose residuals
  would look dependent by chance; a parent whose marginal dependence the test misses, such as a
  weak parent beside a strong one, is not found.
- A set that holds a pair already related, one in the other's candidates, is skipped: Lemma 4
  identifies a sink only with each candidate set outside ``K``. Without the skip, line 18 resets
  ``t`` whenever a known sink is found again, and the search need not end.
- p-values are compared as their logarithms rounded to ``_KEY_DP`` decimal places, as
  :func:`andrey.hc` compares its score changes, and a tie goes to the earlier column. The sets
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

from andrey.core.independence import hsic_moments_from_grams, median_gram
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
#: Memory for the kept kernel matrices; each holds ``n_samples**2`` float64 entries.
_GRAM_BYTES = 2**30

#: A residual: a variable regressed on the sorted variables of the tuple.
Term = tuple[int, tuple[int, ...]]
#: One side of a test: the joint residuals of its terms.
Side = tuple[Term, ...]


def _key(p: float) -> float:
    """``log(p)`` rounded to ``_KEY_DP`` decimal places, ``-inf`` for ``0``."""
    return round(math.log(p), _KEY_DP) if p > 0 else -math.inf


def _running(variables: tuple[int, ...], candidates: list[set[int]]) -> Side:
    """The side of each variable's residual on its current candidates."""
    return tuple((v, tuple(sorted(candidates[v]))) for v in variables)


class _Tests:
    """The regressions and HSIC tests of one fit; the only place a test runs.

    A residual depends only on its variable and regressors, and a test only on its two sides, so the
    search's repeated sweeps look up the p-values of the tests they repeat. Each variable's residual
    on its current candidates is kept, and the kernel matrices of the most recently used sides of
    such residuals, as many as ``_GRAM_BYTES`` holds.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.bases = BasisCache(data)
        n = data.shape[0]
        self._residuals: dict[Term, np.ndarray] = {}
        self._grams: OrderedDict[Side, tuple[np.ndarray, float, float]] = OrderedDict()
        self._capacity = max(4, _GRAM_BYTES // (8 * n * n))
        self._p_values: dict[tuple[Side, Side], float] = {}

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
    def _gram(residuals: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
        """The HSIC kernel of the residuals' joint squared distances."""
        total = np.zeros((residuals[0].size, residuals[0].size))
        for r in residuals:
            diff = r[:, None] - r[None, :]
            total += np.multiply(diff, diff, out=diff)
        return median_gram(total)

    def _side_gram(self, side: Side) -> tuple[np.ndarray, float, float]:
        found = self._grams.pop(side, None)
        if found is None:
            found = self._gram([self._residual(term) for term in side])
            if len(self._grams) >= self._capacity:
                self._grams.popitem(last=False)
        self._grams[side] = found
        return found

    def p_value(self, x: Side, y: Side) -> float:
        """The HSIC p-value between two sides of residuals on current candidates."""
        x, y = min(x, y), max(x, y)
        key = (x, y)
        found = self._p_values.get(key)
        if found is None:
            found = hsic_moments_from_grams(self._side_gram(x), self._side_gram(y))[1]
            self._p_values[key] = found
        return found

    def p_fresh(self, child: int, regressors: set[int], y: Side) -> float:
        """The HSIC p-value of ``child``'s residual on ``regressors`` against the side ``y``.

        The residual is fitted for this test alone, so neither it nor its kernel is kept.
        """
        term = (child, tuple(sorted(regressors)))
        key = ((term,), y)
        found = self._p_values.get(key)
        if found is None:
            residual = self._residuals.get(term)
            gram = self._gram([self._fit(term) if residual is None else residual])
            found = hsic_moments_from_grams(gram, self._side_gram(y))[1]
            self._p_values[key] = found
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
        if tests.p_value(((i, ()),), ((j, ()),)) < alpha:
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
    size = 2
    while size <= min(max_depth + 1, d):
        changed = False
        for members in combinations(range(d), size):
            if any(j in candidates[i] for i in members for j in members):
                continue
            best: tuple[tuple[float, int], int, float] | None = None
            for child in members:
                rest = tuple(j for j in members if j != child)
                if forbidden[list(rest), child].any() or not dependent[list(rest), child].all():
                    continue
                p = tests.p_fresh(child, candidates[child] | set(rest), _running(rest, candidates))
                if best is None or (-_key(p), child) < best[0]:
                    best = ((-_key(p), child), child, p)
            if best is None or not best[2] > alpha:
                continue
            sink = best[1]
            rest = tuple(j for j in members if j != sink)
            own = _running((sink,), candidates)
            if all(tests.p_value(own, _running((j,), candidates)) < alpha for j in rest):
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
            if tests.p_fresh(child, rest, _running((parent,), candidates)) > alpha:
                candidates[child] = rest
    return candidates


def drop_cycles(parents: list[set[int]], required: np.ndarray) -> list[set[int]]:
    """Drop every parent edge on a directed cycle but the required ones; ``parents`` changes."""
    d = len(parents)
    reach = np.zeros((d, d), dtype=bool)  # reach[i, j]: a directed path runs from i to j
    for child, found in enumerate(parents):
        reach[sorted(found), child] = True
    for k in range(d):
        reach |= np.outer(reach[:, k], reach[k, :])
    for child in range(d):
        parents[child] = {p for p in parents[child] if required[p, child] or not reach[child, p]}
    return parents


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
        if tests.p_value(_running((i,), parents), _running((j,), parents)) < alpha:
            pairs[i, j] = pairs[j, i] = True
    return pairs


@dataclass(frozen=True)
class CAMUVResult:
    """A CAM-UV fit: the parent edges and the pairs joined by an unobserved path.

    ``edges[i, j]`` is the edge ``i -> j``; ``pairs`` is symmetric and shares no pair with
    ``edges``.
    """

    edges: np.ndarray
    pairs: np.ndarray


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
    parents = drop_cycles(find_parents(tests, candidates, alpha=alpha, required=required), required)
    pairs = find_pairs(tests, parents, alpha=alpha, dependent=dependent, forbidden=forbidden)
    edges = np.zeros((d, d), dtype=bool)
    for child, found in enumerate(parents):
        edges[sorted(found), child] = True
    return CAMUVResult(edges=edges, pairs=pairs)
