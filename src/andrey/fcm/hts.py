"""Hierarchical topological sorts (LHTS, NHTS) and edge discovery (ED) of Hiremath et al. (2024).

Both sorts return *layers*: layer 0 holds the roots, and a variable's layer is the length of the
longest directed path from a root to it (the paper's hierarchical sort). ED then finds each
variable's parents among the variables of earlier layers.

Every independence answer comes from a small *tests* object, so the same search runs on data
(:class:`LinearData`, :class:`NonlinearData`, a registered conditional-independence test) and on
exact answers from a known model, which the unit tests use. A test's answer is the logarithm of a
p-value; a pair is independent when it is above ``log(alpha)``.

References
----------
Hiremath, S., Maasch, J. R. M. A., Gao, M., Ghosal, P., and Gan, K. (2024). Hybrid top-down global
causal discovery with local search for linear and nonlinear additive noise models. NeurIPS 37.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from scipy.linalg import solve

from andrey.core.ci import count_tests
from andrey.core.independence import hsic_gram, hsic_log_p

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

V = TypeVar("V")

#: Kernel ridge settings of the paper's experiments (Appendix E.2): the polynomial kernel
#: ``(<x, y> / q + 1)^3`` over ``q`` regressors with ridge term 1 for the parent tests, and the
#: Gaussian kernel ``exp(-0.01 ||x - y||^2)`` with ridge term 0.1 for the layer tests.
POLY_DEGREE, POLY_RIDGE = 3, 1.0
RBF_GAMMA, RBF_RIDGE = 0.01, 0.1
#: Significant digits at which two p-values compare equal when a rule picks the largest.
DIGITS = 9


class LinearTests(Protocol[V]):
    """The operations LHTS needs: variables, linear residuals, and an independence test."""

    def column(self, i: int) -> V: ...
    def residual(self, y: V, regressors: Sequence[V]) -> V: ...
    def log_p(self, a: V, b: V) -> float: ...


class NonlinearTests(Protocol):
    """The questions NHTS asks, each answered by the logarithm of a p-value."""

    def marginal(self, i: int, j: int) -> float: ...
    def parent(self, i: int, j: int, regressors: Sequence[int]) -> float: ...
    def layer(self, i: int, done: Sequence[int]) -> list[float]: ...
    def conditional(self, x: int, y: int, given: Sequence[int]) -> float: ...


def _standardized(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0)
    scale = centered.std(axis=0)
    return centered / np.where(scale > 0, scale, 1.0)


def _hsic(a: np.ndarray, b: np.ndarray, method: str) -> float:
    count_tests()
    return hsic_log_p(hsic_gram(a - a.mean()), hsic_gram(b - b.mean()), method)


class LinearData:
    """LHTS's operations on data: standardized columns, least squares, and the gamma HSIC test."""

    def __init__(self, X: np.ndarray, method: str = "lhts") -> None:
        self.X = _standardized(np.asarray(X, dtype=np.float64))
        self.method = method

    def column(self, i: int) -> np.ndarray:
        return self.X[:, i]

    def residual(self, y: np.ndarray, regressors: Sequence[np.ndarray]) -> np.ndarray:
        if not regressors:
            return y - y.mean()
        Z = np.column_stack(regressors)
        Z = Z - Z.mean(axis=0)
        yc = y - y.mean()
        beta = np.linalg.lstsq(Z, yc, rcond=None)[0]
        return yc - Z @ beta

    def log_p(self, a: np.ndarray, b: np.ndarray) -> float:
        return _hsic(a, b, self.method)


def kernel_ridge_residual(y: np.ndarray, Z: np.ndarray, kernel: str) -> np.ndarray:
    """The in-sample residual of a kernel ridge regression of ``y`` on the columns of ``Z``.

    ``kernel="poly"``: ``(<z, z'> / q + 1)^3`` over ``q`` columns, ridge term 1; ``"rbf"``:
    ``exp(-0.01 ||z - z'||^2)``, ridge term 0.1 (the paper's Appendix E.2). No intercept.
    """
    if kernel == "poly":
        K = (Z @ Z.T / Z.shape[1] + 1.0) ** POLY_DEGREE
        ridge = POLY_RIDGE
    else:
        sq = np.sum(Z * Z, axis=1)
        K = np.exp(-RBF_GAMMA * np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0))
        ridge = RBF_RIDGE
    K[np.diag_indices_from(K)] += ridge
    coef = solve(K, y, assume_a="pos")
    K[np.diag_indices_from(K)] -= ridge
    return y - K @ coef


@dataclass
class NonlinearData:
    """NHTS's questions on data: the gamma HSIC test on standardized columns and kernel ridge
    residuals, and ``ci`` (a registered conditional-independence test) for the root step."""

    X: np.ndarray
    ci: Callable[[int, int, Sequence[int]], float]
    method: str = "nhts"
    _grams: dict[int, tuple[np.ndarray, float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = _standardized(np.asarray(self.X, dtype=np.float64))

    def _gram(self, i: int) -> tuple[np.ndarray, float, float]:
        if i not in self._grams:
            self._grams[i] = hsic_gram(self.X[:, i])
        return self._grams[i]

    def marginal(self, i: int, j: int) -> float:
        count_tests()
        return hsic_log_p(self._gram(i), self._gram(j), self.method)

    def parent(self, i: int, j: int, regressors: Sequence[int]) -> float:
        r = kernel_ridge_residual(self.X[:, j], self.X[:, list(regressors)], "poly")
        count_tests()
        return hsic_log_p(self._gram(i), hsic_gram(r - r.mean()), self.method)

    def layer(self, i: int, done: Sequence[int]) -> list[float]:
        r = kernel_ridge_residual(self.X[:, i], self.X[:, list(done)], "rbf")
        side = hsic_gram(r - r.mean())
        count_tests(len(done))
        return [hsic_log_p(self._gram(s), side, self.method) for s in done]

    def conditional(self, x: int, y: int, given: Sequence[int]) -> float:
        p = self.ci(x, y, list(given))
        return math.log(p) if p > 0 else -math.inf


def _key(log_p: float) -> float:
    """``log_p`` to :data:`DIGITS` significant digits, so near-equal p-values tie."""
    return float(f"{log_p:.{DIGITS}g}")


class _Order:
    """A strict partial order kept transitively closed: ``before[a, b]``, ``a`` before ``b``."""

    def __init__(self, d: int, precedence: np.ndarray | None) -> None:
        self.before = np.zeros((d, d), dtype=bool)
        if precedence is not None:
            for a, b in zip(*np.nonzero(precedence), strict=True):
                self.add(int(a), int(b))

    def allows(self, a: int, b: int) -> bool:
        return a != b and not self.before[b, a]

    def add(self, a: int, b: int) -> None:
        up = self.before[:, a].copy()
        up[a] = True
        down = self.before[b].copy()
        down[b] = True
        self.before |= np.outer(up, down)


@dataclass
class SortResult:
    """Layers of a hierarchical sort and the number of placements no test decided."""

    layers: list[list[int]]
    forced: int


def _peel(d: int, before: np.ndarray) -> list[list[int]]:
    """Algorithm 4 (AS): repeatedly take the variables with nothing before them among the rest."""
    left = list(range(d))
    layers = []
    while left:
        layer = [v for v in left if not before[left, v].any()]
        layers.append(layer)
        left = [v for v in left if v not in layer]
    return layers


def lhts_sort(
    tests: LinearTests[V],
    d: int,
    alpha: float,
    *,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> SortResult:
    """LHTS (Algorithm 1): the ancestor table from residual tests, then its layers (Algorithm 4).

    Stage 1 marks independent pairs unrelated. Each later round tests every pair still unknown
    after regressing both variables on their known mutual ancestors jointly: ``x_i`` is an ancestor
    of ``x_j`` when ``x_i``'s residual is independent of the residual of ``x_j`` regressed on it and
    the reverse is dependent; both independent marks the pair unrelated (not in the first round,
    Stage 2). A round's decisions are applied together after it, the table kept transitively closed;
    one that would close a cycle with the table or ``precedence`` is skipped. A round that decides
    nothing makes the ancestor decision with the largest p-value that closes no cycle (ties by the
    lower positions), and counts it in ``forced``. ``required[i, j]`` puts ``x_i`` before ``x_j``
    from the start; ``precedence[i, j]`` forbids ``x_j`` before ``x_i``.
    """
    log_alpha = math.log(alpha)
    order = _Order(d, precedence)
    ancestor = np.zeros((d, d), dtype=bool)
    if required is not None:
        for a, b in zip(*np.nonzero(required), strict=True):
            order.add(int(a), int(b))
            ancestor |= _closure_add(ancestor, int(a), int(b))
    unrelated = np.zeros((d, d), dtype=bool)
    columns = [tests.column(i) for i in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            if not ancestor[i, j] and not ancestor[j, i]:
                unrelated[i, j] = unrelated[j, i] = tests.log_p(columns[i], columns[j]) > log_alpha
    residuals: dict[Hashable, V] = {}
    answers: dict[Hashable, tuple[float, float]] = {}

    def projected(i: int, mutual: tuple[int, ...]) -> V:
        key = (i, mutual)
        if key not in residuals:
            residuals[key] = tests.residual(columns[i], [columns[k] for k in mutual])
        return residuals[key]

    forced = 0
    first = True
    while True:
        pairs = [
            (i, j)
            for i in range(d)
            for j in range(i + 1, d)
            if not (ancestor[i, j] or ancestor[j, i] or unrelated[i, j])
        ]
        if not pairs:
            break
        decisions: list[tuple[str, int, int]] = []
        scores: dict[tuple[int, int], float] = {}
        for i, j in pairs:
            mutual = tuple(int(k) for k in np.flatnonzero(ancestor[:, i] & ancestor[:, j]))
            key = (i, j, mutual)
            if key not in answers:
                xi, xj = projected(i, mutual), projected(j, mutual)
                answers[key] = (
                    tests.log_p(xi, tests.residual(xj, [xi])),
                    tests.log_p(xj, tests.residual(xi, [xj])),
                )
            p_ij, p_ji = answers[key]
            scores[i, j], scores[j, i] = p_ij, p_ji
            free_ij, free_ji = p_ij > log_alpha, p_ji > log_alpha
            if free_ij and free_ji:
                if not first:
                    decisions.append(("unrelated", i, j))
            elif free_ij:
                decisions.append(("ancestor", i, j))
            elif free_ji:
                decisions.append(("ancestor", j, i))
        progress = False
        for kind, a, b in decisions:
            if kind == "unrelated":
                if not (ancestor[a, b] or ancestor[b, a]):
                    unrelated[a, b] = unrelated[b, a] = True
                    progress = True
            elif order.allows(a, b) and not ancestor[a, b]:
                order.add(a, b)
                ancestor |= _closure_add(ancestor, a, b)
                progress = True
        if not progress:
            options = [(a, b) for (a, b) in scores if order.allows(a, b)]
            if not options:
                break
            a, b = min(options, key=lambda ab: (-_key(scores[ab]), ab))
            order.add(a, b)
            ancestor |= _closure_add(ancestor, a, b)
            forced += 1
        first = False
    return SortResult(_peel(d, order.before | ancestor), forced)


def _closure_add(ancestor: np.ndarray, a: int, b: int) -> np.ndarray:
    """The pairs ``a -> b`` adds to a transitively closed ancestor table."""
    up = ancestor[:, a].copy()
    up[a] = True
    down = ancestor[b].copy()
    down[b] = True
    return np.outer(up, down)


def nhts_sort(
    tests: NonlinearTests,
    d: int,
    alpha: float,
    *,
    precedence: np.ndarray | None = None,
) -> SortResult:
    """NHTS (Algorithm 2): roots from parent tests, then one layer per round.

    Stage 1 tests every pair for marginal dependence; a variable independent of all is isolated
    (layer 0). Stage 2 (Lemma 4.3): ``x_i`` is a parent of a dependent ``x_j`` when ``x_i`` is
    independent of the residual of ``x_j`` on ``x_i`` alone, or on ``x_i`` and ``P_ij``, the
    variables independent of ``x_i`` and dependent on ``x_j``. Stage 3 (Lemma 4.4): a variable
    with a found child is a root when it has no found parent and, for every other variable ``x_j``
    with a found child, ``x_i`` and ``x_j`` are independent, ``x_j`` is a found child of ``x_i``,
    or ``x_j`` and some found child of ``x_i`` are dependent given ``x_i``. Stage 4: each round, a
    variable joins the next layer when it depends on some placed variable and the residual of its
    regression on all placed variables is independent of each of them. A round that places none
    places the variable whose smallest p-value is largest (ties by the lower position), counted in
    ``forced``; so does a search whose Stage 3 found no root among dependent variables. A variable
    waits while one ``precedence`` puts before it is unplaced, and is never a root then.
    """
    log_alpha = math.log(alpha)
    before = np.zeros((d, d), dtype=bool) if precedence is None else precedence
    dep = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            dep[i, j] = dep[j, i] = tests.marginal(i, j) <= log_alpha
    isolated = [i for i in range(d) if not dep[i].any() and not before[:, i].any()]
    parent = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(d):
            if i == j or not dep[i, j] or before[j, i]:
                continue
            if tests.parent(i, j, [i]) > log_alpha:
                parent[i, j] = True
                continue
            shared = [k for k in range(d) if k not in (i, j) and not dep[k, i] and dep[k, j]]
            if shared and tests.parent(i, j, [i, *shared]) > log_alpha:
                parent[i, j] = True
    found = [i for i in range(d) if parent[i].any()]
    roots = []
    for i in found:
        if parent[:, i].any() or before[:, i].any():
            continue
        children = np.flatnonzero(parent[i])
        if all(
            not dep[i, j]
            or parent[i, j]
            or any(tests.conditional(j, int(k), [i]) <= log_alpha for k in children if k != j)
            for j in found
            if j != i
        ):
            roots.append(i)
    forced = 0
    layers = [sorted(isolated + roots)]
    done = list(layers[0])
    left = [v for v in range(d) if v not in done]
    while left:
        ready = [v for v in left if not before[left, v].any()]
        scores: dict[int, float] = {}
        layer = []
        for v in ready:
            if not dep[v, done].any():
                continue
            answers = tests.layer(v, done)
            scores[v] = min(answers)
            if scores[v] > log_alpha:
                layer.append(v)
        if not layer:
            pick = min(scores, key=lambda v: (-_key(scores[v]), v)) if scores else min(ready)
            layer = [pick]
            forced += 1
        layers.append(layer)
        done = sorted(done + layer)
        left = [v for v in left if v not in layer]
    return SortResult([sorted(layer) for layer in layers if layer], forced)


def edge_discovery(
    layers: Sequence[Sequence[int]],
    ci: Callable[[int, int, Sequence[int]], float],
    d: int,
    alpha: float,
    *,
    required: np.ndarray | None = None,
    forbidden: np.ndarray | None = None,
) -> np.ndarray:
    """ED over a hierarchical sort (Algorithm 3 in the layered form of Appendix C.2).

    For each variable ``x_j``, layer by layer from the first, the candidates are the variables of
    earlier layers, taken one layer at a time from the nearest. ``x_i`` is a candidate parent when
    ``ci(x_i, x_j, Z)`` is at most ``alpha`` for ``Z`` = the parents of ``x_i`` and the candidates
    already found in layers between them. Every parent passes that step, so the candidates hold
    the parents; each candidate that shares or follows a layer with another is then tested again
    given all other candidates, and stays when the p-value is at most ``alpha``.
    ``required[i, j]`` keeps ``x_i -> x_j`` untested; ``forbidden[i, j]`` drops it. Returns the
    boolean adjacency, ``[i, j]`` for ``x_i -> x_j``.
    """
    layer_of = np.empty(d, dtype=int)
    for k, layer in enumerate(layers):
        layer_of[list(layer)] = k
    parents = np.zeros((d, d), dtype=bool)
    for depth in range(1, len(layers)):
        for j in layers[depth]:
            found: list[int] = []
            for lower in range(depth - 1, -1, -1):
                new = []
                for i in layers[lower]:
                    if forbidden is not None and forbidden[i, j]:
                        continue
                    if required is not None and required[i, j]:
                        new.append(i)
                        continue
                    given = sorted({*np.flatnonzero(parents[:, i]).tolist(), *found})
                    if ci(i, j, given) <= alpha:
                        new.append(i)
                found.extend(new)
            keep = []
            for i in found:
                others = [k for k in found if k != i]
                if (
                    (required is not None and required[i, j])
                    or not any(layer_of[k] <= layer_of[i] for k in others)
                    or ci(i, j, sorted(others)) <= alpha
                ):
                    keep.append(i)
            parents[keep, j] = True
    return parents


def total_order(layers: Sequence[Sequence[int]]) -> list[int]:
    """The layers in turn, each in ascending position."""
    return [v for layer in layers for v in sorted(layer)]


@dataclass
class HTSFit:
    """A fitted hierarchical sort and its DAG, over the columns as given."""

    dag: np.ndarray
    layers: list[list[int]]
    forced: int


def fit_hts(
    X: np.ndarray,
    sort: str,
    *,
    alpha: float,
    ci: Callable[[int, int, Sequence[int]], float],
    required: np.ndarray | None = None,
    forbidden: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> HTSFit:
    """Run ``sort`` (``"lhts"`` or ``"nhts"``) on the columns of ``X``, then ED with ``ci``."""
    d = X.shape[1]
    if sort == "lhts":
        result = lhts_sort(LinearData(X), d, alpha, required=required, precedence=precedence)
    else:
        tests = NonlinearData(X, ci)
        result = nhts_sort(tests, d, alpha, precedence=precedence)
    dag = edge_discovery(result.layers, ci, d, alpha, required=required, forbidden=forbidden)
    return HTSFit(dag, result.layers, result.forced)
