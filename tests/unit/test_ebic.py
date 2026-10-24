"""Test the extended BIC (``ebic_gamma``) against its formula on every engine path.

The reference family score is ``n log R + lambda_value k log n + 2 gamma log C(d - 1, k)``, with
``R`` the residual variance of the node on its parents from the sample covariance and the count
from :func:`math.comb`. Each fast path (full scores, the scalar and batched Schur deltas, their
full-score fallbacks, the Gram score of exact search) is compared with it directly, and each
search is run twice, once on its fast engine and once on the reference score, which takes the
two-full-score path, and must return the same graph.
"""

from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pytest

import andrey
from andrey.core import TAIL
from andrey.core.score import BICScore, ebic_penalty, ebic_step, make_score
from andrey.core.score_delta import (
    DeltaBICScore,
    add_deltas,
    delta_many_with_base,
    remove_deltas,
    score_many_with_base,
)
from andrey.search.exact import _GramBICScore
from andrey.search.gies import InterventionalBICScore

GAMMAS = (0.5, 1.0)


def _sem(seed: int, n: int, d: int, p: float) -> np.ndarray:
    """A linear-Gaussian SEM over a random DAG of edge density ``p``, with weak and strong edges."""
    rng = np.random.default_rng(seed)
    b = np.tril(rng.random((d, d)) < p, -1) * rng.uniform(0.15, 1.0, (d, d))
    b *= rng.choice([-1.0, 1.0], (d, d))
    x = np.zeros((n, d))
    noise = rng.standard_normal((n, d))
    for j in range(d):
        x[:, j] = x @ b[j] + noise[:, j]
    return x


def _log_count(d: int, k: int) -> float:
    return math.log(math.comb(d - 1, k))


class _Reference:
    """The extended BIC from its formula, with no shortcut: one least-squares fit per family."""

    def __init__(self, x: np.ndarray, *, lambda_value: float = 1.0, ebic_gamma: float = 0.0):
        self.cov = np.atleast_2d(np.cov(x, rowvar=False))
        self.n, self.d = x.shape
        self.lam, self.gamma = lambda_value, ebic_gamma

    def score(self, node, parents) -> float:
        pa = sorted(int(p) for p in parents)
        k = len(pa)
        residual = self.cov[node, node]
        if pa:
            c = self.cov[np.ix_(pa, [node])]
            residual -= (c.T @ np.linalg.lstsq(self.cov[np.ix_(pa, pa)], c, rcond=None)[0]).item()
        penalty = self.lam * k * math.log(self.n) + 2 * self.gamma * _log_count(self.d, k)
        return self.n * math.log(residual) + penalty

    def score_many(self, node, parent_sets) -> list[float]:
        return [self.score(node, ps) for ps in parent_sets]

    def add_deltas(self, node, parents) -> np.ndarray:
        """``score(node, parents + u) - score(node, parents)`` per ``u``; ``inf`` for no move."""
        base = self.score(node, parents)
        out = np.full(self.d, np.inf)
        for u in range(self.d):
            if u != node and u not in parents:
                out[u] = self.score(node, [*parents, u]) - base
        return out


_X = _sem(3, 300, 8, 0.45)
_CASES = [(v, ps) for v in (0, 5) for ps in ((), (1,), (2, 6), (1, 3, 4, 7))]


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_term_is_the_log_count_of_parent_sets(gamma):
    for d, k in itertools.product((2, 8, 40), range(4)):
        if k < d:
            term = 2 * gamma * _log_count(d, k)
            assert ebic_penalty(k, d, gamma) == pytest.approx(term, abs=1e-9)
            step = 2 * gamma * (math.log(d - 1 - k) - math.log(k + 1)) if k + 1 < d else math.inf
            assert ebic_step(k, d, gamma) == pytest.approx(step, abs=1e-9)
    assert ebic_penalty(8, 8, gamma) == math.inf


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("cond", [1e10, 0.0], ids=["schur", "full-score fallback"])
def test_every_fast_path_matches_the_formula(gamma, cond):
    """Full scores, the scalar delta, the delta vectors, and both batched surfaces; ``cond=0``
    sends every non-empty base to the two-full-score fallback."""
    ref = _Reference(_X, lambda_value=1.5, ebic_gamma=gamma)
    score = BICScore(_X, lambda_value=1.5, ebic_gamma=gamma)
    engine = DeltaBICScore(score, cond_threshold=cond)
    d = _X.shape[1]
    for v, base in _CASES:
        assert score.score(v, base) == pytest.approx(ref.score(v, base), rel=1e-12)
        adds = [u for u in range(d) if u != v and u not in base]
        moves = [("add", u) for u in adds] + [("remove", u) for u in base]
        want = [
            ref.score(v, [*base, u] if op == "add" else [p for p in base if p != u])
            for op, u in moves
        ]
        diffs = [w - ref.score(v, base) for w in want]
        scalar = [engine.delta(v, tuple(sorted(base)), u, op) for op, u in moves]
        assert scalar == pytest.approx(diffs, rel=1e-9, abs=1e-9)
        batched = score_many_with_base(score, v, base, moves, cond_threshold=cond)
        assert batched == pytest.approx(want, rel=1e-12)
        raw = delta_many_with_base(score, v, base, moves, cond_threshold=cond)
        assert raw == pytest.approx(diffs, rel=1e-9, abs=1e-9)
        up = add_deltas(score, v, list(base), cond_threshold=cond)
        assert up[adds] == pytest.approx(diffs[: len(adds)], rel=1e-9, abs=1e-9)
        if base:
            down = remove_deltas(score, v, list(base), cond_threshold=cond)
            assert down[list(base)] == pytest.approx(diffs[len(adds) :], rel=1e-9, abs=1e-9)


def test_a_full_parent_set_has_no_add():
    """A node whose parents are every other variable has no add; the term is ``inf`` past it."""
    score = BICScore(_X[:, :3], ebic_gamma=1.0)
    assert np.isinf(add_deltas(score, 0, [1, 2])).all()


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_other_scores_carry_the_term(gamma):
    """Exact search's Gram score and the GIES score with no interventions."""
    n, d = _X.shape
    gram = _GramBICScore(_X, gamma)
    centered = _X - _X.mean(axis=0)
    gies_score = InterventionalBICScore(_X, None, ebic_gamma=gamma)
    ref = _Reference(_X, ebic_gamma=gamma)
    for v, base in _CASES:
        pa = list(base)
        rss = float(np.sum(centered[:, v] ** 2))
        if pa:
            fit = np.linalg.lstsq(centered[:, pa], centered[:, v], rcond=None)[0]
            rss = float(np.sum((centered[:, v] - centered[:, pa] @ fit) ** 2))
        want = n * math.log(rss / n) + len(pa) * math.log(n) + 2 * gamma * _log_count(d, len(pa))
        assert gram.score(v, pa) == pytest.approx(want, rel=1e-12)
        assert gies_score.score(v, pa) == pytest.approx(ref.score(v, pa), rel=1e-12)
        adds = [u for u in range(d) if u != v and u not in pa]
        assert gies_score.add_deltas(v, pa)[adds] == pytest.approx(
            ref.add_deltas(v, pa)[adds], rel=1e-9, abs=1e-9
        )


def _module(name: str):
    # ``andrey.search.<name>`` is the re-exported function; the module holds what to patch.
    return importlib.import_module(f"andrey.search.{name}")


def _reference_score(monkeypatch, module: str, gamma: float) -> None:
    def build(score_func, data, *, lambda_value=1.0, ebic_gamma=0.0, categorical=None):
        assert ebic_gamma == gamma
        return _Reference(np.asarray(data), lambda_value=lambda_value, ebic_gamma=ebic_gamma)

    monkeypatch.setattr(_module(module), "make_score", build)


_SEARCH = _sem(11, 120, 9, 0.4)
_SEARCHES = {
    "ges": ("ges", andrey.ges, {}),
    "hc": ("hc", andrey.hc, {}),
    "tabu": ("tabu", andrey.tabu, {}),
    "boss": ("boss", andrey.boss, {"seed": 0}),
    "grasp": ("boss", andrey.grasp, {"seed": 0}),
}


@pytest.mark.parametrize("name", list(_SEARCHES))
def test_each_search_matches_its_run_on_the_formula(monkeypatch, name):
    """The fast engine and the two-full-score path over :class:`_Reference` find one graph, with
    the formula's score; the term changes the graph, so the check has something to find."""
    module, run, kwargs = _SEARCHES[name]
    plain = run(_SEARCH, ebic_gamma=0.0, **kwargs)
    fast = run(_SEARCH, ebic_gamma=1.0, **kwargs)
    with monkeypatch.context() as patch:
        _reference_score(patch, module, 1.0)
        slow = run(_SEARCH, ebic_gamma=1.0, **kwargs)
    assert fast.structure.oriented_edges() == slow.structure.oriented_edges()
    assert fast.metadata["score"] == pytest.approx(slow.metadata["score"], rel=1e-10)
    assert fast.structure.oriented_edges() != plain.structure.oriented_edges()


def test_gies_and_xges_match_their_runs_on_the_formula(monkeypatch):
    """GIES and XGES build their scores themselves; their runs on :class:`_Reference` swap it in
    with its add deltas."""
    fast_gies = andrey.gies(_SEARCH, ebic_gamma=1.0)
    fast_xges = andrey.xges(_SEARCH, ebic_gamma=1.0)
    with monkeypatch.context() as patch:
        patch.setattr(
            _module("gies"),
            "InterventionalBICScore",
            lambda x, mask, **kw: _Reference(x, **kw),
        )
        patch.setattr(_module("xges"), "BICScore", lambda x, **kw: _Reference(x, **kw))
        patch.setattr(_module("xges"), "add_deltas", lambda score, y, e: score.add_deltas(y, e))
        slow_gies = andrey.gies(_SEARCH, ebic_gamma=1.0)
        slow_xges = andrey.xges(_SEARCH, ebic_gamma=1.0)
    for fast, slow in ((fast_gies, slow_gies), (fast_xges, slow_xges)):
        assert fast.structure.oriented_edges() == slow.structure.oriented_edges()
        assert fast.metadata["score"] == pytest.approx(slow.metadata["score"], rel=1e-10)
    plain = andrey.xges(_SEARCH, ebic_gamma=0.0)
    assert fast_xges.structure.oriented_edges() != plain.structure.oriented_edges()


def _dags(d: int):
    """Every DAG on ``d`` nodes, as boolean parent matrices ``a[i, j]`` for ``i -> j``."""
    pairs = list(itertools.combinations(range(d), 2))
    for marks in itertools.product((0, 1, 2), repeat=len(pairs)):
        a = np.zeros((d, d), dtype=bool)
        for (i, j), m in zip(pairs, marks, strict=True):
            if m == 1:
                a[i, j] = True
            elif m == 2:
                a[j, i] = True
        power = a.astype(int)
        acyclic = True
        for _ in range(d):
            if np.trace(power):
                acyclic = False
                break
            power = power @ a
        if acyclic:
            yield a


@pytest.mark.parametrize("gamma", (0.0, 1.0))
def test_exact_search_finds_the_optimum_of_the_formula(gamma):
    """Over all 543 DAGs on four variables, exact search returns one of the lowest EBIC."""
    x = _sem(5, 60, 4, 0.6)
    ref = _Reference(x, ebic_gamma=gamma)

    def total(a: np.ndarray) -> float:
        return sum(ref.score(j, np.flatnonzero(a[:, j]).tolist()) for j in range(4))

    best = min(total(a) for a in _dags(4))
    found = andrey.exact_search(x, ebic_gamma=gamma).structure.to_numpy()
    assert total(found == TAIL) == pytest.approx(best, rel=1e-10)


def test_make_score_checks_the_weight_under_every_score():
    with pytest.raises(ValueError, match="ebic_gamma"):
        make_score("local_score_BDeu", np.zeros((4, 2)), ebic_gamma=-1.0)
