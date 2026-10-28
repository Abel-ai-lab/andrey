"""FCI's ``possible_dsep``: both modes keep their recorded PAGs, and unknown values raise.

``test_fci_oracle`` and the recovery tests check that the default does not depend on the column
order. The step reads the neighbor lists and semidirected reach of the marks once per version of
the marks, and builds them again only after an edge is removed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey.core import NULL, ci
from andrey.core.ci import make_indep_test
from andrey.data import sample_discrete, sample_mixed, sample_scm

fci = importlib.import_module("andrey.constraint.fci")

# Recorded PAGs, the same for both modes on these draws. Row i holds ``to_numpy()``'s marks at
# node i: 0 none, 1 tail, 2 arrowhead, 3 circle.
RECORDED = {
    "fci": (
        {"d": 15, "n": 750, "density": 4.0, "seed": 3},
        ["00000000303300", "00000000020200", "00000002000021", "00003000003000"]
        + ["00030030000000", "00000000000000", "00003000000000", "00200000220001"]
        + ["20000002000300", "02000002001000", "20020000020020", "22000000300000"]
        + ["00200000002000", "00200002000000"],
    ),
    "gfci": (
        {"d": 8, "n": 400, "density": 4.0, "seed": 2},
        ["0200020", "2021001", "0201002", "0220100"] + ["0002022", "2000200", "0210200"],
    ),
}


@pytest.mark.parametrize("possible_dsep", ["order_free", "fast"])
@pytest.mark.parametrize("method", sorted(RECORDED))
def test_keeps_the_recorded_pag(method, possible_dsep):
    draw, rows = RECORDED[method]
    x = sample_scm(graph="erdos_renyi", latents=1, scale="standardize", **draw).data
    recorded = np.array([[int(mark) for mark in row] for row in rows])
    fit = getattr(andrey, method)(x, possible_dsep=possible_dsep)
    np.testing.assert_array_equal(fit.structure.to_numpy(), recorded)


@pytest.mark.parametrize(("method", "value"), [("fci", "order-free"), ("gfci", "FAST")])
def test_unknown_possible_dsep_raises(method, value):
    with pytest.raises(ValueError, match="possible_dsep must be one of 'order_free', 'fast'"):
        getattr(andrey, method)(
            np.random.default_rng(0).standard_normal((50, 3)), possible_dsep=value
        )


def _rebuilt_per_call(marks, test, alpha, sepsets):
    """``_remove_in_node_order`` with the lists and reach built for every possible-d-sep call."""
    n = marks.shape[0]
    edges = [fci._edge_order(marks, i, j) for i in range(n) for j in range(i + 1, n) if marks[i, j]]
    for na, nb in edges:
        if not fci._adjacent(marks, na, nb):
            continue
        adj_a, adj_b = (set(np.flatnonzero(marks[v]).tolist()) for v in (na, nb))
        for a, b in ((na, nb), (nb, na)):
            dsep = fci._possible_dsep(
                marks, a, b, fci._neighbor_lists(marks), fci._semidirected_reach(marks)
            )
            if fci._try_remove_by_dsep(marks, na, nb, dsep, adj_a, adj_b, test, alpha, sepsets):
                break


@pytest.mark.parametrize(
    "draw",
    [
        {"d": 10, "n": 500, "density": 4.0, "latents": 1, "seed": 9},
        {"d": 15, "n": 500, "density": 4.0, "latents": 2, "seed": 7},
        {"d": 15, "n": 1000, "density": 3.0, "latents": 1, "seed": 3},
    ],
)
def test_the_lists_are_built_once_per_version_of_the_marks(monkeypatch, draw):
    alpha = 0.05
    x = sample_scm(graph="erdos_renyi", scale="standardize", **draw).data
    test = make_indep_test("fisherz", x)
    support, sepsets = fci._fast_adjacency_search(x, test, alpha)
    marks = fci._circle_pag(support)
    fci._orient_colliders(marks, sepsets)

    reference, reference_sepsets = marks.copy(), dict(sepsets)
    _rebuilt_per_call(reference, test, alpha, reference_sepsets)
    builds = {"lists": 0, "reach": 0}
    for name, key in (("_neighbor_lists", "lists"), ("_semidirected_reach", "reach")):
        monkeypatch.setattr(fci, name, _counting(getattr(fci, name), builds, key))
    fci._remove_in_node_order(marks, test, alpha, sepsets, None, None)

    removed = (np.count_nonzero(support) - np.count_nonzero(marks != NULL)) // 2
    assert removed >= 2, "the draw must remove edges, so that a stale list would show"
    np.testing.assert_array_equal(marks, reference)
    assert sepsets == reference_sepsets
    assert builds == {"lists": removed + 1, "reach": removed + 1}


def _counting(function, builds, key):
    def counted(*args):
        builds[key] += 1
        return function(*args)

    return counted


class _Scripted:
    """A test whose p-value of each conditioning set is set in advance.

    ``answered`` logs the queries a caller is answered, as the identity check counts them: each
    scalar call, and the sets ``separating_prefix`` returns p-values for, not the calls it makes
    inside. ``raising`` names a set whose evaluation raises, alone or in a batch. Without
    ``batched``, the test has no ``separating_prefix`` and takes one call per subset.
    """

    def __init__(self, pvalues, raising=None, batched=True):
        self.pvalues, self.raising, self.answered, self.inside = pvalues, raising, [], False
        if not batched:
            self.separating_prefix = None

    def _p(self, condition_set):
        key = tuple(sorted(condition_set))
        if key == self.raising:
            raise FloatingPointError(f"set {key}")
        return self.pvalues.get(key, 0.0)

    def __call__(self, x, y, condition_set=()):
        p = self._p(condition_set)
        if not self.inside:
            self.answered.append((tuple(condition_set), p))
        return p

    def batched_call(self, x, y, condition_sets):
        return [self._p(s) for s in condition_sets]

    def separating_prefix(self, x, y, condition_sets, alpha):
        self.inside = True
        try:
            pvalues = ci._separating_prefix(self, x, y, condition_sets, alpha)
        finally:
            self.inside = False
        self.answered.extend(zip(map(tuple, condition_sets), pvalues))
        return pvalues


def _dsep_run(test, dsep, alpha=0.05):
    marks = np.full((12, 12), 1, dtype=np.int8)
    sepsets: dict = {}
    try:
        removed, error = (
            fci._try_remove_by_dsep(marks, 0, 1, dsep, {2}, {3}, test, alpha, sepsets),
            None,
        )
    except FloatingPointError as exc:
        removed, error = None, str(exc)
    return removed, sepsets, int(marks[0, 1]), test.answered, error


@pytest.mark.parametrize("seed", range(48))
def test_runs_answer_the_one_at_a_time_queries(seed, monkeypatch):
    """Runs of 1, 2, 4, ... subsets give the one-at-a-time loop's separating set and answered
    queries, in order, whether the pair separates early, late, or never; a set that raises raises
    only when the loop reaches it, after the queries the loop answers first."""
    monkeypatch.setattr(fci, "_DSEP_RUN", 8)  # several capped runs on a small case
    rng = np.random.default_rng(seed)
    dsep = rng.permutation(np.arange(2, 12)).tolist()[: rng.integers(3, 8)]
    choices = list(fci._iter_dsep_choices(dsep, {2}, {3}))
    pvalues = {tuple(sorted(c)): float(rng.uniform(0.0, 0.04)) for c in choices}
    where = ("early", "late", "never")[seed % 3]
    if where != "never":
        late = int(rng.integers(len(choices) // 2, len(choices)))
        at = int(rng.integers(0, 3)) if where == "early" else late
        pvalues[tuple(sorted(choices[at]))] = 0.5
    raising = tuple(sorted(choices[rng.integers(len(choices))])) if seed % 2 else None
    removed, sepsets, mark, answered, error = _dsep_run(_Scripted(pvalues, raising), dsep)
    want = _dsep_run(_Scripted(pvalues, raising, batched=False), dsep)
    assert (removed, sepsets, mark, error) == (want[0], want[1], want[2], want[4])
    if error is None:
        assert answered == want[3]
    else:  # the fit raises; the queries answered before the failing run are the loop's
        assert answered == want[3][: len(answered)]


@pytest.mark.parametrize("test_name", ["fisherz", "chisq", "degen_gauss", "cond_gauss"])
@pytest.mark.parametrize("possible_dsep", ["order_free", "fast"])
def test_shipped_tests_answer_the_one_at_a_time_queries(test_name, possible_dsep, monkeypatch):
    """FCI with each shipped test that has ``separating_prefix`` returns the PAG and answers the
    possible-d-sep queries, p-values included, of the one-at-a-time loop, which a test without it
    takes."""

    if test_name == "chisq":
        x = sample_discrete(d=12, n=600, seed=5, density=5.0).data
    elif test_name == "fisherz":
        x = sample_scm(d=15, n=750, density=4.0, seed=3).data
    else:
        x = sample_mixed(d=12, n=600, seed=1, density=5.0).data
    asked = []
    try_remove = fci._try_remove_by_dsep

    def logged(marks, x_, y_, dsep, adj_a, adj_b, test, *args):
        def answer(a, b, condition_set):
            p = test(a, b, condition_set)
            asked.append((a, b, tuple(condition_set), p))
            return p

        return try_remove(marks, x_, y_, dsep, adj_a, adj_b, answer, *args)

    monkeypatch.setattr(fci, "_try_remove_by_dsep", logged)
    x = np.asarray(x, float)
    want = andrey.fci(x, indep_test=test_name, possible_dsep=possible_dsep)
    one_at_a_time = list(asked)
    monkeypatch.setattr(fci, "_try_remove_by_dsep", try_remove)
    prefixes = []
    for cls in (ci.PartialCorrelationTest, ci.ChiSquare, ci.DGTest, ci.CGTest):
        method = vars(cls)["separating_prefix"]

        def spied(self, a, b, sets, alpha, _method=method):
            pvalues = _method(self, a, b, sets, alpha)
            prefixes.extend((a, b, tuple(s), p) for s, p in zip(sets, pvalues))
            return pvalues

        monkeypatch.setattr(cls, "separating_prefix", spied)
    got = andrey.fci(x, indep_test=test_name, possible_dsep=possible_dsep)
    assert np.array_equal(got.structure.to_numpy(), want.structure.to_numpy())
    assert prefixes == one_at_a_time
    assert one_at_a_time, "the case must reach the possible-d-sep step"
