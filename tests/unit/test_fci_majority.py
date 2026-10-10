"""Majority collider decisions, separating-set rewrites, and ambiguous orientation paths."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint import fci as fci_module
from andrey.core import ARROW, CIRCLE, TAIL


def circle_graph(n, edges):
    marks = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        marks[a, b] = marks[b, a] = CIRCLE
    return marks


@pytest.mark.parametrize(
    "separating,decision",
    [
        ([], "collider"),
        ([()], "collider"),
        ([(1,)], "noncollider"),
        ([(), (1, 3)], "collider"),
        ([(1,), (3,)], "noncollider"),
        ([(), (1,)], "ambiguous"),
    ],
    ids=["no-separator", "none-containing", "all-containing", "less", "more", "half"],
)
@pytest.mark.parametrize("batched", [False, True])
def test_majority_decisions(separating, decision, batched):
    """Shared subsets count twice; p == alpha passes; recorded sets never vote."""
    marks = circle_graph(5, [(0, 1), (1, 2), (0, 3), (2, 4)])
    before = marks.copy()
    sepsets = {(0, 2): (1, 3), (2, 0): (4,)}
    calls = []

    class Oracle:
        def __call__(self, a, c, subset=()):
            assert not batched, "The batched oracle must not be called one subset at a time"
            calls.append((a, c, subset))
            return 0.05 if tuple(subset) in separating else 0.0

    oracle = Oracle()
    if batched:

        def batch(a, c, subsets):
            calls.extend((a, c, subset) for subset in subsets)
            return [0.05 if subset in separating else 0.0 for subset in subsets]

        oracle.batched_call = batch

    ambiguous = fci_module._majority_sepsets(marks, oracle, 0.05, sepsets)
    queried = [subset for a, c, subset in calls if (a, c) == (0, 2)]
    assert queried == [(), (1,), (3,), (1, 3), (), (1,), (4,), (1, 4)]
    assert ((0, 1, 2) in ambiguous) == (decision == "ambiguous")
    if decision == "collider":
        assert sepsets[0, 2] == (3,)
        assert sepsets[2, 0] == (4,)
    elif decision == "noncollider":
        assert sepsets[0, 2] == (1, 3)
        assert sepsets[2, 0] == (4, 1)
    else:
        assert sepsets[0, 2] == (1, 3)
        assert sepsets[2, 0] == (4,)
    np.testing.assert_array_equal(marks, before)


def test_majority_has_no_conditioning_size_cap():
    marks = circle_graph(6, [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)])
    sepsets = {}
    fci_module._majority_sepsets(
        marks, lambda a, c, subset: float(subset == (1, 3, 4, 5)), 0.05, sepsets
    )
    assert sepsets[0, 2] == sepsets[2, 0] == (1,)


@pytest.mark.parametrize("engine", ["fci", "gfci"])
def test_public_collider_rule_validation(engine):
    with pytest.raises(ValueError, match="collider_rule"):
        getattr(andrey, engine)(
            np.random.default_rng(0).standard_normal((8, 2)), collider_rule="conservative"
        )


@pytest.mark.parametrize("engine", ["fci", "gfci"])
def test_majority_runs_after_possible_dsep_only(engine, monkeypatch):
    module = importlib.import_module(
        "andrey.constraint.fci" if engine == "fci" else "andrey.search.gfci"
    )
    events = []
    snapshots = {}
    for name, event in [
        ("_orient_colliders", "colliders"),
        ("_remove_by_possible_dsep", "possible_dsep"),
        ("_majority_sepsets", "majority"),
        ("_fci_orient", "orient"),
    ]:
        original = getattr(module, name)

        def wrap(marks, *args, original=original, event=event, **kwargs):
            events.append(event)
            if event == "possible_dsep":
                snapshots["before_possible_dsep"] = marks != 0
            if event == "majority":
                snapshots[event] = marks != 0
            result = original(marks, *args, **kwargs)
            if event == "possible_dsep":
                snapshots[event] = marks != 0
            return result

        monkeypatch.setattr(module, name, wrap)
    arr = data.sample_scm(graph="scale_free", d=6, n=500, seed=5, density=2.0, latents=1).data
    fit = getattr(andrey, engine)
    fit(arr, collider_rule="majority")
    assert events == ["colliders", "possible_dsep", "majority", "colliders", "orient"]
    np.testing.assert_array_equal(snapshots["majority"], snapshots["possible_dsep"])
    if engine == "fci":
        assert np.count_nonzero(snapshots["before_possible_dsep"]) > np.count_nonzero(
            snapshots["possible_dsep"]
        )
    events.clear()
    implicit = fit(arr).structure.to_numpy()
    assert "majority" not in events
    explicit = fit(arr, collider_rule="sepsets").structure.to_numpy()
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("rule", ["r0", "r1", "r3", "r7", "r9", "r10"])
@pytest.mark.parametrize("reverse_labels", [False, True])
def test_rules_skip_ambiguous_triple(rule, reverse_labels):
    if rule in ("r0", "r1", "r7"):
        marks = circle_graph(3, [(0, 1), (1, 2)])
        triple = (0, 1, 2)
        sepsets = {(0, 2): ()}
        if rule == "r1":
            marks[1, 0] = ARROW
        elif rule == "r7":
            marks[0, 1] = TAIL
    elif rule == "r3":
        marks = circle_graph(4, [(0, 1), (2, 1), (0, 3), (2, 3), (3, 1)])
        marks[1, 0] = marks[1, 2] = ARROW
        triple = (0, 3, 2)
        sepsets = {(0, 2): (3,)}
    elif rule == "r9":
        marks = circle_graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        marks[3, 0] = ARROW
        triple = (0, 1, 2)
        sepsets = {}
    else:
        marks = circle_graph(4, [(0, 1), (0, 2), (0, 3), (2, 1), (3, 1)])
        marks[1, 0] = marks[1, 2] = marks[1, 3] = ARROW
        marks[2, 1] = marks[3, 1] = TAIL
        triple = (2, 0, 3)
        sepsets = {}
    if reverse_labels:
        n = len(marks)
        marks = marks[::-1, ::-1].copy()
        triple = tuple(n - 1 - node for node in triple)
        sepsets = {
            (n - 1 - a, n - 1 - c): tuple(n - 1 - node for node in cond)
            for (a, c), cond in sepsets.items()
        }
    before = marks.copy()
    ambiguous = frozenset([fci_module._triple(*triple)])

    def orient(m, ties):
        if rule == "r0":
            fci_module._orient_colliders(m, sepsets, ties)
        elif rule == "r1":
            fci_module._rule_r1r2_cycle(m, False, ties)
        elif rule == "r3":
            fci_module._rule_r3(m, sepsets, False, ties)
        else:
            getattr(fci_module, f"_rule_{rule}")(m, False, ties)

    orient(marks, ambiguous)
    if rule == "r7":
        # Other eligible endpoints may orient; the tied triple must not set this tail.
        assert marks[triple[1], triple[2]] == CIRCLE
    else:
        np.testing.assert_array_equal(marks, before)
    marks = before.copy()
    orient(marks, frozenset())
    assert not np.array_equal(marks, before)


@pytest.mark.parametrize("triple", [(0, 2, 4), (0, 3, 5)])
def test_r10_rejects_ambiguous_path(triple):
    marks = circle_graph(6, [(0, 1), (0, 2), (2, 4), (0, 3), (3, 5), (4, 1), (5, 1)])
    marks[1, 0] = marks[1, 4] = marks[1, 5] = ARROW
    marks[4, 1] = marks[5, 1] = TAIL
    before = marks.copy()
    assert not fci_module._rule_r10(marks, False, frozenset([triple]))
    np.testing.assert_array_equal(marks, before)
    assert fci_module._rule_r10(marks, False)
    assert marks[0, 1] == TAIL


@pytest.mark.parametrize(
    "triples",
    [[(0, 1, 2)], [(0, 1, 2), (1, 0, 3)], [(0, 1, 2), (0, 3, 2)]],
    ids=["single", "adjacent", "opposite"],
)
@pytest.mark.parametrize("reverse_labels", [False, True])
def test_r5_rejects_ambiguous_circle_paths(triples, reverse_labels):
    marks = circle_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    if reverse_labels:
        marks = marks[::-1, ::-1].copy()
        triples = [tuple(3 - node for node in triple) for triple in triples]
    before = marks.copy()
    # A collider at a tied node is still possible, so the cycle cannot force all tails.
    ambiguous = frozenset(fci_module._triple(*triple) for triple in triples)
    assert not fci_module._rule_r5(marks, False, ambiguous)
    np.testing.assert_array_equal(marks, before)
    assert fci_module._rule_r5(marks, False)
    np.testing.assert_array_equal(marks, (before != 0) * TAIL)


@pytest.mark.parametrize("ties", [[], [(1, 3, 4)], [(0, 1, 5)]])
def test_circle_path_keeps_global_visited_search(ties):
    marks = circle_graph(6, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
    paths = list(fci_module._uncovered_circle_paths(marks, 0, 5, (-1, -1), frozenset(ties)))
    # A tied first route is rejected; unrelated ties do not expand the search to other prefixes.
    assert paths == ([] if ties == [(1, 3, 4)] else [[0, 1, 3, 4, 5]])


@pytest.mark.parametrize("middle_in_set", [False, True])
def test_r4_reads_rewritten_separating_set(middle_in_set):
    marks = circle_graph(4, [(0, 1), (1, 2), (2, 3), (1, 3)])
    sepsets = {(0, 3): () if middle_in_set else (2,)}
    # For the triple 0-1-3, node 2 is unrelated; create the 0-2-3 adjacency for voting.
    vote_graph = circle_graph(4, [(0, 2), (2, 3)])
    fci_module._majority_sepsets(
        vote_graph, lambda a, c, subset: float((2 in subset) == middle_in_set), 0.05, sepsets
    )
    oriented, changed = fci_module._do_ddp(
        marks, 0, 1, 2, 3, {0: 1, 1: 2}, lambda *args: 0.0, 0.05, sepsets, False
    )
    assert oriented and changed
    assert marks[2, 3] == (TAIL if middle_in_set else ARROW)


@pytest.mark.parametrize("rule", ["r2", "r6", "r8"])
def test_orientation_rules_without_ambiguity_guard(rule):
    marks = circle_graph(3, [(0, 1), (1, 2)])
    if rule == "r6":
        marks[0, 1] = marks[1, 0] = TAIL
        expected_endpoint = (1, 2)
        expected_mark = TAIL
    else:
        marks[0, 2] = marks[2, 0] = CIRCLE
        marks[0, 1] = TAIL
        marks[1, 0] = marks[2, 1] = ARROW
        expected_endpoint = (2, 0)
        expected_mark = ARROW
        if rule == "r8":
            marks[1, 2] = TAIL
            marks[2, 0] = ARROW
            expected_endpoint = (0, 2)
            expected_mark = TAIL
    fci_module._fci_orient(marks, lambda *args: 0.0, 0.05, {}, frozenset([(0, 1, 2)]))
    assert marks[expected_endpoint] == expected_mark


@pytest.mark.parametrize("pvalue", [0.0, 1.0])
def test_missing_recorded_sets(pvalue):
    marks = circle_graph(3, [(0, 1), (1, 2)])
    sepsets = {}
    ambiguous = fci_module._majority_sepsets(marks, lambda *args: pvalue, 0.05, sepsets)
    fci_module._orient_colliders(marks, sepsets, ambiguous)
    if pvalue == 0.0:
        assert sepsets == {(0, 2): (), (2, 0): ()}
        assert not ambiguous
        assert marks[1, 0] == marks[1, 2] == ARROW
    else:
        assert sepsets == {}
        assert ambiguous == frozenset([(0, 1, 2)])
        assert marks[1, 0] == marks[1, 2] == CIRCLE


@pytest.mark.parametrize("pair", [(0, 2), (2, 0)])
@pytest.mark.parametrize("decision", ["collider", "noncollider", "ambiguous"])
def test_majority_preserves_members_of_one_way_recorded_set(pair, decision):
    marks = circle_graph(4, [(0, 1), (1, 2), (0, 3)])
    recorded = (3,) if decision == "noncollider" else (1, 3)
    sepsets = {pair: recorded}

    def oracle(a, c, subset):
        if decision == "ambiguous":
            return 1.0
        return float(decision == "noncollider" and 1 in subset)

    ambiguous = fci_module._majority_sepsets(marks, oracle, 0.05, sepsets)
    if decision == "ambiguous":
        assert (0, 1, 2) in ambiguous
        assert sepsets[pair] == recorded
        assert pair[::-1] not in sepsets
    else:
        expected = frozenset([1, 3] if decision == "noncollider" else [3])
        assert fci_module._sepset(sepsets, 0, 2) == expected
        assert fci_module._sepset(sepsets, 2, 0) == expected


@pytest.mark.parametrize("pair", [(0, 3), (3, 0)])
def test_r4_preserves_unvoted_member_from_one_way_separating_set(pair):
    marks = circle_graph(4, [(0, 1), (1, 2), (2, 3), (1, 3)])
    # Node 2 is not a common neighbor of 0 and 3, so their vote only decides node 1.
    sepsets = {pair: (2,)}
    fci_module._majority_sepsets(marks, lambda *args: 0.0, 0.05, sepsets)
    marks[1, 0] = marks[1, 2] = marks[3, 1] = ARROW
    marks[1, 3] = TAIL
    done, changed = fci_module._do_ddp(
        marks, 0, 1, 2, 3, {0: 1, 1: 2}, lambda *args: 0.0, 0.05, sepsets, False
    )
    assert done and changed
    assert marks[2, 3] == TAIL
