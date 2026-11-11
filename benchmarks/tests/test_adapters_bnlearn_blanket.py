"""bnlearn's set learners, through R: each learner's answer, the settings that reach R, and its
agreement with Andrey on cases both must get right.

These checks run where R is loaded and skip elsewhere (``needs_r``); the table and option checks
that need no R are in ``test_blanket_lane.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_r

from andrey_bench import blanket_lane as lane
from andrey_bench.adapters.andrey_blanket import AndreyBlanket
from andrey_bench.adapters.bnlearn_blanket import FUNCTIONS, BnlearnBlanket, check_version

pytestmark = needs_r()


def _collider(n: int = 3000, seed: int = 0) -> np.ndarray:
    """Columns ``[a, b, c, e]`` with ``a -> c <- b`` and ``e`` independent of the rest."""
    rng = np.random.default_rng(seed)
    a, b, e = rng.normal(size=(3, n))
    return np.column_stack([a, b, a + b + 0.5 * rng.normal(size=n), e])


def _sets(adapter, X: np.ndarray, targets=None, **params) -> dict[int, list[int]]:
    adapter.setup()
    targets = list(range(X.shape[1])) if targets is None else targets
    native = adapter.fit(X, {**adapter.params(), "targets": targets, **params})
    return lane.sets_of(lane.unpack_answer(native)[0], targets)


def test_the_pinned_bnlearn_is_the_one_in_the_library():
    check_version()  # raises SystemExit on another version


@pytest.mark.parametrize("function", [f for f, known in FUNCTIONS.items() if known.kind == "mb"])
def test_a_blanket_learner_finds_the_colliders_blankets(function):
    adapter = BnlearnBlanket(function, indep_test="fisherz")
    assert _sets(adapter, _collider()) == {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: []}


@pytest.mark.parametrize("function", [f for f, known in FUNCTIONS.items() if known.kind == "pc"])
def test_a_pc_learner_finds_the_colliders_parents_and_children(function):
    adapter = BnlearnBlanket(function, indep_test="fisherz")
    assert _sets(adapter, _collider()) == {0: [2], 1: [2], 2: [0, 1], 3: []}


@pytest.mark.parametrize("kind", ["mb", "pc"])
def test_pc_stable_is_read_at_each_target_in_one_fit(kind):
    adapter = BnlearnBlanket("pc.stable", indep_test="fisherz", kind=kind)
    assert adapter.name == f"bnlearn.pc-stable.{kind}"
    adapter.setup()
    X = _collider()
    answer, counts, total = lane.unpack_answer(
        adapter.fit(X, {**adapter.params(), "targets": [0, 2]})
    )
    expected = {"mb": {0: [1, 2], 2: [0, 1]}, "pc": {0: [2], 2: [0, 1]}}[kind]
    assert lane.sets_of(answer, [0, 2]) == expected
    assert not answer[[1, 3]].any()
    assert total > 0 and (counts == lane.NO_COUNT).all()  # one fit, one count


def test_the_tests_of_each_search_are_counted():
    adapter = BnlearnBlanket("iamb", indep_test="fisherz")
    adapter.setup()
    _, counts, total = lane.unpack_answer(
        adapter.fit(_collider(), {**adapter.params(), "targets": [0, 3]})
    )
    assert counts[0] > 0 and counts[3] > 0 and counts[1] == counts[2] == lane.NO_COUNT
    assert total == counts[0] + counts[3]


def test_a_member_of_a_pc_set_holds_the_target_in_its_own():
    """The adapter keeps a variable only when its own search holds the target (the AND rule)."""
    rng = np.random.default_rng(1)
    d, n = 8, 400
    X = rng.normal(size=(n, d))
    for j in range(1, d):  # a chain with weak links, so the searches can disagree
        X[:, j] += 0.35 * X[:, j - 1]
    found = _sets(BnlearnBlanket("mmpc", indep_test="fisherz"), X)
    assert all(t in found[j] for t in found for j in found[t])


def test_the_sparse_table_rule_is_the_one_the_option_picks():
    rng = np.random.default_rng(0)
    n = 60  # fewer than 5 rows for each of the 16 cells of a pair of 4-level variables
    x = rng.integers(0, 4, size=n)
    y = np.where(rng.random(n) < 0.9, x, rng.integers(0, 4, size=n))
    X = np.column_stack([x, y, rng.integers(0, 4, size=n)]).astype(float)
    with_rule = BnlearnBlanket("iamb", indep_test="gsq", sparse_rule=True)
    without = BnlearnBlanket("iamb", indep_test="gsq", sparse_rule=False)
    assert (with_rule.params()["bnlearn_test"], without.params()["bnlearn_test"]) == (
        "mi-adf",
        "mi",
    )
    # mi-adf takes the table as too sparse to test and reports independence; mi tests it.
    assert _sets(with_rule, X, [0]) == {0: []}
    assert _sets(without, X, [0]) == {0: [1]}
    # Andrey's G-square test has no such rule.
    mine = AndreyBlanket("iamb", indep_test="gsq")
    assert _sets(mine, X, [0]) == {0: [1]}


def test_the_limit_on_the_conditioning_set_reaches_bnlearn():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 5))
    X[:, 4] = X[:, :4].sum(axis=1) + 0.5 * rng.normal(size=2000)  # four parents
    free = _sets(BnlearnBlanket("iamb", indep_test="fisherz"), X, [4])[4]
    capped = _sets(BnlearnBlanket("iamb", indep_test="fisherz", max_depth=1), X, [4])[4]
    assert free == [0, 1, 2, 3] and 0 < len(capped) < len(free)
    mine = _sets(AndreyBlanket("iamb", indep_test="fisherz", max_depth=1), X, [4])[4]
    assert 0 < len(mine) < len(free)


@pytest.mark.parametrize("learner", ["iamb", "inter_iamb", "mmpc"])
def test_andrey_and_bnlearn_agree_on_a_case_both_must_get_right(learner):
    from andrey_bench.adapters.bnlearn_blanket import COUNTERPART

    X = _collider()
    reference = BnlearnBlanket(COUNTERPART[learner], indep_test="fisherz")
    assert _sets(AndreyBlanket(learner, indep_test="fisherz"), X) == _sets(reference, X)


def test_a_whole_graph_comes_back_as_the_graph_of_each_structure():
    X = _collider()
    moral = BnlearnBlanket("iamb", indep_test="fisherz", structure="moral")
    skeleton = BnlearnBlanket("iamb", indep_test="fisherz", structure="skeleton")
    cpdag = BnlearnBlanket("iamb", indep_test="fisherz", structure="cpdag")
    answers = {}
    for adapter in (moral, skeleton, cpdag):
        adapter.setup()
        native = adapter.fit(X, adapter.params())
        answers[adapter.structure] = (lane.unpack_answer(native)[0], adapter.to_structure(native))
        assert lane.unpack_answer(native)[2] > 0
    assert np.array_equal(answers["moral"][0], lane.true_graph(_dag(), "moral"))
    assert np.array_equal(answers["skeleton"][0], lane.true_graph(_dag(), "skeleton"))
    assert answers["moral"][1].kind == answers["skeleton"][1].kind == "undirected"
    directed = answers["cpdag"][1]
    assert directed.kind == "cpdag" and directed.to_numpy()[2, 0] == 2  # a -> c


def _dag() -> np.ndarray:
    marks = np.zeros((4, 4), dtype=np.int8)
    for i, j in ((0, 2), (1, 2)):
        marks[i, j], marks[j, i] = 1, 2
    return marks


def test_every_function_runs_per_target_and_for_the_whole_graph():
    X = _collider(800)
    for function, known in FUNCTIONS.items():
        # pc.stable gives either set, so it is told which to read.
        adapter = BnlearnBlanket(
            function, indep_test="fisherz", kind="mb" if known.kind is None else None
        )
        found = _sets(adapter, X, [0, 2])
        assert set(found) == {0, 2}
        for structure in known.structures:
            graph = BnlearnBlanket(function, indep_test="fisherz", structure=structure)
            graph.setup()
            native = graph.fit(X, graph.params())
            assert lane.unpack_answer(native)[0].shape == (4, 4)
            graph.to_structure(native).validate()


def test_discrete_columns_are_read_as_factors_with_the_discrete_tests():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 3, size=(2, 1500))
    c = (a + b + rng.integers(0, 2, size=1500)) % 3
    X = np.column_stack([a, b, c]).astype(float)
    for test in ("gsq", "chisq"):
        for rule in (True, False):
            found = _sets(BnlearnBlanket("iamb", indep_test=test, sparse_rule=rule), X)
            assert found[2] == [0, 1]
