"""FCI's possible-d-sep searches: the array enumeration, the worker pool, and the skipped subsets.

A pair's search tests the subsets of its candidate set by increasing size, in ``combinations``
order, and the first that separates the pair removes it. With a FisherZ, StudentT, or NormalScores
test a larger search enumerates its subsets as arrays, and with more than one worker a search of
at least the cutoff's subsets runs on the worker pool. After the first endpoint's search finds
nothing, the second endpoint's leaves out the subsets of the first endpoint's set. None of these
changes the subset found, so the PAG and the separating sets are the same on any worker count. The
pooled comparisons spy on the pool's creation, so they cannot silently run serial against serial.
"""

from __future__ import annotations

import importlib
from itertools import combinations

import numpy as np
import pytest

import andrey
from andrey.constraint import _parallel_skeleton
from andrey.constraint._skeleton import block_positions, block_subsets, blocks
from andrey.constraint._skeleton import first_separating_subset as first_separating
from andrey.constraint.pc import adjacency_search
from andrey.core import CIRCLE, NULL
from andrey.core.ci import PartialCorrelationTest, make_indep_test
from andrey.data import sample_scm
from tests.recovery.helpers.jci import random_dag, sample_pooled

fci = importlib.import_module("andrey.constraint.fci")

# Draws whose possible-d-sep step removes edges (test_fci_possible_dsep).
DRAWS = [
    {"d": 10, "n": 500, "density": 4.0, "latents": 1, "seed": 9},
    {"d": 15, "n": 500, "density": 4.0, "latents": 2, "seed": 7},
]
GATE = "ANDREY_POSSIBLE_DSEP_PARALLEL_MIN_WORK"


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture
def pools(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The worker count of each pool process executor started during the test."""
    created: list[int] = []
    real = _parallel_skeleton.ProcessPoolExecutor

    def spy(*args, **kwargs):
        created.append(kwargs["max_workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(_parallel_skeleton, "ProcessPoolExecutor", spy)
    return created


@pytest.fixture
def pooled_searches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The candidate count of each search run on the pool during the test."""
    searched: list[int] = []
    real = _parallel_skeleton.SkeletonWorkerPool.first_separating

    def spy(self, x, y, candidates, *args):
        searched.append(len(candidates))
        return real(self, x, y, candidates, *args)

    monkeypatch.setattr(_parallel_skeleton.SkeletonWorkerPool, "first_separating", spy)
    return searched


class _Scripted:
    """A test that finds every set dependent but ``separating`` and logs each set it answers."""

    def __init__(self, separating: tuple[int, ...] | None = None) -> None:
        self.separating, self.asked = separating, []

    def __call__(self, x, y, condition_set=()):
        raise AssertionError("an array search asks in runs")

    def separating_prefix(self, x, y, condition_sets, alpha):
        pvalues = []
        for condition_set in condition_sets:
            self.asked.append(tuple(int(v) for v in condition_set))
            pvalues.append(0.5 if self.asked[-1] == self.separating else 0.0)
            if pvalues[-1] > alpha:
                break
        return pvalues


@pytest.mark.parametrize("most", [1, 2, 5, 40])
@pytest.mark.parametrize("n", range(8))
def test_tasks_hold_the_subsets_in_combinations_order(n: int, most: int) -> None:
    sizes = range(1, n + 1)
    tasks = list(_parallel_skeleton.dsep_tasks(n, sizes, most))
    held = [
        tuple(row) for task in tasks for block in task for row in block_positions(n, block).tolist()
    ]
    assert held == [c for size in sizes for c in combinations(range(n), size)]
    assert all(sum(block_subsets(n, block) for block in task) <= most for task in tasks)


@pytest.mark.parametrize("seed", range(24))
def test_an_array_search_asks_the_tuple_search_s_subsets_in_its_order(seed: int) -> None:
    """The subsets asked, the one found, and the queries answered before it are the tuple
    search's, with the adjacencies and a tested set left out and under a ``max_depth``."""
    rng = np.random.default_rng(seed)
    dsep = rng.permutation(np.arange(2, 14)).tolist()[: rng.integers(1, 10)]
    adj_a, adj_b, tested = (set(rng.choice(dsep, rng.integers(0, len(dsep) + 1))) for _ in "abc")
    tested = tested if seed % 2 else set()
    max_depth = None if seed % 3 else int(rng.integers(1, len(dsep) + 1))
    top = fci._largest_size(len(dsep), max_depth)
    choices = list(fci._iter_dsep_choices(dsep, adj_a, adj_b, max_depth, tested))
    masks = [sum(1 << i for i, v in enumerate(dsep) if v in s) for s in (adj_a, adj_b, tested)]
    most = int(rng.integers(1, 6))

    def search(test: _Scripted) -> tuple[int, ...] | None:
        order = (block for size in range(1, top + 1) for block in blocks(len(dsep), size, most))
        return first_separating(test, 0, 1, dsep, masks, order, 0.05)

    every = _Scripted()
    assert search(every) is None
    assert every.asked == [tuple(sorted(c)) for c in choices]
    if choices:
        at = int(rng.integers(len(choices)))
        found = _Scripted(tuple(sorted(choices[at])))
        assert search(found) == choices[at]
        assert found.asked == every.asked[: at + 1]


def _before_the_step(draw: dict, ranked: bool):
    x = sample_scm(graph="erdos_renyi", scale="standardize", **draw).data
    test = make_indep_test("fisherz", x)
    support, sepsets, _ = adjacency_search(test, x.shape[1], 0.05)
    marks = fci.circle_pag(support)
    fci.orient_colliders(marks, sepsets)
    by_rank, step_test = fci._test_in_rank_order(x, "fisherz", None) if ranked else (None, test)
    return marks, sepsets, step_test, by_rank


@pytest.mark.parametrize(
    ("most", "max_depth", "ranked"), [(7, None, False), (7, None, True), (1, 2, True)]
)
@pytest.mark.parametrize("draw", DRAWS)
def test_a_pooled_step_removes_the_serial_pairs(
    pools: list[int],
    monkeypatch: pytest.MonkeyPatch,
    draw: dict,
    most: int,
    max_depth: int | None,
    ranked: bool,
) -> None:
    marks, sepsets, test, by_rank = _before_the_step(draw, ranked)
    serial, serial_sepsets = marks.copy(), dict(sepsets)
    fci._remove_by_possible_dsep(serial, test, 0.05, serial_sepsets, by_rank, max_depth=max_depth)
    monkeypatch.setenv(GATE, "0")
    monkeypatch.setattr(_parallel_skeleton, "DSEP_TASK_SUBSETS", most)
    pooled, pooled_sepsets = marks.copy(), dict(sepsets)
    fci._remove_by_possible_dsep(
        pooled, test, 0.05, pooled_sepsets, by_rank, max_depth=max_depth, workers=3
    )

    assert np.count_nonzero(serial) < np.count_nonzero(marks), "the draw must remove edges"
    np.testing.assert_array_equal(pooled, serial)
    assert list(pooled_sepsets.items()) == list(serial_sepsets.items())
    assert pools == [3]


def test_a_search_below_the_cutoff_runs_in_process(pools: list[int]) -> None:
    marks, sepsets, test, by_rank = _before_the_step(DRAWS[1], True)
    serial, serial_sepsets = marks.copy(), dict(sepsets)
    fci._remove_by_possible_dsep(serial, test, 0.05, serial_sepsets, by_rank)
    gated, gated_sepsets = marks.copy(), dict(sepsets)
    fci._remove_by_possible_dsep(gated, test, 0.05, gated_sepsets, by_rank, workers=3)
    assert pools == []
    np.testing.assert_array_equal(gated, serial)
    assert list(gated_sepsets.items()) == list(serial_sepsets.items())


@pytest.mark.parametrize(("max_depth", "subsets"), [(None, 63), (4, 56)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_a_search_reaching_the_cutoff_runs_on_the_pool(
    monkeypatch: pytest.MonkeyPatch, max_depth: int | None, subsets: int, offset: int
) -> None:
    """Six candidates hold 63 subsets over sizes one to six, and 56 up to size four."""
    monkeypatch.setenv(GATE, str(subsets + offset))
    searched = []
    monkeypatch.setattr(
        _parallel_skeleton.SkeletonWorkerPool,
        "first_separating",
        lambda self, x, y, candidates, within, sizes, alpha: searched.append(sizes),
    )
    x = np.random.default_rng(0).standard_normal((300, 8))
    test = make_indep_test("fisherz", x)
    marks = np.full((8, 8), NULL, dtype=np.int8)
    marks[0, 1] = marks[1, 0] = CIRCLE
    pool = _parallel_skeleton.SkeletonWorkerPool(test, 2)
    dsep = [7, 6, 5, 4, 3, 2]
    fci._try_remove_by_dsep(marks, 0, 1, dsep, set(), set(), test, 0.05, {}, max_depth, pool)
    assert searched == ([range(1, 7 if max_depth is None else 5)] if offset <= 0 else [])


@pytest.mark.parametrize("indep_test", ["fisherz", "student_t"])
def test_a_singular_test_raises_the_serial_error_on_every_path(
    pools: list[int], monkeypatch: pytest.MonkeyPatch, indep_test: str
) -> None:
    x = np.random.default_rng(1).standard_normal((400, 6))
    x[:, 1] += x[:, 0]  # no subset separates 0 and 1
    x[:, 4] = x[:, 2] + x[:, 3]  # so the search reaches the singular subset (4, 3, 2)
    test = make_indep_test(indep_test, x)
    monkeypatch.setenv(GATE, "0")
    errors = []
    for least, workers in ((10**9, 1), (0, 1), (0, 2)):  # tuples, arrays, the pool
        monkeypatch.setattr(fci, "_ARRAY_MIN_SUBSETS", least)
        pool = _parallel_skeleton.SkeletonWorkerPool(test, 2) if workers > 1 else None
        marks = np.full((6, 6), NULL, dtype=np.int8)
        marks[0, 1] = marks[1, 0] = CIRCLE
        try:
            with pytest.raises(ValueError) as error:
                fci._try_remove_by_dsep(
                    marks, 0, 1, [5, 4, 3, 2], set(), set(), test, 0.05, {}, pool=pool
                )
        finally:
            if pool is not None:
                pool.close()
        errors.append((type(error.value), str(error.value)))
    assert errors[1] == errors[0]
    assert errors[2] == errors[0]
    assert pools == [2]


# ---- whole fits on the recovery tests' pooled two-domain data -----------------------------------

FITS = {
    "fci": lambda x, index: andrey.fci(x),
    "fci_jci": lambda x, index: andrey.fci_jci(x, index),
    "gfci": lambda x, index: andrey.gfci(x, env=index),
}


def _pooled_data(seed: int = 4, d: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Two-domain data; at the defaults its possible-d-sep sets reach 9 to 11 candidates."""
    return sample_pooled(random_dag(seed, d, d // 10), d, 2000, seed)


def _fitted(monkeypatch: pytest.MonkeyPatch, method: str, x, index) -> tuple:
    """The PAG, its metadata, and the separating sets the possible-d-sep step leaves."""
    left: list = []
    real = fci._remove_by_possible_dsep

    def step(marks, test, alpha, sepsets, *args, **kwargs):
        real(marks, test, alpha, sepsets, *args, **kwargs)
        left.append(list(sepsets.items()))

    monkeypatch.setattr(fci, "_remove_by_possible_dsep", step)
    out = FITS[method](x, index)
    monkeypatch.setattr(fci, "_remove_by_possible_dsep", real)
    return out.structure.to_numpy().tolist(), repr(out.metadata), left


@pytest.mark.parametrize("method", sorted(FITS))
def test_the_pag_is_the_same_on_every_path_and_worker_count(
    pools: list[int], pooled_searches: list[int], monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    x, index = _pooled_data()
    monkeypatch.setattr(fci, "_ARRAY_MIN_SUBSETS", 10**9)
    tuples = _fitted(monkeypatch, method, x, index)
    monkeypatch.setattr(fci, "_ARRAY_MIN_SUBSETS", 0)
    searched = []
    real = fci.first_separating_subset
    monkeypatch.setattr(fci, "first_separating_subset", lambda *a: searched.append(a) or real(*a))
    arrays = _fitted(monkeypatch, method, x, index)
    assert searched, "the fit must search as arrays"
    assert arrays == tuples
    monkeypatch.setenv(GATE, "256")  # the searches of nine candidates or more
    for workers in (2, 4):
        with andrey.config(n_jobs=workers):
            assert _fitted(monkeypatch, method, x, index) == tuples
    assert len(pooled_searches) >= 2 and min(pooled_searches) >= 9
    assert set(pools) == {2, 4}


@pytest.mark.parametrize("least", [0, 10**9], ids=["arrays", "tuples"])
@pytest.mark.parametrize(("method", "seed", "d"), [("fci", 4, 30), ("fci_jci", 2, 20)])
def test_skipping_the_tested_subsets_never_changes_the_pag(
    monkeypatch: pytest.MonkeyPatch, method: str, seed: int, d: int, least: int
) -> None:
    """GFCI runs the same step; on these draws its second searches skip nothing."""
    monkeypatch.setattr(fci, "_ARRAY_MIN_SUBSETS", least)
    x, index = _pooled_data(seed, d)
    answered = [0]
    prefix = PartialCorrelationTest.separating_prefix

    def counted(self, *args):
        pvalues = prefix(self, *args)
        answered[0] += len(pvalues)
        return pvalues

    monkeypatch.setattr(PartialCorrelationTest, "separating_prefix", counted)
    skipping = _fitted(monkeypatch, method, x, index), answered[0]
    answered[0] = 0
    real = fci._try_remove_by_dsep
    monkeypatch.setattr(fci, "_try_remove_by_dsep", lambda *args: real(*args[:-1], ()))
    every = _fitted(monkeypatch, method, x, index), answered[0]
    assert skipping[0] == every[0]
    assert skipping[1] < every[1], "the draw must skip some subsets"
