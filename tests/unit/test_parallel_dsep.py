"""FCI's possible-d-sep searches on the skeleton worker pool: the order, the cutoff, the errors.

The pairs are visited one at a time on any worker count. With more than one worker and a FisherZ,
StudentT, or NormalScores test, a pair's search sends its larger conditioning-set sizes to the pool,
which tests the subsets in the serial order and returns the first that separates the pair. The
removals, their order, and the separating sets are then the serial ones. Most tests set the cutoff
to 0, so a small case reaches the pool, and every one spies on the pool's creation, so a pooled
comparison cannot silently run serial against serial.
"""

from __future__ import annotations

import importlib
from itertools import combinations

import numpy as np
import pytest

import andrey
from andrey.constraint import _parallel_skeleton
from andrey.core import CIRCLE, NULL
from andrey.core.ci import make_indep_test
from andrey.data import sample_scm

fci = importlib.import_module("andrey.constraint.fci")

# Draws whose possible-d-sep step removes edges (test_fci_possible_dsep).
DRAWS = [
    {"d": 10, "n": 500, "density": 4.0, "latents": 1, "seed": 9},
    {"d": 15, "n": 500, "density": 4.0, "latents": 2, "seed": 7},
    {"d": 15, "n": 1000, "density": 3.0, "latents": 1, "seed": 3},
]


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture
def pools(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The worker count of each skeleton pool created during the test."""
    created: list[int] = []
    real = _parallel_skeleton.ProcessPoolExecutor

    def spy(*args, **kwargs):
        created.append(kwargs["max_workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(_parallel_skeleton, "ProcessPoolExecutor", spy)
    return created


@pytest.fixture
def open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every possible-d-sep search, and every ``|S| >= 2`` skeleton pass, to the pool."""
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", "0")


@pytest.mark.parametrize("most", [1, 2, 5, 40])
@pytest.mark.parametrize("n", range(8))
def test_tasks_hold_the_subsets_in_combinations_order(n: int, most: int) -> None:
    sizes = range(1, n + 1)
    tasks = list(_parallel_skeleton._dsep_tasks(n, sizes, most))
    held = [
        subset
        for task in tasks
        for block in task
        for subset in _parallel_skeleton._subsets(range(n), block, ())
    ]
    assert held == [c for size in sizes for c in combinations(range(n), size)]
    assert all(
        sum(_parallel_skeleton._block_subsets(n, block) for block in task) <= most for task in tasks
    )


@pytest.mark.parametrize("most", [1, 3, 64])
def test_tasks_leave_out_the_subsets_inside_an_adjacency(most: int) -> None:
    dsep = [9, 7, 5, 4, 2, 1]
    adj_a, adj_b = {9, 7, 4}, {5, 1}
    tasks = _parallel_skeleton._dsep_tasks(len(dsep), range(1, len(dsep) + 1), most)
    held = [
        subset
        for task in tasks
        for block in task
        for subset in _parallel_skeleton._subsets(dsep, block, (frozenset(adj_a), frozenset(adj_b)))
    ]
    assert held == list(fci._iter_dsep_choices(dsep, adj_a, adj_b))


def _before_the_step(draw: dict, ranked: bool):
    x = sample_scm(graph="erdos_renyi", scale="standardize", **draw).data
    test = make_indep_test("fisherz", x)
    support, sepsets = fci._fast_adjacency_search(x, test, 0.05)
    marks = fci._circle_pag(support)
    fci._orient_colliders(marks, sepsets)
    by_rank, step_test = fci._test_in_rank_order(x, "fisherz", None) if ranked else (None, test)
    return marks, sepsets, step_test, by_rank


@pytest.mark.usefixtures("open_gate")
@pytest.mark.parametrize("most", [1, 7, _parallel_skeleton.DSEP_TASK_SUBSETS])
@pytest.mark.parametrize("ranked", [False, True], ids=["fast", "order_free"])
@pytest.mark.parametrize("draw", DRAWS)
def test_a_pooled_step_removes_the_serial_pairs(
    pools: list[int], monkeypatch: pytest.MonkeyPatch, draw: dict, ranked: bool, most: int
) -> None:
    marks, sepsets, test, by_rank = _before_the_step(draw, ranked)
    serial, serial_sepsets = marks.copy(), dict(sepsets)
    fci._remove_by_possible_dsep(serial, test, 0.05, serial_sepsets, by_rank)
    monkeypatch.setattr(_parallel_skeleton, "DSEP_TASK_SUBSETS", most)
    pooled, pooled_sepsets = marks.copy(), dict(sepsets)
    fci._remove_by_possible_dsep(pooled, test, 0.05, pooled_sepsets, by_rank, workers=3)

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


@pytest.mark.parametrize("cutoff", [0, 6, 7, 21, 22, 63, 64])
def test_the_sizes_from_the_cutoff_go_to_the_pool(
    pools: list[int], monkeypatch: pytest.MonkeyPatch, cutoff: int
) -> None:
    """Six candidates hold 6, 21, 41, 56, 62, and 63 subsets through sizes one to six."""
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", str(cutoff))
    pooled_sizes: list[range] = []
    real = _parallel_skeleton.SkeletonWorkerPool.first_separating

    def spy(self, x, y, candidates, skip, sizes, alpha):
        pooled_sizes.append(sizes)
        return real(self, x, y, candidates, skip, sizes, alpha)

    monkeypatch.setattr(_parallel_skeleton.SkeletonWorkerPool, "first_separating", spy)
    x = np.random.default_rng(0).standard_normal((300, 8))
    x[:, 1] += x[:, 0]  # no subset separates 0 and 1, so every size is tested
    test = make_indep_test("fisherz", x)
    marks = np.full((8, 8), NULL, dtype=np.int8)
    marks[0, 1] = marks[1, 0] = CIRCLE
    pool = _parallel_skeleton.SkeletonWorkerPool(test, 2)
    try:
        removed = fci._try_remove_by_dsep(
            marks, 0, 1, [7, 6, 5, 4, 3, 2], set(), set(), test, 0.05, {}, pool=pool
        )
    finally:
        pool.close()

    held = [6, 21, 41, 56, 62, 63]
    first = next((size for size in range(1, 7) if held[size - 1] >= cutoff), 7)
    assert not removed
    assert pooled_sizes == [range(first, 7)]
    assert pools == ([2] if first < 7 else [])


@pytest.mark.usefixtures("open_gate")
@pytest.mark.parametrize("indep_test", ["fisherz", "student_t"])
def test_a_singular_test_raises_the_serial_error(pools: list[int], indep_test: str) -> None:
    x = np.random.default_rng(1).standard_normal((400, 6))
    x[:, 1] += x[:, 0]  # no subset separates 0 and 1
    x[:, 4] = x[:, 2] + x[:, 3]  # so the search reaches the singular subset (4, 3, 2)
    test = make_indep_test(indep_test, x)
    errors = []
    for pool in (None, _parallel_skeleton.SkeletonWorkerPool(test, 2)):
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
        errors.append(error.value)
    assert type(errors[1]) is type(errors[0])
    assert str(errors[1]) == str(errors[0])
    assert pools == [2]


def _shifted(seed: int) -> np.ndarray:
    """Hidden-variable data pooled over two regimes, the second shifting three variables."""
    x = sample_scm(d=12, n=800, density=3.0, latents=1, seed=seed, scale="standardize").data
    x[len(x) // 2 :, :3] += 0.6
    return x


FITS = {"fci": andrey.fci, "gfci": andrey.gfci}


@pytest.mark.usefixtures("open_gate")
@pytest.mark.parametrize("tie_break", ["rank", "position"], ids=["order_free", "fast"])
@pytest.mark.parametrize("method", sorted(FITS))
def test_facades_give_the_serial_pag_on_the_configured_workers(
    pools: list[int], monkeypatch: pytest.MonkeyPatch, method: str, tie_break: str
) -> None:
    searches = []
    real = _parallel_skeleton.SkeletonWorkerPool.first_separating

    def spy(self, *args):
        searches.append(args)
        return real(self, *args)

    monkeypatch.setattr(_parallel_skeleton.SkeletonWorkerPool, "first_separating", spy)
    x = _shifted(4)
    with andrey.config(tie_break=tie_break):
        serial = FITS[method](x)
        assert searches == []
        with andrey.config(n_jobs=2):
            pooled = FITS[method](x)
    # (x, y, candidates, skip, sizes, alpha): some search sent subsets to the pool
    assert any(args[2] and len(args[4]) for args in searches)
    assert set(pools) == {2}
    np.testing.assert_array_equal(pooled.structure.to_numpy(), serial.structure.to_numpy())
    assert repr(pooled.metadata) == repr(serial.metadata)
