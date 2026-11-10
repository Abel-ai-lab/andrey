"""Missing-Value PC: ``pc(missing="mvpc")``, its arguments, and the tests of its correction.

A corrected test reads the regressions of a block and permutations of the rows where its
indicators' parents are observed; the checks below compare it with the test-wise deletion test it
falls back to, build it by hand on a small block, and check what its draws depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.constraint.mvpc import IndicatorTest, PermutationCorrection, candidate_edges
from andrey.core.ci import FisherZ, make_indep_test
from andrey.data import missingness
from tests.recovery.test_mvpc import figure_1b


def _labeled(out) -> dict:
    marks, labels = out.structure.to_numpy(), out.structure.labels
    return {(a, b): int(marks[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}


def test_on_complete_data_mvpc_gives_the_test_wise_result():
    X = np.nan_to_num(figure_1b(0), nan=0.0)
    out, deletion = andrey.pc(X, missing="mvpc"), andrey.pc(X)
    assert out.structure == deletion.structure
    assert dict(out.metadata) == dict(deletion.metadata)


@pytest.mark.parametrize("indep_test", ["chisq", "kci", "cond_gauss"])
def test_mvpc_takes_only_the_partial_correlation_tests(indep_test):
    with pytest.raises(ValueError, match='^pc: missing="mvpc" corrects the partial-correlation'):
        andrey.pc(figure_1b(0), missing="mvpc", indep_test=indep_test)


@pytest.mark.parametrize("name", ["fci", "cdnod", "mmpc"])
def test_only_pc_takes_mvpc(name):
    X = figure_1b(0)
    args = (X, np.repeat([0.0, 1.0], len(X) // 2)) if name == "cdnod" else (X,)
    with pytest.raises(ValueError, match=rf"^{name}: missing must be one of 'testwise'"):
        getattr(andrey, name)(*args, missing="mvpc")


def test_a_fit_repeats_records_its_seed_and_runs_with_student_t():
    first = andrey.pc(figure_1b(1), missing="mvpc", seed=3)
    assert first == andrey.pc(figure_1b(1), missing="mvpc", seed=3)
    assert first.metadata["seed"] == 3
    student = andrey.pc(figure_1b(1), missing="mvpc", indep_test="student_t")
    assert student.metadata["corrected_edges"]["removed"] == [[0, 1]]


def test_column_order_units_and_origin_change_no_result():
    frame = pd.DataFrame(figure_1b(2), columns=["x", "y", "w"])
    want = andrey.pc(frame, missing="mvpc")
    assert want.metadata["missingness_parents"] == {"y": ["w"]}
    moved = frame[["w", "x", "y"]].assign(w=lambda f: 1000.0 * f["w"] - 7.0, y=lambda f: f["y"] + 5)
    out = andrey.pc(moved, missing="mvpc")
    assert _labeled(out) == _labeled(want)
    assert out.metadata["missingness_parents"] == want.metadata["missingness_parents"]
    assert out.metadata["corrected_edges"]["removed"] == [["x", "y"]]


def test_candidates_share_an_adjacent_variable_or_an_indicator_they_cause():
    chain = np.zeros((4, 4), dtype=bool)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        chain[a, b] = chain[b, a] = True
    assert not candidate_edges(chain, {}).any()
    shared = candidate_edges(chain, {3: [0, 1]})  # 0 and 1 both cause the indicator of 3
    assert shared[0, 1] and shared[1, 0] and shared.sum() == 2
    triangle = chain.copy()
    triangle[0, 2] = triangle[2, 0] = True
    assert set(zip(*np.nonzero(np.triu(candidate_edges(triangle, {}))))) == {(0, 1), (0, 2), (1, 2)}


def _correction(X, parents, candidates=None):
    d = X.shape[1]
    if candidates is None:
        candidates = ~np.eye(d, dtype=bool)
    return PermutationCorrection(
        X, "fisherz", parents, candidates, make_indep_test("fisherz", X), 0
    )


def test_the_parents_of_a_query_are_closed_under_their_own_indicators():
    test = _correction(figure_1b(0), {1: [2], 2: [3], 3: [4]})
    assert test.closure((0, 1)) == [2, 3, 4]
    assert test.closure((0,)) == []


def test_off_the_candidates_or_with_no_parents_a_query_is_the_deletion_test():
    X = figure_1b(0)
    deletion = make_indep_test("fisherz", X)
    off = np.zeros((3, 3), dtype=bool)
    assert _correction(X, {1: [2]}, off)(0, 1) == deletion(0, 1)
    assert _correction(X, {})(0, 1, (2,)) == deletion(0, 1, (2,))


def test_a_corrected_test_is_the_test_on_the_virtual_rows_it_builds():
    X = figure_1b(0)
    test = _correction(X, {1: [2]})
    rows = ~np.isnan(X).any(axis=1)
    design = np.column_stack([np.ones(rows.sum()), X[rows, 2]])
    coefficients = np.linalg.lstsq(design, X[rows][:, [0, 1]], rcond=None)[0]
    residuals = X[rows][:, [0, 1]] - design @ coefficients
    drawn = test._generator(0, 1, []).permutation(np.arange(len(X)))[: rows.sum()]
    virtual = np.column_stack([np.ones(rows.sum()), X[drawn, 2]]) @ coefficients + residuals
    assert test(1, 0) == FisherZ(virtual)(0, 1)


def test_a_corrected_test_with_too_few_rows_is_the_deletion_test():
    X = figure_1b(0)[:8]
    X[:, 1] = [0.1, np.nan, 0.3, np.nan, np.nan, 0.2, np.nan, 0.5]
    X[[5, 7], 2] = np.nan  # columns 0, 1, and their indicator's parent 2 share two rows
    test = _correction(X, {1: [2]})
    assert test(0, 1) == make_indep_test("fisherz", X)(0, 1)
    assert test.uncorrected == {(0, 1)}
    assert test(0, 2) == make_indep_test("fisherz", X)(0, 2)  # no indicator parent: not counted
    assert test.uncorrected == {(0, 1)}


def _blocked(seed: int, n: int = 450) -> np.ndarray:
    """A triangle whose three columns are observed together on two rows, each pair on a third."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = x + 0.5 * rng.standard_normal(n)
    w = x + y + 0.5 * rng.standard_normal(n)
    X = np.column_stack([x, y, w])
    third = n // 3
    X[:third, 2] = np.nan
    low = np.zeros(n, dtype=bool)
    low[third:] = w[third:] < np.median(w[third:])
    X[low, 1] = np.nan  # y is missing where w is low: w is a parent of its indicator
    high = np.flatnonzero(~low & (np.arange(n) >= third))
    X[high[2:], 0] = np.nan
    return X


@pytest.mark.parametrize("seed", range(3))
def test_a_fit_whose_corrected_tests_lack_rows_returns_the_test_wise_edges_and_names_them(seed):
    X = _blocked(seed)
    out = andrey.pc(X, missing="mvpc", max_depth=0)
    assert out.metadata["missingness_parents"]
    assert out.metadata["corrected_edges"] == {
        "retested": [[0, 1], [0, 2], [1, 2]],
        "removed": [],
        "uncorrected": [[0, 1], [0, 2], [1, 2]],
    }
    testwise = andrey.pc(X, max_depth=0)
    assert np.array_equal(out.structure.to_numpy(), testwise.structure.to_numpy())


def test_an_indicator_constant_on_a_query_rows_separates_the_pair():
    X = np.random.default_rng(0).standard_normal((50, 3))
    X[:5, 0] = np.nan  # missing only where column 1 is missing too
    X[:5, 1] = np.nan
    R = X.copy()
    R[:, 0] = np.isnan(X[:, 0])
    test = IndicatorTest(make_indep_test("fisherz", R), R, 0)
    assert test(0, 1) == 1.0
    assert test(0, 2) < 1.0


def test_an_mcar_mask_gives_the_indicators_few_parents():
    X = np.random.default_rng(1).standard_normal((2000, 5))
    X[:, 1] += X[:, 0]
    X[:, 2] += X[:, 1]
    mask = missingness.draw(X, missingness.mcar(0.2, observed=0.4), seed=1)
    out = andrey.pc(np.where(mask.mask, np.nan, X), missing="mvpc")
    assert sum(map(len, out.metadata["missingness_parents"].values())) <= 1
    assert out.structure == andrey.pc(np.where(mask.mask, np.nan, X)).structure
