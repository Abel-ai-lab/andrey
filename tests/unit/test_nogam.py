"""NoGAM's contract: its regressions, knowledge, units, and the data it refuses."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core.ties import tie_ranks
from andrey.fcm import nogam as core

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.nogam(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, _ in out.structure.oriented_edges()}


def _named(out: andrey.StructureOutput) -> set[tuple[str, str]]:
    return {(str(i), str(j)) for i, j, _ in out.structure.oriented_edges()}


def _chain(seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with ``x1 = sin(2 x0)`` and ``x2 = x1**2``, each plus uniform noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, N)
    x1 = np.sin(2 * x0) + rng.uniform(-0.5, 0.5, N)
    return np.column_stack([x0, x1, x1**2 + rng.uniform(-0.5, 0.5, N)])


@pytest.fixture(scope="module")
def chain_fit():
    return _fit(_chain())


def test_the_chain_is_found(chain_fit):
    assert _edges(chain_fit) == {(0, 1), (1, 2)}
    assert chain_fit.ordering == (0, 1, 2)
    with andrey.config(tie_break="position"):
        assert _edges(_fit(_chain())) == {(0, 1), (1, 2)}


def test_a_constant_kernel_predicts_the_shrunk_training_sum():
    """With the kernel 1 between every pair of rows, the ridge solution has a closed form: each
    row is predicted by the sum of the other folds' responses over their count plus the ridge
    term."""
    y = np.arange(12.0)
    predicted = core.cross_predict(np.ones((12, 12)), y)
    folds = np.array_split(np.arange(12), core.FOLDS)
    assert [len(fold) for fold in folds] == [3, 3, 2, 2, 2]
    for fold in folds:
        expected = (y.sum() - y[fold].sum()) / (12 - len(fold) + core.RIDGE)
        np.testing.assert_allclose(predicted[fold], expected, rtol=1e-10)


def test_each_row_is_predicted_without_its_own_fold():
    """A response that is noise has nothing to predict: the predictions made without a row's fold
    do not follow it, where a fit on every row would."""
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, 200))
    kernel = np.exp(-core.GAMMA * (x[:, None] - x[None, :]) ** 2)
    assert abs(np.corrcoef(core.cross_predict(kernel, y), y)[0, 1]) < 0.2


def test_the_residual_of_an_effect_is_its_noise_and_a_cause_keeps_its_spread():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N)
    noise = 0.5 * rng.standard_normal(N)
    unit = np.column_stack([x, np.sin(2 * x) + noise])
    unit = (unit - unit.mean(axis=0)) / unit.std(axis=0)
    residual = core.residuals(unit)
    assert np.corrcoef(residual[:, 1], noise)[0, 1] > 0.9
    errors = core.prediction_errors(unit)
    assert errors[1] < errors[0]


def test_alpha_sets_which_parents_stay_and_not_the_order(chain_fit):
    loose = _fit(_chain(), alpha=0.999999)
    assert _edges(chain_fit) < _edges(loose) == {(0, 1), (0, 2), (1, 2)}
    assert loose.ordering == chain_fit.ordering


def test_the_result_follows_the_columns_in_any_order_and_units(chain_fit):
    """Reordered, rescaled, and shifted columns give the same graph and order over the
    variables."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_chain(), columns=list("abc"))
    moved = frame[["c", "a", "b"]] * [1e-3, 250.0, 7.0] + [5.0, -40.0, 1e3]
    out = _fit(moved)
    assert _named(out) == {("a", "b"), ("b", "c")}
    assert [moved.columns[j] for j in out.ordering] == ["a", "b", "c"]


def test_a_required_edge_stays_on_independent_columns():
    X = np.random.default_rng(1).standard_normal((N, 3))
    out = _fit(X, knowledge=andrey.Knowledge(required=[(2, 0)]))
    assert _edges(out) == {(2, 0)}
    assert out.ordering.index(2) < out.ordering.index(0)


def test_a_forbidden_edge_is_never_returned(chain_fit):
    out = _fit(_chain(), knowledge=andrey.Knowledge(forbidden=[(0, 1)]))
    assert (0, 1) not in _edges(out)
    assert out.ordering == chain_fit.ordering


def test_tiers_order_the_variables_and_forbid_edges_into_an_earlier_tier():
    # The highest-ranked column goes first and the lowest-ranked last, against the rank's own pull.
    X = np.random.default_rng(5).standard_normal((200, 5))
    ranks = tie_ranks(X)
    first, last = int(np.argmax(ranks)), int(np.argmin(ranks))
    middle = [j for j in range(5) if j not in (first, last)]
    out = _fit(X, knowledge=andrey.Knowledge(tiers=[[first], middle, [last]]))
    assert out.ordering[0] == first
    assert out.ordering[-1] == last
    assert all(j != first and i != last for i, j in _edges(out))


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_binary_column_is_refused_by_name():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_chain(), columns=["a", "flag", "c"])
    frame["flag"] = np.arange(N) % 2
    with pytest.raises(ValueError) as error:
        _fit(frame)
    assert "flag" in str(error.value)
    assert "nogam" in str(error.value)


def test_twice_as_many_rows_as_columns_fit_and_one_fewer_is_refused():
    X = np.random.default_rng(4).standard_normal((16, 8))
    assert _fit(X).structure.n_nodes == 8
    with pytest.raises(ValueError):
        _fit(X[:15])


def test_fewer_rows_than_folds_are_refused():
    X = np.random.default_rng(4).standard_normal((5, 2))
    assert _fit(X).structure.n_nodes == 2
    with pytest.raises(ValueError, match="at least 5 rows"):
        _fit(X[:4])


def test_rows_that_mostly_repeat_are_refused_by_name():
    X = np.random.default_rng(6).standard_normal((40, 2))
    X[:30] = 0.0
    with pytest.raises(ValueError, match="nogam: the median distance"):
        _fit(X)
