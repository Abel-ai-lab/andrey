"""SCORE's contract: the leaf search, ties, knowledge, units, and the data it refuses."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core.ties import tie_ranks
from andrey.fcm.score_matching import leaf_order

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.score_matching(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, _ in out.structure.oriented_edges()}


def _named(out: andrey.StructureOutput) -> set[tuple[str, str]]:
    return {(str(i), str(j)) for i, j, _ in out.structure.oriented_edges()}


def _chain(seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with ``x1 = sin(2 x0)`` and ``x2 = x1**2``, each plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(N)
    x1 = np.sin(2 * x0) + 0.5 * rng.standard_normal(N)
    return np.column_stack([x0, x1, x1**2 + 0.5 * rng.standard_normal(N)])


@pytest.fixture(scope="module")
def chain_fit():
    return _fit(_chain())


def test_the_chain_is_found(chain_fit):
    assert _edges(chain_fit) == {(0, 1), (1, 2)}
    assert chain_fit.ordering == (0, 1, 2)
    with andrey.config(tie_break="position"):
        assert _edges(_fit(_chain())) == {(0, 1), (1, 2)}


def test_alpha_sets_which_parents_stay_and_not_the_order(chain_fit):
    """A parent stays when its smooth term's p-value is at most ``alpha``, so a larger level keeps
    every predecessor of the order; the order does not read it."""
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


def test_the_smallest_value_is_the_next_leaf_and_a_tie_goes_to_the_earlier_column():
    unit = np.zeros((3, 4))
    values = {4: [0.3, 0.1, 0.2, 0.1 + 1e-12], 3: [0.2, 0.2, 0.2], 2: [0.5, 0.4]}
    order = leaf_order(unit, lambda remaining: np.array(values[remaining.shape[1]]))
    # Columns 1 and 3 tie within the rounding, so 1 leaves first; of 0, 2, 3 all tie, so 0 leaves;
    # of 2 and 3 the smaller value, 3's, leaves.
    assert order.tolist() == [2, 3, 0, 1]


def test_a_variable_that_must_precede_another_waits_for_it_to_leave():
    unit = np.zeros((3, 3))
    precedence = np.zeros((3, 3), dtype=bool)
    precedence[0, 2] = True
    order = leaf_order(unit, lambda remaining: np.arange(remaining.shape[1]), precedence)
    assert order.tolist() == [0, 2, 1]


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
    X = np.random.default_rng(5).standard_normal((300, 6))
    ranks = tie_ranks(X)
    first, last = int(np.argmax(ranks)), int(np.argmin(ranks))
    middle = [j for j in range(6) if j not in (first, last)]
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
    assert "score_matching" in str(error.value)


def test_twice_as_many_rows_as_columns_fit_and_one_fewer_is_refused():
    X = np.random.default_rng(4).standard_normal((16, 8))
    assert _fit(X).structure.n_nodes == 8
    with pytest.raises(ValueError):
        _fit(X[:15])


def test_rows_that_mostly_repeat_are_refused_by_name():
    """With three quarters of the rows equal, more than half of the row pairs are at distance 0,
    which leaves the kernel no bandwidth."""
    X = np.random.default_rng(6).standard_normal((40, 2))
    X[:30] = 0.0
    with pytest.raises(ValueError, match="score_matching: the median distance"):
        _fit(X)
