"""CAM's contract: knowledge, units, the neighbor rule, and the columns it refuses."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import andrey
from andrey.fcm.cam import boosting_counts, neighbors, order_search
from andrey.fcm.gam import BasisCache
from andrey.search.hc import tie_ranks

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.cam(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, _ in out.structure.oriented_edges()}


def _chain(seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with nonlinear effects and Gaussian noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, N)
    x1 = np.sin(2 * x0) + x0 + 0.3 * rng.standard_normal(N)
    x2 = np.tanh(2 * x1) + 0.3 * rng.standard_normal(N)
    return np.column_stack([x0, x1, x2])


@pytest.fixture(scope="module")
def chain_fit():
    return _fit(_chain())


def test_the_chain_is_found(chain_fit):
    assert _edges(chain_fit) == {(0, 1), (1, 2)}
    assert chain_fit.ordering == (0, 1, 2)


def test_rescaling_a_column_changes_neither_the_graph_nor_the_order(chain_fit):
    out = _fit(_chain() * np.array([1e-4, 3.7, 2e3]))
    assert out.structure == chain_fit.structure
    assert out.ordering == chain_fit.ordering


def _mirrored(seed: int = 0) -> np.ndarray:
    """Linear Gaussian ``x``, ``y`` with each row's swap also a row, and independent ``z``.

    Swapping ``x`` and ``y`` only reorders the rows, so ``x -> y`` and ``y -> x`` have the same
    gain up to rounding; at this seed the two differ in the last bit.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N // 2)
    b = 0.8 * a + 0.6 * rng.standard_normal(N // 2)
    x, y = np.concatenate([a, b]), np.concatenate([b, a])
    return np.column_stack([x, y, rng.standard_normal(N)])


def _named(out: andrey.StructureOutput) -> set[tuple[str, str]]:
    return {(str(i), str(j)) for i, j, _ in out.structure.oriented_edges()}


def test_tied_gains_go_to_the_earlier_column():
    X = _mirrored()[:, :2]
    candidates = ~np.eye(2, dtype=bool)
    for columns in (X, X[:, ::-1]):
        assert np.argwhere(order_search(BasisCache(columns), candidates)).tolist() == [[0, 1]]


def test_a_tie_follows_the_rank_and_reordered_columns_relabel_the_result():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_mirrored(), columns=list("xyz"))
    found = {frozenset(_named(_fit(frame[list(order)]))) for order in itertools.permutations("xyz")}
    assert len(found) == 1
    ranks = dict(zip("xyz", tie_ranks(frame.to_numpy()).tolist(), strict=True))
    cause, effect = sorted("xy", key=ranks.get)
    assert found == {frozenset({(cause, effect)})}
    with andrey.config(tie_break="position"):
        assert _named(_fit(frame[["y", "x", "z"]])) == {("y", "x")}


def test_a_forbidden_edge_is_never_in_the_graph():
    out = _fit(_chain(), knowledge=andrey.Knowledge(forbidden=[(0, 1)]))
    assert (0, 1) not in _edges(out)


def test_a_required_edge_stays_on_independent_columns():
    X = np.random.default_rng(1).standard_normal((N, 3))
    out = _fit(X, knowledge=andrey.Knowledge(required=[(2, 0)]))
    assert _edges(out) == {(2, 0)}
    assert out.ordering.index(2) < out.ordering.index(0)


def test_tiers_order_the_variables_and_forbid_edges_into_an_earlier_tier(chain_fit):
    # x2 first: no edge may enter it, and no order may place x0 or x1 before it.
    out = _fit(_chain(), knowledge=andrey.Knowledge(tiers=[[2], [0, 1]]))
    assert all(j != 2 for _, j in _edges(out))
    assert out.ordering[0] == 2


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
@pytest.mark.parametrize("seed", [1, 2])
def test_every_tier_precedes_the_next_in_the_order_of_many_columns(seed):
    # Few possible parents per variable leave most pairs unconnected by the search.
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, 15))
    X[:, 1] = np.sin(2 * X[:, 0]) + 0.3 * rng.standard_normal(300)
    out = _fit(X, knowledge=andrey.Knowledge(tiers=[[14], list(range(14))]))
    assert out.ordering[0] == 14
    assert all(j != 14 for _, j in _edges(out))


def test_required_edges_above_max_parents_raise():
    rng = np.random.default_rng(7)
    X = rng.uniform(-2, 2, (N, 3))
    with pytest.raises(ValueError):
        _fit(X, max_parents=1, knowledge=andrey.Knowledge(required=[(0, 2), (1, 2)]))


def test_required_parents_with_more_coefficients_than_rows_name_the_column():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        np.random.default_rng(8).standard_normal((14, 8)), columns=list("abcdefgh")
    )
    required = [(c, "h") for c in "abcdefg"]
    with pytest.raises(ValueError, match="'h'"):
        _fit(frame, knowledge=andrey.Knowledge(required=required))


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_variable_with_few_distinct_values_is_a_parent():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 5, N).astype(float)
    y = (x - 2) ** 2 + 0.3 * rng.standard_normal(N)
    assert _edges(_fit(np.column_stack([x, y]))) == {(0, 1)}


def test_fewer_rows_than_a_full_model_needs_still_give_a_dag():
    out = _fit(np.random.default_rng(3).standard_normal((12, 8)))
    assert out.structure.kind == "dag" and sorted(out.ordering) == list(range(8))


def test_boosting_picks_the_variable_the_response_depends_on():
    rng = np.random.default_rng(4)
    X = rng.uniform(-2, 2, (N, 3))
    X[:, 2] = np.sin(2 * X[:, 0]) + 0.2 * rng.standard_normal(N)
    counts = boosting_counts(X)
    assert counts[:, 2].sum() == 100 and counts[2, 2] == 0
    assert counts[0, 2] > 90


def test_possible_parents_are_picked_three_times_and_a_tie_at_the_tenth_keeps_fewer():
    counts = np.zeros((14, 3), dtype=np.int64)
    counts[:, 0] = [2, 3, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Eleven picked at least three times: the ten most picked stay.
    counts[:, 1] = [2, 3, 30, 9, 9, 8, 8, 7, 7, 6, 5, 5, 1, 0]
    # Twelve tied at the tenth place: none is picked more often than the eleventh.
    counts[:12, 2] = 8
    out = neighbors(counts)
    assert np.flatnonzero(out[:, 0]).tolist() == [1, 2]
    assert np.flatnonzero(out[:, 1]).tolist() == list(range(2, 12))
    assert not out[:, 2].any()


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_columns_without_a_smooth_basis_are_named():
    X = _chain()
    X[:, 1] = np.arange(N) % 2
    with pytest.raises(ValueError):
        _fit(X)
