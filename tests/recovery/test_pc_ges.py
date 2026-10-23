from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint.pc import pc as pc_engine
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # The BIC/Fisher-Z primitives route through core.stats; pin the numpy oracle so an ambient
    # ANDREY_DEVICE=cpu cannot compare torch numerics against the numpy baselines.
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


_BASELINE_LAMBDA = 2.0


def test_pc_recovery():
    assert_recorded_output(andrey.pc(datagen.generate("gauss_5v")), "PC", case="gauss_5v")


def conflicting_colliders() -> np.ndarray:
    """A seed on which colliders conflict, so a higher-ranked collider makes PC skip another."""
    return data.sample_scm(d=20, n=200, seed=0, density=2.0, scale="standardize").data


# The recorded column-order output on that seed.
COLUMN_ORDER_EDGES = [
    (0, 8, "directed"),
    (0, 17, "undirected"),
    (1, 4, "undirected"),
    (1, 7, "undirected"),
    (6, 2, "directed"),
    (7, 2, "directed"),
    (12, 2, "directed"),
    (3, 4, "undirected"),
    (5, 18, "undirected"),
    (7, 19, "undirected"),
    (9, 8, "directed"),
    (10, 8, "directed"),
    (9, 11, "undirected"),
    (10, 12, "directed"),
    (11, 19, "undirected"),
    (16, 12, "directed"),
    (13, 18, "undirected"),
    (14, 15, "undirected"),
]


def test_pc_column_order_keeps_the_recorded_output():
    x = conflicting_colliders()
    assert andrey.pc(x, collider_order="column").structure.oriented_edges() == COLUMN_ORDER_EDGES
    assert andrey.pc(x).structure.oriented_edges() != COLUMN_ORDER_EDGES


def test_pc_collider_order_follows_tie_break_unless_given():
    x = conflicting_colliders()
    default = andrey.pc(x).structure.oriented_edges()
    with andrey.config(tie_break="position"):
        assert andrey.pc(x).structure.oriented_edges() == COLUMN_ORDER_EDGES
        assert andrey.pc(x, collider_order="p_value").structure.oriented_edges() == default
    with andrey.config(tie_break="rank"):
        assert andrey.pc(x).structure.oriented_edges() == default


def test_pc_is_invariant_to_column_order():
    x = conflicting_colliders()
    trace: list[dict] = []
    pc_engine(x, _trace=trace)
    assert any(
        r["step"] == "collider" and not r["oriented"] and r["triple"][1] not in r["sepset"]
        for r in trace
    )
    marks = andrey.pc(x).structure.to_numpy()
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = andrey.pc(x[:, order]).structure.to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)


def conflicting_orientations() -> np.ndarray:
    """A seed on which Meek's rule R1 directs one edge both ways."""
    return data.sample_scm(d=10, n=100, seed=9, density=4.0).data


def test_pc_orientations_are_invariant_to_column_order():
    x = conflicting_orientations()
    marks = andrey.pc(x).structure.to_numpy()
    in_order = pc_engine(x, in_column_order=True).to_numpy()
    in_order_changed = False
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = andrey.pc(x[:, order]).structure.to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)
        permuted = pc_engine(x[:, order], in_column_order=True).to_numpy()
        in_order_changed |= not np.array_equal(permuted[np.ix_(back, back)], in_order)
    assert in_order_changed


def test_ges_recovery():
    assert_recorded_output(
        andrey.ges(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA),
        "GES",
        case="gauss_5v",
    )


def test_ges_does_not_depend_on_the_data_scale():
    """GES returns the same CPDAG after the data is rescaled so far that squared covariances
    underflow."""
    x = datagen.generate("gauss_5v")
    want = andrey.ges(x).structure.to_numpy()
    np.testing.assert_array_equal(andrey.ges(x * 1e-85).structure.to_numpy(), want)


def fan_in(causes: int, n: int = 2000) -> np.ndarray:
    """``causes`` independent standard-normal variables and a last one, their noisy sum."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, causes))
    return np.column_stack([x, x.sum(axis=1) + 0.5 * rng.standard_normal(n)])


def in_degree(out: andrey.StructureOutput, node: int) -> int:
    return sum(
        1
        for _, head, kind in out.structure.oriented_edges(index=True)
        if head == node and kind == "directed"
    )


def test_ges_max_parents_stops_a_variable_short_of_all_its_causes():
    # Three causes of the last variable: the default limit of 4 / 2 = 2 admits the third.
    # Past one parent the variable gets no new edges, and the graph scores worse.
    x = fan_in(3)
    full, limited = andrey.ges(x), andrey.ges(x, max_parents=1)
    assert in_degree(full, 3) == 3
    assert in_degree(limited, 3) < 3
    assert limited.metadata["score"] > full.metadata["score"]


def test_ges_max_parents_replaces_the_default_limit_of_half_the_variables():
    # Five causes of the last of six variables: the default limit of 3 lets it reach four parents,
    # exactly as max_parents=3 does, and a larger value lifts the limit.
    x = fan_in(5)
    default = andrey.ges(x)
    same = andrey.ges(x, max_parents=3)
    lifted = andrey.ges(x, max_parents=4)
    assert in_degree(default, 5) == 4
    assert same.structure.oriented_edges() == default.structure.oriented_edges()
    assert same.metadata["score"] == default.metadata["score"]
    assert in_degree(lifted, 5) == 5
    assert lifted.metadata["score"] < default.metadata["score"]


@pytest.mark.parametrize(
    ("value", "error"), [(-1, ValueError), (2.5, TypeError), (True, TypeError)]
)
def test_ges_rejects_a_bad_max_parents(value, error):
    with pytest.raises(error, match="max_parents"):
        andrey.ges(fan_in(3), max_parents=value)


def test_pc_output_contract():
    out = andrey.pc(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag"


def test_ges_output_contract():
    out = andrey.ges(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag" and "score" in out.metadata


def test_pc_rejects_unknown_indep_test():
    with pytest.raises(NotImplementedError):
        andrey.pc(datagen.generate("gauss_5v"), indep_test="no_such_test")


def test_ges_rejects_unknown_score():
    with pytest.raises(NotImplementedError):
        andrey.ges(datagen.generate("gauss_5v"), score_func="local_score_CV_general")


def test_pc_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        andrey.pc(datagen.generate("gauss_5v"), alpha=1.5)
