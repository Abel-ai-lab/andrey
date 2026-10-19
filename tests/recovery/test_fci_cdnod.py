from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint.pc import _discover_skeleton, _orient_colliders
from andrey.core import TAIL, GraphStructure
from andrey.core.orient import meek, pdag2dag
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def test_fci_recovery():
    out = andrey.fci(datagen.generate("gauss_5v"))
    assert_recorded_output(out, "FCI", case="gauss_5v")


def test_cdnod_recovery():
    out = andrey.cdnod(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"))
    assert_recorded_output(out, "CDNOD", case="gauss_5v")


def _domain_shift_chain() -> tuple[np.ndarray, np.ndarray]:
    """Two domains of a linear-Gaussian model ``0 -> 1 -> 2 -> 3``, ``1 -> 4``.

    Only the mean of variable 0 differs between the domains. The graph has no collider, so its
    equivalence class leaves every edge undirected.
    """
    rng = np.random.default_rng(0)
    n = 2000
    weights = np.zeros((5, 5))
    weights[0, 1], weights[1, 2], weights[2, 3], weights[1, 4] = 0.8, -0.7, 0.9, 0.6
    domain = np.repeat([0.0, 1.0], n // 2)
    shift = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    x = np.zeros((n, 5))
    for j in range(5):  # columns in causal order
        x[:, j] = x @ weights[:, j] + shift[j] * domain + rng.standard_normal(n)
    return x, domain


def test_cdnod_orients_the_planted_graph_from_a_domain_shift():
    """The domain index points into variable 0, and that one arrowhead orients the whole chain."""
    x, domain = _domain_shift_chain()
    planted = [(0, 1, "directed"), (1, 2, "directed"), (1, 4, "directed"), (2, 3, "directed")]
    assert andrey.cdnod(x, domain).structure.oriented_edges() == planted
    # Without the index the same data leave every edge open.
    assert {kind for *_, kind in andrey.pc(x).structure.oriented_edges()} == {"undirected"}


def test_cdnod_finds_no_edge_between_independent_columns():
    """A shift in one column's mean links that column to the index only, which is dropped."""
    rng = np.random.default_rng(0)
    domain = np.repeat([0.0, 1.0], 400)
    x = rng.standard_normal((800, 4))
    x[:, 0] += 2.0 * domain
    assert andrey.cdnod(x, domain).structure.oriented_edges() == []


def _stub_measure(independent: set[tuple[int, int, tuple[int, ...]]]) -> type:
    """A module measure that is 0.01 for the listed ``(cause, effect, Z)`` and 0.5 otherwise."""

    class Stub:
        def __init__(self, data: np.ndarray, context: np.ndarray) -> None:
            self.n_features = data.shape[1]

        def __call__(self, cause: int, effect: int, cond: list[int]) -> float:
            return 0.01 if (cause, effect, tuple(sorted(cond))) in independent else 0.5

    return Stub


def test_cdnod_independent_change_step_orients_the_papers_square(monkeypatch):
    """Huang et al. 2020, Figure 5: ``V3 -> V1``, ``V3 -> V4``, ``V1 -> V2``, ``V4 -> V2``.

    Every variable changes with the index and the square starts undirected. The stub gives the
    independencies the figure lists: ``P(V3)`` and ``P(V1 | V3)``, ``P(V3)`` and ``P(V4 | V3)``
    with no conditioning set, then ``P(V1, V3)`` and ``P(V2 | V1, V3)``, and ``P(V4, V3)`` and
    ``P(V2 | V4, V3)`` with ``V3``. Columns are ``V1, V2, V3, V4``.
    """
    engine = importlib.import_module("andrey.constraint.cdnod")
    independent = {(2, 0, ()), (2, 3, ()), (0, 1, (2,)), (3, 1, (2,))}
    monkeypatch.setattr(engine, "_ModuleDependence", _stub_measure(independent))
    square = np.zeros((4, 4), dtype=np.int8)
    for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
        square[a, b] = square[b, a] = TAIL
    out = engine._orient_by_independent_changes(
        square, np.zeros((5, 4)), np.zeros(5), [0, 1, 2, 3], 0.05
    )
    edges = GraphStructure.from_numpy(out, kind="cpdag").oriented_edges(index=True)
    assert sorted(edges) == [
        (0, 1, "directed"),
        (2, 0, "directed"),
        (2, 3, "directed"),
        (3, 1, "directed"),
    ]


def _changing_modules(seed: int, edges: list[tuple[int, int]], d: int) -> tuple[np.ndarray, ...]:
    """600 rows from 20 domains of a linear DAG whose modules change independently.

    Each root's noise scale and each edge weight are drawn once per domain; a non-root variable
    adds noise of scale 0.5. Columns are in causal order. Returns ``(data, domain_index)``.
    """
    rng = np.random.default_rng(seed)
    n, n_domains = 600, 20
    domain = np.repeat(np.arange(n_domains), n // n_domains)
    x = np.zeros((n, d))
    for j in range(d):
        parents = [i for i, child in edges if child == j]
        for i in parents:
            x[:, j] += rng.uniform(-2.0, 2.0, n_domains)[domain] * x[:, i]
        if parents:
            x[:, j] += 0.5 * rng.standard_normal(n)
        else:
            x[:, j] = rng.uniform(0.5, 2.0, n_domains)[domain] * rng.standard_normal(n)
    return x, domain.astype(float)


def _step_edges(edges: list[tuple[int, int]], d: int, data, domain, threshold: float) -> list:
    """Run the independent-change step on the undirected skeleton of ``edges``, all changing."""
    engine = importlib.import_module("andrey.constraint.cdnod")
    skeleton = np.zeros((d, d), dtype=np.int8)
    for a, b in edges:
        skeleton[a, b] = skeleton[b, a] = TAIL
    out = engine._orient_by_independent_changes(skeleton, data, domain, list(range(d)), threshold)
    return sorted(GraphStructure.from_numpy(out, kind="cpdag").oriented_edges(index=True))


def test_cdnod_independent_change_step_recovers_a_planted_complete_dag():
    """Given the skeleton of ``0 -> 1``, ``0 -> 2``, ``1 -> 2``, as in the paper's setting 2."""
    truth = [(0, 1), (0, 2), (1, 2)]
    data, domain = _changing_modules(2, truth, 3)
    assert _step_edges(truth, 3, data, domain, 0.06) == [(a, b, "directed") for a, b in truth]


def test_cdnod_independent_change_step_leaves_dependent_changes_undirected():
    """When the measure is above the threshold in both directions, the edge stays undirected."""
    data, domain = _changing_modules(4, [(0, 1)], 2)
    assert _step_edges([(0, 1)], 2, data, domain, 0.06) == [(0, 1, "undirected")]
    assert _step_edges([(0, 1)], 2, data, domain, 0.15) == [(0, 1, "directed")]


def test_cdnod_independent_change_step_never_closes_a_directed_cycle(monkeypatch):
    """The stub calls ``1 -> 0``, ``0 -> 2`` and ``2 -> 1`` independent on an undirected triangle.

    The step orients ``1 -> 0`` and ``0 -> 2``; ``2 -> 1`` would close a cycle, so that edge
    stays undirected, and the result has a DAG extension.
    """
    engine = importlib.import_module("andrey.constraint.cdnod")
    independent = {(1, 0, ()), (0, 2, ()), (2, 1, ())}
    monkeypatch.setattr(engine, "_ModuleDependence", _stub_measure(independent))
    triangle = np.zeros((3, 3), dtype=np.int8)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        triangle[a, b] = triangle[b, a] = TAIL
    out = engine._orient_by_independent_changes(
        triangle, np.zeros((5, 3)), np.zeros(5), [0, 1, 2], 0.05
    )
    edges = GraphStructure.from_numpy(out, kind="cpdag").oriented_edges(index=True)
    assert sorted(edges) == [(0, 2, "directed"), (1, 0, "directed"), (1, 2, "undirected")]
    pdag2dag(meek(out))


def test_cdnod_module_changes_across_two_domains_always_depend():
    """With two index values each module takes two embeddings, so the measure is 1 and no
    threshold in ``(0, 1)`` orients the edge."""
    engine = importlib.import_module("andrey.constraint.cdnod")
    rng = np.random.default_rng(0)
    domain = np.repeat([0.0, 1.0], 300)
    x = (1.0 + domain) * rng.standard_normal(600)
    y = (0.5 + 1.5 * domain) * x + 0.5 * rng.standard_normal(600)
    data = np.column_stack([x, y])
    measure = engine._ModuleDependence(data, domain)
    assert measure(0, 1, []) == pytest.approx(1.0)
    assert measure(1, 0, []) == pytest.approx(1.0)
    assert _step_edges([(0, 1)], 2, data, domain, 0.99) == [(0, 1, "undirected")]


def test_cdnod_runs_the_independent_change_step_on_variables_that_change(monkeypatch):
    """The step sees the variables the index keeps an edge to, and only with a threshold."""
    engine = importlib.import_module("andrey.constraint.cdnod")
    seen: list[list[int]] = []
    step = engine._orient_by_independent_changes

    def spy(pdag, data, context, changing, threshold):
        seen.append(list(changing))
        return step(pdag, data, context, changing, threshold)

    monkeypatch.setattr(engine, "_orient_by_independent_changes", spy)
    monkeypatch.setattr(engine, "_ModuleDependence", _stub_measure({(0, 1, ())}))
    rng = np.random.default_rng(0)
    domain = np.repeat(np.arange(4.0), 200)
    x = 1.0 * domain + rng.standard_normal(800)
    y = x + 0.5 * domain + rng.standard_normal(800)
    z = rng.standard_normal(800)
    data = np.column_stack([x, y, z])
    assert andrey.cdnod(data, domain).structure.oriented_edges() == [(0, 1, "undirected")]
    assert seen == []
    out = andrey.cdnod(data, domain, dependence_threshold=0.05)
    assert seen == [[0, 1]]
    assert out.structure.oriented_edges() == [(0, 1, "directed")]


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_cdnod_rejects_a_dependence_threshold_outside_the_unit_interval(threshold):
    x, c = datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v")
    with pytest.raises(ValueError, match="dependence_threshold"):
        andrey.cdnod(x, c, dependence_threshold=threshold)


def shifted_domains() -> tuple[np.ndarray, np.ndarray]:
    """Two domains, with two variables shifted between them. Colliders of the augmented skeleton
    conflict on this seed, so the collider order decides which are kept."""
    x = data.sample_scm(d=12, n=600, seed=1, density=3.0, scale="standardize").data
    context = np.repeat([0.0, 1.0], 300)
    x[context == 1, 0] += 1.5
    x[context == 1, 6] += 1.5
    return x, context


def test_cdnod_passes_the_collider_order_on():
    """On conflicting colliders the two orders keep different ones."""
    x, context = shifted_domains()
    column = andrey.cdnod(x, context, collider_order="column").structure
    assert column != andrey.cdnod(x, context).structure


def test_cdnod_collider_order_follows_tie_break_unless_given():
    x, context = shifted_domains()
    default = andrey.cdnod(x, context).structure
    column = andrey.cdnod(x, context, collider_order="column").structure
    with andrey.config(tie_break="position"):
        assert andrey.cdnod(x, context).structure == column
        assert andrey.cdnod(x, context, collider_order="p_value").structure == default


def test_cdnod_is_invariant_to_column_order():
    x, context = shifted_domains()
    augmented = np.column_stack([x, context])
    skeleton, sepsets, pvalues = _discover_skeleton(augmented, 0.05)
    trace: list[dict] = []
    _orient_colliders(skeleton, sepsets, pvalues, _trace=trace)
    assert any(
        r["step"] == "collider" and not r["oriented"] and r["triple"][1] not in r["sepset"]
        for r in trace
    )
    marks = andrey.cdnod(x, context).structure.to_numpy()
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = andrey.cdnod(x[:, order], context).structure.to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)


def test_fci_output_contract():
    assert andrey.fci(datagen.generate("gauss_5v")).structure.kind == "pag"


def test_cdnod_output_contract():
    out = andrey.cdnod(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"))
    assert out.structure.kind == "cpdag"


def test_fci_rejects_unknown_indep_test():
    with pytest.raises(NotImplementedError):
        andrey.fci(datagen.generate("gauss_5v"), indep_test="no_such_test")


def test_cdnod_rejects_unknown_indep_test():
    x, c = datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v")
    with pytest.raises(NotImplementedError):
        andrey.cdnod(x, c, indep_test="no_such_test")


def test_cdnod_rejects_mismatched_context():
    x = datagen.generate("gauss_5v")
    with pytest.raises(ValueError, match="one value per row"):
        andrey.cdnod(x, np.zeros((len(x) + 1, 1)))


def test_cdnod_rejects_an_index_with_the_right_size_and_the_wrong_shape():
    """A two-column index with as many cells as rows would be flattened onto the wrong rows."""
    x = datagen.generate("gauss_5v")
    x = x[: len(x) // 2 * 2]
    with pytest.raises(ValueError, match="one value per row"):
        andrey.cdnod(x, np.zeros((len(x) // 2, 2)))
    column = datagen.domain_index("gauss_5v")[: len(x)]
    flat = andrey.cdnod(x, column).structure
    assert andrey.cdnod(x, column.reshape(-1, 1)).structure == flat


# --- general-input behavior ----------------------------------------------------------------------
#
# Random latent-confounded models check repeatability and well-formed output beyond the fixed
# baseline. FCI/GFCI column-order sensitivity is measured by qa.column_order.


def _latent_confounded_sample(seed: int, d: int) -> np.ndarray:
    """An observed sample from a random DAG with one hidden confounder."""
    from andrey import data

    return data.sample_scm(graph="erdos_renyi", d=d, n=1200, seed=seed, density=2.0, latents=1).data


def _fit(learner: str, x: np.ndarray):
    if learner == "cdnod":
        context = np.zeros((len(x), 1))
        context[len(x) // 2 :] = 1.0
        return andrey.cdnod(x, context).structure
    return getattr(andrey, learner)(x).structure


@pytest.mark.parametrize(
    ("learner", "kind"),
    [("fci", "pag"), ("gfci", "pag"), ("cdnod", "cpdag")],
)
def test_constraint_family_is_deterministic_and_well_formed(learner, kind):
    for seed in range(4):
        for d in (6, 8):
            x = _latent_confounded_sample(seed, d)
            observed = x.shape[1]  # d nodes minus the hidden confounder
            first = _fit(learner, x)
            assert first.kind == kind
            assert first.n_nodes == observed
            first.validate()
            repeat = _fit(learner, x).to_numpy()
            assert np.array_equal(repeat, first.to_numpy()), f"{learner} is not deterministic"

            if learner == "cdnod":
                order = np.random.default_rng(seed).permutation(observed)
                back = np.argsort(order)
                permuted = _fit(learner, x[:, order]).to_numpy()[np.ix_(back, back)]
                assert np.array_equal(permuted, first.to_numpy()), (
                    f"{learner} changes when the columns are reordered"
                )
