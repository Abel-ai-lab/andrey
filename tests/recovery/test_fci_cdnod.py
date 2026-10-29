from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint.pc import _discover_skeleton, _orient_colliders
from andrey.core import TAIL, GraphStructure
from andrey.core.orient import meek, pdag2dag
from andrey.search.hc import tie_ranks
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


def _stub_measure(
    independent: set[tuple[int, int, tuple[int, ...]]], columns: np.ndarray | None = None
) -> type:
    """A module measure that is 0.01 for the listed ``(cause, effect, Z)`` and 0.5 otherwise.

    The step runs on its data sorted by rank, so the stub reads its indices as positions in
    ``columns``, the data a fit was given, and finds each of its own columns there by value; with
    no ``columns`` it reads them as they are.
    """

    class Stub:
        def __init__(self, data: np.ndarray, context: np.ndarray) -> None:
            self.n_features = data.shape[1]
            self.label = list(range(self.n_features))
            if columns is not None:
                self.label = [
                    next(j for j, col in enumerate(columns.T) if np.array_equal(col, own))
                    for own in data.T
                ]

        def __call__(self, cause: int, effect: int, cond: list[int]) -> float:
            key = (
                self.label[cause],
                self.label[effect],
                tuple(sorted(self.label[z] for z in cond)),
            )
            return 0.01 if key in independent else 0.5

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

    def spy(pdag, data, context, changing, threshold, **kwargs):
        seen.append(list(changing))
        return step(pdag, data, context, changing, threshold, **kwargs)

    monkeypatch.setattr(engine, "_orient_by_independent_changes", spy)
    rng = np.random.default_rng(0)
    domain = np.repeat(np.arange(4.0), 200)
    x = 1.0 * domain + rng.standard_normal(800)
    y = x + 0.5 * domain + rng.standard_normal(800)
    z = rng.standard_normal(800)
    data = np.column_stack([x, y, z])
    monkeypatch.setattr(engine, "_ModuleDependence", _stub_measure({(0, 1, ())}, data))
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


def _invariant_to_column_order(fit, x: np.ndarray, *args, ordered=None) -> bool:
    """Assert ``fit`` maps back to its own output under seeded column orders; return whether
    ``fit`` with the keywords ``ordered``, ``in_column_order=True`` by default, changes under any
    of them."""
    ordered = ordered or dict(in_column_order=True)
    marks = fit(x, *args).to_numpy()
    in_order = fit(x, *args, **ordered).to_numpy()
    in_order_changed = False
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = fit(x[:, order], *args).to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)
        permuted = fit(x[:, order], *args, **ordered).to_numpy()
        in_order_changed |= not np.array_equal(permuted[np.ix_(back, back)], in_order)
    return in_order_changed


def _position_draw(name: str) -> tuple[np.ndarray, tuple]:
    """A draw and the arguments after it, for ``name``'s facade and engine alike, on which firing
    the orientation rules one at a time differs from rounds under some column order."""
    if name == "pc":
        return data.sample_scm(d=10, n=100, seed=9, density=4.0, scale="raw").data, ()
    if name == "cdnod":
        x = data.sample_scm(d=5, n=200, seed=17, density=3.0, scale="standardize").data
        context = np.repeat([0.0, 1.0], 100)
        x[context == 1, 0] += 1.5
        return x, (context,)
    n, seed = (3000, 9) if name == "fci" else (300, 0)
    return data.sample_scm(d=6, n=n, seed=seed, density=3.0, latents=1, scale="raw").data, ()


@pytest.mark.parametrize("name", ["pc", "cdnod", "fci", "gfci"])
def test_tie_break_position_fires_the_orientation_rules_in_column_order(name):
    """``"position"`` reaches the engine as ``in_column_order``, ``"rank"`` as rounds."""
    x, args = _position_draw(name)
    # Only the rules differ between the two settings, not the order of the colliders or of the
    # possible-d-separation step.
    kwargs = dict(collider_order="p_value") if name in ("pc", "cdnod") else {}
    kwargs |= dict(possible_dsep="fast") if name in ("fci", "gfci") else {}
    module = "andrey.search.gfci" if name == "gfci" else f"andrey.constraint.{name}"
    engine = getattr(importlib.import_module(module), name)
    facade = getattr(andrey, name)
    differs = False
    orders = [np.arange(x.shape[1])] + [
        np.random.default_rng(seed).permutation(x.shape[1]) for seed in range(4)
    ]
    for order in orders:
        columns = x[:, order]
        rounds = facade(columns, *args, **kwargs).structure.to_numpy()
        np.testing.assert_array_equal(
            rounds, engine(columns, *args, **kwargs, in_column_order=False).to_numpy()
        )
        with andrey.config(tie_break="position"):
            one_at_a_time = facade(columns, *args, **kwargs).structure.to_numpy()
        np.testing.assert_array_equal(
            one_at_a_time, engine(columns, *args, **kwargs, in_column_order=True).to_numpy()
        )
        differs |= not np.array_equal(rounds, one_at_a_time)
    assert differs


def test_cdnod_orientations_are_invariant_to_column_order():
    """A seed on which Meek's rules direct an edge both ways."""
    x = data.sample_scm(d=5, n=200, seed=17, density=3.0, scale="standardize").data
    context = np.repeat([0.0, 1.0], 100)
    x[context == 1, 0] += 1.5
    cdnod = importlib.import_module("andrey.constraint.cdnod").cdnod
    assert _invariant_to_column_order(cdnod, x, context)


def _changing_domains(d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """600 rows from 20 domains of a random linear DAG over ``d`` variables, columns shuffled.

    Each edge weight and each variable's noise scale are drawn once per domain; 70% of the
    variables also shift their mean per domain, so the index keeps an edge to each of them under a
    mixed-data test. Returns ``(data, domain_index)``.
    """
    rng = np.random.default_rng(seed)
    n, n_domains = 600, 20
    edge = np.triu(rng.random((d, d)) < 2.0 / (d - 1), 1)
    weight = rng.choice([-1, 1], (d, d)) * rng.uniform(0.6, 1.2, (d, d))
    domain = np.repeat(np.arange(n_domains), n // n_domains)
    shifts = rng.random(d) < 0.7
    x = np.zeros((n, d))
    for j in range(d):
        for i in np.flatnonzero(edge[:, j]):
            x[:, j] += weight[i, j] * rng.uniform(0.4, 1.6, n_domains)[domain] * x[:, i]
        x[:, j] += rng.uniform(0.5, 2.0, n_domains)[domain] * rng.standard_normal(n)
        if shifts[j]:
            x[:, j] += rng.uniform(-3.0, 3.0, n_domains)[domain]
    return x[:, rng.permutation(d)], domain.astype(float)


@pytest.mark.parametrize(("d", "seed"), [(5, 34), (6, 0)])
def test_cdnod_independent_change_step_is_invariant_to_column_order(d, seed):
    """Draws on which the step orients edges, and visiting them in column order depends on it."""
    x, domain = _changing_domains(d, seed)
    engine = importlib.import_module("andrey.constraint.cdnod")
    categorical = np.zeros(d, dtype=bool)

    def fit(x, domain, **kwargs):
        return engine.cdnod(
            x,
            domain,
            indep_test="cond_gauss",
            categorical=categorical,
            dependence_threshold=0.1,
            **kwargs,
        )

    skipped = engine.cdnod(x, domain, indep_test="cond_gauss", categorical=categorical)
    assert not np.array_equal(fit(x, domain).to_numpy(), skipped.to_numpy())
    assert _invariant_to_column_order(fit, x, domain)


@pytest.mark.parametrize(("d", "seed"), [(5, 34), (6, 0)])
def test_cdnod_independent_change_step_visits_columns_by_rank(monkeypatch, d, seed):
    """On the step's own input, relabeling the nodes relabels its orientations; with
    ``in_column_order`` the columns are visited by position, so it does not, and on columns
    already sorted by rank it gives the default's orientations."""
    engine = importlib.import_module("andrey.constraint.cdnod")
    step = engine._orient_by_independent_changes
    calls: list[tuple] = []

    def spy(*args, **kwargs):
        calls.append(args)
        return step(*args, **kwargs)

    monkeypatch.setattr(engine, "_orient_by_independent_changes", spy)
    x, domain = _changing_domains(d, seed)
    engine.cdnod(
        x,
        domain,
        indep_test="cond_gauss",
        categorical=np.zeros(d, dtype=bool),
        dependence_threshold=0.1,
    )
    ((pdag, columns, context, changing, threshold),) = calls

    def relabeled(order: np.ndarray, **kwargs) -> np.ndarray:
        """The step on ``order``'s columns and nodes, its orientations mapped back."""
        nodes = np.append(order, np.arange(d, len(pdag)))
        back = np.argsort(nodes)
        marks = step(
            pdag[np.ix_(nodes, nodes)],
            columns[:, order],
            context,
            [int(back[c]) for c in changing],
            threshold,
            **kwargs,
        )
        return marks[np.ix_(back, back)]

    marks = step(pdag, columns, context, changing, threshold)
    assert not np.array_equal(marks, pdag)
    in_order = step(pdag, columns, context, changing, threshold, in_column_order=True)
    in_order_changed = False
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(d)
        assert np.array_equal(relabeled(order), marks)
        in_order_changed |= not np.array_equal(relabeled(order, in_column_order=True), in_order)
    assert in_order_changed
    by_rank = np.argsort(tie_ranks(columns))
    assert np.array_equal(relabeled(by_rank, in_column_order=True), marks)


@pytest.mark.parametrize(
    ("module", "draw"),
    [
        ("andrey.constraint.fci", dict(n=3000, seed=9)),
        ("andrey.search.gfci", dict(n=300, seed=0)),
    ],
)
def test_pag_orientations_are_invariant_to_column_order(module, draw):
    """Seeds on which FCI's rules mark one endpoint both ways."""
    x = data.sample_scm(d=6, density=3.0, latents=1, **draw, scale="raw").data
    fit = getattr(importlib.import_module(module), module.rsplit(".", 1)[1])
    assert _invariant_to_column_order(fit, x)


@pytest.mark.parametrize("module", ["andrey.constraint.fci", "andrey.search.gfci"])
def test_possible_dsep_is_invariant_to_column_order(module):
    """A draw on which the possible-d-separation step removes different edges when it visits
    them in column order."""
    x = data.sample_scm(d=8, n=300, seed=9, density=3.0, latents=1, scale="raw").data
    fit = getattr(importlib.import_module(module), module.rsplit(".", 1)[1])
    assert _invariant_to_column_order(fit, x, ordered=dict(possible_dsep="fast"))


@pytest.mark.parametrize("name", ["fci", "gfci"])
def test_tie_break_position_gives_the_output_of_possible_dsep_fast(name):
    """On a draw where the two orders remove different edges, ``"position"`` gives the output of
    ``possible_dsep="fast"``, and ``"rank"`` that of ``"order_free"``."""
    x = data.sample_scm(d=8, n=300, seed=9, density=3.0, latents=1, scale="raw").data
    facade = getattr(andrey, name)
    modes_differ = False
    orders = [np.arange(x.shape[1])] + [
        np.random.default_rng(seed).permutation(x.shape[1]) for seed in range(4)
    ]
    for order in orders:
        columns = x[:, order]
        order_free = facade(columns, possible_dsep="order_free").structure
        assert facade(columns).structure == order_free
        modes_differ |= order_free != facade(columns, possible_dsep="fast").structure
        with andrey.config(tie_break="position"):
            assert facade(columns).structure == facade(columns, possible_dsep="fast").structure
    assert modes_differ


@pytest.mark.parametrize("name", ["fci", "gfci"])
@pytest.mark.parametrize(
    ("tie_break", "given", "reaches"),
    [
        ("rank", None, "order_free"),
        ("position", None, "fast"),
        ("rank", "fast", "fast"),
        ("position", "order_free", "order_free"),
    ],
)
def test_possible_dsep_follows_tie_break_unless_given(monkeypatch, name, tie_break, given, reaches):
    """The engine gets ``possible_dsep`` from the setting when the call gives none, and the call's
    value when it gives one."""
    module = importlib.import_module(
        "andrey.search.gfci" if name == "gfci" else "andrey.constraint.fci"
    )
    engine = getattr(module, name)
    received = []

    def spy(*args, **kwargs):
        received.append(kwargs["possible_dsep"])
        return engine(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    x = data.sample_scm(d=4, n=100, seed=0, density=2.0, latents=1, scale="raw").data
    with andrey.config(tie_break=tie_break):
        getattr(andrey, name)(x, **({} if given is None else dict(possible_dsep=given)))
    assert received == [reaches]


def test_fci_output_contract():
    assert andrey.fci(datagen.generate("gauss_5v")).structure.kind == "pag"


def test_cdnod_output_contract():
    out = andrey.cdnod(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"))
    assert out.structure.kind == "cpdag"


def test_fci_rejects_unknown_indep_test():
    with pytest.raises(ValueError):
        andrey.fci(datagen.generate("gauss_5v"), indep_test="no_such_test")


def test_cdnod_rejects_unknown_indep_test():
    x, c = datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v")
    with pytest.raises(ValueError):
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

    return data.sample_scm(
        graph="erdos_renyi", d=d, n=1200, seed=seed, density=2.0, latents=1, scale="raw"
    ).data


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
