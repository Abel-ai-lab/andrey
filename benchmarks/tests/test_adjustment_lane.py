"""The pair lane: the packed answers, the planted models, the scorer, the adapters, and the driver.

The driver tests replace ``run_task`` with an in-process fit, so they need no harness environment
and no reference code; one test runs the real runner on the baselines when the bench interpreter is
there. The references' own tests are in their adapters' files and skip without their environments.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_r

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import adjustment_cpu  # noqa: E402

from andrey.core import adjustment as adj  # noqa: E402
from andrey.core import effects as core_effects  # noqa: E402
from andrey_bench import adjustment_lane as L  # noqa: E402
from andrey_bench import adjustment_models as M  # noqa: E402
from andrey_bench import gaussian_networks, network_models, networks, runner  # noqa: E402
from andrey_bench.adapters import andrey_adjustment, baseline_adjustment  # noqa: E402
from andrey_bench.contracts import SolutionAdapter  # noqa: E402

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]


# --- the packed answer -----------------------------------------------------------------------


def test_the_record_vocabulary_is_andreys():
    assert L.ESTIMANDS == core_effects.ESTIMANDS
    assert L.SET_KINDS == core_effects.SET_KINDS


def test_records_survive_the_packing_with_every_field():
    records = [
        core_effects.effect_record(
            0,
            3,
            estimand="total",
            nonzero=True,
            identifiable=False,
            adjustment_sets=[[1], [1, 2], []],
            set_kind="local_parents",
        ),
        core_effects.effect_record(3, 0, estimand="total", nonzero=False, identifiable=True),
        core_effects.effect_record(
            1, 2, estimand="direct", adjustment_sets=[[0, 3]], set_kind="parents_of_outcome"
        ),
    ]
    packed = L.pack_effects(records, 4, 123)
    assert packed.dtype == np.int64 and packed.shape == (5 + 1, 8 + 4)
    got, tests = L.unpack_effects(packed)
    assert tests == 123
    assert got[0]["adjustment_sets"] == [[], [1], [1, 2]]  # the record's own canonical order
    assert [{k: v for k, v in g.items() if k != "adjustment_sets"} for g in got] == [
        {k: v for k, v in r.items() if k != "adjustment_sets"} for r in records
    ]
    assert got[2]["adjustment_sets"] == [[0, 3]]
    core_effects.check_effects(got)


def test_an_empty_answer_and_an_unreported_count_pack_as_nothing():
    packed = L.pack_effects([], 5, None)
    assert packed.shape == (1, 8 + 5)
    assert L.unpack_effects(packed) == ([], None)
    with pytest.raises(ValueError, match="packed answer"):
        L.unpack_effects(np.zeros((3, 4)))


def test_the_empty_set_is_a_set_and_no_set_is_not():
    with_empty = core_effects.effect_record(
        0, 1, estimand="total", adjustment_sets=[[]], set_kind="valid"
    )
    without = core_effects.effect_record(0, 1, estimand="total", nonzero=False)
    assert L.unpack_effects(L.pack_effects([with_empty], 2))[0][0]["adjustment_sets"] == [[]]
    assert L.unpack_effects(L.pack_effects([without], 2))[0][0]["adjustment_sets"] == []


def test_a_reference_that_returns_a_graph_reads_its_records_in_the_parent():
    class FromGraph:
        def records(self, native, params):
            return [{"graph": native.shape}], 7

    class Plain:
        pass

    native = L.pack_effects([], 3, 5)
    assert L.answer_records(FromGraph(), native, {}) == ([{"graph": (1, 11)}], 7)
    assert L.answer_records(Plain(), native, {}) == ([], 5)


# --- the effect estimate ---------------------------------------------------------------------


def test_the_effect_estimate_is_the_standardized_regression_coefficient():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(20_000)
    x = z + rng.standard_normal(20_000)
    y = 2.0 * x + 3.0 * z + rng.standard_normal(20_000)
    data = np.column_stack([x, y, z])
    adjusted = L.ols_effect(data, 0, 1, [2])
    assert adjusted == pytest.approx(2.0 * x.std() / y.std(), abs=0.02)
    assert L.ols_effect(data, 0, 1, []) > adjusted + 0.1  # the confounder biases the unadjusted one
    constant = np.column_stack([np.ones(10), np.arange(10.0)])
    assert np.isnan(L.ols_effect(constant, 0, 1, []))


def test_an_abstention_has_no_effect_error_and_a_zero_answer_estimates_zero():
    fresh = np.random.default_rng(1).standard_normal((500, 3))
    base = {"treatment": 0, "outcome": 1, "estimand": "total", "identifiable": None}
    abstains = {**base, "nonzero": None, "adjustment_sets": [], "set_kind": None}
    says_zero = {**abstains, "nonzero": False}
    assert L.effect_estimates(abstains, fresh, 0.5) is None
    assert L.effect_estimates(says_zero, fresh, 0.5)["estimates"] == [0.0]
    assert L.effect_estimates(says_zero, fresh, 0.5)["error"] == pytest.approx(0.5)
    two = {**abstains, "adjustment_sets": [[], [2]], "set_kind": "local_parents"}
    got = L.effect_estimates(two, fresh, 0.0)
    assert len(got["estimates"]) == 2 and got["worst"] >= got["error"] >= 0


# --- the planted models ----------------------------------------------------------------------


#: Small parameters for the settings whose defaults are large.
SMALL = {
    "locpc-identifiable": {"v": 10},
    "locpc-nonidentifiable": {"v": 10},
    "random-graph": {"nodes": 30},
    "ld3-er": {"nodes": 30},
}


def test_every_registered_setting_draws_a_model_with_the_conventions_of_the_scorer():
    for name, setting in M.SETTINGS.items():
        if name in ("magic-niab", "sangiovese", "mildew"):  # published files; tested below
            continue
        model = setting.make(0, **{**setting.defaults, **SMALL.get(name, {})})
        data = model.rows(400, M.DATA)
        assert data.shape == (400, model.p) and model.d >= model.p
        assert model.hidden == tuple(range(model.p, model.d))  # hidden nodes last
        assert model.dag.any()
        pair = model.pairs[0]
        assert pair.estimand == setting.estimand and pair.treatment != pair.outcome
        assert max(pair.treatment, pair.outcome) < model.p  # the pair is observed
        assert setting.hidden or not model.hidden


def test_the_columns_are_not_in_a_causal_order_and_follow_the_seed():
    first = M.parse_cell("ldecc@500").model(3)
    again = M.parse_cell("ldecc@500").model(3)
    other = M.parse_cell("ldecc@500").model(4)
    np.testing.assert_array_equal(first.dag, again.dag)
    np.testing.assert_array_equal(first.rows(500, M.DATA), again.rows(500, M.DATA))
    assert not np.array_equal(first.dag, other.dag)
    # the DAG is not upper triangular in column order: some arc points to an earlier column
    assert np.tril(first.dag, -1).any()


def test_the_data_and_the_fresh_sample_are_two_samples_of_one_model():
    model = M.parse_cell("entner-effect-admissible@2000").model(0)
    data, fresh = model.rows(2000, M.DATA), model.rows(2000, M.FRESH)
    assert not np.array_equal(data, fresh)
    np.testing.assert_allclose(
        np.cov(data.T), model.covariance()[: model.p, : model.p], atol=0.5, rtol=0.35
    )
    np.testing.assert_allclose(np.cov(fresh.T), np.cov(data.T), atol=0.5, rtol=0.35)


def test_the_true_effect_is_the_population_regression_coefficient_of_a_valid_set():
    from andrey.metrics import effect_oracle

    for seed in range(3):
        model = M.parse_cell("ldecc@500").model(seed)
        pair = model.pairs[0]
        x, y = pair.treatment, pair.outcome
        record = effect_oracle(model.dag, x, y, set_kind="valid", identified_from="dag")
        cov = model.covariance()
        members = [int(v) for v in record["adjustment_sets"][0]]
        cols = [x, *members]
        beta = np.linalg.solve(cov[np.ix_(cols, cols)], cov[cols, y])
        standardized = beta[0] * np.sqrt(cov[x, x] / cov[y, y])
        assert standardized == pytest.approx(model.true_effect(x, y, "total"))
        assert model.effects()[(y, x)] == 0.0  # the reverse direction is a zero effect


def test_the_direct_effect_is_the_arc_weight_standardized():
    model = M.parse_cell("ld3-dag@500").model(2)
    pair = model.pairs[0]
    cov = model.covariance()
    expected = model.weights[pair.treatment, pair.outcome] * np.sqrt(
        cov[pair.treatment, pair.treatment] / cov[pair.outcome, pair.outcome]
    )
    assert model.true_effect(pair.treatment, pair.outcome, "direct") == pytest.approx(expected)


@pytest.mark.parametrize(
    ("task", "effect", "admissible"),
    [
        ("effect-admissible", True, True),
        ("effect-none", True, False),
        ("zero-admissible", False, True),
        ("zero-none", False, False),
    ],
)
def test_each_of_entners_four_tasks_has_the_property_it_names(task, effect, admissible):
    from andrey.metrics import effect_oracle

    for seed in range(4):
        model = M.parse_cell(f"entner-{task}@100").model(seed)
        assert (model.p, model.d) == (12, 17) and len(model.hidden) == 5
        x, y = model.pairs[0].treatment, model.pairs[0].outcome
        record = effect_oracle(
            model.dag, x, y, set_kind="valid", identified_from="dag", latent=list(model.hidden)
        )
        assert record["nonzero"] is effect and record["identifiable"] is admissible
        # the paper's partial order: the treatment and the outcome have no covariate descendant
        covariates = [v for v in range(model.d) if v not in (x, y)]
        assert not model.dag[x, covariates].any() and not model.dag[y, covariates].any()
        assert not model.dag[y, x]


@pytest.mark.parametrize("identifiable", [True, False])
def test_locpc_cases_are_identifiable_or_not_in_the_true_cpdag(identifiable):
    from andrey.metrics import effect_oracle

    name = "identifiable" if identifiable else "nonidentifiable"
    for seed in range(4):
        model = M.parse_cell(f"locpc-{name}@500,v=20").model(seed)
        x, y = model.pairs[0].treatment, model.pairs[0].outcome
        assert model.dag[x, y]  # X -> Y is in the DAG
        record = effect_oracle(model.dag, x, y, estimand="direct", identified_from="cpdag")
        assert record["identifiable"] is identifiable and record["nonzero"] is True
        assert model.pairs[0].estimand == "direct"


def test_locpc_binary_rows_are_binary_and_carry_no_linear_effect():
    model = M.parse_cell("locpc-identifiable@800,v=10,family=binary").model(1)
    assert model.discrete and model.indep_test == "gsq" and model.effects() == {}
    rows = model.rows(800, M.DATA)
    assert set(np.unique(rows)) <= {0.0, 1.0} and rows.shape == (800, 10)


def test_the_ldp_graph_is_the_papers_with_its_effect_and_its_hidden_variants():
    model = M.parse_cell("ldp@500").model(0)
    names = model.meta["names"]
    arcs = {(names[i], names[j]) for i, j in zip(*np.nonzero(model.dag), strict=True)}
    assert arcs == set(M.LDP_EDGES) and len(names) == 10
    x, y = names.index("X"), names.index("Y")
    assert model.weights[x, y] == 2.75 and model.weights[names.index("Z1"), x] == 1.0
    cov = model.covariance()
    assert model.true_effect(x, y, "total") == pytest.approx(
        3.75 * np.sqrt(cov[x, x] / cov[y, y])
    )  # 2.75 directly and 1.0 through the mediator
    assert (names.index("Z8") not in np.nonzero(model.dag)[0]) and not M.parse_cell(
        "ldp@500,effect=0"
    ).model(0).dag.sum() == model.dag.sum()
    hidden = M.parse_cell("ldp@500,latent=True,hide=B1+Z1").model(0)
    assert hidden.d == 18 and [hidden.meta["names"][v] for v in hidden.hidden] == ["B1", "Z1"]


def test_the_ld3_graph_has_the_parents_the_paper_lists_and_an_outcome_without_descendants():
    model = M.parse_cell("ld3-dag@500").model(0)
    names = model.meta["names"]
    y = names.index("Y")
    parents = {names[i] for i in np.flatnonzero(model.dag[:, y])}
    assert parents == {"X", "Z1", "Z3c", "Z3d", "Z4b", "M2", "B2", "B3"}
    assert not model.dag[y].any() and len(names) == 18
    assert model.weights[names.index("X"), y] == 1.25
    assert model.pairs[0].estimand == "direct"
    lost = M.parse_cell("ld3-dag@500,hide=Z3c+M1").model(0)
    assert lost.p == 16 and len(lost.hidden) == 2


def test_the_ld3_er_outcome_is_the_sink_with_most_parents():
    for seed in range(3):
        model = M.parse_cell("ld3-er@300,nodes=40").model(seed)
        x, y = model.pairs[0].treatment, model.pairs[0].outcome
        dag = model.dag
        assert not dag[y].any() and dag[x, y] and dag.sum() == round(1.5 * 40)
        leaves = [v for v in range(40) if dag[:, v].any() and not dag[v].any()]
        assert dag[:, y].sum() == max(dag[:, v].sum() for v in leaves)


def test_the_ldecc_graph_has_the_structure_the_paper_and_its_code_force():
    for seed in range(3):
        model = M.parse_cell("ldecc@400").model(seed)
        x, y = model.pairs[0].treatment, model.pairs[0].outcome
        assert (model.p, model.d) == (25, 25) and not model.dag[x, y]  # no direct effect
        # X has a covariate parent and a mediator child; Y has a covariate parent
        assert model.dag[:, x].sum() >= 1 and model.dag[x].sum() >= 1 and model.dag[:, y].sum() >= 2
        assert (model.noise_sd**2).min() >= 0.1 and (model.noise_sd**2).max() <= 0.2
        weights = np.abs(model.weights[model.weights != 0])
        assert weights.min() >= 0.25 and weights.max() <= 1.0


def test_the_snap_and_load_pair_is_an_explicit_ancestor_with_an_identifiable_effect():
    from andrey.metrics import effect_oracle, to_cpdag

    for seed in range(3):
        model = M.parse_cell("random-graph@300,nodes=60,degree=2").model(seed)
        x, y = model.pairs[0].treatment, model.pairs[0].outcome
        assert (model.dag.sum(0) + model.dag.sum(1)).max() <= 10
        compelled = adj.directed(to_cpdag(model.dag).to_numpy())
        assert y in adj.reach(compelled, [x])
        assert effect_oracle(model.dag, x, y)["identifiable"] is True
        weights = np.abs(model.weights[model.weights != 0])
        assert weights.min() >= 0.5 and weights.max() <= 3.0


def test_a_cell_is_setting_n_and_sorted_parameters():
    cell = M.parse_cell("locpc-identifiable@5000,v=50,family=binary")
    assert (cell.setting, cell.n, cell.params) == (
        "locpc-identifiable",
        5000,
        (("family", "binary"), ("v", 50)),
    )
    assert M.parse_cell(cell.key) == cell
    for bad in ("nope@10", "ldp", "ldp@x", "ldp@10,hide"):
        with pytest.raises(ValueError):
            M.parse_cell(bad)


# --- the published networks ------------------------------------------------------------------


def _toy_network() -> gaussian_networks.GaussianNetwork:
    nodes = ("a", "b", "c", "d")
    weights = np.zeros((4, 4))
    weights[0, 1], weights[0, 2], weights[1, 3], weights[2, 3] = 0.5, 1.0, 2.0, 1.0
    return gaussian_networks.GaussianNetwork(
        "toy", nodes, weights, np.array([1.0, 0.0, 2.0, -1.0]), np.array([1.0, 0.5, 0.5, 1.0])
    )


def test_a_fitted_gaussian_network_is_sampled_with_its_intercepts_and_effects():
    net = _toy_network()
    model = M.finish(
        "toy", 0, net.weights, net.noise_sd, [], [(0, 3, "total")], intercepts=net.intercepts
    )
    data = model.rows(40_000, M.DATA)
    # E[a] = 1, E[b] = 0.5, E[c] = 2 + 1, E[d] = -1 + 2 * 0.5 + 3: the means follow the structure
    np.testing.assert_allclose(sorted(data.mean(0)), sorted([1.0, 0.5, 3.0, 3.0]), atol=0.05)
    pair, cov = model.pairs[0], model.covariance()
    assert model.true_effect(pair.treatment, pair.outcome, "total") == pytest.approx(
        2.0 * np.sqrt(cov[pair.treatment, pair.treatment] / cov[pair.outcome, pair.outcome])
    )  # 0.5 * 2 through b and 1 * 1 through c


def test_the_random_pair_of_a_network_changes_with_the_seed_and_qualifies():
    nodes = tuple(f"v{i}" for i in range(12))
    rng = np.random.default_rng(5)
    weights = np.triu(rng.random((12, 12)) < 0.25, 1) * rng.uniform(0.5, 1.0, (12, 12))
    net = gaussian_networks.GaussianNetwork("toy", nodes, weights, np.zeros(12), np.ones(12))
    pairs = {
        (m.pairs[0].treatment, m.pairs[0].outcome)
        for m in (network_models.gaussian_model(s, net, pair="random") for s in range(8))
    }
    assert len(pairs) > 1
    with pytest.raises(ValueError, match="pair must be"):
        network_models.gaussian_model(0, net, pair="other")


def test_the_published_gaussian_networks_are_registered_with_pinned_files():
    assert set(gaussian_networks.NETWORKS) == {"magic-niab", "sangiovese"}
    for source in gaussian_networks.NETWORKS.values():
        assert len(source.sha256) == 64 and source.url.endswith(".rds")
    assert network_models.PAIRS["magic-niab"] == ("G266", "HT")
    assert network_models.PAIRS["sangiovese"] == ("pH", "GrapeW")


def test_a_discrete_network_replays_its_tables_with_hidden_variables_last(tmp_path, monkeypatch):
    folder = tmp_path / "networks"
    folder.mkdir()
    shutil.copy(ASIA, folder / "asia.bif.gz")
    monkeypatch.setenv(networks.CACHE_ENV, str(folder))
    network = networks.load_network("asia")
    model = network_models.table_model(
        1, network, ("smoke", "dysp"), ("asia",), setting="asia-test"
    )
    data = model.rows(300, M.DATA)
    assert model.discrete and model.indep_test == "chisq" and model.effects() == {}
    assert data.shape == (300, 7) and model.hidden == (7,) and model.meta["names"][7] == "asia"
    pair = model.pairs[0]
    assert (
        model.meta["names"][pair.treatment] == "smoke"
        and model.meta["names"][pair.outcome] == "dysp"
    )
    np.testing.assert_array_equal(data, model.rows(300, M.DATA))  # a function of the seed and n
    assert not np.array_equal(data, model.rows(300, M.FRESH))


@pytest.fixture
def fitted_networks(tmp_path, monkeypatch):
    """The R lane's reader on the real files: the cache is the environment's, or a download."""
    import os

    folder = Path(os.environ.get(networks.CACHE_ENV, "") or tmp_path / "networks")
    folder.mkdir(parents=True, exist_ok=True)
    # The reader keeps its JSON next to the download; a temporary copy leaves the cache as it was.
    work = tmp_path / "work"
    work.mkdir()
    for name in gaussian_networks.NETWORKS:
        if (folder / f"{name}.rds").is_file():
            shutil.copy(folder / f"{name}.rds", work / f"{name}.rds")
    monkeypatch.setenv(networks.CACHE_ENV, str(work))
    return work


@needs_r()
def test_magic_niab_read_in_r_has_the_structure_and_the_effect_the_papers_state(fitted_networks):
    try:
        net = gaussian_networks.load_gaussian_network("magic-niab")
    except OSError as error:
        pytest.skip(f"the MAGIC-NIAB file is unavailable: {error}")
    assert (net.d, int((net.weights != 0).sum())) == (44, 66)
    x, y = (net.index(v) for v in network_models.PAIRS["magic-niab"])
    # LDECC's own pickle of the network gives G266 -> HT a weight of -0.7827 and no other path
    assert net.weights[x, y] == pytest.approx(-0.7827, abs=1e-3)
    model = network_models.magic_niab(0)
    pair = model.pairs[0]
    cov = model.covariance()
    assert model.true_effect(pair.treatment, pair.outcome, "total") == pytest.approx(
        -0.7827 * np.sqrt(cov[pair.treatment, pair.treatment] / cov[pair.outcome, pair.outcome]),
        abs=1e-3,
    )
    kept = networks.cache_dir() / "magic-niab.structure.json"
    assert kept.is_file()
    again = gaussian_networks.load_gaussian_network("magic-niab")  # from the JSON, no R
    np.testing.assert_array_equal(again.weights, net.weights)


@needs_r()
def test_sangiovese_read_in_r_fixes_the_discrete_root_and_keeps_the_papers_pair(fitted_networks):
    try:
        net = gaussian_networks.load_gaussian_network("sangiovese")
    except OSError as error:
        pytest.skip(f"the SANGIOVESE file is unavailable: {error}")
    assert (net.d, int((net.weights != 0).sum())) == (14, 51) and "Treatment" not in net.nodes
    assert net.level.startswith("T")  # the most probable level of the 16
    x, y = (net.index(v) for v in network_models.PAIRS["sangiovese"])
    assert net.weights[x, y] != 0 and not net.weights[y].any()  # pH -> GrapeW, and GrapeW is a sink
    model = network_models.sangiovese(0)
    assert model.pairs[0].estimand == "direct" and model.p == 14


# --- the adapters ----------------------------------------------------------------------------


def test_the_baselines_answer_with_the_empty_set_and_with_every_other_column():
    nothing, everything = (
        baseline_adjustment.AdjustForNothing(),
        baseline_adjustment.AdjustForEverything(),
    )
    assert isinstance(nothing, L.PairAdapter) and isinstance(everything, SolutionAdapter)
    assert nothing.name != everything.name and nothing.name != "baseline.empty"
    data = np.zeros((5, 4))
    params = {"treatment": 1, "outcome": 3, "estimand": "total"}
    (a,), _ = L.unpack_effects(nothing.fit(data, params))
    (b,), _ = L.unpack_effects(everything.fit(data, params))
    assert a["adjustment_sets"] == [[]] and b["adjustment_sets"] == [[0, 2]]
    assert a["set_kind"] == "valid" and a["nonzero"] is None and a["identifiable"] is None
    (c,), _ = L.unpack_effects(everything.fit(data, {**params, "estimand": "direct"}))
    assert c["estimand"] == "direct" and c["set_kind"] == "parents_of_outcome"
    assert nothing.to_structure(nothing.fit(data, params)).n_nodes == 4


def test_the_table_of_methods_names_the_seven_and_what_each_answers():
    assert set(andrey_adjustment.METHODS) == {
        "covariate_rules",
        "ldecc",
        "ldp",
        "ld3",
        "snap",
        "local_optimal_adjustment",
        "locpc_cde",
    }
    for name, row in andrey_adjustment.METHODS.items():
        assert row.estimand in L.ESTIMANDS and row.identified_from in ("cpdag", "dag")
        assert row.set_kind in (*L.SET_KINDS, L.OPTIMAL_OR_LOCAL), name
        assert 0 < row.alpha < 1
    unordered = {n for n, r in andrey_adjustment.METHODS.items() if r.unordered}
    assert unordered == {"snap", "local_optimal_adjustment"}
    with pytest.raises(ValueError, match="unknown method"):
        andrey_adjustment.AndreyAdjustment("pc")


@pytest.mark.parametrize("method", andrey_adjustment.available())
def test_the_table_says_what_the_facade_offers(method):
    """A row of the table, checked against the facade once the method exists: the level is the
    facade's default and the pair arguments are the ones the adapter passes."""
    import inspect

    import andrey

    row = andrey_adjustment.METHODS[method]
    parameters = inspect.signature(getattr(andrey, method)).parameters
    assert parameters["alpha"].default == row.alpha
    assert ("variables" in parameters) is row.unordered
    assert ("treatment" in parameters) is not row.unordered


def test_the_adapter_reads_the_records_of_a_method_and_passes_its_settings(monkeypatch):
    import andrey

    seen = {}

    def fake(data, **kwargs):
        seen.update(kwargs)
        record = core_effects.effect_record(
            kwargs["treatment"],
            kwargs["outcome"],
            estimand="total",
            nonzero=True,
            identifiable=True,
            adjustment_sets=[[2]],
            set_kind="valid",
        )
        return SimpleNamespace(metadata={"effects": [record], "n_tests": 11})

    monkeypatch.setattr(andrey, "ldp", fake, raising=False)
    adapter = andrey_adjustment.AndreyAdjustment("ldp")
    assert adapter.name == "andrey.ldp" and isinstance(adapter, L.PairAdapter)
    assert adapter.params() == {"alpha": 0.005, "indep_test": "fisherz"}
    params = {**adapter.params(), "treatment": 0, "outcome": 1, "max_depth": 3}
    records, tests = L.unpack_effects(adapter.fit(np.zeros((9, 4)), params))
    assert seen == {
        "treatment": 0,
        "outcome": 1,
        "alpha": 0.005,
        "indep_test": "fisherz",
        "max_depth": 3,
    }
    assert tests == 11 and records[0]["adjustment_sets"] == [[2]]


def test_the_pc_arm_reads_the_optimal_set_where_the_class_identifies_the_effect():
    rng = np.random.default_rng(0)
    n = 20_000
    c = rng.standard_normal(n)
    x = c + rng.standard_normal(n)
    y = x + 2 * c + rng.standard_normal(n)
    z = y + rng.standard_normal(n)  # a descendant of the outcome
    data = np.column_stack([x, y, c, z])
    arm = andrey_adjustment.AndreyPCAdjustment()
    assert isinstance(arm, L.PairAdapter) and isinstance(arm, SolutionAdapter)
    (record,), tests = L.unpack_effects(
        arm.fit(data, {**arm.params(), "treatment": 0, "outcome": 1})
    )
    assert tests is None or tests > 0
    assert record["identifiable"] is True and record["adjustment_sets"] == [[2]]
    assert record["set_kind"] == "optimal" and record["nonzero"] is True


def test_the_pc_arm_reads_a_direct_effect_off_the_edges_at_the_outcome():
    rng = np.random.default_rng(1)
    n = 20_000
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    y = x + z + rng.standard_normal(n)  # a collider at y: every edge at y is oriented
    w = rng.standard_normal(n)  # unrelated
    arm = andrey_adjustment.AndreyPCAdjustment()
    data = np.column_stack([x, y, z, w])
    params = {**arm.params(), "treatment": 0, "outcome": 1, "estimand": "direct"}
    (record,), _ = L.unpack_effects(arm.fit(data, params))
    assert record["estimand"] == "direct" and record["identifiable"] is True
    assert record["nonzero"] is True and record["adjustment_sets"] == [[2]]
    assert record["set_kind"] == "parents_of_outcome"
    (apart,), _ = L.unpack_effects(arm.fit(data, {**params, "treatment": 3}))
    assert apart["nonzero"] is False and apart["adjustment_sets"] == []  # not adjacent


# --- the driver ------------------------------------------------------------------------------


class StandIn:
    """A solution that fills the contract: adjusts for every column that is no descendant of the
    treatment by the data (a covariate marginally uncorrelated with the treatment is left out)."""

    name = "stand-in.pairs"
    package = "test"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    algorithm = "stand_in"
    output_type = "effects"
    estimand = "total"
    set_kind = "valid"
    identified_from = "dag"
    unordered = False

    def params(self):
        return {"alpha": 0.01, "indep_test": "fisherz"}

    def fit(self, data, params):
        x, y = params["treatment"], params["outcome"]
        corr = np.corrcoef(data.T)
        keep = [v for v in range(data.shape[1]) if v not in (x, y) and abs(corr[v, x]) > 0.05]
        record = {
            "treatment": x,
            "outcome": y,
            "estimand": "total",
            "nonzero": True,
            "identifiable": True,
            "adjustment_sets": [keep],
            "set_kind": "valid",
        }
        return L.pack_effects([record], data.shape[1], 42)

    def to_structure(self, native):
        return L.empty_structure(native)


class Perfect(StandIn):
    """Answers with the oracle's records, found from the model the driver is running."""

    name = "stand-in.perfect"
    models: dict = {}

    def fit(self, data, params):
        model = self.models[data.tobytes()]
        pair = model.pairs[0]
        (record,) = L.oracle_records(
            model.dag,
            model.hidden,
            (pair.treatment, pair.outcome),
            estimand="total",
            set_kind="valid",
            identified_from="dag",
        )
        return L.pack_effects([record], data.shape[1], None)


def _perfect_models(monkeypatch):
    models = {}
    for text in ("entner-effect-admissible@300", "ldecc@300"):
        cell = M.parse_cell(text)
        for seed in (0, 1):
            model = cell.model(seed)
            models[model.rows(cell.n, M.DATA).tobytes()] = model
    monkeypatch.setattr(Perfect, "models", models)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """The driver on two cells, its fits run in this process, with the stand-ins as the field."""
    _perfect_models(monkeypatch)
    calls = []

    def fit_here(adapter, data, params, **kwargs):
        calls.append((adapter.name, dict(params)))
        return [{"status": "ok", "adj": adapter.fit(data, params), "wall_s": 0.01, "cpu_s": 0.01}]

    monkeypatch.setattr(adjustment_cpu, "run_task", fit_here)
    monkeypatch.setattr(adjustment_cpu, "check_interpreters", lambda field: {})
    monkeypatch.setattr(adjustment_cpu, "andrey_arms", lambda: [])
    monkeypatch.setattr(adjustment_cpu, "FIXED_ARMS", [baseline_adjustment.AdjustForNothing()])
    monkeypatch.setattr(adjustment_cpu, "REFERENCES", [StandIn(), Perfect()])
    flags = [
        "--out",
        str(tmp_path / "run"),
        "--cells",
        "entner-effect-admissible@300",
        "ldecc@300",
        "--seeds",
        "2",
    ]
    return tmp_path / "run", flags, calls


def test_each_cell_and_seed_is_fitted_once_per_pair_and_solution_that_runs_on_it(lane):
    out, flags, calls = lane
    assert adjustment_cpu.main(flags) == 0
    records = _records(out)
    assert len(records) == len(calls) == 2 * 2 * 3  # two cells, two seeds, three solutions
    assert {(r["cell"], r["seed"]) for r in records} == {
        (c, s) for c in ("entner-effect-admissible@300", "ldecc@300") for s in (0, 1)
    }
    assert all(r["status"] == "ok" and r["pair"] == 0 and r["answer"] for r in records)
    pair = {a: p for a, p in calls}["stand-in.pairs"]
    assert set(pair) >= {"treatment", "outcome", "estimand", "alpha", "indep_test"}
    assert pair["indep_test"] == "fisherz" and pair["estimand"] == "total"


def test_a_perfect_answer_is_all_correct_and_an_unadjusted_one_has_the_larger_error(
    lane, monkeypatch
):
    out, flags, _ = lane
    adjustment_cpu.main(flags)
    records = _records(out)
    for r in records:
        if r["solution"] == "stand-in.perfect":
            answer = r["scores"]["answer"]
            assert answer["adjustment"]["wrong"] == 0 and answer["identifiable"]["wrong"] == 0
            assert all(s["valid"] for s in r["scores"]["sets"])
            assert next(iter(r["scores"]["error"].values()))["error"] < 0.5
    empty = [
        r
        for r in records
        if r["solution"] == "baseline.adjust-none" and r["cell"].startswith("entner")
    ]
    perfect = [
        r for r in records if r["solution"] == "stand-in.perfect" and r["cell"].startswith("entner")
    ]

    def mean_error(rows):
        return np.mean([next(iter(r["scores"]["error"].values()))["error"] for r in rows])

    assert mean_error(empty) > mean_error(perfect)
    rows = adjustment_cpu.summarize(records)
    assert {r["solution"] for r in rows} == {
        "stand-in.pairs",
        "stand-in.perfect",
        "baseline.adjust-none",
    }
    assert all(0 <= r["answered"] <= 1 for r in rows)


def test_a_solution_runs_only_on_the_cells_that_ask_for_its_estimand_and_have_its_graph(lane):
    class Direct(StandIn):
        name = "stand-in.direct"
        estimand = "direct"

    class Classy(StandIn):
        name = "stand-in.cpdag"
        identified_from = "cpdag"

    hidden = M.parse_cell("entner-effect-admissible@100").model(0)
    plain = M.parse_cell("ldecc@100").model(0)
    locpc = M.parse_cell("locpc-identifiable@100,v=10").model(0)
    assert adjustment_cpu.runs_on(StandIn(), hidden) and adjustment_cpu.runs_on(StandIn(), plain)
    assert not adjustment_cpu.runs_on(Classy(), hidden) and adjustment_cpu.runs_on(Classy(), plain)
    assert not adjustment_cpu.runs_on(Direct(), plain) and adjustment_cpu.runs_on(Direct(), locpc)

    class GaussianOnly(StandIn):
        name = "stand-in.gaussian"
        indep_tests = ("fisherz",)
        takes_max_depth = True

    binary = M.parse_cell("locpc-identifiable@100,v=10,family=binary").model(0)
    assert adjustment_cpu.runs_on(GaussianOnly(), plain) and not adjustment_cpu.runs_on(
        GaussianOnly(), binary
    )
    assert adjustment_cpu.runs_on(Direct(), binary)  # no declaration: every test
    assert adjustment_cpu._fit_params(GaussianOnly(), None, 3)["max_depth"] == 3
    assert "max_depth" not in adjustment_cpu._fit_params(StandIn(), None, 3)
    assert adjustment_cpu.runs_on(baseline_adjustment.AdjustForNothing(), locpc)  # "any"
    kind = adjustment_cpu.kind_of(baseline_adjustment.AdjustForNothing(), "direct")
    assert kind["set_kind"] == "parents_of_outcome"


def test_a_method_that_does_not_know_the_order_answers_both_directions_and_extras_are_dropped():
    pair = M.Pair(2, 5, "total")
    assert adjustment_cpu.pair_params(StandIn(), pair) == {
        "treatment": 2,
        "outcome": 5,
        "estimand": "total",
    }

    class Unordered(StandIn):
        unordered = True

    assert adjustment_cpu.pair_params(Unordered(), pair)["variables"] == [2, 5]
    answer = [
        {"treatment": 2, "outcome": 5},
        {"treatment": 5, "outcome": 2},
        {"treatment": 1, "outcome": 3},
    ]
    kept, extra = adjustment_cpu.keep_asked(answer, pair, True)
    assert len(kept) == 2 and extra == 1
    assert adjustment_cpu.keep_asked(answer, pair, False) == ([answer[0]], 2)


def test_an_unordered_solution_is_scored_in_both_directions(tmp_path, monkeypatch):
    class Both(StandIn):
        name = "stand-in.both"
        unordered = True
        set_kind = "optimal+local_parents"
        identified_from = "cpdag"

        def fit(self, data, params):
            i, j = params["variables"]
            records = [
                {
                    "treatment": a,
                    "outcome": b,
                    "estimand": "total",
                    "nonzero": None,
                    "identifiable": None,
                    "adjustment_sets": [],
                    "set_kind": None,
                }
                for a, b in ((i, j), (j, i))
            ]
            return L.pack_effects(records, data.shape[1], 1)

    monkeypatch.setattr(
        adjustment_cpu,
        "run_task",
        lambda adapter, data, params, **kw: [
            {"status": "ok", "adj": adapter.fit(data, params), "wall_s": 0.1, "cpu_s": 0.1}
        ],
    )
    monkeypatch.setattr(adjustment_cpu, "check_interpreters", lambda field: {})
    monkeypatch.setattr(adjustment_cpu, "andrey_arms", lambda: [])
    monkeypatch.setattr(adjustment_cpu, "FIXED_ARMS", [])
    monkeypatch.setattr(adjustment_cpu, "REFERENCES", [Both()])
    out = tmp_path / "run"
    cell = "random-graph@300,nodes=30"
    adjustment_cpu.main(["--out", str(out), "--cells", cell, "--seeds", "1"])
    (r,) = _records(out)
    assert len(r["answer"]) == 2 and r["answer_kind"]["unordered"] is True
    assert r["scores"]["answer"]["nonzero"]["abstained"] == 0
    assert r["scores"]["answer"]["nonzero"]["queries"] == 2  # both directions are queries
    assert set(r["scores"]["error"].values()) == {None}  # abstentions have no error


def test_a_finished_run_is_not_overwritten_and_resumes_without_new_fits(lane):
    out, flags, calls = lane
    adjustment_cpu.main(flags)
    done = len(calls)
    with pytest.raises(SystemExit, match="already holds a run"):
        adjustment_cpu.main(flags)
    assert adjustment_cpu.main(flags + ["--resume"]) == 0
    assert len(calls) == done and len(_records(out)) == done


def test_parts_split_the_models_and_the_solution_filter_and_alpha_reach_the_fits(lane):
    out, flags, calls = lane
    adjustment_cpu.main(
        flags + ["--part", "0", "--parts", "2", "--solutions", "stand-in.pairs", "--alpha", "0.02"]
    )
    assert len(calls) == 2  # two of the four (cell, seed) models, one solution
    assert {p["alpha"] for _, p in calls} == {0.02}
    assert {r["solution"] for r in _records(out)} == {"stand-in.pairs"}


def test_a_run_is_rescored_from_its_stored_answers(lane):
    out, flags, _ = lane
    adjustment_cpu.main(flags)
    damaged = _records(out)
    for r in damaged:
        r["scores"] = {"answer": None, "sets": [], "error": {}}
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in damaged))
    assert adjustment_cpu.main(["--out", str(out), "--rescore"]) == 0
    assert all(r["scores"]["answer"]["adjustment"]["queries"] == 1 for r in _records(out))
    meta = json.loads((out / "run_meta.json").read_text())
    (out / "run_meta.json").write_text(json.dumps({**meta, "sampler": "an older generator"}))
    with pytest.raises(SystemExit, match="rescore it with the build"):
        adjustment_cpu.main(["--out", str(out), "--rescore"])


def test_an_unknown_cell_or_solution_is_refused(lane):
    out, _, _ = lane
    with pytest.raises(SystemExit):
        adjustment_cpu.main(["--out", str(out), "--cells", "nonsense@10"])
    with pytest.raises(SystemExit):
        adjustment_cpu.main(["--out", str(out), "--solutions", "no.such"])


def test_the_default_cells_and_the_smoke_cells_are_cells():
    for text in (*adjustment_cpu.DEFAULT_CELLS, *adjustment_cpu.SMOKE["cells"]):
        assert M.parse_cell(text).key == text
    assert sum(c.startswith("entner") for c in adjustment_cpu.DEFAULT_CELLS) == 12


def _bench_python_ready() -> bool:
    try:
        runner._bench_python()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _bench_python_ready(), reason="the harness environment is absent")
def test_the_driver_runs_the_baselines_and_the_pc_arm_through_the_real_runner(tmp_path):
    out = tmp_path / "run"
    flags = [
        "--out",
        str(out),
        "--cells",
        "ldecc@2000",
        "--seeds",
        "1",
        "--cores",
        "1",
        "--solutions",
        "baseline.adjust-none,baseline.adjust-all,andrey.pc.adjustment",
    ]
    assert adjustment_cpu.main(flags) == 0
    records = _records(out)
    assert {r["solution"] for r in records} == {
        "baseline.adjust-none",
        "baseline.adjust-all",
        "andrey.pc.adjustment",
    }
    assert all(r["status"] == "ok" and r["scores"]["answer"] for r in records)
