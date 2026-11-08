"""Recovery checks for an environment index in the score searches and their FCI hybrids.

With ``env``, BOSS, GRaSP, GES, hill climbing, Tabu search, BOSS-FCI, GRaSP-FCI, and GFCI add the
environment index as a variable no data variable causes. Data from two environments, the second
shifting the means of some variables, check the fixed cases a shift identifies. On random models,
the score searches recover the interventional essential graph of the shifted variables (Hauser and
Buhlmann 2012), built without the code under test, and the hybrids come near the PAG FCI-JCI
reaches with a d-separation oracle. Sachs's nine conditions check that the index removes the
adjacencies pooling adds. Without ``env`` each search is unchanged, which its recorded outputs
check.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import Knowledge, data
from andrey.core import ARROW, NULL, TAIL, GraphStructure
from andrey.core.orient import cpdag_of_pdag, dag2cpdag
from andrey.data.interventions import soft
from andrey.metrics import shd, skeleton_scores
from andrey.search import _parallel_ges
from andrey.search import boss as boss_search
from andrey.search import hc as hc_search
from tests.recovery.helpers.essential import essential
from tests.recovery.helpers.jci import oracle_fit, random_dag, sample_pooled

pytestmark = pytest.mark.filterwarnings(
    "ignore::andrey.ExperimentalWarning", "ignore::andrey.DataTypeWarning"
)

SEARCHES = ("boss", "grasp", "ges", "hc", "tabu")
HYBRIDS = ("boss_fci", "grasp_fci", "gfci")
ALL = SEARCHES + HYBRIDS


def fit(name: str, *args, **settings):
    return getattr(andrey, name)(*args, **settings)


def restarted(names: tuple[str, ...]) -> list:
    """``(name, settings)`` parameters for a check of what the index identifies, with hill climbing
    at 20 starts: one climb can direct a chain against the index and keep an edge from the index to
    each variable on it, which no single move removes. The one-start row is
    ``test_one_hill_climb_finds_every_shifted_variable``."""
    return [
        pytest.param(name, {"n_starts": 20}, id="hc-20")
        if name == "hc"
        else pytest.param(name, {}, id=name)
        for name in names
    ]


def scm(nodes: tuple[str, ...], arcs: tuple[tuple[str, str], ...]) -> data.SCM:
    """A linear-Gaussian model on a fixed graph."""
    return data.SCM(
        graph=data.graphs.FixedDAG(nodes=nodes, arcs=arcs),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )


def shifted(seed: int, d: int = 12):
    """A linear-Gaussian model on a random graph with ``d`` edges, from two environments of 1,000
    rows, the second shifting the means of a third of the variables."""
    model = data.SCM(
        graph=data.graphs.erdos_renyi(d=d, avg_degree=2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    targets = np.random.default_rng([seed, 602]).choice(d, size=d // 3, replace=False)
    return model.sample(
        n=[1000, 1000], seed=seed, envs=[{}, {int(j): soft(shift=1.0) for j in targets}]
    )


CHAIN = scm(("x0", "x1", "x2", "x3"), (("x0", "x1"), ("x1", "x2"), ("x2", "x3")))


def chain():
    """The chain ``x0 -> x1 -> x2 -> x3`` from two environments, the second shifting ``x0``."""
    ds = CHAIN.sample(n=[1000, 1000], seed=0, envs=[{}, {0: soft(shift=2.0)}], shuffle=False)
    return ds.data, ds.env


# --- fixed cases ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "settings"), restarted(SEARCHES))
def test_env_orients_a_chain_from_its_shifted_root(name, settings):
    """The chain's CPDAG leaves every edge undirected; the index points into ``x0`` alone and
    directs the chain."""
    x, env = chain()
    assert {kind for *_, kind in fit(name, x).structure.oriented_edges()} == {"undirected"}
    out = fit(name, x, env, **settings)
    assert out.structure.oriented_edges() == [
        (0, 1, "directed"),
        (1, 2, "directed"),
        (2, 3, "directed"),
    ]
    assert out.metadata["index_edges"] == [[0, "directed"]]
    assert out.metadata["changing_variables"] == [0]


@pytest.mark.parametrize("name", ALL)
def test_env_removes_the_edge_pooling_adds(name):
    """Two independent variables the second environment shifts together depend in the pooled rows,
    and not given the index."""
    ds = scm(("x0", "x1"), ()).sample(
        n=[1000, 1000], seed=0, envs=[{}, {0: soft(shift=2.0), 1: soft(shift=2.0)}], shuffle=False
    )
    assert fit(name, ds.data).structure.oriented_edges() != []
    out = fit(name, ds.data, ds.env)
    assert out.structure.oriented_edges() == []
    assert [variable for variable, _ in out.metadata["index_edges"]] == [0, 1]


@pytest.mark.parametrize("name", ALL)
def test_one_environment_gives_the_search_alone(name):
    x, _ = chain()
    alone = fit(name, x)
    out = fit(name, x, np.ones(len(x)))
    assert out.structure == alone.structure
    assert out.metadata["index_edges"] == []
    assert out.metadata.get("changing_variables", []) == []
    assert "index_edges" not in alone.metadata and "changing_variables" not in alone.metadata


@pytest.mark.parametrize("name", ALL)
def test_one_environment_warns_that_a_mixed_data_setting_runs_linear_gaussian(name):
    """One environment adds no index column, so with no categorical column a mixed-data setting
    runs as its linear-Gaussian form and says so, as it does without ``env``."""
    x, _ = chain()
    settings = {"score_func": "cond_gauss", "categorical": []}
    if name in HYBRIDS:
        settings["indep_test"] = "cond_gauss"
    with pytest.warns(andrey.DataTypeWarning, match="linear-Gaussian"):
        fit(name, x, np.ones(len(x)), **settings)


@pytest.mark.parametrize("name", ALL)
def test_bic_reads_a_numeric_env_as_a_number(name):
    """Three environments coded 0, 1, 2, the middle one shifting ``x0``: the shift does not follow
    the codes, so the linear score and test miss it, and the mixed-data ones, which read each code
    as a category, find it."""
    ds = scm(("x0", "x1"), (("x0", "x1"),)).sample(
        n=[1000] * 3, seed=0, envs=[{}, {0: soft(shift=2.0)}, {}], shuffle=False
    )
    assert fit(name, ds.data, ds.env).metadata["index_edges"] == []
    mixed = {"score_func": "cond_gauss", "categorical": []}
    if name in HYBRIDS:
        mixed["indep_test"] = "cond_gauss"
    found = fit(name, ds.data, ds.env, **mixed).metadata["index_edges"]
    assert [variable for variable, _ in found] == [0]


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_marks_a_hidden_common_cause_the_index_reveals(name):
    """``env -> x0 <- L -> x1 <- x2`` with ``L`` hidden: ``x0 <-> x1``, which the index tells
    apart, and the index edge's mark at the index follows ``confounded_index``."""
    dag = nx.DiGraph([(3, 0), (4, 0), (4, 1), (2, 1)])
    x, env = sample_pooled(dag, 3, 2000, seed=0)
    out = fit(name, x, env)
    assert out.structure.oriented_edges() == [(0, 1, "bidirected"), (2, 1, "partially_directed")]
    assert out.metadata["index_edges"] == [[0, "partially_directed"]]
    randomized = fit(name, x, env, confounded_index=False)
    assert randomized.metadata["index_edges"] == [[0, "directed"]]


# --- ground truth ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "settings"), restarted(SEARCHES))
def test_search_recovers_the_interventional_essential_graph(name, settings):
    """On 10 random models, the graph equals the essential graph of the observational environment
    and the shifted one, and ``changing_variables`` the shifted variables, on most draws."""
    exact, found = 0, 0
    for seed in range(10):
        ds = shifted(seed)
        out = fit(name, ds.data, ds.env, **settings)
        truth = essential(ds.graph.to_numpy(), [ds.targets[1]])
        exact += shd(out.structure, truth) == 0
        found += out.metadata["changing_variables"] == sorted(ds.targets[1])
        assert out.metadata["changing_variables"] == [v for v, _ in out.metadata["index_edges"]]
    # Recorded: 9 of 10 and 10 of 10 for GES, Tabu search, and hill climbing with 20 starts, 9 and 9
    # for the rest.
    assert exact >= 8 and found >= 8


def test_one_hill_climb_finds_every_shifted_variable():
    """At one start, on the models above, ``changing_variables`` holds every shifted variable, and
    the graph is closer to the essential graph than the one without ``env``."""
    with_env, without = 0, 0
    for seed in range(10):
        ds = shifted(seed)
        out = andrey.hc(ds.data, ds.env)
        truth = essential(ds.graph.to_numpy(), [ds.targets[1]])
        assert set(ds.targets[1]) <= set(out.metadata["changing_variables"])
        with_env += shd(out.structure, truth)
        without += shd(andrey.hc(ds.data).structure, truth)
    # Recorded: 64 against 137. The graph equals the essential graph on 1 of 10 models and
    # ``changing_variables`` the shifted variables on 3, since one climb can keep the index's edges
    # to a chain it directs against the index.
    assert with_env < without / 1.5


@pytest.mark.parametrize("confounded", [True, False])
@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_comes_near_the_oracle_jci_pag(name, confounded, monkeypatch):
    """12 variables and 2 hidden ones from two environments, six models: endpoint errors against
    the PAG FCI-JCI reaches with a d-separation oracle, summed, below half the empty graph's."""
    d, errors, empty_errors = 12, 0, 0
    empty = GraphStructure.from_numpy(np.zeros((d, d), dtype=np.int8), kind="pag")
    for seed in range(6):
        dag = random_dag(seed, d, 2)
        x, env = sample_pooled(dag, d, 2000, seed=seed)
        oracle, _ = oracle_fit(dag, d, monkeypatch, confounded_index=confounded)
        out = fit(name, x, env, confounded_index=confounded)
        errors += shd(out.structure, oracle, endpoint_aware=True)
        empty_errors += shd(empty, oracle, endpoint_aware=True)
    # Recorded: 46 or 44 against 138, for each hybrid.
    assert errors < empty_errors / 2


# --- real data ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_the_sachs_conditions_env_removes_the_adjacencies_pooling_adds(name, sachs_conditions):
    """Pooled across the nine conditions, under half of each search's adjacencies are in the
    consensus network; with the condition as a categorical ``env``, over 85% are, and the index
    keeps an edge to every protein, since each changes across the conditions."""
    frame, env, consensus = sachs_conditions
    assert skeleton_scores(fit(name, frame).structure, consensus)["precision"] < 0.5
    settings = {"score_func": "cond_gauss", "categorical": []}
    if name in HYBRIDS:
        settings["indep_test"] = "cond_gauss"
    out = fit(name, frame, env, **settings)
    assert skeleton_scores(out.structure, consensus)["precision"] > 0.85
    assert [variable for variable, _ in out.metadata["index_edges"]] == list(frame.columns)


# --- contract -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_env_fits_repeat_and_follow_the_columns(name):
    """A refit gives the same graph, and reordering the columns reorders the graph's nodes and the
    index's edges and changes nothing else."""
    ds = shifted(5, d=8)
    first = fit(name, ds.data, ds.env)
    first.structure.validate()
    assert fit(name, ds.data, ds.env).structure == first.structure
    order = np.random.default_rng(5).permutation(8)
    back = np.argsort(order)
    permuted = fit(name, ds.data[:, order], ds.env)
    assert np.array_equal(
        permuted.structure.to_numpy()[np.ix_(back, back)], first.structure.to_numpy()
    )
    relabeled = sorted([int(order[j]), kind] for j, kind in permuted.metadata["index_edges"])
    assert relabeled == first.metadata["index_edges"]


@pytest.mark.parametrize(("name", "settings"), restarted(ALL))
def test_env_takes_a_column_of_text_labels(name, settings):
    """A column of text labels leaves the variables and fits what its codes fit, with the edges
    named by column."""
    x, env = chain()
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    by_codes = fit(name, frame, env, **settings)
    site = frame.assign(site=np.where(env == 1, "treated", "control"))
    out = fit(name, site, "site", **settings)
    assert out.structure == by_codes.structure
    assert out.structure.labels == ("a", "b", "c", "d")
    assert [variable for variable, _ in out.metadata["index_edges"]] == ["a"]


@pytest.mark.parametrize("name", ALL)
def test_a_fractional_env_is_refused_under_a_mixed_data_setting(name):
    x, env = chain()
    settings = {"score_func": "cond_gauss", "categorical": []}
    if name in HYBRIDS:
        settings["indep_test"] = "cond_gauss"
    with pytest.raises(ValueError, match="reads env as categories coded by whole numbers"):
        fit(name, x, env + 0.5, **settings)


@pytest.mark.parametrize(("name", "settings"), restarted(ALL))
def test_knowledge_holds_with_env(name, settings):
    """A required edge is directed, a forbidden pair is not adjacent, and the index's edges reach
    only shifted variables: all of them for BOSS and GRaSP, each edge pointing out of the index."""
    ds = shifted(0)
    dag = ds.graph.to_numpy()
    edges = [(int(u), int(v)) for u, v in zip(*np.nonzero((dag == TAIL) & (dag.T == ARROW)))]
    (a, b), (u, v) = edges[0], edges[1]
    knowledge = Knowledge(required=[(u, v)], forbidden=[(a, b), (b, a)])
    out = fit(name, ds.data, ds.env, knowledge=knowledge, **settings)
    marks = out.structure.to_numpy()
    assert marks[u, v] == TAIL and marks[v, u] == ARROW
    assert marks[a, b] == NULL
    variables = [variable for variable, _ in out.metadata["index_edges"]]
    assert variables and set(variables) <= set(ds.targets[1])
    if name in SEARCHES:
        assert variables == sorted(ds.targets[1])
        assert {kind for _, kind in out.metadata["index_edges"]} == {"directed"}


@pytest.mark.parametrize("name", ALL)
def test_a_required_edge_against_the_shift_holds(name):
    """``x0 -> x1`` with ``x0`` shifted, and knowledge requiring ``x1 -> x0``: no DAG of the class
    of ``env -> x0 -> x1`` keeps both the index without a parent and the required edge, so the
    search settles on another class, and the result directs ``x1 -> x0``. A score search keeps an
    edge from the index into ``x1`` too, since ``x1 -> x0 <- env`` alone would make ``x1``
    independent of the index; a hybrid's tests drop it, since ``x1`` is independent of the index
    given ``x0``."""
    ds = scm(("x0", "x1"), (("x0", "x1"),)).sample(
        n=[1000, 1000], seed=0, envs=[{}, {0: soft(shift=2.0)}], shuffle=False
    )
    out = fit(name, ds.data, ds.env, knowledge=Knowledge(required=[(1, 0)]))
    assert out.structure.oriented_edges() == [(1, 0, "directed")]
    if name in HYBRIDS:
        assert out.metadata["index_edges"] == [[0, "partially_directed"]]
    else:
        assert out.metadata["index_edges"] == [[0, "directed"], [1, "directed"]]


@pytest.mark.parametrize("name", ("boss", "grasp", "boss_fci", "grasp_fci"))
def test_the_search_never_gives_the_index_a_parent(name, monkeypatch):
    """The DAG each search settles on, before the edges at the index are directed out of it, has no
    edge into the index: the index leads every order the search tries."""
    dags = []

    def settled(dag):
        dags.append(dag.copy())
        return dag2cpdag(dag)

    monkeypatch.setattr(boss_search, "dag2cpdag", settled)
    for seed in range(10):
        ds = shifted(seed)
        fit(name, ds.data, ds.env)
    assert len(dags) == 10 and all(dag.shape == (13, 13) for dag in dags)
    # The last row holds the marks at the index's end of its edges: an arrowhead there is a parent.
    assert not any((dag[-1] == ARROW).any() for dag in dags)


@pytest.mark.parametrize("name", ("hc", "tabu"))
def test_a_dag_search_never_gives_the_index_a_parent(name, monkeypatch):
    """The DAG hill climbing and Tabu search settle on, before the edges at the index are directed
    out of it, has no edge into the index: no move adds one."""
    dags = []

    def settled(dag):
        dags.append(dag.copy())
        return dag2cpdag(dag)

    monkeypatch.setattr(hc_search, "dag2cpdag", settled)
    for seed in range(10):
        ds = shifted(seed)
        fit(name, ds.data, ds.env, n_starts=1)
    assert len(dags) == 10 and all(dag.shape == (13, 13) for dag in dags)
    assert not any((dag[-1] == ARROW).any() for dag in dags)


@pytest.mark.parametrize("name", ("ges", "gfci"))
def test_every_ges_move_keeps_the_index_a_source(name, monkeypatch):
    """On 30 random models of 5 to 20 variables, each graph a GES move reaches, before it is taken
    to its class, directs every edge at the index out of it, and has a consistent extension: the
    rebuild raises on a graph without one, and no fit raises."""
    moved = []

    def rebuilt(pdag):
        moved.append(pdag.copy())
        return cpdag_of_pdag(pdag)

    monkeypatch.setattr(_parallel_ges, "cpdag_of_pdag", rebuilt)
    for seed in range(30):
        ds = shifted(seed, d=(5, 10, 20)[seed % 3])
        fit(name, ds.data, ds.env)
    assert len(moved) > 100
    for pdag in moved:
        index = pdag.shape[0] - 1
        neighbors = np.flatnonzero(pdag[index] != NULL)
        assert (pdag[index, neighbors] == TAIL).all() and (pdag[neighbors, index] == ARROW).all()
