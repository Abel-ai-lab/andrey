"""``andrey.bootstrap`` and ``BootstrapOutput``: the counts, the averaged graph, and the refits."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import BootstrapOutput, bootstrap
from andrey.constraint.fci import _fci_orient
from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput, TemporalStructure
from andrey.core._convert import _canonical_dag, _observed_ancestors_of, mag_of_pag
from andrey.core.orient import _cpdag_of_pdag, dag2cpdag, pdag2dag
from andrey.data.latent import DSepOracle, _AncestorSepsets, _observed_ancestors
from andrey.resample import EDGES, _default_block_length, _draw, _Job
from andrey.spec import REGISTRY


def with_workers(workers, *args, **kwargs):
    """``bootstrap`` run on ``workers`` worker processes, set the way every method sets them."""
    with andrey.config(n_jobs=workers):
        return bootstrap(*args, **kwargs)


def graph(n, edges, kind=None):
    """A graph on ``n`` nodes from ``(i, j, mark at i, mark at j)`` rows."""
    M = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        M[i, j], M[j, i] = mi, mj
    return StructureOutput.new(GraphStructure.from_numpy(M, kind=kind))


def directed(i, j):
    return (i, j, TAIL, ARROW)


def undirected(i, j):
    return (i, j, TAIL, TAIL)


@pytest.fixture(scope="module")
def counted():
    """Ten fixed fits over four nodes; the counts in the comments are out of ten."""
    fits = []
    for k in range(10):
        edges = []
        edges += [directed(0, 1)] if k < 6 else [directed(1, 0)] if k < 8 else []  # 6 / 2 / 0
        edges += [undirected(1, 2)] if k < 3 else [directed(1, 2)] if k == 3 else []  # 3 / 1 / 0
        edges += [directed(2, 3)] if k < 5 else [directed(3, 2)]  # 5 / 5, a tie
        fits.append(graph(4, edges, "cpdag" if k < 3 else "dag"))
    return fits


def canonical(rows):
    """Table rows with a symmetric edge's ends unordered: reordering the columns reorders them."""
    undirected = ("undirected", "bidirected", "circle")
    return sorted(
        (
            r["type"],
            tuple(sorted((r["source"], r["target"])))
            if r["type"] in undirected
            else (r["source"], r["target"]),
            r["count"],
        )
        for r in rows
    )


def canonical_edges(structure):
    return canonical(
        [
            {"source": s, "target": t, "type": k, "count": 0}
            for s, t, k in structure.oriented_edges()
        ]
    )


def edge_set(structure):
    return {(s, t, kind) for s, t, kind in structure.oriented_edges(index=True)}


def test_frequencies_are_shares_of_the_fits_with_each_edge(counted):
    out = BootstrapOutput.from_fits(counted)
    assert out.n_resamples == 10
    assert out.frequency("->")[0, 1] == 0.6 and out.frequency("<-")[1, 0] == 0.6
    assert out.frequency("<-")[0, 1] == 0.2 and out.frequency("->")[1, 0] == 0.2
    assert out.frequency("absent")[0, 1] == out.frequency("absent")[1, 0] == pytest.approx(0.2)
    assert out.frequency("--")[1, 2] == out.frequency("--")[2, 1] == 0.3
    assert out.frequency("->")[1, 2] == 0.1
    assert out.frequency("absent")[1, 2] == pytest.approx(0.6)
    assert out.frequency("->")[2, 3] == out.frequency("<-")[2, 3] == 0.5
    assert out.frequency("absent")[0, 3] == 1.0 and out.frequency("->")[0, 3] == 0.0
    # a pair holds one edge in each fit, so its shares add to 1
    total = sum(out.frequency(edge) for edge in (*EDGES, "absent"))
    assert np.allclose(total[~np.eye(4, dtype=bool)], 1.0)
    assert not out.frequency("o-o").any()


def test_the_table_lists_each_edge_that_appeared_most_frequent_first(counted):
    rows = BootstrapOutput.from_fits(counted).table()
    assert [(r["source"], r["target"], r["type"], r["count"]) for r in rows] == [
        (0, 1, "directed", 6),
        (2, 3, "directed", 5),
        (3, 2, "directed", 5),
        (1, 2, "undirected", 3),
        (1, 0, "directed", 2),
        (1, 2, "directed", 1),
    ]
    assert rows[0]["frequency"] == 0.6
    assert len(BootstrapOutput.from_fits(counted).table(min_frequency=0.5)) == 3


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.5, {(0, 1, "undirected"), (2, 3, "undirected")}),
        (0.7, {(0, 1, "undirected"), (2, 3, "undirected")}),
        (0.8, {(2, 3, "undirected")}),
        (1.0, {(2, 3, "undirected")}),
    ],
)
def test_the_threshold_decides_which_pairs_are_adjacent(counted, threshold, expected):
    # (0, 1) is adjacent in 8 fits of 10 and (2, 3) in all 10, so 1.0 keeps (2, 3).
    out = BootstrapOutput.from_fits(counted, threshold=threshold)
    assert edge_set(out.structure) == expected
    assert out.structure == out.average(threshold)
    assert out.threshold == threshold
    out.structure.validate()


def dags(*counts):
    """Fits over two nodes, ``counts`` of each of ``0 -> 1``, ``1 -> 0``, and no edge."""
    edges = ([directed(0, 1)], [directed(1, 0)], [])
    return [graph(2, edges[k], "dag") for k, c in enumerate(counts) for _ in range(c)]


def test_a_pair_takes_its_most_frequent_state_with_no_edge_one_of_them():
    # adjacent in 7 fits, and its most frequent edge outnumbers no edge: kept
    assert edge_set(BootstrapOutput.from_fits(dags(4, 3, 3)).structure) == {(0, 1, "directed")}
    # adjacent in 6 fits, but no edge (4) outnumbers each edge (3): dropped
    assert not edge_set(BootstrapOutput.from_fits(dags(3, 3, 4)).structure)
    # a tie between an edge and no edge drops the pair
    assert not edge_set(BootstrapOutput.from_fits(dags(4, 2, 4)).structure)


def test_a_tie_between_edges_takes_the_least_committal_one_then_a_graph_of_the_kind():
    fits = dags(4, 4, 2)
    # fits with no data to rank the nodes leave a dag's tie edge undirected, as a cpdag does
    tied = BootstrapOutput.from_fits(fits).structure
    assert tied.kind == "cpdag" and edge_set(tied) == {(0, 1, "undirected")}
    # the ranks of the data point it, and the reversed ranks point it the other way
    points = {}
    for ranks in (np.array([0, 1]), np.array([1, 0])):
        ranked = BootstrapOutput._count(fits, 0.5, False, False, None, None, ranks)
        assert ranked.structure.kind == "dag" and ranked.metadata["averaged_graph"] == "vote"
        points[tuple(ranks)] = edge_set(ranked.structure)
        assert edge_set(ranked.average(0.5)) == points[tuple(ranks)]
    forward, backward = points.values()
    assert len(forward) == len(backward) == 1 and forward != backward
    pags = [graph(2, [(0, 1, CIRCLE, ARROW)], "pag"), graph(2, [directed(0, 1)], "pag")]
    tied = BootstrapOutput.from_fits(pags)
    assert tied.structure.kind == "pag" and tied.structure.endpoints(0, 1) == (CIRCLE, CIRCLE)
    assert tied.frequency("o->")[0, 1] == tied.frequency("->")[0, 1] == 0.5


def random_fits(kind, seed, n=6, count=15):
    """``count`` fits of ``kind`` over ``n`` nodes, each the result of a method on a random DAG."""
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(count):
        dag = np.zeros((n, n), dtype=np.int8)
        for i, j in zip(*np.nonzero(np.triu(rng.random((n, n)) < 0.4, 1))):
            dag[i, j], dag[j, i] = TAIL, ARROW
        order = rng.permutation(n)
        g = GraphStructure.from_numpy(dag[np.ix_(order, order)], kind="dag")
        fits.append(StructureOutput.new(g.to_pag() if kind == "pag" else g))
    if kind == "cpdag":
        cpdags = (dag2cpdag(f.structure.to_numpy()) for f in fits)
        fits = [StructureOutput.new(GraphStructure.from_numpy(m, kind="cpdag")) for m in cpdags]
    return fits


@pytest.mark.parametrize("kind", ["dag", "cpdag", "pag"])
@pytest.mark.parametrize("seed", range(6))
def test_the_averaged_graph_is_a_valid_graph_of_the_fits_kind(kind, seed):
    ranks = np.random.default_rng(seed).permutation(6)  # a dag vote with a tie edge needs them
    out = BootstrapOutput._count(random_fits(kind, seed), 0.5, False, False, None, None, ranks)
    graph_ = out.structure
    marks = graph_.to_numpy()
    assert graph_.kind == kind
    if kind == "dag":
        assert np.array_equal(pdag2dag(marks), marks)  # acyclic, no undirected edge
    elif kind == "cpdag":
        assert np.array_equal(_cpdag_of_pdag(marks), marks)
    else:
        mag = GraphStructure.from_numpy(mag_of_pag(marks), kind="mag")
        assert mag.to_pag() == graph_


def cycle_fits():
    """Three dags, each two edges of the cycle ``0 -> 1 -> 2 -> 0``: the vote is the cycle."""
    pairs = ([directed(0, 1), directed(1, 2)], [directed(1, 2), directed(2, 0)])
    pairs += ([directed(2, 0), directed(0, 1)],)
    return [graph(3, edges, "dag") for edges in pairs]


def test_a_vote_with_no_valid_graph_returns_the_fit_nearest_the_others_with_a_warning(counted):
    fits = cycle_fits()
    with pytest.warns(andrey.AndreyWarning, match="nearest the others") as caught:
        out = BootstrapOutput.from_fits(fits)
    assert len(caught) == 1
    assert out.metadata["averaged_graph"] == "nearest_resample"
    assert out.structure == fits[0].structure  # every fit agrees as much; the first one wins
    assert BootstrapOutput.from_fits(counted).metadata["averaged_graph"] == "vote"
    with pytest.raises(ValueError, match="no valid dag"):
        BootstrapOutput.from_fits(fits, strict=True)
    with pytest.raises(ValueError, match="no valid dag"):
        BootstrapOutput.from_fits([*fits, *fits], strict=True).average(0.6)


@pytest.mark.parametrize("seed", range(4))
def test_the_nearest_fit_has_the_least_mean_shd_to_the_fits(seed):
    fits = random_fits("pag", seed, count=12)
    out = BootstrapOutput.from_fits(fits)
    marks = [f.structure.to_numpy() for f in fits]

    def shd(a, b):
        upper = np.triu(np.ones(a.shape, dtype=bool), 1)
        return int((((a != b) | (a.T != b.T)) & upper).sum())

    distance = [sum(shd(a, b) for b in marks) for a in marks]
    assert out._nearest is not None and out._nearest[0] == int(np.argmin(distance))


def test_a_lag_graph_counts_each_directed_edge_on_its_own():
    def series(*lag1):
        lag0 = GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8))
        M = np.zeros((2, 2), dtype=np.int8)
        for i, j in lag1:
            M[i, j] = ARROW if (i == j or M[j, i]) else TAIL
            M[j, i] = ARROW
        graphs = [lag0, GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)]
        return StructureOutput.new(TemporalStructure.from_lag_graphs(graphs, lags=[0, 1]))

    fits = [series((0, 1), (1, 0)), series((0, 1)), series((0, 1)), series((1, 1)), series()]
    out = BootstrapOutput.from_fits(fits, threshold=0.5)
    assert out.frequency("->", lag=1)[0, 1] == 0.6
    assert out.frequency("<-", lag=1)[1, 0] == 0.6 and out.frequency("->", lag=1)[1, 0] == 0.2
    assert out.frequency("absent", lag=1)[0, 1] == pytest.approx(0.4)
    assert out.frequency("->", lag=1)[1, 1] == 0.2 and not out.frequency("<->", lag=1).any()
    assert not out.frequency("->", lag=0).any()
    assert out.structure.lags == (0, 1)
    assert edge_set(out.structure.lag(1)) == {(0, 1, "directed")}  # in 3 fits of 5
    assert not edge_set(out.average(0.6).lag(1))  # 3 of 5 is not more than 0.6 of them
    assert out.metadata["averaged_graph"] == "vote"
    rows = out.table()
    assert [(r["lag"], r["source"], r["target"], r["count"]) for r in rows] == [
        (1, 0, 1, 3),
        (1, 1, 0, 1),
        (1, 1, 1, 1),
    ]
    with pytest.raises(ValueError, match="lag must be one of"):
        out.frequency("->")


def test_from_fits_rejects_what_cannot_be_counted(counted):
    with pytest.raises(ValueError, match="empty"):
        BootstrapOutput.from_fits([])
    with pytest.raises(TypeError, match="StructureOutput"):
        BootstrapOutput.from_fits([counted[0].structure])
    with pytest.raises(ValueError, match="differ"):
        BootstrapOutput.from_fits([counted[0], graph(3, [])])
    for threshold in (0, 0.3, 0.49, 1.01, -0.5):
        with pytest.raises(ValueError, match="threshold"):
            BootstrapOutput.from_fits(counted, threshold=threshold)
    with pytest.raises(TypeError, match="threshold"):
        BootstrapOutput.from_fits(counted, threshold=True)
    lagged = GraphStructure.from_numpy(
        np.array([[0, ARROW], [TAIL, 0]], dtype=np.int8), kind="digraph"
    )
    mixed = [counted[0], StructureOutput.new(GraphStructure.from_numpy(np.zeros((4, 4))))]
    assert BootstrapOutput.from_fits(mixed).n_resamples == 2
    other = [graph(2, [directed(0, 1)]), StructureOutput.new(lagged)]
    with pytest.raises(ValueError, match="digraph"):
        BootstrapOutput.from_fits(other)
    joined = np.array([[0, TAIL], [TAIL, 0]], dtype=np.int8)
    undirected = StructureOutput.new(GraphStructure.from_numpy(joined, kind="undirected"))
    with pytest.raises(ValueError, match="undirected"):
        BootstrapOutput.from_fits([graph(2, [directed(0, 1)]), undirected])
    assert BootstrapOutput.from_fits([undirected, undirected]).structure.kind == "undirected"


def test_the_labels_of_the_fits_name_the_nodes():
    frame = pd.DataFrame(
        np.random.default_rng(0).standard_normal((200, 3)), columns=["a", "b", "c"]
    )
    frame["b"] += frame["a"]
    out = bootstrap("hc", frame, n_resamples=3, seed=0)
    assert out.structure.labels == ("a", "b", "c")
    assert {row["source"] for row in out.table()} <= {"a", "b", "c"}
    assert {row["source"] for row in out.table(index=True)} <= {0, 1, 2}
    assert "a" in repr(out) and "3 fits" in repr(out)


# ---- the refits ----------------------------------------------------------------------------------


def lingam_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, n)
    b = 0.9 * a + 0.4 * rng.uniform(-1, 1, n)
    c = 0.9 * b + 0.4 * rng.uniform(-1, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": rng.uniform(-1, 1, n)})


def test_a_seed_gives_the_same_resamples_whatever_the_number_of_fits():
    data = lingam_data()
    short = bootstrap("direct_lingam", data, n_resamples=4, seed=7, keep_fits=True)
    longer = bootstrap("direct_lingam", data, n_resamples=6, seed=7, keep_fits=True)
    again = bootstrap(andrey.direct_lingam, data, n_resamples=4, seed=7, keep_fits=True)
    other = bootstrap("direct_lingam", data, n_resamples=4, seed=8, keep_fits=True)
    assert short.fits == again.fits == longer.fits[:4]
    assert short.table() == again.table()
    weights = [fit.weighted_adjacency for fit in short.fits]
    assert all(not np.array_equal(w, weights[0]) for w in weights[1:])  # the resamples differ
    assert any(not np.array_equal(f.weighted_adjacency, g.weighted_adjacency)
               for f, g in zip(short.fits, other.fits))  # fmt: skip
    assert short.metadata == {
        "method": "direct_lingam",
        "seed": 7,
        "tie_break": andrey.config.tie_break,
        "resampling": "subsample",
        "sample_fraction": 0.5,
        "block_length": None,
        "averaged_graph": "vote",
    }
    assert short.fits is not None and bootstrap("hc", data, n_resamples=2).fits is None


def test_the_former_n_boot_still_sets_the_number_of_resamples_with_a_warning():
    data = lingam_data()
    current = bootstrap("hc", data, n_resamples=3, seed=1)
    with pytest.warns(andrey.DeprecatedArgumentWarning, match="n_boot is now n_resamples"):
        former = bootstrap("hc", data, n_boot=3, seed=1)
    assert former.n_resamples == 3 and former.table() == current.table()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match="BootstrapOutput.n_boot is now"):
        assert former.n_boot == former.n_resamples
    with pytest.raises(ValueError, match="n_boot is the former name of n_resamples"):
        bootstrap("hc", data, n_boot=3, n_resamples=4)


def test_the_default_seed_follows_andrey_seed(monkeypatch):
    data = lingam_data()
    monkeypatch.setenv("ANDREY_SEED", "5")
    from_env = bootstrap("direct_lingam", data, n_resamples=3, keep_fits=True)
    explicit = bootstrap("direct_lingam", data, n_resamples=3, seed=5, keep_fits=True)
    assert from_env.fits == explicit.fits and from_env.metadata["seed"] == 5
    monkeypatch.delenv("ANDREY_SEED")
    assert bootstrap("hc", data, n_resamples=2).metadata["seed"] == 0


@pytest.mark.parametrize("method", ["pc", "direct_lingam", "boss"])
def test_workers_do_not_change_the_result(method):
    data = lingam_data()
    serial = with_workers(1, method, data, n_resamples=6, seed=3, keep_fits=True)
    pooled = with_workers(2, method, data, n_resamples=6, seed=3, keep_fits=True)
    assert serial.fits == pooled.fits and serial.table() == pooled.table()
    assert serial.structure == pooled.structure


@pytest.mark.parametrize(
    ("method", "settings"),
    [("pc", {"indep_test": "rcot"}), ("boss", {}), ("ica_lingam", {})],
)
@pytest.mark.filterwarnings("ignore::andrey.SampleSizeWarning")
def test_a_seeded_method_fits_each_resample_with_a_seed_derived_from_the_bootstraps(
    method, settings
):
    data = lingam_data()

    def seeds(**kwargs):
        out = bootstrap(method, data, n_resamples=3, keep_fits=True, **settings, **kwargs)
        return [fit.metadata["seed"] for fit in out.fits]

    derived = seeds(seed=1)
    assert len(set(derived)) == 3
    assert derived == seeds(seed=1)
    assert derived != seeds(seed=2)


def test_the_bootstrap_does_not_take_a_seed_the_method_has_renamed():
    with pytest.raises(ValueError, match="random_state cannot be set"):
        bootstrap("ica_lingam", lingam_data(), n_resamples=2, random_state=4)


@pytest.mark.parametrize("method", ["pc", "fci", "ges", "boss", "direct_lingam"])
def test_the_frequencies_do_not_depend_on_the_column_order(method):
    """Reordering the columns permutes the table and leaves the draws of rows unchanged."""
    rng = np.random.default_rng(1)
    n = 400
    x1, x2 = rng.uniform(-1, 1, (2, n))
    x3 = 0.9 * x1 + 0.9 * x2 + 0.3 * rng.uniform(-1, 1, n)  # a collider, so edges get oriented
    x4 = 0.9 * x3 + 0.3 * rng.uniform(-1, 1, n)
    x5 = rng.uniform(-1, 1, n)
    frame = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5))
    perm = [3, 0, 4, 2, 1]
    plain = bootstrap(method, frame, n_resamples=8, seed=2)
    shuffled = bootstrap(method, frame.iloc[:, perm], n_resamples=8, seed=2)
    for edge in (*EDGES, "absent"):
        assert np.array_equal(
            shuffled.frequency(edge), plain.frequency(edge)[np.ix_(perm, perm)]
        ), edge
    assert plain.frequency("->").any() or plain.frequency("--").any()
    assert canonical(plain.table()) == canonical(shuffled.table())
    assert plain.structure.kind == shuffled.structure.kind
    assert canonical_edges(plain.structure) == canonical_edges(shuffled.structure)


def test_a_row_aligned_input_is_resampled_with_the_rows():
    data = lingam_data(300)
    domain = np.repeat([0, 1, 2], 100)
    out = bootstrap("cdnod", data, n_resamples=3, seed=0, domain_index=domain, keep_fits=True)
    assert out.n_resamples == 3 and out.structure.n_nodes == 4
    with pytest.raises(ValueError, match="domain_index has 299 rows but data has 300"):
        bootstrap("cdnod", data, n_resamples=3, domain_index=domain[:-1])
    with pytest.raises(ValueError, match="needs domain_index"):
        bootstrap("cdnod", data, n_resamples=3)
    ints = (np.random.default_rng(0).random((300, 4)) < 0.1).astype(int)
    assert bootstrap("gies", data, n_resamples=2, interventions=ints).n_resamples == 2
    assert bootstrap("gies", data, n_resamples=2).n_resamples == 2


def test_a_method_with_a_group_per_dataset_returns_one_result_per_group():
    groups = [lingam_data(150, seed) for seed in range(3)]
    out = bootstrap("multi_group_direct_lingam", groups, n_resamples=3, seed=1, keep_fits=True)
    assert isinstance(out, list) and len(out) == 3
    assert all(o.n_resamples == 3 and o.structure.n_nodes == 4 for o in out)
    assert all(len(o.fits) == 3 for o in out)
    # the groups are resampled on their own, so groups of different sizes work
    sizes = [lingam_data(120, 4), lingam_data(180, 5)]
    assert len(bootstrap("multi_group_direct_lingam", sizes, n_resamples=2, seed=1)) == 2


def series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    for t in range(1, n):
        x[t, 0] += 0.5 * x[t - 1, 0]
        x[t, 1] += 0.8 * x[t - 1, 0]
        x[t, 2] += 0.7 * x[t - 1, 1]
    return pd.DataFrame(x, columns=["x", "y", "z"])


def test_a_time_series_method_resamples_blocks_and_counts_every_lag():
    out = bootstrap(
        "pcmci", series(), n_resamples=4, seed=0, resampling="bootstrap", block_length=20, tau_max=1
    )
    assert isinstance(out.structure, TemporalStructure) and out.structure.lags == (1,)
    assert out.metadata["block_length"] == 20
    assert out.frequency("->", lag=1).shape == (3, 3)
    assert {row["lag"] for row in out.table()} == {1}
    default = bootstrap(
        "granger", series(), n_resamples=2, seed=0, resampling="bootstrap", tau_max=1
    )
    assert default.metadata["block_length"] == 10  # 10 * tau_max, above ceil(300 ** (1 / 3))
    with pytest.raises(ValueError, match="lag is for a time-series result"):
        bootstrap("pc", lingam_data(), n_resamples=2).frequency("->", lag=1)


def job_of(name, data, unit, resampling, fraction=0.5, *, seed=0, block_length=None):
    """The job a worker draws a resample from, without the call that builds it."""
    return _Job(
        name, data, {}, {}, seed, unit, block_length, False, "auto", "rank", resampling, fraction
    )


def test_the_blocks_are_consecutive_rows_of_the_series():
    job = job_of("pcmci", np.zeros((50, 2)), "blocks", "bootstrap", block_length=8)
    for b in range(5):
        (idx,) = _draw(job, b)
        assert len(idx) == 50 and idx.min() >= 0 and idx.max() < 50
        for start in range(0, 48, 8):  # six whole blocks and a cut one
            assert np.array_equal(np.diff(idx[start : start + 8]), np.ones(7, dtype=int))
    assert not np.array_equal(_draw(job, 0)[0], _draw(job, 1)[0])
    assert np.array_equal(_draw(job, 3)[0], _draw(job, 3)[0])
    with pytest.raises(ValueError, match="block_length must be an int from 1 to 50"):
        bootstrap("pcmci", series(50), n_resamples=2, resampling="bootstrap", block_length=51)


def test_the_default_block_is_at_most_the_series():
    assert _default_block_length(REGISTRY["pcmci"], {"tau_max": 30}, 100) == 100


def test_a_panel_of_time_points_resamples_the_units():
    panel = [lingam_data(120, seed) for seed in range(3)]
    out = bootstrap("longitudinal_lingam", panel, n_resamples=2, seed=0)
    assert isinstance(out.structure, TemporalStructure)
    with pytest.raises(ValueError, match="same units"):
        bootstrap("longitudinal_lingam", [panel[0], panel[1].iloc[:100]], n_resamples=2)


# ---- resampling ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "fraction"), [(100, 0.5), (101, 0.5), (7, 0.5), (240, 0.9), (60, 0.35), (50, 1.0)]
)
def test_a_subsample_draws_the_rounded_share_of_distinct_rows_in_their_order(n, fraction):
    job = job_of("pc", np.zeros((n, 2)), "rows", "subsample", fraction, seed=3)
    draws = [_draw(job, b)[0] for b in range(20)]
    for idx in draws:
        assert len(idx) == round(fraction * n) and len(set(idx.tolist())) == len(idx)
        assert idx.min() >= 0 and idx.max() < n and np.all(np.diff(idx) > 0)
    assert (len({tuple(idx) for idx in draws}) == 1) == (fraction == 1)  # the resamples differ
    assert all(np.array_equal(_draw(job, b)[0], draws[b]) for b in range(20))  # (seed, b) alone


def test_a_bootstrap_resample_is_n_rows_drawn_with_replacement_from_the_seed():
    job = job_of("pc", np.zeros((80, 2)), "rows", "bootstrap", seed=3)
    for b in range(5):
        rng = np.random.default_rng(np.random.SeedSequence(3, spawn_key=(b, 0)))
        assert np.array_equal(_draw(job, b)[0], rng.integers(0, 80, size=80))


@pytest.mark.parametrize("resampling", ["bootstrap", "subsample"])
def test_each_fit_is_the_method_on_the_rows_its_resample_draws(resampling):
    data = lingam_data(150)
    out = bootstrap("hc", data, n_resamples=4, seed=5, resampling=resampling, keep_fits=True)
    job = job_of("hc", data, "rows", resampling, seed=5)
    assert out.fits is not None
    for b, fit in enumerate(out.fits):
        assert fit == andrey.hc(data.iloc[_draw(job, b)[0]].reset_index(drop=True))
    if resampling == "bootstrap":  # the scheme of every release before sample_fraction
        for b in range(4):
            rng = np.random.default_rng(np.random.SeedSequence(5, spawn_key=(b, 0)))
            assert np.array_equal(_draw(job, b)[0], rng.integers(0, 150, size=150))


def test_a_series_subsample_is_one_stretch_of_consecutive_rows():
    job = job_of("pcmci", np.zeros((50, 2)), "blocks", "subsample", 0.4, seed=1)
    starts = set()
    for b in range(20):
        (idx,) = _draw(job, b)
        assert len(idx) == 20 and np.array_equal(np.diff(idx), np.ones(19, dtype=int))
        assert idx.min() >= 0 and idx.max() < 50
        starts.add(int(idx[0]))
    assert len(starts) > 1


def test_a_panel_subsample_keeps_the_same_units_in_every_dataset_and_each_groups_own_rows():
    units = job_of("longitudinal_lingam", [np.zeros((30, 2))] * 4, "units", "subsample", 0.5)
    idx = _draw(units, 0)
    assert len(idx) == 4 and all(np.array_equal(ix, idx[0]) for ix in idx)
    assert len(idx[0]) == 15 and len(set(idx[0].tolist())) == 15
    groups = job_of(
        "multi_group_direct_lingam", [np.zeros((120, 2)), np.zeros((180, 2))], "groups", "subsample"
    )
    first, second = _draw(groups, 0)
    assert (len(first), len(second)) == (60, 90) and len(set(second.tolist())) == 90


def test_resampling_defaults_to_a_subsample_for_every_method():
    data = lingam_data()
    default = bootstrap("hc", data, n_resamples=2)
    assert (default.metadata["resampling"], default.metadata["sample_fraction"]) == (
        "subsample",
        0.5,
    )
    drawn = bootstrap("hc", data, n_resamples=2, resampling="bootstrap", sample_fraction=0.2)
    assert drawn.metadata["resampling"] == "bootstrap" and drawn.metadata["sample_fraction"] is None
    assert (
        bootstrap("hc", data, n_resamples=2, sample_fraction=0.2).metadata["sample_fraction"] == 0.2
    )
    stretch = bootstrap("pcmci", series(), n_resamples=2, tau_max=1)
    assert stretch.metadata["resampling"] == "subsample"
    assert stretch.metadata["sample_fraction"] == 0.5 and stretch.metadata["block_length"] is None
    panel = [lingam_data(120, seed) for seed in range(3)]
    assert (
        bootstrap("longitudinal_lingam", panel, n_resamples=2).metadata["resampling"] == "subsample"
    )


@pytest.mark.parametrize(
    ("method", "settings", "n", "block"),
    [
        ("varma_lingam", {}, 300, 7),  # no lag: ceil(300 ** (1 / 3))
        ("pcmci", {"tau_max": 1}, 300, 10),  # ten times the longest lag
        ("pcmciplus", {"tau_max": 3}, 300, 30),
        ("granger", {"maxlag": 2}, 300, 20),  # a former name of tau_max counts
        ("pcmci", {}, 300, 10),  # the method's own tau_max
    ],
)
def test_the_default_block_is_ten_times_the_longest_lag_or_the_cube_root_of_the_length(
    method, settings, n, block
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = bootstrap(method, series(n), n_resamples=1, resampling="bootstrap", **settings)
    assert out.metadata["block_length"] == block


def test_a_subsample_of_every_row_gives_every_fit_the_fit_on_the_data():
    data = lingam_data(200)
    out = bootstrap(
        "direct_lingam", data, n_resamples=3, seed=0, sample_fraction=1.0, keep_fits=True
    )
    assert out.fits is not None
    for fit in out.fits:
        assert np.array_equal(fit.weighted_adjacency, andrey.direct_lingam(data).weighted_adjacency)


# ---- the call ------------------------------------------------------------------------------------


def test_the_method_is_a_name_or_a_facade_and_nothing_else():
    data = lingam_data()
    by_name = bootstrap("hc", data, n_resamples=2, seed=0)
    assert bootstrap(andrey.hc, data, n_resamples=2, seed=0).table() == by_name.table()
    for bad in ("nope", lambda x: x, len, "BootstrapOutput"):
        with pytest.raises(ValueError, match="method must be one of"):
            bootstrap(bad, data, n_resamples=2)
    with pytest.raises(NotImplementedError, match="gin"):
        bootstrap("gin", data, n_resamples=2)


def test_the_arguments_are_checked_before_any_fit():
    data = lingam_data()
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap("hc", data, n_resamples=0)
    with pytest.raises(TypeError, match="n_resamples"):
        bootstrap("hc", data, n_resamples=2.5)
    with pytest.raises(ValueError, match="threshold"):
        bootstrap("hc", data, n_resamples=2, threshold=0.0)
    with pytest.raises(ValueError, match="seed"):
        bootstrap("hc", data, n_resamples=2, seed=-1)
    with pytest.raises(ValueError, match="block_length is for a time-series method"):
        bootstrap("hc", data, n_resamples=2, block_length=5)
    with pytest.raises(TypeError, match="keep_fits"):
        bootstrap("hc", data, n_resamples=2, keep_fits=1)
    with pytest.raises(ValueError, match="resampling must be one of"):
        bootstrap("hc", data, n_resamples=2, resampling="jackknife")
    for fraction in (0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError, match="sample_fraction must be a number in"):
            bootstrap("hc", data, n_resamples=2, sample_fraction=fraction)
    with pytest.raises(TypeError, match="sample_fraction"):
        bootstrap("hc", data, n_resamples=2, sample_fraction="half")
    with pytest.raises(ValueError, match="at least 2 rows"):
        bootstrap("hc", data.iloc[:20], n_resamples=2, sample_fraction=0.05)
    with pytest.raises(ValueError, match="block_length is for resampling='bootstrap'"):
        bootstrap("pcmci", series(), n_resamples=2, resampling="subsample", block_length=5)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_fit_stops_the_run_and_names_its_resample(workers):
    with pytest.raises(TypeError, match="bogus") as raised:
        with_workers(workers, "hc", lingam_data(), n_resamples=3, bogus=1)
    assert any("bootstrap resample 0" in note for note in raised.value.__notes__)


def test_each_warning_of_the_fits_is_raised_once():
    rng = np.random.default_rng(0)
    codes = pd.DataFrame(rng.integers(0, 3, (300, 3)), columns=["a", "b", "c"])
    for workers in (1, 2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with_workers(workers, "ges", codes, n_resamples=4, seed=0)
        kinds = [w for w in caught if issubclass(w.category, andrey.DataTypeWarning)]
        assert len(kinds) == 1, workers


def test_the_workers_follow_the_configured_count_and_never_nest():
    data = lingam_data()
    with andrey.config(n_jobs=2):
        configured = bootstrap("pc", data, n_resamples=4, seed=0, keep_fits=True)
    serial = with_workers(1, "pc", data, n_resamples=4, seed=0, keep_fits=True)
    assert configured.fits == serial.fits
    assert andrey.config.n_jobs == 1 or "ANDREY_N_JOBS" in os.environ


def test_knowledge_travels_to_the_workers():
    data = lingam_data()
    knowledge = andrey.Knowledge(required=[("c", "d")])
    serial = with_workers(1, "pc", data, n_resamples=3, seed=0, knowledge=knowledge)
    pooled = with_workers(2, "pc", data, n_resamples=3, seed=0, knowledge=knowledge)
    assert serial.table() == pooled.table()
    assert serial.frequency("->")[2, 3] == 1.0  # the required edge is in every fit


@pytest.mark.parametrize("method", ["pc", "fci"])
def test_an_edge_the_knowledge_directs_stays_directed_in_the_averaged_graph(method):
    # Without knowledge, a - b - c - d is undirected in a CPDAG and has circles in a PAG.
    knowledge = andrey.Knowledge(required=[("c", "d")])
    out = bootstrap(method, lingam_data(), n_resamples=5, seed=0, knowledge=knowledge)
    assert out.metadata["averaged_graph"] == "vote"
    assert out.structure.endpoints(2, 3) == (TAIL, ARROW)


def linear_data(n, parents, seed=0):
    """Linear data over the columns of ``parents``: 0.8 times the sum of the parents, plus noise."""
    rng = np.random.default_rng(seed)
    columns = {}
    for name, causes in parents.items():
        columns[name] = sum(0.8 * columns[c] for c in causes) + rng.normal(size=n)
    return pd.DataFrame(columns)


def mag_oracle(pag):
    """The d-separation test and separating sets of a MAG of ``pag``'s class, as ``to_pag`` uses."""
    mag = GraphStructure.from_numpy(mag_of_pag(np.asarray(pag.to_numpy())), kind="mag")
    digraph, selection = _canonical_dag(mag)
    nodes = list(range(mag.n_nodes))
    sepsets = _AncestorSepsets(
        np.asarray(mag.to_numpy()) != 0,
        nodes,
        _observed_ancestors(digraph, nodes),
        _observed_ancestors_of(digraph, nodes, selection),
    )
    return DSepOracle(digraph, nodes, selection), sepsets


@pytest.mark.parametrize("method", ["fci", "gfci"])
def test_an_averaged_pag_under_knowledge_is_closed_under_the_orientation_rules(method):
    # Tiers put an arrowhead at c on b - c, from which the rules orient c -> d and c -> e.
    data = linear_data(500, {"a": [], "b": ["a"], "c": ["b"], "d": ["c"], "e": ["c"]})
    knowledge = andrey.Knowledge(tiers=[["a", "b"], ["c", "d", "e"]])
    out = bootstrap(method, data, n_resamples=8, seed=0, knowledge=knowledge, keep_fits=True)
    test, sepsets = mag_oracle(BootstrapOutput.from_fits(out.fits).structure)
    marks = np.array(out.structure.to_numpy(), dtype=np.int8)
    _fci_orient(marks, test, 0.5, sepsets)
    assert np.array_equal(marks, out.structure.to_numpy())


@pytest.mark.parametrize("method", ["fci", "gfci"])
def test_an_averaged_pag_under_knowledge_equals_the_fit_every_resample_agrees_on(method):
    # The required a -> b orients the chain a -> b -> c -> d to its end.
    data = linear_data(1000, {"a": [], "b": ["a"], "c": ["b"], "d": ["c"]})
    knowledge = andrey.Knowledge(required=[("a", "b")])
    fit = getattr(andrey, method)(data, knowledge=knowledge)
    out = bootstrap(method, data, n_resamples=5, seed=0, knowledge=knowledge, keep_fits=True)
    assert not (fit.structure.to_numpy() == CIRCLE).any()
    assert all(resample.structure == fit.structure for resample in out.fits)
    assert out.structure == fit.structure


def tie_data(n, seed):
    """Six columns as an array: `x1 -> x3 <- x2`, `x3 -> x4`, and the separate pair `x5 -> x6`.

    The pair is an edge that nothing else orients, so a tie on it is the rank's to decide.
    """
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-1, 1, (2, n))
    x3 = 0.9 * x1 + 0.9 * x2 + 0.5 * rng.uniform(-1, 1, n)
    x4 = 0.8 * x3 + 0.5 * rng.uniform(-1, 1, n)
    x5 = rng.uniform(-1, 1, n)
    x6 = 0.7 * x5 + 0.5 * rng.uniform(-1, 1, n)
    return np.column_stack([x1, x2, x3, x4, x5, x6])


def tied_pairs(out):
    """The pairs whose most frequent edges tie, and more fits have them than have no edge."""
    n = out.structure.n_nodes
    found = []
    for i in range(n):
        for j in range(i + 1, n):
            counts = [round(out.frequency(e)[i, j] * out.n_resamples) for e in EDGES]
            absent = round(out.frequency("absent")[i, j] * out.n_resamples)
            if max(counts) > absent and counts.count(max(counts)) > 1:
                found.append((i, j))
    return found


@pytest.mark.parametrize("table", [False, True], ids=["array", "table"])
@pytest.mark.parametrize(
    "method, kind, n, n_resamples, seed",
    [("direct_lingam", "dag", 40, 6, 27), ("pc", "cpdag", 60, 4, 5)],
)
def test_a_tie_in_the_vote_does_not_depend_on_the_column_order(
    method, kind, n, n_resamples, seed, table
):
    """The fits tie on a pair, and the averaged graph is the same up to relabeling whatever the
    column order, for the array that carries no labels and the table that does."""
    data = tie_data(n, seed)
    perm = [5, 2, 0, 4, 3, 1]
    shuffled_data = data[:, perm]
    if table:
        data = pd.DataFrame(data, columns=[f"x{k}" for k in range(1, 7)])
        shuffled_data = data.iloc[:, perm]
    plain = bootstrap(method, data, n_resamples=n_resamples, seed=seed)
    shuffled = bootstrap(method, shuffled_data, n_resamples=n_resamples, seed=seed)
    assert tied_pairs(plain) and plain.structure.kind == shuffled.structure.kind == kind
    assert np.array_equal(
        shuffled.structure.to_numpy(), plain.structure.to_numpy()[np.ix_(perm, perm)]
    )


def test_the_tie_break_rule_is_recorded_and_ranks_the_columns_by_it():
    data = tie_data(40, 27)
    perm = [5, 2, 0, 4, 3, 1]
    plain = bootstrap("direct_lingam", data, n_resamples=2, seed=0)
    shuffled = bootstrap("direct_lingam", data[:, perm], n_resamples=2, seed=0)
    assert plain.metadata["tie_break"] == shuffled.metadata["tie_break"] == "rank"
    assert np.array_equal(shuffled._ranks, plain._ranks[perm])  # a rank moves with its column
    with andrey.config(tie_break="position"):
        by_position = bootstrap("direct_lingam", data[:, perm], n_resamples=2, seed=0)
    assert by_position.metadata["tie_break"] == "position"
    assert np.array_equal(by_position._ranks, np.arange(6))
