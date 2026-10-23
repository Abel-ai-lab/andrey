"""``andrey.bootstrap`` and ``BootstrapOutput``: the counts, the averaged graph, and the refits."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import BootstrapOutput, bootstrap
from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput, TemporalStructure
from andrey.resample import EDGES, _draw, _Job


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
    assert out.n_boot == 10
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
        (0.3, {(0, 1, "directed"), (1, 2, "undirected"), (2, 3, "undirected")}),
        (0.4, {(0, 1, "directed"), (1, 2, "undirected"), (2, 3, "undirected")}),
        (0.5, {(0, 1, "directed"), (2, 3, "undirected")}),
        (0.8, {(0, 1, "directed"), (2, 3, "undirected")}),
        (0.9, {(2, 3, "undirected")}),
        (1.0, {(2, 3, "undirected")}),
    ],
)
def test_the_threshold_decides_which_pairs_are_adjacent(counted, threshold, expected):
    out = BootstrapOutput.from_fits(counted, threshold=threshold)
    assert edge_set(out.structure) == expected
    assert out.structure == out.average(threshold)
    assert out.threshold == threshold
    out.structure.validate()


def test_a_pair_takes_its_most_frequent_edge_and_a_tie_the_least_committal(counted):
    out = BootstrapOutput.from_fits(counted)
    assert out.structure.endpoints(0, 1) == (TAIL, ARROW)  # 6 of 10 against 2
    assert out.structure.endpoints(2, 3) == (TAIL, TAIL)  # 5 against 5
    assert out.average(0.3).kind == "cpdag"
    # three dags that disagree widen to a cpdag only because of the tie edge
    dags = [graph(2, [directed(0, 1)]), graph(2, [directed(1, 0)])]
    assert BootstrapOutput.from_fits(dags[:1]).structure.kind == "dag"
    tied = BootstrapOutput.from_fits(dags)
    assert tied.structure.kind == "cpdag" and tied.structure.endpoints(0, 1) == (TAIL, TAIL)
    pags = [graph(2, [(0, 1, CIRCLE, ARROW)], "pag"), graph(2, [directed(0, 1)], "pag")]
    tied = BootstrapOutput.from_fits(pags)
    assert tied.structure.kind == "pag" and tied.structure.endpoints(0, 1) == (CIRCLE, CIRCLE)
    assert tied.frequency("o->")[0, 1] == tied.frequency("->")[0, 1] == 0.5


def test_a_lag_graph_counts_each_directed_edge_on_its_own():
    def series(*lag1):
        lag0 = GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8))
        M = np.zeros((2, 2), dtype=np.int8)
        for i, j in lag1:
            M[i, j] = ARROW if (i == j or M[j, i]) else TAIL
            M[j, i] = ARROW
        graphs = [lag0, GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)]
        return StructureOutput.new(TemporalStructure.from_lag_graphs(graphs, lags=[0, 1]))

    fits = [series((0, 1), (1, 0)), series((0, 1)), series((1, 1)), series()]
    out = BootstrapOutput.from_fits(fits, threshold=0.5)
    assert out.frequency("->", lag=1)[0, 1] == 0.5
    assert out.frequency("<-", lag=1)[1, 0] == 0.5 and out.frequency("->", lag=1)[1, 0] == 0.25
    assert out.frequency("absent", lag=1)[0, 1] == 0.5
    assert out.frequency("->", lag=1)[1, 1] == 0.25 and not out.frequency("<->", lag=1).any()
    assert not out.frequency("->", lag=0).any()
    assert out.structure.lags == (0, 1)
    assert edge_set(out.structure.lag(1)) == {(0, 1, "directed")}
    assert edge_set(out.average(0.25).lag(1)) == {
        (0, 1, "directed"),
        (1, 0, "directed"),
        (1, 1, "directed"),
    }
    rows = out.table()
    assert [(r["lag"], r["source"], r["target"], r["count"]) for r in rows] == [
        (1, 0, 1, 2),
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
    for threshold in (0, 1.01, -0.5):
        with pytest.raises(ValueError, match="threshold"):
            BootstrapOutput.from_fits(counted, threshold=threshold)
    with pytest.raises(TypeError, match="threshold"):
        BootstrapOutput.from_fits(counted, threshold=True)
    lagged = GraphStructure.from_numpy(
        np.array([[0, ARROW], [TAIL, 0]], dtype=np.int8), kind="digraph"
    )
    mixed = [counted[0], StructureOutput.new(GraphStructure.from_numpy(np.zeros((4, 4))))]
    assert BootstrapOutput.from_fits(mixed).n_boot == 2
    other = [graph(2, [directed(0, 1)]), StructureOutput.new(lagged)]
    with pytest.raises(ValueError, match="digraph"):
        BootstrapOutput.from_fits(other)


def test_the_labels_of_the_fits_name_the_nodes():
    frame = pd.DataFrame(
        np.random.default_rng(0).standard_normal((200, 3)), columns=["a", "b", "c"]
    )
    frame["b"] += frame["a"]
    out = bootstrap("hc", frame, n_boot=3, seed=0)
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
    short = bootstrap("direct_lingam", data, n_boot=4, seed=7, keep_fits=True)
    longer = bootstrap("direct_lingam", data, n_boot=6, seed=7, keep_fits=True)
    again = bootstrap(andrey.direct_lingam, data, n_boot=4, seed=7, keep_fits=True)
    other = bootstrap("direct_lingam", data, n_boot=4, seed=8, keep_fits=True)
    assert short.fits == again.fits == longer.fits[:4]
    assert short.table() == again.table()
    weights = [fit.weighted_adjacency for fit in short.fits]
    assert all(not np.array_equal(w, weights[0]) for w in weights[1:])  # the resamples differ
    assert any(not np.array_equal(f.weighted_adjacency, g.weighted_adjacency)
               for f, g in zip(short.fits, other.fits))  # fmt: skip
    assert short.metadata == {"method": "direct_lingam", "seed": 7, "block_length": None}
    assert short.fits is not None and bootstrap("hc", data, n_boot=2).fits is None


def test_the_default_seed_follows_andrey_seed(monkeypatch):
    data = lingam_data()
    monkeypatch.setenv("ANDREY_SEED", "5")
    from_env = bootstrap("direct_lingam", data, n_boot=3, keep_fits=True)
    explicit = bootstrap("direct_lingam", data, n_boot=3, seed=5, keep_fits=True)
    assert from_env.fits == explicit.fits and from_env.metadata["seed"] == 5
    monkeypatch.delenv("ANDREY_SEED")
    assert bootstrap("hc", data, n_boot=2).metadata["seed"] == 0


@pytest.mark.parametrize("method", ["pc", "direct_lingam", "boss"])
def test_workers_do_not_change_the_result(method):
    data = lingam_data()
    serial = bootstrap(method, data, n_boot=6, seed=3, num_workers=1, keep_fits=True)
    pooled = bootstrap(method, data, n_boot=6, seed=3, num_workers=2, keep_fits=True)
    assert serial.fits == pooled.fits and serial.table() == pooled.table()
    assert serial.structure == pooled.structure


def test_a_seeded_method_fits_each_resample_with_its_own_seed_unless_given_one():
    data = lingam_data()
    derived = bootstrap("ica_lingam", data, n_boot=3, seed=1, keep_fits=True)
    pinned = bootstrap("ica_lingam", data, n_boot=3, seed=1, random_state=4, keep_fits=True)
    again = bootstrap("ica_lingam", data, n_boot=3, seed=1, random_state=4, keep_fits=True)
    assert pinned.fits == again.fits
    assert derived.fits != pinned.fits


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
    plain = bootstrap(method, frame, n_boot=8, seed=2)
    shuffled = bootstrap(method, frame.iloc[:, perm], n_boot=8, seed=2)
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
    out = bootstrap("cdnod", data, n_boot=3, seed=0, domain_index=domain, keep_fits=True)
    assert out.n_boot == 3 and out.structure.n_nodes == 4
    with pytest.raises(ValueError, match="domain_index has 299 rows but data has 300"):
        bootstrap("cdnod", data, n_boot=3, domain_index=domain[:-1])
    with pytest.raises(ValueError, match="needs domain_index"):
        bootstrap("cdnod", data, n_boot=3)
    ints = (np.random.default_rng(0).random((300, 4)) < 0.1).astype(int)
    assert bootstrap("gies", data, n_boot=2, interventions=ints).n_boot == 2
    assert bootstrap("gies", data, n_boot=2).n_boot == 2


def test_a_method_with_a_group_per_dataset_returns_one_result_per_group():
    groups = [lingam_data(150, seed) for seed in range(3)]
    out = bootstrap("multi_group_direct_lingam", groups, n_boot=3, seed=1, keep_fits=True)
    assert isinstance(out, list) and len(out) == 3
    assert all(o.n_boot == 3 and o.structure.n_nodes == 4 for o in out)
    assert all(len(o.fits) == 3 for o in out)
    # the groups are resampled on their own, so groups of different sizes work
    sizes = [lingam_data(120, 4), lingam_data(180, 5)]
    assert len(bootstrap("multi_group_direct_lingam", sizes, n_boot=2, seed=1)) == 2


def series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    for t in range(1, n):
        x[t, 0] += 0.5 * x[t - 1, 0]
        x[t, 1] += 0.8 * x[t - 1, 0]
        x[t, 2] += 0.7 * x[t - 1, 1]
    return pd.DataFrame(x, columns=["x", "y", "z"])


def test_a_time_series_method_resamples_blocks_and_counts_every_lag():
    out = bootstrap("pcmci", series(), n_boot=4, seed=0, block_length=20, tau_max=1)
    assert isinstance(out.structure, TemporalStructure) and out.structure.lags == (1,)
    assert out.metadata["block_length"] == 20
    assert out.frequency("->", lag=1).shape == (3, 3)
    assert {row["lag"] for row in out.table()} == {1}
    default = bootstrap("granger", series(), n_boot=2, seed=0, maxlag=1)
    assert default.metadata["block_length"] == 7  # ceil(300 ** (1 / 3))
    with pytest.raises(ValueError, match="lag is for a time-series result"):
        bootstrap("pc", lingam_data(), n_boot=2).frequency("->", lag=1)


def test_the_blocks_are_consecutive_rows_of_the_series():
    job = _Job("pcmci", np.zeros((50, 2)), {}, {}, 0, "blocks", 8, None, "auto", "rank")
    for b in range(5):
        (idx,) = _draw(job, b)
        assert len(idx) == 50 and idx.min() >= 0 and idx.max() < 50
        for start in range(0, 48, 8):  # six whole blocks and a cut one
            assert np.array_equal(np.diff(idx[start : start + 8]), np.ones(7, dtype=int))
    assert not np.array_equal(_draw(job, 0)[0], _draw(job, 1)[0])
    assert np.array_equal(_draw(job, 3)[0], _draw(job, 3)[0])
    with pytest.raises(ValueError, match="block_length must be an int from 1 to 50"):
        bootstrap("pcmci", series(50), n_boot=2, block_length=51)


def test_a_panel_of_time_points_resamples_the_units():
    panel = [lingam_data(120, seed) for seed in range(3)]
    out = bootstrap("longitudinal_lingam", panel, n_boot=2, seed=0)
    assert isinstance(out.structure, TemporalStructure)
    with pytest.raises(ValueError, match="same units"):
        bootstrap("longitudinal_lingam", [panel[0], panel[1].iloc[:100]], n_boot=2)


# ---- the call ------------------------------------------------------------------------------------


def test_the_method_is_a_name_or_a_facade_and_nothing_else():
    data = lingam_data()
    by_name = bootstrap("hc", data, n_boot=2, seed=0)
    assert bootstrap(andrey.hc, data, n_boot=2, seed=0).table() == by_name.table()
    for bad in ("nope", lambda x: x, len, "BootstrapOutput"):
        with pytest.raises(ValueError, match="method must be one of"):
            bootstrap(bad, data, n_boot=2)
    with pytest.raises(NotImplementedError, match="gin"):
        bootstrap("gin", data, n_boot=2)


def test_the_arguments_are_checked_before_any_fit():
    data = lingam_data()
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap("hc", data, n_boot=0)
    with pytest.raises(TypeError, match="n_boot"):
        bootstrap("hc", data, n_boot=2.5)
    with pytest.raises(ValueError, match="threshold"):
        bootstrap("hc", data, n_boot=2, threshold=0.0)
    with pytest.raises(ValueError, match="seed"):
        bootstrap("hc", data, n_boot=2, seed=-1)
    with pytest.raises(ValueError, match="block_length is for a time-series method"):
        bootstrap("hc", data, n_boot=2, block_length=5)
    with pytest.raises(TypeError, match="keep_fits"):
        bootstrap("hc", data, n_boot=2, keep_fits=1)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_fit_stops_the_run_and_names_its_resample(workers):
    with pytest.raises(TypeError, match="bogus") as raised:
        bootstrap("hc", lingam_data(), n_boot=3, num_workers=workers, bogus=1)
    assert any("bootstrap resample 0" in note for note in raised.value.__notes__)


def test_each_warning_of_the_fits_is_raised_once():
    rng = np.random.default_rng(0)
    codes = pd.DataFrame(rng.integers(0, 3, (300, 3)), columns=["a", "b", "c"])
    for workers in (1, 2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap("ges", codes, n_boot=4, seed=0, num_workers=workers)
        kinds = [w for w in caught if issubclass(w.category, andrey.DataTypeWarning)]
        assert len(kinds) == 1, workers


def test_the_workers_follow_the_configured_count_and_never_nest():
    data = lingam_data()
    with andrey.config(num_workers=2):
        configured = bootstrap("pc", data, n_boot=4, seed=0, keep_fits=True)
    serial = bootstrap("pc", data, n_boot=4, seed=0, num_workers=1, keep_fits=True)
    assert configured.fits == serial.fits
    assert andrey.config.num_workers == 1 or "ANDREY_NUM_WORKERS" in os.environ


def test_knowledge_travels_to_the_workers():
    data = lingam_data()
    knowledge = andrey.Knowledge(required=[("c", "d")])
    serial = bootstrap("pc", data, n_boot=3, seed=0, num_workers=1, knowledge=knowledge)
    pooled = bootstrap("pc", data, n_boot=3, seed=0, num_workers=2, knowledge=knowledge)
    assert serial.table() == pooled.table()
    assert serial.frequency("->")[2, 3] == 1.0  # the required edge is in every fit
