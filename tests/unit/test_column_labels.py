"""A DataFrame's column names become the result's node labels, for every method that takes one.

The labels also name each edge under any column order, which checks the registry's column-order
fields on every method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import TemporalStructure, backend
from andrey.core.knowledge import column_position
from andrey.core.structure import node_labels
from andrey.spec import list_specs

COLUMNS = ["x", "y", "z"]
N = 400


def frame(seed: int = 0) -> pd.DataFrame:
    """A non-Gaussian chain x -> y -> z, so every method family finds structure."""
    u = np.random.default_rng(seed).uniform(-1, 1, size=(N, 3)) ** 3
    x = u[:, 0]
    y = 1.2 * x + u[:, 1]
    return pd.DataFrame({"x": x, "y": y, "z": 0.7 * y + u[:, 2]})


def inputs(spec, frames) -> list:
    """The method's required positional inputs, with a DataFrame wherever it takes a data matrix."""
    args = []
    for inp in spec.data.inputs:
        if not inp.required:
            continue
        if inp.kind == "matrix":
            pair = spec.family == "pairwise"
            args.append(frames[0].iloc[:, :2] if pair else frames[0])
        elif inp.kind == "panel":
            args.append(frames)
        elif inp.kind == "context":
            args.append(np.repeat([0.0, 1.0], N // 2).reshape(-1, 1))
    return args


def graphs(structure):
    """Every graph a structure holds: itself, or each lag and each (time, lag) cell."""
    if isinstance(structure, TemporalStructure):
        cells = [structure.at(t, k) for t in structure.times for k in structure.lags]
        return [structure.lag(k) for k in structure.lags] + cells
    return [structure]


TABLE_SPECS = [
    spec for spec in list_specs() if {i.kind for i in spec.data.inputs} & {"matrix", "panel"}
]

#: Settings that shorten a slow fit; the labels come from the same facade path.
SHORT = {"calm": {"max_outer": 1, "subproblem_iter": 20}}


@pytest.mark.parametrize("spec", TABLE_SPECS, ids=lambda spec: spec.name)
def test_dataframe_columns_become_labels_as_text(spec):
    if spec.name == "calm":
        pytest.importorskip("torch")
    # Integer names that are also positions, beside a string: every label is the name as text.
    frames = [frame(seed).set_axis([1, 0, "z"], axis=1) for seed in range(3)]
    out = getattr(andrey, spec.name)(*inputs(spec, frames), **SHORT.get(spec.name, {}))
    for result in out if isinstance(out, list) else [out]:
        for graph in graphs(result.structure):
            # GIN appends its latents after the observed columns; a pairwise test takes two.
            labels = ("1", "0", "z")[: 2 if spec.family == "pairwise" else 3]
            assert graph.labels[: len(labels)] == labels
            assert graph.labels == result.structure.labels


@pytest.mark.parametrize("spec", TABLE_SPECS, ids=lambda spec: spec.name)
def test_columns_named_alike_as_text_raise_before_the_fit(spec):
    # One row: a fit, or any other check of the rows, would fail on it first.
    frames = [frame(seed).head(1).set_axis(["1", 1, 0], axis=1) for seed in range(3)]
    with pytest.raises(ValueError, match=r"'1' names columns \[0, 1\]"):
        getattr(andrey, spec.name)(*inputs(spec, frames))


def named_marks(out) -> list[dict]:
    """Each graph of a result as ``{(label, label): mark}``, which no column order changes."""
    found = []
    for result in out if isinstance(out, list) else [out]:
        for graph in graphs(result.structure):
            m, labels = graph.to_numpy(), graph.labels
            found.append(
                {(a, b): int(m[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}
            )
    return found


@pytest.mark.parametrize("spec", TABLE_SPECS, ids=lambda spec: spec.name)
def test_reordering_the_columns_relabels_the_result(spec, monkeypatch):
    """A method marked order-free names the same edges under reordered columns, and exactly the
    methods marked ``tie_break`` read the setting during a fit. A pairwise method's two columns
    swap."""
    if spec.name == "calm":
        pytest.importorskip("torch")
    reads = []
    resolve = backend._resolve_tie_break
    monkeypatch.setattr(backend, "_resolve_tie_break", lambda: reads.append(1) or resolve())

    def fit(columns: list[str], **given):
        frames = [frame(seed)[columns] for seed in range(3)]
        return getattr(andrey, spec.name)(
            *inputs(spec, frames), **SHORT.get(spec.name, {}), **given
        )

    want = named_marks(fit(COLUMNS))
    assert bool(reads) == spec.tie_break
    for given in ({"collider_order": "p_value"}, {"possible_dsep": "order_free"}):
        if given.keys() <= {param.name for param in spec.params}:
            # An order given in the call does not consult the setting for that order.
            read_by_default = len(reads)
            reads.clear()
            fit(COLUMNS, **given)
            assert len(reads) == read_by_default - 1
    if spec.column_order != "can_change":
        pair = spec.family == "pairwise"
        for order in ([1, 0, 2],) if pair else ([1, 2, 0], [2, 0, 1]):
            assert named_marks(fit([COLUMNS[k] for k in order])) == want


@pytest.mark.parametrize("data", [frame().to_numpy(), pd.DataFrame(frame().to_numpy())])
def test_arrays_and_default_columns_keep_integer_nodes(data):
    out = andrey.pc(data)
    assert out.structure.labels is None
    edges = out.structure.oriented_edges()
    assert edges and all(isinstance(s, int) and isinstance(t, int) for s, t, _ in edges)


def test_reversed_columns_give_the_same_named_edges():
    df = frame()
    forward = andrey.direct_lingam(df).structure.oriented_edges()
    reverse = andrey.direct_lingam(df[COLUMNS[::-1]]).structure.oriented_edges()
    assert sorted(forward) == sorted(reverse) == [("x", "y", "directed"), ("y", "z", "directed")]


def test_reversed_columns_give_the_same_named_edges_on_torch(monkeypatch):
    # A reversed frame's numpy array is a view with a negative stride, which torch rejects.
    pytest.importorskip("torch")
    monkeypatch.setenv("ANDREY_DEVICE", "cpu")  # a pinned device runs even a small fit on torch
    noise = np.random.default_rng(0).standard_normal((N, 3))
    collider = noise[:, 0] + noise[:, 1] + noise[:, 2]
    df = pd.DataFrame(np.column_stack([noise[:, 0], collider, noise[:, 1]]), columns=COLUMNS)
    forward = andrey.pc(df).structure
    reverse = andrey.pc(df[COLUMNS[::-1]]).structure
    assert reverse.labels == tuple(COLUMNS[::-1])
    assert sorted(reverse.oriented_edges()) == sorted(forward.oriented_edges())
    assert sorted(forward.oriented_edges()) == [("x", "y", "directed"), ("z", "y", "directed")]


def test_gin_explicit_labels_beat_columns():
    out = andrey.gin(frame(), labels=("a", "b", "c"))
    assert out.structure.labels[:3] == ("a", "b", "c")


@pytest.mark.parametrize(
    ("labels", "message"), [(("a", "a", "b"), r"'a' names columns \[0, 1\]"), ((1, "1", 2), "'1'")]
)
def test_gin_explicit_labels_must_differ_as_text(labels, message):
    with pytest.raises(ValueError, match=message):
        andrey.gin(frame(), labels=labels)


def test_gin_latents_skip_the_names_of_observed_columns():
    latents = andrey.gin(frame().set_axis(["L1", "x", "L3"], axis=1)).structure.labels[3:]
    assert latents[0] == "L2" and not set(latents) & {"L1", "x", "L3"}
    assert andrey.gin(frame(), labels=(1, 2, 3)).structure.labels[:4] == ("1", "2", "3", "L1")


@pytest.mark.parametrize(
    ("names", "labels"),
    [
        ([10, 20, 30], ("10", "20", "30")),
        ([1.5, None, ("a", 1)], ("1.5", "None", "('a', 1)")),
        ([1, "y", np.str_("z")], ("1", "y", "z")),
    ],
)
def test_node_labels_take_each_name_as_text(names, labels):
    assert node_labels(names) == labels
    assert all(type(label) is str for label in node_labels(names))


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["x", "x", "z"], r"'x' names columns \[0, 1\]"),
        ([1, "1", "z", "z"], r"'1' names columns \[0, 1\]; 'z' names columns \[2, 3\]"),
    ],
)
def test_node_labels_name_every_label_columns_share(names, message):
    with pytest.raises(ValueError, match=message):
        node_labels(names)


@pytest.mark.parametrize(
    ("node", "labels", "position"),
    [("1", ("1", "0", "2"), 0), (1, ("1", "0", "2"), 1), (2, None, 2), ("b", ("a", "b"), 1)],
)
def test_column_position_reads_a_string_as_a_label_and_an_integer_as_a_position(
    node, labels, position
):
    assert column_position(node, labels, 3, "categorical") == position


@pytest.mark.parametrize(
    ("node", "labels", "message"),
    [
        ("w", ("a", "b", "c"), "names column 'w', which data does not have"),
        ("a", None, "data has no column names"),
        (3, ("a", "b", "c"), "lists column 3, but data has 3 columns$"),
        (10, ("10", "20", "30"), "to name the column '10', pass its label as a string"),
    ],
)
def test_column_position_names_a_missing_column(node, labels, message):
    with pytest.raises(ValueError, match=message):
        column_position(node, labels, 3, "categorical")


def test_groups_must_name_their_columns_alike():
    renamed = frame(1).rename(columns={"x": "w"})
    with pytest.raises(ValueError, match="differently"):
        andrey.multi_group_direct_lingam([frame(0), renamed])
    # An unnamed array group takes the names the others share.
    outs = andrey.multi_group_direct_lingam([frame(0), frame(1).to_numpy()])
    assert [o.structure.labels for o in outs] == [tuple(COLUMNS)] * 2
