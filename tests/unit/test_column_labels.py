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
def test_dataframe_columns_become_labels(spec):
    if spec.name == "calm":
        pytest.importorskip("torch")
    frames = [frame(0), frame(1), frame(2)]
    out = getattr(andrey, spec.name)(*inputs(spec, frames), **SHORT.get(spec.name, {}))
    for result in out if isinstance(out, list) else [out]:
        for graph in graphs(result.structure):
            # GIN appends its latents after the observed columns; a pairwise test takes two.
            columns = COLUMNS[:2] if spec.family == "pairwise" else COLUMNS
            assert graph.labels[: len(columns)] == tuple(columns)
            assert graph.labels == result.structure.labels


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


def test_duplicate_column_names_are_rejected():
    df = frame()
    df.columns = ["x", "x", "z"]
    with pytest.raises(ValueError, match="unique"):
        andrey.pc(df)


def test_groups_must_name_their_columns_alike():
    renamed = frame(1).rename(columns={"x": "w"})
    with pytest.raises(ValueError, match="differently"):
        andrey.multi_group_direct_lingam([frame(0), renamed])
    # An unnamed array group takes the names the others share.
    outs = andrey.multi_group_direct_lingam([frame(0), frame(1).to_numpy()])
    assert [o.structure.labels for o in outs] == [tuple(COLUMNS)] * 2
