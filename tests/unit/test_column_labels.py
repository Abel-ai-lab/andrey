"""A DataFrame's column names become the result's node labels, for every method that takes one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import TemporalStructure
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
    """The method's positional inputs, with a DataFrame wherever it takes a data matrix."""
    args = []
    for inp in spec.data.inputs:
        if inp.kind == "matrix":
            args.append(frames[0][COLUMNS[:2]] if spec.name == "pnl" else frames[0])  # a pair
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


@pytest.mark.parametrize("spec", TABLE_SPECS, ids=lambda spec: spec.name)
def test_dataframe_columns_become_labels(spec):
    if spec.name == "calm":
        pytest.importorskip("torch")
    out = getattr(andrey, spec.name)(*inputs(spec, [frame(0), frame(1), frame(2)]))
    for result in out if isinstance(out, list) else [out]:
        for graph in graphs(result.structure):
            # GIN appends its latents after the observed columns; PNL takes the first two.
            columns = COLUMNS[:2] if spec.name == "pnl" else COLUMNS
            assert graph.labels[: len(columns)] == tuple(columns)
            assert graph.labels == result.structure.labels


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
