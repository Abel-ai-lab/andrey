"""A column's offset changes no method's graph.

A method's result depends on where a column's zero lies only where its model does. A step that
reads the offset through its arithmetic, such as squared distances from ``x^2 + y^2 - 2xy`` or a
sum of terms far from 0, is a numerical artifact. The data here lie on a grid that an offset of
``1e9`` keeps exactly, so every fit sees the same values at both offsets and any difference comes
from the method.

``varma_lingam`` also has a case of its own on a larger series: its likelihood fit takes a constant
trend, so a column's mean is not part of the model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core import TemporalStructure
from andrey.core.independence import gamma_gram, hsic_gamma_from_grams
from andrey.spec import list_specs
from tests.recovery.helpers import datagen

N = 400
#: The grid spacing, ``2**-23``: the spacing of the floats in ``[2**29, 2**30)``, so a column on it
#: plus any offset below ``2**29`` stays exactly the same values.
GRID = 2.0**-23
#: Shifts of the columns: a large offset, and a scale with an offset.
SHIFTS = {"offset": (1.0, 1e9), "scale_and_offset": (3.7, 1e6)}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        warnings.simplefilter("ignore", andrey.SearchLimitWarning)
        yield


def on_grid(x: np.ndarray) -> np.ndarray:
    """``x`` rounded to the multiples of :data:`GRID`."""
    return np.round(x / GRID) * GRID


def chain(seed: int = 0) -> np.ndarray:
    """A non-Gaussian chain 0 -> 1 -> 2, so every method family finds structure."""
    u = np.random.default_rng(seed).uniform(-1, 1, size=(N, 3)) ** 3
    x = u.copy()
    x[:, 1] += 1.2 * u[:, 0]
    x[:, 2] += 0.7 * x[:, 1]
    return on_grid(x)


def graphs(structure) -> list[np.ndarray]:
    """The endpoint marks of every graph a structure holds: itself, or each lag and each cell."""
    if isinstance(structure, TemporalStructure):
        cells = [structure.at(t, k) for t in structure.times for k in structure.lags]
        return [structure.lag(k).to_numpy() for k in structure.lags] + [c.to_numpy() for c in cells]
    return [structure.to_numpy()]


def marks(out) -> list[np.ndarray]:
    return [
        m for result in (out if isinstance(out, list) else [out]) for m in graphs(result.structure)
    ]


def inputs(spec, data: np.ndarray, shift: tuple[float, float]) -> list:
    """The method's required inputs built from ``data`` shifted to ``data * scale + offset``."""
    scale, offset = shift

    def moved(x: np.ndarray) -> np.ndarray:
        return x * scale + offset

    args = []
    for inp in spec.data.inputs:
        if not inp.required:
            continue
        if inp.kind == "matrix":
            args.append(moved(data[:, :2] if spec.family == "pairwise" else data))
        elif inp.kind == "panel":
            args.append([moved(chain(seed)) for seed in range(3)])
        elif inp.kind == "context":
            args.append(np.repeat([0.0, 1.0], N // 2).reshape(-1, 1))
    return args


#: Settings that shorten a slow fit.
SHORT = {"calm": {"max_iter": 1, "subproblem_iter": 20}}

TABLE_SPECS = [
    spec for spec in list_specs() if {i.kind for i in spec.data.inputs} & {"matrix", "panel"}
]


@pytest.mark.parametrize("spec", TABLE_SPECS, ids=lambda spec: spec.name)
def test_shifting_the_columns_keeps_every_methods_graphs(spec):
    if spec.name == "calm":
        pytest.importorskip("torch")
    fit = getattr(andrey, spec.name)
    data = chain()
    want = marks(fit(*inputs(spec, data, (1.0, 0.0)), **SHORT.get(spec.name, {})))
    for shift in SHIFTS.values():
        got = marks(fit(*inputs(spec, data, shift), **SHORT.get(spec.name, {})))
        assert len(got) == len(want)
        for have, expected in zip(got, want, strict=True):
            np.testing.assert_array_equal(have, expected, err_msg=f"{shift}")


@pytest.mark.parametrize("shift", SHIFTS.values(), ids=SHIFTS.keys())
@pytest.mark.parametrize(
    ("method", "case"),
    [
        ("rcd", "confounded_fork_4v"),
        ("rcd", "fork_5v_observed"),
        ("gin", "measurement_6v"),
        ("gin", "measurement_chain_9v"),
    ],
)
@pytest.mark.parametrize("replicate", range(3))
def test_rcd_and_gin_keep_their_graph_on_shifted_columns(method, case, replicate, shift):
    scale, offset = shift
    data = on_grid(datagen.generate(case, replicate=replicate))
    fit = getattr(andrey, method)
    want, got = fit(data), fit(data * scale + offset)
    for have, expected in zip(marks(got), marks(want), strict=True):
        np.testing.assert_array_equal(have, expected)
    assert got.metadata == want.metadata


@pytest.mark.parametrize("offset", [1e3, 1e6, 1e9])
def test_gamma_hsic_does_not_read_the_columns_offset(offset):
    rng = np.random.default_rng(0)
    x = on_grid(rng.uniform(-1, 1, N) ** 3)
    y = on_grid(x**2 + 0.3 * rng.standard_normal(N))
    want = hsic_gamma_from_grams(gamma_gram(x), gamma_gram(y), N)
    got = hsic_gamma_from_grams(gamma_gram(x + offset), gamma_gram(y - offset), N)
    assert got == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("shift", SHIFTS.values(), ids=SHIFTS.keys())
def test_varma_lingam_keeps_its_graph_on_shifted_columns(shift):
    """A series 400 rows by 9 columns, fit with one optimizer iteration, keeps its graphs."""
    scale, offset = shift
    data = on_grid(datagen.generate("measurement_chain_9v", replicate=4))
    want = andrey.varma_lingam(data, max_iter=1)
    got = andrey.varma_lingam(data * scale + offset, max_iter=1)
    for have, expected in zip(marks(got), marks(want), strict=True):
        np.testing.assert_array_equal(have, expected)
    assert list(got.ordering) == list(want.ordering)
