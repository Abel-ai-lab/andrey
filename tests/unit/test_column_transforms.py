"""A change of units, or a transform a test cannot see, changes no method's graph.

Rescaling each column by its own positive factor is a change of units, and a method returns the
same graph in any units unless its meaning requires otherwise: the methods whose registry entry sets
``reads_units`` are those exceptions, and their graph changes here. The factors are powers of two
from about ``1e-3`` to ``1e3``: a power of two rescales a float exactly, so both fits see the same
values up to the units, and any difference comes from the method. A categorical column keeps its
codes.

A rank-based test reads each column only through its ranks, so under ``indep_test="normal_scores"``
a strictly increasing transform of each column, each its own, gives the same graph. A method that
also scores the data with ``score_func`` reads the values through its score, so the check covers the
methods whose only reading of the data is the test. A categorical test, and a categorical score,
read a column only through which rows share a category, so recoding each column's categories with
other whole numbers in the same order gives the same graph under ``"chisq"`` and ``"gsq"``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest

import andrey
from andrey.core import TemporalStructure
from andrey.spec import list_specs
from tests.unit.methods import fit_settings

N = 300
P = 6
#: Each column's factor: powers of two from ``2**-10`` to ``2**10``, out of order across columns.
FACTORS = 2.0 ** np.array([-10, 7, -3, 10, -7, 3])
#: A strictly increasing transform of each column, each its own.
MONOTONE: tuple[Callable[[np.ndarray], np.ndarray], ...] = (
    np.exp,
    lambda x: x**3,
    lambda x: np.log(x - x.min() + 1.0),
    np.sinh,
    lambda x: x + x**3,
    lambda x: -np.exp(-x),
)
#: Each column's categories 0, 1, 2 recoded as other whole numbers in the same order.
RECODE = np.array([[-4, 1, 7], [10, 20, 30], [0, 5, 6], [-9, -8, 40], [2, 3, 100], [-1, 0, 1]])
#: Three environments in row order: observational, one that cuts the edge 0 -> 2, and one that
#: shifts the noise of 3.
ENV = np.arange(N) * 3 // N
#: The settings of a fit on the environments: the targets of gies, igsp, and ut_igsp, which reads
#: the third environment's as unknown, and the variable ICP explains, 4, whose parent 3 shifts.
ENV_SETTINGS = {
    "gies": {"targets": {0: [], 1: [2], 2: [3]}},
    "icp": {"target": 4},
    "igsp": {"targets": {0: [], 1: [2], 2: [3]}},
    "ut_igsp": {"targets": {0: [], 1: [2], 2: None}},
}
#: CALM's schedule here: the package-wide one finds no edge on these data.
CALM = {"max_iter": 3, "subproblem_iter": 300}
TABLE_SPECS = [
    spec for spec in list_specs() if {i.kind for i in spec.data.inputs} & {"matrix", "panel"}
]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        warnings.simplefilter("ignore", andrey.SearchLimitWarning)
        yield


def dag(seed: int = 0) -> np.ndarray:
    """A linear non-Gaussian DAG 0 -> 1 -> 3 -> 4, 0 -> 2 -> 3, 2 -> 5 over the environments of
    :data:`ENV`, so every method family finds structure."""
    u = np.random.default_rng(seed).uniform(-1, 1, size=(N, P)) ** 3
    u[ENV == 2, 3] += 1.0
    x = u.copy()
    x[:, 1] += 1.2 * x[:, 0]
    x[:, 2] += -0.8 * x[:, 0] * (ENV != 1)
    x[:, 3] += 0.9 * x[:, 1] + 0.7 * x[:, 2]
    x[:, 4] += 1.1 * x[:, 3]
    x[:, 5] += 0.8 * x[:, 2]
    return x


def series(seed: int = 0) -> np.ndarray:
    """A non-Gaussian series in which each variable follows its own past, with 0 -> 1 -> 2 and
    3 -> 4 at lag 1 and 1 -> 5 at lag 0."""
    e = np.random.default_rng(seed).uniform(-1, 1, size=(N, P)) ** 3
    x = np.zeros((N, P))
    for t in range(N):
        past = x[t - 1] if t else np.zeros(P)
        x[t] = 0.4 * past + e[t]
        x[t, 1] += 0.8 * past[0]
        x[t, 2] += 0.7 * past[1]
        x[t, 4] += -0.8 * past[3]
        x[t, 5] += 0.9 * x[t, 1]
    return x


def categories(x: np.ndarray) -> np.ndarray:
    """Each column of ``x`` cut at its terciles into the categories 0, 1, 2."""
    edges = np.quantile(x, [1 / 3, 2 / 3], axis=0)
    return np.stack([np.digitize(x[:, j], edges[:, j]) for j in range(x.shape[1])], axis=1)


def with_category(x: np.ndarray) -> np.ndarray:
    """``x`` with its last column replaced by the categories of column 2, a categorical child."""
    out = x.copy()
    out[:, -1] = categories(x[:, [2]])[:, 0]
    return out


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


def always_env(spec) -> bool:
    """Whether every fit of the method reads environments: those it requires, gnies's, and those
    that igsp's and ut_igsp's targets need."""
    required = any(i.kind == "context" and i.required for i in spec.data.inputs)
    return required or spec.name in ("gnies", "igsp", "ut_igsp")


def fit(spec, change: Callable[[np.ndarray], np.ndarray], *, env: bool = False, **settings):
    """The method's fit on its inputs, each data matrix passed through ``change``: a series for a
    time-series method, a pair of columns for a pairwise one, and the DAG otherwise. With ``env``,
    or when the method always reads them, the fit also reads the environments."""
    env = env or always_env(spec)
    data = series if spec.family == "temporal" else dag
    args = []
    for inp in spec.data.inputs:
        if inp.kind == "context" and env:
            args.append(ENV)
        elif inp.kind == "matrix":
            args.append(change(data(0)[:, :2] if spec.family == "pairwise" else data(0)))
        elif inp.kind == "panel":
            args.append([change(data(seed)) for seed in range(3)])
    extra = ENV_SETTINGS.get(spec.name, {}) if env else {}
    schedule = CALM if spec.name == "calm" else {}
    kwargs = {**fit_settings(spec.name), **schedule, **extra, **settings}
    return getattr(andrey, spec.name)(*args, **kwargs)


def same(got, want) -> bool:
    """Whether two fits hold the same graphs, mark for mark."""
    got, want = marks(got), marks(want)
    return len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want))


def found(out) -> bool:
    """Whether a fit found an edge, so a comparison with it can fail."""
    return any(m.any() for m in marks(out))


def rescale(x: np.ndarray) -> np.ndarray:
    return x * FACTORS[: x.shape[1]]


@pytest.mark.parametrize("spec", TABLE_SPECS, ids=lambda spec: spec.name)
def test_rescaling_each_column_keeps_the_graph_unless_the_method_reads_units(spec):
    if spec.name == "calm":
        pytest.importorskip("torch")
    want = fit(spec, lambda x: x)
    assert found(want)
    assert same(fit(spec, rescale), want) != spec.reads_units


@pytest.mark.parametrize(
    "spec",
    [
        spec
        for spec in TABLE_SPECS
        if any(i.kind == "context" for i in spec.data.inputs) and not always_env(spec)
    ],
    ids=lambda spec: spec.name,
)
def test_rescaling_each_column_keeps_the_graph_across_environments(spec):
    want = fit(spec, lambda x: x, env=True)
    assert found(want)
    assert same(fit(spec, rescale, env=True), want)


@pytest.mark.parametrize(
    "spec",
    [spec for spec in TABLE_SPECS if any(p.name == "categorical" for p in spec.params)],
    ids=lambda spec: spec.name,
)
def test_rescaling_each_continuous_column_of_mixed_data_keeps_the_graph(spec):
    want = fit(spec, with_category, categorical=[P - 1])
    assert found(want)
    got = fit(spec, lambda x: with_category(rescale(x)), categorical=[P - 1])
    assert same(got, want)


def accepts(spec, test: str) -> bool:
    return any(p.name == "indep_test" and test in (p.choices or ()) for p in spec.params)


def scores(spec) -> bool:
    return any(p.name == "score_func" for p in spec.params)


def monotone(x: np.ndarray) -> np.ndarray:
    return np.stack([f(x[:, j]) for j, f in enumerate(MONOTONE[: x.shape[1]])], axis=1)


#: The methods that take ``normal_scores`` and read the data only through their test; the others
#: that take it, gfci, mmhc, rsmax2, boss_fci, and grasp_fci, also take ``score_func``, whose score
#: reads the values, so a monotone transform can change their graph.
RANK_TESTED = [spec for spec in TABLE_SPECS if accepts(spec, "normal_scores") and not scores(spec)]


@pytest.mark.parametrize("spec", RANK_TESTED, ids=lambda spec: spec.name)
def test_a_monotone_transform_of_each_column_keeps_the_normal_scores_graph(spec):
    want = fit(spec, lambda x: x, indep_test="normal_scores")
    assert found(want)
    assert same(fit(spec, monotone, indep_test="normal_scores"), want)


def recoded(x: np.ndarray) -> np.ndarray:
    codes = categories(x)
    return np.stack([RECODE[j][codes[:, j]] for j in range(x.shape[1])], axis=1).astype(float)


#: Each categorical test, and the categorical score a method that also scores pairs with it.
CATEGORICAL = {"chisq": "bdeu", "gsq": "discrete_bic"}


@pytest.mark.parametrize("test", CATEGORICAL)
@pytest.mark.parametrize(
    "spec", [spec for spec in TABLE_SPECS if accepts(spec, "gsq")], ids=lambda spec: spec.name
)
def test_recoding_each_columns_categories_in_order_keeps_the_graph(spec, test):
    settings = {"indep_test": test} | ({"score_func": CATEGORICAL[test]} if scores(spec) else {})
    want = fit(spec, lambda x: categories(x).astype(float), **settings)
    assert found(want)
    assert same(fit(spec, recoded, **settings), want)
