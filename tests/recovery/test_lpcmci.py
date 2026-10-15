"""LPCMCI recovery: a planted process with a hidden common cause, and the null case."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.data import temporal

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

# Six variables; X5 is white noise and hidden. B0[effect, cause] within one step:
# X5 -> X1, X5 -> X2, X2 -> X3. A1[effect, cause] at lag 1: memory in X0..X4, X0 -> X1, X3 -> X4.
B0 = np.zeros((6, 6))
B0[1, 5], B0[2, 5], B0[3, 2] = 0.7, 0.6, 0.6
LAG_1 = np.diag([0.5, 0.4, 0.5, 0.4, 0.5, 0.0])
LAG_1[1, 0] = LAG_1[4, 3] = 0.5

# Over X0..X4 with tau_max=2 the hidden cause is a two-headed link X1 <-> X2; every other planted
# link is a definite cause.
TRUE_LINKS = {
    "1<->2@0",
    "2-->3@0",
    "0-->1@1",
    "3-->4@1",
    *(f"{i}-->{i}@1" for i in range(5)),
}


def _links(out: andrey.StructureOutput) -> set[str]:
    """Every link as ``"i<marks>j@lag"``, a lag-0 link once with ``i < j``."""
    return {
        f"{i}{link}{j}@{lag}"
        for lag, block in enumerate(out.metadata["links"])
        for i, row in enumerate(block)
        for j, link in enumerate(row)
        if link and (lag > 0 or i < j)
    }


@pytest.fixture(scope="module")
def planted() -> np.ndarray:
    innovations = np.random.default_rng(1).standard_normal((1200, 6))
    x = temporal.generate(np.linalg.inv(np.eye(6) - B0), LAG_1[None], None, innovations, 200)
    return x[:, :5]


@pytest.mark.parametrize("indep_test", ["student_t", "fisherz"])
def test_recovers_the_planted_graph_with_its_hidden_cause(planted, indep_test):
    out = andrey.lpcmci(planted, tau_max=2, indep_test=indep_test)
    assert _links(out) == TRUE_LINKS
    assert out.structure.lag(0).oriented_edges() == [(1, 2, "bidirected"), (2, 3, "directed")]
    assert out.structure.lag(1).oriented_edges() == [
        (0, 0, "directed"),
        (0, 1, "directed"),
        (1, 1, "directed"),
        (2, 2, "directed"),
        (3, 3, "directed"),
        (3, 4, "directed"),
        (4, 4, "directed"),
    ]
    assert out.structure.lag(2).oriented_edges() == []
    assert out.metadata["conflicts"] == []


def test_same_result_on_every_run(planted):
    first, second = (andrey.lpcmci(planted, tau_max=2) for _ in range(2))
    assert first.structure == second.structure
    assert first.metadata == second.metadata


def test_the_result_does_not_depend_on_the_variable_order(planted):
    order = [3, 0, 4, 2, 1]
    out = andrey.lpcmci(planted, tau_max=2)
    permuted = andrey.lpcmci(planted[:, order], tau_max=2)
    back = set()
    for link in _links(permuted):
        i, marks, rest = int(link[0]), link[1:4], link[4:]
        j, lag = int(rest[0]), int(rest.split("@")[1])
        i, j = order[i], order[j]
        if lag == 0 and i > j:
            i, j, marks = j, i, marks[::-1].replace("<", "#").replace(">", "<").replace("#", ">")
        back.add(f"{i}{marks}{j}@{lag}")
    assert back == _links(out)


def test_independent_autocorrelated_series_give_only_autodependencies():
    innovations = np.random.default_rng(1).standard_normal((700, 4))
    data = temporal.generate(np.eye(4), 0.9 * np.eye(4)[None], None, innovations, burn_in=200)
    out = andrey.lpcmci(data, tau_max=2, pc_alpha=0.01)
    assert _links(out) == {f"{i}-->{i}@1" for i in range(4)}
