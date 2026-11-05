"""GnIES recovers the targets and the interventional essential graph of noise interventions.

The fixed case has one certain answer: a cause whose effect's noise variance changes between two
environments gives the directed edge, with the effect as the one target. The ground truth is
linear-Gaussian data from ``andrey.data`` with an observational environment and three that each
scale one variable's noise variance, at 10 variables and 10,000 rows per environment: the true
class and targets on each of the first five seeds the shipped defaults recover (seeds 0-19
recovered on 19 of 20). The recorded outputs are the marks and targets of three small draws, and
the graph, score, and targets of the ``gauss_5v_noise_shift`` baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import GraphStructure
from andrey.core.orient import to_structure
from andrey.data import SCM, graphs
from andrey.data.functional import linear
from andrey.data.interventions import soft
from andrey.data.noise import gaussian
from andrey.metrics import shd
from andrey.search.gies import replace_unprotected
from andrey.search.gnies import singleton_cut
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output


def _draw(
    seed: int, sizes: int | list[int], d: int, avg_degree: float, n_targets: int
) -> tuple[np.ndarray, np.ndarray, list[int], GraphStructure]:
    """Signed weights in ``[0.5, 1]``, unit noise variance, and ``n_targets`` single-target
    environments beside an observational one, each multiplying its target's noise variance by
    ``U[5, 10] / U[1, 2]``; standardized on the pooled rows. Returns the data, each row's
    environment, the targets, and the true interventional essential graph."""
    scm = SCM(
        graph=graphs.erdos_renyi(d, avg_degree=avg_degree),
        functional=linear(weight_range=(0.5, 1.0)),
        noise=gaussian(),
    )
    rng = np.random.default_rng([d, seed])
    targets = sorted(int(j) for j in rng.choice(d, size=n_targets, replace=False))
    ratios = [rng.uniform(5, 10) / rng.uniform(1, 2) for _ in targets]
    envs = [{}] + [{j: soft(noise_scale=float(np.sqrt(r)))} for j, r in zip(targets, ratios)]
    ds = scm.sample(n=sizes, seed=seed, envs=envs, scale="raw")
    X = (ds.data - ds.data.mean(axis=0)) / ds.data.std(axis=0)
    dag = np.array(ds.graph.adjacency, dtype=np.int8)
    truth = to_structure(replace_unprotected(dag, singleton_cut(targets, d)), kind="cpdag")
    return X, np.asarray(ds.env), targets, truth


def _targets(out: andrey.StructureOutput) -> list:
    return out.metadata["changing_variables"]


def test_effect_with_a_changed_noise_variance_is_the_target():
    rng = np.random.default_rng(0)
    env = np.repeat([0, 1], 1000)
    x = rng.standard_normal(2000)
    y = 0.8 * x + rng.standard_normal(2000) * np.where(env == 1, 3.0, 1.0)
    out = andrey.gnies(np.column_stack([x, y]), env)
    assert out.structure.oriented_edges() == [(0, 1, "directed")]
    assert _targets(out) == [1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5])
def test_recovers_the_class_and_targets(seed):
    X, env, targets, truth = _draw(seed, 10_000, d=10, avg_degree=2.7, n_targets=3)
    out = andrey.gnies(X, env)
    assert shd(out.structure, truth) == 0
    assert _targets(out) == targets


#: Each small draw's marks and targets, as ``andrey.gnies`` returned them at its defaults.
RECORDED = {
    0: (
        [
            (0, 1, "undirected"),
            (1, 2, "directed"),
            (2, 3, "directed"),
            (2, 4, "directed"),
            (3, 4, "directed"),
        ],
        [2, 4],
    ),
    1: (
        [
            (2, 0, "directed"),
            (0, 3, "directed"),
            (1, 3, "directed"),
            (1, 4, "undirected"),
            (2, 3, "directed"),
        ],
        [0, 3],
    ),
    2: (
        [
            (1, 0, "directed"),
            (2, 0, "directed"),
            (2, 1, "directed"),
            (2, 3, "directed"),
            (2, 4, "undirected"),
        ],
        [1, 3],
    ),
}


@pytest.mark.parametrize("seed", sorted(RECORDED))
def test_recorded_output(seed):
    X, env, _, _ = _draw(seed, [400, 300, 200], d=5, avg_degree=2.0, n_targets=2)
    out = andrey.gnies(X, env)
    assert (out.structure.oriented_edges(), _targets(out)) == RECORDED[seed]


def test_recorded_baseline():
    case = "gauss_5v_noise_shift"
    out = andrey.gnies(datagen.generate(case), datagen.environment_index(case))
    assert_recorded_output(out, "GnIES", case=case)
