"""IGSP recovers the true interventional essential graph in its papers' settings.

Yang, Katcoff, and Uhler (2018, Section 5): Erdos-Renyi DAGs on 10 variables with mean degree 1.5,
weights uniform on [0.25, 1] in absolute value, standard normal noise, an observational
environment and 4 settings, each a perfect intervention on one variable that cuts it from its
causes and keeps its noise. Wang et al. (2017, Section 5.1): the same graphs with weights uniform on
[0.1, 1] in absolute value and one setting of 5 variables, each cut from its causes and given the
noise N(0, 0.2). Both with 10,000 rows per environment. A perfect intervention that keeps a root's
noise leaves its distribution unchanged, so the truth is the essential graph under the targets the
interventions change. The fits run at the facade's defaults.

Wang et al. (2017, Figure 1): the 7-variable DAG on which the score search of GIES is not
consistent, with the setting ``{4}``, ``{5}``, or ``{4, 5}`` and 100,000 rows per environment.

The recorded output is the graph and objective of the ``gauss_5v_noise_shift`` baseline, whose two
settings each change the noise of one variable, both given as targets.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
import andrey.data as data
from andrey.core import ARROW, TAIL
from andrey.data.interventions import hard
from andrey.metrics import shd
from tests.recovery.helpers import datagen
from tests.recovery.helpers.essential import essential
from tests.recovery.helpers.harness import assert_recorded_output

P = 10
#: Each paper's weights, intervention, and settings: (number of settings, targets per setting).
SETTINGS = {
    "yang": ((0.25, 1.0), hard(0.0, sd=1.0), (4, 1)),
    "wang": ((0.1, 1.0), hard(0.0, sd=0.2**0.5), (1, 5)),
}
#: Wang et al. (2017), Figure 1: 5 is the collider of 3 and 7.
FIGURE_1 = data.graphs.FixedDAG(
    nodes=("1", "2", "3", "4", "5", "6", "7"),
    arcs=(("1", "4"), ("2", "4"), ("3", "4"), ("3", "5"), ("7", "5"), ("4", "6")),
)


def paper_dataset(paper: str, seed: int, n: int):
    """The dataset and each environment's targets, by column position."""
    weights, intervention, (settings, size) = SETTINGS[paper]
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=P, avg_degree=1.5),
        functional=data.functional.linear(weight_range=weights),
        noise=data.noise.gaussian(),
    )
    drawn = np.random.default_rng([seed, 655]).permutation(P)[: settings * size]
    chosen = [sorted(int(v) for v in drawn[k * size : (k + 1) * size]) for k in range(settings)]
    envs = [{}, *({v: intervention for v in t} for t in chosen)]
    ds = scm.sample(n=[n] * (settings + 1), seed=seed, scale="raw", envs=envs)
    return ds, {0: [], **{k + 1: t for k, t in enumerate(chosen)}}


def effective(ds, targets: dict[int, list[int]], keeps_root_noise: bool) -> list[set[int]]:
    """Each setting's targets whose distribution the intervention changes: all of them, less the
    roots when a perfect intervention keeps their noise."""
    dag = ds.graph.to_numpy()
    roots = {j for j in range(dag.shape[0]) if not ((dag[:, j] == TAIL) & (dag[j] == ARROW)).any()}
    return [set(t) - (roots if keeps_root_noise else set()) for k, t in targets.items() if k]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("paper", list(SETTINGS))
def test_igsp_recovers_the_paper_setting(paper, seed):
    ds, targets = paper_dataset(paper, seed, 10_000)
    out = andrey.igsp(ds.data, ds.env, targets=targets)
    truth = essential(ds.graph.to_numpy(), effective(ds, targets, paper == "yang"))
    errors = shd(out.structure, truth)
    assert errors < len(truth.oriented_edges())  # the empty graph's SHD
    # Recorded: 0 on every dataset. One pair of slack for a test decision at the level that the
    # last bits of another machine's arithmetic move.
    assert errors <= 1


@pytest.mark.parametrize("family", [["4"], ["5"], ["4", "5"]])
def test_igsp_recovers_the_dag_where_gies_is_not_consistent(family):
    columns = {name: j for j, name in enumerate(FIGURE_1.nodes)}
    scm = data.SCM(
        graph=FIGURE_1,
        functional=data.functional.linear(weight_range=(0.25, 1.0)),
        noise=data.noise.gaussian(),
    )
    setting = {columns[v]: hard(0.0, sd=0.2**0.5) for v in family}
    ds = scm.sample(n=[100_000] * 2, seed=0, scale="raw", envs=[{}, setting], shuffle=False)
    out = andrey.igsp(ds.data, ds.env, targets={0: [], 1: sorted(setting)})
    truth = essential(ds.graph.to_numpy(), [set(setting)])
    assert np.array_equal(out.structure.to_numpy(), truth.to_numpy())


def test_recorded_baseline():
    case = "gauss_5v_noise_shift"
    column = list(datagen.column_order(case)).index
    targets = {0: [], 1: [column(3)], 2: [column(4)]}
    out = andrey.igsp(datagen.generate(case), datagen.environment_index(case), targets=targets)
    assert_recorded_output(out, "IGSP", case=case)
