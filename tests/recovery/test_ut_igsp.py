"""UT-IGSP recovers the true interventional essential graph and targets in its paper's setting.

The setting is Squires, Wang, and Uhler (2020, Section 5.1): Erdos-Renyi DAGs on 20 variables with
mean degree 1.5, weights uniform on [0.25, 1] in absolute value, standard normal noise, an
observational environment and 5 settings with n rows each. Each setting intervenes on one known
target and 0 or 3 unknown ones: a shift of the noise mean to 1, or a perfect intervention that sets
the target to N(1, 0.1). The fits run at the facade's defaults. The recorded output is the graph,
objective, and targets of the ``gauss_5v_noise_shift`` baseline, whose two settings each change
the noise of one variable, neither of them given.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
import andrey.data as data
from andrey.core import ARROW, TAIL
from andrey.data.interventions import hard, soft
from andrey.search.gies import cut_pairs, replace_unprotected
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output

P, K = 20, 5
INTERVENTIONS = {"shift": soft(shift=1.0), "perfect": hard(1.0, sd=0.1**0.5)}
#: Each dataset's recorded SHD and target errors (false plus missed targets over the settings).
RECORDED = {
    (seed, off, kind): (0, 0) for seed in range(3) for off in (0, 3) for kind in INTERVENTIONS
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


def paper_dataset(seed: int, n: int, off: int, kind: str):
    """The dataset, with ``off`` unknown targets per setting, and the known targets."""
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=P, avg_degree=1.5),
        functional=data.functional.linear(weight_range=(0.25, 1.0)),
        noise=data.noise.gaussian(),
    )
    designed = np.random.default_rng([seed, 654]).choice(P, size=K, replace=False)
    envs: list[dict] = [{}]
    for k, target in enumerate(designed):
        others = [v for v in range(P) if v != target]
        unknown = np.random.default_rng([seed, 654, off, k]).choice(others, off, replace=False)
        envs.append({int(v): INTERVENTIONS[kind] for v in (target, *unknown)})
    ds = scm.sample(n=[n] * (K + 1), seed=seed, scale="raw", envs=envs)
    known = {0: [], **{k + 1: [int(t)] for k, t in enumerate(designed)}}
    return ds, known


def essential(dag: np.ndarray, targets: dict[int, list[int]]) -> np.ndarray:
    """The interventional essential graph of the DAG's marks under the observational setting and
    each setting's targets."""
    family = np.zeros((len(targets) + 1, dag.shape[0]), dtype=bool)
    for row, variables in enumerate(targets.values(), start=1):
        family[row, variables] = True
    return replace_unprotected(dag.astype(np.int8), cut_pairs(family))


def shd(a: np.ndarray, b: np.ndarray) -> int:
    """The variable pairs whose endpoint marks differ."""
    upper = np.triu_indices(a.shape[0], 1)
    return int(((a[upper] != b[upper]) | (a.T[upper] != b.T[upper])).sum())


@pytest.mark.parametrize("kind", list(INTERVENTIONS))
@pytest.mark.parametrize("off", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_ut_igsp_recovers_the_paper_setting(seed, off, kind):
    ds, known = paper_dataset(seed, 5000, off, kind)
    out = andrey.ut_igsp(ds.data, ds.env, targets=known)
    truth = essential(ds.graph.to_numpy(), {k: t for k, t in ds.targets.items() if k})
    found = {int(k): v for k, v in out.metadata["targets"].items()}
    target_errors = sum(len(set(found[k]) ^ set(ds.targets[k])) for k in range(1, K + 1))
    errors = shd(out.structure.to_numpy(), truth)
    recorded_shd, recorded_targets = RECORDED[seed, off, kind]
    assert errors < int((truth != 0).sum() // 2)  # the empty graph's SHD
    assert errors <= recorded_shd + 1
    assert target_errors <= recorded_targets + 1


@pytest.mark.parametrize("kind", list(INTERVENTIONS))
def test_a_chain_with_an_unknown_and_a_known_target_is_fully_oriented(kind):
    """The chain 0 -> 1 -> 2, intervened on 0 in a setting with unknown targets and on 2 in one
    where 2 is known: both edges are oriented and both targets found."""
    rng = np.random.default_rng(0)
    n = 2000
    env = np.repeat([0, 1, 2], n)
    noise = rng.standard_normal((3 * n, 3))

    def intervened(rows: np.ndarray, value: np.ndarray) -> np.ndarray:
        if kind == "shift":
            return np.where(rows, value + 1.0, value)
        return np.where(rows, 1.0 + 0.1**0.5 * rng.standard_normal(3 * n), value)

    x0 = intervened(env == 1, noise[:, 0])
    x1 = 0.8 * x0 + noise[:, 1]
    x2 = intervened(env == 2, 0.8 * x1 + noise[:, 2])
    out = andrey.ut_igsp(np.column_stack([x0, x1, x2]), env, targets={0: [], 2: [2]})
    assert out.structure.oriented_edges() == [(0, 1, "directed"), (1, 2, "directed")]
    assert out.metadata["targets"] == {"0": [], "1": [0], "2": [2]}
    dag = np.zeros((3, 3), dtype=np.int8)
    dag[0, 1] = dag[1, 2] = TAIL
    dag[1, 0] = dag[2, 1] = ARROW
    assert np.array_equal(out.structure.to_numpy(), dag)


def test_recorded_baseline():
    case = "gauss_5v_noise_shift"
    env = datagen.environment_index(case)
    out = andrey.ut_igsp(datagen.generate(case), env, targets={0: [], 1: None, 2: None})
    assert_recorded_output(out, "UT-IGSP", case=case)
