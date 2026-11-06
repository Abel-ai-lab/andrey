"""Recovery checks for BOSS-FCI and GRaSP-FCI: GFCI's FCI phase on a BOSS or GRaSP CPDAG.

Each hybrid is checked to equal FCI's phase applied to its own search's CPDAG. Fixed cases are a
planted hidden common cause and independent columns. Ground truth comes from simulated
linear-Gaussian data with hidden variables at the paper's sizes, from data simulated on the
published ASIA network with one node hidden, and from the Sachs measurements against their
consensus network. Mixed data, determinism, column-order invariance, the recorded outputs, and
input validation complete them.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core import LATENT, OBSERVED, GraphStructure
from andrey.data import load_dataset
from andrey.data.latent import marginal
from andrey.metrics import shd, skeleton_scores
from andrey.search import gfci as engines
from andrey.search.boss import boss
from andrey.search.ges import ges
from andrey.search.grasp import grasp
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output
from tests.unit.conftest import serve_datasets_offline

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

HYBRIDS = ("boss_fci", "grasp_fci")
FIT = {
    "boss_fci": lambda x, **params: andrey.boss_fci(x, **params),
    "grasp_fci": lambda x, **params: andrey.grasp_fci(x, **params),
}
_SEARCH = {"boss_fci": boss, "grasp_fci": grasp}
_RECORDED = {"boss_fci": "BOSSFCI", "grasp_fci": "GRaSPFCI"}
_FCI_PHASE = dict(
    alpha=0.05,
    indep_test="fisherz",
    collider_rule="sepsets",
    categorical=None,
    possible_dsep="order_free",
    in_column_order=False,
)


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """The Sachs and ASIA datasets, served from the unit tests' fixtures with no network."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        serve_datasets_offline(monkeypatch, tmp_path_factory.mktemp("cache"))
        yield {
            "sachs": load_dataset("sachs", as_frame=False),
            "asia": load_dataset("asia", n=2000, seed=0, as_frame=False),
        }


def _empty_errors(truth: GraphStructure) -> int:
    empty = GraphStructure.from_numpy(np.zeros((truth.n_nodes,) * 2, dtype=np.int8), kind="pag")
    return shd(empty, truth, endpoint_aware=True)


# --- the composition ------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_each_hybrid_is_the_fci_phase_on_its_search(seed):
    x = data.sample_scm(d=12, n=600, seed=seed, density=3.0, latents=1).data
    for name in HYBRIDS:
        pattern, _ = _SEARCH[name](x, ebic_gamma=0.0)  # the standard BIC, as the facades
        expected = engines.fci_from_pattern(x, pattern, **_FCI_PHASE)
        assert FIT[name](x).structure == expected
    pattern, _ = ges(x, ebic_gamma=0.0)
    assert andrey.gfci(x).structure == engines.fci_from_pattern(x, pattern, **_FCI_PHASE)


# --- fixed cases ----------------------------------------------------------------------------------

# Observed columns a, b, c, d, e of ``confounded_gauss_5v``; h is hidden.
_LATENT = "confounded_gauss_5v"
_PLANTED_PAG = [
    (0, 1, "partially_directed"),  # a o-> b
    (1, 2, "bidirected"),  # b <-> c: the hidden common cause
    (2, 4, "directed"),  # c -> e
    (3, 2, "partially_directed"),  # d o-> c
]


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_recovers_the_planted_pag(name):
    x = datagen.generate(_LATENT)
    assert sorted(datagen.to_variables(_LATENT, datagen.planted_pag(_LATENT)).oriented_edges()) == (
        _PLANTED_PAG
    )
    found = datagen.to_variables(_LATENT, FIT[name](x).structure)
    assert sorted(found.oriented_edges(index=True)) == _PLANTED_PAG


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_finds_no_edges_between_independent_columns(name):
    x = np.random.default_rng(0).standard_normal((1000, 4))
    out = FIT[name](x)
    assert out.structure.oriented_edges() == []
    assert out.structure.kind == "pag"


# --- ground truth ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", HYBRIDS)
@pytest.mark.parametrize(
    "params", [{}, {"penalty_discount": 2.0, "alpha": 0.01}], ids=["defaults", "paper"]
)
def test_hybrid_recovers_the_pag_near_the_papers_size(name, params):
    """20 measured linear-Gaussian variables, 4 hidden common causes, mean degree 4, and 1,000
    rows, the paper's sizes on ``sample_scm``'s defaults, against the true PAG, at the defaults and
    at the paper's penalty discount and alpha. Nearly every adjacency found is true, and the
    endpoint errors stay below 0.85 times the empty graph's."""
    ds = data.sample_scm(d=24, n=1000, seed=0, density=4.0, latents=4)
    truth = marginal(ds.graph, "pag")
    found = FIT[name](ds.data, **params).structure
    assert skeleton_scores(found, truth)["precision"] >= 0.9
    assert shd(found, truth, endpoint_aware=True) < 0.85 * _empty_errors(truth)


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_recovers_the_asia_pag_with_smoking_hidden(name, datasets):
    """Data simulated on the ASIA network, with ``smoke`` (the cause of lung and bronc) hidden."""
    asia = datasets["asia"]
    hidden = asia.graph.labels.index("smoke")
    order = [j for j in range(asia.graph.n_nodes) if j != hidden] + [hidden]
    types = np.array([OBSERVED] * (len(order) - 1) + [LATENT], dtype=np.int8)
    dag = GraphStructure.from_numpy(
        asia.graph.to_numpy()[np.ix_(order, order)], kind="dag", node_types=types
    )
    truth = marginal(dag, "pag")
    found = FIT[name](np.asarray(asia.data)[:, order[:-1]]).structure
    assert shd(found, truth, endpoint_aware=True) == 0


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_finds_only_known_sachs_adjacencies(name, datasets):
    """The Sachs measurements: every adjacency found is in the consensus network.

    The network has 17 of the 55 possible adjacencies, so a guess would be right about 31% of the
    time; PC, FCI, and GFCI also find 8 adjacencies, all in it.
    """
    sachs = datasets["sachs"]
    scores = skeleton_scores(FIT[name](sachs.data).structure, sachs.graph)
    assert scores["precision"] == 1.0
    assert scores["recall"] > 0.4


# --- mixed data -----------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_runs_the_mixed_data_score_and_test(name):
    sample = data.sample_mixed(d=5, n=3000, seed=0)
    out = FIT[name](sample.data, score_func="degen_gauss", indep_test="degen_gauss")
    assert out.structure.kind == "pag"
    out.structure.validate()
    assert "categorical" in out.metadata


# --- contract -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_matches_the_recorded_output(name):
    out = FIT[name](datagen.generate("gauss_5v"))
    assert_recorded_output(out, _RECORDED[name], case="gauss_5v")


def test_n_starts_reaches_the_search():
    """More starts hand the test phase the best-scoring search's CPDAG; on this input three starts
    end at another CPDAG than one start, for both searches."""
    x = data.sample_scm(d=12, n=600, seed=7, density=4.0, latents=1).data
    for name in HYBRIDS:
        pattern, _ = _SEARCH[name](x, ebic_gamma=0.0, n_starts=3)
        assert pattern != _SEARCH[name](x, ebic_gamma=0.0)[0]
        expected = engines.fci_from_pattern(x, pattern, **_FCI_PHASE)
        assert FIT[name](x, n_starts=3).structure == expected


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_is_deterministic_for_a_seed(name):
    x = data.sample_scm(d=10, n=600, seed=4, density=3.0, latents=1).data
    first = FIT[name](x, seed=3)
    assert first.metadata["algorithm"] == name
    assert FIT[name](x, seed=3).structure == first.structure


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_is_invariant_to_column_order(name):
    x = data.sample_scm(d=10, n=600, seed=5, density=3.0, latents=1).data
    marks = FIT[name](x).structure.to_numpy()
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = FIT[name](x[:, order]).structure.to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)


@pytest.mark.parametrize("name", HYBRIDS)
@pytest.mark.parametrize(
    ("params", "error", "message"),
    [
        (dict(indep_test="degen_gauss"), ValueError, "read the data alike"),
        (dict(score_func="bdeu"), ValueError, "read the data alike"),
        (dict(alpha=1.5), ValueError, "alpha"),
        (dict(collider_rule="vote"), ValueError, "collider_rule"),
    ],
)
def test_hybrid_rejects_bad_settings(name, params, error, message):
    with pytest.raises(error, match=message):
        FIT[name](datagen.generate("gauss_5v"), **params)


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_rejects_an_unknown_test_before_the_score_search(name, monkeypatch):
    def unexpected_search(*args, **kwargs):
        pytest.fail("An unknown indep_test must be rejected before the score search")

    monkeypatch.setattr(engines, name.removesuffix("_fci"), unexpected_search)
    with pytest.raises(ValueError, match="indep_test"):
        FIT[name](datagen.generate("gauss_5v"), indep_test="fishrz")


def test_grasp_fci_rejects_a_grasp_depth_below_one():
    with pytest.raises(ValueError, match="grasp_depth"):
        andrey.grasp_fci(datagen.generate("gauss_5v"), grasp_depth=0)
