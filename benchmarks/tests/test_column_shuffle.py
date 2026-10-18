"""The column shuffle: a shuffled dataset is its unshuffled one with the columns relabeled.

Each check reads the permutation where the store records it, the registry's ``column_order``, and
compares the two datasets through it: the data columns, the truth on both node axes of every block,
and the identity. Hidden nodes stay last, so the permutation of the truth fixes them. Rows keep their
order, so a domain-shift dataset keeps its domains, and a multi-group order covers every group.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from andrey.data.sortability import causal_depth, index_correlation
from andrey_bench import datasets, regime_models
from andrey_bench.datasets import DatasetKey
from andrey_bench.window import ARROW, TAIL

#: A summary measured on columns in the generated order that names its datasets.
DATA_TYPES_SUMMARY = Path(__file__).resolve().parents[1] / "published" / "0.6.0" / "summary.json"

#: Regimes whose generators take the shuffle step, and those whose column order is already free.
SHUFFLED_REGIMES = sorted(
    regime for regime, spec in datasets.REGIMES.items() if spec[1] not in datasets.ORDER_FREE
)
ORDER_FREE_REGIMES = sorted(set(datasets.REGIMES) - set(SHUFFLED_REGIMES))


def _key(regime: str, seed: int = 2, **overrides) -> DatasetKey:
    """A small key of `regime`; seed 2 draws a column order other than the identity in each."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    d = 2 if regime.endswith("_pair") else 10
    density = datasets.TEMPORAL_REGIMES.get(regime, 1.0 if d == 2 else datasets.CANONICAL_DENSITY)
    if functional == "measurement":  # three latents with their indicators
        d, density = 12, float(regime_models.MODELS[functional].indicators)
    coordinates = {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": density,
        "d": d,
        "n": 60,
        "seed": seed,
        "latents": datasets.latents_for(regime, d),
        "times": 4 if functional == "panel2013" else 0,
    }
    return DatasetKey(**{**coordinates, **overrides})


@pytest.mark.parametrize("regime", SHUFFLED_REGIMES)
def test_a_shuffled_dataset_is_the_unshuffled_one_relabeled(regime: str, tmp_path) -> None:
    shuffled = _key(regime)
    plain = dataclasses.replace(shuffled, shuffle=False)
    registry = datasets.materialize([shuffled, plain], tmp_path)
    assert registry["n_datasets"] == 2
    data, truth = datasets.load(tmp_path, shuffled)
    plain_data, plain_truth = datasets.load(tmp_path, plain)
    columns = registry["keys"][shuffled.label]["column_order"]
    observed = plain_data.shape[-1]  # every group's columns in a multi-group dataset
    assert registry["keys"][plain.label]["column_order"] == list(range(observed))
    assert sorted(columns) == list(range(observed)) and columns != sorted(columns)

    np.testing.assert_array_equal(data, plain_data[:, columns])
    nodes = np.r_[columns, observed : plain_truth.shape[-1]]  # hidden nodes stay last
    np.testing.assert_array_equal(truth, plain_truth[..., nodes, :][..., nodes])

    assert shuffled.digest != plain.digest and shuffled.label == f"{plain.label}-shuffled"
    call = datasets.generation_kwargs(shuffled)
    assert call == {**datasets.generation_kwargs(plain), "shuffle": True}


def test_a_multi_group_order_is_one_order_of_the_variables_in_every_group() -> None:
    key = _key("multi_group_er")
    columns = np.asarray(datasets.generate_one(key)[3])
    within = columns.reshape(regime_models.N_GROUPS, key.d)
    within -= key.d * np.arange(regime_models.N_GROUPS)[:, None]  # each group keeps its block
    assert (within == within[0]).all()
    assert sorted(within[0]) == list(range(key.d)) and list(within[0]) != list(range(key.d))


def test_unshuffled_regime_model_keys_keep_their_identity() -> None:
    """A domain-shift or multi-group key keeps its pre-shuffle digest unshuffled; GIN's keeps it."""
    digests = {}
    for regime in ("domain_shift_er", "multi_group_er", "latent_measurement"):
        key = dataclasses.replace(_key(regime, seed=0), n=500)
        if key.functional in datasets.ORDER_FREE:
            digests[regime] = key.digest
        else:
            digests[regime] = (key.digest, dataclasses.replace(key, shuffle=False).digest)
    assert digests == {
        "domain_shift_er": ("800d32c36f12", "bdddcb42f327"),
        "multi_group_er": ("d8f51f5bfa1e", "88e37b41768c"),
        "latent_measurement": "073ce1386d4e",
    }


@pytest.mark.parametrize("regime", ORDER_FREE_REGIMES)
def test_an_order_free_regime_takes_no_shuffle_step(regime: str) -> None:
    key = _key(regime)
    assert "shuffle" not in datasets.generation_kwargs(key)
    assert not key.label.endswith("-shuffled")
    assert datasets.generate_one(key)[3] is None  # no permutation to record
    with pytest.raises(ValueError, match="shuffle"):
        datasets.generation_kwargs(dataclasses.replace(key, shuffle=False))


@pytest.mark.parametrize("regime", ["pcmciplus_paper", "lpcmci_paper", "longitudinal_lingam_paper"])
def test_an_order_free_regime_orients_its_instantaneous_links_both_ways(regime: str) -> None:
    """Instantaneous links run from a later column to an earlier one as well as the other way."""
    forward = backward = 0
    for seed in range(3):
        _, truth, kind, _ = datasets.generate_one(_key(regime, seed=seed))
        if kind == datasets.WINDOW_KIND:  # [lag, cause, effect] marks, a tail at the cause
            causes, effects = np.nonzero(truth[0] == TAIL)
        else:  # [time, block, effect, cause] coefficients
            effects, causes = np.nonzero((truth[:, 0] != 0).any(axis=0))
        forward += int((causes < effects).sum())
        backward += int((causes > effects).sum())
    assert forward and backward


def test_a_measurement_model_lists_its_indicators_in_a_random_order() -> None:
    """No seed lists the indicators grouped by latent, as drawing them latent by latent would."""
    for seed in range(3):
        key = _key("latent_measurement", seed=seed)
        _, truth, _, _ = datasets.generate_one(key)
        observed = key.d - key.latents
        parent = (truth == TAIL) & (truth.T == ARROW)  # parent[i, j]: i -> j
        latent_of = parent[observed:, :observed].argmax(axis=0)  # each indicator's latent
        assert (np.diff(latent_of) < 0).any()


def test_the_datasets_a_published_summary_names_are_unshuffled_keys() -> None:
    """Every dataset id the summary lists resolves to the unshuffled key of its coordinates."""
    summary = json.loads(DATA_TYPES_SUMMARY.read_text())
    named = 0
    for method in summary["methods"]:
        topology, functional, noise, standardize = datasets.REGIMES[method["regime"]]
        for row in method["rows"]:
            for entry in row.get("different_graph", []):
                coordinates = {
                    "topology": topology,
                    "functional": functional,
                    "noise": noise,
                    "standardize": standardize,
                    "density": row["mean_degree"],
                    "d": row["d"],
                    "n": row["n"],
                    "seed": entry["seed"],
                    "shuffle": False,
                }
                if functional == datasets.MIXED_FUNCTIONAL:
                    low, high = entry["categories"].split("-")
                    coordinates["categories"] = (int(low), int(high))
                assert DatasetKey(**coordinates).digest == entry["dataset"]
                named += 1
    assert named


#: Seeds of each column-order check.
ORDER_SEEDS = range(100)


def _order_edges(truth: np.ndarray, kind: str) -> np.ndarray:
    """The ``(cause, effect)`` rows that fix a draw's causal order, in its node numbering.

    The DAG, or a CPDAG's directed edges; for a time series, its same-step DAG, or, with lagged
    links only, its links between different variables; for a coefficient stack, its instantaneous
    matrix (a panel's at the last time point).
    """
    if kind in ("dag", "cpdag"):
        return np.argwhere((truth == TAIL) & (truth.T == ARROW))
    if kind == "temporal":  # [lag, cause, effect] marks, a tail at the cause
        links = np.argwhere((truth == TAIL).any(axis=0))
        return links[links[:, 0] != links[:, 1]]
    if kind == datasets.WINDOW_KIND:
        return np.argwhere((truth[0] == TAIL) & (truth[0].T == ARROW))
    instantaneous = truth[0] if kind == "temporal_coefficients" else truth[-1][0]
    effect, cause = np.nonzero(instantaneous)  # [effect, cause] coefficients
    return np.stack([cause, effect], axis=1)


def _correlation(regime: str, seed: int, by_depth: bool) -> tuple[float, int]:
    """The draw's rank correlation of column index and causal depth, and its column count.

    ``by_depth`` sorts the columns by causal depth first: an order that follows the graph. ``nan``
    when the links form a cycle or every column has the same depth.
    """
    try:
        x, truth, kind, _ = datasets.generate_one(_key(regime, seed=seed))
    except ValueError as err:  # a seed with fewer nodes of two children than nodes to hide
        assert "cannot hide" in str(err)
        return float("nan"), 0
    truth = np.asarray(truth)
    d, k = truth.shape[-1], x.shape[-1]
    edges = _order_edges(truth, kind)
    try:
        depth = causal_depth(edges, d)
    except ValueError:  # lagged links in a cycle order no variable before another
        return float("nan"), k
    if by_depth:
        column = np.arange(d)
        column[:k] = np.argsort(np.argsort(depth[:k], kind="stable"), kind="stable")
        edges = column[edges]
    return index_correlation(edges, d, n_observed=k), k


@pytest.mark.parametrize("regime", sorted(datasets.REGIMES))
def test_a_regimes_column_order_says_nothing_of_the_causal_order(regime: str) -> None:
    """Across seeds the mean rank correlation of column index and causal depth is near zero.

    Under a uniform column order the correlation on ``k`` columns has mean 0 and variance
    ``1 / (k - 1)``, so the mean over ``S`` draws stays within four standard deviations,
    ``4 sqrt(sum 1 / (k - 1)) / S``. A draw whose links form a cycle has no causal order and is left
    out, which depends on its graph only. The positive control is the same draws with their columns
    sorted by causal depth.
    """
    drawn = [_correlation(regime, seed, by_depth=False) for seed in ORDER_SEEDS]
    rho, k = np.array([pair for pair in drawn if np.isfinite(pair[0])]).T
    bound = 4 * np.sqrt(np.sum(1 / (k - 1))) / len(k)
    assert abs(rho.mean()) < bound
    control = [_correlation(regime, seed, by_depth=True)[0] for seed in ORDER_SEEDS]
    assert abs(np.nanmean(control)) > bound


def test_the_varma_paper_model_needs_its_shuffle() -> None:
    """VARMA-LiNGAM's paper model draws a lower-triangular instantaneous matrix, so its unshuffled
    columns come in causal order: their mean correlation with causal depth is far outside the
    bound."""
    rhos = []
    for seed in range(30):
        key = _key("varma_lingam_paper", seed=seed, shuffle=False)
        _, stack, kind, _ = datasets.generate_one(key)
        rhos.append(index_correlation(_order_edges(np.asarray(stack), kind), key.d))
    assert np.nanmean(rhos) > 4 / np.sqrt(30 * (key.d - 1))
