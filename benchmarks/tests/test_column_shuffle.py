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
