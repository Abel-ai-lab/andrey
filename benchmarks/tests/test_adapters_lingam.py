"""The causal-learn DirectLiNGAM and ICA-LiNGAM adapters against their Andrey counterparts.

Each causal-learn solution must return the same graph as its Andrey counterpart. DirectLiNGAM's
order search is deterministic. Both ICA-LiNGAM solutions start FastICA from the same seed. The
fit-to-valid-DAG check for every LiNGAM adapter is in :mod:`test_adapters_contract`.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey_bench.adapters.andrey_direct_lingam import andrey_direct_lingam_adapters
from andrey_bench.adapters.andrey_ica_lingam import AndreyICALiNGAM
from andrey_bench.adapters.causal_learn_direct_lingam import CausalLearnDirectLiNGAM
from andrey_bench.adapters.causal_learn_ica_lingam import CausalLearnICALiNGAM
from andrey_bench.datasets import CANONICAL_DENSITY, DatasetKey, generation_kwargs
from andrey_bench.scoring import structure_hash

_ANDREY_DIRECT = next(a for a in andrey_direct_lingam_adapters() if a.backend == "numpy")
PAIRS = [(_ANDREY_DIRECT, CausalLearnDirectLiNGAM()), (AndreyICALiNGAM(), CausalLearnICALiNGAM())]


def _lingam_sf(d: int, n: int) -> np.ndarray:
    """Return a `lingam_sf` sample: scale-free and linear with uniform noise."""
    import andrey.data as data

    key = DatasetKey(
        topology="scale_free",
        functional="linear",
        noise="uniform",
        standardize="standardized",
        density=CANONICAL_DENSITY,
        d=d,
        n=n,
        seed=0,
    )
    return np.asarray(data.sample_scm(**generation_kwargs(key)).data)


@pytest.mark.parametrize("andrey_adapter, cl_adapter", PAIRS, ids=lambda a: a.name)
def test_causal_learn_returns_the_andrey_graph(andrey_adapter, cl_adapter) -> None:
    data = _lingam_sf(20, 200)
    ours = andrey_adapter.to_structure(andrey_adapter.fit(data, andrey_adapter.params()))
    theirs = cl_adapter.to_structure(cl_adapter.fit(data, cl_adapter.params()))
    assert structure_hash(ours) == structure_hash(theirs)


def test_the_causal_learn_conversion_reads_the_true_direction() -> None:
    """Check x -> y with uniform noise; untransposed weights would give y -> x."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 2000)
    y = 0.8 * x + rng.uniform(-1, 1, 2000)
    adj = CausalLearnDirectLiNGAM().fit(np.column_stack([x, y]), {})
    assert np.array_equal(adj, [[0, 1], [0, 0]])


def test_both_ica_solutions_run_one_seed() -> None:
    assert AndreyICALiNGAM().params() == CausalLearnICALiNGAM().params()
