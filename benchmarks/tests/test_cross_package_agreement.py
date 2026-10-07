"""Do two packages given one dataset return one graph?

`structure_hash` exists to make that answerable: each package returns its own encoding, so comparing
what `fit` returned compares encodings rather than graphs.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.adapters.gcastle_ges import GCastleGES
from andrey_bench.scoring import structure_hash


def _linear_gaussian(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.triu(rng.normal(0, 1, (d, d)), 1) * (rng.random((d, d)) < 0.2)
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = x @ weights[:, j] + rng.normal(0, 1, n)
    return (x - x.mean(0)) / x.std(0)


def _fit_hash(adapter, x: np.ndarray) -> str:
    return structure_hash(adapter.to_structure(np.asarray(adapter.fit(x, adapter.params()))))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("d", [10, 20])
def test_andrey_and_causal_learn_ges_return_the_same_graph(d: int, seed: int):
    """At the matched penalty the two engines return the same graph."""
    x = _linear_gaussian(d, 1000, seed)
    assert _fit_hash(andrey_ges_adapters()[0], x) == _fit_hash(CausalLearnGES(), x)


def test_the_hash_still_separates_a_package_that_disagrees():
    """A hash that collapsed everything to one value would pass the test above for the wrong reason."""
    x = _linear_gaussian(10, 1000, 0)
    assert _fit_hash(GCastleGES(), x) != _fit_hash(andrey_ges_adapters()[0], x)
