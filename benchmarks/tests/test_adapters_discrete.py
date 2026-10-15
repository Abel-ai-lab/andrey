"""The categorical-data adapters, their regime, and agreement with causal-learn.

Andrey's chi-square and G-square p-values and BDeu scores follow causal-learn's definitions, so
they must agree with causal-learn's to rounding; on the same data both packages' PC and GES must
return the same graph. Each adapter must also meet the harness contract the shared contract test
checks for the others.
"""

from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations

import discrete_cpu
import numpy as np
import pytest

from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore
from andrey.data import sample_discrete
from andrey_bench import datasets
from andrey_bench.contracts import SolutionAdapter
from andrey_bench.datasets import DatasetKey
from andrey_bench.scoring import structure_hash

ADAPTERS = [a for pair in discrete_cpu.PAIRS.values() for a in pair]

#: The module each Andrey adapter's ``fit`` imports on first use.
DEFERRED_MODULES = {
    "andrey.pc.chisq": "andrey.constraint.pc",
    "andrey.pc.gsq": "andrey.constraint.pc",
    "andrey.ges.bdeu": "andrey.search.ges",
}


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_an_adapter_meets_the_harness_contract(adapter: SolutionAdapter) -> None:
    assert isinstance(adapter, SolutionAdapter)
    json.dumps(adapter.params(), sort_keys=True)
    clone = pickle.loads(pickle.dumps(adapter))
    assert (clone.name, clone.params()) == (adapter.name, adapter.params())


def test_every_solution_has_its_own_name() -> None:
    names = [a.name for a in discrete_cpu.SOLUTIONS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("method", list(discrete_cpu.PAIRS))
def test_both_packages_run_one_method_with_one_setting(method: str) -> None:
    mine, theirs = discrete_cpu.PAIRS[method]
    assert (mine.algorithm, mine.output_type, mine.mode) == (
        theirs.algorithm,
        theirs.output_type,
        theirs.mode,
    )
    assert mine.params() == theirs.params()


@pytest.mark.parametrize("name", list(DEFERRED_MODULES))
def test_setup_imports_the_module_fit_defers(name: str) -> None:
    (adapter,) = [a for a in ADAPTERS if a.name == name]
    sys.modules.pop(DEFERRED_MODULES[name], None)
    adapter.setup()
    assert DEFERRED_MODULES[name] in sys.modules


def _key(d: int = 6, n: int = 400, seed: int = 0) -> DatasetKey:
    return DatasetKey(**discrete_cpu.regime_slice(), d=d, n=n, seed=seed)


def test_the_regime_stores_category_codes_and_the_true_dag() -> None:
    key = _key()
    call = datasets.generation_kwargs(key)
    assert call["generator"] == "sample_discrete"
    assert {k: call[k] for k in datasets.DISCRETE_SETTINGS} == datasets.DISCRETE_SETTINGS
    data, marks, kind = datasets.generate_one(key)
    assert kind == "dag" and marks.shape == (6, 6)
    assert data.shape == (400, 6) and np.array_equal(data, np.round(data))


def test_the_regime_rejects_rescaled_codes() -> None:
    key = DatasetKey(
        **{**discrete_cpu.regime_slice(), "standardize": "standardized"}, d=5, n=50, seed=0
    )
    with pytest.raises(ValueError, match="raw codes"):
        datasets.generation_kwargs(key)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_a_fit_gives_a_cpdag(adapter: SolutionAdapter) -> None:
    data, _, _ = datasets.generate_one(_key())
    adapter.setup()
    structure = adapter.to_structure(np.asarray(adapter.fit(data, adapter.params())))
    assert structure.kind == "cpdag" and structure.n_nodes == 6


# --- agreement with causal-learn ------------------------------------------------------------------


def _codes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 300
    x = np.column_stack([rng.integers(0, k, n) for k in (2, 3, 4, 3, 2)]).astype(float)
    x[:, 1] = np.where(rng.random(n) < 0.6, x[:, 0], x[:, 1])
    x[:, 2] = 3 * x[:, 2] - 7  # codes need not start at 0
    x[:, 4] = x[:, 3] % 2  # constant within each value of column 3: no degrees of freedom
    return x


@pytest.mark.parametrize("method", ["chisq", "gsq"])
def test_p_values_agree_with_causal_learn(method: str) -> None:
    from causallearn.utils.cit import CIT

    for seed in range(3):
        x = _codes(seed)
        mine, theirs = ChiSquare(x, g_square=method == "gsq"), CIT(x, method)
        for a, b in combinations(range(x.shape[1]), 2):
            rest = [c for c in range(x.shape[1]) if c not in (a, b)]
            for size in range(3):
                for s in combinations(rest, size):
                    assert mine(a, b, s) == pytest.approx(
                        theirs(a, b, list(s)), rel=1e-9, abs=1e-12
                    )


def test_bdeu_is_minus_twice_causal_learns_score() -> None:
    from causallearn.score.LocalScoreFunction import local_score_BDeu

    x = _codes(0)
    mine = BDeuScore(x)
    for node in range(x.shape[1]):
        rest = [c for c in range(x.shape[1]) if c != node]
        for size in range(3):
            for parents in combinations(rest, size):
                expected = -2 * local_score_BDeu(x, node, list(parents))
                assert mine.score(node, list(parents)) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("method", list(discrete_cpu.PAIRS))
def test_both_packages_return_the_same_graph(method: str, seed: int) -> None:
    data = sample_discrete(d=8, n=2000, seed=seed).data
    hashes = []
    for adapter in discrete_cpu.PAIRS[method]:
        adapter.setup()
        native = adapter.fit(data, adapter.params())
        hashes.append(structure_hash(adapter.to_structure(np.asarray(native))))
    assert hashes[0] == hashes[1]
