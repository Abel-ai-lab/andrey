"""Test that GES enumeration and reachability paths return the same CPDAG and score.

Compare pruned with unpruned subset-lattice enumeration; above 64 nodes the reference also uses set
reachability (below the cap GES always takes the single-word bitmask). Separate checks cover forced
and automatic reachability above 64 nodes and compare serial and two-chunk decomposition.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from andrey.core import _bitset, backend
from andrey.core.orient import to_structure
from andrey.core.score import BICScore
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import ges

# ``andrey.search.ges`` the *attribute* is the re-exported ``ges`` function (the package __init__
# shadows the submodule), so flags on the module must be reached through sys.modules, not the name.
_GES_MOD = sys.modules["andrey.search.ges"]


def _gaussian_sem(seed: int, n: int, d: int, p: float) -> np.ndarray:
    """A random strictly-lower-triangular linear-Gaussian SEM (edge density ``p``)."""
    rng = np.random.default_rng(seed)
    b = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                b[j, i] = rng.uniform(0.4, 1.2) * rng.choice([-1.0, 1.0])
    e = rng.standard_normal((n, d))
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = e[:, j] + x @ b[j]
    return x


def _run(x: np.ndarray, *, reach: str, prune: bool, maxP: float | None):
    """Run GES with explicit reachability mode + pruning toggle (restored afterwards).

    ``reach`` takes effect only above 64 variables; below the cap GES always uses the bitmask.
    """
    old_reach = getattr(_GES_MOD, "_REACH_MODE", "auto")
    old_prune = getattr(ops, "_LATTICE_PRUNE", True)
    _GES_MOD._REACH_MODE = reach
    ops._LATTICE_PRUNE = prune
    try:
        # Pin serial: the toggles are module globals that never reach pool workers, so under
        # ANDREY_NUM_WORKERS > 1 a pooled pass would silently run the live path inside the
        # reference run and the comparison would be fast-vs-fast.
        with backend.config(num_workers=1):
            struct, score = ges(x, maxP=maxP)
    finally:
        _GES_MOD._REACH_MODE = old_reach
        ops._LATTICE_PRUNE = old_prune
    return struct.to_numpy(), score


# Every row reaches the pruned subset lattice (a t0 of at least ``ops._LATTICE_MIN_M`` nodes), so
# the pruning toggle changes the path taken. Only the d = 66 row also swaps in set reachability.
# maxP bounds density so the reference per-candidate set BFS stays cheap.
_GRID = [
    # (seed, d, n, p, maxP)
    (0, 15, 300, 0.40, None),
    (0, 40, 240, 0.18, 6),
    (1, 40, 240, 0.18, 6),
    (0, 66, 180, 0.05, 3),
]


@pytest.mark.parametrize("seed,d,n,p,maxP", _GRID)
def test_fast_enum_matches_reference(
    seed: int, d: int, n: int, p: float, maxP: float | None
) -> None:
    """Pruned and unpruned enumeration return the same CPDAG and score below and above 64 nodes."""
    x = _gaussian_sem(seed, n, d, p)
    live_adj, live_score = _run(x, reach="auto", prune=True, maxP=maxP)
    ref_adj, ref_score = _run(x, reach="set", prune=False, maxP=maxP)
    assert np.array_equal(live_adj, ref_adj), (
        f"CPDAG differs from the reference at seed={seed}, d={d}"
    )
    assert live_score == ref_score, f"score differs from the reference at seed={seed}, d={d}"


@pytest.mark.parametrize(
    "seed,d,n,p,maxP",
    [(0, 66, 180, 0.06, 3)],
)
def test_wide_and_set_reachability_agree_in_ges(
    seed: int, d: int, n: int, p: float, maxP: float | None
) -> None:
    """Wide bitmask and set reachability return the same CPDAG and score above 64 nodes."""
    x = _gaussian_sem(seed, n, d, p)
    wide_adj, wide_score = _run(x, reach="wide", prune=True, maxP=maxP)
    set_adj, set_score = _run(x, reach="set", prune=True, maxP=maxP)
    assert np.array_equal(wide_adj, set_adj), f"wide vs set differ at seed={seed}, d={d}"
    assert wide_score == set_score


def test_auto_reachability_uses_wide_and_matches_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatic reachability uses both traversals and preserves the CPDAG and BIC exactly."""
    # 65 variables cross the single-word limit; few samples make the learned graph dense.
    x = np.random.default_rng(0).standard_normal((8, 65))
    wide_calls = 0
    set_states = 0
    reaches_wide = _bitset.semidirected_reaches_wide
    successor_lists = ops.tail_successor_lists

    def counted_wide(succ: list[int], start: int, target: int, barrier: int) -> bool:
        nonlocal wide_calls
        wide_calls += 1
        return reaches_wide(succ, start, target, barrier)

    def counted_set(adj: np.ndarray) -> list[list[int]]:
        nonlocal set_states
        set_states += 1
        return successor_lists(adj)

    monkeypatch.setattr(_bitset, "semidirected_reaches_wide", counted_wide)
    monkeypatch.setattr(ops, "tail_successor_lists", counted_set)
    # Pin covariance to NumPy so backend environment settings do not change the scores.
    # The traversals above 64 variables do not consult the backend.
    with backend.config(backend="numpy"):
        auto_adj, auto_score = _run(x, reach="auto", prune=True, maxP=5)
        assert wide_calls > 0, "auto never used wide reachability"
        assert set_states > 0, "auto never used set reachability"
        set_adj, set_score = _run(x, reach="set", prune=True, maxP=5)
    assert np.array_equal(auto_adj, set_adj)
    # Matching non-finite scores would pass the hex comparison without a valid BIC.
    assert np.isfinite(auto_score)
    assert auto_score.hex() == set_score.hex()  # Hex equality also distinguishes signed zero.


def _ges_decomposed(x: np.ndarray, n_chunks: int, maxP: float | None):
    """Full GES via the in-process N-chunk decomposition, mirroring ``ges()``'s serial setup.

    Drives the fast enumerator through ``forward_chunk_best`` / ``backward_chunk_best`` and the
    ``(stable_key, x, y, index)`` reduce -- the path the subset pruning must leave bit-identical --
    then recomputes the objective from the final CPDAG as ``ges()`` does.
    """
    from andrey.core.score_delta import DeltaBICScore
    from andrey.search.ges import _recompute_objective

    X = np.asarray(x, dtype=np.float64)
    n = X.shape[1]
    max_parents = n / 2 if maxP is None else maxP
    score = BICScore(X, lambda_value=1.0)
    engine = DeltaBICScore(score)
    adj = np.zeros((n, n), dtype=np.int8)
    with threadpool_limits(limits=1):
        adj = pg._forward_inprocess(adj, engine, n, max_parents, n_chunks)
        adj = pg._backward_inprocess(adj, engine, n, n_chunks)
        total = _recompute_objective(adj, score, n)
    return to_structure(adj, kind="cpdag").to_numpy(), total


# Chunk counts and the d > 64 traversals have their own tests.
@pytest.mark.parametrize("seed,d,n,p,maxP", [(0, 20, 200, 0.30, None), (1, 40, 240, 0.18, 6)])
def test_serial_equals_parallel_decomposition(
    seed: int, d: int, n: int, p: float, maxP: float | None
) -> None:
    """Serial GES and two-chunk decomposition return the same CPDAG and score."""
    x = _gaussian_sem(seed, n, d, p)
    serial_adj, serial_score = _run(x, reach="auto", prune=True, maxP=maxP)
    decomp_adj, decomp_score = _ges_decomposed(x, 2, maxP)
    assert np.array_equal(serial_adj, decomp_adj), (
        f"decomposition differs from serial at seed={seed}, d={d}"
    )
    assert serial_score == decomp_score
