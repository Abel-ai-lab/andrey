"""Test that GES enumeration, scan, and reachability paths return the same CPDAG and score.

The forward scan that keeps candidates between passes is compared with the full rescan, move by
move, from 8 to 200 nodes on dense and sparse data, and with the slowest reference: the full rescan
over unpruned subset lattices and set reachability. Separate checks cover the full rescan's forced
and automatic reachability above 64 nodes, the bound the kept scan's screening deltas rely on, and
serial against two-chunk decomposition.
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


def _run(
    x: np.ndarray,
    *,
    reach: str = "auto",
    prune: bool = True,
    maxP: float | None,
    lambda_value: float = 1.0,
    full_rescan: bool = False,
    moves: list[tuple] | None = None,
):
    """Run GES with explicit scan, reachability mode, and pruning toggles (restored afterwards).

    ``reach`` takes effect only in the full rescan above 64 variables; below the cap it always uses
    the bitmask. ``moves``, when given, collects each applied Insert and Delete in order.
    """
    old_reach = getattr(_GES_MOD, "_REACH_MODE", "auto")
    old_prune = getattr(ops, "_LATTICE_PRUNE", True)
    insert, delete = ops.apply_insert, ops.apply_delete
    _GES_MOD._REACH_MODE = reach
    ops._LATTICE_PRUNE = prune
    pg._FULL_RESCAN = full_rescan
    if moves is not None:

        def record_insert(adj, x, y, T):
            moves.append(("insert", x, y, tuple(T)))
            insert(adj, x, y, T)

        def record_delete(adj, x, y, H):
            moves.append(("delete", x, y, tuple(H)))
            delete(adj, x, y, H)

        ops.apply_insert, ops.apply_delete = record_insert, record_delete
    try:
        # Pin serial: the toggles are module globals that never reach pool workers, so under
        # ANDREY_NUM_WORKERS > 1 a pooled pass would silently run the live path inside the
        # reference run and the comparison would be fast-vs-fast.
        with backend.config(num_workers=1):
            struct, score = ges(x, lambda_value=lambda_value, maxP=maxP)
    finally:
        _GES_MOD._REACH_MODE = old_reach
        ops._LATTICE_PRUNE = old_prune
        pg._FULL_RESCAN = False
        ops.apply_insert, ops.apply_delete = insert, delete
    return struct.to_numpy(), score


# Dense rows put several undirected neighbors on most targets, so subsets and reachability matter;
# sparse rows reach 200 nodes. The full rescan costs about one scan of every target per accepted
# move, so the largest rows take a heavier penalty, which keeps chance edges and moves few.
_SCAN_GRID = [
    # (seed, d, n, p, lambda_value)
    (0, 8, 80, 0.60, 1.0),
    (1, 8, 200, 0.80, 1.0),
    (0, 15, 300, 0.40, 1.0),
    (2, 30, 300, 0.25, 1.0),
    (0, 60, 300, 0.08, 1.0),
    (1, 100, 500, 0.02, 2.0),
    (0, 200, 1000, 0.002, 2.0),
]


@pytest.mark.parametrize("seed,d,n,p,lambda_value", _SCAN_GRID)
def test_kept_scan_matches_full_rescan(
    seed: int, d: int, n: int, p: float, lambda_value: float
) -> None:
    """The kept forward scan applies the full rescan's moves and returns its CPDAG and score."""
    x = _gaussian_sem(seed, n, d, p)
    kept_moves: list[tuple] = []
    full_moves: list[tuple] = []
    kept_adj, kept_score = _run(x, maxP=None, lambda_value=lambda_value, moves=kept_moves)
    full_adj, full_score = _run(
        x, maxP=None, lambda_value=lambda_value, full_rescan=True, moves=full_moves
    )
    assert kept_moves == full_moves, f"move sequences differ at seed={seed}, d={d}"
    assert np.array_equal(kept_adj, full_adj)
    assert kept_score.hex() == full_score.hex()


def test_screening_bound_covers_batch_differences() -> None:
    """Scoring a source in a different batch over the same base stays far inside the bound.

    The kept scan's screening deltas come from one batch of every candidate sharing a base; the
    full rescan's come from a batch of the candidates that pass the reachability check. The scan
    selects the full rescan's move whenever the two differ by less than ``_SCREEN_TOL`` per sample.
    """
    from andrey.core.score_delta import DeltaBICScore

    rng = np.random.default_rng(0)
    worst = 0.0
    with threadpool_limits(limits=1):
        for _ in range(20):
            d = int(rng.integers(20, 80))
            n = int(rng.choice([200, 2000]))
            x = _gaussian_sem(int(rng.integers(1000)), n, d, 0.2)
            engine = DeltaBICScore(BICScore(x))
            for _ in range(20):
                y = int(rng.integers(d))
                rest = [v for v in range(d) if v != y]
                base = sorted(rng.choice(rest, int(rng.integers(0, 10)), replace=False).tolist())
                pool = [v for v in rest if v not in base]
                sources = rng.choice(pool, min(len(pool), int(rng.integers(2, 30))), replace=False)
                full = engine.delta_many_with_base(y, base, [("add", int(u)) for u in sources])
                for _ in range(3):
                    keep = rng.random(len(sources)) < 0.5
                    part = engine.delta_many_with_base(
                        y, base, [("add", int(u)) for u in sources[keep]]
                    )
                    for a, b in zip(np.asarray(full)[keep], part, strict=True):
                        worst = max(worst, abs(a - b) / n)
    assert worst < 1e-3 * pg._SCREEN_TOL


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
    ref_adj, ref_score = _run(x, reach="set", prune=False, maxP=maxP, full_rescan=True)
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
    wide_adj, wide_score = _run(x, reach="wide", prune=True, maxP=maxP, full_rescan=True)
    set_adj, set_score = _run(x, reach="set", prune=True, maxP=maxP, full_rescan=True)
    assert np.array_equal(wide_adj, set_adj), f"wide vs set differ at seed={seed}, d={d}"
    assert wide_score == set_score


def test_auto_reachability_uses_wide_and_matches_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full rescan's automatic reachability uses both traversals and keeps CPDAG and BIC."""
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
        auto_adj, auto_score = _run(x, reach="auto", prune=True, maxP=5, full_rescan=True)
        assert wide_calls > 0, "auto never used wide reachability"
        assert set_states > 0, "auto never used set reachability"
        set_adj, set_score = _run(x, reach="set", prune=True, maxP=5, full_rescan=True)
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
