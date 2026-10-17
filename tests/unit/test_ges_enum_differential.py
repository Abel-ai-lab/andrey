"""Test that GES enumeration, scan, and reachability paths return the same CPDAG and score.

The forward scan that keeps candidates between passes is compared with the full rescan, move by
move, from 8 to 200 nodes on dense and sparse data, on duplicate and near-duplicate columns, and
with the slowest reference: the full rescan over unpruned subset lattices and set reachability.
Separate checks cover the full rescan's forced and automatic reachability above 64 nodes, the
bounds the kept scan's screening deltas rely on, the kept scan under an engine whose deltas move
with the batch by up to those bounds, and serial against two-chunk decomposition.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from andrey.core import _bitset, backend
from andrey.core import score as score_module
from andrey.core.orient import to_structure
from andrey.core.score import BICScore
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import ges

# ``andrey.search.ges`` the *attribute* is the re-exported ``ges`` function (the package __init__
# shadows the submodule), so flags on the module must be reached through sys.modules, not the name.
_GES_MOD = sys.modules["andrey.search.ges"]


@pytest.fixture
def unchecked_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a BIC score accept the duplicate and near-duplicate columns its input check rejects.

    The engine tests below run on such columns to cover the scan and its bounds where scores are
    unavailable or rank-one updates cancel.
    """
    monkeypatch.setattr(score_module, "check_columns", lambda cov, n: None)


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


def _collinear_sem(seed: int, n: int, d: int, p: float, k: int, noise: float) -> np.ndarray:
    """:func:`_gaussian_sem` with ``k`` columns replaced by another column plus scaled noise.

    The noise has standard deviation ``noise``; 0 makes exact duplicates.
    """
    x = _gaussian_sem(seed, n, d, p)
    rng = np.random.default_rng(seed + 10_000)
    for _ in range(k):
        src, dst = rng.choice(d, 2, replace=False)
        x[:, dst] = x[:, src] + noise * rng.standard_normal(n)
    return x


def _random_collinear_case(seed: int) -> tuple[np.ndarray, int | None, float]:
    """Near-collinear data with size, density, ``maxP``, penalty, and noise drawn from ``seed``.

    Returns the data, ``maxP``, and ``lambda_value``.
    """
    rng = np.random.default_rng(seed)
    d = int(rng.integers(5, 26))
    n = int(rng.choice([25, 50, 100, 300]))
    p = float(rng.uniform(0.05, 0.9))
    max_p = rng.choice([None, 1, 2, 3, 6], p=[0.5, 0.1, 0.15, 0.15, 0.1])
    lambda_value = float(rng.choice([0.5, 1.0, 2.0]))
    noise = float(10.0 ** rng.uniform(-8, -2))
    x = _collinear_sem(seed, n, d, p, max(1, d // 5), noise)
    return x, None if max_p is None else int(max_p), lambda_value


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


def _assert_kept_matches_full(
    x: np.ndarray, maxP: float | None, lambda_value: float, label: str
) -> None:
    """The kept scan and the full rescan apply the same moves and return one CPDAG and score."""
    kept_moves: list[tuple] = []
    full_moves: list[tuple] = []
    kept_adj, kept_score = _run(x, maxP=maxP, lambda_value=lambda_value, moves=kept_moves)
    full_adj, full_score = _run(
        x, maxP=maxP, lambda_value=lambda_value, full_rescan=True, moves=full_moves
    )
    assert kept_moves == full_moves, f"move sequences differ at {label}"
    assert np.array_equal(kept_adj, full_adj)
    assert kept_score.hex() == full_score.hex()


@pytest.mark.parametrize("seed,d,n,p,lambda_value", _SCAN_GRID)
def test_kept_scan_matches_full_rescan(
    seed: int, d: int, n: int, p: float, lambda_value: float
) -> None:
    """The kept forward scan applies the full rescan's moves and returns its CPDAG and score."""
    x = _gaussian_sem(seed, n, d, p)
    _assert_kept_matches_full(x, None, lambda_value, f"seed={seed}, d={d}")


# Duplicate columns give deltas of minus infinity, which tie across sources; near-duplicates make
# the rank-one update cancel, so two batches over one base can give far-apart deltas.
_COLLINEAR_GRID = [
    # (seed, d, n, p, k, noise)
    (0, 4, 25, 0.0, 1, 0.0),
    (0, 8, 50, 0.50, 2, 0.0),
    (1, 15, 100, 0.40, 3, 0.0),
    (2, 15, 300, 0.40, 3, 1e-6),
    (3, 20, 200, 0.30, 4, 1e-7),
    (4, 20, 300, 0.60, 4, 1e-4),
]


@pytest.mark.usefixtures("unchecked_columns")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log of a zero residual
@pytest.mark.parametrize("seed,d,n,p,k,noise", _COLLINEAR_GRID)
def test_kept_scan_matches_full_rescan_on_collinear_columns(
    seed: int, d: int, n: int, p: float, k: int, noise: float
) -> None:
    """The kept scan matches the full rescan move by move on duplicate and near-duplicate data."""
    x = _collinear_sem(seed, n, d, p, k, noise)
    _assert_kept_matches_full(x, None, 1.0, f"seed={seed}, d={d}, noise={noise}")


# Seeds of random near-collinear cases that include ties at minus infinity and cancelled updates.
_RANDOM_COLLINEAR_SEEDS = [3, 59, 104, 128, 131, 162, 175, 250, 269]


@pytest.mark.usefixtures("unchecked_columns")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log of a zero residual
@pytest.mark.parametrize("seed", _RANDOM_COLLINEAR_SEEDS)
def test_kept_scan_matches_full_rescan_on_random_collinear_data(seed: int) -> None:
    """The kept scan matches the full rescan, move by move, on random near-collinear data."""
    x, max_p, lambda_value = _random_collinear_case(seed)
    _assert_kept_matches_full(x, max_p, lambda_value, f"seed={seed}")


@pytest.mark.usefixtures("unchecked_columns")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log of a zero residual
def test_screening_bounds_cover_batch_differences() -> None:
    """A delta scored in another batch over the same base stays within the first batch's bound.

    The kept scan's screening deltas come from one batch of every candidate sharing a base; the
    full rescan's come from a batch of the candidates that pass the reachability check. The scan
    selects the full rescan's move whenever every difference stays within the bound. The sample
    covers well-conditioned data, mixed column scales, and near-duplicate columns, where the
    rank-one update cancels. On well-conditioned data every bound is finite and small, so the scan
    still skips most candidates.
    """
    from andrey.core.score_delta import DeltaBICScore

    rng = np.random.default_rng(0)
    with threadpool_limits(limits=1):
        for noise in [None, 1e-2, 1e-4, 1e-6, 1e-8, 0.0]:
            for _ in range(8):
                d = int(rng.integers(12, 40))
                n = int(rng.choice([100, 2000]))
                seed = int(rng.integers(1000))
                if noise is None:
                    x = _gaussian_sem(seed, n, d, 0.2) * rng.uniform(1e-3, 1e3, size=d)
                else:
                    x = _collinear_sem(seed, n, d, 0.3, d // 4, noise)
                engine = DeltaBICScore(BICScore(x))
                for _ in range(20):
                    y = int(rng.integers(d))
                    rest = [v for v in range(d) if v != y]
                    base = rng.choice(rest, int(rng.integers(0, 8)), replace=False).tolist()
                    pool = [v for v in rest if v not in base]
                    sources = rng.choice(pool, min(len(pool), int(rng.integers(2, 20))), False)
                    moves = [("add", int(u)) for u in sources]
                    full_bounds: list[float] = []
                    full = engine.delta_many_with_base(y, base, moves, bounds=full_bounds)
                    if noise is None:
                        assert max(full_bounds) < 1e-9 * n
                    for _ in range(3):
                        keep = rng.random(len(sources)) < 0.5
                        part_bounds: list[float] = []
                        part = engine.delta_many_with_base(
                            y, base, [m for m, kept in zip(moves, keep) if kept], bounds=part_bounds
                        )
                        pairs = zip(
                            np.asarray(full)[keep],
                            np.asarray(full_bounds)[keep],
                            part,
                            part_bounds,
                            strict=True,
                        )
                        for a, a_bound, b, b_bound in pairs:
                            if a.hex() == b.hex():
                                continue
                            assert abs(a - b) <= a_bound, (noise, y, base)
                            assert abs(a - b) <= b_bound, (noise, y, base)


class _BatchShiftedDelta:
    """A delta engine whose add deltas change with the batch by up to the bounds it reports.

    An add's delta is two full scores plus a shift drawn from the target, base, source, and whole
    batch. The shift stays within half the reported bound, so two batches differ by at most the
    bound. A third of the sources report an infinite bound and shift anywhere in ``spread``.
    Removes are two full scores.
    """

    def __init__(self, score: BICScore, bound: float, spread: float) -> None:
        self.score_obj = score
        self.bound = bound
        self.spread = spread

    def delta_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
        *,
        bounds: list[float] | None = None,
    ) -> list[float]:
        from andrey.core.score_delta import _delta_two_scores

        base = tuple(sorted(int(p) for p in base_parents))
        batch = sorted(int(u) for _, u in deltas)
        out: list[float] = []
        out_bounds: list[float] = []
        for op, u in deltas:
            delta = _delta_two_scores(self.score_obj, int(i), base, int(u), op)
            bound = 0.0
            if op == "add":
                bound = math.inf if (int(i) + int(u)) % 3 == 0 else self.bound
                rng = np.random.default_rng([int(i), int(u), len(base), *base, *batch])
                delta += rng.uniform(-0.5, 0.5) * (self.spread if bound == math.inf else bound)
            out.append(delta)
            out_bounds.append(bound)
        if bounds is not None:
            bounds[:] = out_bounds
        return out


@pytest.mark.parametrize("seed,d,n,p", [(0, 12, 200, 0.40), (1, 20, 300, 0.30), (2, 30, 300, 0.15)])
def test_kept_scan_honors_any_engine_bounds(
    seed: int, d: int, n: int, p: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kept scan selects the full rescan's Inserts whenever deltas stay within their bounds.

    The engine moves each delta with its batch by up to its reported bound, and by any amount where
    the bound is infinite, so the scan's selection rests on the bounds alone.
    """
    score = BICScore(_gaussian_sem(seed, n, d, p))

    def forward(full_rescan: bool) -> tuple[np.ndarray, list[tuple]]:
        engine = _BatchShiftedDelta(score, bound=0.05 * n, spread=2.0 * n)
        moves: list[tuple] = []
        insert = ops.apply_insert

        def record_insert(adj, x, y, T):
            moves.append((x, y, tuple(T)))
            insert(adj, x, y, T)

        with monkeypatch.context() as patch:
            patch.setattr(ops, "apply_insert", record_insert)
            patch.setattr(pg, "_FULL_RESCAN", full_rescan)
            adj = pg._forward_inprocess(np.zeros((d, d), dtype=np.int8), engine, d, 3, 1)
        return adj, moves

    kept_adj, kept_moves = forward(False)
    full_adj, full_moves = forward(True)
    assert len(full_moves) > d  # the shifts reorder many candidates over many passes
    assert kept_moves == full_moves
    assert np.array_equal(kept_adj, full_adj)


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
