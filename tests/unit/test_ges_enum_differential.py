"""Test that GES's forward scans and subset enumerations return one CPDAG and score.

The kept forward scan is compared with the full rescan (``_FULL_RESCAN``), pruned enumeration
with unpruned subset lattices (``_LATTICE_PRUNE``), the CPDAG rewritten around a move with the
CPDAG rebuilt whole (``_LOCAL_REFORM``), and serial GES with two-chunk decomposition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from andrey.core import ARROW, TAIL, backend
from andrey.core import score as score_module
from andrey.core.knowledge import Constraints, Knowledge, resolve
from andrey.core.orient import _cpdag_of_pdag, _cpdag_of_pdag_near, dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import ges
from andrey.search.hc import tie_ranks


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
    prune: bool = True,
    maxP: float | None,
    lambda_value: float = 1.0,
    full_rescan: bool = False,
    local_reform: bool = True,
    constraints: Constraints | None = None,
    moves: list[tuple] | None = None,
):
    """Run GES with explicit scan, pruning, and CPDAG-rewrite toggles (restored afterwards).

    ``moves``, when given, collects each applied Insert and Delete in order.
    """
    old_prune = getattr(ops, "_LATTICE_PRUNE", True)
    insert, delete = ops.apply_insert, ops.apply_delete
    ops._LATTICE_PRUNE = prune
    ops._LOCAL_REFORM = local_reform
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
            struct, score = ges(x, lambda_value=lambda_value, maxP=maxP, constraints=constraints)
    finally:
        ops._LATTICE_PRUNE = old_prune
        ops._LOCAL_REFORM = True
        pg._FULL_RESCAN = False
        ops.apply_insert, ops.apply_delete = insert, delete
    return struct.to_numpy(), score


# Dense rows put several undirected neighbors on most targets, so subsets and reachability matter;
# sparse rows reach 100 nodes. The full rescan costs about one scan of every target per accepted
# move, so the largest rows take a heavier penalty, which keeps chance edges and moves few.
_SCAN_GRID = [
    # (seed, d, n, p, lambda_value)
    (0, 8, 80, 0.60, 1.0),
    (1, 8, 200, 0.80, 1.0),
    (0, 15, 300, 0.40, 1.0),
    (2, 30, 300, 0.25, 1.0),
    (0, 60, 300, 0.08, 1.0),
    (1, 100, 500, 0.02, 2.0),
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


def _valid_moves(adj: np.ndarray) -> list[tuple[str, int, int, tuple[int, ...]]]:
    """Every valid Insert and Delete of the CPDAG ``adj``, as GES enumerates them."""
    n = adj.shape[0]
    state = pg.neighborhood_state(adj)
    reachable = pg._reachable(adj)
    moves = []
    for y in range(n):
        for x, _t_index, T, clique in pg._insert_candidates(y, adj, state):
            if not reachable(y, x, clique):
                moves.append(("insert", x, y, T))
    adj_sets, _parents, undirected = state
    for y in range(n):
        for x in sorted(adj_sets[y]):
            if adj[y, x] == ARROW or (adj[y, x] == TAIL and adj[x, y] == TAIL):
                for _h_index, H in ops.clique_residuals(adj, sorted(undirected[y] & adj_sets[x])):
                    moves.append(("delete", x, y, H))
    return moves


@pytest.mark.parametrize(
    "seed,d,steps,p",
    [
        (0, 6, 40, 0.40),
        (1, 10, 60, 0.20),
        (2, 16, 80, 0.12),
        (3, 30, 80, 0.07),
        (4, 8, 60, 0.50),
        (5, 12, 80, 0.35),
    ],
)
def test_cpdag_after_a_move_matches_a_full_rebuild(
    seed: int, d: int, steps: int, p: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A walk of valid Inserts and Deletes gives, after each, the CPDAG a full rebuild gives.

    The region limit is lifted, so every move takes the rewrite of the nodes it can reach. The
    walk starts from the CPDAG of a random DAG of edge density ``p``, so it begins with directed
    and undirected edges; the denser rows put triangles and long undirected chains in the way.
    """
    monkeypatch.setattr(ops, "_REGION_SHARE", 1.0)
    rng = np.random.default_rng(seed)
    dag = np.zeros((d, d), dtype=np.int8)
    order = rng.permutation(d)
    for lo in range(d):
        for hi in range(lo + 1, d):
            if rng.random() < p:
                dag[order[lo], order[hi]], dag[order[hi], order[lo]] = TAIL, ARROW
    adj = dag2cpdag(dag)
    kinds = set()
    for _ in range(steps):
        moves = _valid_moves(adj)
        kind, x, y, subset = moves[int(rng.integers(len(moves)))]
        kinds.add(kind)
        moved = adj.copy()
        apply = ops.apply_insert if kind == "insert" else ops.apply_delete
        apply(moved, x, y, subset)
        expected = _cpdag_of_pdag(moved)
        change = ops.insert_cpdag if kind == "insert" else ops.delete_cpdag
        adj = change(adj, x, y, subset)
        assert np.array_equal(adj, expected), (kind, x, y, subset)
    assert kinds == {"insert", "delete"}


def test_cpdag_rewrite_declines_a_region_past_its_limit() -> None:
    """A move that can reach more nodes than the limit leaves the matrix as it was."""
    adj = np.zeros((6, 6), dtype=np.int8)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        adj[a, b] = adj[b, a] = TAIL
    before = adj.copy()
    assert _cpdag_of_pdag_near(adj, [0], 2) is None
    assert np.array_equal(adj, before)
    assert _cpdag_of_pdag_near(adj, [0], 5) is adj


@pytest.mark.parametrize("seed,d,n,p,lambda_value", [_SCAN_GRID[2], _SCAN_GRID[4]])
def test_local_reform_matches_full_rebuild(
    seed: int, d: int, n: int, p: float, lambda_value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GES applies the same moves and returns the same CPDAG and score when each move rewrites
    only the nodes it can reach as when each move rebuilds the whole CPDAG."""
    monkeypatch.setattr(ops, "_REGION_SHARE", 1.0)
    x = _gaussian_sem(seed, n, d, p)
    local_moves: list[tuple] = []
    full_moves: list[tuple] = []
    local_adj, local_score = _run(
        x, maxP=None, lambda_value=lambda_value, local_reform=True, moves=local_moves
    )
    full_adj, full_score = _run(
        x, maxP=None, lambda_value=lambda_value, local_reform=False, moves=full_moves
    )
    assert local_moves == full_moves
    assert np.array_equal(local_adj, full_adj)
    assert local_score.hex() == full_score.hex()


# Duplicate columns make families unavailable, so their adds score an infinite delta;
# near-duplicates make the rank-one update cancel, so two batches over one base can give far-apart
# deltas.
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


# Random near-collinear cases whose updates can cancel: one without maxP, one whose maxP changes
# the graph.
_RANDOM_COLLINEAR_SEEDS = [59, 131]


@pytest.mark.usefixtures("unchecked_columns")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log of a zero residual
@pytest.mark.parametrize("seed", _RANDOM_COLLINEAR_SEEDS)
def test_kept_scan_matches_full_rescan_on_random_collinear_data(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kept scan matches the full rescan, move by move, on random near-collinear data."""
    from andrey.core.score_delta import DeltaBICScore

    x, max_p, lambda_value = _random_collinear_case(seed)
    batch = DeltaBICScore.delta_many_with_base
    infinite_bounds = 0

    def counted(self, i, base_parents, deltas, *, bounds=None):
        nonlocal infinite_bounds
        out = batch(self, i, base_parents, deltas, bounds=bounds)
        infinite_bounds += sum(b == math.inf for b in bounds or ())
        return out

    monkeypatch.setattr(DeltaBICScore, "delta_many_with_base", counted)
    _assert_kept_matches_full(x, max_p, lambda_value, f"seed={seed}")
    assert infinite_bounds, "no update could cancel"
    if max_p is not None:
        capped, _ = _run(x, maxP=max_p, lambda_value=lambda_value)
        uncapped, _ = _run(x, maxP=None, lambda_value=lambda_value)
        assert not np.array_equal(capped, uncapped), "maxP does not change the graph"


@pytest.mark.usefixtures("unchecked_columns")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log of a zero residual
def test_screening_bounds_cover_batch_differences() -> None:
    """A delta scored in another batch over the same base stays within the first batch's bound.

    The kept scan screens with one batch of every candidate sharing a base; the full rescan scores
    a batch of those passing reachability. On well-conditioned data every bound is also small.
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


@pytest.mark.parametrize("seed,d,n,p", [(0, 12, 200, 0.40), (1, 20, 300, 0.30)])
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
            adj = pg._forward_inprocess(np.zeros((d, d), dtype=np.int8), engine, d, 3, 1, range(d))
        return adj, moves

    kept_adj, kept_moves = forward(False)
    full_adj, full_moves = forward(True)
    assert len(full_moves) > d  # the shifts reorder many candidates over many passes
    assert kept_moves == full_moves
    assert np.array_equal(kept_adj, full_adj)


# Every row reaches the pruned subset lattice (a t0 of at least ``ops._LATTICE_MIN_M`` nodes), so
# the pruning toggle changes the path taken; the d = 66 row crosses a 64-bit word. maxP bounds
# density so the reference per-candidate set BFS stays cheap.
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
    live_adj, live_score = _run(x, prune=True, maxP=maxP)
    ref_adj, ref_score = _run(x, prune=False, maxP=maxP, full_rescan=True)
    assert np.array_equal(live_adj, ref_adj), (
        f"CPDAG differs from the reference at seed={seed}, d={d}"
    )
    assert live_score == ref_score, f"score differs from the reference at seed={seed}, d={d}"


def _ges_decomposed(x: np.ndarray, n_chunks: int, maxP: float | None):
    """Full GES via the in-process N-chunk decomposition, mirroring ``ges()``'s serial setup.

    Drives the fast enumerator through ``forward_chunk_best`` / ``backward_chunk_best`` and the
    ``move_key`` reduce -- the path the subset pruning must leave bit-identical --
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
        ranks = tie_ranks(X).tolist()
        adj = pg._forward_inprocess(adj, engine, n, max_parents, n_chunks, ranks)
        adj = pg._backward_inprocess(adj, engine, n, n_chunks, ranks)
        total = _recompute_objective(adj, score, n)
    return to_structure(adj, kind="cpdag").to_numpy(), total


# Chunk counts and the d > 64 traversals have their own tests.
@pytest.mark.parametrize("seed,d,n,p,maxP", [(0, 20, 200, 0.30, None), (1, 40, 240, 0.18, 6)])
def test_serial_equals_parallel_decomposition(
    seed: int, d: int, n: int, p: float, maxP: float | None
) -> None:
    """Serial GES and two-chunk decomposition return the same CPDAG and score."""
    x = _gaussian_sem(seed, n, d, p)
    serial_adj, serial_score = _run(x, prune=True, maxP=maxP)
    decomp_adj, decomp_score = _ges_decomposed(x, 2, maxP)
    assert np.array_equal(serial_adj, decomp_adj), (
        f"decomposition differs from serial at seed={seed}, d={d}"
    )
    assert serial_score == decomp_score


def test_local_reform_matches_full_rebuild_under_knowledge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required and forbidden edges and tiers: both rebuilds apply the same moves and reject the
    same ones."""
    monkeypatch.setattr(ops, "_REGION_SHARE", 1.0)
    d = 20
    x = _gaussian_sem(3, 400, d, 0.2)
    knowledge = Knowledge(
        required=[(0, 1), (2, 3)],
        forbidden=[(4, 5), (5, 4), (6, 7), (8, 9)],
        tiers=[[10, 11, 12], [13, 14, 15]],
    )
    constraints = resolve(knowledge, d, None, method="ges")
    local_moves: list[tuple] = []
    full_moves: list[tuple] = []
    local_adj, local_score = _run(
        x, maxP=None, constraints=constraints, local_reform=True, moves=local_moves
    )
    full_adj, full_score = _run(
        x, maxP=None, constraints=constraints, local_reform=False, moves=full_moves
    )
    assert local_moves == full_moves
    assert np.array_equal(local_adj, full_adj)
    assert local_score.hex() == full_score.hex()
