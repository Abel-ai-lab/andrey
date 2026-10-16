"""Differential guard for the PC/FCI skeleton fast paths: prescreen, batched scan, worker pool.

The prescreen replaces the scalar size-0 and size-1 skeleton passes with vectorized reads of the
Fisher-Z correlation matrix, the ``|S| >= 2`` scan evaluates conditioning sets through
``FisherZ.batched_call``, and the skeleton pass may fan out across worker processes. All three are
*output-preserving*: the recovered adjacency and every removed pair's separating set must be
identical to the plain scalar adjacency search, pair for pair and byte for byte, and PC's largest
separating p-value per removed pair must be the same float on every path.

This module pins that with an independent, deliberately naive scalar reference adjacency search --
one that calls the Fisher-Z oracle (:meth:`FisherZ.__call__`) directly, with no batching, no
prescreen, and no vectorization -- and asserts the production ``_discover_skeleton`` (PC) and
``_fast_adjacency_search`` (FCI) reproduce it exactly across a sweep of random linear-Gaussian DAGs
(seeds x dimensions x densities x sample sizes, including small ``n`` near the Fisher-Z
degrees-of-freedom boundary). The reference encodes the exact semantics each search must preserve:
both PC and FCI condition on either endpoint's snapshot neighborhood and union the separating sets
(FCI records a removed pair both ways); both snapshot neighborhoods at the start of a conditioning
size and apply removals only after the size completes.

The toggle and serial-vs-parallel checks compare each fast path with the same search's scalar path.
"""

from __future__ import annotations

from functools import partial
from itertools import combinations

import numpy as np
import pytest

from andrey.constraint.fci import _fast_adjacency_search
from andrey.constraint.pc import _discover_skeleton
from andrey.core import NULL, TAIL
from andrey.core.ci import make_indep_test

# Adjacency, separating sets, and, from PC's search, the largest separating p-value per pair.
Skeleton = (
    tuple[np.ndarray, dict[tuple[int, int], tuple[int, ...]]]
    | tuple[np.ndarray, dict[tuple[int, int], tuple[int, ...]], dict[tuple[int, int], float]]
)


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the numpy oracle so an ambient ANDREY_DEVICE cannot make the correlation matrix (and thus
    # the p-values the reference and the production search compare) differ across backends.
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _gaussian_dag(seed: int, n: int, d: int, density: float) -> np.ndarray:
    """A random strictly-lower-triangular linear-Gaussian SEM with edge probability ``density``."""
    rng = np.random.default_rng(seed)
    b = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < density:
                b[j, i] = rng.uniform(0.4, 1.2) * rng.choice([-1.0, 1.0])
    e = rng.standard_normal((n, d))
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = e[:, j] + x @ b[j]
    return x


# Random linear-Gaussian DAGs across a range of dimensions. Densities stay modest at high
# d so the naive O(C(deg, k)) reference terminates quickly; small-n rows (n only a little above d)
# sit near the Fisher-Z df boundary, where near-alpha p-values stress decision agreement most.
_SWEEP = [
    (8, 11, 0.30),
    (8, 400, 0.35),
    (15, 19, 0.25),
    (15, 500, 0.30),
    (30, 400, 0.18),
    (60, 400, 0.10),
    (100, 600, 0.05),
]
_CASES = [(seed, d, n, p) for seed in range(3) for (d, n, p) in _SWEEP]
_CASE_IDS = [f"seed{s}-d{d}-n{n}-p{p}" for (s, d, n, p) in _CASES]


def _reference_skeleton(
    X: np.ndarray, alpha: float, *, mode: str, indep_test: str = "fisherz"
) -> Skeleton:
    """A naive scalar adjacency search -- the semantic oracle the fast paths must reproduce.

    Both conventions condition on either endpoint's snapshot neighborhood and union the separating
    sets; ``mode`` selects only how the pair is recorded -- ``"pc"`` once (``x < y``), ``"fci"``
    both ways.
    Every conditioning set is scored with the scalar test call directly -- no batching, no
    vectorization -- so this path shares nothing with the production search but the oracle itself.
    """
    test = make_indep_test(indep_test, X)
    d = X.shape[1]
    adj = np.full((d, d), TAIL, dtype=np.int8)
    np.fill_diagonal(adj, NULL)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}

    size = 0
    while True:
        neighbors = [np.flatnonzero(adj[i] != NULL).tolist() for i in range(d)]
        if size > max((len(nb) for nb in neighbors), default=0) - 1:
            break
        remove: list[tuple[int, int]] = []
        for x in range(d):
            for y in range(x + 1, d):
                if adj[x, y] == NULL:
                    continue
                separating: set[int] = set()
                found = False
                for a, b in ((x, y), (y, x)):
                    candidates = [c for c in neighbors[a] if c != b]
                    if len(candidates) < size:
                        continue
                    for subset in combinations(candidates, size):
                        if test(a, b, subset) > alpha:
                            found = True
                            separating.update(subset)
                if found:
                    remove.append((x, y))
                    key = tuple(sorted(separating))
                    sepsets[(x, y)] = key
                    if mode == "fci":
                        sepsets[(y, x)] = key
        for x, y in remove:
            adj[x, y] = NULL
            adj[y, x] = NULL
        size += 1

    return adj, sepsets


def _assert_same(lhs: Skeleton, rhs: Skeleton, label: str) -> None:
    """Assert two skeletons share adjacency (byte for byte) and separating sets (key for key).

    When both carry PC's separating p-values, every removed pair has one, and they are equal
    floats, pair for pair.
    """
    assert np.array_equal(lhs[0] != NULL, rhs[0] != NULL), f"{label}: adjacency differs"
    assert lhs[1] == rhs[1], f"{label}: separating sets differ"
    if len(lhs) == len(rhs) == 3:
        assert lhs[2].keys() == lhs[1].keys(), f"{label}: a removed pair has no p-value"
        assert lhs[2] == rhs[2], f"{label}: separating p-values differ"


@pytest.mark.parametrize(("seed", "d", "n", "p"), _CASES, ids=_CASE_IDS)
def test_pc_skeleton_matches_scalar_reference(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    prod = _discover_skeleton(x, 0.05, "fisherz")
    ref = _reference_skeleton(x, 0.05, mode="pc")
    _assert_same(prod, ref, f"PC seed={seed} d={d} n={n} p={p}")


@pytest.mark.parametrize(("seed", "d", "n", "p"), _CASES, ids=_CASE_IDS)
def test_fci_skeleton_matches_scalar_reference(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    test = make_indep_test("fisherz", x)
    prod = _fast_adjacency_search(x, test, 0.05)
    ref = _reference_skeleton(x, 0.05, mode="fci")
    _assert_same(prod, ref, f"FCI seed={seed} d={d} n={n} p={p}")


# ---- prescreen on/off is bit-identical (the vectorized path vs the scalar scan) ------------------


@pytest.mark.parametrize(("seed", "d", "n", "p"), _CASES, ids=_CASE_IDS)
def test_pc_prescreen_matches_unprescreened(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    on = _discover_skeleton(x, 0.05, "fisherz", prescreen=True)
    off = _discover_skeleton(x, 0.05, "fisherz", prescreen=False)
    _assert_same(on, off, f"PC prescreen toggle seed={seed} d={d} n={n} p={p}")


@pytest.mark.parametrize(("seed", "d", "n", "p"), _CASES, ids=_CASE_IDS)
def test_fci_prescreen_matches_unprescreened(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    on = _fast_adjacency_search(x, make_indep_test("fisherz", x), 0.05, prescreen=True)
    off = _fast_adjacency_search(x, make_indep_test("fisherz", x), 0.05, prescreen=False)
    _assert_same(on, off, f"FCI prescreen toggle seed={seed} d={d} n={n} p={p}")


# ---- the prescreen leaves the full CPDAG / PAG unchanged (covers downstream queries) ----------
#
# The skeleton comparisons above pin the adjacency and separating sets. The full pipelines also
# issue CI queries downstream (FCI's possible-d-sep pass and discriminating-path rule), so these
# assert the recovered CPDAG / PAG itself is byte-identical with the prescreen on versus off --
# forcing prescreen off by wrapping the internal search each facade calls.

_PIPELINE_SWEEP = [(6, 300, 0.40), (8, 400, 0.30), (10, 500, 0.30)]
_PIPELINE_CASES = [(seed, d, n, p) for seed in range(2) for (d, n, p) in _PIPELINE_SWEEP]
_PIPELINE_IDS = [f"seed{s}-d{d}-n{n}-p{p}" for (s, d, n, p) in _PIPELINE_CASES]


@pytest.mark.parametrize(("seed", "d", "n", "p"), _PIPELINE_CASES, ids=_PIPELINE_IDS)
def test_pc_pipeline_prescreen_invariant(
    monkeypatch: pytest.MonkeyPatch, seed: int, d: int, n: int, p: float
) -> None:
    import importlib

    import andrey

    pc_mod = importlib.import_module("andrey.constraint.pc")
    x = _gaussian_dag(seed, n, d, p)
    prescreened = andrey.pc(x).structure.to_numpy()
    monkeypatch.setattr(pc_mod, "_discover_skeleton", partial(_discover_skeleton, prescreen=False))
    baseline = andrey.pc(x).structure.to_numpy()
    assert np.array_equal(prescreened, baseline), f"PC CPDAG differs seed={seed} d={d}"


@pytest.mark.parametrize(("seed", "d", "n", "p"), _PIPELINE_CASES, ids=_PIPELINE_IDS)
def test_fci_pipeline_prescreen_invariant(
    monkeypatch: pytest.MonkeyPatch, seed: int, d: int, n: int, p: float
) -> None:
    import importlib

    import andrey

    fci_mod = importlib.import_module("andrey.constraint.fci")
    x = _gaussian_dag(seed, n, d, p)
    prescreened = andrey.fci(x).structure.to_numpy()
    monkeypatch.setattr(
        fci_mod, "_fast_adjacency_search", partial(_fast_adjacency_search, prescreen=False)
    )
    baseline = andrey.fci(x).structure.to_numpy()
    assert np.array_equal(prescreened, baseline), f"FCI PAG differs seed={seed} d={d}"


@pytest.mark.parametrize(("seed", "d", "n", "p"), _PIPELINE_CASES, ids=_PIPELINE_IDS)
def test_cdnod_pipeline_prescreen_invariant(
    monkeypatch: pytest.MonkeyPatch, seed: int, d: int, n: int, p: float
) -> None:
    import importlib

    import andrey

    cdnod_mod = importlib.import_module("andrey.constraint.cdnod")
    x = _gaussian_dag(seed, n, d, p)
    context = np.zeros((n, 1))
    context[n // 2 :, 0] = 1.0
    prescreened = andrey.cdnod(x, context).structure.to_numpy()
    monkeypatch.setattr(
        cdnod_mod, "_discover_skeleton", partial(_discover_skeleton, prescreen=False)
    )
    baseline = andrey.cdnod(x, context).structure.to_numpy()
    assert np.array_equal(prescreened, baseline), f"CDNOD CPDAG differs seed={seed} d={d}"


# ---- serial and parallel skeleton searches are byte-identical ------------------------------------
#
# At each conditioning size the pairs are scanned independently and removals applied only after the
# size completes, so distributing the pairs over worker processes and reducing by union cannot alter
# the outcome. These pin that: the same skeleton whether the |S| >= 2 scan runs in-process or fans
# out. Densities are chosen so the search reaches |S| >= 2 (the sizes the pool handles).

_PARALLEL_SWEEP = [(15, 500, 0.30), (20, 600, 0.25), (30, 500, 0.18)]
_PARALLEL_CASES = [(seed, d, n, p) for seed in range(2) for (d, n, p) in _PARALLEL_SWEEP]
_PARALLEL_IDS = [f"seed{s}-d{d}-n{n}-p{p}" for (s, d, n, p) in _PARALLEL_CASES]


@pytest.mark.parametrize(("seed", "d", "n", "p"), _PARALLEL_CASES, ids=_PARALLEL_IDS)
def test_pc_serial_matches_parallel(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    serial = _discover_skeleton(x, 0.05, "fisherz", workers=1)
    parallel = _discover_skeleton(x, 0.05, "fisherz", workers=3)
    _assert_same(serial, parallel, f"PC serial/parallel seed={seed} d={d} n={n} p={p}")


@pytest.mark.parametrize(("seed", "d", "n", "p"), _PARALLEL_CASES, ids=_PARALLEL_IDS)
def test_fci_serial_matches_parallel(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    serial = _fast_adjacency_search(x, make_indep_test("fisherz", x), 0.05, workers=1)
    parallel = _fast_adjacency_search(x, make_indep_test("fisherz", x), 0.05, workers=3)
    _assert_same(serial, parallel, f"FCI serial/parallel seed={seed} d={d} n={n} p={p}")


# ---- Student-t takes the same fast paths ---------------------------------------------------------
#
# The Student-t test shares the partial-correlation base, so the prescreen, the batched scan, and
# the worker pool engage for it too. The checks above repeat on a smaller sweep with
# ``indep_test="student_t"``. The parallel cases are small samples where Student-t and Fisher-Z
# give different skeletons, so a worker that rebuilt the wrong test would be caught.

_T_SWEEP = [(8, 11, 0.30), (15, 500, 0.30), (30, 400, 0.18)]
_T_CASES = [(seed, d, n, p) for seed in range(2) for (d, n, p) in _T_SWEEP]
_T_IDS = [f"seed{s}-d{d}-n{n}-p{p}" for (s, d, n, p) in _T_CASES]
_T_PARALLEL_CASES = [(3, 10, 25, 0.30), (4, 10, 30, 0.30), (0, 8, 30, 0.30)]
_T_PARALLEL_IDS = [f"seed{s}-d{d}-n{n}-p{p}" for (s, d, n, p) in _T_PARALLEL_CASES]


@pytest.mark.parametrize(("seed", "d", "n", "p"), _T_CASES, ids=_T_IDS)
def test_student_t_skeletons_match_scalar_reference(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    label = f"seed={seed} d={d} n={n} p={p}"
    pc = _discover_skeleton(x, 0.05, "student_t")
    _assert_same(pc, _reference_skeleton(x, 0.05, mode="pc", indep_test="student_t"), f"PC {label}")
    fci = _fast_adjacency_search(x, make_indep_test("student_t", x), 0.05)
    _assert_same(
        fci, _reference_skeleton(x, 0.05, mode="fci", indep_test="student_t"), f"FCI {label}"
    )


@pytest.mark.parametrize(("seed", "d", "n", "p"), _T_CASES, ids=_T_IDS)
def test_student_t_prescreen_matches_unprescreened(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    label = f"seed={seed} d={d} n={n} p={p}"
    _assert_same(
        _discover_skeleton(x, 0.05, "student_t", prescreen=True),
        _discover_skeleton(x, 0.05, "student_t", prescreen=False),
        f"PC {label}",
    )
    _assert_same(
        _fast_adjacency_search(x, make_indep_test("student_t", x), 0.05, prescreen=True),
        _fast_adjacency_search(x, make_indep_test("student_t", x), 0.05, prescreen=False),
        f"FCI {label}",
    )


@pytest.mark.parametrize(("seed", "d", "n", "p"), _T_PARALLEL_CASES, ids=_T_PARALLEL_IDS)
def test_student_t_serial_matches_parallel(seed: int, d: int, n: int, p: float) -> None:
    x = _gaussian_dag(seed, n, d, p)
    label = f"seed={seed} d={d} n={n} p={p}"
    serial = _discover_skeleton(x, 0.05, "student_t", workers=1)
    fisherz = _discover_skeleton(x, 0.05, "fisherz", workers=1)
    assert serial[1] != fisherz[1] or not np.array_equal(serial[0], fisherz[0]), "case too easy"
    _assert_same(serial, _discover_skeleton(x, 0.05, "student_t", workers=3), f"PC {label}")
    fci_serial = _fast_adjacency_search(x, make_indep_test("student_t", x), 0.05, workers=1)
    fci_parallel = _fast_adjacency_search(x, make_indep_test("student_t", x), 0.05, workers=3)
    _assert_same(fci_serial, fci_parallel, f"FCI {label}")
