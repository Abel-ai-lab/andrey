"""Test GES reachability on both sides of the 64-node bitmask limit.

Compare GES with an independent Python-integer reachability oracle at 63, 64, 65, and 80 nodes.
Also check agreement between set and bitmask search below the limit and rejection of oversized
single-word successor masks.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from andrey.core import backend
from andrey.core.structure import TAIL
from andrey.search.ges import ges

# ``andrey.search.ges`` the attribute is the re-exported ``ges`` function, so monkeypatching the
# module's internals goes through the submodule object rather than the shadowed attribute path.
_GES_MOD = sys.modules["andrey.search.ges"]

_LARGE_D = [63, 64, 65, 80]
_MAXP = 3  # bounds density so the O(edges * d^2) serial search stays cheap


def _large_d_data(d: int, seed: int = 0) -> np.ndarray:
    """A sparse linear-Gaussian SEM on ``d`` variables with ~10 active nodes spread across 0..d-1.

    The small active set bounds the accepted-move count (hence runtime). Edges follow a *random*
    topological order over the active nodes, not index order, so nodes above 64 sit on both ends of
    semi-directed paths (not only as sinks) -- a truncated high bit then changes a search decision,
    which is what makes the test falsify a packing bug instead of passing straight through it.
    """
    rng = np.random.default_rng(seed)
    active = np.unique(np.linspace(0, d - 1, 10).round().astype(int))
    order = rng.permutation(active)  # random topological order: high-index nodes are not all sinks
    b = np.zeros((d, d))
    for ai in range(len(order)):
        for aj in range(ai + 1, len(order)):
            if rng.random() < 0.35:
                parent, child = int(order[ai]), int(order[aj])  # parent precedes child in the order
                b[child, parent] = rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0])
    e = rng.standard_normal((150, d))
    x = np.zeros((150, d))
    active_set = {int(v) for v in order}
    # Fill columns in topological order (active parents first), then the pure-noise inactive nodes.
    fill_order = [int(v) for v in order] + [j for j in range(d) if j not in active_set]
    for node in fill_order:
        x[:, node] = e[:, node] + x @ b[node]
    return x


def _oracle_reach_factory(n: int):
    """Independent reachability for GES: correct Python-int successor masks + a bit-iterating BFS.

    Algorithmically distinct from both shipped paths (the ``uint64`` bitmask BFS and the set BFS
    over successor lists) and width-unlimited by construction, so agreement is real validation.
    Monkeypatched over ``ges._reach_factory`` to rerun the same search with a different oracle.
    """

    def build_state(adj: np.ndarray):
        succ = [
            int(sum(1 << int(c) for c in np.nonzero(adj[node] == TAIL)[0])) for node in range(n)
        ]

        def reachable(y: int, x: int, clique) -> bool:
            if y == x:
                return True
            bar = 0
            for c in clique:
                if c != y and c != x:
                    bar |= 1 << int(c)
            seen = 1 << y
            frontier = [y]
            while frontier:
                u = frontier.pop()
                mask = succ[u]
                v = 0
                while mask:
                    if mask & 1:
                        if v == x:
                            return True
                        bit = 1 << v
                        if not (bar & bit) and not (seen & bit):
                            seen |= bit
                            frontier.append(v)
                    mask >>= 1
                    v += 1
            return False

        return reachable

    return build_state


@pytest.mark.parametrize("d", _LARGE_D)
def test_ges_large_d_serial(d: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial GES matches an independent reachability oracle at 63, 64, 65, and 80 nodes."""
    x = _large_d_data(d)
    shipped, score = ges(x, maxP=_MAXP)  # bitmask path for d <= 64, set BFS above -- must not raise
    assert shipped.kind == "cpdag"
    assert shipped.n_nodes == d

    monkeypatch.setattr(_GES_MOD, "_reach_factory", _oracle_reach_factory)
    oracle, oracle_score = ges(x, maxP=_MAXP)
    assert np.array_equal(shipped.to_numpy(), oracle.to_numpy()), (
        f"CPDAG differs from oracle at d={d}"
    )
    assert score == oracle_score  # identical move sequence -> bit-identical accumulated BIC


@pytest.mark.parametrize("d", [63, 64])
def test_ges_large_d_numba_numpy_agree(d: int) -> None:
    """At the bit-63/64 boundary the numba kernel and the numpy reference return the same CPDAG."""
    if backend.numba() is None:
        pytest.skip("numba (the [numba] extra) is not installed")
    x = _large_d_data(d)
    with backend.config(backend="numpy"):
        cpdag_np, _ = ges(x, maxP=_MAXP)
    with backend.config(backend="numba"):
        cpdag_nb, _ = ges(x, maxP=_MAXP)
    assert cpdag_np == cpdag_nb


@pytest.mark.parametrize("d", [8, 20, 64])
def test_ges_setbfs_matches_bitset_below_cap(d: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set and bitmask reachability produce the same CPDAG at or below 64 nodes."""
    x = _large_d_data(d)
    bitmask_cpdag, bitmask_score = ges(x, maxP=_MAXP)
    monkeypatch.setattr(_GES_MOD, "_BITSET_MAX_D", 0)  # every d now takes the set-BFS branch
    setbfs_cpdag, setbfs_score = ges(x, maxP=_MAXP)
    assert np.array_equal(bitmask_cpdag.to_numpy(), setbfs_cpdag.to_numpy())
    assert bitmask_score == setbfs_score  # byte-for-byte: same moves, same accumulated BIC


def test_tail_successor_masks_rejects_above_cap() -> None:
    """The single-``uint64`` packing raises above 64 instead of silently truncating high bits."""
    from andrey.core import _bitset

    _bitset.tail_successor_masks(np.zeros((64, 64), dtype=np.int8))  # the cap itself is fine
    with pytest.raises(ValueError, match="d <= 64"):
        _bitset.tail_successor_masks(np.zeros((65, 65), dtype=np.int8))
