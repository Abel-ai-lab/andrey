"""Reference paths the kept GES scans must agree with.

``forward_chunk_best`` scans every target's valid Inserts afresh at each pass, where
``_ForwardScan`` keeps each target's candidates between passes; the naive clique scans test every
subset, where ``clique_extensions`` and ``clique_residuals`` collapse the lattice with bitmasks
for large neighborhoods; ``full_rebuild`` re-forms the whole CPDAG after a move, where ``_reform``
rewrites the nodes the move reaches. :func:`reference_paths` swaps any of them in for a test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import pytest

from andrey.core.orient import _cpdag_of_pdag
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search.ges import _reachable


def insert_candidates(
    y: int, adj: np.ndarray, state: pg.NeighborState, forbidden: np.ndarray | None = None
) -> Iterator[tuple[int, int, tuple[int, ...], set[int]]]:
    """Yield ``(x, t_index, T, clique)`` for every Insert into ``y`` whose ``NA u T`` is a clique.

    ``clique = NA u T`` is the node set every semi-directed ``y ~> x`` path must meet for the
    Insert to be valid; the caller checks that. Sources come in ascending ``x``, and each source's
    subsets in :func:`~andrey.search.operators.clique_extensions` order. With ``forbidden``
    (``[i, j]``: no edge ``i -> j``), neither ``x`` nor a member of ``T`` is forbidden into ``y``.
    """
    adj_sets, _par_of, nbr = state
    nbrs_y = nbr[y]
    sources = np.flatnonzero((adj[:, y] == ops.NULL) & (adj[y] == ops.NULL))
    free = nbrs_y  # the undirected neighbors T may hold
    if forbidden is not None:
        sources = sources[~forbidden[sources, y]]
        free = nbrs_y - set(np.flatnonzero(forbidden[:, y]).tolist())
    for x in sources.tolist():
        if x == y:
            continue
        adj_x = adj_sets[x]
        na = nbrs_y & adj_x
        t0 = sorted(free - adj_x)
        for t_index, T in ops.clique_extensions(adj, na, t0):
            yield x, t_index, T, na | set(T)


def forward_chunk_best(
    targets: Sequence[int],
    adj: np.ndarray,
    delta_engine: pg.DeltaEngine,
    n: int,
    max_parents: float | None,
    state: pg.NeighborState,
    reachable: Callable[[int, int, set[int]], bool],
    ranks: Sequence[int],
    forbidden: np.ndarray | None = None,
    excluded: pg.Excluded = None,
    cap: float | None = None,
) -> pg.ForwardResult:
    """The best-improving Insert whose target is in ``targets``, by ``move_key``.

    Enumerates the chunk's valid Insert candidates score-free (clique + reachability), groups them
    per target by ``base = sorted(NA u T u Pa)``, and scores each group with one batched
    ``delta_many_with_base`` call over its ``("add", x)`` moves -- so one shared ``inv(cov[base,
    base])`` amortizes across all sources sharing a base. A candidate is improving iff its raw delta
    is below ``-_EPS_IMPROVE``; the argmin is over the quantized key, so the visitation order
    cannot change the result. Grouping is per target, and a target lives wholly in one
    chunk, so the batch shape (hence each delta) is identical however the targets are partitioned.
    ``forbidden`` filters the candidates as in :func:`insert_candidates`; a move in ``excluded``
    is never selected. Under ``cap``, an Insert whose family exceeds it is left out before scoring
    and one whose CPDAG breaks it is never selected.
    """
    par_of = state[1]
    best: pg.ForwardResult = None
    best_key: tuple | None = None
    for y in targets:
        paj = par_of[y]
        if max_parents is not None and len(paj) > max_parents:
            continue  # y already carries the maximum admissible parent count.
        if pg._over_cap(cap, len(paj) + 1):
            continue  # every family of y holds its parents and one source more.
        pa_set = set(int(p) for p in paj)
        # Group valid candidates by their base parent set; each base scores its adds in one batch.
        groups: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
        for x, _t_index, T, clique in insert_candidates(y, adj, state, forbidden):
            if reachable(y, x, clique):
                continue  # some semi-directed y ~> x path misses the clique: invalid Insert.
            base = tuple(sorted(pa_set | clique))
            if pg._over_cap(cap, len(base) + 1):
                continue
            groups.setdefault(base, []).append((x, T))
        for base, cands in groups.items():
            moves = [("add", x) for x, _ in cands]
            for (x, T), delta in zip(
                cands, delta_engine.delta_many_with_base(y, base, moves), strict=True
            ):
                if delta < -_EPS_IMPROVE:
                    key = pg.move_key(delta, n, x, y, T, ranks)
                    if best_key is None or key < best_key:
                        if excluded and key[1:] in excluded:
                            continue
                        if pg._breaks_cap(adj, ops.insert_cpdag, (x, y, T), cap):
                            continue
                        best_key = key
                        best = (key, x, y, T)
    return best


def naive_clique_extensions(
    adj: np.ndarray, na: Iterable[int], t0: Sequence[int]
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """:func:`~andrey.search.operators.clique_extensions` by a clique test of every subset."""
    na_set = {int(v) for v in na}
    for t_index, T in enumerate(ops.subsets(list(t0))):
        if ops.is_clique(adj, na_set | set(T)):
            yield t_index, T


def naive_clique_residuals(
    adj: np.ndarray, h0: Sequence[int]
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """:func:`~andrey.search.operators.clique_residuals` by a clique test of every residual."""
    h0 = list(h0)
    h0_set = set(h0)
    for h_index, H in enumerate(ops.subsets(h0)):
        if ops.is_clique(adj, h0_set - set(H)):
            yield h_index, H


def full_rebuild(adj: np.ndarray, touched: list[int]) -> np.ndarray:
    """The CPDAG of ``adj`` rebuilt whole, whichever nodes the move touched."""
    return _cpdag_of_pdag(adj)


def full_rescan_best(
    self: pg._ForwardScan, adj: np.ndarray, excluded: pg.Excluded = None
) -> pg.ForwardResult:
    """:meth:`_ForwardScan.best` by a scan of every target's candidates on ``adj``."""
    return forward_chunk_best(
        self.targets,
        adj,
        self.delta_engine,
        self.n,
        self.max_parents,
        pg.neighborhood_state(adj),
        _reachable(adj),
        self.ranks,
        self.forbidden,
        excluded,
        self.cap,
    )


@contextmanager
def reference_paths(
    *, naive_lattice: bool = False, full_rescan: bool = False, rebuild: bool = False
) -> Iterator[None]:
    """Run the enclosed GES in this process with the chosen reference paths in place of the kept.

    Pooled workers are other processes and keep their own paths, so a caller pins one worker.
    """
    with pytest.MonkeyPatch.context() as patch:
        if naive_lattice:
            patch.setattr(ops, "clique_extensions", naive_clique_extensions)
            patch.setattr(ops, "clique_residuals", naive_clique_residuals)
        if full_rescan:
            patch.setattr(pg._ForwardScan, "best", full_rescan_best)
        if rebuild:
            patch.setattr(ops, "_reform", full_rebuild)
        yield
