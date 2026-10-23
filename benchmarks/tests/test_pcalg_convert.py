"""``GraphStructure.from_pcalg`` and ``to_pcalg`` against the harness's pcalg decoders.

The benchmark's R adapters read pcalg's ``amat.pag`` with ``r_convert.marks_from_amat_pag`` and its
``amat.cpdag`` by transposing it in R, then ``contracts.structure_from_adjacency``. The package's
methods must read every matrix pcalg can write as the harness does, so a benchmark and a user
exchanging matrices with R agree. Needs only numpy and Andrey; R is not needed.
"""

from __future__ import annotations

import itertools

import numpy as np

from andrey import GraphStructure
from andrey_bench.adapters.r_convert import marks_from_amat_pag
from andrey_bench.contracts import structure_from_adjacency

N = 3


def test_from_pcalg_reads_every_amat_pag_as_the_harness_does():
    # Per node pair: no edge, or a mark (circle, arrowhead, tail) at each end: 10 states.
    states = [(0, 0), *itertools.product((1, 2, 3), repeat=2)]
    pairs = list(itertools.combinations(range(N), 2))
    for combination in itertools.product(states, repeat=len(pairs)):
        amat = np.zeros((N, N), dtype=int)
        for (a, b), (code_a, code_b) in zip(pairs, combination):
            amat[b, a], amat[a, b] = code_a, code_b  # amat[a, b] is the mark at b
        pag = GraphStructure.from_pcalg(amat, "pag")
        assert np.array_equal(pag.to_numpy(), marks_from_amat_pag(amat))
        assert np.array_equal(pag.to_pcalg(), amat)


def test_from_pcalg_reads_every_amat_cpdag_as_the_harness_does():
    # The R adapters hand the harness t(amat), which reads row -> column.
    pairs = [(a, b) for a in range(N) for b in range(N) if a != b]
    for cells in itertools.product((0, 1), repeat=len(pairs)):
        amat = np.zeros((N, N), dtype=int)
        for (a, b), cell in zip(pairs, cells):
            amat[a, b] = cell
        cpdag = GraphStructure.from_pcalg(amat, "cpdag")
        assert cpdag == structure_from_adjacency(amat.T, kind="cpdag")
        assert np.array_equal(cpdag.to_pcalg(), amat)
