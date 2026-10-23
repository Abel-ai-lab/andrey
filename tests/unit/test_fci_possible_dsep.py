"""FCI's ``possible_dsep``: both modes keep their recorded PAGs, and unknown values raise.

``test_fci_oracle`` and the recovery tests check that the default does not depend on the column
order. The step reads the neighbor lists and semidirected reach of the marks once per version of
the marks, and builds them again only after an edge is removed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey.core import NULL
from andrey.core.ci import make_indep_test
from andrey.data import sample_scm

fci = importlib.import_module("andrey.constraint.fci")

# Recorded PAGs, the same for both modes on these draws. Row i holds ``to_numpy()``'s marks at
# node i: 0 none, 1 tail, 2 arrowhead, 3 circle.
RECORDED = {
    "fci": (
        {"d": 15, "n": 750, "density": 4.0, "seed": 3},
        ["00000000303300", "00000000020200", "00000002000021", "00003000003000"]
        + ["00030030000000", "00000000000000", "00003000000000", "00200000220001"]
        + ["20000002000300", "02000002001000", "20020000020020", "22000000300000"]
        + ["00200000002000", "00200002000000"],
    ),
    "gfci": (
        {"d": 8, "n": 400, "density": 4.0, "seed": 2},
        ["0200020", "2021001", "0201002", "0220100"] + ["0002022", "2000200", "0210200"],
    ),
}


@pytest.mark.parametrize("possible_dsep", ["order_free", "fast"])
@pytest.mark.parametrize("method", sorted(RECORDED))
def test_keeps_the_recorded_pag(method, possible_dsep):
    draw, rows = RECORDED[method]
    x = sample_scm(graph="erdos_renyi", latents=1, scale="standardize", **draw).data
    recorded = np.array([[int(mark) for mark in row] for row in rows])
    fit = getattr(andrey, method)(x, possible_dsep=possible_dsep)
    np.testing.assert_array_equal(fit.structure.to_numpy(), recorded)


@pytest.mark.parametrize(("method", "value"), [("fci", "order-free"), ("gfci", "FAST")])
def test_unknown_possible_dsep_raises(method, value):
    with pytest.raises(ValueError, match="possible_dsep must be 'order_free' or 'fast'"):
        getattr(andrey, method)(
            np.random.default_rng(0).standard_normal((50, 3)), possible_dsep=value
        )


def _rebuilt_per_call(marks, test, alpha, sepsets):
    """``_remove_in_node_order`` with the lists and reach built for every possible-d-sep call."""
    n = marks.shape[0]
    edges = [fci._edge_order(marks, i, j) for i in range(n) for j in range(i + 1, n) if marks[i, j]]
    for na, nb in edges:
        if not fci._adjacent(marks, na, nb):
            continue
        adj_a, adj_b = (set(np.flatnonzero(marks[v]).tolist()) for v in (na, nb))
        for a, b in ((na, nb), (nb, na)):
            dsep = fci._possible_dsep(
                marks, a, b, fci._neighbor_lists(marks), fci._semidirected_reach(marks)
            )
            if fci._try_remove_by_dsep(marks, na, nb, dsep, adj_a, adj_b, test, alpha, sepsets):
                break


@pytest.mark.parametrize(
    "draw",
    [
        {"d": 10, "n": 500, "density": 4.0, "latents": 1, "seed": 9},
        {"d": 15, "n": 500, "density": 4.0, "latents": 2, "seed": 7},
        {"d": 15, "n": 1000, "density": 3.0, "latents": 1, "seed": 3},
    ],
)
def test_the_lists_are_built_once_per_version_of_the_marks(monkeypatch, draw):
    alpha = 0.05
    x = sample_scm(graph="erdos_renyi", scale="standardize", **draw).data
    test = make_indep_test("fisherz", x)
    support, sepsets = fci._fast_adjacency_search(x, test, alpha)
    marks = fci._circle_pag(support)
    fci._orient_colliders(marks, sepsets)

    reference, reference_sepsets = marks.copy(), dict(sepsets)
    _rebuilt_per_call(reference, test, alpha, reference_sepsets)
    builds = {"lists": 0, "reach": 0}
    for name, key in (("_neighbor_lists", "lists"), ("_semidirected_reach", "reach")):
        monkeypatch.setattr(fci, name, _counting(getattr(fci, name), builds, key))
    fci._remove_in_node_order(marks, test, alpha, sepsets, None, None)

    removed = (np.count_nonzero(support) - np.count_nonzero(marks != NULL)) // 2
    assert removed >= 2, "the draw must remove edges, so that a stale list would show"
    np.testing.assert_array_equal(marks, reference)
    assert sepsets == reference_sepsets
    assert builds == {"lists": removed + 1, "reach": removed + 1}


def _counting(function, builds, key):
    def counted(*args):
        builds[key] += 1
        return function(*args)

    return counted
