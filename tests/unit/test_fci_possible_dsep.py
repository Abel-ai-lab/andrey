"""FCI's ``possible_dsep``: the default keeps its recorded PAGs; ``"order_free"`` ignores labels.

``"order_free"`` builds possible-d-separation sets that do not depend on the node labels;
``test_fci_oracle`` checks its PAGs under exact tests.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import andrey
from andrey.constraint.fci import _possible_dsep
from andrey.core.structure import ARROW, CIRCLE, TAIL
from andrey.data import sample_scm


def _pag() -> np.ndarray:
    """``0 o-o 1``, ``0 <-> 3``, ``2 <-> 3``, ``4 --o 2``, and ``4 o-> 3``."""
    marks = np.zeros((5, 5), dtype=np.int8)
    for u, v, at_u, at_v in [
        (0, 1, CIRCLE, CIRCLE),
        (0, 3, ARROW, ARROW),
        (2, 3, ARROW, ARROW),
        (2, 4, CIRCLE, TAIL),
        (3, 4, ARROW, CIRCLE),
    ]:
        marks[u, v], marks[v, u] = at_u, at_v
    return marks


def test_order_free_set_ignores_node_labels():
    """The order-free set is the same under every relabeling, mapped back to the original labels.

    Node 3 is adjacent to 0. Nodes 2 and 4 are reached through the collider at 3, and each is a
    possible-parent target of the other: ``4 --o 2`` and ``2 o-- 4`` are both semidirected. The
    default set misses one of them under some labelings.
    """
    marks = _pag()
    assert set(_possible_dsep(marks, 0, 1, order_free=True)) == {2, 3, 4}
    fast_sets = set()
    for order in permutations(range(5)):
        label = np.array(order)  # node i is called label[i]
        node = np.argsort(label)
        relabeled = marks[np.ix_(node, node)]
        x, y = int(label[0]), int(label[1])
        found = _possible_dsep(relabeled, x, y, order_free=True)
        assert {int(node[v]) for v in found} == {2, 3, 4}, order
        fast_sets.add(frozenset(int(node[v]) for v in _possible_dsep(relabeled, x, y)))
    assert len(fast_sets) > 1


# The default's recorded PAGs on draws where ``"order_free"`` removes one more edge. Row i holds
# ``to_numpy()``'s marks at node i: 0 none, 1 tail, 2 arrowhead, 3 circle.
RECORDED = {
    "fci": (
        {"d": 15, "n": 750, "density": 4.0, "seed": 3},
        ["00000000201200", "00000000020200", "00000002000021", "00003000003000"]
        + ["00030030000000", "00000000000000", "00003000000000", "00200000220001"]
        + ["20000002000100", "02000002001000", "20020000020020", "22000000200000"]
        + ["00200000002000", "00200002000000"],
    ),
    "gfci": (
        {"d": 8, "n": 400, "density": 4.0, "seed": 2},
        ["0200020", "2021001", "0301001", "0220100"] + ["0002022", "2000200", "0220200"],
    ),
}


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize("method", sorted(RECORDED))
def test_default_keeps_the_recorded_pag(method):
    draw, rows = RECORDED[method]
    x = sample_scm(graph="erdos_renyi", latents=1, scale="standardize", **draw).data
    recorded = np.array([[int(mark) for mark in row] for row in rows])
    fit = getattr(andrey, method)
    np.testing.assert_array_equal(fit(x).structure.to_numpy(), recorded)
    order_free = fit(x, possible_dsep="order_free").structure.to_numpy()
    assert np.count_nonzero(order_free) < np.count_nonzero(recorded)


@pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")
@pytest.mark.parametrize(("method", "value"), [("fci", "order-free"), ("gfci", "FAST")])
def test_unknown_possible_dsep_raises(method, value):
    with pytest.raises(ValueError, match="possible_dsep must be 'fast' or 'order_free'"):
        getattr(andrey, method)(
            np.random.default_rng(0).standard_normal((50, 3)), possible_dsep=value
        )
