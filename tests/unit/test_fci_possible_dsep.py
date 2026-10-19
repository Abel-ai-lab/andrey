"""FCI's ``possible_dsep``: ``"fast"`` keeps its recorded PAGs, and unknown values raise.

``test_fci_oracle`` and the recovery tests check that the default does not depend on the column
order.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.data import sample_scm

# ``"fast"``'s recorded PAGs. Row i holds ``to_numpy()``'s marks at node i: 0 none, 1 tail,
# 2 arrowhead, 3 circle.
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


@pytest.mark.parametrize("method", sorted(RECORDED))
def test_fast_keeps_the_recorded_pag(method):
    draw, rows = RECORDED[method]
    x = sample_scm(graph="erdos_renyi", latents=1, scale="standardize", **draw).data
    recorded = np.array([[int(mark) for mark in row] for row in rows])
    fast = getattr(andrey, method)(x, possible_dsep="fast")
    np.testing.assert_array_equal(fast.structure.to_numpy(), recorded)


@pytest.mark.parametrize(("method", "value"), [("fci", "order-free"), ("gfci", "FAST")])
def test_unknown_possible_dsep_raises(method, value):
    with pytest.raises(ValueError, match="possible_dsep must be 'fast' or 'order_free'"):
        getattr(andrey, method)(
            np.random.default_rng(0).standard_normal((50, 3)), possible_dsep=value
        )
