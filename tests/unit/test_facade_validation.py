"""Facade input-validation guards: shape errors surface as the documented ``ValueError``.

Direct Python calls (not the CLI, which validates inputs of its own) must not leak a raw
``IndexError`` / ``LinAlgError`` on a bad shape -- the docstrings promise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey


def test_direct_lingam_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        andrey.direct_lingam(np.random.default_rng(0).standard_normal(100))


@pytest.mark.parametrize("shape", [(100,), (100, 1), (100, 3)])
def test_pnl_rejects_anything_but_a_pair(shape):
    with pytest.raises(ValueError, match=r"\(n_samples, 2\)"):
        andrey.pnl(np.random.default_rng(0).standard_normal(shape))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_pnl_rejects_an_alpha_outside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        andrey.pnl(np.random.default_rng(0).standard_normal((100, 2)), alpha=alpha)
