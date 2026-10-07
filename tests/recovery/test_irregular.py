from __future__ import annotations

import numpy as np
import pytest

from andrey.irregular.anm import anm
from andrey.irregular.pnl import pnl
from tests.recovery.helpers import datagen


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


# PNL draws no random state, so a fixed input pins its output. The tolerance is loose on purpose:
# these are p-values from a kernel statistic, and holding them to the last bit would gate on the
# host's SIMD kernels rather than on the method.
_PNL_CUBIC_PVALUES = (0.22597081473440506, 0.01989273405111902)
_ANM_CUBIC_PVALUES = (0.07269113995001186, 0.0)


def _pnl_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One ground-truth post-nonlinear pair ``x -> y`` with standardized, non-Gaussian variables."""
    n = int(rng.integers(200, 351))

    def nongaussian(m: int) -> np.ndarray:
        kind = rng.integers(0, 3)
        if kind == 0:
            v = rng.uniform(-2.0, 2.0, m)
        elif kind == 1:
            v = rng.laplace(0.0, 1.0, m)
        else:
            u = rng.normal(0.0, 1.0, m)
            v = np.sign(u) * np.abs(u) ** 1.5
        return (v - v.mean()) / (v.std() + 1e-12)

    inner_maps = (
        lambda z: z**3 + z,
        lambda z: np.tanh(2 * z),
        lambda z: np.sinh(z),
        lambda z: z + 0.5 * np.sin(3 * z),
    )
    outer_maps = (lambda z: z**3 + z, lambda z: np.tanh(2 * z), lambda z: np.sinh(z))
    x = nongaussian(n)
    inner = inner_maps[int(rng.integers(0, len(inner_maps)))](x)
    inner = (inner - inner.mean()) / (inner.std() + 1e-12)
    noise = 0.5 * nongaussian(n)
    y = outer_maps[int(rng.integers(0, len(outer_maps)))](inner + noise)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return x, y


def test_pnl_resolves_the_planted_direction():
    x = datagen.generate("anm_pair_cubic")
    pval_forward, pval_backward = pnl(x[:, 0], x[:, 1])
    assert pval_forward > pval_backward  # anm_pair_cubic plants x -> y
    assert np.allclose((pval_forward, pval_backward), _PNL_CUBIC_PVALUES, atol=1e-6)


def test_pnl_is_deterministic_over_varied_pairs():
    rng = np.random.default_rng(0)
    for _ in range(6):
        x, y = _pnl_pair(rng)
        first = pnl(x, y)
        assert all(0.0 <= p <= 1.0 for p in first)
        assert pnl(x, y) == first, "PNL is not deterministic"


def test_pnl_output_contract():
    from andrey.api.irregular import pnl as pnl_facade

    x = datagen.generate("anm_pair_cubic")
    out = pnl_facade(x)
    assert out.structure.type == "graph"
    assert set(out.metadata) == {"algorithm", "pval_forward", "pval_backward"}
    assert (out.metadata["pval_forward"], out.metadata["pval_backward"]) == pnl(x[:, 0], x[:, 1])


def _pairs() -> dict[str, np.ndarray]:
    """Return three sample pairs: directed, unrelated, and linear Gaussian."""
    planted = datagen.generate("anm_pair_cubic")
    rng = np.random.default_rng(0)
    n = planted.shape[0]
    unrelated = np.column_stack([rng.laplace(size=n), rng.laplace(size=n)])
    g = rng.standard_normal(n)
    gaussian = np.column_stack([g, 0.8 * g + 0.6 * rng.standard_normal(n)])
    return {"planted": planted, "unrelated": unrelated, "gaussian": gaussian}


# The measured p-values are (0.226, 0.020), (0.594, 0.298), and (0.881, 0.922): the comparison
# rule always draws an edge; the threshold rule draws one only for the planted pair.
@pytest.mark.parametrize(
    ("pair", "alpha", "edges"),
    [
        ("planted", None, [(0, 1, "directed")]),
        ("planted", 0.05, [(0, 1, "directed")]),
        ("unrelated", None, [(0, 1, "directed")]),
        ("unrelated", 0.05, []),
        ("gaussian", None, [(1, 0, "directed")]),
        ("gaussian", 0.05, []),
    ],
)
def test_pnl_decision_rules_on_measured_pairs(pair, alpha, edges):
    from andrey.api.irregular import pnl as pnl_facade

    assert pnl_facade(_pairs()[pair], alpha=alpha).structure.oriented_edges() == edges


def test_pnl_names_the_direction_whatever_the_column_order():
    import pandas as pd

    from andrey.api.irregular import pnl as pnl_facade

    df = pd.DataFrame(datagen.generate("anm_pair_cubic"), columns=["cause", "effect"])
    for columns in (["cause", "effect"], ["effect", "cause"]):
        edges = pnl_facade(df[columns]).structure.oriented_edges()
        assert edges == [("cause", "effect", "directed")]


def test_anm_resolves_the_planted_direction():
    from andrey.api._irregular import _adapt_anm

    x = datagen.generate("anm_pair_cubic")
    pval_forward, pval_backward = anm(x[:, 0], x[:, 1])
    assert pval_forward > pval_backward
    # Loosely pinned as well: the direction alone would survive a numerical regression in the GP
    # residual path that still left the forward p-value above zero.
    assert np.allclose((pval_forward, pval_backward), _ANM_CUBIC_PVALUES, atol=1e-4)
    out = _adapt_anm(pval_forward, pval_backward)
    assert out.structure.type == "graph"
    assert out.structure.oriented_edges() == [(0, 1, "directed")]
    assert set(out.metadata) == {"algorithm", "pval_forward", "pval_backward"}
