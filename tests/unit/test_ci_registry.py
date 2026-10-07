"""The pluggable CI registry: any registered ``CITest`` is selectable end-to-end via ``indep_test``.

The constraint facades (pc / fci / cdnod / gfci) resolve ``indep_test`` through
:mod:`andrey.core.ci`'s registry. These tests pin the falsifiable claims: a registered stub is
dispatched, the bare ``__call__`` Protocol is the guaranteed floor, an unregistered name still
raises, and the default Fisher-Z path is unchanged. CALM is out of scope (internal Markov-blanket
pruning, no facade ``indep_test``).
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import ci


@pytest.fixture
def clean_registry():
    """Snapshot and restore the CI registry so a test's registrations do not leak into others."""
    snapshot = dict(ci._CI_REGISTRY)
    try:
        yield
    finally:
        ci._CI_REGISTRY.clear()
        ci._CI_REGISTRY.update(snapshot)


def _chain_data(seed: int = 0, n: int = 500, d: int = 6) -> np.ndarray:
    """A linear chain so a sparse skeleton leaves non-adjacent pairs for the CI test to query."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    for j in range(1, d):
        x[:, j] += 0.6 * x[:, j - 1]
    return x


class _RecordingTest:
    """A minimal ``CITest`` (only ``__call__``) recording its queries, delegating the p-value."""

    def __init__(self, data: np.ndarray) -> None:
        self.calls: list[tuple] = []
        self._fisherz = ci.FisherZ(data)  # delegate the answer so the search still converges

    def __call__(self, x: int, y: int, condition_set=()) -> float:
        self.calls.append((x, y, tuple(condition_set)))
        return self._fisherz(x, y, condition_set)


@pytest.mark.parametrize("facade", ["pc", "fci", "gfci", "cdnod"])
def test_registered_test_is_dispatched_end_to_end(facade, clean_registry):
    """Every constraint facade routes its CI queries to the registered test, not FisherZ."""
    recorders: list[_RecordingTest] = []

    def factory(data):
        rec = _RecordingTest(data)
        recorders.append(rec)
        return rec

    ci.register_indep_test("recording", factory)
    x = _chain_data(d=6)
    if facade == "cdnod":
        c = np.repeat([0.0, 1.0], len(x) // 2).reshape(-1, 1)
        andrey.cdnod(x, c, indep_test="recording")
    else:
        getattr(andrey, facade)(x, indep_test="recording")
    assert recorders, f"{facade} never constructed the registered test"
    assert any(rec.calls for rec in recorders), f"{facade} issued no CI query to the stub"


def test_minimal_protocol_is_the_guaranteed_floor(clean_registry):
    """A ``CITest`` defining only ``__call__`` (no batch, no extras) is fully usable by PC."""

    class OnlyCall:
        def __init__(self, data):
            self._fisherz = ci.FisherZ(data)

        def __call__(self, x, y, condition_set=()):
            return self._fisherz(x, y, condition_set)

    ci.register_indep_test("only_call", OnlyCall)
    out = andrey.pc(_chain_data(), indep_test="only_call")
    assert out.structure.kind == "cpdag"


def test_unknown_indep_test_raises():
    """An unregistered name raises NotImplementedError, naming what is available."""
    with pytest.raises(NotImplementedError, match="unknown indep_test"):
        andrey.pc(_chain_data(), indep_test="does_not_exist")


def test_fisherz_default_matches_explicit():
    """The default path and an explicit ``indep_test='fisherz'`` give the identical graph."""
    x = _chain_data()
    default = andrey.pc(x).structure.to_numpy()
    explicit = andrey.pc(x, indep_test="fisherz").structure.to_numpy()
    assert np.array_equal(default, explicit)


def test_registry_surface(clean_registry):
    """``register`` / ``make`` / ``available`` behave as documented; ``fisherz`` is shipped."""
    assert "fisherz" in ci.available_indep_tests()
    assert ci.SHIPPED_INDEP_TESTS == ("fisherz",)
    ci.register_indep_test("delegate", lambda data: ci.FisherZ(data))
    assert "delegate" in ci.available_indep_tests()
    assert isinstance(ci.make_indep_test("fisherz", _chain_data()), ci.FisherZ)
