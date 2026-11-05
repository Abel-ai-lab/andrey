from __future__ import annotations

import pytest

import andrey


@pytest.fixture(scope="module", autouse=True)
def _numpy_backend():
    """Fit on NumPy, the backend the baselines record, whatever the ``ANDREY_*`` variables pin.

    Module scope also covers the module fixtures that fit; the context outranks the variables.
    """
    with andrey.config(backend="numpy"):
        yield


@pytest.fixture(autouse=True)
def _pin_seed_to_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient ``ANDREY_SEED`` so seeded adapters use the baselines' default seed, 0."""
    monkeypatch.delenv("ANDREY_SEED", raising=False)
