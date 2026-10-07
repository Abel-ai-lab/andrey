from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _pin_seed_to_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient ``ANDREY_SEED`` so seeded adapters use the baselines' default seed, 0."""
    monkeypatch.delenv("ANDREY_SEED", raising=False)
