from __future__ import annotations

import numpy as np
import pandas as pd
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


@pytest.fixture(scope="module")
def sachs_conditions(tmp_path_factory):
    """The Sachs cells of all nine conditions, log-transformed as Mooij et al. (2020) use them: the
    data, each cell's condition, and the consensus network."""
    from andrey.data import load_dataset
    from tests.unit.conftest import serve_datasets_offline

    with pytest.MonkeyPatch.context() as monkeypatch:
        serve_datasets_offline(monkeypatch, tmp_path_factory.mktemp("cache"))
        pooled = load_dataset("sachs", conditions=True)
    logged = pd.DataFrame(np.log(pooled.data.to_numpy()), columns=list(pooled.feature_names))
    return logged, pooled.env, pooled.graph
