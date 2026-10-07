"""``ANDREY_*`` parsing at the sites that read each variable: blank values, flags, bad values."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import backend, env, stats
from andrey.core.seeding import resolve_seed


def test_blank_value_counts_as_unset(monkeypatch):
    monkeypatch.setenv("ANDREY_SEED", " ")
    assert resolve_seed() == 0


@pytest.mark.parametrize("value", ["-1", str(env.SEED_MAX + 1)])
def test_seed_outside_the_range_every_seeded_method_takes_raises(value, monkeypatch):
    monkeypatch.setenv("ANDREY_SEED", value)
    with pytest.raises(ValueError, match="ANDREY_SEED must be an integer from 0 to 4294967295"):
        resolve_seed()


def test_largest_seed_is_accepted(monkeypatch):
    monkeypatch.setenv("ANDREY_SEED", str(env.SEED_MAX))
    assert resolve_seed() == 2**32 - 1


@pytest.mark.parametrize(("value", "expected"), [("On", True), ("0", False)])
def test_flag_reads_both_states(value, expected, monkeypatch):
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", value)
    assert env.GPU_CALIBRATE.read() is expected


def test_bad_flag_raises_where_it_is_read(monkeypatch):
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "maybe")
    with pytest.raises(ValueError, match="ANDREY_GPU_CALIBRATE must be one of"):
        stats.cov(np.eye(3))


def test_bad_alias_raises_while_backend_takes_precedence(monkeypatch):
    monkeypatch.setenv("ANDREY_BACKEND", "numpy")
    monkeypatch.setenv("ANDREY_DEVICE", "quantum")
    with pytest.raises(ValueError, match="ANDREY_DEVICE"):
        _ = backend.config.backend
