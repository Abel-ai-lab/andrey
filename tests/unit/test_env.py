"""``ANDREY_*`` parsing at the sites that read each variable: blank values, flags, bad values."""

from __future__ import annotations

import random

import numpy as np
import pytest

from andrey.core import backend, env, stats
from andrey.core.seeding import resolve_seed, seed_all


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


@pytest.fixture
def global_generators():
    """Restore the global ``random``, NumPy, and torch streams that ``seed_all`` reseeds."""
    torch = backend.torch()
    python, numpy = random.getstate(), np.random.get_state()
    torch_state = None if torch is None else torch.get_rng_state()
    yield torch
    random.setstate(python)
    np.random.set_state(numpy)
    if torch is not None:
        torch.set_rng_state(torch_state)


def test_seed_all_seeds_every_global_generator_from_one_number(global_generators, monkeypatch):
    torch = global_generators

    def draws() -> tuple:
        return random.random(), np.random.rand(), None if torch is None else float(torch.rand(1))

    assert seed_all(5) == 5
    first = draws()
    seed_all(5)
    assert draws() == first
    monkeypatch.setenv("ANDREY_SEED", "7")
    assert seed_all() == 7
    monkeypatch.delenv("ANDREY_SEED")
    assert seed_all() == 0
    monkeypatch.setenv("ANDREY_SEED", "seven")
    with pytest.raises(ValueError, match="ANDREY_SEED"):
        seed_all()


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
