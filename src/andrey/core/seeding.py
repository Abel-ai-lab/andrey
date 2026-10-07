"""Unified seeding for reproducible stochastic runs, across every backend.

Some algorithms draw randomness from more than one place -- stdlib ``random`` (the permutation
searches BOSS / GRaSP), numpy's global RNG, and torch (CALM's optimizer). :func:`seed_all` seeds all
of them from one number, torch-style: one call, everything reproducible on a fixed machine. The
effective seed comes from :func:`resolve_seed` -- an explicit ``seed`` wins, else the global
``ANDREY_SEED`` (default ``0``) -- so ``ANDREY_SEED=42`` makes every stochastic adapter reproducible
without threading ``seed=`` through each call. See ``docs/docs/configuration.md``.

torch is touched only through :mod:`andrey.core.backend` (behind the ``[torch]`` extra), so
importing this module -- and Andrey -- stays torch-free; torch is seeded only when installed.
"""

from __future__ import annotations

import random

import numpy as np

from . import backend, env

_DEFAULT_SEED = 0


def resolve_seed(seed: int | None = None) -> int:
    """The effective seed: an explicit ``seed`` if given, else ``ANDREY_SEED`` (default ``0``).

    Adapters whose engine takes its own ``random_state`` call this to pick the number to pass down;
    adapters that rely on a global RNG call :func:`seed_all` (which resolves through here).
    """
    if seed is not None:
        return int(seed)
    env_seed = env.SEED.read()
    return _DEFAULT_SEED if env_seed is None else env_seed


def seed_all(seed: int | None = None) -> int:
    """Seed Python's ``random``, NumPy's global generator, and torch from one number.

    Use it to make random draws outside Andrey repeatable, like data simulated with ``np.random``.
    Andrey's methods with a ``seed`` or ``random_state`` argument seed their own generator from
    it (else ``ANDREY_SEED``, else ``0``), so ``seed_all`` does not change their results. torch is
    seeded only when installed, on the CPU and on every CUDA device.

    Parameters
    ----------
    seed : int or None, default=None
        The seed. ``None`` reads ``ANDREY_SEED``, and uses ``0`` when it is unset. Any integer is
        accepted; NumPy receives its lowest 32 bits.

    Returns
    -------
    int
        The seed used.

    Raises
    ------
    ValueError
        If ``seed`` is ``None`` and ``ANDREY_SEED`` is not an integer from ``0`` to ``2**32 - 1``.
    """
    s = resolve_seed(seed)
    random.seed(s)
    np.random.seed(s & 0xFFFFFFFF)
    torch = backend.torch()
    if torch is not None:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    return s
