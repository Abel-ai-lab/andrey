"""Parse the ``ANDREY_*`` environment variables, one parser per variable.

Every variable in :mod:`andrey.spec.environment` has one :class:`Setting` here. The
package reads each variable through its setting, and ``andrey config`` and ``andrey run`` check
every variable with the same settings, so a run and the report accept the same values. A
blank value counts as unset. A bad value raises ``ValueError`` naming the variable, the value, and
the accepted form.

The parsers live apart from the specs so that reading a setting imports only the standard library.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")

BACKENDS = ("auto", "numpy", "numba", "cuda", "mps", "cpu")
BACKEND_ALIASES = {"none": "cpu", "metal": "mps"}
_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")
# scikit-learn's random_state bound, the narrowest range among the seeded methods.
SEED_MAX = 2**32 - 1


def parse_backend(value: str, source: str) -> str:
    """Return the backend ``value`` names, ignoring case, with ``none`` and ``metal`` resolved."""
    b = value.strip().lower()
    b = BACKEND_ALIASES.get(b, b)
    if b not in BACKENDS:
        raise ValueError(
            f"{source} must be one of {'/'.join(BACKENDS)} (or none/metal), got {value!r}"
        )
    return b


def parse_num_workers(value: int | str, source: str) -> int:
    """Return a worker count: an integer ``>= -1``, where ``-1`` means all usable cores."""
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer, got {value!r}") from exc
    if n < -1:
        raise ValueError(
            f"{source} must be >= -1 (-1 = all usable cores, 0/1 = serial), got {value!r}"
        )
    return n


def parse_int(value: str, source: str) -> int:
    """Return ``value`` as an integer."""
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{source} must be an integer, got {value!r}") from exc


def parse_seed(value: str, source: str) -> int:
    """Return a seed: an integer from ``0`` to ``SEED_MAX``."""
    seed = parse_int(value, source)
    if not 0 <= seed <= SEED_MAX:
        raise ValueError(f"{source} must be an integer from 0 to {SEED_MAX}, got {value!r}")
    return seed


def parse_flag(value: str, source: str) -> bool:
    """Return ``True`` for ``1``/``true``/``yes``/``on`` and ``False`` for their opposites."""
    v = value.strip().lower()
    if v in _ON:
        return True
    if v in _OFF:
        return False
    raise ValueError(f"{source} must be one of {'/'.join(_ON)} or {'/'.join(_OFF)}, got {value!r}")


def parse_path(value: str, source: str) -> Path:
    """Return ``value`` as a path, with ``~`` expanded."""
    return Path(value.strip()).expanduser()


@dataclass(frozen=True)
class Setting(Generic[T]):
    """One ``ANDREY_*`` variable and its parser."""

    name: str
    parse: Callable[[str, str], T]

    def read(self) -> T | None:
        """Return the parsed value, or ``None`` when the variable is unset or blank."""
        raw = os.environ.get(self.name)
        if raw is None or not raw.strip():
            return None
        return self.parse(raw, self.name)


BACKEND = Setting("ANDREY_BACKEND", parse_backend)
DEVICE = Setting("ANDREY_DEVICE", parse_backend)
NUM_WORKERS = Setting("ANDREY_NUM_WORKERS", parse_num_workers)
GES_PARALLEL_MIN_WORK = Setting("ANDREY_GES_PARALLEL_MIN_WORK", parse_int)
HC_PARALLEL_MIN_WORK = Setting("ANDREY_HC_PARALLEL_MIN_WORK", parse_int)
COV_GPU_THRESHOLD = Setting("ANDREY_COV_GPU_THRESHOLD", parse_int)
ENTROPY_GPU_THRESHOLD = Setting("ANDREY_ENTROPY_GPU_THRESHOLD", parse_int)
GPU_CALIBRATE = Setting("ANDREY_GPU_CALIBRATE", parse_flag)
SEED = Setting("ANDREY_SEED", parse_seed)
REQUIRE_GPU = Setting("ANDREY_REQUIRE_GPU", parse_flag)
DATA_DIR = Setting("ANDREY_DATA_DIR", parse_path)

SETTINGS: dict[str, Setting] = {
    s.name: s
    for s in (
        BACKEND,
        DEVICE,
        NUM_WORKERS,
        GES_PARALLEL_MIN_WORK,
        HC_PARALLEL_MIN_WORK,
        COV_GPU_THRESHOLD,
        ENTROPY_GPU_THRESHOLD,
        GPU_CALIBRATE,
        SEED,
        REQUIRE_GPU,
        DATA_DIR,
    )
}
