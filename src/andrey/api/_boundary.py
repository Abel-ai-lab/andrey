"""The warning a fit gives when its data are larger than the method's measured boundary.

The published boundary lane steps each method's size on several settings until its fits stop
finishing within the caps. A fit warns, once per method, when its size is above the largest size
that finished at the nearest measured setting and a ladder there ended on the time or memory cap.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

from andrey.core.warning_policy import BoundaryWarning, warn_once

__all__ = ["Ladder", "measured", "past_boundary", "warn_past_boundary"]

#: The release cards' address: a card's page is ``<CARDS><release>.html``.
CARDS = "https://andrey.abel.ai/docs/benchmarks/"
#: The stops that end a ladder on a cap; past any other stop the method was not measured.
CAP_STOPS = ("time", "memory")


class Ladder(NamedTuple):
    """One ladder of the boundary lane: one method on one setting, stepped until a stop.

    ``rows`` is the setting's rows, its rows per variable when ``per_variable``, and ``None`` for
    the pair methods, whose ladder steps the rows. ``last_through`` is the largest size whose fits
    finished within the caps, counted in ``unit``: ``"variables"``, the data's columns, or
    ``"rows"``. Where the generator hides nodes, it counts the observed ones and
    ``last_through_d`` every node. ``stop`` ended the ladder: ``"time"`` or ``"memory"`` (three
    fits of the next size ran past a cap), ``"error"``, ``"accuracy"``, or ``"ladder_end"``.
    """

    label: str
    rows: int | None
    per_variable: bool
    mean_degree: int | None
    unit: str
    last_through: int | None
    last_through_d: int | None
    stop: str


@functools.cache
def measured() -> dict[str, tuple[Ladder, ...]]:
    """Return every method's measured ladders, read once from the generated table."""
    from andrey.api._boundary_table import LADDERS

    return {method: tuple(Ladder(*row) for row in rows) for method, rows in LADDERS.items()}


def past_boundary(method: str, shape: tuple[int, ...]) -> tuple[Ladder, int] | None:
    """Return the ladder whose size data of ``shape`` pass, and how many ladders its setting has.

    The size is the data's columns, or their rows for a method whose ladders step rows. The
    nearest setting has the rows, or rows per variable, closest to the data's on a log scale;
    a tie takes both. The threshold is the largest size any ladder at that setting finished
    within the caps, the most at any measured density, as the data's density is unknown. Data
    pass it when their size is larger and a ladder that reached it ended on the time or memory
    cap; a larger size after an accuracy, error, or last-rung stop was not measured, so it
    returns ``None``, as do sizes at or below the threshold and methods with no ladder.
    """
    ladders = measured().get(method, ())
    if not ladders:
        return None
    n_rows, n_columns = shape[0], shape[1]
    size = n_rows if ladders[0].unit == "rows" else n_columns
    if all(ladder.last_through is None or size <= ladder.last_through for ladder in ladders):
        return None

    def distance(ladder: Ladder) -> float:
        if ladder.rows is None:
            return 0.0
        rows = n_rows / n_columns if ladder.per_variable else n_rows
        return abs(math.log(rows / ladder.rows))

    nearest = min(distance(ladder) for ladder in ladders)
    setting = [ladder for ladder in ladders if math.isclose(distance(ladder), nearest)]
    limit = max((lad.last_through for lad in setting if lad.last_through is not None), default=size)
    capped = [lad for lad in setting if lad.last_through == limit and lad.stop in CAP_STOPS]
    return (capped[0], len(setting)) if size > limit and capped else None


def warn_past_boundary(method: str, shape: tuple[int, ...]) -> None:
    """Warn once per method when data of ``shape`` are larger than the method's boundary."""
    found = past_boundary(method, shape)
    if found is None:
        return
    from andrey.api._boundary_table import CORES, MEMORY_CAP, RELEASE, TIME_CAP

    ladder, settings = found
    n_rows, n_columns = shape[0], shape[1]
    if ladder.unit == "rows":
        given, limit = f"{n_rows:,} rows", f"{ladder.last_through:,} rows"
    else:
        given, limit = f"{n_columns:,} variables and {n_rows:,} rows", f"{ladder.last_through:,}"
        if ladder.last_through_d != ladder.last_through:
            limit += f" observed variables, {ladder.last_through_d:,} counting the hidden ones"
        else:
            limit += " variables"
    most = ", the most at any of the mean degrees measured at that setting" if settings > 1 else ""
    cap = f"ran past {TIME_CAP}" if ladder.stop == "time" else f"ran out of the {MEMORY_CAP}"
    warn_once(
        f"andrey.{method} got {given}. In the {RELEASE} benchmark, its fits finished within the "
        f"caps ({TIME_CAP}, {MEMORY_CAP}, {CORES} cores) through {limit} ({ladder.label}{most}), "
        f"and three fits of the next size {cap}. The measured size is a reference, not a hard "
        "limit: it changes with the data generator, the density, the method's settings, and the "
        f"machine. See the Limitations paragraph of help(andrey.{method}) and the release card, "
        f"{CARDS}{RELEASE}.html.",
        BoundaryWarning,
        key=method,
    )
