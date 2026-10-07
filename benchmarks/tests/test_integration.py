"""A failure to read one estimate costs one record, over an end-to-end micro-run.

Runs the real pipeline (keys -> materialize -> subprocess fit -> to_structure -> score) over ONE
tiny dataset (d=6, n=200). Slow-ish on a shared node (subprocess spawns + imports), but nothing is
timed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from tiny_keys import TINY_KEYS

from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.contracts import Status
from andrey_bench.integration import run_batch


def _one_tiny_key():
    """The smallest dataset of the tiny set (d=6)."""
    return [k for k in TINY_KEYS if k.d == 6][:1]


#: Parser calls, counted in the parent — ``to_structure`` runs there, so a module-level counter sees
#: every repeat in order.
_PARSE_CALLS: list[int] = []


@dataclass(frozen=True)
class StubOneBadRepeat(BaselineEmpty):
    """The empty graph, with a parser that chokes on the second repeat's output.

    The shape a real solution takes at a size it has not met before: the fit returns, and what it
    returns is not something this adapter can read.
    """

    name: str = "stub.one-bad-repeat"

    def to_structure(self, native):
        _PARSE_CALLS.append(1)
        if len(_PARSE_CALLS) == 2:
            raise ValueError("unparsable estimate")
        return super().to_structure(native)


def test_a_repeat_that_cannot_be_scored_keeps_the_repeats_that_measured(tmp_path):
    """One unreadable estimate is one ``error`` record, not a missing solution.

    Dropping the whole (dataset, solution) would take timings that already succeeded out of the
    table, and the run would still exit having written a parquet that does not mention them.
    """
    _PARSE_CALLS.clear()
    result = run_batch(
        tmp_path,
        [StubOneBadRepeat()],
        _one_tiny_key(),
        machine_id="tiny-cpu",
        cap_wall_s=240.0,
        repeats=3,
        warmup=0,
    )

    assert [r.status for r in result.records] == [
        Status.OK.value,
        Status.ERROR.value,
        Status.OK.value,
    ]
    assert len(result.errors) == 1 and "repeat 1" in result.errors[0]

    # The two that measured keep their timing and their score.
    for record in (result.records[0], result.records[2]):
        assert record.wall_s is not None and "shd" in record.metrics

    # The one that did not keeps its timing — the fit ran, the parse is what failed — and is given
    # no score standing in for the one that could not be computed.
    assert result.records[1].wall_s is not None
    assert result.records[1].metrics == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-q", "-p", "no:cacheprovider"]))
