"""The ``--shard`` option splits the collected tests into disjoint parts that together hold all."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.conftest import parse_shard, shard_of

ROOT = Path(__file__).resolve().parents[2]
IDS = [f"test_{name}.py::test_{case}[{p}]" for name in "abc" for case in "xy" for p in range(500)]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_every_test_belongs_to_one_part_and_the_parts_are_of_similar_size(count):
    sizes = Counter(shard_of(nodeid, count) for nodeid in IDS)
    assert set(sizes) == set(range(1, count + 1))
    mean = len(IDS) / count
    assert all(abs(size - mean) < 0.1 * mean for size in sizes.values())


@pytest.mark.parametrize("spec", ["0/2", "3/2", "1", "a/b", "1/2/3", "", "-1/2"])
def test_a_malformed_part_is_a_usage_error(spec):
    with pytest.raises(pytest.UsageError, match="--shard takes K/N"):
        parse_shard(spec)


def _collected(*options: str) -> set[str]:
    """The ids of this file's tests that a collect-only run selects."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + [*options, str(Path(__file__))],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return {line for line in run.stdout.splitlines() if "::" in line}


def test_the_parts_of_a_run_are_disjoint_and_together_hold_every_test():
    everything = _collected()
    parts = [_collected(f"--shard={k}/2") for k in (1, 2)]
    assert len(everything) > 2
    assert all(parts)
    assert set().union(*parts) == everything
    assert sum(len(part) for part in parts) == len(everything)
