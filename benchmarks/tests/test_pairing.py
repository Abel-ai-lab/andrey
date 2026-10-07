"""The rules that decide whether two measurements may be divided by each other."""

from __future__ import annotations

import pandas as pd

from andrey_bench import pairing


def _rows(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_a_ratio_never_spans_two_runs():
    """Solutions in one run ran back to back under one node load; rows from two runs did not."""
    fast = _rows(run_dir=["a", "b"], dataset_id=["d1", "d1"], data_seed=[0, 0], wall_s=[1.0, 2.0])
    slow = _rows(run_dir=["a"], dataset_id=["d1"], data_seed=[0], wall_s=[10.0])
    # Only run "a" holds both arms, so the 2.0 from run "b" must not enter the median.
    assert pairing.paired_ratios(fast, slow) == [10.0]


def test_one_seed_at_two_sample_sizes_is_two_datasets():
    """A seed is one coordinate of a dataset: two datasets can share one."""
    fast = _rows(
        run_dir=["a", "a"], dataset_id=["small", "big"], data_seed=[0, 0], wall_s=[1.0, 10.0]
    )
    slow = _rows(
        run_dir=["a", "a"], dataset_id=["small", "big"], data_seed=[0, 0], wall_s=[10.0, 200.0]
    )
    assert pairing.paired_ratios(fast, slow) == [10.0, 20.0]


def test_blocked_is_not_a_ceiling():
    """A step that never ran says nothing about reach, so it must not read as a timeout."""
    assert pairing.ceiling(_rows(status=["blocked"])) == "blocked"
    assert pairing.ceiling(_rows(status=["timeout"])) == "timeout"
    assert pairing.ceiling(_rows(status=["ok", "timeout"])) is None


def test_skew_sees_the_rows_the_ratio_used():
    """An unpaired row reaches neither the ratio nor the skew reported beside it."""
    # One row unpaired by run ("b"), one unpaired by dataset ("higher").
    fast = _rows(
        run_dir=["a", "b", "a"],
        dataset_id=["shared", "shared", "higher"],
        wall_s=[1.0, 1000.0, 500.0],
    )
    slow = _rows(run_dir=["a"], dataset_id=["shared"], wall_s=[4.0])

    assert pairing.paired_ratios(fast, slow) == [4.0]
    assert pairing.duration_skew(fast, slow) == 4.0
