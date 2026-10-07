"""Orientation traces replay exactly and preserve ordinary completion."""

import numpy as np
import pytest

from andrey.core.orient import meek


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("d", [4, 8, 16])
def test_trace_replays_exactly(seed, d):
    rng = np.random.default_rng(seed)
    adj = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 0.35:
                adj[i, j], adj[j, i] = 1, rng.choice([1, 2])
    original = adj.copy()
    expected = meek(adj)
    steps = []
    actual = meek(adj, _trace=steps)
    np.testing.assert_array_equal(adj, original)
    np.testing.assert_array_equal(actual, expected)
    replay = original
    for step in steps:
        assert step["rule"] in {"R1", "R2", "R3"}
        np.testing.assert_array_equal(step["before"], replay)
        delta = step["after"] != step["before"]
        assert delta.sum() == 1
        assert step["before"][delta].item() == 1
        assert step["after"][delta].item() == 2
        replay = step["after"]
    np.testing.assert_array_equal(replay, expected)


def test_trace_snapshots_do_not_alias():
    adj = np.array([[0, 1, 0, 0], [2, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    steps = []
    result = meek(adj, _trace=steps)
    assert len(steps) == 2
    before = steps[0]["before"].copy()
    result[:] = 0
    steps[-1]["after"][:] = 0
    np.testing.assert_array_equal(steps[0]["before"], before)
