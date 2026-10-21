"""The profiler's corrected shares form a whole, and its call counts repeat from fit to fit."""

from __future__ import annotations

import math

from qa import identity, profile


def test_shares_sum_to_one_and_call_counts_repeat() -> None:
    call, x = identity.FITS["PC"].call, profile.sample("asia_gauss", 0)
    with profile.single_thread():
        call(x)  # warm-up: imports and caches
        overhead = profile.calibrate()
        first, second = (profile.profile(call, x, overhead) for _ in range(2))
    assert overhead >= 0
    assert math.isclose(sum(r.share for r in first), 1.0, abs_tol=1e-9)
    assert all(0 <= r.share <= 1 and 0 <= r.cumulative_share <= 1 for r in first)
    assert sorted((r.function, r.calls) for r in first) == sorted(
        (r.function, r.calls) for r in second
    )


def test_keys_list_the_gate_datasets_then_the_larger_one() -> None:
    assert profile.keys("PC")[-1] == identity.LARGER["PC"]
    assert len(set(profile.keys("CALM"))) == len(profile.keys("CALM"))
