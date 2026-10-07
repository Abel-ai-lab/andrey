"""The choices `r_packages.py` makes before any fit runs.

A driver decides which solutions run and which environment each one runs under. A mistake in either
still writes a full table of plausible numbers: a run that dropped the baseline, or handed the
parallel work gate to a solution that reads none, looks like any other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import r_packages  # noqa: E402


def _lane(name: str) -> list:
    build, _packages = r_packages.LANES[name]
    return build()


@pytest.mark.parametrize("lane", sorted(r_packages.LANES))
def test_every_lane_carries_the_empty_graph_floor(lane: str) -> None:
    assert "baseline.empty" in {a.name for a in _lane(lane)}


@pytest.mark.parametrize("lane", sorted(r_packages.LANES))
def test_every_lane_pairs_andrey_against_another_package(lane: str) -> None:
    """Each comparison requires an Andrey solution and a competitor."""
    packages = {a.package for a in _lane(lane)} - {"baseline"}
    assert "andrey" in packages and len(packages) > 1, packages


#: Searches without an Andrey implementation, with reasons. Measure quality and reach, but exclude
#: speed ratios: comparing different methods would attribute method differences to implementations.
UNIMPLEMENTED = {
    "tabu": "bnlearn's tabu escapes a local optimum by accepting worse moves; andrey has none",
    "rfci": "RFCI skips FCI's possible-d-sep pass and may return a different PAG; andrey has none",
}


@pytest.mark.parametrize("lane", sorted(r_packages.LANES))
def test_every_method_in_a_lane_has_an_andrey_counterpart(lane: str) -> None:
    """Each competitor search needs an Andrey counterpart or an explicit exemption.

    bnlearn's `hc` moves one edge at a time through DAG space. GES searches CPDAG space over
    subsets of a neighbor set. Pairing those methods would measure different searches.
    """
    solutions = _lane(lane)
    mine = {a.algorithm for a in solutions if a.package == "andrey"}
    theirs = {a.algorithm for a in solutions if a.package not in ("andrey", "baseline")}
    unmatched = theirs - mine
    assert unmatched <= set(UNIMPLEMENTED), (
        f"{lane}: no andrey counterpart for {sorted(unmatched - set(UNIMPLEMENTED))}; either add "
        "one or record it in UNIMPLEMENTED with the reason"
    )


def test_no_solution_is_filed_under_another_search() -> None:
    """`algorithm` identifies the search a solution runs.

    This check detects relabeling a solution to satisfy the separate counterpart check.
    """
    declared = {a.name: a.algorithm for lane in r_packages.LANES for a in _lane(lane)}
    assert declared["bnlearn.tabu"] == "tabu"
    assert declared["pcalg.rfci"] == "rfci"
    assert declared["bnlearn.hc"] == "hc"
    assert declared["pcalg.fci"] == "fci"


def test_selecting_a_subset_keeps_the_baseline() -> None:
    """Runs above the shared size range retain the baseline for comparison."""
    chosen = r_packages.select(_lane("score"), ["pcalg.ges"])
    assert {a.name for a in chosen} == {"pcalg.ges", "baseline.empty"}


def test_selecting_nothing_runs_the_whole_lane() -> None:
    lane = _lane("pc")
    for only in (None, []):
        assert [a.name for a in r_packages.select(lane, only)] == [a.name for a in lane]


def test_selecting_a_solution_the_lane_does_not_have_stops_the_run() -> None:
    """Reject unknown solutions before materializing data or running incomplete comparisons."""
    with pytest.raises(SystemExit, match="pcalg.ges"):
        r_packages.select(_lane("pc"), ["pcalg.ges"])


GATES = {"ANDREY_GES_PARALLEL_MIN_WORK": 28104, "ANDREY_HC_PARALLEL_MIN_WORK": 30000}
NO_GATES = dict.fromkeys(GATES, None)


def test_each_work_gate_reaches_only_the_solutions_that_read_it() -> None:
    """GES and HC each read their own threshold variable.

    Assigning the GES threshold to parallel HC would record an unused value.
    """
    solutions = _lane("score")
    reached = {
        env: {a.name for a in group}
        for group, env_map in r_packages.groups(solutions, GATES)
        if env_map
        for env in env_map
    }
    assert reached == {
        "ANDREY_GES_PARALLEL_MIN_WORK": {"andrey.ges.numpy.parallel"},
        "ANDREY_HC_PARALLEL_MIN_WORK": {"andrey.hc.numpy.parallel"},
    }


def test_a_gate_left_unset_reaches_nobody() -> None:
    """An unset threshold preserves the shipped default without an overlay of `None`."""
    solutions = _lane("score")
    partial = {**NO_GATES, "ANDREY_GES_PARALLEL_MIN_WORK": 28104}
    overlays = [env for _group, env in r_packages.groups(solutions, partial) if env]
    assert overlays == [{"ANDREY_GES_PARALLEL_MIN_WORK": "28104"}]


def test_no_gate_means_one_group_and_no_overlay() -> None:
    solutions = _lane("score")
    assert r_packages.groups(solutions, NO_GATES) == [(solutions, None)]


def test_a_lane_without_a_parallel_solution_gets_no_overlay() -> None:
    """The PC solutions have no parallel mode and receive no parallel threshold."""
    solutions = _lane("pc")
    assert [env for _group, env in r_packages.groups(solutions, GATES)] == [None]
