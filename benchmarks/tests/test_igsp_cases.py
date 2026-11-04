"""Andrey's IGSP returns the true graph on each fixed case where the reference does.

The cases and the true essential graphs are in ``igsp_cases.json``, recorded by
``igsp_cases.py record`` in the reference's environment; this test needs only Andrey.
"""

from __future__ import annotations

import igsp_cases as cases


def test_andrey_finds_the_truth_on_every_seed_of_every_recorded_case():
    rows = cases.compare()
    assert len(rows) == cases.N_CASES
    assert all(row["found"] == len(cases.ANDREY_SEEDS) for row in rows), rows
