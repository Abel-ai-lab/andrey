"""Andrey's UT-IGSP returns the reference's answer on each recorded fixed case.

The cases and the reference's answers are in ``ut_igsp_cases.json``, recorded by
``ut_igsp_cases.py record`` in the reference's environment; this test needs only Andrey.
"""

from __future__ import annotations

import ut_igsp_cases as cases


def test_andrey_agrees_with_the_reference_on_every_recorded_case():
    rows = cases.compare()
    assert len(rows) == cases.N_CASES
    assert all(row["shd_between"] == 0 and row["same_targets"] for row in rows), rows
