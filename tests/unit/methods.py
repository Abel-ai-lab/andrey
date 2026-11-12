"""What the package-wide tests pass a method besides its data and its own settings.

Several package-wide tests share these tables and contexts to fit a method; a test keeps beside them
only what is particular to that test, such as the targets that name its own columns.
"""

from __future__ import annotations

import numpy as np

#: Settings that shorten a slow fit; the facade path stays the same, and the properties the
#: package-wide tests check do not depend on the fit converging.
SHORT = {
    "calm": {"max_iter": 1, "subproblem_iter": 20},
    "varma_lingam": {"max_iter": 1},
    "jpcmciplus": {"time_dummy": False},
}
#: The arguments a method needs beyond its inputs: the target of ICP and MB-by-MB, by position.
REQUIRED = {"icp": {"target": 2}, "mb_by_mb": {"target": 2}}


def fit_settings(method: str) -> dict:
    """The settings of a package-wide fit of ``method``: its short schedule and the arguments it
    requires."""
    return {**SHORT.get(method, {}), **REQUIRED.get(method, {})}


def environments(n: int) -> np.ndarray:
    """Three environments of about ``n / 3`` rows each, in row order: the optional context of a
    method that reads environments, such as igsp and ut_igsp."""
    return np.arange(n) * 3 // n


def domains(n: int) -> np.ndarray:
    """Two domains of ``n / 2`` rows each, as a column: a required context, such as CD-NOD's
    domain index."""
    return np.repeat([0.0, 1.0], n // 2).reshape(-1, 1)
