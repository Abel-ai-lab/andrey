from __future__ import annotations

from typing import Any

import numpy as np

from tests.recovery.helpers import harness
from tests.recovery.helpers.project import project_output

__all__ = ["assert_recorded_output", "assert_recorded_output_mec", "project_output"]


def assert_recorded_output(out: Any, algorithm: str, *, case: str) -> harness.RegressionResult:
    return harness.assert_baseline(algorithm, project_output(out, algorithm), case=case)


def assert_recorded_output_mec(out: Any, algorithm: str, *, case: str) -> None:
    """Compare the CPDAGs of the candidate and recorded DAGs.

    Score-equivalent optima can differ in tie-breaking, so ExactSearch compares Markov
    equivalence classes. Check the input hash before comparing to detect generator drift.
    """
    from andrey.core.orient import dag2cpdag

    baseline = harness.check_input(case, algorithm)["output"]["graph"]
    baseline_dag = np.asarray(baseline, dtype=np.int8)
    native_dag = out.structure.to_numpy()
    if not np.array_equal(dag2cpdag(native_dag), dag2cpdag(baseline_dag)):
        raise AssertionError(f"{algorithm}: CPDAG (Markov-equivalence-class) mismatch vs baseline")
