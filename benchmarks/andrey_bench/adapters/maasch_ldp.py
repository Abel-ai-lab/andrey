"""LDP and LD3 adapters for the authors' code (Maasch et al., 2024 and 2025), run unmodified in
their shared environment.

The code is ``jmaasch/ldp`` at commit :data:`LDP_COMMIT` and ``jmaasch/LD3`` at commit
:data:`LD3_COMMIT`, repositories with no license file. They are measurement references only:
``benchmarks/start_maasch.sh`` downloads the archive of each commit, checks its SHA-256, and builds
the one environment both run in (the two modules import the same packages); the benchmark imports
the code from the directories ``$ANDREY_BENCH_LDP`` and ``$ANDREY_BENCH_LD3`` name and copies none
of it.

Both fits build a frame whose columns are named ``V0`` to ``V{d-1}`` and call the code as its demo
and notebooks do:

- :class:`MaaschLDP`: ``LDP(data=df, independence_test=...).partition_z(exposure, outcome, alpha)``.
  The code returns the partition of the other variables; its Z5 criterion is the attribute
  ``vas_exists``, set when some instrument candidate is independent of the outcome given the
  confounders and the treatment. The record is identifiable with the set ``Z1`` as a valid set when
  the criterion holds, and not identifiable with no set when it does not. The code never returns
  the empty set as a valid set: with no confounder found it reports "not identifiable".
- :class:`MaaschLD3`: ``LD3(data=df, independence_test=...).get_sdc_cde_adjustment(exposure,
  outcome, alpha)``. The code returns the structural direct criterion (1 when the treatment and the
  outcome are dependent given the confounder-or-mediator parents of the outcome) and ``ADE``, the
  parents of the outcome it found. The record is a direct effect whose ``nonzero`` is the
  criterion, with ``ADE`` as its one set (possibly empty); ``identifiable`` is ``None``.

The test is Fisher's z test (the code's ``"fisher"``) for ``indep_test="fisherz"`` and the
chi-square test (the code's ``"chi"``) for ``"chisq"``. Both use the code's default level, 0.005,
passed by the lane. The number of tests is the code's own ``total_tests``, which counts every call
of its test wrapper including those the test object answers from its cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adjustment_lane import (
    OUTCOME_PARAM,
    TREATMENT_PARAM,
    empty_structure,
    pack_effects,
)

#: Name the directories holding the extracted code of each repository.
LDP_ENV = "ANDREY_BENCH_LDP"
LD3_ENV = "ANDREY_BENCH_LD3"

#: The pinned commits; ``start_maasch.sh`` fetches them.
LDP_COMMIT = "838f6e26b669ab049b797f0e2eda1f31faa316fa"
LD3_COMMIT = "74b91cb9ae13e05d26c23095410ee908616256e8"

#: The lane's settings: the code's default level, and Fisher's z test.
DEFAULTS = {"alpha": 0.005, "indep_test": "fisherz"}

#: The code's name of each test the lane can ask for.
_TESTS = {"fisherz": "fisher", "chisq": "chi"}


def _code_dir(variable: str, module: str) -> Path:
    raw = os.environ.get(variable, "").strip()
    directory = Path(raw)
    if not raw or not (directory / f"{module}.py").is_file():
        raise FileNotFoundError(
            f"${variable}={raw!r} does not hold {module}.py; run benchmarks/start_maasch.sh"
        )
    return directory


def _test_name(indep_test: str) -> str:
    """The code's name of the lane's test."""
    if indep_test not in _TESTS:
        raise ValueError(f"the lane runs {sorted(_TESTS)} on LDP and LD3, not {indep_test!r}")
    return _TESTS[indep_test]


def _frame(data: np.ndarray, indep_test: str) -> Any:
    """The columns as a frame named ``V0``, ``V1``, ... as the code's tests index them."""
    import pandas as pd

    X = np.asarray(data, dtype=np.float64) if indep_test == "fisherz" else np.asarray(data)
    return pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])


def _position(label: str) -> int:
    return int(label[1:])


def _ldp_class() -> Any:
    """The repository's ``LDP`` class, imported from the downloaded directory."""
    directory = str(_code_dir(LDP_ENV, "ldp"))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from ldp import LDP  # the repository's module of that name

    return LDP


def _ld3_class() -> Any:
    """The repository's ``LD3`` class, imported from the downloaded directory."""
    directory = str(_code_dir(LD3_ENV, "ld3"))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from ld3 import LD3  # the repository's module of that name

    return LD3


class _Maasch:
    """What the two solutions share: the environment, the settings, and the untimed members."""

    backend = "numpy"
    mode = "serial"
    output_type = "effects"
    identified_from = "dag"
    unordered = False
    #: Fisher's z test for continuous data and the chi-square test for categorical.
    indep_tests = ("fisherz", "chisq")
    python_env = "ANDREY_BENCH_MAASCH_PYTHON"

    def params(self) -> dict[str, Any]:
        """The lane's settings (:data:`DEFAULTS`), without the pair."""
        return dict(DEFAULTS)

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; the lane scores the records. Untimed."""
        return empty_structure(native)


class MaaschLDP(_Maasch):
    """The authors' LDP at one pair.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    name = "ldp.authors"
    package = "ldp"
    package_version = LDP_COMMIT[:12]
    algorithm = "ldp"
    estimand = "total"
    set_kind = "valid"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _ldp_class()

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code for the pair in ``params``; return the packed record and test count."""
        settings = {**self.params(), **params}
        test = _test_name(str(settings["indep_test"]))
        treatment, outcome = int(params[TREATMENT_PARAM]), int(params[OUTCOME_PARAM])
        ldp = _ldp_class()
        frame = _frame(data, str(settings["indep_test"]))
        run = ldp(data=frame, independence_test=test)
        result = run.partition_z(
            exposure=f"V{treatment}", outcome=f"V{outcome}", alpha=float(settings["alpha"])
        )
        valid = bool(run.vas_exists)
        record = {
            "treatment": treatment,
            "outcome": outcome,
            "estimand": self.estimand,
            "nonzero": None,
            "identifiable": valid,
            "adjustment_sets": [sorted({_position(v) for v in result["Z1"]})] if valid else [],
            "set_kind": self.set_kind if valid else None,
        }
        return pack_effects([record], frame.shape[1], run.total_tests)


class MaaschLD3(_Maasch):
    """The authors' LD3 at one pair.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    name = "ld3.authors"
    package = "ld3"
    package_version = LD3_COMMIT[:12]
    algorithm = "ld3"
    estimand = "direct"
    set_kind = "parents_of_outcome"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _ld3_class()

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code for the pair in ``params``; return the packed record and test count."""
        settings = {**self.params(), **params}
        test = _test_name(str(settings["indep_test"]))
        treatment, outcome = int(params[TREATMENT_PARAM]), int(params[OUTCOME_PARAM])
        ld3 = _ld3_class()
        frame = _frame(data, str(settings["indep_test"]))
        run = ld3(data=frame, independence_test=test)
        sdc, ade = run.get_sdc_cde_adjustment(
            exposure=f"V{treatment}", outcome=f"V{outcome}", alpha=float(settings["alpha"])
        )
        record = {
            "treatment": treatment,
            "outcome": outcome,
            "estimand": self.estimand,
            "nonzero": bool(sdc),
            "identifiable": None,
            "adjustment_sets": [sorted({_position(v) for v in ade})],
            "set_kind": self.set_kind,
        }
        return pack_effects([record], frame.shape[1], run.total_tests)
