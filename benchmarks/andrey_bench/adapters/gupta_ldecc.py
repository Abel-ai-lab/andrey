"""LDECC adapter for the authors' code (Gupta, Childers and Lipton, 2023), run unmodified in its own
environment.

The code is ``acmi-lab/local-causal-discovery`` at commit :data:`COMMIT`, a repository with no
license file. It is a measurement reference only: ``benchmarks/start_ldecc.sh`` downloads the
archive of that commit, checks its SHA-256, and builds the environment; the benchmark imports the
code from the directory ``$ANDREY_BENCH_LDECC`` names and copies none of it.

A fit calls ``LDECCAlgorithm(use_ci_oracle=False, alpha=alpha, ldecc_do_checks=True).run(df)`` as
the repository's notebooks do for the variant they label ``ldecc-checks``, on a frame whose columns
are named ``V0`` to ``V{d-1}``. The test is Fisher's z test, the only one the code has, and
``alpha`` is passed by the lane (the class default is 0.05; the notebooks set
``0.01 * 32000 / n``, which the lane does not copy). The code never reads the outcome: its answer
is the local structure around the treatment.

The answer is read as the notebooks read it: the code returns the treatment's parents, children,
and unoriented neighbors, and the non-collider pairs among them, and the locally valid parent sets
are the parents plus each subset of the unoriented neighbors that holds no non-collider pair
(``causal_discovery.utils.get_all_combinations``). The record is one per pair with those sets, the
outcome removed from each and repeated sets merged; the effect is identified when exactly one set
remains. The code does not say whether the effect is zero, so ``nonzero`` is ``None``. The number of
tests is the code's own ``ci_test_calls["total"]``, which counts every call of its test wrapper.
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

#: Names the directory holding the extracted code.
CODE_ENV = "ANDREY_BENCH_LDECC"

#: The pinned commit of ``acmi-lab/local-causal-discovery``; ``start_ldecc.sh`` fetches it.
COMMIT = "605d8d1e82d0cd63dce717a5618baa0b5fa153fb"

#: The files a fit imports, relative to the code directory. The package has no ``__init__.py``.
REQUIRED_FILES = (
    "causal_discovery/ldecc.py",
    "causal_discovery/utils.py",
    "causal_discovery/causal_discovery_base.py",
)

#: The lane's settings: the code's default level and its only test.
DEFAULTS = {"alpha": 0.05, "indep_test": "fisherz"}


def _code_dir() -> Path:
    raw = os.environ.get(CODE_ENV, "").strip()
    directory = Path(raw)
    if not raw or not all((directory / name).is_file() for name in REQUIRED_FILES):
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} does not hold the LDECC code; run benchmarks/start_ldecc.sh"
        )
    return directory


def _reference() -> tuple[Any, Any]:
    """The repository's ``LDECCAlgorithm`` and ``get_all_combinations``, imported from the
    downloaded directory."""
    directory = str(_code_dir())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from causal_discovery.ldecc import LDECCAlgorithm  # the repository's package of that name
    from causal_discovery.utils import get_all_combinations

    return LDECCAlgorithm, get_all_combinations


def _position(label: str) -> int:
    return int(label[1:])


class GuptaLDECC:
    """The authors' LDECC at one treatment, with the "checks" variant and Fisher's z test.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    name = "ldecc.authors"
    package = "ldecc"
    package_version = COMMIT[:12]
    backend = "numpy"
    mode = "serial"
    algorithm = "ldecc"
    output_type = "effects"
    estimand = "total"
    set_kind = "local_parents"
    identified_from = "cpdag"
    unordered = False
    #: LDECC runs Fisher's z test only.
    indep_tests = ("fisherz",)
    python_env = "ANDREY_BENCH_LDECC_PYTHON"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _reference()

    def params(self) -> dict[str, Any]:
        """The lane's settings (:data:`DEFAULTS`), without the pair."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code for the pair in ``params``; return the packed record and test count."""
        settings = {**self.params(), **params}
        if settings["indep_test"] != "fisherz":
            raise ValueError(
                f"LDECC's code has Fisher's z test only, not {settings['indep_test']!r}"
            )
        X = np.asarray(data, dtype=np.float64)
        d = X.shape[1]
        treatment, outcome = int(params[TREATMENT_PARAM]), int(params[OUTCOME_PARAM])
        algorithm, all_combinations = _reference()
        import pandas as pd

        run = algorithm(
            treatment_node=f"V{treatment}",
            outcome_node=f"V{outcome}",
            alpha=float(settings["alpha"]),
            use_ci_oracle=False,
            ldecc_do_checks=True,
        )
        result = run.run(pd.DataFrame(X, columns=[f"V{i}" for i in range(d)]))
        parents = {_position(v) for v in result["tmt_parents"]}
        sets: list[list[int]] = []
        for extra in all_combinations(result["unoriented"], result["non_colliders"]):
            members = sorted((parents | {_position(v) for v in extra}) - {outcome})
            if members not in sets:
                sets.append(members)
        sets.sort()
        record = {
            "treatment": treatment,
            "outcome": outcome,
            "estimand": self.estimand,
            "nonzero": None,
            "identifiable": len(sets) == 1,
            "adjustment_sets": sets,
            "set_kind": self.set_kind,
        }
        return pack_effects([record], d, run.ci_test_calls["total"])

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; the lane scores the records. Untimed."""
        return empty_structure(native)
