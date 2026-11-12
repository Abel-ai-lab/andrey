"""LOAD adapter for the authors' code (Schubert, Claassen, and Magliacane, 2026), run unmodified.

The code is ``Matyasch/load`` (main) at commit :data:`COMMIT`, under the BSD 2-clause license, a
NumPy-only package with a ``pyproject.toml``. It is a measurement reference only:
``benchmarks/start_schubert.sh`` downloads the archive of the commit, checks its SHA-256, and
builds the environment; the benchmark imports the package from the ``src`` directory of
``$ANDREY_BENCH_LOAD`` and copies none of it. The branch of the paper's experiments is not the
reference.

A fit calls ``load(nodes=, ci_test=, alpha=, targets=)`` as the repository's README does, with the
two columns of the pair as the targets and Fisher's z test on the data's correlation matrix
(:class:`~andrey_bench.adapters.schubert_fisherz.FisherZ`) as ``ci_test``. The code returns
``{"adj_sets": {(treatment, outcome): [set, ...]}, "identifiable": bool}``: one flag for the pair
and the adjustment sets of the directions it did not find to be zero.

The records name both directions of the pair, in column order, one per ordered pair:

- a direction with sets and ``identifiable`` true: the optimal adjustment set (the code's
  Lemma 4.4 step), the effect nonzero (the treatment was found an explicit ancestor);
- a direction with sets and ``identifiable`` false: the local parent sets of the treatment, the
  outcome left out, the effect undecided: the output does not say whether the treatment was found
  an explicit or only a possible ancestor, and the second does not make the effect nonzero;
- a direction without sets: the code's "identified as zero" (its README), a definite
  non-ancestor: the effect is zero, identified, and has no set.

The count of tests is that of the Fisher-z callable: the distinct queries the code evaluated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.schubert_fisherz import FisherZ, check_settings
from andrey_bench.adjustment_lane import (
    OPTIMAL_OR_LOCAL,
    VARIABLES_PARAM,
    empty_structure,
    pack_effects,
)

#: Names the directory holding the downloaded code.
CODE_ENV = "ANDREY_BENCH_LOAD"

#: The pinned commit of ``Matyasch/load``; ``start_schubert.sh`` fetches it.
COMMIT = "7febfc68d477b82b07e52aa50e454339f6e0ebcd"

#: The files a fit imports, relative to the code directory.
REQUIRED_FILES = ("src/load/__init__.py", "src/load/load_alg.py", "src/load/mb_by_mb.py")

#: The authors' settings: the level of the paper's experiments.
DEFAULTS: dict[str, Any] = {"alpha": 0.01, "indep_test": "fisherz"}


def _code_dir() -> Path:
    raw = os.environ.get(CODE_ENV, "").strip()
    directory = Path(raw)
    if not raw or not all((directory / name).is_file() for name in REQUIRED_FILES):
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} does not hold the LOAD files; run benchmarks/start_schubert.sh"
        )
    return directory


def _load() -> Any:
    """The repository's ``load`` function, imported from the downloaded ``src`` directory."""
    source = str(_code_dir() / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import load  # the repository's package of that name

    return load.load


def effect_records(answer: Mapping[str, Any], pair: tuple[int, int]) -> list[dict[str, Any]]:
    """The ``effects`` records of both directions of ``pair`` from what the code returned."""
    sets_of = answer["adj_sets"]
    identifiable = bool(answer["identifiable"])
    i, j = pair
    records = []
    for x, y in ((i, j), (j, i)):
        record: dict[str, Any] = {
            "treatment": x,
            "outcome": y,
            "estimand": "total",
            "nonzero": False,
            "identifiable": True,
            "adjustment_sets": [],
            "set_kind": None,
        }
        if (x, y) in sets_of:
            members = {tuple(sorted(int(v) for v in s if v not in (x, y))) for s in sets_of[x, y]}
            record.update(
                nonzero=True if identifiable else None,
                identifiable=identifiable,
                adjustment_sets=[list(s) for s in sorted(members)],
                set_kind="optimal" if identifiable else "local_parents",
            )
        records.append(record)
    return records


class SchubertLOAD:
    """The authors' LOAD at one pair of targets.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    name: str = "load.authors"
    package: str = "load"
    package_version: str = COMMIT[:12]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "local_optimal_adjustment"
    output_type: str = "effects"
    estimand: str = "total"
    set_kind: str = OPTIMAL_OR_LOCAL
    identified_from: str = "cpdag"
    unordered: bool = True
    #: The code takes a callable test; the lane gives it Fisher's z.
    indep_tests = ("fisherz",)
    python_env: str = "ANDREY_BENCH_SCHUBERT_PYTHON"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _load()

    def params(self) -> dict[str, Any]:
        """The authors' settings (:data:`DEFAULTS`)."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code with the pair in ``params["variables"]`` as its targets; return the
        packed records and the number of tests. The only timed region."""
        X = np.asarray(data, dtype=np.float64)
        settings = {**DEFAULTS, **params}
        check_settings(settings, "LOAD")
        i, j = (int(v) for v in params[VARIABLES_PARAM])
        run = _load()
        test = FisherZ(X)
        answer = run(
            nodes=set(range(X.shape[1])),
            ci_test=test,
            alpha=float(settings["alpha"]),
            targets=[i, j],
        )
        return pack_effects(effect_records(answer, (i, j)), X.shape[1], test.n_tests)

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; the lane scores the records. Untimed."""
        return empty_structure(native)
