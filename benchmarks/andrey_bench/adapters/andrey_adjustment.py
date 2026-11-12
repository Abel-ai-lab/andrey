"""Andrey's treatment-outcome methods for the adjustment lane, and the PC arm every paper reports.

:class:`AndreyAdjustment` runs ``andrey.<method>`` on a pair and reads the answer from
``metadata["effects"]``. The seven methods of the group are rows of :data:`METHODS`: what the method
answers (the estimand, the kind of set), how its effect is identified (from the Markov equivalence
class or from observed sets of the true graph), whether it is told which variable is the treatment,
and the level of its tests. A method is added to the lane by naming its row; :func:`available` lists
the rows the installed Andrey has, and ``adjustment_cpu.py`` takes them. Every setting is passed to
the method, never defaulted: ``alpha``, ``indep_test`` and, when given, ``max_depth``.

:class:`AndreyPCAdjustment` is the global baseline of every paper in the group: PC's CPDAG read at
the pair, the optimal adjustment set where the class identifies the effect and the local parent
sets of the treatment (Maathuis et al., 2009) where it does not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version
from andrey_bench.adjustment_lane import (
    OPTIMAL_OR_LOCAL,
    OUTCOME_PARAM,
    TREATMENT_PARAM,
    VARIABLES_PARAM,
    empty_structure,
    pack_effects,
)


@dataclass(frozen=True)
class Method:
    """What the lane knows of one method: its answer and its settings."""

    estimand: str
    set_kind: str
    identified_from: str
    unordered: bool
    #: The level of the method's tests: the paper's or the authors' value (DESIGN.md section 1).
    alpha: float


#: The seven methods of the group, by the name of their function in ``andrey``.
METHODS: dict[str, Method] = {
    "covariate_rules": Method("total", "valid", "dag", False, 0.05),
    "ldecc": Method("total", "local_parents", "cpdag", False, 0.05),
    "ldp": Method("total", "valid", "dag", False, 0.005),
    "ld3": Method("direct", "parents_of_outcome", "dag", False, 0.005),
    "snap": Method("total", OPTIMAL_OR_LOCAL, "cpdag", True, 0.05),
    "local_optimal_adjustment": Method("total", OPTIMAL_OR_LOCAL, "cpdag", True, 0.01),
    "locpc_cde": Method("direct", "parents_of_outcome", "cpdag", False, 0.05),
}

#: The test of a Gaussian cell and its default level when the cell does not set one.
DEFAULT_INDEP_TEST = "fisherz"


def available() -> tuple[str, ...]:
    """The rows of :data:`METHODS` the installed Andrey has."""
    import andrey

    return tuple(name for name in METHODS if callable(getattr(andrey, name, None)))


class AndreyAdjustment:
    """``andrey.<method>`` on a treatment-outcome pair.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    package = "andrey"
    package_version = _andrey_version()
    backend = "numpy"
    mode = "serial"
    output_type = "effects"

    def __init__(self, method: str, *, indep_test: str = DEFAULT_INDEP_TEST) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; the methods are {sorted(METHODS)}")
        row = METHODS[method]
        self.method, self.indep_test = method, indep_test
        self.algorithm = method
        self.name = f"andrey.{method}"
        self.estimand, self.set_kind = row.estimand, row.set_kind
        self.identified_from, self.unordered = row.identified_from, row.unordered

    def setup(self) -> None:
        """Import Andrey and the method before timing, so no fit pays for them."""
        import andrey

        if not callable(getattr(andrey, self.method, None)):
            raise RuntimeError(f"the installed Andrey has no {self.method!r}")

    def params(self) -> dict[str, Any]:
        """The settings passed to every fit: the method's level and the test."""
        return {"alpha": METHODS[self.method].alpha, "indep_test": self.indep_test}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the method on ``data`` for the pair in ``params``; return the packed records and
        the test count. The only timed region."""
        import andrey

        data = np.asarray(data, dtype=np.float64)
        settings = {**self.params(), **params}
        pair = (
            {"variables": [int(v) for v in params[VARIABLES_PARAM]]}
            if self.unordered
            else {
                "treatment": int(params[TREATMENT_PARAM]),
                "outcome": int(params[OUTCOME_PARAM]),
            }
        )
        extra = {} if settings.get("max_depth") is None else {"max_depth": settings["max_depth"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = getattr(andrey, self.method)(
                data,
                **pair,
                alpha=float(settings["alpha"]),
                indep_test=str(settings["indep_test"]),
                **extra,
            )
        return pack_effects(out.metadata["effects"], data.shape[1], out.metadata.get("n_tests"))

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; the lane scores the records. Untimed."""
        return empty_structure(native)


class AndreyPCAdjustment:
    """PC's CPDAG read at the pair, as the global baseline of the group's papers.

    Where the CPDAG identifies the total effect (:func:`andrey.core.adjustment.adjustment_set`
    finds a set) the answer is the optimal adjustment set, and the effect is zero when there is
    none; where it does not, the answer is the local parent sets of the treatment with the outcome
    left out. A CPDAG the adjustment core refuses (PC can return a graph that is no CPDAG of any
    class) is an answer that decides nothing.
    """

    name = "andrey.pc.adjustment"
    package = "andrey"
    package_version = _andrey_version()
    backend = "numpy"
    mode = "serial"
    algorithm = "pc"
    output_type = "effects"
    estimand = "total"
    set_kind = OPTIMAL_OR_LOCAL
    identified_from = "cpdag"
    unordered = False

    def __init__(self, *, indep_test: str = DEFAULT_INDEP_TEST) -> None:
        self.indep_test = indep_test

    def setup(self) -> None:
        """Import Andrey and the adjustment core before timing."""
        import andrey  # noqa: F401
        from andrey.core import adjustment  # noqa: F401

    def params(self) -> dict[str, Any]:
        """PC's own defaults, passed."""
        return {"alpha": 0.05, "indep_test": self.indep_test}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run PC and read the pair off its CPDAG; the timed region."""
        import andrey
        from andrey.core import adjustment as adj

        data = np.asarray(data, dtype=np.float64)
        d = data.shape[1]
        settings = {**self.params(), **params}
        x, y = int(params[TREATMENT_PARAM]), int(params[OUTCOME_PARAM])
        out = andrey.pc(
            data, alpha=float(settings["alpha"]), indep_test=str(settings["indep_test"])
        )
        record: dict[str, Any] = {
            "treatment": x,
            "outcome": y,
            "estimand": "total",
            "nonzero": None,
            "identifiable": None,
            "adjustment_sets": [],
            "set_kind": None,
        }
        graph = out.structure
        try:
            adj.checked(graph, "the PC arm", ("cpdag",))
            refused = False
        except ValueError:
            refused = True  # not the CPDAG of a class: the answer decides nothing
        if not refused:
            if adj.adjustment_set(graph, x, y) is not None:
                optimal = adj.optimal_set(graph, x, y)
                record["identifiable"] = True
                record["nonzero"] = optimal is not None
                if optimal is not None:
                    record.update(adjustment_sets=[sorted(optimal)], set_kind="optimal")
            else:
                sets = {
                    tuple(sorted(v for v in s if v != y)) for s in adj.local_parent_sets(graph, x)
                }
                record.update(
                    identifiable=False,
                    adjustment_sets=[list(s) for s in sorted(sets)],
                    set_kind="local_parents",
                )
        return pack_effects([record], d, out.metadata.get("n_tests"))

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; untimed."""
        return empty_structure(native)
