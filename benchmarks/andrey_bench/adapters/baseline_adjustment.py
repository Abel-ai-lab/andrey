"""The two baselines of the adjustment lane: adjust for nothing, and adjust for everything.

Every paper of the group compares its sets with these: the unadjusted regression (the empty set),
which a method that finds a set has to beat, and the regression on every other variable (the
strategy Entner et al., 2013, Sect. 2, and Maasch et al., 2025, Remark 4, argue against), which
stands for a user who does not select covariates. Neither says whether the effect is zero or
identifiable, so each leaves both fields at "cannot tell" and the scorer judges the set alone: for
a total effect by the adjustment criterion in the true graph, for a direct effect against the
outcome's parents.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adjustment_lane import (
    ESTIMAND_PARAM,
    OUTCOME_PARAM,
    TREATMENT_PARAM,
    empty_structure,
    pack_effects,
)


class _Baseline:
    """A baseline answers either estimand, with the one set kind its estimand allows."""

    package = "baseline"
    package_version = "1.0"
    backend = "native"
    mode = "serial"
    algorithm = "baseline"
    output_type = "effects"
    estimand = "any"
    set_kind = "valid"
    identified_from = "dag"
    unordered = False

    def params(self) -> dict[str, Any]:
        """The baselines have nothing to configure."""
        return {}

    def _members(self, d: int, x: int, y: int) -> list[int]:
        raise NotImplementedError

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """The record of the pair with the baseline's set; the timed region."""
        d = int(np.asarray(data).shape[1])
        x, y = int(params[TREATMENT_PARAM]), int(params[OUTCOME_PARAM])
        estimand = params.get(ESTIMAND_PARAM, "total")
        record = {
            "treatment": x,
            "outcome": y,
            "estimand": estimand,
            "nonzero": None,
            "identifiable": None,
            "adjustment_sets": [self._members(d, x, y)],
            "set_kind": "valid" if estimand == "total" else "parents_of_outcome",
        }
        return pack_effects([record], d, None)

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; untimed."""
        return empty_structure(native)


class AdjustForNothing(_Baseline):
    """The unadjusted regression: the empty set."""

    name = "baseline.adjust-none"

    def _members(self, d: int, x: int, y: int) -> list[int]:
        return []


class AdjustForEverything(_Baseline):
    """The regression on every other column."""

    name = "baseline.adjust-all"

    def _members(self, d: int, x: int, y: int) -> list[int]:
        return [v for v in range(d) if v not in (x, y)]
