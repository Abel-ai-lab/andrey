"""causal-learn CD-NOD adapter for data pooled from several domains.

Calls ``cdnod(X, c_indx, alpha=0.05, indep_test="fisherz", stable=True, uc_rule=0,
uc_priority=2, show_progress=False)``. Those are causal-learn's defaults, and the settings of
Andrey's CD-NOD: a stable skeleton search (removals take effect after each conditioning size), a
collider wherever the middle node is in no separating set, and a collider that would reverse an
edge already directed is skipped. causal-learn returns a graph over the data and the domain index,
its last node; the index is dropped before the CPDAG is read. ``fit`` rebuilds the index from
``n_domains`` as :mod:`~andrey_bench.adapters.andrey_cdnod` does.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency
from andrey_bench.regime_models import N_DOMAINS, domain_index

_DEFAULTS = {
    "alpha": 0.05,
    "indep_test": "fisherz",
    "stable": True,
    "uc_rule": 0,
    "uc_priority": 2,
    "n_domains": N_DOMAINS,
}


def _causal_learn_version() -> str:
    """The installed causal-learn version (distribution metadata; the module exposes none)."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnCDNOD:
    """Solution adapter for causal-learn's CD-NOD.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.cdnod"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "cdnod"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.ConstraintBased.CDNOD import cdnod  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the settings passed to ``cdnod`` and the number of domains in the rows."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn CD-NOD; return the 0/1 CPDAG adjacency of the data's variables. Timed."""
        from causallearn.search.ConstraintBased.CDNOD import cdnod

        settings = {**_DEFAULTS, **params}
        X = np.asarray(data, dtype=np.float64)
        cg = cdnod(
            X,
            domain_index(len(X), int(settings["n_domains"])),
            alpha=float(settings["alpha"]),
            indep_test=settings["indep_test"],
            stable=bool(settings["stable"]),
            uc_rule=int(settings["uc_rule"]),
            uc_priority=int(settings["uc_priority"]),
            show_progress=False,
        )
        d = X.shape[1]
        return generalgraph_to_adjacency(np.asarray(cg.G.graph)[:d, :d])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
