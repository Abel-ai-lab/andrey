"""causal-learn ANM adapter for cause-effect pairs.

Calls ``ANM().cause_or_effect(x, y)`` on the two columns, which returns the p-values of the two
directions and decides nothing. The adapter decides the edge with the function that decides Andrey's,
so both solutions apply the same rule to their p-values: with ``alpha=None``, the direction with the
larger p-value. causal-learn reports no test statistic, so equal p-values give no edge, where Andrey
takes the direction with the smaller HSIC statistic relative to its mean under independence.

causal-learn regresses with the same Gaussian process and tests with the same HSIC as Andrey, but on
the data as given; Andrey standardizes each column first. On standardized data the two agree.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS: dict[str, Any] = {"alpha": None}


def _causal_learn_version() -> str:
    """Return the installed causal-learn distribution version."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnANM:
    """Solution adapter for causal-learn's ANM.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.anm"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "anm"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import causal-learn and the shared decision before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.FCMBased.ANM.ANM import ANM  # noqa: F401

        from andrey.api._irregular import _adapt_anm  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the decision rule applied to the p-values; causal-learn takes no setting."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn ANM; return the two-node DAG's endpoint marks. The only timed region."""
        from causallearn.search.FCMBased.ANM.ANM import ANM

        from andrey.api._irregular import _adapt_anm

        X = np.asarray(data, dtype=np.float64)
        forward, backward = ANM().cause_or_effect(X[:, [0]], X[:, [1]])
        alpha = {**_DEFAULTS, **params}["alpha"]
        return np.asarray(_adapt_anm(forward, backward, alpha=alpha).structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
