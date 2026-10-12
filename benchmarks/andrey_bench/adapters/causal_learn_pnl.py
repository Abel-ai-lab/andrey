"""causal-learn PNL adapter for cause-effect pairs.

Calls ``PNL().cause_or_effect(x, y)`` on the two columns, which returns the p-values of the two
directions and decides nothing. The adapter decides the edge with the function that decides Andrey's,
so both solutions apply the same rule to their p-values: with ``alpha=None``, the direction with the
larger p-value.

The two packages fit the same model by different means and test it differently. causal-learn trains
two small neural networks per direction with Adam, 3000 epochs in batches of 128 rows that drop the
last partial batch, so a pair of fewer than 128 rows trains nothing. It then compares the mean of
the assumed cause with the mean of the estimated noise by a two-sample t-test, rounded to three
decimals, rather than testing their independence. Andrey fits smooth basis expansions by Newton's
method and tests independence with HSIC.
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


class CausalLearnPNL:
    """Solution adapter for causal-learn's PNL.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.pnl"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "torch-cpu"
    mode: str = "serial"
    algorithm: str = "pnl"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import torch, causal-learn, and the shared decision before timing.

        See `contracts.SetupAdapter`.
        """
        import torch  # noqa: F401
        from causallearn.search.FCMBased.PNL.PNL import PNL  # noqa: F401

        from andrey.api._irregular import _adapt_pnl  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the decision rule applied to the p-values; causal-learn's defaults otherwise."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn PNL; return the two-node DAG's endpoint marks. The only timed region."""
        from causallearn.search.FCMBased.PNL.PNL import PNL

        from andrey.api._irregular import _adapt_pnl

        X = np.asarray(data, dtype=np.float64)
        forward, backward = PNL().cause_or_effect(X[:, [0]], X[:, [1]])
        alpha = {**_DEFAULTS, **params}["alpha"]
        return np.asarray(_adapt_pnl(forward, backward, alpha=alpha).structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
