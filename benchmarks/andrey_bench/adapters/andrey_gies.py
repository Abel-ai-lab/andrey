"""Andrey GIES adapter: greedy interventional equivalence search on interventional data.

A fit needs the samples and which variables were intervened on in each row. The harness passes a
fit one matrix, so the GIES lane hands each fit the samples and the 0/1 intervention matrix side by
side: :func:`stack` builds that matrix and :func:`unstack` splits it again inside `fit`.

pcalg's `gies` is the counterpart. Both score BIC with a penalty of `log(n) / 2` per parameter in
log-likelihood units, Andrey's `penalty_discount=1.0` in deviance units, with `n` every row of the
data, and both run the forward, backward, and turning phases until none improves, with no degree
cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_hc import _andrey_version
from andrey_bench.adapters.bic_penalty import ANDREY_EBIC_GAMMA

_DEFAULT_LAMBDA = 1.0


def stack(samples: np.ndarray, interventions: np.ndarray) -> np.ndarray:
    """The ``(n, 2 d)`` matrix a GIES fit takes: the samples, then the 0/1 interventions."""
    return np.hstack([samples, np.asarray(interventions, dtype=samples.dtype)])


def unstack(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a :func:`stack` matrix into the ``(n, d)`` samples and boolean interventions."""
    d = data.shape[1] // 2
    if data.shape[1] != 2 * d:
        raise ValueError(
            f"a GIES fit takes samples and interventions side by side, got {data.shape}"
        )
    return np.asarray(data[:, :d], dtype=np.float64), np.asarray(data[:, d:]) != 0


@dataclass(frozen=True)
class AndreyGIES:
    """Benchmark adapter for Andrey GIES."""

    name: str = "andrey.gies.numpy"
    backend: str = "numpy"
    mode: str = "serial"
    package: str = "andrey"
    package_version: str = _andrey_version()
    algorithm: str = "gies"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey's GIES before timing, outside `fit`."""
        from andrey.search.gies import gies  # noqa: F401

    def params(self) -> dict[str, Any]:
        """BIC in deviance units."""
        return {"lambda_value": _DEFAULT_LAMBDA, "ebic_gamma": ANDREY_EBIC_GAMMA}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run GIES and return endpoint marks of the essential graph. The only timed region.

        Calls the search directly, so the facade's input checks stay outside the timing.
        """
        import andrey
        from andrey.search.gies import gies

        X, mask = unstack(data)
        with andrey.config(backend=self.backend):
            graph, _score = gies(
                X,
                mask,
                penalty_discount=float(params.get("lambda_value", _DEFAULT_LAMBDA)),
                ebic_gamma=float(params.get("ebic_gamma", ANDREY_EBIC_GAMMA)),
            )
        return np.asarray(graph.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
