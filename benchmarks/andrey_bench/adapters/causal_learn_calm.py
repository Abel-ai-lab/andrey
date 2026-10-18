"""causal-learn CALM adapter: the CALM authors' implementation.

The CALM paper's first author contributed it to causal-learn. The paper's own code repository,
github.com/kaifeng-jin/CALM, runs the same optimizer, but returns only the weight matrix,
standardizes by default, and carries no license; causal-learn's copy is MIT-licensed and installs
from PyPI. Calls ``causallearn.search.ScoreBased.CALM.calm`` with its defaults, which are the
paper's settings. Its names map to Andrey's (``htol`` is ``h_tol``); the learning rate (0.001), the round
cap (100), and ``rho_max`` (1e16) are fixed inside it. It draws from torch's global generator, so
``fit`` seeds it first. Importing it sets torch's default dtype to float64 for the process; each
fit runs in its own process, so no other solution sees that.

The native array is the 0/1 support of the returned weight matrix, ``[i, j] = 1`` for ``i -> j``.
The method stops after 100 rounds whether or not the support is acyclic, and keeps every edge
whose gate passes 0.5; a returned cycle fails the conversion to a DAG and ends the fit in an error
record.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_fci import _causal_learn_version

#: causal-learn's defaults, in Andrey's names.
_DEFAULTS: dict[str, Any] = {
    "lambda1": 0.005,
    "alpha": 0.01,
    "tau": 0.5,
    "rho_init": 1e-5,
    "rho_mult": 3.0,
    "h_tol": 1e-8,
    "subproblem_iter": 40000,
    "standardize": False,
    "seed": 0,
}


class CausalLearnCALM:
    """Solution adapter for causal-learn's CALM."""

    name: str = "causal-learn.calm"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "torch-cpu"
    mode: str = "serial"
    algorithm: str = "calm"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import causal-learn's CALM and torch before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.ScoreBased.CALM import calm  # noqa: F401

    def params(self) -> dict[str, Any]:
        """causal-learn's defaults, in Andrey's names."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn CALM; return the 0/1 DAG adjacency. The only timed region."""
        import torch
        from causallearn.search.ScoreBased.CALM import calm as _calm

        settings = {**_DEFAULTS, **params}
        torch.manual_seed(int(settings["seed"]))
        record = _calm(
            np.asarray(data, dtype=np.float64),
            lambda1=float(settings["lambda1"]),
            alpha=float(settings["alpha"]),
            tau=float(settings["tau"]),
            rho_init=float(settings["rho_init"]),
            rho_mult=float(settings["rho_mult"]),
            htol=float(settings["h_tol"]),
            subproblem_iter=int(settings["subproblem_iter"]),
            standardize=bool(settings["standardize"]),
            device="cpu",
        )
        return (np.asarray(record["B_weighted"]) != 0).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
