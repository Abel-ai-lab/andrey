"""Benchmark Andrey CALM under two optimizer schedules.

``andrey.calm.torch-cpu`` runs Andrey's defaults. ``andrey.calm.torch-cpu.paper_schedule`` runs
the schedule of the CALM paper, which causal-learn's ``calm`` uses as its defaults: 40,000 Adam
steps per round at learning rate 0.001, ``rho`` from 1e-5, up to 100 rounds, and no weight
threshold. Unlike causal-learn's, Andrey's rounds can end before their last step once the loss
stops changing. Both use the paper's penalty, gate temperature, and moral-graph test level, and fix
the optimizer seed at 0. Fits return the DAG's endpoint marks.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_fci import _andrey_version

#: Andrey's defaults, named in full so the record says what ran.
DEFAULTS: dict[str, Any] = {
    "lambda1": 0.005,
    "alpha": 0.01,
    "tau": 0.5,
    "rho_init": 1e-3,
    "rho_mult": 3.0,
    "rho_max": 1e16,
    "h_tol": 1e-8,
    "max_outer": 25,
    "subproblem_iter": 3000,
    "lr": 0.01,
    "w_threshold": 0.3,
    "gate_threshold": 0.5,
    "standardize": False,
    "seed": 0,
}

#: The paper's schedule in Andrey's names. causal-learn's ``calm`` fixes the learning rate, the
#: round cap, and ``rho_max`` inside; its gate threshold is 0.5 and it has no weight threshold.
PAPER_SCHEDULE: dict[str, Any] = {
    **DEFAULTS,
    "rho_init": 1e-5,
    "max_outer": 100,
    "subproblem_iter": 40000,
    "lr": 0.001,
    "w_threshold": 0.0,
}

_SCHEDULES = {"default": DEFAULTS, "paper": PAPER_SCHEDULE}


class AndreyCALM:
    """CALM solution adapter, one solution name per schedule."""

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "torch-cpu"
    mode: str = "serial"
    algorithm: str = "calm"
    output_type: str = "dag"

    def __init__(self, schedule: str = "default") -> None:
        if schedule not in _SCHEDULES:
            raise ValueError(f"schedule must be one of {sorted(_SCHEDULES)}, got {schedule!r}")
        self.schedule = schedule
        self.name = "andrey.calm.torch-cpu" + (".paper_schedule" if schedule == "paper" else "")

    def setup(self) -> None:
        """Import Andrey, torch, and the CALM engine before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.search.calm import calm  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the schedule's settings in Andrey's names."""
        return dict(_SCHEDULES[self.schedule])

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey CALM; return the DAG's endpoint-mark array. The only timed region."""
        import andrey

        settings = {**_SCHEDULES[self.schedule], **params}
        seed = settings.pop("seed")
        out = andrey.calm(np.asarray(data, dtype=np.float64), seed=seed, **settings)
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
