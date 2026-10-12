"""Andrey PCMCI solution adapters, one per partial-correlation test.

Calls the public facade ``andrey.pcmci(X, tau_max=..., pc_alpha=0.2, alpha_level=0.05,
indep_test=...)``. ``pc_alpha`` and ``alpha_level`` are Andrey's defaults, which are also
tigramite's ``run_pcmci`` settings in :mod:`~andrey_bench.adapters.tigramite_pcmci`. Andrey tests
lags ``1..tau_max`` and no lag 0.

``indep_test`` makes two solutions, not one with a parameter: ``params`` are not part of a
``run_id``, so two tests under one name would share a part file and a resume would keep one of
them. ``tau_max`` is in every name for the same reason.

``fit`` returns the ``(tau_max, n, n)`` 0/1 link stack read from the result's
``TemporalStructure``; ``links[k - 1, i, j]`` is the link ``X_i(t-k) -> X_j(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: The two ``indep_test`` values ``andrey.pcmci`` accepts; ``student_t`` is its default.
INDEP_TESTS = ("student_t", "fisherz")

#: The largest lag tested unless a driver says otherwise.
DEFAULT_TAU_MAX = 2

#: Andrey's defaults, and the settings the tigramite adapter passes.
DEFAULT_PC_ALPHA = 0.2
DEFAULT_ALPHA_LEVEL = 0.05


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey-core")
    except metadata.PackageNotFoundError:  # a checkout run without its distribution metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


@dataclass(frozen=True)
class AndreyPCMCI:
    """Solution adapter for Andrey's PCMCI with one partial-correlation test.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Build it with
    :func:`andrey_pcmci_adapters`, which names each test and lag window.
    """

    name: str
    indep_test: str
    tau_max: int = DEFAULT_TAU_MAX
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pcmci"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its PCMCI engine before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.temporal.pcmci import pcmci  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, both significance levels, and the test."""
        return {
            "tau_max": self.tau_max,
            "pc_alpha": DEFAULT_PC_ALPHA,
            "alpha_level": DEFAULT_ALPHA_LEVEL,
            "indep_test": self.indep_test,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PCMCI and return the ``(tau_max, n, n)`` int8 link stack. The timed region."""
        import andrey
        from andrey_bench.contracts import links_from_temporal

        out = andrey.pcmci(
            np.asarray(data, dtype=np.float64),
            tau_max=int(params["tau_max"]),
            pc_alpha=float(params["pc_alpha"]),
            alpha_level=float(params["alpha_level"]),
            indep_test=str(params["indep_test"]),
        )
        return links_from_temporal(out.structure)

    def to_structure(self, native: np.ndarray) -> Any:
        """Link stack -> ``TemporalStructure`` with lags ``1..tau_max``; untimed."""
        from andrey_bench.contracts import temporal_from_links

        return temporal_from_links(native)


def andrey_pcmci_adapters(tau_max: int = DEFAULT_TAU_MAX) -> list[AndreyPCMCI]:
    """One :class:`AndreyPCMCI` per test in :data:`INDEP_TESTS`, named for the test and lag window."""
    return [
        AndreyPCMCI(name=f"andrey.pcmci.{test}.tau{tau_max}", indep_test=test, tau_max=tau_max)
        for test in INDEP_TESTS
    ]
