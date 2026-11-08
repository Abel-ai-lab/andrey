"""Andrey's J-PCMCI+, and its PCMCI+ on the pooled system variables, for the J-PCMCI+ lane.

Both read one ``(n_datasets, n_steps, n_columns)`` array, the datasets of
:class:`~andrey_bench.context_models.ContextDraw`: the system variables, then the observed time
and space contexts, at the positions ``params["time_contexts"]`` and ``params["space_contexts"]``
name. Both return the window marks of :mod:`andrey_bench.window`.

``AndreyJPCMCIPlus`` calls ``andrey.jpcmciplus(list(data), time_contexts=..., space_contexts=...,
time_dummy=True, tau_max=..., alpha=..., indep_test=...)``; every other argument keeps its default.
Its marks cover every column: the links between system variables from the structure, and the links
from the contexts into them from ``metadata["context_links"]``, a tail at the context and an
arrowhead at the variable.

``AndreyPooledPCMCIPlus`` calls ``andrey.pcmciplus`` on the system variables of the datasets one
after another, pooled with no index of the dataset, the baseline of the J-PCMCI+ paper; its marks
cover the system variables.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pcmci import _andrey_version

#: The window marks of a link from a context: the tail at the context, the arrowhead at the variable.
_TAIL, _ARROW = 1, 2

_DEFAULTS = {"tau_max": 2, "alpha": 0.05, "indep_test": None}


class AndreyJPCMCIPlus:
    """Solution adapter for Andrey's J-PCMCI+ with both dummies.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.jpcmciplus.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "jpcmciplus"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its J-PCMCI+ engine before timing."""
        import andrey  # noqa: F401
        from andrey.temporal.jpcmciplus import jpcmciplus  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, the level, the test (``None``: Andrey's default), and no contexts."""
        return {**_DEFAULTS, "time_contexts": [], "space_contexts": []}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey J-PCMCI+; return the ``(tau_max + 1, n, n)`` marks over every column. Timed."""
        import warnings

        import andrey
        from andrey_bench.adapters.andrey_pcmci import pcmciplus_link_strings
        from andrey_bench.window import marks_from_link_strings

        settings = {**_DEFAULTS, **params}
        x = np.asarray(data, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.jpcmciplus(
                list(x),
                time_contexts=[int(c) for c in settings["time_contexts"]],
                space_contexts=[int(c) for c in settings["space_contexts"]],
                time_dummy=True,
                tau_max=int(settings["tau_max"]),
                alpha=float(settings["alpha"]),
                indep_test=settings["indep_test"],
            )
        system = marks_from_link_strings(pcmciplus_link_strings(out))
        n = system.shape[1]
        marks = np.zeros((system.shape[0], x.shape[2], x.shape[2]), dtype=np.int8)
        marks[:, :n, :n] = system
        for context, lag, variable in out.metadata["context_links"]:
            c, k, j = int(context), int(lag), int(variable)
            marks[k, c, j] = _TAIL
            if k == 0:
                marks[0, j, c] = _ARROW
        return marks

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> a ``WindowGraph`` whose lag 0 is a CPDAG; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="cpdag")


class AndreyPooledPCMCIPlus:
    """Solution adapter for Andrey's PCMCI+ on the pooled system variables.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.pcmciplus.pooled.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pcmciplus"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its PCMCI+ engine before timing."""
        import andrey  # noqa: F401
        from andrey.temporal.pcmciplus import pcmciplus  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, the level, the test (``None``: Andrey's default), and the system size."""
        return {**_DEFAULTS, "n_system": 0}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PCMCI+ on the pooled rows; return the system variables' marks. Timed."""
        import andrey
        from andrey_bench.adapters.andrey_pcmci import pcmciplus_link_strings
        from andrey_bench.window import marks_from_link_strings

        settings = {**_DEFAULTS, **params}
        n = int(settings["n_system"])
        x = np.asarray(data, dtype=np.float64)
        out = andrey.pcmciplus(
            x[:, :, :n].reshape(-1, n),
            tau_max=int(settings["tau_max"]),
            alpha=float(settings["alpha"]),
            indep_test=settings["indep_test"],
        )
        return marks_from_link_strings(pcmciplus_link_strings(out))

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> a ``WindowGraph`` whose lag 0 is a CPDAG; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="cpdag")
