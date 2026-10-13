"""Empty-graph baseline solution — the measured correctness floor.

This adapter runs the empty graph (no edges) on every dataset and is scored like any other
solution, so "better than nothing" is a measured record rather than an assumed number. A method
that recovers less than nothing has a worse record, in the same units, on the same dataset.

The floor is a non-trivial bar. The empty graph's SHD to a sparse ground truth is exactly the
truth's edge count ``m``, which on a sparse ER DGP grows only linearly in ``d`` while the number of
node pairs grows quadratically. Beating it therefore forces a solution to recover structure, not
merely to avoid catastrophic errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BaselineEmpty:
    """Empty-graph correctness-floor solution.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Its ``algorithm`` is ``baseline``,
    not the family it is read against: the empty graph implements no method, and filing it as one
    would pool the correctness floor into that method's field on any ``algorithm`` groupby.
    """

    name: str = "baseline.empty"
    package: str = "baseline"
    package_version: str = "1.0"  # a constant graph — versionless, pinned for provenance only
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "baseline"
    output_type: str = "cpdag"

    def params(self) -> dict[str, Any]:
        """The empty graph has nothing to configure."""
        return {}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the ``d×d`` all-zero (no-edge) 0/1 adjacency. The timed region — near-zero cost.

        ``d`` is the data's column count; ``params`` is ignored (the empty graph has no knobs).
        """
        d = int(np.asarray(data).shape[1])
        return np.zeros((d, d), dtype=np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """All-zero adjacency -> the empty Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


@dataclass(frozen=True)
class BaselineEmptyTemporal:
    """Empty lag-stack correctness floor for temporal solutions.

    The temporal counterpart of :class:`BaselineEmpty`: no link at any lag ``1..tau_max``, scored
    lag by lag like the solutions it bounds. Its lag-resolved SHD is the truth's link count within
    the window. ``tau_max`` is in the name because ``params`` are not part of a ``run_id``.
    """

    name: str = "baseline.empty.temporal.tau2"
    tau_max: int = 2
    package: str = "baseline"
    package_version: str = "1.0"
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "baseline"
    output_type: str = "temporal"

    def params(self) -> dict[str, Any]:
        """The lag window the empty stack covers."""
        return {"tau_max": self.tau_max}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the all-zero ``(tau_max, d, d)`` link stack. The timed region."""
        d = int(np.asarray(data).shape[1])
        return np.zeros((int(params["tau_max"]), d, d), dtype=np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """All-zero link stack -> the empty ``TemporalStructure``; untimed."""
        from andrey_bench.contracts import temporal_from_links

        return temporal_from_links(native)


def baseline_empty_temporal(tau_max: int = 2) -> BaselineEmptyTemporal:
    """A :class:`BaselineEmptyTemporal` named for its lag window."""
    return BaselineEmptyTemporal(name=f"baseline.empty.temporal.tau{tau_max}", tau_max=tau_max)


@dataclass(frozen=True)
class BaselineEmptyWindow:
    """Empty window correctness floor for PCMCI+ and LPCMCI.

    No link at any lag ``0..tau_max``, scored link by link like the solutions it bounds
    (:func:`andrey_bench.scoring.window_scores`): its endpoint SHD is twice the truth's link count
    within the window. ``tau_max`` is in the name because ``params`` are not part of a ``run_id``.
    """

    name: str = "baseline.empty.window.tau1"
    tau_max: int = 1
    package: str = "baseline"
    package_version: str = "1.0"
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "baseline"
    output_type: str = "temporal"

    def params(self) -> dict[str, Any]:
        """The lag window the empty graph covers."""
        return {"tau_max": self.tau_max}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the all-zero ``(tau_max + 1, d, d)`` window marks. The timed region."""
        d = int(np.asarray(data).shape[1])
        return np.zeros((int(params["tau_max"]) + 1, d, d), dtype=np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """All-zero window marks -> the empty ``WindowGraph``; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="dag")


def baseline_empty_window(tau_max: int = 1) -> BaselineEmptyWindow:
    """A :class:`BaselineEmptyWindow` named for its lag window."""
    return BaselineEmptyWindow(name=f"baseline.empty.window.tau{tau_max}", tau_max=tau_max)
