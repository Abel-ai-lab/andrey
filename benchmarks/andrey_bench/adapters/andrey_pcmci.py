"""Andrey PCMCI, PCMCI+, and LPCMCI solution adapters.

PCMCI calls the public facade ``andrey.pcmci(X, tau_max=..., pc1_alpha=0.2, alpha=0.05,
indep_test=...)``, one solution per partial-correlation test. ``pc1_alpha`` and ``alpha`` are
Andrey's defaults, which are also tigramite's ``run_pcmci`` settings in
:mod:`~andrey_bench.adapters.tigramite_pcmci`. Andrey's PCMCI tests lags ``1..tau_max`` and no lag 0.

PCMCI+ calls ``andrey.pcmciplus(X, tau_max=..., alpha=0.01, collider_rule="majority")`` and
LPCMCI ``andrey.lpcmci(X, tau_max=..., alpha=0.05, n_preliminary_iterations=1)``, both Andrey's
defaults and tigramite's settings, with the Student-t test of the authors' reference
implementation. They test lags
``0..tau_max``, and their ``fit`` returns the ``(tau_max + 1, n, n)`` window marks of
:mod:`andrey_bench.window`, read from the same three-character link strings tigramite returns, so
both packages' graphs are compared mark for mark.

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

#: Andrey's endpoint marks, as the structures' ``to_numpy`` returns them.
TAIL, ARROW = 1, 2

#: The two ``indep_test`` values ``andrey.pcmci`` accepts.
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
            "alpha": DEFAULT_ALPHA_LEVEL,
            "indep_test": self.indep_test,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PCMCI and return the ``(tau_max, n, n)`` int8 link stack. The timed region."""
        import andrey
        from andrey_bench.contracts import links_from_temporal

        out = andrey.pcmci(
            np.asarray(data, dtype=np.float64),
            tau_max=int(params["tau_max"]),
            pc1_alpha=float(params["pc_alpha"]),
            alpha=float(params["alpha"]),
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


#: Andrey's PCMCI+ defaults, which are also tigramite's ``run_pcmciplus`` defaults.
PCMCIPLUS_PC_ALPHA = 0.01
PCMCIPLUS_COLLIDER_RULE = "majority"

#: Andrey's LPCMCI defaults, which are also tigramite's ``run_lpcmci`` defaults.
LPCMCI_PC_ALPHA = 0.05
LPCMCI_PRELIMINARY_ITERATIONS = 1

#: The largest lag PCMCI+ and LPCMCI test unless a driver says otherwise: the true processes have
#: lag-1 links only.
SAME_STEP_TAU_MAX = 1


def pcmciplus_link_strings(out: Any) -> np.ndarray:
    """Return the ``(n, n, tau_max + 1)`` link strings of an ``andrey.pcmciplus`` result.

    The layout of tigramite's ``results["graph"]``: lag 0 holds ``-->`` and ``<--`` for an oriented
    link, ``o-o`` for an unoriented one, and ``x-x`` for a conflict, which the structure shows as
    undirected and ``metadata["conflicts"]`` lists; lags ``>= 1`` hold ``-->``.
    """
    structure = out.structure
    n, tau_max = structure.n_nodes, max(structure.lags)
    graph = np.full((n, n, tau_max + 1), "", dtype="<U3")
    marks = np.asarray(structure.lag(0).to_numpy())
    conflicts = {tuple(sorted(int(v) for v in pair)) for pair in out.metadata["conflicts"]}
    names = {(TAIL, ARROW): "-->", (ARROW, TAIL): "<--", (TAIL, TAIL): "o-o"}
    for i, j in zip(*np.nonzero(marks), strict=True):
        pair = (int(marks[i, j]), int(marks[j, i]))
        if pair not in names:
            raise RuntimeError(f"pcmciplus returned the lag-0 marks {pair} at ({i}, {j})")
        graph[i, j, 0] = "x-x" if (min(i, j), max(i, j)) in conflicts else names[pair]
    for k in range(1, tau_max + 1):
        # The arrowhead of `i -> j` is the mark at j, `marks[j, i]`, so the links are the transpose.
        graph[:, :, k][np.asarray(structure.lag(k).to_numpy()).T == ARROW] = "-->"
    return graph


def lpcmci_link_strings(out: Any) -> np.ndarray:
    """Return the ``(n, n, tau_max + 1)`` link strings of an ``andrey.lpcmci`` result.

    ``metadata["links"][k][i][j]`` is already a link string; this reorders it to tigramite's
    ``graph[i, j, k]``.
    """
    return np.asarray(out.metadata["links"], dtype="<U3").transpose(1, 2, 0)


@dataclass(frozen=True)
class AndreyPCMCIPlus:
    """Solution adapter for Andrey's PCMCI+ with the Student-t test.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Build it with
    :func:`andrey_pcmciplus_adapter`, which names the lag window.
    """

    name: str = f"andrey.pcmciplus.student_t.tau{SAME_STEP_TAU_MAX}"
    tau_max: int = SAME_STEP_TAU_MAX
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pcmciplus"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its PCMCI+ engine before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.temporal.pcmciplus import pcmciplus  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, the significance level, the collider rule, and the test."""
        return {
            "tau_max": self.tau_max,
            "alpha": PCMCIPLUS_PC_ALPHA,
            "collider_rule": PCMCIPLUS_COLLIDER_RULE,
            "indep_test": "student_t",
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PCMCI+ and return its ``(tau_max + 1, n, n)`` window marks. Timed."""
        import andrey
        from andrey_bench.window import marks_from_link_strings

        out = andrey.pcmciplus(
            np.asarray(data, dtype=np.float64),
            tau_max=int(params["tau_max"]),
            alpha=float(params["alpha"]),
            collider_rule=str(params["collider_rule"]),
            indep_test=str(params["indep_test"]),
        )
        return marks_from_link_strings(pcmciplus_link_strings(out))

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> a ``WindowGraph`` whose lag 0 is a CPDAG; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="cpdag")


@dataclass(frozen=True)
class AndreyLPCMCI:
    """Solution adapter for Andrey's LPCMCI with the Student-t test.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Build it with
    :func:`andrey_lpcmci_adapter`, which names the lag window.
    """

    name: str = f"andrey.lpcmci.student_t.tau{SAME_STEP_TAU_MAX}"
    tau_max: int = SAME_STEP_TAU_MAX
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "lpcmci"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its LPCMCI engine before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.temporal.lpcmci import lpcmci  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, the significance level, the preliminary rounds, and the test."""
        return {
            "tau_max": self.tau_max,
            "alpha": LPCMCI_PC_ALPHA,
            "n_preliminary_iterations": LPCMCI_PRELIMINARY_ITERATIONS,
            "indep_test": "student_t",
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey LPCMCI and return its ``(tau_max + 1, n, n)`` window marks. Timed."""
        import andrey
        from andrey_bench.window import marks_from_link_strings

        out = andrey.lpcmci(
            np.asarray(data, dtype=np.float64),
            tau_max=int(params["tau_max"]),
            alpha=float(params["alpha"]),
            n_preliminary_iterations=int(params["n_preliminary_iterations"]),
            indep_test=str(params["indep_test"]),
        )
        return marks_from_link_strings(lpcmci_link_strings(out))

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> a ``WindowGraph`` whose lag 0 is a PAG; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="pag")


def andrey_pcmciplus_adapter(tau_max: int = SAME_STEP_TAU_MAX) -> AndreyPCMCIPlus:
    """An :class:`AndreyPCMCIPlus` named for its lag window."""
    return AndreyPCMCIPlus(name=f"andrey.pcmciplus.student_t.tau{tau_max}", tau_max=tau_max)


def andrey_lpcmci_adapter(tau_max: int = SAME_STEP_TAU_MAX) -> AndreyLPCMCI:
    """An :class:`AndreyLPCMCI` named for its lag window."""
    return AndreyLPCMCI(name=f"andrey.lpcmci.student_t.tau{tau_max}", tau_max=tau_max)
