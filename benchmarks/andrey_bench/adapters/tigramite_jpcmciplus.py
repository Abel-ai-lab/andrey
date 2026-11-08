"""tigramite's JPCMCIplus, the reference of the J-PCMCI+ lane, with a fix applied at import.

Calls tigramite's public API only: ``JPCMCIplus(dataframe=DataFrame(blocks,
analysis_mode="multiple", vector_vars=...), cond_ind_test=ParCorrMult(),
node_classification=..., verbosity=0).run_jpcmciplus(tau_max=..., pc_alpha=...)``, every other
argument at its default; ``params["test"] = "gpdc"`` takes ``GPDC()`` instead. Each dataset's
block holds its columns, then the one-hot dummy of the time step (a column per step) and that of
the dataset (a column per dataset), each one vector variable, as the J-PCMCI+ authors' experiment
code builds them. The data are the ``(n_datasets, n_steps, n_columns)`` array of
:class:`~andrey_bench.context_models.ContextDraw`; ``fit`` returns the window marks
(:mod:`andrey_bench.window`) over its columns, the dummies dropped.

**The fix.** In tigramite 5.2.10.1, ``add_found_context_link_assumptions`` gives both ends of a
same-step link between two observed contexts the assumption ``"-->"``, which ``_set_link_assumptions``
rejects, so the fit raises ``ValueError("Inconsistent link assumptions ...")``. :data:`PATCH` leaves
that link ``"o?o"``. With ``params["patch"]`` true, the default, ``fit`` applies it to the module's
source in memory (:func:`apply_patch`) and runs the patched class; the installed package does not
change. ``patch=False`` runs the stock class, so the lane can check that the fix changes no fit that
finishes without it.

**It runs in an environment of its own** (``benchmarks/start_tigramite.sh``), as the PCMCI lane's
tigramite adapters do: this module imports nothing but numpy and the standard library at import
time, and only ``fit`` runs there.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.tigramite_pcmci import PYTHON_ENV_VAR, _tigramite_version

#: The fix, a unified diff against tigramite 5.2.10.1's ``tigramite/jpcmciplus.py``.
PATCH = Path(__file__).resolve().parents[2] / "patches" / "tigramite-jpcmciplus-context-links.patch"

#: The tests ``params["test"]`` names, each at its defaults.
TESTS = ("parcorr_mult", "gpdc")


def patch_id() -> str:
    """The patch's file name and the first 12 hex digits of its SHA-1, for the records."""
    return f"{PATCH.name}@{hashlib.sha1(PATCH.read_bytes()).hexdigest()[:12]}"


def apply_patch(source: str, patch: str) -> str:
    """Apply the hunks of a one-file unified diff to `source`.

    Raises ``RuntimeError`` unless the patch has a hunk and each hunk's old lines occur in `source`
    exactly once, so a changed tigramite fails the fit instead of running unpatched. An empty line
    in a hunk is an empty context line.
    """
    hunks = re.split(r"^@@ [^\n]* @@\n", patch, flags=re.MULTILINE)[1:]
    if not hunks:
        raise RuntimeError("the patch has no hunk")
    for hunk in hunks:
        lines = [" \n" if line == "\n" else line for line in hunk.splitlines(keepends=True)]
        old = "".join(line[1:] for line in lines if line[0] in " -")
        new = "".join(line[1:] for line in lines if line[0] in " +")
        if source.count(old) != 1:
            raise RuntimeError(f"a hunk of {PATCH.name} matches {source.count(old)} places, not 1")
        source = source.replace(old, new)
    return source


def jpcmciplus_class(patched: bool) -> type:
    """tigramite's ``JPCMCIplus``, from the source with :data:`PATCH` applied if `patched`."""
    import importlib.util

    from tigramite.jpcmciplus import JPCMCIplus

    if not patched:
        return JPCMCIplus
    spec = importlib.util.find_spec("tigramite.jpcmciplus")
    source = apply_patch(Path(spec.origin).read_text(), PATCH.read_text())
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, f"{spec.origin} ({PATCH.name})", "exec"), module.__dict__)  # noqa: S102
    return module.JPCMCIplus


@dataclass(frozen=True)
class TigramiteJPCMCIPlus:
    """Benchmark adapter for tigramite's JPCMCIplus with both dummies.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`, plus the optional ``python_env``.
    Build it with :func:`tigramite_jpcmciplus_adapter`, which reads the version.
    """

    name: str = "tigramite.jpcmciplus"
    package: str = "tigramite"
    package_version: str = ""
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "jpcmciplus"
    output_type: str = "temporal"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import tigramite before timing; see `contracts.SetupAdapter`."""
        from tigramite import data_processing  # noqa: F401
        from tigramite.independence_tests.parcorr_mult import ParCorrMult  # noqa: F401
        from tigramite.jpcmciplus import JPCMCIplus  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, the level, the test, the patch, and the columns' kinds (none yet)."""
        return {
            "tau_max": 2,
            "pc_alpha": 0.05,
            "test": "parcorr_mult",
            "patch": True,
            "n_system": 0,
            "time_contexts": [],
            "space_contexts": [],
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run tigramite JPCMCIplus; return the ``(tau_max + 1, n, n)`` marks over the columns."""
        from tigramite import data_processing

        from andrey_bench.window import marks_from_link_strings

        if params["test"] == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC as Test
        elif params["test"] == "parcorr_mult":
            from tigramite.independence_tests.parcorr_mult import ParCorrMult as Test
        else:
            raise ValueError(f"test must be one of {TESTS}, got {params['test']!r}")
        x = np.asarray(data, dtype=np.float64)
        n_datasets, n_steps, width = x.shape
        blocks = {}
        for m in range(n_datasets):
            space = np.zeros((n_steps, n_datasets))
            space[:, m] = 1.0
            blocks[m] = np.hstack([x[m], np.eye(n_steps), space])
        time_node, space_node = width, width + 1
        kinds = dict.fromkeys(range(int(params["n_system"])), "system")
        kinds |= dict.fromkeys((int(c) for c in params["time_contexts"]), "time_context")
        kinds |= dict.fromkeys((int(c) for c in params["space_contexts"]), "space_context")
        kinds |= {time_node: "time_dummy", space_node: "space_dummy"}
        vectors = {j: [(j, 0)] for j in range(width)}
        vectors[time_node] = [(width + k, 0) for k in range(n_steps)]
        vectors[space_node] = [(width + n_steps + m, 0) for m in range(n_datasets)]
        frame = data_processing.DataFrame(
            data=blocks, analysis_mode="multiple", vector_vars=vectors
        )
        model = jpcmciplus_class(bool(params["patch"]))(
            dataframe=frame, cond_ind_test=Test(), node_classification=kinds, verbosity=0
        )
        results = model.run_jpcmciplus(
            tau_max=int(params["tau_max"]), pc_alpha=float(params["pc_alpha"])
        )
        return marks_from_link_strings(np.asarray(results["graph"])[:width, :width])

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> a ``WindowGraph`` whose lag 0 is a CPDAG. Runs in the parent; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="cpdag")


def tigramite_jpcmciplus_adapter() -> TigramiteJPCMCIPlus:
    """A :class:`TigramiteJPCMCIPlus` carrying its environment's tigramite version."""
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    return TigramiteJPCMCIPlus(package_version=_tigramite_version(python) if python else "unknown")
