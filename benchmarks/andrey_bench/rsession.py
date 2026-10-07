"""Embedded R session for the pcalg and bnlearn adapters.

rpy2 loads `libR` into the benchmark worker. The worker's `ru_maxrss`, tree RSS sampler,
and the parent's `killpg` cover R without separate process management. R dies with its worker;
a killed fit leaves no R process behind.

The worker runs the adapter's `setup()` outside timing to start the session. :func:`session` raises
if setup has not run, preventing R startup from being included in fit timing.

Every process fitting an R solution needs the modules `benchmarks/start_r.sh --env` names loaded - R
and the compiler it was built with - and `R_LIBS_USER` set to the library that script built. The
runner passes its environment to each child; exporting these settings in the driver's shell is
sufficient.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

#: Names for matrices passed to R. bnlearn requires column names; results are decoded by name.
COLUMN_PREFIX = "V"

# rpy2 ships an ABI-mode binding; building the API-mode binding requires R headers. Selecting
# ABI mode avoids failed API-mode import attempts and their stderr messages. Call overhead differs
# by microseconds; fits take seconds.
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

_SESSION: "RSession | None" = None


def columns(d: int) -> list[str]:
    """The column names for a ``d``-variable dataset, in Andrey's column order."""
    return [f"{COLUMN_PREFIX}{i}" for i in range(d)]


def start(*packages: str) -> "RSession":
    """Start embedded R and load `packages`; untimed and idempotent.

    Each worker has one session. Repeated calls reuse it and load only packages not yet loaded.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = RSession()
    _SESSION.library(*packages)
    return _SESSION


def session() -> "RSession":
    """Return the running session.

    Raises
    ------
    RuntimeError
        If no session has started. Starting R here would include interpreter and package loading
        in the caller's timing, including an adapter's timed fit.
    """
    if _SESSION is None:
        raise RuntimeError(
            "no R session: rsession.start() runs in the adapter's setup(), which the worker calls "
            "untimed. Starting R here would time it as part of the fit."
        )
    return _SESSION


def package_version(name: str) -> str:
    """Read an installed R package's version from its `DESCRIPTION` file.

    Adapters evaluate the `package_version` class attribute when the driver imports them.
    Reading the filesystem avoids starting R to build the solution list.
    """
    for root in library_paths():
        description = root / name / "DESCRIPTION"
        if not description.exists():
            continue
        for line in description.read_text(errors="replace").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def _rpy2_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("rpy2")
    except metadata.PackageNotFoundError:
        return "unknown"


def check_environment(*packages: str) -> dict[str, str]:
    """Start R and report the fit environment.

    Called before the driver's first fit. A missing module or library produces one error instead
    of an error record for every campaign unit. Record the returned versions in `run_meta.json`
    to distinguish results from different R builds. Fails when R loaded a package version other
    than the one the adapters record.
    """
    try:
        session = start(*packages)
    except Exception as exc:
        raise SystemExit(
            f"the R lane is not usable in this environment: {exc}\n"
            "Run benchmarks/start_r.sh, then load the modules and export the R_LIBS_USER that "
            "`benchmarks/start_r.sh --env` prints. A libstdc++ or GLIBCXX error means the compiler "
            "module is not loaded."
        ) from exc
    versions = {
        "R": session.version,
        "rpy2": _rpy2_version(),
        "R_LIBS_USER": os.environ.get("R_LIBS_USER", "<unset>"),
    }
    for name in packages:
        loaded, where = session.loaded_package(name)
        # Adapters record the version `package_version` reads off disk; R may have loaded another.
        if loaded != package_version(name):
            raise SystemExit(
                f"R loaded {name} {loaded} from {where}, but the records would say "
                f"{package_version(name)}: keep one {name} on the R library path"
            )
        versions[name] = loaded
    return versions


def library_paths() -> list[Path]:
    """The R library directories in the order R searches them: `R_LIBS`, `R_LIBS_USER`, the site
    library, then R's own."""
    home = os.environ.get("R_HOME", "")
    roots = [
        p
        for var in ("R_LIBS", "R_LIBS_USER")
        for p in os.environ.get(var, "").split(os.pathsep)
        if p
    ]
    site = os.environ.get("R_LIBS_SITE")
    if site is not None:
        roots += [p for p in site.split(os.pathsep) if p]
    elif home:
        roots.append(str(Path(home) / "site-library"))
    if home:
        roots.append(str(Path(home) / "library"))
    return [Path(p) for p in roots]


class RSession:
    """One embedded R interpreter, with the packages and functions an adapter needs.

    Constructed by :func:`start`; adapters reach it through :func:`session`.
    """

    def __init__(self) -> None:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.vectors import FloatMatrix

        self._ro = ro
        self._numpy2rpy = numpy2ri.numpy2rpy
        self._float_matrix = FloatMatrix
        self._is_null = ro.r["is.null"]
        self._loaded: set[str] = set()
        self.version: str = str(ro.r("R.version.string")[0])

    def library(self, *packages: str) -> None:
        """``library(pkg)`` for each package not already attached."""
        for name in packages:
            if name in self._loaded:
                continue
            try:
                self._ro.r(f"library({name})")
            except Exception as exc:  # rpy2 wraps R errors; include setup instructions.
                raise RuntimeError(
                    f"R package {name!r} is not installed for this R. Run benchmarks/start_r.sh, "
                    f"and export R_LIBS_USER (currently {os.environ.get('R_LIBS_USER', '<unset>')})"
                ) from exc
            self._loaded.add(name)

    def loaded_package(self, name: str) -> tuple[str, str]:
        """The version of attached package `name`, and the library directory R loaded it from.

        Read from the loaded copy's `DESCRIPTION` as written - `2.7-12`, the form `package_version`
        and the records carry - not through `packageVersion`, which normalizes it to `2.7.12`.
        """
        version = str(self._ro.r(f'utils::packageDescription("{name}")$Version')[0])
        where = str(self._ro.r(f'dirname(find.package("{name}"))')[0])
        return version, where

    def define(self, code: str) -> None:
        """Evaluate R source that defines the functions an adapter calls. Untimed setup work."""
        self._ro.r(code)

    def function(self, name: str) -> Any:
        """A handle to an R function defined by :meth:`define`."""
        return self._ro.r[name]

    def matrix(self, data: np.ndarray) -> Any:
        """A ``(n, d)`` numpy array as an R numeric matrix with columns ``V0 .. V{d-1}``.

        Part of the timed fit: this is the data transfer into R.
        """
        array = np.ascontiguousarray(data, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {array.shape}")
        # `FloatMatrix` assigns column names in R. On the vector returned by `numpy2rpy`,
        # assigning `colnames` only sets a Python attribute and leaves R's `dimnames` `NULL`.
        # bnlearn then supplies names, making decoding positional.
        r_matrix = self._float_matrix(self._numpy2rpy(array))
        r_matrix.colnames = self._ro.StrVector(columns(array.shape[1]))
        return r_matrix

    def square_matrix(self, value: Any, *, names: list[str]) -> np.ndarray:
        """Return an R square matrix as a numpy array with rows and columns in `names` order.

        Reindexing by `dimnames` prevents a solution's variable reordering from producing a
        transposed or permuted graph.
        """
        d = len(names)
        array = np.asarray(value)
        if array.shape != (d, d):
            raise ValueError(f"expected a {d}x{d} matrix from R, got shape {array.shape}")
        labels = self._dimnames(value)
        if labels is None:
            raise ValueError("R returned a matrix with no dimnames; cannot verify variable order")
        rows, cols = labels
        if sorted(rows) != sorted(names) or sorted(cols) != sorted(names):
            raise ValueError(f"R returned variables {sorted(set(rows))}, expected {sorted(names)}")
        order = [rows.index(name) for name in names]
        return array[np.ix_(order, [cols.index(name) for name in names])]

    def _dimnames(self, value: Any) -> tuple[list[str], list[str]] | None:
        """The ``(rownames, colnames)`` of an R matrix, or ``None`` when it carries neither."""
        names = self._ro.r["dimnames"](value)
        if bool(self._is_null(names)[0]) or len(names) != 2:
            return None
        rows, cols = names[0], names[1]
        if bool(self._is_null(rows)[0]) or bool(self._is_null(cols)[0]):
            return None
        return [str(x) for x in rows], [str(x) for x in cols]
