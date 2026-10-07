"""Compare recorded outputs using per-algorithm tolerances.

Fixture schema (``baselines/<algorithm>/<case>.json``)::

    {
      "schema_version": int,
      "algorithm": str, "case": str,
      "input": {"case": str, "seed": int, "shape": [n, d], "sha256": str}
             | {"case": str, "seeds": [int], "shapes": [[n, d]], "sha256": str},
      "output": {"<field>": <list>, ...}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import datagen
from .tolerances import SPEC

SCHEMA_VERSION = 3
BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"


def baseline_path(algorithm: str, case: str) -> Path:
    return BASELINES_DIR / algorithm / f"{case}.json"


def load_baseline(algorithm: str, case: str) -> dict:
    """Read a baseline fixture; fail fast on a schema-version mismatch."""
    with open(baseline_path(algorithm, case)) as fh:
        doc = json.load(fh)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{baseline_path(algorithm, case)}: schema_version "
            f"{doc.get('schema_version')} != {SCHEMA_VERSION}"
        )
    return doc


@dataclass
class RegressionResult:
    algorithm: str
    ok: bool
    failures: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return f"{self.algorithm}: regression OK"
        return f"{self.algorithm}: regression FAILED\n  - " + "\n  - ".join(self.failures)


def _structural_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Exact equality; matching NaNs count as equal (a reference may carry NaN
    # sentinels in a float-coded structural field).
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _is_rectangular(value: object) -> bool:
    """True if ``value`` forms a homogeneous numpy array, not a ragged nested list."""
    try:
        return np.asarray(value).dtype != object
    except ValueError:  # inhomogeneous / ragged nested lists
        return False


def _structural_field_failure(name: str, candidate: object, baseline: object) -> str | None:
    """``None`` if a structural field matches exactly, else a diagnostic string.

    Handles both rectangular arrays (graph adjacency: NaN-aware, cell-diff diagnostics) and ragged
    nested lists (for example, CAMUV parent-index lists), which numpy cannot stack into one array --
    those fall back to exact nested equality.
    """
    if not (_is_rectangular(candidate) and _is_rectangular(baseline)):
        return None if candidate == baseline else f"{name}: structural mismatch"
    a = np.asarray(candidate)
    b = np.asarray(baseline)
    if a.shape != b.shape:
        return f"{name}: shape {a.shape} != baseline {b.shape}"
    if not _structural_equal(a, b):
        return f"{name}: structural mismatch ({int(np.sum(a != b))} cells differ)"
    return None


def _finite_maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a - b)
    finite = diff[np.isfinite(diff)]
    return float(finite.max()) if finite.size else float("nan")


def compare(
    algorithm: str,
    candidate: dict,
    baseline_output: dict,
) -> RegressionResult:
    """Compare ``candidate`` against ``baseline_output`` for ``algorithm``.

    Structural fields must match exactly, numeric fields within ``atol``/``rtol``.
    Matching NaNs count as equal. Raises ``ValueError`` on an unusable spec: an
    empty spec, a baseline field not in the spec, or a spec field not in the baseline.
    """
    if algorithm not in SPEC:
        raise KeyError(f"no tolerance spec for {algorithm!r}")
    spec = SPEC[algorithm]
    spec_fields = set(spec["structural"]) | set(spec["numeric"])
    if not spec_fields:
        raise ValueError(f"empty comparison spec for {algorithm!r}: nothing would be compared")
    uncovered = set(baseline_output) - spec_fields
    if uncovered:
        raise ValueError(
            f"{algorithm}: baseline fields not in the comparison spec (would go unchecked): "
            f"{sorted(uncovered)}"
        )
    failures: list[str] = []

    for name in spec["structural"]:
        if name not in baseline_output:
            raise ValueError(f"{algorithm}: spec field {name!r} missing from baseline output")
        if name not in candidate:
            failures.append(f"{name}: missing from candidate")
            continue
        failure = _structural_field_failure(name, candidate[name], baseline_output[name])
        if failure is not None:
            failures.append(failure)

    for name, tol in spec["numeric"].items():
        if name not in baseline_output:
            raise ValueError(f"{algorithm}: spec field {name!r} missing from baseline output")
        if name not in candidate:
            failures.append(f"{name}: missing from candidate")
            continue
        atol = tol["atol"]
        rtol = tol["rtol"]
        a = np.asarray(candidate[name], dtype=np.float64)
        b = np.asarray(baseline_output[name], dtype=np.float64)
        if a.shape != b.shape:
            failures.append(f"{name}: shape {a.shape} != baseline {b.shape}")
        elif not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            maxdiff = _finite_maxdiff(a, b)
            failures.append(
                f"{name}: numeric mismatch (max|Δ|={maxdiff:.3e}, atol={atol:g}, rtol={rtol:g})"
            )

    return RegressionResult(algorithm=algorithm, ok=not failures, failures=failures)


def check_input(case: str, algorithm: str) -> dict:
    """Load the baseline and re-check the regenerated input against its recorded SHA-256.

    Returns the baseline document. Every gate calls this before comparing: if the generator drifts,
    the comparison downstream is against different data and its verdict means nothing, so drift has
    to fail loudly rather than surface as an output mismatch.
    """
    doc = load_baseline(algorithm, case)
    recomputed = datagen.input_sha256(case)
    if recomputed != doc["input"]["sha256"]:
        raise AssertionError(
            f"input drift for case {case!r}: regenerated sha256 {recomputed} "
            f"!= baseline {doc['input']['sha256']}"
        )
    return doc


def assert_baseline(
    algorithm: str,
    candidate: dict,
    *,
    case: str,
) -> RegressionResult:
    """Assert a candidate matches the baseline for ``(algorithm, case)``.

    Also re-checks the regenerated input against the baseline's recorded SHA-256 so
    input drift surfaces as a clear failure.
    """
    doc = check_input(case, algorithm)
    result = compare(algorithm, candidate, doc["output"])
    if not result.ok:
        raise AssertionError(result.describe())
    return result
