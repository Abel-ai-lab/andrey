"""Self-tests for the tolerance harness and the committed baselines."""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

from tests.recovery.helpers import datagen, harness
from tests.recovery.helpers.tolerances import SPEC

# --- fixture discovery ---------------------------------------------------------


def _all_baselines() -> list[tuple[str, str]]:
    found = []
    for path in sorted(harness.BASELINES_DIR.rglob("*.json")):
        found.append((path.parent.name, path.stem))
    return found


BASELINES = _all_baselines()
IDS = [f"{a}/{c}" for a, c in BASELINES]


def test_regression_baselines_exist():
    assert BASELINES, "no baseline fixtures discovered under tests/recovery/baselines/"


# --- schema + clean-room -------------------------------------------------------

_REQUIRED_KEYS = {
    "schema_version",
    "algorithm",
    "case",
    "input",
    "output",
}
# Clean-room: no host/hardware identifiers may leak into a committed fixture.
# Defense-in-depth (the capture script also refuses to leak); high-signal tokens
# only, word-boundaried so GPU names can't false-match inside a hex SHA, and
# private-IP patterns that won't match a version string like "0.1.4.0".
_FORBIDDEN = re.compile(
    r"\b(?:a800|a100|h100|v100)\b"  # NVIDIA datacenter GPUs
    r"|\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b"  # MAC address
    r"|\b10(?:\.\d{1,3}){3}\b"  # private IPv4 10.0.0.0/8
    r"|\b192\.168(?:\.\d{1,3}){2}\b"  # private IPv4 192.168.0.0/16
    r"|\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b"  # private IPv4 172.16.0.0/12
    r"|/home/|/users/|/tmp/|\.local\b|\.ad\.|hostname",
    re.IGNORECASE,
)


@pytest.mark.parametrize(("algorithm", "case"), BASELINES, ids=IDS)
def test_schema_and_cleanroom(algorithm, case):
    path = harness.baseline_path(algorithm, case)
    raw = path.read_text()
    assert not _FORBIDDEN.search(raw), f"forbidden identifier in {path.name}"

    doc = json.loads(raw)
    assert set(doc) == _REQUIRED_KEYS
    assert doc["schema_version"] == harness.SCHEMA_VERSION
    assert doc["algorithm"] == algorithm
    assert doc["case"] == case
    assert algorithm in SPEC


def test_forbidden_regex_catches_leaks():
    # Must catch host/hardware identifiers...
    for leak in (
        "NVIDIA A100-SXM4-80GB",
        "ran on A800",
        "00:1b:44:11:3a:b7",
        "192.168.1.42",
        "10.0.0.5",
        "172.16.3.9",
        "/home/alice/run",
        "/Users/bob/data",
        "box.ad.example.edu",
    ):
        assert _FORBIDDEN.search(leak), f"should flag: {leak!r}"
    # ...without false-positives on legitimate fixture content. The dotted quads are the sharp
    # edge: a version string and a float both look like a private IPv4 to a loose pattern.
    for ok in (
        "0.1.4.0",
        "12d28970e00555c9a611578a24c472ab797fcd02c41038b19d62e4061a0e8822",  # sha256
        "lingam_5v_uniform",
        "0.5813538891885072",
    ):
        assert not _FORBIDDEN.search(ok), f"false positive: {ok!r}"


# --- input integrity -----------------------------------------------------------


@pytest.mark.parametrize(("algorithm", "case"), BASELINES, ids=IDS)
def test_input_integrity(algorithm, case):
    doc = harness.load_baseline(algorithm, case)
    assert datagen.input_sha256(case) == doc["input"]["sha256"], "input drift vs recorded sha256"


# --- the gate PASSES on a matching candidate -----------------------------------


@pytest.mark.parametrize(("algorithm", "case"), BASELINES, ids=IDS)
def test_gate_passes_on_identity(algorithm, case):
    doc = harness.load_baseline(algorithm, case)
    result = harness.assert_baseline(algorithm, doc["output"], case=case)
    assert result.ok


def _numeric_baselines():
    return [(a, c) for a, c in BASELINES if SPEC[a]["numeric"]]


def _structural_baselines():
    return [(a, c) for a, c in BASELINES if SPEC[a]["structural"]]


def test_derived_baselines_nonempty():
    # Guard the parametrize sources: an empty list silently collects 0 cases and
    # reports PASS, vanishing the gate-fail coverage below.
    assert _numeric_baselines(), "no numeric baselines — numeric gate-fail tests would vanish"
    assert _structural_baselines(), (
        "no structural baselines — structural gate-fail tests would vanish"
    )


# --- compare() fails fast on an unusable spec/baseline (no silent pass) -----------


def test_compare_rejects_empty_spec():
    harness.SPEC["_tmp_empty"] = {"structural": [], "numeric": {}}
    try:
        with pytest.raises(ValueError):
            harness.compare("_tmp_empty", {}, {})  # nothing would be compared -> reject
    finally:
        del harness.SPEC["_tmp_empty"]


def test_compare_rejects_uncovered_baseline_field():
    algorithm, case = _structural_baselines()[0]
    baseline = dict(harness.load_baseline(algorithm, case)["output"])
    baseline["_unspecced"] = [1, 2, 3]  # a field not in SPEC would ship unverified
    with pytest.raises(ValueError):
        harness.compare(algorithm, baseline, baseline)


def test_compare_rejects_baseline_missing_spec_field():
    algorithm, case = _structural_baselines()[0]
    doc = harness.load_baseline(algorithm, case)
    baseline = dict(doc["output"])
    del baseline[SPEC[algorithm]["structural"][0]]  # broken fixture
    with pytest.raises(ValueError):
        harness.compare(algorithm, doc["output"], baseline)


def test_compare_is_nan_aware_on_both_paths():
    # No current baseline carries NaN, so exercise compare()'s equal_nan branch on a
    # synthetic spec: a candidate reproducing a reference NaN must PASS (no false
    # fail); a one-sided NaN must FAIL - on both the structural and numeric paths.
    # Guards the NaN handling needed once NaN-bearing methods (p-values) are captured.
    harness.SPEC["_tmp_nan"] = {
        "structural": ["g"],
        "numeric": {"v": {"atol": 1e-6, "rtol": 1e-6}},
    }
    try:
        baseline = {"g": [[float("nan"), 1.0], [2.0, 3.0]], "v": [float("nan"), 2.0, 3.0]}
        assert harness.compare("_tmp_nan", copy.deepcopy(baseline), baseline).ok  # matching NaN

        nbad = copy.deepcopy(baseline)
        nbad["v"] = [9.0, 2.0, 3.0]  # NaN -> finite on numeric path
        assert not harness.compare("_tmp_nan", nbad, baseline).ok

        sbad = copy.deepcopy(baseline)
        sbad["g"] = [[9.0, 1.0], [2.0, 3.0]]  # NaN -> finite on structural path
        assert not harness.compare("_tmp_nan", sbad, baseline).ok
    finally:
        del harness.SPEC["_tmp_nan"]


# --- the gate FAILS out of tolerance (so it can bite) --------------------------


def test_gate_fails_on_shape_mismatch():
    algorithm, case = _structural_baselines()[0]
    doc = harness.load_baseline(algorithm, case)
    field = SPEC[algorithm]["structural"][0]
    candidate = copy.deepcopy(doc["output"])
    candidate[field] = np.asarray(candidate[field])[:-1].tolist()  # drop a row -> wrong shape
    assert not harness.compare(algorithm, candidate, doc["output"]).ok


def test_gate_fails_on_missing_candidate_field():
    algorithm, case = _structural_baselines()[0]
    doc = harness.load_baseline(algorithm, case)
    candidate = copy.deepcopy(doc["output"])
    del candidate[SPEC[algorithm]["structural"][0]]
    assert not harness.compare(algorithm, candidate, doc["output"]).ok


@pytest.mark.parametrize(("algorithm", "case"), _numeric_baselines())
def test_gate_fails_on_numeric_perturbation(algorithm, case):
    doc = harness.load_baseline(algorithm, case)
    field, tol = next(iter(SPEC[algorithm]["numeric"].items()))
    candidate = copy.deepcopy(doc["output"])

    perturbed = np.asarray(candidate[field], dtype=np.float64)
    bump = tol["atol"] * 1e3 + 1.0  # comfortably beyond atol/rtol
    flat = perturbed.ravel()
    flat[0] += bump
    candidate[field] = perturbed.reshape(np.shape(candidate[field])).tolist()

    assert not harness.compare(algorithm, candidate, doc["output"]).ok


@pytest.mark.parametrize(("algorithm", "case"), _numeric_baselines())
def test_gate_tolerates_subthreshold_noise(algorithm, case):
    doc = harness.load_baseline(algorithm, case)
    field, tol = next(iter(SPEC[algorithm]["numeric"].items()))
    candidate = copy.deepcopy(doc["output"])

    perturbed = np.asarray(candidate[field], dtype=np.float64) + tol["atol"] * 0.5
    candidate[field] = perturbed.tolist()

    assert harness.compare(algorithm, candidate, doc["output"]).ok


def _perturb_structural(value):
    """Flip an array cell or append to a ragged list, which numpy cannot ravel."""
    if harness._is_rectangular(value):
        arr = np.asarray(value)
        flat = arr.ravel().copy()
        flat[0] = flat[0] + 1 if flat[0] == 0 else 0  # change a structural cell
        return flat.reshape(arr.shape).tolist()
    changed = copy.deepcopy(value)
    changed.append([-1])  # ragged: a phantom entry breaks exact structural equality
    return changed


@pytest.mark.parametrize(("algorithm", "case"), _structural_baselines())
def test_gate_fails_on_structural_perturbation(algorithm, case):
    doc = harness.load_baseline(algorithm, case)
    field = SPEC[algorithm]["structural"][0]
    candidate = copy.deepcopy(doc["output"])
    candidate[field] = _perturb_structural(candidate[field])
    assert not harness.compare(algorithm, candidate, doc["output"]).ok


def test_assert_baseline_raises_on_mismatch():
    algorithm, case = _numeric_baselines()[0]
    doc = harness.load_baseline(algorithm, case)
    field = next(iter(SPEC[algorithm]["numeric"]))
    candidate = copy.deepcopy(doc["output"])
    candidate[field] = (np.asarray(candidate[field], dtype=np.float64) + 1.0).tolist()
    with pytest.raises(AssertionError):
        harness.assert_baseline(algorithm, candidate, case=case)
