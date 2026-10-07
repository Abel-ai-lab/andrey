"""The public-datasets summary: every fit is accounted for, and nothing machine-specific ships."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andrey.data import list_datasets

SUMMARY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "published"
    / "alpha-2026-09"
    / "public-datasets.json"
)


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_every_dataset_is_a_load_dataset_name(summary):
    names = [d["dataset"] for d in summary["datasets"]]
    assert set(names) <= set(list_datasets())


def test_every_fit_finished_or_is_named_as_not_finishing(summary):
    for dataset in summary["datasets"]:
        for method in dataset["methods"]:
            assert any(f["package"] == "Andrey" for f in method["fits"])
            for fit in method["fits"]:
                if fit["status"] == "ok":
                    assert fit["seconds"] is not None and fit["shd"] is not None
                else:
                    assert fit["status"] in ("capped", "failed") and fit["note"]
                    assert fit["seconds"] is None  # no speed without a finished fit


def test_the_lingam_methods_run_only_on_measured_data(summary):
    for dataset in summary["datasets"]:
        methods = {m["method"] for m in dataset["methods"]}
        lingam = {"DirectLiNGAM", "ICA-LiNGAM"} & methods
        assert bool(lingam) is (dataset["data"] == "measured"), dataset["dataset"]


def test_the_summary_records_versions_settings_and_hardware(summary):
    """Every package version, the settings, and the hardware."""
    assert all(summary["packages"].values()) and summary["settings"] and summary["hardware"]
