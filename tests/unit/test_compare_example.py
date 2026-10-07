"""The one-dataset comparison page and its recorded causal-learn graphs."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SVG = "{http://www.w3.org/2000/svg}"


def load():
    spec = importlib.util.spec_from_file_location("compare_example", ROOT / "apps" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load()
RECORDING = json.loads(compare.RECORDING.read_text())
PACKAGES = ("Andrey", "causal-learn")


@pytest.fixture(scope="module")
def page():
    return compare.compare_page()


@pytest.fixture(scope="module")
def estimates():
    graphs = {"Andrey": compare.andrey_graphs(), "causal-learn": compare.causal_learn_graphs()}
    return {
        (package, method): graphs[package][method]
        for package in PACKAGES
        for method in graphs[package]
    }


def edge_count(graph):
    marks = graph.to_numpy()
    return int(np.count_nonzero(np.triu(marks | marks.T)))


def section(page, method):
    return re.search(rf'<section data-method="{method}".*?</section>', page, re.S).group(0)


def test_recording_matches_the_current_dataset():
    data, _ = compare.dataset()
    assert RECORDING["data"]["sha256"] == compare.fingerprint(data)
    assert RECORDING["data"]["labels"] == list(compare.LABELS)
    assert RECORDING["causal-learn"] == compare.CAUSAL_LEARN
    assert set(RECORDING["methods"]) == set(compare.METHODS)


@pytest.mark.parametrize(
    "key, value",
    [("sha256", "0" * 64), ("labels", sorted(compare.LABELS)), ("shape", [999, 10])],
)
def test_a_recording_from_other_data_is_refused(key, value):
    stale = copy.deepcopy(RECORDING)
    stale["data"][key] = value
    with pytest.raises(ValueError, match=r"apps/compare\.py --record"):
        compare.causal_learn_graphs(stale)


def test_edge_statuses_add_up_to_the_metrics(estimates):
    truth = compare.truth_cpdag()
    for (package, method), estimate in estimates.items():
        statuses = compare.edge_statuses(estimate, truth)
        assert {status for *_, status in statuses} <= set(compare.STATUSES)
        tally = compare.counts(statuses)
        values = compare.metrics(estimate, truth)
        # Structural SHD charges one edit per pair whose endpoint marks differ.
        assert tally["missing"] + tally["extra"] + tally["wrong"] == values["shd"], package
        assert tally["correct"] + tally["missing"] + tally["wrong"] == edge_count(truth)
        assert tally["correct"] + tally["extra"] + tally["wrong"] == edge_count(estimate)
        joined = tally["correct"] + tally["wrong"]
        skeleton = 2 * joined / (2 * joined + tally["extra"] + tally["missing"])
        assert values["skeleton_f1"] == pytest.approx(skeleton), (package, method)


def test_the_default_dataset_shows_mistakes_for_each_package(estimates):
    truth = compare.truth_cpdag()
    for key, estimate in estimates.items():
        assert compare.metrics(estimate, truth)["shd"] >= 1, key


def test_page_shows_three_panels_and_six_numbers_per_method(page, estimates):
    truth = compare.truth_cpdag()
    for method in compare.METHODS:
        body = section(page, method)
        panels = re.findall(r'<figure class="panel[^"]*" data-package="([^"]+)"', body)
        assert panels == ["truth", *PACKAGES]
        shown = re.findall(r'<dd data-metric="(\w+)">([^<]+)</dd>', body)
        expected = [
            (key, compare.number(key, value))
            for package in PACKAGES
            for key, value in compare.metrics(estimates[package, method], truth).items()
        ]
        assert shown == expected
    opening = re.findall(r"<section [^>]*>", page)
    assert ["hidden" in tag for tag in opening] == [method != "PC" for method in compare.METHODS]


def test_page_names_variables_and_explains_every_edge(page):
    for label in compare.LABELS:
        assert f">{label}</text>" in page
    legend = re.search(r'<ul class="legend".*?</ul>', page, re.S).group(0)
    for name in compare.STATUSES.values():
        assert name in legend
    edges = re.findall(r'<g data-status="(\w+)">(<title>[^<]+</title>)?', page)
    assert edges and all(title for _, title in edges)
    missing = re.findall(r'<g data-status="missing">.*?</g></g>', page)
    assert missing and all("stroke-dasharray" in group for group in missing)


def test_page_is_standalone_and_names_no_internal_locations(page):
    markup = re.sub(r"<(style|script)>.*?</\1>", "", page, flags=re.S)
    for needle in ("/scratch", "src/andrey", "apps/", "tmp/", ".json", "abel.ai"):
        assert needle not in markup
    assert not re.search(r"""(src|href)=["']?https?:""", page)
    text = re.sub(r"<[^>]+>", " ", markup).lower()
    assert not re.search(r"#\d", text)
    for phrase in ("against", " vs", "cause and effect", "wins"):
        assert phrase not in text


def test_thumbnail_is_a_concrete_svg_with_mistakes():
    svg = compare.compare_thumbnail()
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG}svg"
    assert "var(" not in svg
    assert 1.2 <= float(root.get("width")) / float(root.get("height")) <= 1.45
    statuses = {g.get("data-status") for g in root.iter(f"{SVG}g")} - {None}
    assert statuses - {"correct"}
