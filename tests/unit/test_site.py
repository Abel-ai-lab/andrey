"""The homepage states only what the benchmark summary and the method registry support.

The tests that fill the page's template need Jinja, from the docs group; without it they skip.
"""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import andrey.viz
from andrey import GraphStructure
from andrey.core.structure import TAIL
from andrey.spec import get_spec, list_specs

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

PLACEHOLDER = ROOT / "tests" / "unit" / "fixtures" / "summary.placeholder.json"
HEADLINE = "Over 100x faster on PC, and faster on the other supported methods."
# A stand-in for pydata-sphinx-theme's layout: the blocks the homepage template fills or keeps.
LAYOUT = (
    "<head>{% block htmltitle %}{% endblock %}{% block extrahead %}{% endblock %}</head>"
    "<body>{% block content %}"
    "{% block search_dialog %}{% endblock %}"
    '{% block docs_navbar %}<header id="pst-header">navbar</header>{% endblock %}'
    "{% block scripts_end %}{% endblock %}{% endblock %}{% block footer %}{% endblock %}</body>"
)


def render_homepage(summary, *, placeholder=False):
    """The homepage template filled from ``summary``, as the site build fills it."""
    jinja2 = pytest.importorskip("jinja2")
    loader = jinja2.ChoiceLoader(
        [
            jinja2.DictLoader(
                {
                    "layout.html": LAYOUT,
                    "sections/sidebar-primary.html": "",
                    "sections/announcement.html": "",
                }
            ),
            jinja2.FileSystemLoader(ROOT / "site"),
        ]
    )
    template = jinja2.Environment(loader=loader).get_template("homepage.html")
    return template.render(home=build.homepage_context(summary, placeholder=placeholder))


@pytest.fixture(scope="module", autouse=True)
def sachs_offline(offline_datasets_module):
    """The homepage example loads Sachs; the fixture's copy holds the same 853 rows."""


@pytest.fixture(scope="module")
def placeholder_page():
    return render_homepage(build.load_summary(PLACEHOLDER, placeholder=True))


def _row(d, andrey_s, comparator_s, ratio, andrey_shd, comparator_shd, empty_shd):
    return {
        "d": d,
        "seeds": 3,
        "andrey_s": andrey_s,
        "comparator_s": comparator_s,
        "ratio": ratio,
        "andrey_shd": andrey_shd,
        "comparator_shd": comparator_shd,
        "empty_shd": empty_shd,
    }


def _method(name, family, comparator, rows, regime="linear_gauss_er"):
    return {
        "method": name,
        "family": family,
        "comparator": comparator,
        "regime": regime,
        "hardware": "AMD EPYC 7713, 16 cores",
        "build": "pre-release build, 15 September 2026",
        "rows": rows,
    }


SUMMARY = {
    "generated": "2026-09-15",
    "headline": HEADLINE,
    "methods": [
        _method(
            "PC", "constraint-based", "causal-learn", [_row(400, 0.82, 617, 941.05, 357, 357, 400)]
        ),
        _method(
            "FCI",
            "constraint-based",
            "causal-learn",
            [
                _row(20, 0.061, 0.038, 0.62, 12, 12, 20),
                _row(50, 0.2, 0.68, 3.4, 30, 31, 50),
                _row(400, 9.0, 80.0, 8.9, 450, 452, 400),
                _row(800, 40.0, 350.0, 8.75, 900, 905, 800),
            ],
            regime="latent_gauss_er",
        ),
        _method(
            "ICA-LiNGAM",
            "functional",
            "causal-learn",
            # Timed, but its datasets could not be paired within a run: no ratio at any size.
            [_row(20, 1.00, 1.01, None, 25, 25, 19), _row(50, 2.02, 2.00, None, 60, 60, 49)],
            regime="lingam_sf",
        ),
        _method(
            "DirectLiNGAM",
            "functional",
            "lingam",
            [
                _row(100, 37.7, 135.6, 3.59, 6.7, 6.3, 99),
                _row(200, 481.9, None, None, 14, None, 199),
            ],
            regime="lingam_sf",
        ),
    ],
}


def test_headline_must_be_the_claim_the_numbers_support():
    summary = copy.deepcopy(SUMMARY)
    summary["headline"] = "Up to 941x faster than other popular causal discovery packages."
    with pytest.raises(ValueError, match="Over 100x faster on PC"):
        build.fastest(summary)


def _speed_up(summary: dict, method: str, ratio: float) -> None:
    """Set the ratio at the largest size of every comparison of ``method``."""
    for m in summary["methods"]:
        if m["method"] == method:
            rows = [r for r in m["rows"] if r["ratio"] is not None]
            max(rows, key=lambda r: r["d"])["ratio"] = ratio


def test_the_claim_rounds_down_and_says_how_many_other_methods_are_faster():
    summary = copy.deepcopy(SUMMARY)
    # PC's 941x rounds down to 100x; FCI and DirectLiNGAM are both faster.
    assert build.speed_claim(summary) == (
        "Over 100x faster on PC",
        "faster on the other supported methods",
    )
    # One of the two not faster at its largest size: no majority, so the claim stops at PC.
    _speed_up(summary, "FCI", 0.98)
    assert build.speed_claim(summary) == ("Over 100x faster on PC", "")
    # A ratio that is exactly a power of ten is not "over" it.
    _speed_up(summary, "PC", 100.0)
    assert build.speed_claim(summary)[0] == "At least 100x faster on PC"


def test_the_published_claim_says_most_when_one_method_is_not_faster():
    summary = copy.deepcopy(SUMMARY)
    summary["methods"].append(
        {**summary["methods"][1], "method": "GES", "rows": [_row(150, 1.0, 4.3, 4.3, 5, 5, 150)]}
    )
    _speed_up(summary, "FCI", 0.98)
    assert build.speed_claim(summary)[1] == "faster on most other supported methods"


def test_notes_cover_every_cell_without_a_speedup():
    notes = build.exceptions(SUMMARY)
    assert (
        "FCI is slower than causal-learn at d = 20: 0.62x, 0.061 s compared with 0.038 s." in notes
    )
    assert "lingam did not finish DirectLiNGAM within the time cap at d = 200." in notes
    # Accuracy never withholds a speed ratio, so no note speaks of it.
    assert not any("empty graph" in note or "speed claim" in note for note in notes)


def test_charts_cover_only_comparisons_with_a_published_ratio():
    chart = build.speed_chart(SUMMARY)
    # One chart per method with a paired ratio, each titled with its largest one. The belt repeats
    # them once, hidden from assistive technology, so the loop is seamless.
    assert chart.count('<figure class="tchart">') == 3  # PC, FCI, DirectLiNGAM
    assert chart.count('<figure class="tchart" aria-hidden="true">') == 3
    ups = re.findall(r"up to <b>([^<]+)</b>", chart)
    # FCI's largest is at d = 400, where both graphs are worse than the empty graph: a speed
    # ratio does not depend on accuracy.
    assert ups == ["941x", "8.9x", "3.6x"] * 2
    # Sizes without a ratio are plotted hollow: DirectLiNGAM 200 (Andrey only; lingam passed its
    # time cap).
    assert chart.count("is-open") == 1 * 2
    assert "ICA-LiNGAM" not in chart and "ICA-LiNGAM" not in build.accuracy_chart(SUMMARY)
    # A comparison the charts skip is in the table, not in the notes.
    assert not any("ICA-LiNGAM" in note for note in build.exceptions(SUMMARY))
    assert "ICA-LiNGAM" in build.speed_table(SUMMARY)


def test_speed_intro_names_the_packages_and_leaves_accuracy_out():
    intro = build.speed_intro(SUMMARY)
    assert "alongside causal-learn and lingam" in intro
    assert "errors" not in intro


def test_quality_counts_every_scored_size():
    # PC 357/357, FCI 12/12, 30/31, 450/452, 900/905, DirectLiNGAM 6.7/6.3: five as good. At
    # DirectLiNGAM 200 lingam passed its time cap, so it has no graph to count.
    intro = build.quality_intro(SUMMARY)
    assert "or fewer, in 5 of 6 comparisons, and every release is tested" in intro
    assert build.accuracy_tally(SUMMARY) == (5, 6)


def test_the_release_promise_is_the_quality_gate():
    # The intro says every release is tested against true graphs. The quality gate is that test,
    # and tests/unit/test_release_gates.py holds every PyPI publish job to needing it.
    assert "every release is tested against the true graphs" in build.quality_intro(SUMMARY)
    gate = (ROOT / ".github" / "workflows" / "quality-gate.yml").read_text()
    assert "workflow_call:" in gate and "python -m qa.quality_gate" in gate


def test_errors_compare_as_a_share_rounded_against_andrey():
    exact = {"d": 50, "seeds": 3, "andrey_s": 1.0, "comparator_s": 2.0, "ratio": 2.0}
    exact |= {"andrey_shd": 0, "comparator_shd": 0, "empty_shd": 50}
    rows = [
        exact,
        dict(exact, d=100, andrey_shd=2, comparator_shd=0),  # no share of no errors: a count
        dict(exact, d=200, andrey_shd=30, comparator_shd=31),  # 3.2% fewer rounds down
        dict(exact, d=400, andrey_shd=450, comparator_shd=452),  # 0.4% fewer
        dict(exact, d=800, andrey_shd=6.7, comparator_shd=6.3),  # 6.3% more rounds up
        dict(exact, d=1600, andrey_shd=12, comparator_shd=None),  # past the time cap: no row
    ]
    compared = build.error_comparisons([{"comparator": "other", "rows": rows}])
    assert [(c["d"], c["kind"], c["label"]) for c in compared] == [
        (50, "same", "same"),
        (100, "more", "2 more"),
        (200, "fewer", "3% fewer"),
        (400, "fewer", "<1% fewer"),
        (800, "more", "7% more"),
    ]


def test_methods_list_every_registered_method():
    title, html = build.methods(build.load_summary(PLACEHOLDER, placeholder=True))
    specs = list_specs()
    assert title.startswith(f"{len(specs)} methods in ")
    assert html.count('class="method"') == len(specs)
    for spec in specs:
        assert f"docs/code/generated/andrey.{spec.name}.html" in html
    # PC, FCI, GES, BOSS, GRaSP, and DirectLiNGAM are in the placeholder summary: each has the
    # seal, and a tooltip that describes it.
    assert html.count('class="method-bench"') == 6
    assert html.count('role="tooltip"') == 6 and "Benchmarked</a>" not in html
    experimental = [s for s in specs if s.status == "experimental"]
    assert html.count('class="method-status"') == len(experimental)
    assert "Experimental Score-then-FCI hybrid" in re.sub(r"<[^>]+>", "", html)  # read as text


def test_method_descriptions_stop_at_the_first_clause():
    assert build.first_clause(get_spec("pc").summary) == (
        "General-purpose structure learning from i.i.d. observational data"
    )
    for spec in list_specs():
        assert 0 < len(build.first_clause(spec.summary)) <= 140, spec.name


def test_layout_is_checked(tmp_path):
    summary = copy.deepcopy(SUMMARY)
    del summary["methods"][0]["rows"][0]["empty_shd"]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="empty_shd"):
        build.load_summary(path, placeholder=True)


def test_unpublished_summary_needs_the_placeholder_flag():
    with pytest.raises(ValueError, match="andrey_placeholder=1"):
        build.load_summary(PLACEHOLDER)
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    marked = build.homepage_context(summary, placeholder=True)
    assert 'content="noindex"' in marked["robots"] and "Not for release" in marked["banner"]
    unmarked = build.homepage_context(summary)
    assert unmarked["robots"] == unmarked["banner"] == ""


def test_page_shows_every_claimed_ratio_and_no_internal_locations(placeholder_page):
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    page = placeholder_page
    # Each chart's largest speedup, as "up to"; every size is on the benchmarks page.
    for _, methods in build.time_groups(summary):
        claimed = [r["ratio"] for m in methods for r in m["rows"] if r["ratio"] is not None]
        assert f"up to <b>{build.ratio_text(max(claimed))}</b>" in page
    assert build.escape(build.home_claim(summary)) in page
    assert not re.search(r"/scratch/|/home/|\bsc\d{3}\b|\bsol\b", page, re.IGNORECASE)


def test_example_prints_the_recovered_edges():
    source, printed, svg = build.run_example()
    assert "andrey.pc(df)" in source
    assert printed.splitlines() == [
        "PC  cpdag  |  11 nodes  |  8 edges (6 undirected, 2 directed)",
        "",
        "  raf -- mek",
        "  plc -- pip3",
        "  pip2 -- pip3",
        "  erk -- akt",
        "  erk -- pka",
        "  akt -- pka",
        "  p38 -> pkc",
        "  jnk -> pkc",
    ]
    assert svg.startswith("<svg") and "var(--andrey-" in svg


def test_the_example_copy_counts_the_links_pc_finds_in_the_known_network(placeholder_page):
    from andrey.data import load_dataset

    found = build.run_script("example.py")[2]["out"].structure.oriented_edges()
    known = load_dataset("sachs").graph.oriented_edges()
    found, known = ({frozenset(edge[:2]) for edge in edges} for edges in (found, known))
    assert found <= known
    claim = (
        f"PC finds {len(found)} of the {len(known)} links in the known signalling network, "
        "and none outside it"
    )
    guide = (ROOT / "docs" / "docs" / "guides" / "getting-started.md").read_text(encoding="utf-8")
    for text in (placeholder_page, guide):
        assert claim in " ".join(text.split())


def test_a_graph_in_parts_is_drawn_part_by_part():
    marks = np.zeros((4, 4), np.int8)
    marks[0, 1] = marks[1, 0] = marks[2, 3] = marks[3, 2] = TAIL  # two undirected pairs
    positions = build.parts_layout(GraphStructure.from_numpy(marks, kind="cpdag"))
    assert max(positions[0][0], positions[1][0]) < min(positions[2][0], positions[3][0])
    assert all(0 <= c <= 1 for position in positions.values() for c in position)


def test_wordmark_takes_the_palette_ink():
    for name, palette in (("light", andrey.viz.LIGHT), ("dark", andrey.viz.DARK)):
        svg = (ROOT / "site" / "brand" / f"wordmark-{name}.svg").read_text()
        assert re.findall(r'fill="([^"]+)"', svg) == [palette.ink]


def test_a_ratio_needs_both_times_and_not_better_graphs(tmp_path):
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    # gCastle at d = 400 scores SHD 792.5 against the empty graph's 400; its ratio still stands.
    gcastle = summary["methods"][1]
    assert gcastle["comparator"] == "gCastle" and gcastle["rows"][4]["ratio"] == 253.52
    lingam = next(m for m in summary["methods"] if m["comparator"] == "lingam")
    lingam["rows"][-1]["ratio"] = 5.0  # d = 200: lingam has no time
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="a time is missing"):
        build.load_summary(path, placeholder=True)


def test_a_skipped_comparison_still_reports_a_time_cap():
    summary = copy.deepcopy(SUMMARY)
    ica = summary["methods"][2]
    ica["rows"].append(_row(100, 30.0, None, None, 90, None, 99))
    notes = build.exceptions(summary)
    assert [n for n in notes if "ICA-LiNGAM" in n] == [
        "causal-learn did not finish ICA-LiNGAM within the time cap at d = 100."
    ]


def test_numbers_read_as_written():
    assert build.number_text(1473.0) == "1,473"
    assert build.number_text(792.5) == "792.5"
    assert build.number_text(0.66667) == "0.67"
    assert build.number_text(None) == "-"
    assert build.ratio_text(9.96) == "10x"
    assert build.ratio_text(0.996) == "1.0x"


def test_intro_claims_one_machine_per_size_not_per_comparison():
    # Two comparisons list two CPU models across their sizes, so the pairing holds within a size.
    intro = build.speed_intro(SUMMARY)
    assert "on the same datasets and the same machine at each size" in intro
    assert "Each comparison ran both packages" not in intro


def test_placeholder_numbers_are_refused_wherever_the_file_sits(tmp_path, monkeypatch):
    published = tmp_path / "published"
    published.mkdir()
    copied = published / "summary.json"
    copied.write_text(PLACEHOLDER.read_text())
    monkeypatch.setattr(build, "PUBLISHED", published)
    with pytest.raises(ValueError, match="placeholder numbers"):
        build.load_summary(copied)


def test_the_headline_is_the_slogan_and_the_claim_drops_what_it_says(placeholder_page):
    # The heading says "causal discovery", so the claim under it leaves those words out.
    found = re.search(r'<h1 id="headline">(.*?)</h1>\s*<p class="claim">([^<]*)<', placeholder_page)
    assert re.sub(r"<[^>]+>", "", found[1]) == "Causal discovery, fast."
    assert found[2] == "Over 100x faster on PC, and faster on the other supported methods."


def test_the_page_title_is_the_one_line_description(placeholder_page):
    # The description pyproject.toml and the README give, as the browser tab shows it.
    assert "<title>Andrey, a very fast causal discovery package</title>" in placeholder_page


# The registry as it is, and without one method the description does not name: the count of "more"
# methods follows the registry.
@pytest.mark.parametrize("dropped", [None, "gin"])
def test_the_description_states_the_speed_claim_and_the_method_count(monkeypatch, dropped):
    specs = [s for s in list_specs() if s.name != dropped]
    monkeypatch.setattr(build, "list_specs", lambda: specs)
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    description = build.page_description(summary)
    assert "over 100x faster on PC than other popular packages" in description
    assert f"with PC, GES, FCI and {len(specs) - 3} more methods" in description
    # Search results cut a description at about 160 characters.
    assert len(description) <= 160


@pytest.mark.parametrize("name", ["pc", "ges", "fci"])
def test_the_description_names_only_registered_methods(monkeypatch, name):
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    monkeypatch.setattr(build, "list_specs", lambda: [s for s in list_specs() if s.name != name])
    with pytest.raises(ValueError, match=f"unregistered methods: {name.upper()}$"):
        build.page_description(summary)


@pytest.fixture(scope="module")
def conf():
    """The namespace of ``docs/conf.py``, which needs the docs group."""
    for module in ("ablog", "resvg_py", "sphinx"):
        pytest.importorskip(module)
    path = list(sys.path)  # conf.py puts apps/ on the path
    try:
        return runpy.run_path(str(ROOT / "docs" / "conf.py"))
    finally:
        sys.path[:] = path


FRONT, FIRST, HOME = (
    "From the front matter.",
    "The page's first sentence.",
    "The homepage's sentence.",
)


# An empty or blank front-matter value counts as none. A page with no paragraph, such as search,
# takes the homepage's description.
@pytest.mark.parametrize(
    ("meta", "with_paragraph", "without"),
    [
        ({"description": FRONT}, FRONT, FRONT),
        ({"description": ""}, FIRST, HOME),
        ({"description": "  "}, FIRST, HOME),
        ({}, FIRST, HOME),
    ],
)
def test_a_page_without_a_description_takes_its_first_paragraph(
    conf, meta, with_paragraph, without
):
    from docutils import nodes
    from docutils.utils import new_document

    doctree = new_document("page")
    doctree += nodes.paragraph(text=FIRST)
    app = types.SimpleNamespace(andrey_home={"description": HOME})
    assert conf["_description"](app, {"meta": meta}, doctree) == with_paragraph
    assert conf["_description"](app, {"meta": meta}, None) == without


def test_the_speedup_panel_shows_one_method_at_a_time(placeholder_page):
    # The page's own style shows the checked method's panel and size; the stylesheets carry no
    # per-method rule.
    assert ".race:has(#race-m-pc:checked) :is(#race-panel-pc, #race-size-pc)" in placeholder_page
    assert placeholder_page.count('<figure class="race') == 1


def _labels(html, method):
    panel = re.search(
        rf'id="race-panel-{method}"[^>]*>(.*?)(?=<div class="race-panel|</div></figure>)', html
    )[1]
    names = re.findall(r'<span class="race-name">([^<]+)</span>', panel)
    values = re.findall(r'<span class="race-time">([^<]+)</span>', panel)
    return dict(zip(names, values)), panel


def test_the_speedup_panel_has_a_slide_per_method_at_its_largest_claimed_size():
    summary = build.load_summary(build.SUMMARY) if build.SUMMARY.exists() else SUMMARY
    html = build.speedup_panel(summary)
    sizes = re.findall(r'id="race-size-([\w-]+)">([\d,]+) variables', html)
    assert sizes == [
        ("pc", "400"),
        ("fci", "800"),
        ("ges", "150"),
        ("boss", "800"),
        ("grasp", "400"),
        ("directlingam", "200"),
    ]
    assert re.findall(r'<label for="race-m-[\w-]+">([^<]+)</label>', html) == list(
        build.SPEEDUP_METHODS
    )
    assert 'id="race-m-pc" checked' in html and html.count(" checked") == 1
    assert '<legend class="visually-hidden">Method</legend>' in html
    assert '<a class="race-link" href="docs/benchmarks.html">Benchmark report</a>' in html


def test_every_speedup_bar_comes_from_andreys_paired_ratios():
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    labels, panel = _labels(build.speedup_panel(summary), "pc")
    pc = {m["comparator"]: m for m in summary["methods"] if m["method"] == "PC"}
    row = {name: next(r for r in m["rows"] if r["d"] == 400) for name, m in pc.items()}
    # Andrey's bar is its paired ratio against the slowest package (median times give 756x);
    # gCastle's is that ratio over Andrey's paired ratio against gCastle (times give 3.3x).
    ratio = {name: r["ratio"] for name, r in row.items()}
    assert labels["Andrey"] == build.ratio_text(ratio["causal-learn"]) == "941x"
    assert labels["gCastle"] == build.ratio_text(ratio["causal-learn"] / ratio["gCastle"]) == "3.7x"
    assert labels["causal-learn"] == "1x"
    # Nothing under the bars: no times, no notes, no tags.
    assert "race-note" not in panel and " s<" not in panel and "race-tag" not in panel


def test_every_timed_package_needs_a_published_ratio():
    """No silent fallback to a time quotient for a package without a paired ratio."""
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    grasp = next(m for m in summary["methods"] if m["method"] == "GRaSP")
    row = next(r for r in grasp["rows"] if r["d"] == 400)
    other = copy.deepcopy(grasp)
    other["comparator"] = "slower package"
    other["rows"] = [dict(row, comparator_s=row["comparator_s"] * 10, ratio=None)]
    summary["methods"].append(other)
    with pytest.raises(ValueError, match="no published ratio against slower package"):
        build.speedup_panel(summary)


def test_without_has_every_panel_is_captioned_and_the_header_drops_the_size():
    css = build.speedup_css(build.load_summary(PLACEHOLDER, placeholder=True))
    fallback = css[css.index("@supports not selector(:has(a))") :]
    assert ".race-sizes-now { display: none; }" in fallback
    assert "content: attr(aria-label)" in fallback


def test_the_speedup_panel_needs_every_method():
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    summary["methods"] = [m for m in summary["methods"] if m["method"] != "GRaSP"]
    with pytest.raises(ValueError, match="GRaSP"):
        build.speedup_panel(summary)


def test_the_speedup_panel_plays_only_without_reduced_motion():
    script = (ROOT / "site" / "assets" / "figures.js").read_text(encoding="utf-8")
    block = script[script.index('document.querySelectorAll(".race--speedup")') :]
    block = block[: block.index("\n  }\n")]
    assert "motion.matches" in block and "continue" in block  # reduced motion: PC stays
    assert "clearInterval" in block and "focusin" in block and "pointerenter" in block
    # A click stops it too: clicking the method already shown fires no change event.
    assert 'addEventListener("click"' in block and 'addEventListener("change"' in block
    # Setting .checked fires no change event, so each step restarts the shown bars itself.
    assert "restart(race)" in block


def test_the_speedup_bars_grow_on_every_page_that_shows_them():
    figures = (ROOT / "site" / "assets" / "figures.css").read_text(encoding="utf-8")
    motion = figures[figures.index("@media (prefers-reduced-motion: no-preference)") :]
    rule = motion[motion.index(".race--speedup .race-fill") :]
    rule = rule[: rule.index("}")]
    assert "animation: race-fill" in rule and "var(--dur)" not in rule  # no clock to wait for
    # The homepage and the launch post share figures.css; the homepage's sheet keeps the growth.
    site = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    assert not re.search(r"race-fill\s*\{[^}]*animation:\s*none", site)
    summary = build.load_summary(PLACEHOLDER, placeholder=True)
    assert 'style="--i: 1"' in build.speedup_panel(summary)


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_published_summary_builds():
    page = render_homepage(build.load_summary(build.SUMMARY))
    assert "Not for release" not in page and "e+0" not in page


def test_one_header_on_every_page():
    # The homepage keeps pydata's navbar and draws no header of its own.
    template = (ROOT / "site" / "homepage.html").read_text()
    assert re.search(r"\{%-? block docs_navbar %\}\{\{ super\(\) \}\}\{% endblock", template)
    assert "<header" not in template
    # The navbar: the wordmark and the release, then the site's sections in order.
    conf = (ROOT / "docs" / "conf.py").read_text()
    assert '"navbar_start": ["site-logo", "site-version"]' in conf
    assert '"navbar_center": ["site-nav"]' in conf
    nav = (ROOT / "docs" / "_templates" / "site-nav.html").read_text()
    assert re.findall(r'\("([\w/]+)", "(\w+)"', nav) == [
        ("docs/index", "Docs"),
        ("docs/examples/index", "Examples"),
        ("docs/demos/index", "Demos"),
        ("blog/index", "Blog"),
        ("docs/faq", "FAQ"),
    ]


def test_the_wordmark_leads_to_the_homepage():
    # pathto('', 1) is the site root from any page; the homepage is the root's index.html.
    logo = (ROOT / "docs" / "_templates" / "site-logo.html").read_text()
    assert "{%- set home = pathto('', 1) %}" in logo and "href=\"{{ home or './' }}\"" in logo
    assert '"index": "homepage.html"' in (ROOT / "docs" / "conf.py").read_text()


def test_the_navbar_names_only_the_page_being_read_as_current():
    # A section's link is highlighted on each of its pages; aria-current goes only on a link to the
    # page itself.
    jinja2 = pytest.importorskip("jinja2")
    templates = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "docs" / "_templates"))
    nav = templates.get_template("site-nav.html")

    def marks(pagename):
        page = nav.render(pagename=pagename, _=str, pathto=lambda doc: doc)
        highlighted = re.findall(r'<li class="nav-item current active">\s*<a [^>]*>(\w+)<', page)
        return highlighted, re.findall(r'aria-current="page">([^<]+)<', page)

    assert marks("index") == ([], [])
    assert marks("docs/index") == (["Docs"], ["Docs"])
    assert marks("docs/guides/getting-started") == (["Docs"], [])
    assert marks("docs/examples/index") == (["Examples"], ["Examples"])
    assert marks("docs/examples/ges_quickstart") == (["Examples"], [])
    assert marks("docs/demos/index") == (["Demos"], ["Demos"])
    assert marks("docs/demos/pc-replay") == (["Demos"], [])
    assert marks("blog/index") == (["Blog"], ["Blog"])
    assert marks("blog/introducing-andrey") == (["Blog"], [])
    assert marks("docs/faq") == (["FAQ"], ["FAQ"])


def test_homepage_styles_are_scoped_to_the_homepage():
    # Every rule in site.css is under .andrey-home, so it cannot restyle the navbar or the docs.
    css = re.sub(r"/\*.*?\*/", "", (ROOT / "site" / "assets" / "site.css").read_text(), flags=re.S)
    # A keyframes block's steps (from, to) style nothing until a scoped rule names its animation.
    css = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    selectors = [
        selector.strip()
        for prelude in re.findall(r"([^{}]+)\{", css)
        if not prelude.strip().startswith("@")
        for selector in prelude.split(",")
    ]
    assert selectors
    # A theme's rule may open with its html attribute; what it styles is still under .andrey-home.
    selectors = [re.sub(r'^html\[data-theme="(dark|light)"\]\s+', "", s) for s in selectors]
    unscoped = [s for s in selectors if not s.startswith((".andrey-home", ":where(.andrey-home)"))]
    assert unscoped == []


def test_every_partial_mark_says_what_it_covers():
    notes = build.comparison_notes()
    assert all(mark == "part" or package != "Andrey" for _, package, mark, _ in notes)
    table = build.comparison_table()
    assert table.count('class="mark mark--') == len(build.COMPARISON) * len(build.PEERS)
    # Each partial mark carries its note as a tooltip and in its screen-reader text.
    assert table.count(" title=") == len(notes) and "compare-note-" not in table
    for _, _, _, note in notes:
        assert f"Partly: {build.escape(note)}</span>" in table
    bad = list(build.COMPARISON) + [("Methods", "Unexplained", ("yes", "part", "no", "no"), {})]
    original, build.COMPARISON = build.COMPARISON, tuple(bad)
    try:
        with pytest.raises(ValueError, match="in part"):
            build.comparison_notes()
    finally:
        build.COMPARISON = original


def test_interface_example_returns_three_graph_types():
    code, printed, figures = build.run_interface()
    assert code.startswith("for learn in") and "graphs[" not in code
    assert [line.split()[:2] for line in printed.splitlines()] == [
        ["pc", "cpdag"],
        ["fci", "pag"],
        ["direct_lingam", "dag"],
    ]
    assert figures.count("<figure") == 3 and "var(--andrey-" in figures
    # The two edges out of season get three answers: plain lines (PC), circles at both ends (FCI),
    # and arrows (DirectLiNGAM); everything downstream agrees. Marks: 1 tail, 2 arrow, 3 circle.
    graphs = build.run_script("interface.py")[2]["graphs"]
    downstream = [(1, 3, 1, 2), (2, 3, 1, 2), (3, 4, 1, 2)]
    expected = {
        "pc": [(0, 1, 1, 1), (0, 2, 1, 1), *downstream],
        "fci": [(0, 1, 3, 3), (0, 2, 3, 3), *downstream],
        "direct_lingam": [(0, 1, 1, 2), (0, 2, 1, 2), *downstream],
    }
    for name, edges in expected.items():
        assert sorted(map(tuple, graphs[name].to_edges().tolist())) == edges


@pytest.fixture(scope="module")
def agent_frames():
    return build.agent_frames()


def test_the_agent_session_is_the_command_lines_json(agent_frames):
    commands = [command for command, _, _ in agent_frames]
    assert commands == [build.AGENT_LIST, build.AGENT_HELP, build.AGENT_RUN]
    (_, listed, more), (_, alpha, _), (_, edges, _) = agent_frames
    # Every method is counted, the first shown in full; alpha is the registry's own parameter.
    assert len(listed) + more == len(list_specs())
    assert listed[0].split("\t") == ["pc", "cpdag", "supported"]
    param = get_spec("pc").params[0]
    assert (param.name, json.loads(alpha[0])["default"]) == ("alpha", param.default)
    assert [json.loads(line) for line in edges][-2:] == [
        ["p38", "pkc", "directed"],
        ["jnk", "pkc", "directed"],
    ]
    assert len(edges) == 8


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
def test_the_agent_sessions_filters_print_what_jq_prints(agent_frames, tmp_path):
    data = tmp_path / "sachs.csv"
    frame = runpy.run_path(str(ROOT / "site" / "example.py"), run_name="example")["df"]
    frame.to_csv(data, index=False)
    for command, lines, more in agent_frames:
        cli, query = command.split(" | jq ", 1)
        flag, _, program = query.partition(" ")
        args = [str(data) if a == "sachs.csv" else a for a in shlex.split(cli)[1:]]
        answer = json.dumps(build.run_cli(args))
        printed = subprocess.run(
            ["jq", flag, shlex.split(program)[0]], input=answer, capture_output=True, text=True
        ).stdout.splitlines()
        assert printed[: len(lines)] == lines and len(printed) == len(lines) + more, command


def test_the_homepage_shows_the_whole_agent_session(placeholder_page):
    start = placeholder_page.index('id="agents"')
    section = placeholder_page[start : placeholder_page.index("</section>", start)]
    # Every command and answer is in the page, so it reads without the script; Replay appears
    # only when the script can play it.
    assert section.count('<div class="frame">') == 3
    for command in (build.AGENT_LIST, build.AGENT_HELP, build.AGENT_RUN):
        assert html.escape(command) in section
    assert '<button class="term-replay" type="button" hidden>' in section
    assert section.count('class="agent-step"') == 3 and 'class="agent-graph"' in section
    # The terminal's bar has no title; Replay keeps the bar's right end.
    assert 'class="window-title"' not in section
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    replay = css[css.index(".andrey-home .term-replay {") :]
    assert "grid-column: 3" in replay[: replay.index("}")]


def test_the_footer_credits_abel_with_its_mark_and_name(placeholder_page):
    start = placeholder_page.index('<footer class="site-footer')
    footer = placeholder_page[start : placeholder_page.index("</footer>", start)]
    link = footer[footer.index('<a class="footer-abel" href="https://abel.ai/">') :]
    link = link[: link.index("</a>")]
    # One link holds the mark and the name; the mark is drawn in the text's color.
    assert link.endswith("Abel AI Lab") and "Built by" not in footer
    assert '<svg class="abel-mark" viewBox="0 0 726 460" aria-hidden="true"' in link
    assert 'fill="currentColor"' in link and "white" not in link


def test_the_agent_session_wraps_long_lines_inside_the_terminal(placeholder_page):
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    rule = css[css.index(".andrey-home .cmd,\n.andrey-home .ln {") :]
    rule = rule[: rule.index("}")]
    assert "white-space: pre-wrap" in rule and "overflow-wrap: anywhere" in rule
    # Typing reveals a command in place; growing its width would reflow it once it wraps.
    assert "agent-type" not in css and '<span class="cmd-text">' in placeholder_page
    script = (ROOT / "site" / "assets" / "figures.js").read_text(encoding="utf-8")
    assert 'rest.className = "untyped"' in script and ".untyped" in css
    # The example's printed output wraps the same way.
    output = css[css.index(".andrey-home pre.output {") :]
    assert "white-space: pre-wrap" in output[: output.index("}")]


def test_code_and_its_output_sit_in_windows_named_for_the_file(placeholder_page):
    for name in ("interface.py", "example.py"):
        assert placeholder_page.count(f'<span class="window-title">{name}</span>') == 2
    assert 'class="term-bar window-bar"' in placeholder_page
    # In the launch post a captioned code block is a window, its caption the title; the install
    # command's window has no title.
    post = (ROOT / "docs" / "blog" / "introducing-andrey.md").read_text(encoding="utf-8")
    quick_start = post.split("## Try Andrey", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r":caption: (.*)", quick_start) == ["example.py", "example.py"]
    blocks = re.findall(r"```\{code-block\} shell\n(?::[^\n]*\n)*\n(.*?)\n```", quick_start, re.S)
    assert "uv add andrey-core" in blocks


def test_the_claim_cards_lead_to_their_sections_in_page_order(placeholder_page):
    links = re.findall(r'<div class="pillar-more">.*?<a href="#([\w-]+)">', placeholder_page)
    assert links == ["speed", "accuracy", "graphs", "agents"]
    at = [placeholder_page.index(f'<section class="section wrap" id="{name}"') for name in links]
    assert at == sorted(at)
    # Then the reference sections: getting started, the methods, and the comparison.
    assert at[-1] < placeholder_page.index('<section class="section wrap" id="start"')


def test_the_homepage_keeps_every_note_at_its_foot(placeholder_page):
    page = placeholder_page
    for inline in ('class="race-note"', 'class="speed-notes"', 'class="compare-notes"'):
        assert inline not in page
    refs = re.findall(r'<a href="#fn-(\d+)" id="fn-ref-\1">', page)
    notes = re.findall(r'<li id="fn-(\d+)">', page)
    assert refs == notes == [str(n) for n in range(1, len(notes) + 1)]
    assert len(notes) == 2  # the headline's source, then the comparison's
    assert 'href="docs/benchmarks.html"' in page and 'href="blog/introducing-andrey.html"' in page


def test_the_accuracy_cards_show_every_scored_comparison():
    chart = build.accuracy_chart(SUMMARY)
    shown = chart.split('<div class="tc-copy" aria-hidden="true">')[0]
    # A card per speed chart, repeated once on the belt and hidden from assistive technology.
    assert shown.count('<figure class="tchart rtable">') == 3  # PC, FCI, DirectLiNGAM
    assert chart.count('<figure class="tchart rtable" aria-hidden="true">') == 3
    scores = re.findall(r'<span class="rt-score"><b>(\d+) of (\d+)</b>', shown)
    assert scores == [("1", "1"), ("4", "4"), ("0", "1")]
    assert (sum(int(a) for a, _ in scores), sum(int(b) for _, b in scores)) == (5, 6)
    labels = re.findall(r'<span class="rt-v">([^<]+)</span>', shown)
    assert labels == ["same", "same", "3% fewer", "&lt;1% fewer", "&lt;1% fewer", "7% more"]
    # Each count is in the row's tooltip; the bar is as long as the share of errors avoided.
    assert 'title="causal-learn, 50 variables: Andrey 30 errors, causal-learn 31"' in shown
    assert 'style="--f: 3.2%"' in shown
    # DirectLiNGAM at 200: lingam passed its time cap, so the card has no row for it.
    assert "200" not in re.findall(r'<span class="rt-d">([^<]*)</span>', shown)


def test_the_speed_section_opens_why_andrey_is_fast(placeholder_page):
    start = placeholder_page.index('id="speed"')
    section = placeholder_page[start : placeholder_page.index("</section>", start)]
    assert 'data-wf-open="why-fast"' in section
    assert '<dialog class="wf wf-dialog" id="why-fast"' in section


def test_link_preview_fonts_ship_with_their_license():
    # The preview cards render text with the site's Inter, which the OFL permits bundling with it.
    fonts = ROOT / "site" / "brand" / "fonts"
    names = sorted(p.name for p in fonts.glob("*.ttf"))
    assert names == ["Inter-Regular.ttf", "Inter-SemiBold.ttf"]
    assert "SIL Open Font License" in (fonts / "OFL.txt").read_text()
    conf = (ROOT / "docs" / "conf.py").read_text()
    assert 'HOME_PREVIEW = "assets/preview/home.jpg"' in conf
    assert '"og:image:type": "image/jpeg"' in conf
    assert '"twitter:card": "summary_large_image"' in conf


@pytest.fixture(scope="module")
def cards():
    """``site/cards.py``, which draws with Pillow: in the viz extra and the docs group, not dev."""
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location("site_cards", ROOT / "site" / "cards.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_card_shows_its_title_with_the_accent_word_and_no_version(cards):
    svg = cards.card_svg(cards.TAGLINE, cards.ACCENT)
    # A card opens the page it previews, and every page's navbar shows the version.
    assert andrey.__version__ not in svg
    assert " ".join(word for word, _ in _title_words(svg)) == cards.TAGLINE
    amber = [word for word, fill in _title_words(svg) if fill == cards.grain.AMBER]
    assert amber == [cards.ACCENT]
    plain = cards.card_svg(cards.TAGLINE)
    assert {fill for _, fill in _title_words(plain)} == {"#ffffff"}


SVG = "{http://www.w3.org/2000/svg}"


def _title_words(svg: str) -> list[tuple[str, str]]:
    """The title's words in reading order, each with its fill."""
    words = [t for t in ET.fromstring(svg).iter(f"{SVG}text") if t.get("font-weight") == "600"]
    words.sort(key=lambda t: (round(float(t.get("y"))), float(t.get("x"))))
    return [(t.text, t.get("fill")) for t in words]


def test_a_card_is_a_small_jpeg_the_size_platforms_crop_to(cards):
    import io

    from PIL import Image

    pytest.importorskip("resvg_py")  # the docs group's renderer; CI's extras jobs lack it
    data = cards.card_jpeg("Introducing Andrey")
    with Image.open(io.BytesIO(data)) as image:
        assert (image.format, image.size) == ("JPEG", (1200, 630))
    assert len(data) <= cards.MAX_BYTES == 300 * 1024


def test_a_long_card_title_keeps_every_word_or_fails(cards):
    title = "Learning causal graphs from observational data with hidden confounders"
    svg = cards.card_svg(title)
    assert " ".join(word for word, _ in _title_words(svg)) == title
    with pytest.raises(ValueError, match="needs more than 3 lines"):
        cards.card_svg("word " * 60)


def test_the_card_ground_is_the_same_on_every_build(cards):
    band = (cards.grain.Band(600, 300, 520, 95),)
    first, second = cards.grain.stipple(120, 63, band), cards.grain.stipple(120, 63, band)
    assert first.tobytes() == second.tobytes()


def test_the_homepage_has_no_latest_strip(placeholder_page):
    # The alpha banner links the launch post, and the header links the blog and the FAQ.
    assert 'class="latest' not in placeholder_page
    assert "latest" not in (ROOT / "site" / "homepage.html").read_text()


def test_the_hero_grain_is_small_and_the_same_on_every_build(cards):
    tiles = dict(cards.grain.hero_tiles())
    assert sum(len(t) for t in tiles.values()) <= 150 * 1024
    assert dict(cards.grain.hero_tiles()) == tiles
    css = (ROOT / "site" / "assets" / "site.css").read_text()
    assert all(f'url("{name}")' in css for name in tiles)
    assert "cards.grain.hero_tiles()" in (ROOT / "docs" / "conf.py").read_text()


def test_the_homepage_and_docs_css_name_no_colour(cards):
    # Every color is a generated token, so a palette edit reaches the hero and the chip too.
    sheets = ("site.css", "figures.css", "why-fast.css")
    for path in [*(f"site/assets/{name}" for name in sheets), "docs/_static/andrey.css"]:
        rules = re.sub(r"/\*.*?\*/", "", (ROOT / path).read_text(), flags=re.S)
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", rules), path
    tokens = cards.grain.ground_tokens()
    assert tokens["--andrey-ground-cobalt"] == andrey.viz.LIGHT.ramp[3]
    assert tokens["--andrey-ground-amber"] == andrey.viz.LIGHT.categorical[1]


def test_the_readme_banner_is_the_committed_rounded_svg_the_readme_links(cards):
    import base64
    import io

    import numpy as np
    from PIL import Image

    pytest.importorskip("resvg_py")  # the docs group's renderer; CI's extras jobs lack it

    def pixels(svg):
        data = base64.b64decode(re.search(r'href="data:image/jpeg;base64,([^"]+)"', svg)[1])
        with Image.open(io.BytesIO(data)) as image:
            assert (image.format, image.size) == ("JPEG", (cards.BANNER_W, cards.BANNER_H))
            return np.asarray(image.convert("RGB"), dtype=int)

    committed = (ROOT / ".github" / "readme" / "banner.svg").read_text()
    # One JPEG, clipped to the README chart's corner radius, under the cards' byte limit.
    assert f'rx="{cards.BANNER_RADIUS}"' in committed and committed.count("<image ") == 1
    assert len(committed.encode()) <= cards.MAX_BYTES
    # Matches a fresh draw. Another platform's renderer shifts a few pixels by a few levels; a
    # one-letter tagline edit shifts about 500 pixels by more than 64.
    moved = np.abs(pixels(committed) - pixels(cards.readme_banner())).max(axis=2)
    assert (moved > 64).sum() < 100, "run apps/build_launch.py --banner"
    readme = (ROOT / "README.md").read_text()
    header = readme[: readme.index("</p>")]
    # One tagline: the banner carries the name and the line, so no heading repeats a slogan.
    assert 'src=".github/readme/banner.svg"' in header
    assert 'alt="Andrey, a very fast causal discovery package"' in header
    assert "<picture" not in header and "<h1" not in readme
    assert cards.TAGLINE == "A very fast causal discovery package"


@pytest.mark.parametrize("kind", ["pair", "graph"])
def test_a_post_cover_is_a_small_jpeg_drawn_the_same_every_build(cards, kind):
    import io

    from PIL import Image

    pytest.importorskip("resvg_py")  # the docs group's renderer; CI's extras jobs lack it
    data = cards.cover_jpeg(kind)
    with Image.open(io.BytesIO(data)) as image:
        assert (image.format, image.size) == ("JPEG", (cards.COVER_W, cards.COVER_H))
    assert len(data) <= cards.MAX_BYTES
    assert cards.cover_jpeg(kind) == data
    assert cards.cover_alt(kind).strip()


def test_a_post_cover_is_named_in_front_matter_and_fails_on_a_bad_name(cards):
    with pytest.raises(ValueError, match="unknown cover"):
        cards.cover_jpeg("photo")
    post = (ROOT / "docs" / "blog" / "introducing-andrey.md").read_text()
    assert "cover: pair" in post
    assert 'cover_caption: "Andrey Kolmogorov and Andrey Markov."' in post


def test_the_launch_post_links_where_the_name_comes_from():
    faq = (ROOT / "docs" / "docs" / "faq.md").read_text()
    assert '### Where does the name "Andrey" come from?' in faq
    post = (ROOT / "docs" / "blog" / "introducing-andrey.md").read_text()
    assert "(../docs/faq.md#where-does-the-name-andrey-come-from)" in post


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_the_published_summary_gives_every_paired_ratio():
    """A ratio is withheld only where a time is missing, never for accuracy."""
    summary = build.load_summary(build.SUMMARY)
    assert (
        summary["headline"] == "Over 100x faster on PC, and faster on most other supported methods."
    )
    for method in summary["methods"]:
        for row in method["rows"]:
            timed = row["andrey_s"] is not None and row["comparator_s"] is not None
            assert (row["ratio"] is not None) == timed, (method["method"], method["comparator"])


def test_no_speed_cell_is_withheld_in_the_report_or_the_readme():
    for page in (ROOT / "docs" / "docs" / "benchmarks.md", ROOT / "README.md"):
        text = page.read_text(encoding="utf-8").lower()
        assert "no claim" not in text and "no speed claim" not in text, page.name


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_every_hand_written_copy_of_the_headline_is_the_summarys():
    """The docs index and the notebook repeat the headline; the live demo repeats the homepage's
    heading and the shorter claim under it. Each must match."""
    summary = build.load_summary(build.SUMMARY)
    for page in (
        ROOT / "docs" / "docs" / "index.md",
        ROOT / "examples" / "speed_on_your_machine.ipynb",
    ):
        assert summary["headline"] in page.read_text(encoding="utf-8"), page.name
    app = (ROOT / "apps" / "live-discovery" / "app.py").read_text(encoding="utf-8")
    assert "<h1>Causal discovery, fast.</h1>" in app and build.home_claim(summary) in app
    space = (ROOT / "apps" / "live-discovery" / "README.md").read_text(encoding="utf-8")
    assert f"Causal discovery, fast. {build.home_claim(summary)}" in space
    assert 'title="Andrey | Causal discovery, fast"' in app


def test_the_citation_title_is_the_tagline():
    # The package's own line, as on the README banner; the homepage's headline is campaign wording.
    cff = (ROOT / "CITATION.cff").read_text()
    assert 'title: "Andrey: A very fast causal discovery package"' in cff
    for page in (ROOT / "README.md", ROOT / "docs" / "docs" / "faq.md"):
        text = page.read_text()
        assert "title = {Andrey: A very fast causal discovery package}," in text, page.name
