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
from collections import Counter
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
    pytest.importorskip("PIL")  # the sweep traces the art; Pillow is the viz extra's, not dev's
    loader = jinja2.ChoiceLoader(
        [
            jinja2.DictLoader(
                {
                    "layout.html": LAYOUT,
                    "sections/sidebar-primary.html": "",
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


def placeholder_summaries() -> list[dict]:
    """Every release's summary, the placeholder standing in for the first, as the preview loads
    them."""
    return build.release_summaries(build.load_summary(PLACEHOLDER, placeholder=True))


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
    assert build.speed_claim([summary]) == (
        "Over 100x faster on PC",
        "faster on the other supported methods",
    )
    # One of the two not faster at its largest size: no majority, so the claim stops at PC.
    _speed_up(summary, "FCI", 0.98)
    assert build.speed_claim([summary]) == ("Over 100x faster on PC", "")
    # A ratio that is exactly a power of ten is not "over" it.
    _speed_up(summary, "PC", 100.0)
    assert build.speed_claim([summary])[0] == "At least 100x faster on PC"


def test_the_published_claim_says_most_when_one_method_is_not_faster():
    summary = copy.deepcopy(SUMMARY)
    summary["methods"].append(
        {**summary["methods"][1], "method": "GES", "rows": [_row(150, 1.0, 4.3, 4.3, 5, 5, 150)]}
    )
    _speed_up(summary, "FCI", 0.98)
    assert build.speed_claim([summary])[1] == "faster on most other supported methods"


def test_only_supported_methods_count_toward_the_claim(monkeypatch):
    summary = copy.deepcopy(SUMMARY)
    _speed_up(summary, "FCI", 0.98)
    assert build.speed_claim([summary])[1] == ""
    # FCI, the one method not faster, is experimental here: the rest are all faster.
    specs = [
        s.model_copy(update={"status": "experimental"}) if s.name == "fci" else s
        for s in list_specs()
    ]
    monkeypatch.setattr(build, "list_specs", lambda: specs)
    assert build.speed_claim([summary])[1] == "faster on the other supported methods"
    assert "FCI" not in build.speed_chart([summary])


def test_notes_cover_every_cell_without_a_speedup():
    notes = build.exceptions(SUMMARY)
    assert (
        "FCI is slower than causal-learn at d = 20: 0.62x, 0.061 s compared with 0.038 s." in notes
    )
    assert "lingam did not finish DirectLiNGAM within the time cap at d = 200." in notes
    # Accuracy never withholds a speed ratio, so no note speaks of it.
    assert not any("empty graph" in note or "speed claim" in note for note in notes)


def test_charts_cover_only_comparisons_with_a_published_ratio():
    chart = build.speed_chart([SUMMARY])
    # One chart per method with a paired ratio, each titled with its largest one. The belt repeats
    # them once, hidden from assistive technology, so the loop is seamless.
    assert chart.count('<figure class="tchart">') == 3  # PC, FCI, DirectLiNGAM
    assert chart.count('<figure class="tchart" aria-hidden="true">') == 3
    ups = re.findall(r'class="tchart-up"><b>([^<]+)</b>', chart)
    # Each title rounds its largest ratio (941.05x, 8.9x, 3.59x) down, to one digit or below 10 to
    # a tenth, and marks it "+". FCI's largest is at d = 400, where both graphs are worse than the
    # empty graph: a speed ratio does not depend on accuracy.
    assert ups == ["900x+", "8.9x+", "3.5x+"] * 2
    # Sizes without a ratio are plotted hollow: DirectLiNGAM 200 (Andrey only; lingam passed its
    # time cap).
    assert chart.count("is-open") == 1 * 2
    assert "ICA-LiNGAM" not in chart and "ICA-LiNGAM" not in build.accuracy_chart([SUMMARY])
    # A comparison the charts skip is in the table, not in the notes.
    assert not any("ICA-LiNGAM" in note for note in build.exceptions(SUMMARY))
    assert "ICA-LiNGAM" in build.speed_table(SUMMARY)


def test_a_chart_joins_only_comparisons_from_the_same_runs():
    # PC against gCastle on the causal-learn runs with one more size, against causal-learn at a
    # size no other comparison measures, and against pcalg on runs of its own.
    runs = [_row(20, 0.1, 1.0, 10.0, 5, 6, 20), _row(50, 0.3, 4.0, 13.3, 12, 13, 50)]
    larger = _row(100, 0.9, 20.0, 22.2, 30, 31, 100)
    summary = {
        **SUMMARY,
        "methods": [
            _method("PC", "constraint-based", "causal-learn", runs),
            _method("PC", "constraint-based", "gCastle", [*runs, larger]),
            _method("PC", "constraint-based", "causal-learn", [_row(200, 2, 90, 45, 60, 62, 200)]),
            _method("PC", "constraint-based", "pcalg", [_row(50, 0.35, 2.0, 5.7, 12, 12, 50)]),
        ],
    }
    groups = build.time_groups([summary])
    assert [[m["comparator"] for m in methods] for _, methods in groups] == [
        ["causal-learn", "gCastle"],
        ["causal-learn"],
        ["pcalg"],
    ]
    # Andrey's line covers every size a chart's comparisons time.
    for name, methods in groups:
        chart = build.time_chart(name, methods, (0.01, 100))
        sizes = {r["d"] for m in methods for r in m["rows"]}
        assert chart.count('class="tc-point tc-andrey') == len(sizes)
    # The featured rule reads PC over the sizes its first chart's packages share.
    name, methods = build.default_settings([summary])["pc"]
    assert (name, sorted(build.shared_rows(methods))) == ("PC", [20, 50])


def test_a_setting_shown_under_its_method_name_comes_from_its_first_chart():
    # ICP's default test in two runs: first in the method's claim, then outside it.
    claimed = [_row(5, 0.004, 0.04, 10.0, 1, 1, 2), _row(10, 0.013, 1.2, 92.3, 1, 1, 3)]
    outside = [_row(5, 0.003, 0.04, 13.3, 1, 1, 2), _row(10, 0.008, 1.1, 137.5, 1, 1, 3)]
    summary = {
        **SUMMARY,
        "methods": [
            {**_method("ICP, approximate test", "constraint", "R", claimed, "icp_appendix_g"),
             "solution": "andrey.icp.approximate"},
            {**_method("ICP, approximate test", "constraint", "R", outside,
                       "icp_appendix_g_target"), "solution": "andrey.icp.approximate"},
        ],
    }  # fmt: skip
    assert len(build.time_groups([summary])) == 2
    name, methods = build.default_settings([summary])["icp"]
    assert name == "ICP"
    shown = build.shared_rows(methods)
    assert [(d, rows[0]["andrey_s"]) for d, rows in shown.items()] == [
        (r["d"], r["andrey_s"]) for r in claimed
    ]


def test_a_featured_row_takes_the_largest_sample_size_at_each_size():
    # PC at two sample sizes per graph size, the larger listed first.
    rows = [
        dict(_row(20, 0.3, 3.0, 10.0, 5, 6, 20), n=5000),
        dict(_row(20, 0.1, 0.5, 5.0, 7, 7, 20), n=1000),
        dict(_row(50, 0.9, 12.0, 13.3, 12, 13, 50), n=5000),
        dict(_row(50, 0.4, 2.0, 5.0, 15, 15, 50), n=1000),
    ]
    summary = {**SUMMARY, "methods": [_method("PC", "constraint-based", "causal-learn", rows)]}
    _, methods = build.default_settings([summary])["pc"]
    larger = [r for r in rows if r["n"] == 5000]
    shown = build.shared_rows(methods)
    for who in ("andrey", "comparator"):
        assert [(d, rs[0][f"{who}_s"]) for d, rs in shown.items()] == [
            (r["d"], r[f"{who}_s"]) for r in larger
        ]
    # Every published comparison that repeats a size records each row's sample size.
    for m in build.claimed_comparisons(placeholder_summaries()):
        repeated = {d for d, k in Counter(r["d"] for r in m["rows"]).items() if k > 1}
        assert all(r["n"] is not None for r in m["rows"] if r["d"] in repeated), m["method"]


def test_every_chart_draws_its_comparisons():
    summaries = placeholder_summaries()
    groups = build.time_groups(summaries)
    assert sum(len(methods) for _, methods in groups) == len(build.claimed_comparisons(summaries))
    for name, methods in groups:
        times = []
        for m in methods:
            times.append({})
            for r in m["rows"]:
                times[-1].setdefault(r["d"], Counter())[r["andrey_s"]] += 1
        # A chart's comparisons share a size, and Andrey's times agree wherever two share one.
        assert set(times[0]).intersection(*times[1:]), name
        for d in {d for t in times for d in t}:
            seen = [t[d] for t in times if d in t]
            assert all(s == seen[0] for s in seen), (name, d)
        # The chart draws every Andrey time its comparisons record.
        drawn = build.andrey_rows(methods)
        assert {(r["d"], r["andrey_s"]) for m in methods for r in m["rows"]} <= {
            (r["d"], r["andrey_s"]) for r in drawn
        }
        chart = build.time_chart(name, methods, (1e-4, 1e5))
        assert chart.count('class="tc-point tc-andrey') == sum(
            r["andrey_s"] is not None for r in drawn
        )


def test_errors_compare_as_a_share_rounded_against_andrey():
    exact = {"d": 50, "andrey_s": 1.0, "comparator_s": 2.0, "ratio": 2.0, "at_least": False}
    exact |= {"andrey_errors": 0, "comparator_errors": 0}
    rows = [
        exact,
        dict(exact, d=100, andrey_errors=2, comparator_errors=0),  # no share of none: a count
        dict(exact, d=200, andrey_errors=30, comparator_errors=31),  # 3.2% fewer rounds down
        dict(exact, d=400, andrey_errors=450, comparator_errors=452),  # 0.4% fewer
        dict(exact, d=800, andrey_errors=6.7, comparator_errors=6.3),  # 6.3% more rounds up
        dict(exact, d=1600, andrey_errors=12, comparator_errors=None),  # past the time cap
    ]
    compared = build.error_comparisons([{"comparator": "other", "rows": rows}])
    assert [(c["d"], c["kind"], c["label"]) for c in compared] == [
        (50, "same", "same"),
        (100, "more", "2 more"),
        (200, "fewer", "3% fewer"),
        (400, "fewer", "<1% fewer"),
        (800, "more", "7% more"),
    ]


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
    pytest.importorskip("PIL")  # the context traces the art's sweep
    marked = build.homepage_context(summary, placeholder=True)
    assert 'content="noindex"' in marked["robots"] and "Not for release" in marked["banner"]
    unmarked = build.homepage_context(summary)
    assert unmarked["robots"] == unmarked["banner"] == ""


def test_page_shows_no_internal_locations(placeholder_page):
    summaries = placeholder_summaries()
    page = placeholder_page
    assert not re.search(r"/scratch/|/home/", page, re.IGNORECASE)
    # A summary's notes on its runs stay on the release cards.
    for summary in summaries:
        for method in summary["methods"]:
            for text in [method.get("timing_note"), *(r.get("note") for r in method["rows"])]:
                assert not text or text not in page


def test_the_example_fits_pc_and_draws_in_the_palette():
    source, _, svg = build.run_example()
    assert "andrey.pc(df)" in source
    assert svg.startswith("<svg") and "var(--andrey-" in svg


def test_the_example_copy_counts_the_links_pc_finds_in_the_known_network(placeholder_page):
    from andrey.data import load_dataset

    found = build.run_script("example.py")[2]["out"].structure.oriented_edges()
    known = load_dataset("sachs").graph.oriented_edges()
    found, known = ({frozenset(edge[:2]) for edge in edges} for edges in (found, known))
    assert found <= known
    claim = (
        f"PC finds {len(found)} of the {len(known)} links in the known signaling network, "
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


def test_wordmark_takes_the_palette_ink_and_its_a_one_amber_stroke():
    # The word uses the palette's ink, with the A's right stroke in the art's amber. The icon is
    # that A alone, white on the art's navy, so it reads on a light or a dark browser bar.
    pytest.importorskip("PIL")  # site/grain.py, which holds the navy, draws with Pillow
    amber = andrey.viz.LIGHT.categorical[1]
    for name, palette in (("light", andrey.viz.LIGHT), ("dark", andrey.viz.DARK)):
        svg = (ROOT / "site" / "brand" / f"wordmark-{name}.svg").read_text()
        assert re.findall(r'fill="([^"]+)"', svg) == [palette.ink, amber]
    icon = (ROOT / "site" / "brand" / "favicon.svg").read_text()
    navy = runpy.run_path(str(ROOT / "site" / "grain.py"))["NAVY"]
    assert re.findall(r'fill="([^"]+)"', icon) == [navy, andrey.viz.LIGHT.surface, amber]
    assert "prefers-color-scheme" not in icon


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


def test_placeholder_numbers_are_refused_wherever_the_file_sits(tmp_path, monkeypatch):
    published = tmp_path / "published"
    published.mkdir()
    copied = published / "summary.json"
    copied.write_text(PLACEHOLDER.read_text())
    monkeypatch.setattr(build, "PUBLISHED", published)
    with pytest.raises(ValueError, match="placeholder numbers"):
        build.load_summary(copied)


def test_the_headline_is_one_sentence_with_no_number(placeholder_page):
    found = re.search(r'<h1 id="headline">(.*?)</h1>', placeholder_page, re.S)[1]
    text = re.sub(r"<[^>]+>", "", found).strip()
    assert re.fullmatch(r"[^.!?]+[.!?]", text) and not re.search(r"\d", text), text


def test_the_page_title_names_what_people_search_for(placeholder_page):
    # The title includes the product name and search terms within the length limit.
    assert f"<title>{build.SEARCH_TITLE}</title>" in placeholder_page
    assert len(build.SEARCH_TITLE) <= 60
    for words in ("Andrey", "fast", "causal discovery", "Python", "agents"):
        assert words in build.SEARCH_TITLE


# The registry as it is, and without one method the description does not name: the count of "more"
# methods follows the registry.
@pytest.mark.parametrize("dropped", [None, "gin"])
def test_the_description_states_the_speed_claim_and_the_method_count(monkeypatch, dropped):
    summaries = placeholder_summaries()
    specs = [s for s in list_specs() if s.name != dropped]
    monkeypatch.setattr(build, "list_specs", lambda: specs)
    description = build.page_description(summaries)
    assert description.endswith(build.CLAIM)
    assert f"with PC, GES, FCI and {len(specs) - 3} more methods" in description
    # Search results cut a description at about 160 characters.
    assert len(description) <= 160


@pytest.mark.parametrize("name", ["pc", "ges", "fci"])
def test_the_description_names_only_registered_methods(monkeypatch, name):
    summaries = placeholder_summaries()
    monkeypatch.setattr(build, "list_specs", lambda: [s for s in list_specs() if s.name != name])
    with pytest.raises(ValueError, match=f"unregistered methods: {name.upper()}$"):
        build.page_description(summaries)


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


# ABlog's All Posts, years, archive, tags, and authors have a catalog or a collection and no
# document; a post or an index page has a document.
@pytest.mark.parametrize(
    ("context", "has_document", "generated"),
    [
        ({"catalog": []}, False, True),
        ({"collection": []}, False, True),
        ({"catalog": []}, True, False),
        ({}, False, False),
    ],
)
def test_a_page_ablog_generates_has_a_catalog_or_collection_and_no_document(
    conf, context, has_document, generated
):
    from docutils.utils import new_document

    doctree = new_document("page") if has_document else None
    assert conf["_blog_collection"](context, doctree) is generated


def test_the_theme_stylesheet_keeps_only_the_icons_a_page_names(conf, tmp_path):
    styles = tmp_path / "_static" / "styles"
    styles.mkdir(parents=True)
    (styles / "pydata-sphinx-theme.css").write_text(
        '.fa-lg{font-size:1.25em}.fa-house,.fa-home{--fa:"\\f015"}.fa-user{--fa:"\\f007"}'
        '.fa-sun{--fa:"\\f185"}'
    )
    (tmp_path / "page.html").write_text('<i class="fa-solid fa-home"></i>')
    (tmp_path / "theme.js").write_text('icon.className = "fa-solid fa-sun"')
    # Font Awesome's own script names every icon, so it does not count.
    (tmp_path / "fontawesome.js").write_text("fa-user")
    app = types.SimpleNamespace(outdir=str(tmp_path), builder=types.SimpleNamespace(name="html"))
    conf["_trim_icon_rules"](app, None)
    css = (styles / "pydata-sphinx-theme.css").read_text()
    assert css == '.fa-lg{font-size:1.25em}.fa-house,.fa-home{--fa:"\\f015"}.fa-sun{--fa:"\\f185"}'


def test_every_intersphinx_inventory_falls_back_on_a_stored_copy(conf):
    from sphinx.util.inventory import InventoryFile

    # Without a timeout, a site that never answers holds the build forever.
    assert conf["intersphinx_timeout"] > 0
    # A download kept from an earlier build would otherwise stand in for the site on a rebuild.
    assert conf["intersphinx_cache_limit"] == 0
    for name, (site, locations) in conf["intersphinx_mapping"].items():
        site_first, stored = locations
        assert site_first is None, name  # None is the site's own objects.inv
        assert "://" not in stored, name
        raw = (ROOT / "docs" / stored).read_bytes()
        assert InventoryFile.loads(raw, uri=site).data, name


def _remeasured() -> tuple[dict, dict]:
    """``SUMMARY`` with DirectLiNGAM also measured against causal-learn, and a later summary that
    measures DirectLiNGAM against lingam again."""
    earlier = copy.deepcopy(SUMMARY)
    earlier["methods"].append(
        _method(
            "DirectLiNGAM",
            "functional",
            "causal-learn",
            [_row(100, 8.4, 57.8, 6.87, 7, 7, 99)],
            regime="lingam_sf",
        )
    )
    later = {
        "generated": "2026-10-15",
        "methods": [
            _method(
                "DirectLiNGAM",
                "functional",
                "lingam",
                [_row(100, 10.4, 110.2, 10.6, 6.3, 6.3, 99)],
                regime="lingam_sf",
            )
        ],
    }
    return earlier, later


def test_a_remeasured_comparison_shows_only_the_newest_summarys_rows():
    earlier, later = _remeasured()
    for summaries in ([earlier, later], [later, earlier]):
        kept = build.newest(copy.deepcopy(summaries))
        assert len(kept) == len(summaries)
        found = build.supported(kept)
        pairs = [(m["spec"], m["comparator"]) for m in found]
        assert len(pairs) == len(set(pairs))
        lingam = next(
            m for m in found if (m["spec"], m["comparator"]) == ("direct_lingam", "lingam")
        )
        assert lingam["rows"] == build.comparisons(later)[0]["rows"]
        # The earlier summary keeps every comparison the later one does not measure.
        assert ("direct_lingam", "causal-learn") in pairs
        assert {"PC", "FCI", "ICA-LiNGAM"} <= {m["method"] for m in found}


def test_two_summaries_of_one_day_cannot_both_measure_a_comparison():
    earlier, later = _remeasured()
    later["generated"] = earlier["generated"]
    with pytest.raises(ValueError, match="two summaries"):
        build.newest([earlier, later])


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_each_published_comparison_comes_from_one_summary():
    summaries = build.release_summaries(build.load_summary(build.SUMMARY))
    shown = [
        (s["generated"], m["method"], m["comparator"])
        for s in summaries
        for m in build.comparisons(s)
    ]
    dates = {}
    for generated, *pair in shown:
        dates.setdefault(tuple(pair), set()).add(generated)
    assert all(len(found) == 1 for found in dates.values())
    # No published summary measures a comparison later than the one shown.
    every = [build.load_summary(build.SUMMARY)] + [
        build.load_summary(build.PUBLISHED / f / "summary.json") for f in build.RELEASES[1:]
    ]
    for summary in every:
        for m in summary["methods"]:
            (kept,) = dates[(m["method"], m["comparator"])]
            assert kept >= summary["generated"]


def _lower_bound(summaries, name, bound):
    """Flag ``name``'s largest row with a ratio, in every comparison, as a lower bound."""
    for summary in summaries:
        for method in summary["methods"]:
            if method["method"] == name:
                row = max((r for r in method["rows"] if r["ratio"]), key=lambda r: r["d"])
                row.update(ratio_is_lower_bound=True, ratio_lower_bound=bound)


def test_a_lower_bound_prints_as_one():
    summaries = placeholder_summaries()
    # FCI's largest size; its largest ratio is at a smaller one, so the bound tops its chart too.
    _lower_bound(summaries, "FCI", 20.0)
    assert "<b>20x+</b>" in build.speed_chart(summaries)


def test_the_speedup_bars_grow_on_every_page_that_shows_them():
    figures = (ROOT / "site" / "assets" / "figures.css").read_text(encoding="utf-8")
    motion = figures[figures.index("@media (prefers-reduced-motion: no-preference)") :]
    rule = motion[motion.index(".race--speedup .race-fill") :]
    rule = rule[: rule.index("}")]
    assert "animation: race-fill" in rule and "var(--dur)" not in rule  # no clock to wait for


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_published_summary_builds():
    page = render_homepage(build.load_summary(build.SUMMARY))
    assert "Not for release" not in page and "e+0" not in page


def test_the_navbar_names_only_the_page_being_read_as_current():
    # A section's link is highlighted on each of its pages; aria-current goes only on a link to the
    # page itself.
    jinja2 = pytest.importorskip("jinja2")
    templates = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "docs" / "_templates"))
    nav = templates.get_template("site-nav.html")

    def marks(pagename):
        page = nav.render(pagename=pagename, _=str, pathto=lambda doc: doc)
        page = re.sub(r"<svg .*?</svg>", "", page)
        highlighted = re.findall(r'<li class="nav-item current active">\s*<a [^>]*>([^<]+)<', page)
        return highlighted, re.findall(r'aria-current="page">([^<]+)<', page)

    assert marks("index") == ([], [])
    assert marks("docs/index") == (["Docs"], ["Docs"])
    assert marks("docs/guides/getting-started") == (["Docs"], [])
    assert marks("docs/examples/index") == (["Examples"], ["Examples"])
    assert marks("docs/examples/quick_start") == (["Examples"], [])
    assert marks("docs/examples/pc-replay") == (["Examples"], [])
    assert marks("blog/index") == (["Blog"], ["Blog"])
    assert marks("blog/20261008-introducing-andrey") == (["Blog"], [])
    assert marks("docs/faq") == (["FAQ"], ["FAQ"])
    assert marks("docs/benchmarks") == (["Release cards"], ["Release cards"])
    assert marks("docs/benchmarks/0.4.0") == (["Release cards"], [])


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


def test_the_homepage_shows_the_whole_agent_session(placeholder_page, agent_frames):
    band = placeholder_page[placeholder_page.index('id="agents"') :]
    band = band[: band.index("</section>")]
    # Every command and the first lines it prints are in the page, so it reads without script.
    for command, printed, _ in agent_frames:
        assert html.escape(f"$ {command}") in band
        assert all(html.escape(line) in band for line in printed[:3])


def test_abels_mark_takes_the_text_color():
    mark = build.abel_mark()
    assert 'fill="currentColor"' in mark and "white" not in mark


def test_the_agent_session_wraps_long_lines_inside_the_terminal():
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    for rule in (".andrey-home .term {", ".andrey-home pre.output {"):
        body = css[css.index(rule) :]
        body = body[: body.index("}")]
        assert "white-space: pre-wrap" in body and "overflow-wrap: anywhere" in body, rule


def test_code_and_its_output_sit_in_windows_named_for_the_file(placeholder_page):
    assert placeholder_page.count('<span class="window-title">example.py</span>') == 2
    # In the launch post the install command has a window of its own.
    post = (ROOT / "docs" / "blog" / "20261008-introducing-andrey.md").read_text(encoding="utf-8")
    quick_start = post.split("## Try Andrey", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```\{code-block\} shell\n(?::[^\n]*\n)*\n(.*?)\n```", quick_start, re.S)
    assert "uv add andrey-core" in blocks


@pytest.mark.parametrize(
    ("ratio", "shown"), [(941, 900), (1470, 1000), (16, 10), (6.9, 6), (2.0, 2), (99.9, 90)]
)
def test_a_family_number_rounds_down_to_one_digit(ratio, shown):
    assert build.round_down(ratio) == shown


def test_the_agent_pill_copies_the_agent_prompt_the_readme_and_the_faq_give(placeholder_page):
    assert f'data-prompt="{html.escape(build.AGENT_PROMPT)}"' in placeholder_page
    for name in build.AGENTS.values():  # each agent's mark, named
        assert f"<title>{html.escape(name)}</title>" in placeholder_page
    for page in (ROOT / "README.md", ROOT / "docs" / "docs" / "faq.md"):
        assert build.AGENT_PROMPT in " ".join(page.read_text().split()), page.name


def test_a_row_its_release_card_does_not_count_better_is_never_a_win():
    # The README's accuracy line counts a comparison only where its card agrees.
    def cut(row):
        return -build.accuracy_change(row)

    best = max(build.accuracy_rows([SUMMARY]), key=cut)

    def judged(verdict):
        methods = [
            {
                **m,
                "rows": [
                    {**r, "card_verdict": verdict(i) if callable(verdict) else verdict}
                    for i, r in enumerate(m["rows"])
                ],
            }
            if m["method"] == best["name"]
            else m
            for m in SUMMARY["methods"]
        ]
        rows = build.accuracy_rows([{**SUMMARY, "methods": methods}])
        return [r["name"] for r in rows if cut(r) > 0]

    # A comparison with a row the card counts better and none worse stays a win; one of ties,
    # of rows the card did not judge, or with a worse row, drops out.
    assert best["name"] in judged("better")
    assert best["name"] not in judged("tie")
    assert best["name"] not in judged(None)
    assert best["name"] not in judged(lambda i: "worse" if i == 0 else "better")


def test_the_quality_intro_states_each_recorded_outcome_and_names_every_setting_with_more():
    def cells(outcomes):
        return {k: {(m["method"], r["d"]) for m, r in v} for k, v in outcomes.items()}

    # Without card verdicts, each setting the charts show compares the two error counts. ICA-LiNGAM
    # has no ratio, so no chart, and DirectLiNGAM's other package has no graph at 200 variables.
    judged, counted = build.recorded_outcomes([SUMMARY])
    assert not any(judged.values())
    assert cells(counted) == {
        "same": {("PC", 400), ("FCI", 20)},
        "fewer": {("FCI", 50), ("FCI", 400), ("FCI", 800)},
        "more": {("DirectLiNGAM", 100)},
    }
    intro = build.quality_intro([SUMMARY])
    assert "DirectLiNGAM compared with lingam" in intro and "FCI compared" not in intro
    assert "as accurate" not in intro.lower()

    # Where a card judges, its verdict decides, not the counts; a setting it did not judge is
    # left out.
    verdict = {("PC", 400): "better", ("FCI", 50): "worse", ("FCI", 800): None}
    summary = {
        **SUMMARY,
        "methods": [
            {
                **m,
                "rows": [
                    {**r, "card_verdict": verdict.get((m["method"], r["d"]), "tie")}
                    for r in m["rows"]
                ],
            }
            for m in SUMMARY["methods"]
        ],
    }
    judged, counted = build.recorded_outcomes([summary])
    assert not any(counted.values())
    assert cells(judged) == {
        "tie": {("FCI", 20), ("FCI", 400), ("DirectLiNGAM", 100), ("DirectLiNGAM", 200)},
        "better": {("PC", 400)},
        "worse": {("FCI", 50)},
    }
    intro = build.quality_intro([summary])
    assert "FCI compared with causal-learn" in intro and "50 variables" in intro
    assert "DirectLiNGAM" not in intro


def _timed_row(d: int, n: int, ratio: float, hardware: str | None = None) -> dict:
    return {"d": d, "n": n, "ratio": ratio, "hardware": hardware}


def test_a_grid_comparison_shows_its_smallest_sample_size_and_its_slowest_cpu():
    rows = [
        _timed_row(5, 500, 10.0, "Intel Xeon Gold 6542Y, 4 cores"),
        _timed_row(5, 1000, 9.0, "Intel Xeon Gold 6542Y, 4 cores"),
        _timed_row(10, 500, 2.4, "AMD EPYC 7713, 4 cores"),
        _timed_row(10, 1000, 2.5, "AMD EPYC 7713, 4 cores"),
        _timed_row(20, 500, None),
    ]
    grid = {"lane": "nonlinear", "rows": rows}
    assert [(r["d"], r["n"]) for r in build.shown_rows(grid)] == [(10, 500), (20, 500)]
    # Other lanes retain every measured size, regardless of sample size.
    sweep = [_timed_row(5, 100, 3.0), _timed_row(5, 200, 4.0)]
    assert build.shown_rows({"lane": "other", "rows": sweep}) == sweep


def test_a_second_machines_timings_and_an_unshown_regime_are_not_counted(tmp_path):
    summary = json.loads(build.SUMMARY.read_text(encoding="utf-8"))
    base = summary["methods"][0]
    extra = [
        {**base, "comparison": "second-machine timing"},
        {**base, "lane": "nonlinear", "regime": "anm_er"},
    ]
    written = tmp_path / "summary.json"
    written.write_text(json.dumps({**summary, "methods": [*summary["methods"], *extra]}))
    loaded = build.load_summary(written, placeholder=True)
    assert len(loaded["methods"]) == len(summary["methods"])


def test_every_raster_is_srgb_with_no_profile_and_meets_the_css_ground(cards):
    import numpy as np
    from PIL import Image

    # Browsers color-manage a JPEG with an embedded profile and leave one without it as sRGB, so
    # every picture carries none and all render in the same space.
    rasters = [
        *(ROOT / "site" / "brand").rglob("*.jpg"),
        *(ROOT / "docs" / "_static").rglob("*.jpg"),
    ]
    assert len(rasters) > 15
    for path in rasters:
        with Image.open(path) as image:
            assert image.mode == "RGB" and not image.info.get("icc_profile"), path.name
    # The page's ground behind and around the art is the art's own median color.
    ground = cards._rgb(cards.ground_tokens()["--andrey-ground-art"])
    for path in [*cards.ART.values(), *cards.CENTERED.values()]:
        with Image.open(path) as image:
            median = np.median(np.asarray(image.convert("RGB"), float).reshape(-1, 3), axis=0)
        assert np.abs(median - ground).max() < 16, path.name


def test_the_accuracy_cards_show_every_scored_comparison():
    chart = build.accuracy_chart([SUMMARY])
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


def test_link_preview_fonts_ship_with_their_license():
    # The preview cards render text with the site's Inter, which the OFL permits bundling with it.
    fonts = ROOT / "site" / "brand" / "fonts"
    names = sorted(p.name for p in fonts.glob("*.ttf"))
    assert names == ["Inter-Regular.ttf", "Inter-SemiBold.ttf"]
    assert "SIL Open Font License" in (fonts / "OFL.txt").read_text()


@pytest.fixture(scope="module")
def cards():
    """``site/cards.py``, which draws with Pillow: in the viz extra and the docs group, not dev."""
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location("site_cards", ROOT / "site" / "cards.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_social_preview_is_a_jpeg_at_githubs_size_that_link_previews_still_show(cards):
    import io

    from PIL import Image

    pytest.importorskip("resvg_py")  # the docs group's renderer; CI's extras jobs lack it
    data = cards.social_jpeg()
    with Image.open(io.BytesIO(data)) as image:
        assert (image.format, image.size) == ("JPEG", (1280, 640))
    assert len(data) <= cards.MAX_BYTES == 300 * 1024


def test_the_card_ground_is_the_same_on_every_build(cards):
    band = (cards.grain.Band(600, 300, 520, 95),)
    first, second = cards.grain.stipple(120, 63, band), cards.grain.stipple(120, 63, band)
    assert first.tobytes() == second.tobytes()


def test_the_newest_release_post_is_the_latest_version(conf):
    newest = conf["_newest_post"]
    posts = [
        ("blog/20261008-introducing-andrey", "2026-10-08", "0.1.0"),
        ("blog/20261008-why-andrey-is-fast", "2026-10-08", "0.1.0"),
        ("blog/20261012-andrey-0-2-0", "2026-10-12", "0.2.0"),
    ]
    assert newest(posts) == ("blog/20261012-andrey-0-2-0", "0.2.0")
    # A later version breaks a tie in date, and the later file name a tie in both.
    assert newest([*posts, ("blog/andrey-0-2-1", "2026-10-12", "0.2.1")])[1] == "0.2.1"
    assert newest(posts[:2]) == ("blog/20261008-why-andrey-is-fast", "0.1.0")
    moved = [("blog/20261029-andrey-0-11-0", "2026-11-11", "0.17.0"), *posts]
    moved.append(("blog/20261103-andrey-0-13-0", "2026-11-11", "0.17.0"))
    assert newest(moved)[0] == "blog/20261103-andrey-0-13-0"


def test_the_blog_lists_the_newest_post_first(conf):
    newest_first = conf["_newest_first"]
    posts = [
        ("blog/20261008-why-andrey-is-fast", "2026-10-08", "0.1.0"),
        ("blog/20261008-introducing-andrey", "2026-10-08", "0.1.0"),
        ("blog/20261012-andrey-0-2-0", "2026-10-12", "0.2.0"),
        ("blog/a-note", "2026-10-10", ""),
        ("blog/20261012-andrey-0-3-0", "2026-10-12", "0.3.0"),
    ]
    # By date, then version, then file name.
    assert newest_first(posts) == [
        "blog/20261012-andrey-0-3-0",
        "blog/20261012-andrey-0-2-0",
        "blog/a-note",
        "blog/20261008-why-andrey-is-fast",
        "blog/20261008-introducing-andrey",
    ]
    # A post without a version leads when it is the newest, and follows one with a version on the
    # same day.
    assert newest_first([*posts, ("blog/later", "2026-10-13", "")])[0] == "blog/later"
    assert newest_first([*posts, ("blog/same-day", "2026-10-12", "")])[2] == "blog/same-day"
    assert newest_first(posts[1::-1]) == newest_first(posts[:2])


def test_the_homepage_and_docs_styles_and_scripts_name_no_color(cards):
    # Every color is a generated token, so a palette edit reaches the hero and the chip too.
    assets = ("site.css", "figures.css", "why-fast.css", "why-fast.js")
    docs = ("docs/_static/andrey.css", "docs/_static/benchmarks.css")
    for path in [*(f"site/assets/{name}" for name in assets), *docs]:
        rules = re.sub(r"/\*.*?\*/", "", (ROOT / path).read_text(), flags=re.S)
        if path.endswith(".js"):
            rules = re.sub(r"//[^\n]*", "", rules)
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\b(rgba?|hsla?|oklch|lab)\(", rules), path
        assert not re.search(r":\s*(black|white)\b", rules), path
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
    # The banner links the website's homepage.
    assert '<a href="https://andrey.abel.ai/"><img' in header
    assert 'alt="Andrey: accelerated, accurate, agentic causal discovery"' in header
    assert "<picture" not in header and "<h1" not in readme
    assert cards.TAGLINE == "Accelerated, accurate, agentic causal discovery"


@pytest.mark.parametrize("kind", ["pair", "lanes", "iceberg", "contour", "hourglass", "ground"])
def test_a_post_cover_preview_and_thumbnail_are_small_jpegs_drawn_the_same_every_build(cards, kind):
    import io

    from PIL import Image

    pytest.importorskip("resvg_py")  # the docs group's renderer; CI's extras jobs lack it
    sizes = {
        "cover": (cards.COVER_W, cards.COVER_H),
        "preview": (cards.W, cards.H),
        "thumb": (cards.THUMB_W, cards.THUMB_W * cards.COVER_H // cards.COVER_W),
    }
    draw = {
        "cover": lambda: cards.cover_jpeg(kind),
        "preview": lambda: cards.preview_jpeg(kind, "Hidden common causes", "Release 0.2.0"),
        "thumb": lambda: cards.thumb_jpeg(kind),
    }
    data = {name: make() for name, make in draw.items()}
    for name, image_bytes in data.items():
        with Image.open(io.BytesIO(image_bytes)) as image:
            assert (image.format, image.size) == ("JPEG", sizes[name]), name
        assert len(image_bytes) <= cards.MAX_BYTES, name
        assert draw[name]() == image_bytes, name
    # The front page loads every earlier post's thumbnail, so each is a small part of a cover.
    assert len(data["thumb"]) <= len(data["cover"]) // 4
    assert cards.cover_alt(kind).strip()


def test_every_published_art_is_stored_in_the_brand_tones(cards):
    import numpy as np
    from PIL import Image

    pytest.importorskip("resvg_py")

    def drift(path, match_ground):
        with Image.open(path) as image:
            art = image.convert("RGB")
        toned = cards.brand_tones(art, match_ground=match_ground)
        return np.abs(np.asarray(art, float) - np.asarray(toned, float)).mean()

    # Art already in the tones moves a few levels when recolored again (JPEG noise); art drawn in
    # other blues or ambers moves tens of levels. A failure: store the file through
    # cards.brand_tones (match_ground=False for a release card's art).
    ground = cards._ground_lightness()[1]
    for path in [*cards.ART.values(), *cards.CENTERED.values()]:
        assert drift(path, True) < 8, path.name
        with Image.open(path) as image:
            light = np.asarray(image.convert("RGB"), float) @ cards.LUMA
        assert abs(np.median(light) - ground) < 4, path.name
    cards_art = sorted((ROOT / "docs" / "_static" / "release-cards").glob("*.jpg"))
    assert cards_art
    for path in cards_art:
        assert drift(path, False) < 8, path.name
    # Pure blue and pure amber come back as the palette's tones at their own lightness.
    swatch = Image.new("RGB", (40, 10), (40, 60, 200))
    swatch.paste((230, 150, 40), (20, 0, 40, 10))
    toned = np.asarray(cards.brand_tones(swatch), float)
    assert toned[0, 0, 2] > toned[0, 0, 0] and toned[0, 39, 0] > toned[0, 39, 2]


def test_a_cover_that_is_not_in_the_brand_kit_fails(cards):
    with pytest.raises(ValueError, match="unknown cover"):
        cards.cover_jpeg("photo")


def _release_covers():
    """Each release cover and tile in the Release cards page, the cards, and the example in
    ``docs/DEVELOPMENT.md``: how many folders deep its page is built, its classes, and its body's
    paragraphs. The example is the Release cards page's cover."""
    docs = ROOT / "docs"
    pages = [docs / "docs" / "benchmarks.md", *sorted((docs / "docs" / "benchmarks").glob("*.md"))]
    for page in [*pages, docs / "DEVELOPMENT.md"]:
        depth = max(len(page.relative_to(docs).parts) - 1, 1)
        text = page.read_text(encoding="utf-8")
        for indent, block in re.findall(
            r"^( *):::\{card\}\n(.*?)\n\1:::$", text, flags=re.S | re.M
        ):
            lines = [line.removeprefix(indent) for line in block.split("\n")]
            options = dict(re.findall(r"^:([\w-]+): *(.*)$", "\n".join(lines), flags=re.M))
            classes = options.get("class-card", "").split()
            body = "\n".join(line for line in lines if not re.match(r":[\w-]+:", line)).strip()
            if {"release-cover", "release-tile"} & set(classes):
                yield depth, classes, re.split(r"\n\s*\n", body)


def test_a_release_without_art_takes_the_default_cover_and_one_with_art_keeps_it():
    css = re.sub(
        r"/\*.*?\*/", "", (ROOT / "docs" / "_static" / "benchmarks.css").read_text(), flags=re.S
    )
    art_dir = ROOT / "docs" / "_static" / "release-cards"
    defaults = 0
    for depth, classes, paragraphs in _release_covers():
        version = paragraphs[-1]
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), paragraphs
        art = [name for name in classes if name.startswith("release-art")]
        if "release-default" in classes:
            # The default stands in for art: a release with art shows it, and no art class mixes in.
            defaults += 1
            assert not (art_dir / f"{version}.jpg").exists() and not art, classes
            if "release-cover" in classes:
                # Masthead, the wordmark's image (never typed text) in the title's place, version.
                assert len(paragraphs) == 3, paragraphs
                image = re.fullmatch(
                    r'<p class="sd-card-text">(<img [^>]*>)</p>', paragraphs[1], re.S
                )
                assert image, paragraphs[1]
                src = re.search(r'src="([^"]+)"', image[1])[1]
                assert src == "../" * depth + "_static/wordmark-dark.svg"
                assert (ROOT / "site" / "brand" / "wordmark-dark.svg").exists()
                assert 'class="dark-light"' in image[1] and re.search(r'alt="[^"]+"', image[1])
        elif art:
            # The art shown, `release-art-<release>` or the bare class 0.2.0's card keeps, is the
            # release's own.
            named = [name for name in art if name != "release-art"]
            assert len(named) <= 1, classes
            name = re.escape(named[0] if named else "release-art")
            rule = re.search(rf"\.{name}(?![\w-])[^{{]*\{{([^}}]*)\}}", css)[1]
            assert f'url("release-cards/{version}.jpg")' in rule, (version, classes)
        else:
            # 0.1.0's tile, released before the default, keeps the stipple ground.
            assert version == "0.1.0" and classes == ["release-tile"], classes
    assert defaults, "no default cover to check; DEVELOPMENT.md shows one"


def test_the_default_cover_is_the_stipple_ground_with_no_amber_dot():
    css = re.sub(
        r"/\*.*?\*/", "", (ROOT / "docs" / "_static" / "benchmarks.css").read_text(), flags=re.S
    )
    rules = [(" ".join(s.split()), body) for s, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]
    grounds = [body for selector, body in rules if selector.endswith(".release-default")]
    assert len(grounds) == 1
    assert 'url("release-ground.jpg")' in grounds[0]
    # The art covers' dot above the version, and the default's rule that removes it.
    dot = ".release-cover .sd-card-text:nth-of-type(3)::before"
    assert "var(--andrey-ground-amber)" in dict(rules)[dot]
    no_dot = dot.replace(".release-cover", ".release-cover.release-default")
    assert re.search(r"display:\s*none|content:\s*none", dict(rules)[no_dot])


def test_a_banner_without_art_is_the_default_with_one_amber_accent(cards, conf):
    pytest.importorskip("resvg_py")
    # A post that names no art gets the "ground" kind.
    app = types.SimpleNamespace(env=types.SimpleNamespace(metadata={"blog/post": {}}))
    kind = conf["_cover_kind"](app, "blog/post")
    svg = cards.preview_svg(kind, "Calibration", "Release 0.16.0")
    # Navy, the white wordmark with its amber stroke, and the release: no art, title, or dot.
    assert f'fill="{cards.grain.NAVY}"' in svg and "<image" not in svg
    assert svg.count(cards.grain.AMBER) == 1 and "<circle" not in svg
    assert "Release 0.16.0" in svg and "Calibration" not in svg
    assert cards.preview_jpeg(kind, "Calibration", "Release 0.16.0")[:2] == b"\xff\xd8"
    # A banner over art keeps its title and the dot before its release.
    art = cards.preview_svg("crystal", "Multi-domain data", "Release 0.15.0")
    assert "<circle" in art and "Multi-domain" in art and "<image" in art


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_the_published_summary_gives_every_paired_ratio():
    """A ratio is withheld only where a time is missing, never for accuracy."""
    summary = build.load_summary(build.SUMMARY)
    for release in build.release_summaries(summary):
        for method in release["methods"]:
            for row in method["rows"]:
                timed = row["andrey_s"] is not None and row["comparator_s"] is not None
                assert (row["ratio"] is not None) == timed, (method["method"], row["d"])


def test_every_hand_written_copy_of_the_claim_is_the_builds():
    """The docs index, the notebook, and the live demo repeat the speed claim; the live demo also
    repeats its own heading. Each must match."""
    claim = build.CLAIM
    for page in (
        ROOT / "docs" / "docs" / "index.md",
        ROOT / "examples" / "speed_on_your_machine.ipynb",
    ):
        assert claim in page.read_text(encoding="utf-8"), page.name
    app = (ROOT / "apps" / "live-discovery" / "app.py").read_text(encoding="utf-8")
    assert "<h1>Causal discovery, fast.</h1>" in app and claim in app
    space = (ROOT / "apps" / "live-discovery" / "README.md").read_text(encoding="utf-8")
    assert f"Causal discovery, fast. {claim}" in space
    assert 'title="Andrey | Causal discovery, fast"' in app


def test_the_citation_title_is_the_tagline(cards):
    # The package's own line, as on the README banner; the FAQ cites what the README cites.
    title = f"Andrey: {cards.TAGLINE}"
    assert f'title: "{title}"' in (ROOT / "CITATION.cff").read_text()
    assert f"title = {{{title}}}," in (ROOT / "README.md").read_text()


@pytest.fixture(scope="module")
def published():
    if not build.SUMMARY.exists():
        pytest.skip("the published summary is absent")
    return build.release_summaries(build.load_summary(build.SUMMARY))


@pytest.fixture(scope="module")
def published_page(published):
    return render_homepage(build.load_summary(build.SUMMARY))


def test_every_family_features_one_supported_method_by_the_rule(published):
    featured = build.featured_methods(published)
    assert list(featured) == list(build.SPEC_FAMILIES)
    specs = {s.name: s for s in list_specs()}
    order = list(specs)
    packages = build.compared_packages(published)

    def reach(spec):
        """The largest size at which every package in the shown setting ran and Andrey's graph
        beats the empty graph, or None."""
        _, methods = build.default_settings(published)[spec]
        family = methods[0]["family"]
        sizes = [
            d
            for d, rows in build.shared_rows(methods).items()
            if build.beats_empty(family, rows[0])
        ]
        return max(sizes, default=None)

    for family, entry in featured.items():
        spec = entry["spec"]
        assert specs[spec].status == "supported" and specs[spec].family == family
        assert entry["packages"] == len(packages[spec]) and entry["size"] == reach(spec)
        # No other candidate in the family has more packages, or as many and a larger reach, or
        # both equal and comes first in the registry.
        for other in build.default_settings(published):
            if other == spec or specs[other].family != family or reach(other) is None:
                continue
            mine = (len(packages.get(spec, ())), reach(spec), -order.index(spec))
            theirs = (len(packages.get(other, ())), reach(other), -order.index(other))
            assert mine > theirs, (family, spec, other)
        # Every package in the row ran at its size, and Andrey's graph beats the empty graph.
        for _, row in entry["cells"]:
            assert row["andrey_s"] is not None and row["comparator_s"] is not None
            assert build.beats_empty(family, row)


def test_research_code_is_not_a_package():
    summary = {
        **SUMMARY,
        "methods": [
            _method("PC", "constraint-based", "causal-learn", [_row(20, 0.1, 1.0, 10, 5, 6, 20)]),
            _method(
                "PC", "constraint-based", "SCORE authors' code", [_row(20, 0.1, 2, 20, 5, 6, 20)]
            ),
        ],
    }
    assert build.compared_packages([summary]) == {"pc": {"causal-learn"}}


def test_a_method_whose_graph_never_beats_the_empty_graph_is_not_featured():
    worse = [_row(20, 0.1, 1.0, 10.0, 25, 26, 20)]
    better = [_row(20, 0.1, 1.0, 10.0, 5, 6, 20)]
    summary = {
        **SUMMARY,
        "methods": [
            _method("PC", "constraint-based", "causal-learn", worse),
            _method("PC", "constraint-based", "gCastle", worse),
            _method("FCI", "constraint-based", "causal-learn", better, regime="latent_gauss_er"),
        ],
    }
    assert build.featured_methods([summary])["constraint"]["spec"] == "fci"


def test_a_note_gives_the_cards_verdict_or_else_the_error_counts():
    row = {"andrey_errors": 5, "comparator_errors": 7}
    for verdict in ("tie", "better", "worse"):
        assert build.outcome({**row, "card_verdict": verdict}) == verdict
    for verdict in (None, "unrecorded"):
        assert build.outcome({**row, "card_verdict": verdict}) == "fewer"
        assert build.outcome({**row, "andrey_errors": 7, "card_verdict": verdict}) == "same"
        assert build.outcome({**row, "andrey_errors": 9, "card_verdict": verdict}) == "more"


def test_the_ledger_has_a_row_per_family_with_each_packages_time_and_note(
    published, published_page
):
    featured = build.featured_methods(published)
    rows = [
        (row.split('"', 1)[0], row)
        for row in published_page.split('<div class="l-row" id="featured-')[1:]
    ]
    assert [family for family, _ in rows] == list(featured)
    for family, row in rows:
        entry = featured[family]
        assert f"<b>{html.escape(entry['name'])}</b>" in row
        assert f'href="{build.card_href(entry["release"])}"' in row
        notes = re.findall(r'<li class="v v--(\w+)">', row)
        assert notes == [build.outcome(r) for _, r in entry["cells"]]
        assert build.seconds_text(entry["cells"][0][1]["andrey_s"]) in row
        for m, r in entry["cells"]:
            assert f"{html.escape(build.package_name(m['comparator']))} " in row
            assert build.seconds_text(r["comparator_s"]) in row


def test_the_homepage_counts_come_from_the_registry(published, published_page):
    specs = list_specs()
    supported_count = sum(s.status == "supported" for s in specs)
    families = len({s.family for s in specs})
    lede = re.search(r'<p class="lede">(.*?)</p>', published_page, re.S)[1]
    assert f"{supported_count} supported methods in {build.NUMBER_WORDS[families]} families" in lede
    compared = len({m["spec"] for m in build.supported(published)})
    share = f"all {supported_count}" if compared == supported_count else f"{compared} of the"
    assert f"compare {share}" in published_page
    cards = [p for p in (ROOT / "docs" / "docs" / "benchmarks").glob("*.md")]
    assert f"All {len(cards)} release cards" in published_page


def test_the_card_row_shows_the_newest_cards_with_their_covers(published_page):
    newest = sorted(
        (p.stem for p in (ROOT / "docs" / "docs" / "benchmarks").glob("*.md")),
        key=lambda v: tuple(map(int, v.split("."))),
        reverse=True,
    )[: build.LATEST_CARDS]
    row = published_page[published_page.index('<ul class="cards">') :]
    row = row[: row.index("</ul>")]
    assert re.findall(r'<a href="docs/benchmarks/([\d.]+)\.html">', row) == newest
    for version in newest:
        art = ROOT / "docs" / "_static" / "release-cards" / f"{version}.jpg"
        tile = row[row.index(f"docs/benchmarks/{version}.html") :]
        tile = tile[: tile.index("</li>")]
        assert (f'src="_static/release-cards/{version}.jpg"' in tile) == art.exists()
        assert ("card-ground" in tile) != art.exists()


def test_no_list_on_the_homepage_is_longer_than_the_families(published_page):
    # The page features one method per family; every list of methods lives in the catalog.
    from html.parser import HTMLParser

    class Lists(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.sizes = [], []

        def handle_starttag(self, tag, attrs):
            if tag in ("ul", "ol", "tbody"):
                self.stack.append(0)
            elif tag in ("li", "tr") and self.stack:
                self.stack[-1] += 1

        def handle_endtag(self, tag):
            if tag in ("ul", "ol", "tbody") and self.stack:
                self.sizes.append(self.stack.pop())

    parser = Lists()
    parser.feed(published_page)
    assert parser.sizes and max(parser.sizes) <= len(build.SPEC_FAMILIES)
    assert published_page.count('class="l-row"') == len(build.SPEC_FAMILIES)


def test_every_homepage_link_leads_to_a_page_the_site_builds(published_page):
    # Each relative link names a page built from a source in docs/; the built-site test checks
    # them again in the built site, anchors included.
    links = set(re.findall(r'<a [^>]*href="([^"#]+)(?:#[^"]*)?"', published_page))
    local = sorted(link for link in links if not link.startswith(("http", "mailto:")))
    assert local
    for link in local:
        stem = str(ROOT / "docs" / link.removesuffix(".html"))
        sources = [Path(stem + suffix) for suffix in (".md", ".ipynb", ".rst")]
        assert link.endswith(".html") and any(p.exists() for p in sources), link


def test_amber_text_reads_on_the_paper_in_both_themes():
    def luminance(color):
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    for palette in (andrey.viz.LIGHT, andrey.viz.DARK):
        a, b = luminance(palette.amber_text), luminance(palette.paper)
        assert (max(a, b) + 0.05) / (min(a, b) + 0.05) >= 3, palette.amber_text
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    fast = css[css.index(".andrey-home .fast {") :]
    assert "color: var(--andrey-amber-text);" in fast[: fast.index("}")]
