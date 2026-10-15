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
import warnings
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
    ups = re.findall(r"up to <b>([^<]+)</b>", chart)
    # FCI's largest is at d = 400, where both graphs are worse than the empty graph: a speed
    # ratio does not depend on accuracy.
    assert ups == ["941x", "8.9x", "3.6x"] * 2
    # Sizes without a ratio are plotted hollow: DirectLiNGAM 200 (Andrey only; lingam passed its
    # time cap).
    assert chart.count("is-open") == 1 * 2
    assert "ICA-LiNGAM" not in chart and "ICA-LiNGAM" not in build.accuracy_chart([SUMMARY])
    # A comparison the charts skip is in the table, not in the notes.
    assert not any("ICA-LiNGAM" in note for note in build.exceptions(SUMMARY))
    assert "ICA-LiNGAM" in build.speed_table(SUMMARY)


def test_a_family_intro_names_its_packages_and_leaves_accuracy_out():
    constraint = [m for m in build.claimed_comparisons([SUMMARY]) if m["family"] == "constraint"]
    intro = build.family_intro(constraint)
    assert "alongside causal-learn and gCastle" in intro or "alongside causal-learn" in intro
    assert "errors" not in intro


def test_the_release_promise_is_the_quality_gate():
    # The accuracy section says every release is tested against true graphs. The quality gate is
    # that test, and tests/unit/test_release_gates.py holds every PyPI publish job to needing it.
    tiles = build.accuracy_tiles([SUMMARY]).lower()
    assert "every release is tested against the true graphs" in tiles
    assert "every release is tested against the true graphs" in build.quality_intro([SUMMARY])
    gate = (ROOT / ".github" / "workflows" / "quality-gate.yml").read_text()
    assert "workflow_call:" in gate and "python -m qa.quality_gate" in gate


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


def test_methods_list_every_registered_method():
    summaries = placeholder_summaries()
    title, html = build.methods(summaries)
    specs = list_specs()
    assert title.startswith(f"{len(specs)} methods in ")
    assert html.count('class="method"') == len(specs)
    for spec in specs:
        assert f"docs/code/generated/andrey.{spec.name}.html" in html
    # Each supported method a release's summary compares with another package has the seal, and a
    # tooltip that describes it.
    measured = {build.registry_name(m) for s in summaries for m in s["methods"]}
    sealed = sorted(re.findall(r'aria-describedby="bench-(\w+)"', html))
    assert sealed == sorted(n for n in measured if get_spec(n).status == "supported")
    assert html.count('role="tooltip"') == len(sealed) and "Benchmarked</a>" not in html
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
    summaries = placeholder_summaries()
    page = placeholder_page
    # Each chart's largest speedup, as "up to"; every size is on the release cards.
    for _, methods in build.time_groups(summaries):
        if all(build.faster(m) for m in methods):
            claimed = [r for m in methods for r in m["rows"] if r["ratio"] is not None]
            best = max(claimed, key=lambda r: r["ratio"])
            family = methods[0]["family"]
            dialog = page[page.index(f'id="speed-{family}"') :]
            assert f"up to <b>{build.ratio_label(best)}</b>" in dialog[: dialog.index("</dialog>")]
    assert not re.search(r"/scratch/|/home/|\bsc\d{3}\b|\bsol\b", page, re.IGNORECASE)
    # A summary's notes on its runs stay on the release cards.
    for summary in summaries:
        for method in summary["methods"]:
            for text in [method.get("timing_note"), *(r.get("note") for r in method["rows"])]:
                assert not text or text not in page


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
    # that A alone, with ink that follows the browser's light or dark bar.
    amber = andrey.viz.LIGHT.categorical[1]
    for name, palette in (("light", andrey.viz.LIGHT), ("dark", andrey.viz.DARK)):
        svg = (ROOT / "site" / "brand" / f"wordmark-{name}.svg").read_text()
        assert re.findall(r'fill="([^"]+)"', svg) == [palette.ink, amber]
    icon = (ROOT / "site" / "brand" / "favicon.svg").read_text()
    assert f".ink{{fill:{andrey.viz.LIGHT.ink}}}" in icon and f'fill="{amber}"' in icon
    assert f"prefers-color-scheme: dark){{.ink{{fill:{andrey.viz.DARK.ink}}}" in icon


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
    intro = build.family_intro(build.claimed_comparisons([SUMMARY]))
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


def test_the_headline_is_one_sentence_with_no_number(placeholder_page):
    found = re.search(r'<h1 id="headline">(.*?)</h1>', placeholder_page, re.S)[1]
    text = re.sub(r"<[^>]+>", "", found)
    assert text == "Causal discovery: accelerated, accurate, agentic."
    assert '<span class="word">accelerated, accurate, agentic.</span>' in found


def test_the_hero_cards_lead_to_the_sections_and_repeat_their_claims(placeholder_page):
    hero = placeholder_page[placeholder_page.index('<section class="hero"') :]
    hero = hero[: hero.index("</section>")]
    claims = build.act_claims(placeholder_summaries())
    cards = re.findall(r'<a href="#([\w-]+)">.*?<b>([^<]+)</b><span>([^<]+)</span></a>', hero)
    assert [key for key, _, _ in cards] == [key for key, *_ in build.ACTS] == list(claims)
    for key, title, short in cards:
        assert re.search(rf'<section class="section wrap act[\w -]*" id="{key}"', placeholder_page)
        assert f'<h2 id="{key}-title" class="reveal">{title}</h2>' in placeholder_page
        # A card's number is its section's: the short claim and the sentence agree.
        assert short == claims[key][0]
        cover = re.sub(r"<[^>]+>", "", claims[key][1])
        assert set(re.findall(r"\d[\d,]*", short)) <= set(re.findall(r"\d[\d,]*", cover))


def test_the_lede_names_every_registry_family_in_one_sentence(placeholder_page):
    lede = build.lede()
    families = dict.fromkeys(spec.family for spec in list_specs())
    for family in families:
        assert build.FAMILY_TERMS[family] in lede
    assert lede.count(". ") == 0 and lede.endswith(" methods.")
    assert lede in " ".join(placeholder_page.split())


def test_the_page_title_is_the_one_line_description(placeholder_page):
    # The description pyproject.toml and the README give, as the browser tab shows it.
    assert "<title>Andrey, a very fast causal discovery package</title>" in placeholder_page


# The registry as it is, and without one method the description does not name: the count of "more"
# methods follows the registry.
@pytest.mark.parametrize("dropped", [None, "gin"])
def test_the_description_states_the_speed_claim_and_the_method_count(monkeypatch, dropped):
    summaries = placeholder_summaries()
    specs = [s for s in list_specs() if s.name != dropped]
    monkeypatch.setattr(build, "list_specs", lambda: specs)
    description = build.page_description(summaries)
    assert "over 1,000x faster on PNL than other popular packages" in description
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


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_the_readme_chart_has_six_cards_each_a_published_ratio():
    summaries = build.release_summaries(build.load_summary(build.SUMMARY))
    slides = build.speedup_slides(summaries)
    assert [name for _, name in build.PANEL] == [
        "PC",
        "FCI",
        "GES",
        "XGES",
        "Tabu search",
        "DirectLiNGAM",
    ]
    assert len(slides) == 6
    by_release = dict(zip(build.RELEASES, summaries, strict=True))
    for (release, name), (shown, size, bars) in zip(build.PANEL, slides, strict=True):
        assert shown == name
        # The slide is the method's largest size with a published ratio, at the panel's mean
        # degree, and Andrey's bar prints that ratio, "≥" before a lower bound. A pair's size is
        # its rows.
        methods = [m for m in by_release[release]["methods"] if m["method"] == name]
        rows = [
            r
            for m in methods
            for r in m["rows"]
            if r["ratio"] is not None and r.get("mean_degree", build.MEAN_DEGREE) == 2.0
        ]
        pairs = methods[0].get("family") == "pairwise"
        unit = "rows" if pairs else "variables"
        size_of = (lambda r: r["n"]) if pairs else (lambda r: r.get("observed", r["d"]))
        largest = max(size_of(r) for r in rows)
        assert size == f"{largest:,} {unit}"
        ratio, at_least = max(build.claimed_ratio(r) for r in rows if size_of(r) == largest)
        andrey = next(label for package, _, label in bars if package == "Andrey")
        assert andrey == ("\u2265" if at_least else "") + build.ratio_text(ratio)
        assert bars[-1][2] == "1x" and andrey == bars[0][2]


def test_the_readme_chart_takes_rows_at_its_mean_degree():
    summaries = placeholder_summaries()
    release, name = build.PANEL[0]
    summary = summaries[build.RELEASES.index(release)]
    method = next(m for m in summary["methods"] if m["method"] == name)
    # The same sizes again at a denser mean degree, each with a larger ratio: the panel keeps the
    # rows at MEAN_DEGREE.
    denser = [dict(r, mean_degree=4.0, ratio=r["ratio"] * 2) for r in method["rows"]]
    before = build.speedup_slides(summaries)[0]
    method["rows"] = [dict(r, mean_degree=build.MEAN_DEGREE) for r in method["rows"]] + denser
    assert build.speedup_slides(summaries)[0] == before
    kept = {r["ratio"] for m in build.comparisons(summary) for r in m["rows"]}
    assert kept.isdisjoint(r["ratio"] for r in denser)


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
    slide = next(s for s in build.speedup_slides(summaries) if s[0] == "FCI")
    assert slide[2][0] == ("Andrey", 20.0, "\u226520x")
    assert "up to <b>\u226520x</b>" in build.speed_chart(summaries)
    # A bound divides into no other package's bar: PC is compared with two packages.
    _lower_bound(summaries, "PC", 1000.0)
    with pytest.raises(ValueError, match="lower bound"):
        build.speedup_slides(summaries)


def test_the_panel_shows_only_supported_methods(monkeypatch):
    summaries = placeholder_summaries()
    _, name = build.PANEL[1]
    spec = build.registry_name({"method": name})
    specs = [
        s.model_copy(update={"status": "experimental"}) if s.name == spec else s
        for s in list_specs()
    ]
    monkeypatch.setattr(build, "list_specs", lambda: specs)
    with pytest.raises(ValueError, match=f"{name} is not"):
        build.speedup_slides(summaries)


def test_every_speedup_bar_comes_from_andreys_paired_ratios():
    summaries = placeholder_summaries()
    summary = summaries[0]
    bars = next(bars for name, _, bars in build.speedup_slides(summaries) if name == "PC")
    labels = {package: label for package, _, label in bars}
    pc = {m["comparator"]: m for m in summary["methods"] if m["method"] == "PC"}
    row = {name: next(r for r in m["rows"] if r["d"] == 400) for name, m in pc.items()}
    # Andrey's bar is its paired ratio against the slowest package (median times give 756x);
    # gCastle's is that ratio over Andrey's paired ratio against gCastle (times give 3.3x).
    ratio = {name: r["ratio"] for name, r in row.items()}
    assert labels["Andrey"] == build.ratio_text(ratio["causal-learn"]) == "941x"
    assert labels["gCastle"] == build.ratio_text(ratio["causal-learn"] / ratio["gCastle"]) == "3.7x"
    assert labels["causal-learn"] == "1x"


def test_every_timed_package_needs_a_published_ratio():
    """No silent fallback to a time quotient for a package without a paired ratio."""
    summaries = placeholder_summaries()
    release, name = build.PANEL[2]
    summary = summaries[build.RELEASES.index(release)]
    method = next(m for m in summary["methods"] if m["method"] == name)
    row = max((r for r in method["rows"] if r["ratio"]), key=lambda r: r["d"])
    other = copy.deepcopy(method)
    other["comparator"] = "slower package"
    other["rows"] = [dict(row, comparator_s=row["comparator_s"] * 10, ratio=None)]
    summary["methods"].append(other)
    with pytest.raises(ValueError, match="no published ratio against slower package"):
        build.speedup_slides(summaries)


def test_the_readme_chart_needs_every_method():
    summaries = placeholder_summaries()
    release, name = build.PANEL[-1]
    summary = summaries[build.RELEASES.index(release)]
    summary["methods"] = [m for m in summary["methods"] if m["method"] != name]
    with pytest.raises(ValueError, match=name):
        build.speedup_slides(summaries)


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
    assert re.findall(r'\("([\w/]+)", "([\w ]+)"', nav) == [
        ("docs/index", "Docs"),
        ("docs/examples/index", "Examples"),
        ("docs/demos/index", "Demos"),
        ("blog/index", "Blog"),
        ("docs/benchmarks", "Release cards"),
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
        highlighted = re.findall(r'<li class="nav-item current active">\s*<a [^>]*>([^<]+)<', page)
        return highlighted, re.findall(r'aria-current="page">([^<]+)<', page)

    assert marks("index") == ([], [])
    assert marks("docs/index") == (["Docs"], ["Docs"])
    assert marks("docs/guides/getting-started") == (["Docs"], [])
    assert marks("docs/examples/index") == (["Examples"], ["Examples"])
    assert marks("docs/examples/ges_quickstart") == (["Examples"], [])
    assert marks("docs/demos/index") == (["Demos"], ["Demos"])
    assert marks("docs/demos/pc-replay") == (["Demos"], [])
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


def test_every_partial_mark_says_what_it_covers():
    notes = build.comparison_notes()
    assert all(mark == "part" or package != "Andrey" for _, package, mark, _ in notes)
    table = build.comparison_table()
    assert table.count('class="mark mark--') == len(build.COMPARISON) * len(build.PEERS)
    # Each partial mark carries its note as a tooltip and in its screen-reader text.
    assert table.count(" title=") == len(notes) and "compare-note-" not in table
    for _, _, _, note in notes:
        assert f"Partly: {build.escape(note)}</span>" in table
    bad = list(build.COMPARISON) + [("Coverage", "Unexplained", ("yes", "part", "no", "no"), {})]
    original, build.COMPARISON = build.COMPARISON, tuple(bad)
    try:
        with pytest.raises(ValueError, match="in part"):
            build.comparison_notes()
    finally:
        build.COMPARISON = original


def test_the_comparison_table_has_three_groups_with_a_mark_for_every_package():
    groups = list(dict.fromkeys(group for group, *_ in build.COMPARISON))
    assert groups == ["Interface", "Coverage", "Speed"]
    # A group's rows sit together, so each group heads the table once.
    table = build.comparison_table()
    assert table.count('<tr class="compare-group">') == len(groups)
    for _, feature, marks, _ in build.COMPARISON:
        assert len(marks) == len(build.PEERS) and set(marks) <= build.MARKS.keys(), feature
    assert table.count('<th scope="col">') == len(build.PEERS)


def test_interface_example_returns_every_kind_of_graph():
    code, printed, figures = build.run_interface()
    assert code.startswith("for learn in") and "graphs[" not in code
    assert [line.split()[:2] for line in printed.splitlines()] == [
        ["pc", "cpdag"],
        ["fci", "pag"],
        ["direct_lingam", "dag"],
    ]
    assert figures.count("<figure") == 5 and "var(--andrey-" in figures
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
    # Day by day, VARMA-LiNGAM finds the rain wetting the lawn the same day and the sprinkler and
    # the lawn driving each other a day later, which time collapsed draws as a cycle.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        over_time = build.run_script("interface.py")[2]["over_time"]
    sprinkler, wet, rain = range(3)
    assert over_time.labels == ("sprinkler", "wet", "rain") and over_time.lags == (0, 1)
    summary = over_time.summary_graph()
    assert summary.kind == "digraph"
    assert summary.lags_of(rain, wet) == (0,)
    assert summary.lags_of(sprinkler, wet) == summary.lags_of(wet, sprinkler) == (1,)
    others = [(i, j) for i in range(3) for j in range(3) if summary.lags_of(i, j)]
    assert sorted(others) == sorted([(rain, wet), (sprinkler, wet), (wet, sprinkler)])
    # Drawn as two days, the lag-1 edges cross from the first day into the second.
    days, _ = build.two_days(over_time)
    edges = days.oriented_edges()
    assert {kind for *_, kind in edges} == {"directed"}
    assert {(source, target) for source, target, _ in edges} == {
        ("rain(t-1)", "wet(t-1)"),
        ("rain(t)", "wet(t)"),
        ("sprinkler(t-1)", "wet(t)"),
        ("wet(t-1)", "sprinkler(t)"),
    }


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
    section = _dialog(placeholder_page, "agents")
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
    assert placeholder_page.count('<span class="window-title">example.py</span>') == 2
    assert 'class="term-bar window-bar"' in placeholder_page
    # In the launch post a captioned code block is a window, its caption the title; the install
    # command's window has no title.
    post = (ROOT / "docs" / "blog" / "20261008-introducing-andrey.md").read_text(encoding="utf-8")
    quick_start = post.split("## Try Andrey", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r":caption: (.*)", quick_start) == ["example.py", "example.py"]
    blocks = re.findall(r"```\{code-block\} shell\n(?::[^\n]*\n)*\n(.*?)\n```", quick_start, re.S)
    assert "uv add andrey-core" in blocks


def test_the_graphs_tile_draws_every_kind_of_causal_graph(placeholder_page):
    # Every kind a GraphStructure can be has the homepage's name for it, and the tile lists them
    # all, with graphs over time, and draws each one; its dialog draws them with captions, and
    # leads to the guide that explains them.
    from typing import get_args

    from andrey.core.structure import Kind

    assert set(get_args(Kind)) == build.GRAPH_KINDS.keys()
    kinds = build.graph_kinds()
    tile = placeholder_page[placeholder_page.index('data-tile="graphs"') :]
    tile = tile[: tile.index("</li>")]
    assert f">{build.graph_title()}<" in tile and f"{build.listed(kinds)}." in tile
    drawn = ["DAG", "CPDAG", "PAG", "Graph over time", "Cyclic graph"]
    assert re.findall(r"<figcaption>([^<]+)</figcaption>", tile) == drawn
    assert len(drawn) == len(kinds) and tile.count('role="img"') == len(kinds)
    dialog = _dialog(placeholder_page, "graphs")
    assert re.findall(r"<figcaption><strong>([^<]+)</strong>,", dialog) == drawn
    assert '<a href="docs/guides/graph-types.html">' in dialog
    assert (ROOT / "docs" / "docs" / "guides" / "graph-types.md").is_file()
    assert "<pre" not in dialog  # examples, not code


def _dialog(page, key):
    """The markup of the dialog with this id."""
    start = page.index(f'<dialog class="wf tile-dialog" id="{key}"')
    return page[start : page.index("</dialog>", start)]


def _tiles(page):
    """Each tile as (link target, dialog id the script opens or None); a tile that is its own
    evidence, with nothing more behind it, as (None, None)."""
    tiles = []
    for tile in re.findall(r'<li class="tile[^"]*">.*?</li>', page, re.S):
        opener = re.search(r'data-tile="([^"]+)"', tile)
        link = re.search(r'<a class="tile-face" href="([^"]+)"', tile)
        if link is None:
            assert "wf-plus" not in tile  # no plus where nothing more opens
        tiles.append((link and link[1], opener and opener[1]))
    return tiles


def test_every_tile_leads_to_its_details_without_script(placeholder_page):
    page = placeholder_page
    tiles = _tiles(page)
    assert len(tiles) >= 10
    for target, key in tiles:
        if target is None:
            continue  # the tile is its own evidence
        # Every tile that leads on behaves alike: it opens its dialog, whose content shows in the
        # page when the script is absent, and none leaves the page.
        assert key and target == f"#{key}", target
        assert f'<dialog class="wf tile-dialog" id="{key}"' in page
    assert ".tile-dialog { display: block; position: static;" in page


def test_the_page_has_a_tile_per_family_with_a_comparison(placeholder_page):
    summaries = placeholder_summaries()
    tiles = build.family_tiles(summaries)
    families = {m["family"] for m in build.claimed_comparisons(summaries)}
    assert [t["family"] for t in tiles] == [f for f in build.SPEC_FAMILIES if f in families]
    for info in tiles:
        assert placeholder_page.count(f'data-tile="speed-{info["family"]}"') == 1
        dialog = _dialog(placeholder_page, f"speed-{info['family']}")
        assert 'href="docs/benchmarks.html"' in dialog
        best = max(r["ratio"] for m in info["comparisons"] for r in m["rows"] if r["ratio"])
        if info["number"] is None:
            assert best < 2
        else:
            # Rounded down to one digit: never above the family's ratio, always more than half.
            assert best / 2 < info["number"] <= best
            assert len(str(info["number"]).rstrip("0")) == 1
            assert f"<b>{info['number']:,}x</b>" in placeholder_page


@pytest.mark.parametrize(
    ("ratio", "shown"), [(941, 900), (1470, 1000), (16, 10), (6.9, 6), (2.0, 2), (99.9, 90)]
)
def test_a_family_number_rounds_down_to_one_digit(ratio, shown):
    assert build.round_down(ratio) == shown


def test_a_family_where_andrey_is_not_faster_shows_no_number():
    summaries = placeholder_summaries()
    for summary in summaries:
        for method in summary["methods"]:
            if method["method"] in ("PC", "FCI"):
                for row in method["rows"]:
                    row["ratio"] = 0.8
    tiles = {t["family"]: t for t in build.family_tiles(summaries)}
    assert tiles["constraint"]["number"] is None
    assert "faster</span>" not in build.family_tile(tiles["constraint"])
    assert tiles["lingam"]["number"] is not None


def test_a_new_family_gets_a_tile_with_no_code_change(monkeypatch):
    summaries = placeholder_summaries()
    monkeypatch.setattr(build, "SPEC_FAMILIES", {**build.SPEC_FAMILIES, "temporal": "Time series"})
    assert "temporal" not in [t["family"] for t in build.family_tiles(summaries)]
    original = build.comparisons

    def moved(summary):
        out = original(summary)
        for m in out:
            if m["method"] == "PNL":
                m["family"] = "temporal"
        return out

    monkeypatch.setattr(build, "comparisons", moved)
    titles = [t["title"] for t in build.family_tiles(summaries)]
    assert "Time series" in titles


def test_the_sections_follow_the_headline_and_hold_their_evidence(placeholder_page):
    sections = re.findall(
        r'<section class="section wrap act[\w -]*" id="([\w-]+)"', placeholder_page
    )
    assert sections == ["accelerated", "accurate", "agentic", "all-in-one"]
    start = {k: placeholder_page.index(f'id="{k}"') for k in sections}
    ends = [*list(start.values())[1:], placeholder_page.index('<div class="tile-details">')]
    held = {
        "accelerated": ('data-scene="parallel"', 'data-tile="speed-constraint"'),
        "accurate": ('class="err-pairs"', 'href="docs/benchmarks.html"'),
        "agentic": ('data-tile="agents"', "andrey run pc --help --json"),
        "all-in-one": (
            'data-tile="methods"',
            'data-tile="graphs"',
            'data-tile="compare"',
            'id="example"',
        ),
    }
    for (key, at), end in zip(start.items(), ends, strict=True):
        body = placeholder_page[at:end]
        # Every section shows its evidence after its cover: tiles, and notes in plain type where
        # a figure would add nothing.
        cover = re.search(r'<div class="[\w -]*\bact-cover\b', body)
        assert cover and cover.start() < body.index('<ul class="bento-grid">'), key
        assert body.count('<li class="tile') + body.count('<li class="bento-note">') >= 1, key
        for mark in held[key]:
            assert mark in body, (key, mark)
    # No two tiles open the same dialog: what one would show twice is one tile.
    opened = re.findall(
        r'<a class="tile-face" href="#[\w-]+" data-tile="([\w-]+)"', placeholder_page
    )
    assert opened and len(opened) == len(set(opened))


def test_the_hero_pill_copies_the_agent_prompt_the_readme_and_the_faq_give(placeholder_page):
    assert f'data-prompt="{html.escape(build.AGENT_PROMPT)}"' in placeholder_page
    for name in build.AGENTS.values():  # each agent's mark, named
        assert f"<title>{html.escape(name)}</title>" in placeholder_page
    for page in (ROOT / "README.md", ROOT / "docs" / "docs" / "faq.md"):
        assert build.AGENT_PROMPT in " ".join(page.read_text().split()), page.name


def test_the_hero_and_the_first_section_share_one_art(placeholder_page):
    # The fast lanes stay on the screen from the hero into the first section's title, so nothing
    # breaks between them: the first section shows no art of its own before its tiles.
    run = placeholder_page[placeholder_page.index('<div class="run">') :]
    art = run[: run.index('<section class="hero"')]
    assert f'src="_static/art-{build.HERO_ART}.jpg"' in art and 'fetchpriority="high"' in art
    assert 'class="cover-sweep"' in art
    first = run[run.index('id="accelerated"') : run.index('id="accurate"')]
    assert "<img" not in first[: first.index('<ul class="bento-grid">')]
    assert first.index('<div class="run-sheet">') < first.index('<ul class="bento-grid">')
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    rule = css[css.index(".andrey-home .run-art {") :]
    assert "position: sticky;" in rule[: rule.index("}")]
    # The run clips the art at its foot, so the next section opens on its own art alone.
    rule = css[css.index(".andrey-home .run {") :]
    assert "overflow: clip;" in rule[: rule.index("}")]
    # "From data to graph" opens on the art the hero had, and only the first screen's art loads
    # before the rest.
    example = placeholder_page[placeholder_page.index('id="data-to-graph"') :]
    example = example[: example.index('id="example"')]
    assert f'src="_static/art-{build.EXAMPLE_ART}.jpg"' in example and 'loading="lazy"' in example
    assert placeholder_page.count('fetchpriority="high"') == 1


def test_the_accuracy_bars_draw_only_the_methods_with_fewer_errors():
    rows = build.accuracy_rows([SUMMARY])
    figure = build.accuracy_wins([SUMMARY])
    # No count of comparisons: a pair of bars only where Andrey has fewer errors, each saying how
    # many fewer, and the other tile bounds the other methods.
    wins = [r for r in rows if build.accuracy_change(r) <= -0.5]
    assert 0 < len(wins) < len(rows)
    # Each win is a pair of bars, Andrey's then the other package's, on the row's own scale.
    assert figure.count('<div class="err-bar err-bar--andrey">') == len(wins)
    assert figure.count('<div class="err-bar">') == len(wins)
    assert re.findall(r"--f: ([\d.]+)", figure)[1::2] == ["1.000"] * len(wins)
    assert len(re.findall(r">\d+% fewer errors<", figure)) == len(wins)
    tiles = build.accuracy_tiles([SUMMARY])
    # The homepage shows the wins only; every other method's errors are on the release cards.
    assert not any(hedge in tiles.lower() for hedge in ("as accurate", "on par", "within"))
    assert '<a href="docs/benchmarks.html">Release cards</a>' in tiles


def test_the_accuracy_section_leads_with_its_largest_win(placeholder_page):
    # The claim is the largest cut in errors, the chart's first row, not a count of methods.
    fewer = f"{build.fewest_errors(placeholder_summaries()):.0f}%"
    section = placeholder_page[placeholder_page.index('id="accurate"') :]
    section = section[: section.index("</section>")]
    assert f"Up to <b>{fewer}</b> fewer errors than other popular packages." in section
    assert re.findall(r">(\d+%) fewer errors<", section)[0] == fewer
    # The tile under it names what its chart shows rather than repeating the number.
    assert '<h3 class="tile-title">Fewer errors than the other package</h3>' in section
    assert "More accurate in" not in placeholder_page
    assert not any(hedge in section.lower() for hedge in ("as accurate", "on par", "within"))
    # The win rows name the package each is against; PC is more accurate only against gCastle.
    assert re.findall(r'<div class="err-bar"><span>([^<]+)</span>', section)


def test_the_first_section_tells_why_it_is_fast_one_scene_at_a_time(placeholder_page):
    # Under the first section's title, on the shared art, every scene in order with its title and
    # narration, each played by the scroll: no tile, no opener, no dialog of their own.
    section = placeholder_page[placeholder_page.index('id="accelerated"') :]
    section = section[: section.index('<div class="run-sheet">')]
    pin = section[section.index('<div class="wf-pin">') :]
    assert pin.index('id="accelerated-title"') < pin.index('<div class="wf wf-story">')
    scenes = build._why_fast["SCENES"]
    assert re.findall(r'data-scene="(\w+)" data-play="scroll"', pin) == [s["key"] for s in scenes]
    titles = re.findall(r'<li class="wf-step"><div class="wf-words"><h3>([^<]+)</h3>', pin)
    assert titles == [html.escape(s["title"]) for s in scenes]
    assert all(html.escape(s["text"]) in pin for s in scenes)
    assert "data-wf-open" not in placeholder_page and "wf-dialog" not in placeholder_page
    assert '<li class="tile' not in pin and "wf-plus" not in pin
    assert "wf-note" not in pin  # no footnote under the story


def test_a_list_is_a_note_in_plain_type_not_a_card(placeholder_page):
    # The release promise and the methods by family need no figure, which a card would only
    # frame: each is a note beside a tile, its heading and text on the page's paper.
    notes = re.findall(r'<li class="bento-note">(.*?)</li>', placeholder_page, re.S)
    assert [re.search(r"<h3>([^<]+)</h3>", note)[1] for note in notes] == [
        "Tested against the true graphs",
        f"{len(list_specs())} methods",
    ]
    assert not any("tile-glass" in note or "wf-plus" in note for note in notes)
    # The family list opens the methods dialog, which holds the full table.
    assert '<a href="#methods" data-tile="methods">' in notes[1]


def test_the_covers_move_only_without_reduced_motion_and_only_by_transform_and_opacity():
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    for name in ("cover-settle", "hero-lift", "card-arrive"):
        frames = css[css.index(f"@keyframes {name}") :]
        frames = frames[: frames.index("\n}\n")]
        props = set(re.findall(r"^\s+([a-z-]+):", frames, re.M))
        assert props and props <= {"opacity", "translate", "scale"}, name
        # Each runs only inside a block for readers who have not asked for reduced motion.
        use = css.index(f"animation: {name}")
        assert css.rfind("@media (prefers-reduced-motion: no-preference)", 0, use) > css.rfind(
            "\n}\n", 0, use
        ), name
    # A cover settles to its own size, so its art is never shown enlarged at rest.
    settle = css[css.index("@keyframes cover-settle") :]
    assert "scale: 1;" in settle[settle.index("50%") :]


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
    for path in cards.ART.values():
        with Image.open(path) as image:
            median = np.median(np.asarray(image.convert("RGB"), float).reshape(-1, 3), axis=0)
        assert np.abs(median - ground).max() < 16, path.name


def test_every_hero_and_tile_art_is_stored_in_the_brand_tones_at_its_size(placeholder_page, cards):
    import numpy as np
    from PIL import Image

    brand = ROOT / "site" / "brand"
    # The hero's art, which the first section shares, each other section's, the example's, and the
    # documentation's are stored series art, each shown once.
    covers = re.findall(
        r'<div class="cover-frame"><img src="_static/(art-[\w-]+\.jpg)"', placeholder_page
    )
    arts = {f"art-{art}.jpg" for _, _, art, _ in build.ACTS}
    arts |= {f"art-{build.EXAMPLE_ART}.jpg", f"art-{build.DOCS_ART}.jpg"}
    assert sorted(covers) == sorted(arts)
    assert all((brand / c) in cards.ART.values() for c in covers)
    # The tiles' art, the hero cards' pictures, and the section covers, each at its stored size.
    tiles = re.findall(
        r'<img (?:class="tile-art" )?src="_static/((?:bento/)?[\w-]+\.jpg)" alt="" '
        r'width="(\d+)" height="(\d+)"',
        placeholder_page,
    )
    assert len(tiles) >= len(build.FAMILY_ART) + 2 * len(build.ACTS)
    for path, width, height in tiles:
        with Image.open(brand / path) as image:
            assert image.size == (int(width), int(height)), path
            art = image.convert("RGB")
        toned = cards.brand_tones(art, match_ground=False)
        assert np.abs(np.asarray(art, float) - np.asarray(toned, float)).mean() < 8, path


def test_the_homepage_names_its_sources_where_they_are_used(placeholder_page):
    # No notes at the page's foot: the comparison dialog names what its marks were read from, and
    # each partial mark carries its note as a tooltip.
    page = placeholder_page
    for note in (
        'class="footnotes"',
        'class="fn-ref"',
        'class="race-note"',
        'class="compare-notes"',
    ):
        assert note not in page
    dialog = page[page.index('id="compare"') :]
    assert build.comparison_source() in dialog[: dialog.index('<table class="compare">')]
    for feature, package, _, note in build.comparison_notes():
        assert f'title="{html.escape(note)}"' in dialog, (feature, package)
    assert 'href="docs/benchmarks.html"' in page


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


def test_the_newest_release_post_is_the_latest_version(conf):
    newest = conf["_newest_post"]
    posts = [
        ("blog/20261008-introducing-andrey", "2026-10-08", "0.1.0"),
        ("blog/20261008-why-andrey-is-fast", "2026-10-08", "0.1.0"),
        ("blog/20261012-andrey-0-2-0", "2026-10-12", "0.2.0"),
    ]
    assert newest(posts) == ("blog/20261012-andrey-0-2-0", "0.2.0")
    # A later version breaks a tie in date; a tie in both names the posts.
    assert newest([*posts, ("blog/andrey-0-2-1", "2026-10-12", "0.2.1")])[1] == "0.2.1"
    with pytest.raises(ValueError, match="introducing-andrey, blog/20261008-why-andrey-is-fast"):
        newest(posts[:2])


def test_the_blog_lists_the_newest_post_first(conf):
    newest_first = conf["_newest_first"]
    posts = [
        ("blog/20261008-why-andrey-is-fast", "2026-10-08", "0.1.0"),
        ("blog/20261008-introducing-andrey", "2026-10-08", "0.1.0"),
        ("blog/20261012-andrey-0-2-0", "2026-10-12", "0.2.0"),
        ("blog/a-note", "2026-10-10", ""),
        ("blog/20261012-andrey-0-3-0", "2026-10-12", "0.3.0"),
    ]
    # By date, then version; posts that tie in both keep their order.
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
    with pytest.raises(ValueError, match="introducing-andrey, blog/20261008-why-andrey-is-fast"):
        newest_first(posts[:2])


def test_the_homepage_has_no_latest_strip(placeholder_page):
    # The header's dot on Release cards marks a newer release, and the header links the blog and
    # the FAQ.
    assert 'class="latest' not in placeholder_page
    assert "latest" not in (ROOT / "site" / "homepage.html").read_text()


def test_the_hero_grain_is_small_and_the_same_on_every_build(cards):
    tiles = dict(cards.grain.hero_tiles())
    assert sum(len(t) for t in tiles.values()) <= 150 * 1024
    assert dict(cards.grain.hero_tiles()) == tiles
    css = (ROOT / "site" / "assets" / "site.css").read_text()
    assert all(f'url("{name}")' in css for name in tiles)
    assert "cards.grain.hero_tiles()" in (ROOT / "docs" / "conf.py").read_text()


def test_the_sweep_follows_the_streak_in_the_art():
    # The traced path lies on the art's amber pixels from its foot to its tip, so the light runs
    # up the streak, the trail's dots, or the arm, and nowhere else.
    import numpy as np
    from PIL import Image

    assert set(build.SWEPT) == {art for _, _, art, _ in build.ACTS} | {
        build.EXAMPLE_ART,
        build.DOCS_ART,
    }  # every cover
    for art in build.SWEPT:
        path = build.streak(art)
        (x0, y0), (x1, y1) = path[0], path[-1]
        # From the bottom up, or a level beam from left to right, the way its light travels.
        assert y0 > y1 or (abs(y1 - y0) < 20 and x0 < x1), art
        with Image.open(ROOT / "site" / "brand" / f"art-{art}.jpg") as image:
            rgb = np.asarray(image.convert("RGB"), dtype=float)
        warm = (rgb[..., 0] - rgb[..., 2] > 40) & (rgb[..., 0] > 120)
        for (ax, ay), (bx, by) in zip(path, path[1:]):
            for t in np.linspace(0, 1, 6):
                x, y = round(ax + (bx - ax) * t), round(ay + (by - ay) * t)
                if not 0 <= y < rgb.shape[0]:
                    continue  # the foot runs off the frame's bottom edge
                near = warm[max(y - 6, 0) : y + 7, max(x - 6, 0) : x + 7]
                assert near.any(), (art, x, y)
        assert np.hypot(x1 - x0, y1 - y0) > rgb.shape[0] * 0.45  # the streak, not a speck of it
    # Its markup sits in the cover's frame, and the sweep runs only for readers who have not asked
    # for reduced motion; otherwise the overlay stays hidden.
    assert 'class="cover-sweep"' in build.sweep(build.SWEPT[0])
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    rule = css[css.index(".andrey-home .cover-sweep {") :]
    assert "opacity: 0;" in rule[: rule.index("}")]
    use = css.index("animation: streak-sweep")
    assert css.rfind("@media (prefers-reduced-motion: no-preference)", 0, use) > css.rfind(
        "\n}\n", 0, use
    )
    frames = css[css.index("@keyframes streak-sweep") :]
    frames = frames[: frames.index("\n}\n")]
    props = set(re.findall(r"^\s+([a-z-]+):", frames, re.M)) - {"animation-timing-function"}
    assert props == {"stroke-dashoffset", "opacity"}


def test_the_homepage_shades_its_art_with_the_covers_fade(cards):
    # One navy fade, the blog's and the release cards' covers' own, under the hero's, the section
    # covers', and the tiles' type.
    css = (ROOT / "site" / "assets" / "site.css").read_text(encoding="utf-8")
    fade = css[css.index("--cover-fade:") :]
    fade = fade[: fade.index(");") + 2]
    stops = re.findall(r"ground-navy\) (\d+)%, transparent\)(?: ([\d.]+)%)?", fade)
    peak, end = cards.BANNER_FADE_PEAK, cards.BANNER_FADE
    found = [float(v or 0) for stop in stops for v in stop]
    assert found == pytest.approx([peak * 100, 0, peak * 75, end * 55])
    assert re.search(rf"transparent {end * 100:g}%", fade)
    for rule in (".run-art::after", ".tile-face::after", ".cover-art::after"):
        body = css[css.index(f".andrey-home {rule} {{") :]
        assert "background: var(--cover-fade);" in body[: body.index("}")], rule


def test_the_homepage_and_docs_css_name_no_color(cards):
    # Every color is a generated token, so a palette edit reaches the hero and the chip too.
    sheets = ("site.css", "figures.css", "why-fast.css")
    docs = ("docs/_static/andrey.css", "docs/_static/benchmarks.css")
    for path in [*(f"site/assets/{name}" for name in sheets), *docs]:
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
    # The banner links the website's homepage.
    assert '<a href="https://andrey.abel.ai/"><img' in header
    assert 'alt="Andrey, a very fast causal discovery package"' in header
    assert "<picture" not in header and "<h1" not in readme
    assert cards.TAGLINE == "A very fast causal discovery package"


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
    for path in cards.ART.values():
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


def test_a_post_cover_is_named_in_front_matter_and_fails_on_a_bad_name(cards):
    with pytest.raises(ValueError, match="unknown cover"):
        cards.cover_jpeg("photo")
    post = (ROOT / "docs" / "blog" / "20261008-introducing-andrey.md").read_text()
    assert "cover: pair" in post


def test_the_launch_post_links_where_the_name_comes_from():
    faq = (ROOT / "docs" / "docs" / "faq.md").read_text()
    assert '### Where does the name "Andrey" come from?' in faq
    post = (ROOT / "docs" / "blog" / "20261008-introducing-andrey.md").read_text()
    assert "(../docs/faq.md#where-does-the-name-andrey-come-from)" in post


@pytest.mark.skipif(not build.SUMMARY.exists(), reason="the published summary is absent")
def test_the_published_summary_gives_every_paired_ratio():
    """A ratio is withheld only where a time is missing, never for accuracy."""
    summary = build.load_summary(build.SUMMARY)
    assert (
        summary["headline"] == "Over 100x faster on PC, and faster on most other supported methods."
    )
    for release in build.release_summaries(summary):
        for method in release["methods"]:
            for row in method["rows"]:
                timed = row["andrey_s"] is not None and row["comparator_s"] is not None
                assert (row["ratio"] is not None) == timed, (method["method"], row["d"])


def test_no_speed_cell_is_withheld_in_the_reports_or_the_readme():
    reports = sorted((ROOT / "docs" / "docs" / "benchmarks").glob("*.md"))
    for page in (*reports, ROOT / "docs" / "docs" / "benchmarks.md", ROOT / "README.md"):
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
    claim = build.home_claim(build.release_summaries(summary))
    app = (ROOT / "apps" / "live-discovery" / "app.py").read_text(encoding="utf-8")
    assert "<h1>Causal discovery, fast.</h1>" in app and claim in app
    space = (ROOT / "apps" / "live-discovery" / "README.md").read_text(encoding="utf-8")
    assert f"Causal discovery, fast. {claim}" in space
    assert 'title="Andrey | Causal discovery, fast"' in app


def test_the_citation_title_is_the_tagline():
    # The package's own line, as on the README banner; the homepage's headline is campaign wording.
    cff = (ROOT / "CITATION.cff").read_text()
    assert 'title: "Andrey: A very fast causal discovery package"' in cff
    for page in (ROOT / "README.md", ROOT / "docs" / "docs" / "faq.md"):
        text = page.read_text()
        assert "title = {Andrey: A very fast causal discovery package}," in text, page.name
