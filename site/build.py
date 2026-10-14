"""The homepage's content, generated from the release summaries and the method registry.

``docs/conf.py`` fills the homepage template (``site/homepage.html``) with
:func:`homepage_context`, and ``apps/build_launch.py`` draws the README's chart with
:func:`readme_chart`. ``docs/DEVELOPMENT.md`` has the build command.

Every number on the homepage comes from a release's published benchmark summary (``RELEASES``) or
from the method registry. The speed and accuracy sections cover every comparison with another
package whose method the registry supports; each family's tile shows its largest ratio rounded
down to one digit, and the README claim is the one :func:`claim_sentence` words from them. The
README chart shows one card per method in ``PANEL``. The method list comes from the method
registry. A summary outside ``benchmarks/published/`` loads only as a placeholder, which marks the
page as not for release. Links are relative to the site root.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import re
import runpy
import tempfile
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

import networkx as nx
import numpy as np
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

import andrey.viz
from andrey import GraphStructure
from andrey.cli import main as andrey_cli
from andrey.spec import list_specs

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = ROOT / "docs"
PUBLISHED = ROOT / "benchmarks" / "published"
# Every release's published comparison with other packages, oldest first, by folder under
# PUBLISHED. The speed and accuracy sections cover them all. The first is SUMMARY, the one a
# placeholder preview replaces.
RELEASES = (
    "alpha-2026-09",
    "hidden-confounders-2026-10",
    "score-search-2026-10",
    "pairwise-2026-10",
)
SUMMARY = PUBLISHED / RELEASES[0] / "summary.json"

palette_block = runpy.run_path(str(DOCS / "palette_css.py"))["palette_block"]
# The number formats the live demo shares.
_formatting = runpy.run_path(str(SITE / "formatting.py"))
_why_fast = runpy.run_path(str(SITE / "why_fast.py"))
ratio_text, seconds_text, number_text = (
    _formatting[name] for name in ("ratio_text", "seconds_text", "number_text")
)

# The method registry's own families, in the order the method list shows them.
SPEC_FAMILIES = {
    "constraint": "Constraint-based",
    "score": "Score-based",
    "permutation": "Permutation-based",
    "lingam": "Linear non-Gaussian",
    "temporal": "Time series",
    "latent": "Latent variables",
    "pairwise": "Pairwise direction",
}
# Each registry family as the hero's first sentence names its methods.
FAMILY_TERMS = {
    "constraint": "constraint-based",
    "score": "score-based",
    "permutation": "permutation-based",
    "lingam": "linear non-Gaussian",
    "temporal": "time-series",
    "latent": "latent-variable",
    "pairwise": "pairwise",
}
REGIMES = {
    "linear_gauss_er": "linear-Gaussian data from Erdos-Renyi graphs",
    "latent_gauss_er": "linear-Gaussian data from Erdos-Renyi graphs with hidden confounders",
    "lingam_sf": "linear non-Gaussian data from scale-free graphs",
    "latent_lingam_er": "linear non-Gaussian data from Erdos-Renyi graphs with hidden confounders",
    "anm_pair": "cause-effect pairs with additive noise",
    "pnl_pair": "cause-effect pairs with post-nonlinear noise",
}
METHOD_KEYS = {"method", "comparator", "regime", "hardware", "build", "rows"}
ROW_KEYS = {"d", "seeds", "andrey_s", "comparator_s", "ratio"}
# How each kind of summary counts a graph's errors: the row keys for Andrey's count, the other
# package's, and the empty graph's (None where a summary records none), then what one error is, as
# the accuracy section says it.
ERRORS = {
    "shd": (
        ("andrey_shd", "comparator_shd", "empty_shd"),
        "an edge to add, remove, or reorient to reach the true graph",
    ),
    "directed": (
        ("andrey_directed_errors", "comparator_directed_errors", "empty_directed_errors"),
        "on data with hidden common causes, a directed edge that only one of the learned and "
        "true graphs has",
    ),
    "direction": (
        ("andrey_reversed", "comparator_reversed", None),
        "on a cause-effect pair, the reversed direction",
    ),
}
# The density the homepage shows. A summary that measures more than one mean degree at a size
# gives the panel, the charts, and the counts its rows at mean degree 2, the density of the other
# summaries' Erdos-Renyi graphs.
MEAN_DEGREE = 2.0
DEMO_URL: str | None = None  # the live demo; link it once its Space is public
# The one-line description, as pyproject.toml and the README give it.
DESCRIPTION = "Andrey, a very fast causal discovery package."

# Graph colors resolve through the page's tokens, so the drawing follows the light or dark theme.
TOKEN_PALETTE = andrey.viz.Palette(
    **{
        field.name: (
            getattr(andrey.viz.LIGHT, field.name)
            if isinstance(getattr(andrey.viz.LIGHT, field.name), tuple)
            else f"var(--andrey-{field.name.replace('_', '-')})"
        )
        for field in dataclasses.fields(andrey.viz.LIGHT)
    }
)


class Footnotes:
    """A page's notes, numbered in the order they are added and listed together at its foot.

    Add them in reading order; each call returns the superscript that points at its note.
    """

    def __init__(self) -> None:
        self.notes: list[str] = []

    def add(self, note: str) -> str:
        """Record ``note`` (HTML) and return its reference."""
        self.notes.append(note)
        n = len(self.notes)
        return f'<sup class="fn-ref"><a href="#fn-{n}" id="fn-ref-{n}">{n}</a></sup>'

    def render(self) -> str:
        """The numbered notes, each with a link back to where it is referenced.

        A note that ends in a list takes its link after the lead-in, not below the list.
        """
        items = []
        for n, note in enumerate(self.notes, start=1):
            back = (
                f' <a class="fn-back" href="#fn-ref-{n}" '
                f'aria-label="Back to reference {n}">&#8617;</a>'
            )
            lead, listed, rest = note.partition("<ul>")
            items.append(f'<li id="fn-{n}">{lead}{back}{listed}{rest}</li>')
        return "".join(items)


# ---- the summaries ---------------------------------------------------------------------------
def load_summary(path: Path, *, placeholder: bool = False) -> dict:
    """Read a release's benchmark summary and check its layout; refuse unpublished files for
    release."""
    if not placeholder and not path.resolve().is_relative_to(PUBLISHED):
        raise ValueError(
            f"{path} is not under benchmarks/published/; preview it with -D andrey_placeholder=1"
        )
    if not path.exists():
        raise ValueError(f"{path} does not exist; it ships in benchmarks/published/")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("placeholder") and not placeholder:
        raise ValueError(
            f"{path} holds placeholder numbers; preview it with -D andrey_placeholder=1"
        )
    missing = {"generated", "methods"} - summary.keys()
    if missing:
        raise ValueError(f"summary lacks {sorted(missing)}")
    for method in summary["methods"]:
        name = method.get("method", "?")
        if METHOD_KEYS - method.keys():
            raise ValueError(f"{name} lacks {sorted(METHOD_KEYS - method.keys())}")
        if method["regime"] not in REGIMES:
            raise ValueError(f"{name}: describe regime {method['regime']!r} in REGIMES")
        registry_name(method)
        kind = error_kind(method)
        keys = ROW_KEYS | {key for key in ERRORS[kind][0] if key}
        for row in method["rows"]:
            if keys - row.keys():
                raise ValueError(f"{name} d={row.get('d')} lacks {sorted(keys - row.keys())}")
            if kind == "direction" and (row.get("andrey_no_edge") or row.get("comparator_no_edge")):
                raise ValueError(f"{name} n={row['n']}: a pair without an edge is not counted")
            timed = row["andrey_s"] is not None and row["comparator_s"] is not None
            if row["ratio"] is not None and not timed:
                raise ValueError(
                    f"{name} d={row['d']}: a ratio is published, but a time is missing"
                )
            if row.get("ratio_is_lower_bound") and row.get("ratio_lower_bound") is None:
                raise ValueError(f"{name} d={row['d']}: a lower bound is flagged, but not given")
    return summary


def release_summaries(summary: dict) -> list[dict]:
    """Every release's summary, in ``RELEASES`` order, with ``summary`` standing in for the
    first."""
    return [summary, *(load_summary(PUBLISHED / f / "summary.json") for f in RELEASES[1:])]


def normalized(name: str) -> str:
    """Match a summary method ("DirectLiNGAM") to a registry name ("direct_lingam")."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def registry_name(method: dict) -> str:
    """The registry's name for a summary's method: the method its ``solution`` runs
    (``andrey.tabu.numpy.serial`` is ``tabu``), or else its own name matched to the registry's
    ("DirectLiNGAM" is ``direct_lingam``)."""
    names = {s.name for s in list_specs()}
    if "solution" in method:
        name = method["solution"].split(".")[1]
    else:
        name = next((n for n in names if normalized(n) == normalized(method["method"])), "")
    if name not in names:
        raise ValueError(f"{method['method']}: no registered method runs it")
    return name


def error_kind(method: dict) -> str:
    """How a comparison counts a graph's errors: the kind in ``ERRORS`` whose keys its rows give.

    The accuracy comparison sets Andrey's count against the other package's on the same datasets
    at each size, in the measure the summary records for both:

    - ``shd``: the structural Hamming distance to the true graph, the edges to add, remove, or
      reorient to reach it (the alpha-2026-09 and score-search-2026-10 summaries).
    - ``directed``: on data with hidden common causes, the directed edges in one graph but not the
      other, against the true graph's projection onto the observed variables; a reversed edge
      counts twice (the hidden-confounders-2026-10 summary).
    - ``direction``: on cause-effect pairs, the pairs whose direction is reversed (the
      pairwise-2026-10 summary). Each package names a direction on every pair it fits, so a fitted
      pair is either right or reversed; a summary with a pair left without an edge fails to load.

    Fewer is better in each. A comparison whose rows give none of them fails, rather than drop out
    of the count.
    """
    kinds = [
        kind
        for kind, ((ours, _, _), _) in ERRORS.items()
        if all(ours in row for row in method["rows"])
    ]
    if len(kinds) != 1:
        raise ValueError(f"{method.get('method', '?')}: name how its rows count errors in ERRORS")
    return kinds[0]


def claimed_ratio(row: dict) -> tuple[float | None, bool]:
    """A row's speedup and whether it is a lower bound. Where a package hit its time cap on some
    datasets, the summary flags ``ratio_is_lower_bound`` and gives the bound as
    ``ratio_lower_bound``, which stands for ``ratio``."""
    if row.get("ratio_is_lower_bound"):
        return row["ratio_lower_bound"], True
    return row["ratio"], False


def comparisons(summary: dict) -> list[dict]:
    """A summary's comparisons in one layout, whatever kind of summary it is.

    Each keeps the summary's method, package (``comparator``), regime, hardware, and build, and
    adds the registry's name (``spec``), family, and status, its ``error_kind``, and the ``unit``
    of its sizes: "variables", or "rows" for a pairwise method. Each row gives:

    - ``d``: the size, in ``unit``: the variables the method gets, the observed ones on data with
      hidden variables, or a cause-effect pair's rows, since every pair has two variables;
    - ``andrey_s`` and ``comparator_s``: the median fit times, ``None`` past a time cap;
    - ``ratio`` and ``at_least``: the speedup, from :func:`claimed_ratio`;
    - ``andrey_errors`` and ``comparator_errors``: each graph's errors, from :func:`error_kind`.

    Rows at a mean degree other than ``MEAN_DEGREE`` are left out. So are the summary's ``reach``
    rows, the sizes the other package did not finish or did not run: they have no paired time.
    """
    specs = {s.name: s for s in list_specs()}
    out = []
    for method in summary["methods"]:
        spec = specs[registry_name(method)]
        kind = error_kind(method)
        ours, theirs, _ = ERRORS[kind][0]
        pairs = spec.family == "pairwise"
        rows = []
        for row in method["rows"]:
            if row.get("mean_degree", MEAN_DEGREE) != MEAN_DEGREE:
                continue
            ratio, at_least = claimed_ratio(row)
            rows.append(
                {
                    "d": row["n"] if pairs else row.get("observed", row["d"]),
                    "andrey_s": row["andrey_s"],
                    "comparator_s": row["comparator_s"],
                    "ratio": ratio,
                    "at_least": at_least,
                    "andrey_errors": row[ours],
                    "comparator_errors": row[theirs],
                }
            )
        out.append(
            {
                **method,
                "spec": spec.name,
                "family": spec.family,
                "status": spec.status,
                "error_kind": kind,
                "unit": "rows" if pairs else "variables",
                "rows": rows,
            }
        )
    return out


def supported(summaries: list[dict]) -> list[dict]:
    """Every comparison in ``summaries`` whose method the registry supports, grouped by family in
    ``SPEC_FAMILIES`` order, each family's in the summaries' order. Only these count on the
    page."""
    rank = {family: i for i, family in enumerate(SPEC_FAMILIES)}
    found = [m for s in summaries for m in comparisons(s) if m["status"] == "supported"]
    return sorted(found, key=lambda method: rank[method["family"]])


def sizes_text(sizes: list[int]) -> str:
    """Join problem sizes: d = 400; d = 200 and 400; d = 20, 50, and 100."""
    words = [f"{d:,}" for d in sizes]
    if len(words) > 2:
        words = [", ".join(words[:-1]) + ",", words[-1]]
    return "d = " + " and ".join(words)


def listed(words: list[str]) -> str:
    """Join words as a list: a; a and b; a, b, and c."""
    if len(words) > 2:
        return ", ".join(words[:-1]) + f", and {words[-1]}"
    return " and ".join(words)


def ordered(summary: dict) -> list[dict]:
    """A summary's comparisons by registry family, in ``SPEC_FAMILIES`` order, keeping the
    summary's order within one."""
    rank = {family: i for i, family in enumerate(SPEC_FAMILIES)}
    family = {s.name: s.family for s in list_specs()}
    return sorted(summary["methods"], key=lambda m: rank[family[registry_name(m)]])


def _largest(found: list[dict]) -> tuple[dict, dict]:
    """The comparison and row with the largest ratio."""
    cells = [(m, r) for m in found for r in m["rows"] if r["ratio"] is not None]
    method, row = max(cells, key=lambda cell: cell[1]["ratio"])
    if row["ratio"] <= 1:
        raise ValueError("no comparison shows Andrey faster")
    return method, row


def faster(method: dict) -> bool:
    """Whether Andrey is faster in a comparison at the largest size with a published ratio."""
    rows = [r for r in method["rows"] if r["ratio"] is not None]
    return bool(rows) and max(rows, key=lambda r: r["d"])["ratio"] > 1


def speed_claim(summaries: list[dict]) -> tuple[str, str]:
    """The speed claim over every supported comparison, in two parts, with no single ratio a rerun
    would move.

    The lead rounds the largest ratio down to a power of ten and names its method ("Over 100x
    faster on PC"). The rest says how many of the other supported methods with a published ratio
    are faster against every package they were compared with: "the other" when all are, "most
    other" when more than half are, and nothing when fewer are. A method is the registry's, so
    every comparison of one method, with any of its settings, has to be faster.
    """
    found = supported(summaries)
    method, row = _largest(found)
    floor = 10 ** math.floor(math.log10(row["ratio"]))
    lead = f"{'Over' if row['ratio'] > floor else 'At least'} {ratio_text(floor)} faster on "
    lead += method["method"]
    others: dict[str, bool] = {}
    for m in found:
        if m["spec"] != method["spec"] and any(r["ratio"] is not None for r in m["rows"]):
            others[m["spec"]] = others.get(m["spec"], True) and faster(m)
    count = sum(others.values())
    if count == len(others):
        rest = "faster on the other supported methods"
    elif 2 * count > len(others):
        rest = "faster on most other supported methods"
    else:
        rest = ""
    return lead, rest


def claim_sentence(summaries: list[dict]) -> str:
    """The speed claim as one sentence: the homepage's, under its heading, and the README's."""
    lead, rest = speed_claim(summaries)
    return f"{lead}, and {rest}." if rest else f"{lead}."


def fastest(summary: dict) -> tuple[dict, dict]:
    """Return the comparison and row with the largest ratio in one summary; its headline must be
    the claim its numbers support."""
    method, row = _largest(supported([summary]))
    if summary.get("headline") != claim_sentence([summary]):
        raise ValueError(
            f"headline {summary.get('headline')!r} is not the claim the numbers support, "
            f"{claim_sentence([summary])!r} ({method['method']} at d = {row['d']}: "
            f"{ratio_text(row['ratio'])})"
        )
    return method, row


def home_claim(summaries: list[dict]) -> str:
    """The homepage's speed claim, the line under its heading."""
    return claim_sentence(summaries)


def ratio_label(row: dict) -> str:
    """A row's speedup as the page prints it, with "≥" before a lower bound."""
    return ("≥" if row["at_least"] else "") + ratio_text(row["ratio"])


# ---- the README's speedup cards --------------------------------------------------------------
# The methods the README chart shows, in order, each from a release's summary: (folder in
# RELEASES, method).
PANEL = (
    ("alpha-2026-09", "PC"),
    ("alpha-2026-09", "FCI"),
    ("alpha-2026-09", "GES"),
    ("score-search-2026-10", "XGES"),
    ("score-search-2026-10", "Tabu search"),
    ("alpha-2026-09", "DirectLiNGAM"),
)


def speedup_text(speed: float) -> str:
    """A speedup as the README chart prints it: the slowest package is 1x."""
    return "1x" if speed == 1 else ratio_text(speed)


def speedup_slides(summaries: list[dict]) -> list[tuple[str, str, list[tuple[str, float, str]]]]:
    """Each ``PANEL`` method's speedups at its largest size with a published ratio, fastest first.

    A slide is the method, its size as it prints, and its bars: each package, its speed relative
    to the slowest package (1x), and the label it prints. ``summaries`` are in ``RELEASES`` order,
    and each method comes from its own release's summary, rows at ``MEAN_DEGREE``.

    The packages at that size come from the comparisons run with the same data, hardware, and
    build as the one with the ratio. Every bar comes from Andrey's paired ratios: Andrey's is its
    ratio against the slowest; another package's is that ratio divided by Andrey's ratio against
    it. No run pairs two other packages, so theirs is a quotient of medians. Every package timed
    at that size must have a ratio. A ratio published as a lower bound prints with "≥"; a bound
    divides into no other package's bar, so its slide has one other package. Every method must be
    supported by the registry.
    """
    by_release = dict(zip(RELEASES, summaries, strict=True))
    keys = ("regime", "hardware", "build")
    slides = []
    for release, name in PANEL:
        found = [m for m in comparisons(by_release[release]) if m["method"] == name]
        if not found:
            raise ValueError(f"the README chart needs {name} in {release}")
        if found[0]["status"] != "supported":
            raise ValueError(f"the README chart shows supported methods; {name} is not")
        claimed = [(row["d"], m) for m in found for row in m["rows"] if row["ratio"]]
        if not claimed:
            raise ValueError(f"the README chart needs a published {name} ratio")
        d, lead = max(claimed, key=lambda item: item[0])
        runs = [m for m in found if all(m[k] == lead[k] for k in keys)]
        rows = {m["comparator"]: r for m in runs for r in m["rows"] if r["d"] == d}
        andrey = {r["andrey_s"] for r in rows.values()}
        if len(andrey) != 1:
            raise ValueError(f"{name} d = {d}: Andrey's time differs between comparisons")
        timed = {p: r for p, r in rows.items() if r["comparator_s"] is not None}
        if unpaired := [p for p, r in timed.items() if r["ratio"] is None]:
            raise ValueError(f"{name} d = {d}: no published ratio against {', '.join(unpaired)}")
        if len(timed) > 1 and any(r["at_least"] for r in timed.values()):
            raise ValueError(f"{name} d = {d}: a lower bound divides into no other package's bar")
        slowest = max(timed, key=lambda p: timed[p]["ratio"])
        ours = timed[slowest]["ratio"]
        bars = [("Andrey", ours, ratio_label(timed[slowest]))]
        for package, row in timed.items():
            speed = ours / row["ratio"]
            bars.append((package, speed, speedup_text(speed)))
        slides.append((name, f"{d:,} {lead['unit']}", sorted(bars, key=lambda bar: -bar[1])))
    return slides


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def claimed_comparisons(summaries: list[dict]) -> list[dict]:
    """The comparisons the charts cover: the supported ones with a published ratio at some size."""
    return [m for m in supported(summaries) if any(r["ratio"] is not None for r in m["rows"])]


# ---- speed across methods --------------------------------------------------------------------
def family_intro(methods: list[dict]) -> str:
    """Say what a family's charts show and which packages they cover."""
    packages = sorted({m["comparator"] for m in methods}, key=str.lower)
    return (
        f"Andrey was measured alongside {listed(packages)} on the same datasets and the same "
        "machine at each size. Each chart plots the median fit time against the number of "
        "variables, or of rows for a pair of variables: up and to the left is better."
    )


def time_groups(summaries: list[dict]) -> list[tuple[str, list[dict]]]:
    """The comparisons a chart each: a method's comparisons share one chart when they come from
    the same runs (Andrey's times agree at every common size), otherwise one chart apiece."""
    groups: list[tuple[str, list[dict]]] = []
    for method in claimed_comparisons(summaries):
        same = [
            g
            for g in groups
            if g[0] == method["method"]
            and all(
                a["andrey_s"] == b["andrey_s"]
                for a in g[1][0]["rows"]
                for b in method["rows"]
                if a["d"] == b["d"]
            )
        ]
        if same:
            same[0][1].append(method)
        else:
            groups.append((method["method"], [method]))
    return groups


def log_scale(lo: float, hi: float, a: float, b: float):
    """Map ``lo..hi`` on a log scale onto ``a..b``."""
    span = math.log10(hi / lo)
    return lambda v: a + (b - a) * math.log10(v / lo) / span


# The sizes a time chart labels on its axis, those inside the charts' range.
SIZE_TICKS = (5, 10, 20, 50, 100, 200, 400, 800, 1600, 3200)


def time_chart(name: str, methods: list[dict], xr: tuple[float, float], sizes: list[int]) -> str:
    """Fit time (x) against problem size (y) for Andrey and each package, both on log scales.

    Up and to the left is better: larger problems in less time. The size is in the comparisons'
    ``unit``, variables or rows. A hollow point is a size with no paired ratio; a package past its
    time cap has no point at that size.
    """
    unit = methods[0]["unit"]
    w, h, left, right, top, bottom = 320, 232, 44, 306, 22, 196
    x = log_scale(xr[0], xr[1], left, right)
    y = log_scale(sizes[0], sizes[-1], bottom, top)
    grid = []
    for k in range(round(math.log10(xr[0])), round(math.log10(xr[1])) + 1):
        v = 10.0**k
        grid.append(
            f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{bottom}" class="tc-grid"/>'
            f'<text x="{x(v):.1f}" y="{bottom + 14}" text-anchor="middle">{seconds_axis(v)}</text>'
        )
    for d in (v for v in SIZE_TICKS if sizes[0] <= v <= sizes[-1]):
        grid.append(
            f'<line x1="{left}" x2="{right}" y1="{y(d):.1f}" y2="{y(d):.1f}" class="tc-grid"/>'
            f'<text x="{left - 6}" y="{y(d) + 3:.1f}" text-anchor="end">{d:,}</text>'
        )
    ours = [(r["d"], r["andrey_s"], r["ratio"] is not None) for r in methods[0]["rows"]]
    series = [("Andrey", ours, "tc-andrey")]
    series += [
        (
            m["comparator"],
            [(r["d"], r["comparator_s"], r["ratio"] is not None) for r in m["rows"]],
            f"tc-other tc-other-{i}",
        )
        for i, m in enumerate(methods)
    ]
    marks = []
    for label, points, cls in series:
        timed = [(d, t, claim) for d, t, claim in points if t is not None]
        if not timed:
            continue
        path = " ".join(f"{x(t):.1f},{y(d):.1f}" for d, t, _ in timed)
        marks.append(f'<polyline class="tc-line {cls}" points="{path}"/>')
        for d, t, claim in timed:
            tip = f"{label}, {d:,} {unit}: {seconds_text(t)}" + (
                "" if claim else ", no paired ratio"
            )
            marks.append(
                f'<circle class="tc-point {cls}{"" if claim else " is-open"}" cx="{x(t):.1f}" '
                f'cy="{y(d):.1f}" r="4"><title>{escape(tip)}</title></circle>'
            )
    best = max(
        (r for m in methods for r in m["rows"] if r["ratio"] is not None),
        key=lambda r: r["ratio"],
    )
    peers = " and ".join(m["comparator"] for m in methods)
    legend = "".join(
        f'<span class="tc-key {cls}"><i aria-hidden="true"></i>{escape(label)}</span>'
        for label, _, cls in series
    )
    return (
        '<figure class="tchart"><figcaption>'
        f'<span class="tchart-name"><strong>{escape(name)}</strong></span>'
        f'<span class="tchart-up">up to <b>{escape(ratio_label(best))}</b></span></figcaption>'
        f'<p class="tc-legend">{legend}</p>'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(name)}: fit time against '
        f'the number of {unit} for Andrey and {escape(peers)}">'
        f'<g class="tc-axis">{"".join(grid)}</g>'
        f'<text class="tc-title" x="{left}" y="{h - 4}">seconds (log)</text>'
        f'<text class="tc-title" x="4" y="{top - 8}">{unit}</text>'
        f"{''.join(marks)}</svg></figure>"
    )


def seconds_axis(v: float) -> str:
    """A decade of seconds as an axis label: 0.01, 0.1, 1, 10, 100, 1k, 10k."""
    return {1000: "1k", 10000: "10k"}.get(v, f"{v:g}")


def speed_figures(summaries: list[dict]) -> list[tuple[str, str]]:
    """A chart per method (or per run where a method's comparisons differ), each with its largest
    speedup, as (registry family, chart). The charts share axes so they compare at a glance: one
    time axis, and one size axis per unit."""
    # A speedup chart: a comparison where Andrey is not faster stays in the release cards' tables.
    groups = [(n, ms) for n, ms in time_groups(summaries) if all(faster(m) for m in ms)]
    times = [
        t
        for _, methods in groups
        for m in methods
        for r in m["rows"]
        for t in (r["andrey_s"], r["comparator_s"])
        if t is not None
    ]
    xr = (10 ** math.floor(math.log10(min(times))), 10 ** math.ceil(math.log10(max(times))))
    sizes: dict[str, set[int]] = {}
    for _, methods in groups:
        for m in methods:
            sizes.setdefault(m["unit"], set()).update(r["d"] for r in m["rows"])
    return [
        (methods[0]["family"], time_chart(name, methods, xr, sorted(sizes[methods[0]["unit"]])))
        for name, methods in groups
    ]


def speed_chart(summaries: list[dict]) -> str:
    """The speed charts on a belt, for the benchmark page and the live demo."""
    return belt([figure for _, figure in speed_figures(summaries)])


def belt(figures: list[str]) -> str:
    """Figures on a belt that slides sideways in a loop. The second copy makes the loop seamless
    and is hidden from assistive technology; the duration grows with the count, so the speed
    holds."""
    shown = "".join(figures)
    copy = re.sub(r'<figure class="([^"]*)">', r'<figure class="\1" aria-hidden="true">', shown)
    return (
        f'<div class="tc-belt"><div class="tc-track" style="--n: {len(figures)}">'
        f'{shown}<div class="tc-copy" aria-hidden="true">{copy}</div></div></div>'
    )


def exceptions(summary: dict) -> list[str]:
    """Every cell where Andrey is slower, has no paired ratio, or a package ran past its cap."""
    notes = []
    for method in ordered(summary):
        name, comparator, rows = method["method"], method["comparator"], method["rows"]
        timed = [r for r in rows if r["andrey_s"] is not None and r["comparator_s"] is not None]
        if timed and all(r["ratio"] is None for r in rows):
            # The charts skip this comparison, and the table below lists its rows, so the notes
            # say nothing about it beyond a run that hit its cap.
            notes += capped_notes(name, comparator, rows)
            continue
        for r in rows:
            if r["ratio"] is not None and r["ratio"] < 1:
                notes.append(
                    f"{name} is slower than {comparator} at d = {r['d']:,}: "
                    f"{ratio_text(r['ratio'])}, {seconds_text(r['andrey_s'])} compared with "
                    f"{seconds_text(r['comparator_s'])}."
                )
        unpaired = [r["d"] for r in timed if r["ratio"] is None]
        if unpaired:
            notes.append(
                f"No paired datasets for {name} compared with {comparator} at "
                f"{sizes_text(unpaired)}, so no ratio."
            )
        notes += capped_notes(name, comparator, rows)
    return notes


def capped_notes(name: str, comparator: str, rows: list[dict]) -> list[str]:
    """Name the sizes where a package ran past its time cap."""
    notes = []
    for side, key in ((comparator, "comparator_s"), ("Andrey", "andrey_s")):
        capped = [r["d"] for r in rows if r[key] is None]
        if capped:
            notes.append(
                f"{side} did not finish {name} within the time cap at {sizes_text(capped)}."
            )
    return notes


# ---- quality in the same comparisons ---------------------------------------------------------
def error_comparisons(methods: list[dict]) -> list[dict]:
    """Each size where both packages have a graph, per package: Andrey's errors against theirs.

    The errors are each comparison's own count (:func:`error_kind`). The change is a share of the
    other package's errors, rounded against Andrey: fewer rounds down and more rounds up. Against
    a package with no errors it is a count. Past a time cap a package has no graph to count.
    """
    out = []
    for d in sorted({r["d"] for m in methods for r in m["rows"]}):
        for method in methods:
            row = next((r for r in method["rows"] if r["d"] == d), None)
            if row is None or row["andrey_errors"] is None or row["comparator_errors"] is None:
                continue
            ours, theirs = row["andrey_errors"], row["comparator_errors"]
            if abs(ours - theirs) < 1e-9:
                kind, change, label = "same", 0.0, "same"
            elif theirs == 0:
                kind, change, label = "more", 100.0, f"{number_text(ours)} more"
            else:
                change = 100 * abs(ours - theirs) / theirs
                kind = "fewer" if ours < theirs else "more"
                whole = math.floor(change) if kind == "fewer" else math.ceil(change)
                label = f"{whole}% {kind}" if whole else f"<1% {kind}"
            out.append(
                {
                    "d": d,
                    "method": method,
                    "row": row,
                    "kind": kind,
                    "change": change,
                    "label": label,
                }
            )
    return out


def quality_intro(summaries: list[dict]) -> str:
    """Say what an error is in each count the comparisons use, and that releases are tested
    against true graphs (the quality gate, which every publish job needs)."""
    kinds = dict.fromkeys(m["error_kind"] for m in claimed_comparisons(summaries))
    errors = ", or, ".join(ERRORS[kind][1] for kind in kinds)
    return (
        f"An error is {errors}. Andrey's graphs are as accurate as the reference packages' (the "
        "other packages it is compared with) on the same data: the charts give both packages' "
        "errors at each size, and every release is tested against the true graphs that generated "
        "the data."
    )


def accuracy_card(name: str, methods: list[dict]) -> str:
    """One method's comparisons as a table: each size, each package, and Andrey's errors against
    theirs. Every card has the same columns, so rows line up across cards; the bar grows with the
    share of errors Andrey avoids."""
    rows, first_at = [], set()
    scored = error_comparisons(methods)
    for c in scored:
        method, row, d = c["method"], c["row"], c["d"]
        size = "" if d in first_at else f"{d:,}"
        first_at.add(d)
        if c["kind"] == "fewer":
            mark = f'<span class="rt-bar" style="--f: {min(c["change"], 100):.1f}%"></span>'
        elif c["kind"] == "same":
            mark = '<span class="rt-dot"></span>'
        else:
            mark = ""
        tip = (
            f"{method['comparator']}, {d:,} {method['unit']}: Andrey "
            f"{number_text(row['andrey_errors'])} errors, {method['comparator']} "
            f"{number_text(row['comparator_errors'])}"
        )
        rows.append(
            f'<span class="rt-d">{size}</span><span class="rt-who">'
            f"{escape(method['comparator'])}</span>"
            f'<span class="rt-track" title="{escape(tip)}">{mark}</span>'
            f'<span class="rt-v">{escape(c["label"])}</span>'
        )
    good = sum(c["kind"] != "more" for c in scored)
    return (
        '<figure class="tchart rtable"><figcaption>'
        f'<strong class="rt-name">{escape(name)}</strong>'
        f'<span class="rt-score"><b>{good} of {len(scored)}</b>'
        "<span>same or fewer errors</span></span></figcaption>"
        f'<div class="rt-grid"><span class="rt-h rt-d">{methods[0]["unit"].capitalize()}</span>'
        '<span class="rt-h">Compared with</span><span class="rt-h">Fewer errors &rarr;</span>'
        f'<span class="rt-h"></span>{"".join(rows)}</div></figure>'
    )


def accuracy_cards(summaries: list[dict]) -> list[str]:
    """A card per speed chart, for the comparisons with a graph each."""
    groups = [(n, ms) for n, ms in time_groups(summaries) if error_comparisons(ms)]
    return [accuracy_card(name, methods) for name, methods in groups]


def accuracy_rows(summaries: list[dict]) -> list[dict]:
    """Per method and package compared: both packages' errors summed over the sizes where both
    have a graph, and how many of those sizes Andrey has the same or fewer."""
    rows = []
    for name, methods in time_groups(summaries):
        by: dict[str, dict] = {}
        for c in error_comparisons(methods):
            row = by.setdefault(
                c["method"]["comparator"],
                {"name": name, "comparator": c["method"]["comparator"], "ours": 0.0, "theirs": 0.0,
                 "good": 0, "sizes": 0},
            )  # fmt: skip
            row["ours"] += c["row"]["andrey_errors"]
            row["theirs"] += c["row"]["comparator_errors"]
            row["good"] += c["kind"] != "more"
            row["sizes"] += 1
        rows += by.values()
    return rows


def accuracy_counts(summaries: list[dict]) -> tuple[int, int]:
    """How many comparisons have the same or fewer errors for Andrey, and how many there are."""
    rows = accuracy_rows(summaries)
    return sum(r["good"] for r in rows), sum(r["sizes"] for r in rows)


def accuracy_line(summaries: list[dict]) -> str:
    """The accuracy tile's line: how many comparisons have the same or fewer errors, in amber."""
    good, total = accuracy_counts(summaries)
    return (
        '<p class="tile-text tile-score">Same or fewer errors in '
        f"<b>{good} of {total}</b> comparisons</p>"
    )


# The accuracy figure's frame: the label column, the plot's width, a row's height, and its scale's
# top (Andrey's errors as a share of the other package's).
ACC_LABEL, ACC_PLOT, ACC_ROW, ACC_TOP = 280, 340, 30, 120


def accuracy_figure(summaries: list[dict]) -> str:
    """Andrey against the other package, a row per method: the other package's errors as a ring
    at 100%, Andrey's as a cobalt dot at its share of them, both summed over the sizes the two
    share. Above it, in amber, how many of those comparisons have the same or fewer errors."""
    rows = accuracy_rows(summaries)
    good, total = accuracy_counts(summaries)

    def x(share: float) -> float:
        return ACC_LABEL + ACC_PLOT * min(share, ACC_TOP) / ACC_TOP

    marks = []
    for i, r in enumerate(rows):
        y = 40 + i * ACC_ROW
        share = 100 * r["ours"] / r["theirs"] if r["theirs"] else 100.0
        change = share - 100
        label = "same" if abs(change) < 0.5 else f"{change:+.0f}%".replace("-", "\u2212")
        tip = (
            f"{r['name']} with {r['comparator']}: Andrey {number_text(r['ours'])} errors, "
            f"{r['comparator']} {number_text(r['theirs'])}"
        )
        marks.append(
            f"<g><title>{escape(tip)}</title>"
            f'<text class="acc-name" x="0" y="{y + 4}">{escape(r["name"])}'
            f'<tspan class="acc-with" dx="8">{escape(r["comparator"])}</tspan></text>'
            f'<line class="acc-link" x1="{x(share):.1f}" x2="{x(100):.1f}" y1="{y}" y2="{y}"/>'
            f'<circle class="acc-ref" cx="{x(100):.1f}" cy="{y}" r="7"/>'
            f'<circle class="acc-ours" cx="{x(share):.1f}" cy="{y}" r="5"/>'
            f'<text class="acc-change" x="{ACC_LABEL + ACC_PLOT + 14}" y="{y + 4}">{label}</text>'
            "</g>"
        )
    height = 40 + len(rows) * ACC_ROW
    ticks = "".join(
        f'<line class="acc-grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="22" y2="{height - 14}"/>'
        f'<text class="acc-tick" x="{x(t):.1f}" y="14">{t}%</text>'
        for t in (0, 50, 100)
    )
    return (
        '<figure class="acc-figure">'
        f'<figcaption><b class="acc-score">{good} of {total}</b> comparisons with the same or '
        "fewer errors. A comparison is one method against one package at one size."
        '<span class="acc-note"> Each row sums both packages\' errors over the sizes '
        "they share: the ring is the other package's, the dot Andrey's, as a share of them."
        "</span></figcaption>"
        '<div class="acc-scroll" role="region" aria-label="Accuracy by method" tabindex="0">'
        f'<svg class="acc-chart" viewBox="0 0 {ACC_LABEL + ACC_PLOT + 70} {height}" role="img" '
        f'aria-label="Andrey has the same or fewer errors in {good} of {total} comparisons">'
        f"{ticks}{''.join(marks)}</svg></div></figure>"
    )


def accuracy_chart(summaries: list[dict]) -> str:
    """The accuracy cards on a belt."""
    return belt(accuracy_cards(summaries))


def notes_list(notes: list[str]) -> str:
    """The exceptions as a list under the chart; nothing when there are none."""
    if not notes:
        return ""
    return '<ul class="speed-notes">' + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>"


def speed_source(summary: dict) -> str:
    """Where the numbers come from, and what SHD measures."""
    return (
        f"Numbers from the benchmark summary generated on {escape(summary['generated'])}. SHD is "
        "the structural Hamming distance to the true graph; lower is better."
    )


def speed_table(summary: dict, *, notes: bool = True) -> str:
    """Every row of an SHD summary, the chart's table-view equivalent.

    With ``notes=False`` the exceptions and the source line are left out.
    """
    head = (
        "<tr><th scope='col'>d</th><th scope='col'>Datasets</th><th scope='col'>Andrey</th>"
        "<th scope='col'>Other package</th><th scope='col'>Speedup</th>"
        "<th scope='col'>Andrey SHD</th><th scope='col'>Other SHD</th>"
        "<th scope='col'>Empty-graph SHD</th></tr>"
    )
    body = []
    for method in ordered(summary):
        body.append(
            f"<tr><th scope='rowgroup' colspan='8'><strong>{escape(method['method'])}</strong> "
            f"with {escape(method['comparator'])}: {REGIMES[method['regime']]}; "
            f"{escape(method['hardware'])}; Andrey {escape(method['build'])}"
            "</th></tr>"
        )
        for r in method["rows"]:
            ratio = "-" if r["ratio"] is None else ratio_text(r["ratio"])
            cells = [
                f"{r['d']:,}",
                str(r["seeds"]),
                seconds_text(r["andrey_s"]),
                seconds_text(r["comparator_s"]),
                ratio,
                number_text(r["andrey_shd"]),
                number_text(r["comparator_shd"]),
                number_text(r["empty_shd"]),
            ]
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    listed_notes = notes_list(exceptions(summary)) if notes else ""
    source = f'<p class="generated">{speed_source(summary)}</p>' if notes else ""
    return (
        '<details class="speed-table"><summary>Show every measurement</summary>'
        f"{listed_notes}"
        f'<div class="table-scroll"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody>'
        f"</table></div>{source}</details>"
    )


# ---- the method list -------------------------------------------------------------------------
def returns_text(spec) -> str:  # noqa: ANN001 -- an andrey.spec MethodSpec
    """What a method returns, in one or two words."""
    if spec.family == "pairwise":
        return "Two-node DAG"
    if spec.output.structure_type == "temporal":
        return "Temporal graph"
    return spec.output.graph_kind.upper()


def first_clause(summary: str) -> str:
    """The registry summary up to its first colon, semicolon, or sentence end."""
    return re.split(r"(?<=[a-z)]{2})\. |[:;] ", summary, maxsplit=1)[0].rstrip(".")


def seal_points() -> str:
    """A twelve-point seal around (12, 12) in a 24-unit box: the radius alternates 11 and 9.3."""
    return " ".join(
        f"{12 + r * math.sin(math.pi * k / 12):.2f},{12 - r * math.cos(math.pi * k / 12):.2f}"
        for k, r in ((k, 11 if k % 2 == 0 else 9.3) for k in range(24))
    )


# The benchmarked methods' seal, a badge with a check, and what it means.
BENCHMARKED_SEAL = (
    '<svg class="method-seal" viewBox="0 0 24 24" aria-label="Benchmarked" role="img">'
    f'<polygon points="{seal_points()}"/><polyline points="7.8 12.4 10.6 15.2 16.2 9.2"/></svg>'
)
BENCHMARKED = "Benchmarked"
EXPERIMENTAL = '<span class="method-status">Experimental</span> '


def methods(summaries: list[dict]) -> tuple[str, str]:
    """Every registered method by family, marking the benchmarked and the experimental ones. A
    method is benchmarked when a release's summary compares it with another package and the
    registry supports it."""
    specs = list_specs()
    unknown = {s.family for s in specs} - SPEC_FAMILIES.keys()
    if unknown:
        raise ValueError(f"name the registry families {sorted(unknown)} in SPEC_FAMILIES")
    measured = {m["spec"] for m in supported(summaries)}
    blocks = []
    for family, label in SPEC_FAMILIES.items():
        items = []
        for spec in (s for s in specs if s.family == family):
            bench = (
                '<a class="method-bench" href="docs/benchmarks.html" '
                f'aria-describedby="bench-{spec.name}">'
                f"{BENCHMARKED_SEAL}"
                f'<span class="method-tip" role="tooltip" id="bench-{spec.name}">{BENCHMARKED}'
                "</span></a>"
                if spec.name in measured
                else ""
            )
            status = EXPERIMENTAL if spec.status == "experimental" else ""
            items.append(
                f'<li class="method"><div class="method-head"><span class="method-title">'
                f'<a class="method-name" href="docs/code/generated/andrey.{spec.name}.html">'
                f"<code>andrey.{spec.name}</code></a>{bench}</span>"
                f'<span class="method-kind">{returns_text(spec)}</span></div>'
                f'<p class="method-what">{status}{escape(first_clause(spec.summary))}</p></li>'
            )
        if items:
            blocks.append(
                f'<div class="method-family"><h3>{label}</h3><ul>{"".join(items)}</ul></div>'
            )
    title = f"{len(specs)} methods in {len(blocks)} families"
    return title, f'<div class="methods">{"".join(blocks)}</div>'


def lede() -> str:
    """The hero's first sentence: what Andrey is, and every family the registry has methods in."""
    present = {s.family for s in list_specs()}
    if unknown := present - FAMILY_TERMS.keys():
        raise ValueError(f"name the registry families {sorted(unknown)} in FAMILY_TERMS")
    terms = [term for family, term in FAMILY_TERMS.items() if family in present]
    return (
        "Andrey is a Python package for learning causal graphs from observational data, with "
        f"{listed(terms)} methods."
    )


# ---- the README chart ------------------------------------------------------------------------
README_CHART = "benchmark-chart-{theme}.svg"  # in .github/readme/; the README links both themes
# The README chart is one card per ``PANEL`` method: two rows of three, never more.
README_COLUMNS, README_ROWS = 3, 2
CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def readme_slides(summary: dict) -> list[tuple[str, str, list[tuple[str, float, str]]]]:
    """The README chart's cards, from ``summary`` and the later releases' summaries."""
    slides = speedup_slides(release_summaries(summary))
    if len(slides) > README_COLUMNS * README_ROWS:
        raise ValueError(
            f"the README chart holds {README_ROWS} rows of {README_COLUMNS} cards, and PANEL "
            f"has {len(slides)} methods; replace a method in PANEL instead of adding one"
        )
    return slides


def readme_chart(summary: dict, palette: andrey.viz.Palette) -> str:
    """The speedup of each ``PANEL`` method as a standalone SVG: one card per method, Andrey
    accented, from :func:`speedup_slides`. A README can show only an image. Each card is as tall as
    its own bars; a row of cards starts below the tallest card of the row above. The chart sits on
    an opaque card, so it stays legible where a renderer shows the light file on a dark page.
    """
    slides = readme_slides(summary)
    width, pad, gap, columns = 960, 24, 16, README_COLUMNS
    card_w = (width - 2 * pad - gap * (columns - 1)) / columns
    top = pad + 44

    def card_h(bars: list) -> int:
        return 70 + len(bars) * 30

    row_tops, y_row = [], top
    for start in range(0, len(slides), columns):
        row_tops.append(y_row)
        y_row += max(card_h(bars) for _, _, bars in slides[start : start + columns]) + gap
    height = y_row - gap + pad
    ink, muted, line, accent = palette.ink, palette.muted, palette.line, palette.accent

    def text(x: float, y: float, content: str, *, size=13, fill=ink, weight=400, anchor="start"):
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">'
            f"{escape(content)}</text>"
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:.0f}" '
        f'viewBox="0 0 {width} {height:.0f}" font-family="{escape(CHART_FONT)}" role="img" '
        'aria-labelledby="chart-title chart-desc">',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1:.0f}" rx="16" '
        f'fill="{palette.paper}" stroke="{line}"/>',
        '<title id="chart-title">Speedup on the same data</title>',
        f'<desc id="chart-desc">{escape(readme_chart_alt(summary))}</desc>',
        text(pad, pad + 16, "Speedup on the same data", size=20, weight=600),
    ]
    for i, (name, size, bars) in enumerate(slides):
        x = pad + (i % columns) * (card_w + gap)
        y = row_tops[i // columns]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{card_w:.1f}" height="{card_h(bars)}" rx="12" '
            f'fill="{palette.surface}" stroke="{line}"/>'
        )
        # A long name steps down a size so it clears the size label on the right.
        parts.append(text(x + 18, y + 26, name, size=16 if len(name) <= 14 else 14, weight=600))
        parts.append(text(x + card_w - 18, y + 26, size, size=12.5, fill=muted, anchor="end"))
        top_speed = bars[0][1]
        track_x, track_w = x + 112, card_w - 112 - 70
        for j, (package, speed, label) in enumerate(bars):
            cy = y + 62 + j * 30
            ours = package == "Andrey"
            parts.append(
                text(x + 18, cy, package, fill=ink if ours else muted, weight=600 if ours else 400)
            )
            bar = max(track_w * speed / top_speed, 6)
            fill = f'fill="{accent}"' if ours else f'fill="{muted}" fill-opacity="0.45"'
            parts.append(
                f'<rect x="{track_x:.1f}" y="{cy - 5:.1f}" width="{bar:.1f}" height="10" '
                f'rx="5" {fill}/>'
            )
            parts.append(text(track_x + bar + 10, cy, label, weight=600))
    parts.append("</svg>")
    return "".join(parts) + "\n"


def readme_chart_alt(summary: dict) -> str:
    """The README chart's alt text: each method's speedup, as its card shows it."""
    parts = [
        f"{name} {next(label for package, _, label in bars if package == 'Andrey')} at {size}"
        for name, size, bars in readme_slides(summary)
    ]
    return "Andrey's speedup over the slowest package on the same data: " + ", ".join(parts)


# ---- the example, the wordmark, and the page --------------------------------------------------
def drawing(
    structure,
    label: str,
    positions: dict | None = None,
    size: tuple[int, int] | None = None,
) -> str:
    """Draw a structure with its column names, in the page's theme colors.

    ``size`` sets the drawing's own canvas; a smaller canvas shows the labels larger on the page.
    """
    width, height = size or (None, None)
    markup = str(
        andrey.viz.draw(
            structure,
            palette=TOKEN_PALETTE,
            background=False,
            positions=positions,
            width=width,
            height=height,
        )
    )
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svg = ET.fromstring(markup)
    svg.set("role", "img")
    svg.set("aria-label", label)
    return ET.tostring(svg, "unicode")


def run_script(name: str) -> tuple[str, str, dict]:
    """Run one of the site's example scripts; return its source, what it prints, and its names."""
    path = SITE / name
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        namespace = runpy.run_path(str(path), run_name="__main__")
    return path.read_text(encoding="utf-8"), printed.getvalue().strip(), namespace


def parts_layout(structure: GraphStructure) -> dict[int, tuple[float, float]]:
    """``andrey.viz.layout`` for each connected part of ``structure``, the parts in a grid, and a
    part laid out as a column turned into a row.

    ``layout`` layers nodes along directed edges, so a graph of small parts with few of those
    stacks into one tall column.
    """
    marks = structure.to_numpy()
    graph = nx.from_numpy_array((marks != 0).astype(int))
    parts = sorted(sorted(part) for part in nx.connected_components(graph))
    across = math.ceil(math.sqrt(len(parts)))
    gap = 0.5  # between parts, in a part's width
    positions = {}
    for i, nodes in enumerate(parts):
        row, column = divmod(i, across)
        part = GraphStructure.from_numpy(marks[np.ix_(nodes, nodes)], kind=structure.kind)
        local = andrey.viz.layout(part, engine="builtin")  # a ring for each undirected part
        if len({x for x, _ in local.values()}) == 1:
            local = {k: (y, x) for k, (x, y) in local.items()}
        for k, (x, y) in local.items():
            positions[nodes[k]] = (column * (1 + gap) + x, row * (1 + gap) + y)
    span = max(max(x for x, _ in positions.values()), max(y for _, y in positions.values())) or 1
    return {k: (x / span, y / span) for k, (x, y) in positions.items()}


def run_example() -> tuple[str, str, str]:
    """Run ``site/example.py``; return its source, what it prints, and its graph as an SVG."""
    code, printed, namespace = run_script("example.py")
    structure = namespace["out"].structure
    label = "The graph PC recovers from the Sachs data"
    return code, printed, drawing(structure, label, parts_layout(structure), size=(360, 260))


# The three methods answer differently only on season-rain and season-sprinkler; the drawings'
# edge colors (undirected, circle, directed) carry the difference. What each drawing shows about
# those two edges, by method:
INTERFACE_CAPTIONS = {
    "pc": (
        "PC",
        "a CPDAG. The data cannot say which way the two edges from season point, so they stay "
        "plain lines.",
    ),
    "fci": (
        "FCI",
        "a PAG. Circles on both ends: either direction, or a hidden common cause, fits the data.",
    ),
    "direct_lingam": (
        "DirectLiNGAM",
        "a DAG. Non-Gaussian noise orients every edge, those two included.",
    ),
}


def run_interface() -> tuple[str, str, str]:
    """Run ``site/interface.py``; return the loop it shows, what it prints, and a figure per
    method."""
    source, printed, namespace = run_script("interface.py")
    graphs = namespace["graphs"]
    if graphs.keys() != INTERFACE_CAPTIONS.keys():
        raise ValueError(f"caption every method site/interface.py runs: {sorted(graphs)}")
    # The page shows the loop; the line that keeps each graph for the drawings is the build's.
    loop = source[source.index("for learn in") :].splitlines()
    shown = "\n".join(line for line in loop if "graphs[" not in line)
    # One layout for all three, from the DAG, so the drawings differ only in their edge marks.
    positions = andrey.viz.layout(graphs["direct_lingam"])
    figures = "".join(
        '<figure class="interface-figure">'
        + drawing(
            graphs[key],
            f"The graph {name} returns",
            positions,
            size=(300, 210),
        )
        + f"<figcaption><strong>{name}</strong>: {escape(what)}</figcaption></figure>"
        for key, (name, what) in INTERFACE_CAPTIONS.items()
    )
    return shown, printed, figures


# Each edge as [source, target, type], named by the CSV header: the rows oriented_edges() returns.
AGENT_QUERY = ".structure | .labels as $n | .edges[] | [$n[.source], $n[.target], .type]"


# What an agent runs, in order: find a method, read its parameters, run it. The jq filters are
# applied here, to the command line's own JSON; a test checks them against jq.
AGENT_LIST = "andrey list | jq -r '.[] | [.name, .output, .status] | @tsv'"
AGENT_HELP = "andrey run pc --help --json | jq -c '.method.params.properties.alpha'"
AGENT_RUN = f"andrey run pc --data sachs.csv | jq -c '{AGENT_QUERY}'"
AGENT_LISTED = 3  # the list's first rows; the rest are counted


def run_cli(args: list[str]) -> dict | list:
    """Run the command line as a program would, so it answers in JSON; fail on a non-zero exit."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):  # not a terminal, so the answer is JSON
        status = andrey_cli(args)
    if status != 0:
        raise RuntimeError(f"andrey {' '.join(args)} exited with {status}")
    return json.loads(printed.getvalue())


def compact(value) -> str:  # noqa: ANN001
    """JSON as jq -c prints it."""
    return json.dumps(value, separators=(",", ":"))


def agent_frames() -> list[tuple[str, list[str], int]]:
    """Run the command line on the example's data as an agent would. Each frame is a command, the
    lines its jq filter prints, and how many of them are left out."""
    _, _, namespace = run_script("example.py")
    with tempfile.TemporaryDirectory(prefix="andrey-site-") as directory:
        data = Path(directory) / "sachs.csv"
        namespace["df"].to_csv(data, index=False)  # the header row names the nodes
        methods = run_cli(["list"])
        contract = run_cli(["run", "pc", "--help", "--json"])
        alpha = contract["method"]["params"]["properties"]["alpha"]
        structure = run_cli(["run", "pc", "--data", str(data)])["structure"]
    listed = ["\t".join((m["name"], m["output"], m["status"])) for m in methods]
    names = structure["labels"]
    edges = [[names[e["source"]], names[e["target"]], e["type"]] for e in structure["edges"]]
    return [
        (AGENT_LIST, listed[:AGENT_LISTED], len(listed) - AGENT_LISTED),
        (AGENT_HELP, [compact(alpha)], 0),
        (AGENT_RUN, [compact(edge) for edge in edges], 0),
    ]


def agent_demo(frames: list[tuple[str, list[str], int]], figure: str) -> str:
    """The agent's three steps in a row, then its session in a terminal beside the graph it
    returned.

    Every line is in the page; the page script only reveals them in turn, and without it, or with
    reduced motion, the session is shown whole.
    """
    count = len(frames[0][1]) + frames[0][2]
    steps = (
        (
            "Find a method",
            f"<code>andrey list</code> names all {count} methods, with the graph each returns.",
        ),
        (
            "Read its contract",
            "<code>andrey run pc --help --json</code> gives each parameter's type, "
            "default, and meaning as JSON Schema.",
        ),
        ("Run it", "The run prints the graph as JSON, its nodes named by the data's header row."),
    )
    shown = []
    for command, lines, more in frames:
        lexer = "json" if lines[0].startswith(("{", "[")) else "text"
        body = [
            f'<span class="ln" style="--i: {i}">{code_html(line, lexer)}</span>'
            for i, line in enumerate(lines)
        ]
        if more:
            body.append(f'<span class="ln elide" style="--i: {len(lines)}">... {more} more</span>')
        shown.append(
            '<div class="frame"><div class="cmd"><span class="prompt">$</span> '
            f'<span class="cmd-text">{escape(command)}</span></div>'
            f'<div class="out hl">{"".join(body)}</div><span class="exit">exit 0</span></div>'
        )
    return (
        '<div class="agent-demo"><ol class="agent-steps">'
        + "".join(f'<li class="agent-step"><b>{t}</b><p>{text}</p></li>' for t, text in steps)
        + '</ol><div class="agent-run"><div class="term"><div class="term-bar window-bar">'
        '<button class="term-replay" type="button" hidden>Replay'
        '</button></div><div class="term-body" tabindex="0" role="region" '
        f'aria-label="Agent session">{"".join(shown)}</div></div><figure class="agent-graph">'
        f"{figure}<figcaption>The graph in that JSON, drawn with <code>andrey.viz.draw</code>."
        "</figcaption></figure></div></div>"
    )


# Icons from Feather (MIT, https://github.com/feathericons/feather), and the robot from Lucide (ISC,
# https://github.com/lucide-icons/lucide), drawn at one size and stroke.
ICON = {
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
    "bot": (
        '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/>'
        '<path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>'
    ),
    "chart": (
        '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
    ),
    "graph": (
        '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/>'
        '<circle cx="18" cy="19" r="3"/>'
        '<line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>'
        '<line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>'
    ),
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
    '<circle cx="12" cy="12" r="2"/>',
    "minus": '<line x1="5" y1="12" x2="19" y2="12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
}
# The documentation pages a first visit wants, in the order the band lists them.
DOC_PAGES = (
    (
        "terminal",
        "docs/guides/getting-started.html",
        "Getting started",
        "Install, then recover your first graph.",
    ),
    (
        "graph",
        "docs/guides/graph-types.html",
        "Graph types",
        "What a DAG, CPDAG, PAG, or temporal graph claims.",
    ),
    ("code", "docs/code/index.html", "API reference", "Every method, structure, and command."),
    (
        "chart",
        "docs/benchmarks.html",
        "Benchmarks",
        "Timings, accuracy, limits, and how to rerun them.",
    ),
)


def glyph(icon: str) -> str:
    """One line symbol, inheriting the color of the text beside it."""
    return (
        '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON[icon]}</svg>'
    )


def chip(icon: str, url: str, text: str, *, primary: bool = False) -> str:
    """A link with its symbol, boxed; the hero uses one treatment for all of them."""
    kind = "chip chip--primary" if primary else "chip"
    return f'<a class="{kind}" href="{escape(url)}">{glyph(icon)}{escape(text)}</a>'


# The hero's art, one cover per word of its scroll story (site/brand/art-<name>.jpg), and each
# bento tile's (site/brand/bento/<name>.jpg): a crop of a cover's art at twice the tile's size. A
# family without art shows the stipple ground.
HERO_ART = ("lanes", "contour", "agentic")
FAMILY_ART = {
    "constraint": "constraint",
    "score": "contour",
    "permutation": "permutation",
    "lingam": "lingam",
    "latent": "iceberg",
    "pairwise": "hourglass",
}
TILE_SIZES = {"4": (768, 416), "6": (1168, 416)}
PRIORITY = ' fetchpriority="high"'


def hero_art() -> str:
    """The hero's covers, stacked: the first shows alone without the story, and the story
    (``story.js``) crossfades them with the word, as ``story-step`` ``data-i`` 1 to 3."""
    return "".join(
        f'<img class="hero-cover story-step" data-i="{i}" src="_static/art-{name}.jpg" alt="" '
        f'width="1600" height="640"{PRIORITY if i == 1 else ""}>'
        for i, name in enumerate(HERO_ART, 1)
    )


def tile_art(name: str, size: str) -> str:
    """A tile's art, behind its text; decorative, so it has no text alternative."""
    width, height = TILE_SIZES[size]
    return (
        f'<img class="tile-art" src="_static/bento/{name}.jpg" alt="" width="{width}" '
        f'height="{height}" loading="lazy">'
    )


def hero_chips() -> str:
    """The buttons beside the install line: Get started, and the live demo once its Space is
    public."""
    start = chip("terminal", "docs/guides/getting-started.html", "Get started", primary=True)
    return start + (chip("graph", DEMO_URL, "Live demo") if DEMO_URL else "")


def docs_grid() -> str:
    """The documentation band's four entries, each a page with one line about it."""
    return "".join(
        f'<li><a class="doc-link" href="{escape(url)}">{glyph(icon)}{escape(text)}</a>'
        f"<span>{escape(blurb)}</span></li>"
        for icon, url, text, blurb in DOC_PAGES
    )


def round_down(value: float) -> int:
    """A ratio rounded down to one significant digit: 941 -> 900, 1,470 -> 1,000, 6.9 -> 6."""
    unit = 10 ** math.floor(math.log10(value))
    return int(value // unit * unit)


def family_tiles(summaries: list[dict]) -> list[dict]:
    """A tile per registry family with a supported method that has a published comparison, in
    ``SPEC_FAMILIES`` order, so a new family gets one with no change here.

    A tile gives the family's ``title``, its ``methods``, the ``comparisons`` its charts draw, its
    best ``(comparison, row)``, and its ``number``: the largest published ratio rounded down to
    one digit (:func:`round_down`), or ``None`` where Andrey is not at least twice as fast, so
    the number is always more than half the ratio and never above it.
    """
    claimed = claimed_comparisons(summaries)
    tiles = []
    for family, title in SPEC_FAMILIES.items():
        found = [m for m in claimed if m["family"] == family]
        if not found:
            continue
        cells = [(m, r) for m in found for r in m["rows"] if r["ratio"] is not None]
        method, row = max(cells, key=lambda cell: cell[1]["ratio"])
        tiles.append(
            {
                "family": family,
                "title": title,
                "methods": list(dict.fromkeys(m["method"] for m in found)),
                "comparisons": found,
                "best": (method, row),
                "number": round_down(row["ratio"]) if row["ratio"] >= 2 else None,
            }
        )
    return tiles


def plus() -> str:
    """The round plus every tile carries, the one the "Why Andrey is fast" pill has."""
    return f'<span class="wf-plus" aria-hidden="true">{_why_fast["PLUS"]}</span>'


def tile(key: str, size: str, body: str, *, kind: str = "", href: str | None = None) -> str:
    """One bento tile: a link that opens the dialog ``key`` and, without script, jumps to the same
    id, where the dialog's content is shown in the page. ``href`` is another page that holds the
    details, which the tile's script opener replaces."""
    classes = f"tile tile--{size}" + (f" tile--{kind}" if kind else "")
    return (
        f'<li class="{classes}"><a class="tile-face" href="{href or "#" + key}" '
        f'data-tile="{key}">{body}{plus()}</a></li>'
    )


def method_names(names: list[str]) -> str:
    """A family's methods for its tile: all of them up to three, else two and the count."""
    if len(names) <= 3:
        return listed(names)
    return f"{names[0]}, {names[1]}, and {len(names) - 2} more"


def family_tile(info: dict) -> str:
    """A family's tile: its name, its methods, and its best speedup in words."""
    number = (
        f'<p class="tile-number"><span>up to</span><b>{info["number"]:,}x</b>'
        "<span>faster</span></p>"
        if info["number"]
        else ""
    )
    art = FAMILY_ART.get(info["family"])
    return tile(
        f"speed-{info['family']}",
        "4",
        (tile_art(art, "4") if art else "") + f'<h3 class="tile-title">{escape(info["title"])}</h3>'
        f'<p class="tile-text">{escape(method_names(info["methods"]))}</p>{number}',
        kind="family",
    )


def figure_carousel(summaries: list[dict], graphs: str) -> str:
    """A wide tile that turns through the figures: each family's first speed chart, the accuracy
    figure, and the graph figures. Without script it is a row that scrolls sideways; the script
    (``tiles.js``) adds the controls, turns it slowly, and holds while the pointer or focus is on
    it, or the reader asks for reduced motion."""
    first: dict[str, str] = {}
    for family, figure in speed_figures(summaries):
        first.setdefault(family, figure)
    slides = [
        *first.values(),
        accuracy_figure(summaries),
        f'<div class="carousel-graphs">{graphs}</div>',
    ]
    track = "".join(
        f'<div class="carousel-slide" role="group" aria-roledescription="slide" '
        f'aria-label="{i} of {len(slides)}">{slide}</div>'
        for i, slide in enumerate(slides, 1)
    )
    return (
        '<li class="tile tile--12 tile--carousel"><section class="carousel" '
        'aria-roledescription="carousel" aria-labelledby="carousel-title">'
        '<div class="carousel-head"><h3 class="tile-title" id="carousel-title">In figures</h3>'
        '<p class="tile-text">Each family\'s speed, the accuracy on the same data, and one graph '
        "for every method.</p>"
        '<div class="carousel-controls" hidden><button class="carousel-step" type="button" '
        'data-step="-1" aria-label="Previous figure">&lsaquo;</button><span class="carousel-dots">'
        '</span><button class="carousel-step" type="button" data-step="1" aria-label="Next figure">'
        "&rsaquo;</button></div></div>"
        f'<div class="carousel-track" tabindex="0">{track}</div></section></li>'
    )


def dialog_head(key: str, title: str, intro: str = "") -> str:
    """The start of a tile's dialog, its heading and close button; ``DIALOG_TAIL`` ends it."""
    return (
        f'<dialog class="wf tile-dialog" id="{key}" aria-labelledby="{key}-title">'
        f'<div class="wf-top"><div><h2 id="{key}-title">{escape(title)}</h2>'
        f"{f'<p>{intro}</p>' if intro else ''}</div>"
        f'<button class="wf-close" type="button" aria-label="Close">{_why_fast["CLOSE"]}'
        '</button></div><div class="tile-body">'
    )


DIALOG_TAIL = "</div></dialog>"


def family_dialog(info: dict, charts: list[str]) -> str:
    """A family's dialog: its charts, one per method, and the way to the release cards."""
    body = (
        f'<div class="tile-charts">{"".join(charts)}</div>'
        if charts
        else "<p>No chart for this family compares like with like yet.</p>"
    )
    return (
        dialog_head(
            f"speed-{info['family']}", f"{info['title']} speed", family_intro(info["comparisons"])
        )
        + body
        + '<p class="tile-more"><a href="docs/benchmarks.html">Every measurement, in the release '
        "cards</a></p>" + DIALOG_TAIL
    )


def family_dialogs(summaries: list[dict]) -> str:
    """The dialogs of ``family_tiles``, each with the charts of its family."""
    charts: dict[str, list[str]] = {}
    for family, figure in speed_figures(summaries):
        charts.setdefault(family, []).append(figure)
    return "".join(family_dialog(i, charts.get(i["family"], [])) for i in family_tiles(summaries))


def accuracy_dialog(summaries: list[dict]) -> str:
    """The accuracy dialog: what an error is, then a card per method."""
    return (
        dialog_head("accuracy", "Fast, and just as accurate", escape(quality_intro(summaries)))
        + accuracy_figure(summaries)
        + DIALOG_TAIL
    )


# ---- compared with other packages ------------------------------------------------------------
# Checked against each package's released code (causal-learn 0.1.4.8, gCastle 1.0.4, lingam
# 1.13.0); the published-speed row against its README, documentation, and release notes. A family
# counts every method a package ships, experimental or not.
PEERS = ("Andrey", "causal-learn", "gCastle", "lingam")
PEER_VERSIONS = "causal-learn 0.1.4.8, gCastle 1.0.4, and lingam 1.13.0"
# Each mark's icon on the homepage, its words for screen readers, and its README symbol.
MARKS = {
    "yes": ("check", "Yes", "\u2713"),
    "part": ("minus", "Partly", "\u25d0"),
    "no": ("x", "No", "\u2717"),
}
# (group, feature, one mark per package in PEERS order, a note for every "part" mark by package).
COMPARISON = (
    (
        "Interface",
        "One result type from every method, with its graph type and edge marks",
        ("yes", "part", "part", "part"),
        {
            "causal-learn": "PC, FCI, GES, BOSS, and GRaSP give a GeneralGraph with endpoint "
            "marks, bare or inside an object, a tuple, or a dict; the LiNGAM family gives a "
            "fitted model.",
            "gCastle": "Every method gives an adjacency matrix, without a graph type or edge "
            "marks.",
            "lingam": "Every method gives weighted adjacency matrices, without a graph type; RCD, "
            "CAM-UV, and BottomUpParceLiNGAM mark hidden confounding with NaN.",
        },
    ),
    ("Interface", "Command line with JSON output", ("yes", "no", "no", "no"), {}),
    (
        "Interface",
        "A machine-readable schema for every method, and a guide for agents",
        ("yes", "no", "no", "no"),
        {},
    ),
    (
        "Coverage",
        "Constraint-, score-, and permutation-based search",
        ("yes", "yes", "part", "no"),
        {"gCastle": "PC and GES; no permutation-based search."},
    ),
    ("Coverage", "LiNGAM family", ("yes", "yes", "yes", "yes"), {}),
    (
        "Coverage",
        "Time series",
        ("yes", "yes", "part", "yes"),
        {"gCastle": "Event sequences only (TTPM)."},
    ),
    ("Coverage", "Hidden variables", ("yes", "yes", "no", "yes"), {}),
    (
        "Speed",
        "CPU-parallel and GPU backends",
        ("yes", "part", "part", "part"),
        {
            "causal-learn": "A GPU for CALM only, through its device argument; several CPU cores "
            "for RLCD and the FastKCI test only.",
            "gCastle": "A GPU for the PyTorch-based methods only, through device_type; several "
            "CPU cores for PC's parallel variant only.",
            "lingam": "A GPU for DirectLiNGAM only, through its pwling_fast measure and the "
            "separate culingam package; no search runs on several CPU cores.",
        },
    ),
    (
        "Speed",
        "Speed against other packages, published with each feature release",
        ("yes", "no", "no", "no"),
        {},
    ),
)


def comparison_notes() -> list[tuple[str, str, str, str]]:
    """Check the table and number its notes: (feature, package, mark, note) in reading order."""
    notes = []
    for _, feature, marks, row_notes in COMPARISON:
        if len(marks) != len(PEERS) or set(marks) - MARKS.keys():
            raise ValueError(f"{feature!r}: give one of {sorted(MARKS)} for each of {PEERS}")
        if set(row_notes) - set(PEERS):
            unknown = set(row_notes) - set(PEERS)
            raise ValueError(f"{feature!r}: notes name unknown packages {unknown}")
        for package, mark in zip(PEERS, marks):
            if mark == "part" and package not in row_notes:
                raise ValueError(f"{feature!r}: say what {package} covers in part")
            if package in row_notes:
                notes.append((feature, package, mark, row_notes[package]))
    return notes


def comparison_table() -> str:
    """The features table, one column per package.

    Each partial mark keeps its note as text for screen readers and as a tooltip; the page lists
    the notes in one footnote (``comparison_footnote``).
    """
    notes = {(f, p): note for f, p, _, note in comparison_notes()}
    head = "".join(f'<th scope="col">{escape(p)}</th>' for p in PEERS)
    rows, group = [], None
    for section, feature, marks, _ in COMPARISON:
        if section != group:
            rows.append(
                f'<tr class="compare-group"><th scope="colgroup" colspan="{1 + len(PEERS)}">'
                f"{section}</th></tr>"
            )
            group = section
        cells = []
        for package, mark in zip(PEERS, marks):
            icon, word, _ = MARKS[mark]
            note = notes.get((feature, package))
            said = f"{word}: {escape(note)}" if note else word
            tip = f' title="{escape(note)}"' if note else ""
            cells.append(
                f'<td class="mark mark--{mark}"{tip}>{glyph(icon)}'
                f'<span class="visually-hidden">{said}</span></td>'
            )
        # Where only Andrey has it, its mark is the row's one amber.
        alone = marks[0] == "yes" and "yes" not in marks[1:]
        row_class = ' class="compare-alone"' if alone else ""
        rows.append(f'<tr{row_class}><th scope="row">{escape(feature)}</th>{"".join(cells)}</tr>')
    return (
        '<p class="compare-hint">Scroll sideways for every package.</p>'
        '<div class="table-scroll" role="region" aria-label="Feature comparison" tabindex="0">'
        '<table class="compare">'
        f"<thead><tr><td></td>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def comparison_source() -> str:
    """What the table's marks were read from."""
    return (
        "Read from each package's released code, and the published-speed row from its README, "
        f"documentation, and release notes: {PEER_VERSIONS}."
    )


def comparison_footnote() -> str:
    """The homepage's one note on the table: its source, then every partial mark's note."""
    listed = "".join(
        f"<li>{escape(package)}, {escape(feature)}: {escape(note)}</li>"
        for feature, package, _, note in comparison_notes()
    )
    return f"{comparison_source()} Where a mark is partial:<ul>{listed}</ul>"


# The docs' code colors (pydata-sphinx-theme's defaults), so code reads the same on every page.
CODE_STYLES = {"light": "a11y-high-contrast-light", "dark": "a11y-high-contrast-dark"}


def code_html(source: str, language: str) -> str:
    """Source as highlighted HTML for a ``<pre class="hl"><code>``, by Pygments."""
    return highlight(source, get_lexer_by_name(language), HtmlFormatter(nowrap=True)).rstrip()


def code_css(scope: str, theme: str) -> str:
    """One theme's token colors under ``scope``; the page's own code-block frame stays as it is."""
    rules = HtmlFormatter(style=CODE_STYLES[theme]).get_style_defs(".hl").splitlines()
    return "\n".join(f"{scope}{rule}" for rule in rules if rule.startswith(".hl ."))


def tokens_css() -> str:
    """CSS custom properties for both palettes, and the code colors for each: the system theme, or
    the one the docs stored."""
    return "\n\n".join(
        [
            "/* Generated by site/build.py from andrey.viz.theme; do not edit. */",
            palette_block(":root", andrey.viz.LIGHT),
            code_css("", "light"),
            "@media (prefers-color-scheme: dark) {\n"
            + palette_block(':root:not([data-theme="light"])', andrey.viz.DARK)
            + "\n"
            + code_css(':root:not([data-theme="light"]) ', "dark")
            + "\n}",
            palette_block(':root[data-theme="dark"]', andrey.viz.DARK),
            code_css(':root[data-theme="dark"] ', "dark"),
        ]
    )


def homepage_code_css() -> str:
    """The code colors in both themes, keyed on the ``data-theme`` that the navbar's switch sets."""
    return code_css("", "light") + "\n" + code_css('html[data-theme="dark"] ', "dark")


# The methods the homepage's description names; every other registered method counts as "more".
DESCRIPTION_METHODS = ("PC", "GES", "FCI")


def page_description(summaries: list[dict]) -> str:
    """The homepage's summary for search results and link previews, as plain text."""
    registered = {spec.name for spec in list_specs()}
    if missing := [m for m in DESCRIPTION_METHODS if m.lower() not in registered]:
        raise ValueError(f"the description names unregistered methods: {', '.join(missing)}")
    lead, _ = speed_claim(summaries)  # the rest would take it past what search results show
    named, more = ", ".join(DESCRIPTION_METHODS), len(registered) - len(DESCRIPTION_METHODS)
    return (
        f"Causal discovery for Python. Learn causal graphs with {named} and {more} more methods, "
        f"{lead[0].lower() + lead[1:]} than other popular packages."
    )


def abel_mark() -> str:
    """Abel's mark, the leaf of its logotype, as inline SVG in the text color around it."""
    logotype = (ROOT / "site" / "brand" / "abel-logo-white.svg").read_text(encoding="utf-8")
    paths = re.findall(r'<path d="([^"]+)"', logotype)
    if len(paths) != 4:  # the three letters, then the mark
        raise ValueError(f"expected the logotype's four paths, got {len(paths)}")
    return (
        '<svg class="abel-mark" viewBox="0 0 726 460" aria-hidden="true" focusable="false">'
        f'<path d="{paths[-1]}" fill="currentColor"/></svg>'
    )


def homepage_context(summary: dict, *, placeholder: bool = False) -> dict[str, str]:
    """Everything the homepage template shows, as HTML, from the first release's summary and the
    later releases' published ones."""
    fastest(summary)
    summaries = release_summaries(summary)
    code, printed, figure = run_example()
    interface_code, interface_output, interface_figures = run_interface()
    # The page's one note sits at its foot: the comparison table's sources.
    notes = Footnotes()
    compare_note = notes.add(comparison_footnote())
    comparison = comparison_table()
    methods_title, methods_list = methods(summaries)
    banner = (
        '<p class="placeholder-banner" role="note">Placeholder numbers, copied from the '
        "documentation's benchmarks page. Not for release.</p>"
    )
    return {
        "title": escape(DESCRIPTION.rstrip(".")),
        # Plain text: docs/conf.py escapes it into the page's description and link-preview tags.
        "description": page_description(summaries),
        "robots": '<meta name="robots" content="noindex">' if placeholder else "",
        "banner": banner if placeholder else "",
        "lede": escape(lede()),
        "hero_chips": hero_chips(),
        "hero_art": hero_art(),
        "tile_art": tile_art,
        "plus": plus(),
        "family_tiles": "".join(family_tile(info) for info in family_tiles(summaries)),
        "figure_carousel": figure_carousel(summaries, interface_figures),
        "family_dialogs": family_dialogs(summaries),
        "accuracy_dialog": accuracy_dialog(summaries),
        "accuracy_line": accuracy_line(summaries),
        "why_fast": _why_fast["dialog"](),
        "dialog_head": dialog_head,
        "dialog_tail": DIALOG_TAIL,
        "method_count": f"{len(list_specs())} methods",
        "interface_code": code_html(interface_code, "python"),
        "interface_output": escape(interface_output),
        "interface_figures": interface_figures,
        "example_code": code_html(code.rstrip(), "python"),
        "example_output": escape(printed),
        "example_figure": figure,
        "agent_demo": agent_demo(agent_frames(), figure),
        "methods_title": methods_title,
        "methods": methods_list,
        "compare_note": compare_note,
        "comparison": comparison,
        "docs_grid": docs_grid(),
        "footnotes": notes.render(),
        "abel_mark": abel_mark(),
        "style": homepage_code_css(),
    }
