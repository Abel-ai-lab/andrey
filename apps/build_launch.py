"""Generate the README's claims, package card, methods, and quick start blocks and its images, or
check them for drift.

The images live in the repository, under ``.github/readme/``, so each commit's README shows its own.
PyPI cannot follow the relative paths; the package build rewrites them to the release tag
(``pyproject.toml``, ``[tool.hatch.metadata.hooks.fancy-pypi-readme]``). The banner does not
depend on the benchmark numbers, so only ``--banner`` redraws it.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
import runpy
import textwrap
from html import escape
from pathlib import Path

import numpy as np
from build_assets import site_builder

import andrey.viz
from andrey.spec import list_specs

ROOT = Path(__file__).resolve().parents[1]
SITE_URL = "https://andrey.abel.ai"
IMAGES = ".github/readme"  # the README's images, relative to the repository root
BANNER = "banner.svg"


CARD = "package-card-{theme}.svg"  # in IMAGES; the README links both themes
CARD_TITLE = "Accelerated, accurate, agentic"
# The package card's rows: what Andrey supports, a label and a line each.
CARD_ROWS = (
    ("Data types", "continuous, categorical, mixed, time series, nonlinear"),
    ("Real data processing", "missing values, background knowledge, edge confidence"),
    ("Graphs", "DAG, CPDAG, PAG, temporal; hidden confounders"),
    ("Acceleration", "parallel CPU; GPU with PyTorch (CUDA, MPS)"),
    ("Portability", "NetworkX, pcalg, Graphviz DOT, NumPy"),
    ("Interface", "Python, CLI, skill"),
)
CARD_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
CARD_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def supported_count() -> int:
    """The registry's supported methods, the ones every release benchmarks."""
    return sum(spec.status == "supported" for spec in list_specs())


def card_alt() -> str:
    """The package card's alt text: everything it says, in reading order."""
    rows = ". ".join(f"{label}: {line}" for label, line in CARD_ROWS)
    return f"Andrey: {CARD_TITLE.lower()}. {supported_count()} supported methods. {rows}."


def package_card(theme: str) -> str:
    """The README's package card as a standalone SVG: the wordmark, the line under it, the number
    of supported methods, and the install command on the left; ``CARD_ROWS`` on the right. Light is
    ink on paper, dark pale ink on night, each on an opaque ground with a hairline edge."""
    palette = andrey.viz.LIGHT if theme == "light" else andrey.viz.DARK
    ink, accent, line = palette.ink, palette.accent, palette.line
    width, height, left, right = 960, 480, 56, 470
    mark = (ROOT / "site" / "brand" / f"wordmark-{theme}.svg").read_text()
    view = re.search(r'viewBox="([^"]+)"', mark)[1]
    mark_w = 52 * float(view.split()[2]) / float(view.split()[3])
    paths = mark[mark.index(">") + 1 : mark.rindex("</svg>")]

    def text(x, y, content, size, fill=ink, weight=400, spacing=0, font=CARD_FONT):
        return (
            f'<text x="{x}" y="{y}" font-family="{escape(font)}" font-size="{size}" '
            f'font-weight="{weight}" letter-spacing="{spacing}" fill="{fill}">'
            f"{escape(content)}</text>"
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="card-title card-desc">',
        '<title id="card-title">Andrey</title>',
        f'<desc id="card-desc">{escape(card_alt())}</desc>',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="18" '
        f'fill="{palette.paper}" stroke="{line}"/>',
        f'<svg x="{left}" y="56" width="{mark_w:.1f}" height="52" viewBox="{view}">{paths}</svg>',
        text(left, 166, CARD_TITLE, 23, weight=600),
        text(left, 320, str(supported_count()), 120, weight=600, spacing=-4),
        text(left + 4, 354, "SUPPORTED METHODS", 13, accent, 600, 2),
        text(left + 4, 420, "pip install andrey-core", 16, font=CARD_MONO),
    ]
    for i, (label, content) in enumerate(CARD_ROWS):
        y = 74 + i * 64
        parts += [
            text(right, y, label.upper(), 12, accent, 600, 1.6),
            text(right, y + 24, content, 16),
            f'<line x1="{right}" y1="{y + 42}" x2="{width - left}" y2="{y + 42}" stroke="{line}"/>',
        ]
    return "".join([*parts, "</svg>"]) + "\n"


def readme_cards() -> dict[str, str]:
    """The README's package card in both themes, by file name."""
    return {CARD.format(theme=theme): package_card(theme) for theme in ("light", "dark")}


def write_banner():
    """Draw the README's header banner, with the docs group's renderer."""
    spec = importlib.util.spec_from_file_location("andrey_cards", ROOT / "site" / "cards.py")
    cards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cards)
    (ROOT / IMAGES).mkdir(parents=True, exist_ok=True)
    (ROOT / IMAGES / BANNER).write_text(cards.readme_banner())


def markers(name):
    """The comment pair around one generated README block."""
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


def pitch_block(summary):
    """The one-line claims under the badges, the homepage's, their numbers from every release's
    summary and the registry, and the kinds of data the methods cover."""
    builder = site_builder()
    summaries = builder.release_summaries(summary)
    accuracy, fewer = builder.more_than(builder.fewest_errors(summaries))
    start, end = markers("PITCH")
    return "\n".join(
        [
            start,
            "<!-- Generated by apps/build_launch.py; do not edit. -->",
            "",
            f"- **Fast**: {builder.CLAIM}",
            f"- **Accurate**: {accuracy.lower()} {fewer}% fewer errors than other popular"
            " packages.",
            f"- **One API**: {len(list_specs())} methods, each one function call.",
            "- **Any data**: continuous, categorical, mixed, time series, nonlinear, hidden"
            " confounders.",
            "- **Agent-ready**: a command line that answers in JSON, and a skill for coding"
            " agents.",
            "",
            end,
        ]
    )


def methods_block():
    """Every registered method, one row per family, from the registry; experimental ones are starred
    and footnoted."""
    builder = site_builder()
    specs = builder.list_specs()
    rows = []
    for family, label in builder.SPEC_FAMILIES.items():
        members = [s for s in specs if s.family == family]
        if not members:
            continue
        names = ", ".join(
            f"[`{s.name}`]({SITE_URL}/docs/api/andrey.{s.name}.html)"
            + ("\\*" if s.status == "experimental" else "")
            for s in members
        )
        returns = ", ".join(dict.fromkeys(builder.returns_text(s) for s in members))
        rows.append(f"| {label} | {names} | {returns} |")
    footnote = [
        "\\* [Experimental](" + SITE_URL + "/docs/api.html#experimental-methods):",
        "no published benchmark yet, and the API may change.",
        "",
    ]
    start, end = markers("METHODS")
    return "\n".join(
        [
            start,
            "<!-- Generated from andrey.spec by apps/build_launch.py; do not edit. -->",
            "",
            "| Family | Methods | Returns |",
            "|---|---|---|",
            *rows,
            "",
            *(footnote if any(s.status == "experimental" for s in specs) else []),
            end,
        ]
    )


QUICKSTART = """\
import andrey
from andrey.data import load_dataset
from andrey.metrics import score

sachs = load_dataset("sachs")  # {cells} cells, {proteins} proteins, and the known network
out = andrey.pc(sachs.data)    # learn a graph
print(out)

scores = score(out.structure, sachs.graph)
print({{k: round(scores[k], 2) for k in ("shd", "skeleton_precision", "skeleton_recall")}})
"""


def shd_parts(learned, truth):
    """The learned graph's differences from ``truth``, by kind, in words; they sum to its SHD."""
    est, ref = learned.to_numpy(), truth.to_numpy()
    counts = {"missing": 0, "extra": 0, "oriented": 0, "reversed": 0}
    for i, j in zip(*np.triu_indices(len(est), 1)):
        if bool(est[i, j]) != bool(ref[i, j]):
            counts["missing" if ref[i, j] else "extra"] += 1
        elif (est[i, j], est[j, i]) != (ref[i, j], ref[j, i]):
            undirected = ref[i, j] == ref[j, i]
            counts["oriented" if undirected else "reversed"] += 1
    words = {
        "missing": "are missing",
        "extra": "are extra",
        "oriented": "are oriented that the known network leaves undirected",
        "reversed": "are oriented differently",
    }
    parts = [f"{n} {words[kind]}" for kind, n in counts.items() if n]
    if not parts:
        return 0, "no differences"
    parts[0] = parts[0].replace(" ", " edges ", 1)
    return sum(counts.values()), ", and ".join(
        [", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts
    )


def quickstart_block():
    """The quick start's code, the output it prints when run, and what the numbers mean."""
    from andrey.core import GraphStructure
    from andrey.data import load_dataset
    from andrey.metrics import score as score_graph
    from andrey.metrics import to_cpdag

    cells, proteins = load_dataset("sachs").data.shape
    code = QUICKSTART.format(cells=cells, proteins=proteins)
    printed, run = io.StringIO(), {}
    with contextlib.redirect_stdout(printed):
        exec(compile(code, "README quick start", "exec"), run)
    sachs, learned, score = run["sachs"], run["out"].structure, run["scores"]
    n_learned, n_true = learned.to_edges().shape[0], sachs.graph.to_edges().shape[0]
    found = round(score["skeleton_precision"] * n_learned)
    total, parts = shd_parts(learned, to_cpdag(sachs.graph))
    if total != score["shd"]:
        raise ValueError(f"the SHD's parts sum to {total}, not {score['shd']}")
    empty = GraphStructure.from_numpy(
        np.zeros((proteins, proteins), dtype=np.int8), kind="cpdag", labels=sachs.feature_names
    )
    empty_shd = score_graph(empty, sachs.graph)["shd"]
    intro = (
        f"Learn the protein-signaling network of Sachs et al. (2005) from {cells} cells, and score"
        " it against the known network:"
    )
    among = "all" if found == n_learned else f"{found} of them"
    meaning = (
        f"PC finds {n_learned} edges, {among} among the known network's {n_true}. The"
        " structural Hamming distance (SHD) counts the edges to add, remove, or reorient to reach"
        f" the known network: {score['shd']} here ({parts}), and {empty_shd} for a graph with no"
        " edges."
    )
    start, end = markers("QUICKSTART")
    return "\n".join(
        [
            start,
            "<!-- Generated by apps/build_launch.py, which runs the code; do not edit. -->",
            "",
            *textwrap.wrap(intro, width=99, break_long_words=False, break_on_hyphens=False),
            "",
            "```python",
            code.rstrip("\n"),
            "```",
            "",
            "```text",
            printed.getvalue().rstrip("\n"),
            "```",
            "",
            *textwrap.wrap(meaning, width=99, break_long_words=False, break_on_hyphens=False),
            "",
            end,
        ]
    )


def card_block():
    """The package card in both themes, its alt text, and the release cards link."""
    alt = textwrap.wrap(card_alt(), 88)
    light, dark = (CARD.format(theme=theme) for theme in ("light", "dark"))
    start, end = markers("CARD")
    return "\n".join(
        [
            start,
            "<!-- Generated from andrey.spec by apps/build_launch.py; do not edit. -->",
            "",
            '<p align="center">',
            "  <picture>",
            '    <source media="(prefers-color-scheme: dark)"',
            f'      srcset="{IMAGES}/{dark}">',
            f'    <img alt="{alt[0]}',
            *(f"      {line}" for line in alt[1:-1]),
            *([f'      {alt[-1]}"'] if len(alt) > 1 else []),
            f'      src="{IMAGES}/{light}" width="100%">',
            "  </picture>",
            "</p>",
            "",
            f'<p align="center"><a href="{SITE_URL}/docs/benchmarks.html">Release cards</a></p>',
            "",
            end,
        ]
    )


def why_fast_block():
    """The four ideas behind the speed, one line each, for readers who cannot play the animations
    (the README, and so the package index); the guide animates each."""
    scenes = runpy.run_path(str(ROOT / "site" / "why_fast.py"))["SCENES"]
    start, end = markers("WHY_FAST")
    lines = [
        start,
        "<!-- Generated from site/why_fast.py by apps/build_launch.py; do not edit. -->",
        "",
        f"Four ideas make it [fast]({SITE_URL}/blog/20261008-why-andrey-is-fast.html):",
        "",
    ]
    for scene in scenes:
        text = f"- **{scene['title']}**: {scene['line']}"
        lines += textwrap.wrap(text, width=100, subsequent_indent="  ", break_on_hyphens=False)
    return "\n".join([*lines, "", end])


def update_readme(summary, check=False):
    """Replace only the generated marker blocks, or check them for drift."""
    path = ROOT / "README.md"
    source = path.read_text()
    updated = source
    blocks = (
        ("PITCH", pitch_block(summary)),
        ("CARD", card_block()),
        ("WHY_FAST", why_fast_block()),
        ("METHODS", methods_block()),
        ("QUICKSTART", quickstart_block()),
    )
    for name, block in blocks:
        start, end = markers(name)
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        if len(pattern.findall(updated)) != 1:
            raise ValueError(f"README needs exactly one {name} marker block.")
        updated = pattern.sub(lambda _, block=block: block, updated)
    if check and updated != source:
        raise ValueError(
            "README generated blocks differ from the summaries, andrey.spec, or a run."
        )
    if not check:
        path.write_text(updated)
    for name, card in readme_cards().items():
        target = ROOT / IMAGES / name
        if check and (not target.is_file() or target.read_text() != card):
            raise ValueError(f"{IMAGES}/{name} differs from andrey.spec.")
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(card)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", action="store_true")
    parser.add_argument("--check-readme", action="store_true")
    parser.add_argument("--banner", action="store_true", help=write_banner.__doc__)
    args = parser.parse_args()
    if not (args.readme or args.check_readme or args.banner):
        parser.error("Choose --readme, --check-readme, or --banner.")
    if args.banner:
        write_banner()
    if args.readme or args.check_readme:
        builder = site_builder()
        update_readme(builder.load_summary(builder.SUMMARY), check=args.check_readme)
