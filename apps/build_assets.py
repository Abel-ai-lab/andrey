"""The interactive examples' pages and thumbnails, for the Space and the docs' Examples gallery.

Every page is a standalone document that takes its theme from the page that embeds it. The
benchmark explorer renders from the published summary with the homepage's chart functions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from html import escape
from pathlib import Path

import numpy as np

import andrey.viz
from andrey import GraphStructure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "live-discovery"))

from learn import (  # noqa: E402
    FRAME_SCRIPT,
    HOST_SCRIPT,
    MEEK_INPUTS,
    dsep_document,
    meek_frames,
    meek_replay,
    pc_replay,
    preset_dag,
    sprinkler_data,
)

__all__ = ["HOST_SCRIPT", "build_assets"]
SITE_URL = "https://andrey.abel.ai/docs/"


def site_builder():
    """Load the same chart functions used by the homepage."""
    spec = importlib.util.spec_from_file_location("andrey_site", ROOT / "site" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmarks_page(summary, *, links=SITE_URL, race_tab=False, heading=True):
    """The controlled benchmark results: the speed charts, every measurement, and the accuracy.

    ``links`` leads to the documentation: absolute in the Space, relative in the docs. With
    ``race_tab``, the page points to the Space's Race tab for its live timings.
    ``heading=False`` leaves out the headline, for a page that gives its own title.
    """
    builder = site_builder()
    css = (
        builder.tokens_css()
        + (ROOT / "site" / "assets" / "figures.css").read_text()
        + "body{margin:0;padding:20px;background:var(--andrey-paper);color:var(--andrey-ink);"
        "font:16px/1.6 Inter,system-ui,sans-serif}h1{font-size:1.6rem;font-weight:600}"
        "h2{font-weight:600}details{margin-block:12px}a{color:var(--andrey-accent)}"
        "table{border-collapse:collapse}td,th{padding:4px 10px;text-align:right}"
    )
    body = (
        (f"<h1>{escape(summary['headline'])}</h1>" if heading else "")
        + "<p>Controlled runs on dedicated hardware. Times are median fit times; ratios "
        "are medians of per-dataset ratios."
        + (" The Race tab identifies the server used for live timings." if race_tab else "")
        + "</p>"
        + builder.speed_chart(summary)
        + builder.speed_table(summary)
        + "<h2>Accuracy</h2>"
        + f"<p>{escape(builder.quality_intro(summary))}</p>"
        + builder.accuracy_chart(summary)
        + "<h2>Hardware, builds, and seeds</h2>"
    )
    for method in summary["methods"]:
        body += (
            f"<details><summary>{escape(method['method'])} compared with "
            f"{escape(method['comparator'])}</summary><p>{escape(method['hardware'])}; "
            f"{escape(method['build'])}; {escape(method['regime'])}</p>"
            "<table><tr><th>Variables</th><th>Seeds</th><th>Andrey SHD</th>"
            "<th>Comparison SHD</th><th>Empty SHD</th></tr>"
        )
        for row in method["rows"]:
            body += (
                "<tr>"
                + "".join(
                    f"<td>{builder.number_text(row[k])}</td>"
                    for k in ("d", "seeds", "andrey_shd", "comparator_shd", "empty_shd")
                )
                + "</tr>"
            )
        body += "</table></details>"
    body += (
        "<p>FCI uses endpoint SHD. Other methods use structural SHD. "
        "A missing ratio means a package has no time there (it passed its time cap) or no "
        "datasets could be paired.</p>"
        f'<a href="{links}benchmarks.html" target="_blank" rel="noopener">'
        "Read the full benchmarks</a>"
    )
    return (
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Andrey benchmarks</title><style>{css}</style><body>"
        + body.replace('href="docs/', f'href="{links}')
        # figures.js drives the belts, so they can be dragged and swiped here too.
        + f"<script>{FRAME_SCRIPT}</script>"
        + f"<script>{(ROOT / 'site' / 'assets' / 'figures.js').read_text()}</script></body></html>"
    )


def pc_page(heading=True):
    """PC replayed on rain, sprinkler, wet, and slippery."""
    return pc_replay(*sprinkler_data(), heading=heading)


# ---- thumbnails ------------------------------------------------------------------------------
# A gallery card shows its thumbnail as an image, where the page's color tokens do not reach, so
# each is drawn in the light palette on its own paper.
def graph_thumbnail(structure, **options):
    """A structure drawn in the light palette, on its paper."""
    return str(andrey.viz.draw(structure, palette=andrey.viz.LIGHT, **options))


def meek_thumbnail():
    """The replay's last frame, with the arrowhead its rule added ringed."""
    *_, before, last = meek_frames()
    ((head, tail),) = np.argwhere(last["marks"] != before["marks"]).tolist()
    return graph_thumbnail(
        GraphStructure.from_numpy(last["marks"], kind="cpdag"),
        positions=MEEK_INPUTS[last["replay"]][3],
        highlight=[(head, tail)],
    )


def pc_thumbnail():
    """The graph PC recovers from the replay's data, the replay's last frame."""
    X, labels = sprinkler_data()
    marks = andrey.pc(X).structure.to_numpy()
    return graph_thumbnail(GraphStructure.from_numpy(marks, kind="cpdag", labels=labels))


def dsep_thumbnail():
    """The explorer's graph: a fork, a collider, and the collider's descendant."""
    return graph_thumbnail(GraphStructure.from_networkx(preset_dag()))


def benchmarks_thumbnail(summary):
    """The explorer's first chart: fit time against the number of variables."""
    chart = site_builder().speed_chart(summary)
    svg = chart[chart.index("<svg") : chart.index("</svg>") + len("</svg>")]
    light = andrey.viz.LIGHT
    # The chart's classes resolve to the light palette; the image cannot read the page's tokens.
    style = (
        f".tc-grid{{stroke:{light.line}}}.tc-line{{fill:none;stroke-width:2}}"
        f".tc-andrey{{stroke:{light.accent}}}.tc-other{{stroke:{light.muted}}}"
        f".tc-other-1{{stroke-dasharray:4 3}}circle.tc-andrey{{fill:{light.accent}}}"
        f"circle.tc-other{{fill:{light.muted}}}circle.is-open{{fill:{light.surface}}}"
        f"circle{{stroke:{light.surface};stroke-width:2}}"
        f"text{{fill:{light.muted};font:500 9px Inter,system-ui,sans-serif}}"
    )
    return svg.replace(
        "<svg ",
        f'<svg xmlns="http://www.w3.org/2000/svg" style="background:{light.paper}" ',
        1,
    ).replace(">", f"><style>{style}</style>", 1)


# ---- the Space's files -----------------------------------------------------------------------
def build_assets(out):
    """Write the Space's teaching pages, its benchmark page, and the summary they come from."""
    builder = site_builder()
    summary = builder.load_summary(builder.SUMMARY)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(builder.SUMMARY, data / "summary.json")
    (data / "benchmarks.html").write_text(benchmarks_page(summary, race_tab=True))
    (data / "pc.html").write_text(pc_page())
    (data / "meek.html").write_text(meek_replay())
    (data / "d-separation.html").write_text(dsep_document())
    brand = out / "brand"
    brand.mkdir(exist_ok=True)
    for path in (ROOT / "site" / "brand").glob("wordmark-*.svg"):
        shutil.copy(path, brand / path.name)
    (data / "provenance.json").write_text(
        json.dumps(
            {"summary": str(builder.SUMMARY.relative_to(ROOT)), "generated": summary["generated"]},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "apps" / "live-discovery")
    build_assets(parser.parse_args().out)
