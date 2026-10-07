"""The Examples and Demos galleries: a card for every page, and every card leads somewhere real."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "docs"
EXAMPLES = (DOCS / "examples" / "index.md").read_text()
DEMOS = (DOCS / "demos" / "index.md").read_text()
CONF = (ROOT / "docs" / "conf.py").read_text()
NOTEBOOKS = {p.stem for p in (ROOT / "examples").glob("*.ipynb")}
DEMO_PAGES = {p.stem for p in (DOCS / "demos").glob("*.md")} - {"index"}


def cards(gallery):
    """Each card's title and options, from a gallery's grid-item-card blocks."""
    found = []
    for title, body in re.findall(r":::\{grid-item-card\} ([^\n]+)\n(.*?)\n:::\n", gallery, re.S):
        options = dict(re.findall(r"^:([\w-]+): (.+)$", body, re.M))
        found.append((title, options))
    return found


def toctree_entries(gallery):
    return {
        line.strip()
        for block in re.findall(r"```\{toctree\}\n(.*?)```", gallery, re.S)
        for line in block.splitlines()
        if line.strip() and not line.startswith(":")
    }


def test_each_gallery_links_its_own_pages_with_a_card_and_a_sidebar_entry():
    for gallery, pages in ((EXAMPLES, NOTEBOOKS), (DEMOS, DEMO_PAGES)):
        assert sorted(options["link"] for _, options in cards(gallery)) == sorted(pages)
        assert toctree_entries(gallery) == pages


def test_every_demo_has_a_generator_in_conf():
    # conf.py writes each demo's page and thumbnail at build time; the strict build and
    # docs/test_built_site.py check that the cards' links and images resolve.
    for stem in DEMO_PAGES:
        assert f'"docs/demos/{stem}": (' in CONF, stem


def test_the_live_demo_card_waits_for_a_public_space():
    # The Demos page includes the card conf.py writes, which is empty until DEMO_URL is set.
    assert "{include} ../../_generated/examples/live-demo.md" in DEMOS
    assert "if site.DEMO_URL:" in CONF
