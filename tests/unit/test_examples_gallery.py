"""The Examples gallery: a card for every page, grouped as the sidebar groups them."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs" / "docs" / "examples"
GALLERY = (EXAMPLES / "index.md").read_text()
NOTEBOOKS = {p.stem for p in (ROOT / "examples").glob("*.ipynb")}
DEMOS = {p.stem for p in EXAMPLES.glob("*.md")} - {"index"}


def sections(gallery):
    """Each section of the gallery by its heading: its cards' options and its toctree."""
    found = {}
    for title, body in re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", gallery, re.M | re.S):
        cards = [
            dict(re.findall(r"^:([\w-]+): (.+)$", card, re.M))
            for card in re.findall(r":::\{grid-item-card\} [^\n]+\n(.*?)\n:::\n", body, re.S)
        ]
        (toctree,) = re.findall(r"```\{toctree\}\n(.*?)```", body, re.S)
        options = dict(re.findall(r"^:([\w-]+):(?: (.+))?$", toctree, re.M))
        entries = [line for line in toctree.splitlines() if line and not line.startswith(":")]
        found[title] = (cards, options, entries)
    return found


def test_every_notebook_and_demo_has_one_card():
    links = [card["link"] for cards, _, _ in sections(GALLERY).values() for card in cards]
    assert sorted(links) == sorted(NOTEBOOKS | DEMOS)


def test_each_section_lists_its_cards_pages_in_the_sidebar_under_its_heading():
    for title, (cards, options, entries) in sections(GALLERY).items():
        assert entries == [card["link"] for card in cards], title
        assert options["caption"] == title
