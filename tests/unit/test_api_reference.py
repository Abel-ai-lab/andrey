"""The API reference (``docs/docs/api.md``) can be built on any file system.

autosummary writes each name it lists to a page named after the full dotted name, so two names
that differ only by case would write one file on a case-insensitive file system, such as macOS's
default: one page would replace the other, and the build would warn that a page is missing.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

API_MD = pathlib.Path(__file__).resolve().parents[2] / "docs" / "docs" / "api.md"


def documented_names() -> list[str]:
    """Each name an autosummary table in the API reference lists, under its current module."""
    names, module, listing = [], "", False
    for line in API_MD.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if line.startswith(".. currentmodule::"):
            module, listing = line.split("::", 1)[1].strip(), False
        elif line.startswith(".. autosummary::"):
            listing = True
        elif listing and line.startswith(" ") and entry and not entry.startswith(":"):
            names.append(f"{module}.{entry}")
        elif listing and entry and not line.startswith(" "):
            listing = False
    return names


def test_no_two_documented_names_differ_only_by_case():
    names = documented_names()
    assert "andrey.pc" in names, "the API reference lists no autosummary names"
    by_page = defaultdict(set)
    for name in names:
        by_page[name.casefold()].add(name)
    clashes = sorted(sorted(group) for group in by_page.values() if len(group) > 1)
    assert not clashes, f"names whose reference pages are one file on macOS: {clashes}"
