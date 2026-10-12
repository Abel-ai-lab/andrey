"""Pull-request check: one type label, a milestone, and the CHANGELOG bullets the labels call for.

A pull request carries exactly one type label (``TYPE_LABELS``) and, unless Dependabot opened it,
a milestone. A ``feature`` or ``bug`` pull request adds a bullet under ``## [Unreleased]`` in
``CHANGELOG.md`` that links it as ``[#<number>]``. The ``highlight`` label and a bullet linking the
pull request under ``### Highlights`` come together, and so do ``breaking`` and ``### Breaking``.
Other labels do not count.

The pull request comes from a ``pull_request`` event payload; the two CHANGELOG files are the base
branch's and the one with the pull request merged.

    python3 scripts/check_pr_meta.py --event "$GITHUB_EVENT_PATH" --base base.md --head head.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# In release-notes order; .github/release.yml has one section for each.
TYPE_LABELS = ("feature", "bug", "docs", "chore", "dependencies")
ENTRY_LABELS = ("feature", "bug")
FLAG_SECTIONS = {"highlight": "Highlights", "breaking": "Breaking"}
MILESTONE_EXEMPT = ("dependabot[bot]",)


def unreleased_bullets(changelog: str) -> dict[str, list[str]]:
    """Bullets under ``## [Unreleased]`` by ``###`` subsection, each joined across its lines.

    Bullets before the first subsection are under ``""``.
    """
    sections: dict[str, list[str]] = {}
    bullets: list[str] | None = None
    for line in changelog.splitlines():
        if line.startswith("## "):
            bullets = sections.setdefault("", []) if line.startswith("## [Unreleased]") else None
        elif bullets is None:
            continue
        elif line.startswith("### "):
            bullets = sections.setdefault(line[4:].strip(), [])
        elif line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith(" ") and bullets:
            bullets[-1] += " " + line.strip()
    return sections


def problems(pull: dict, base_changelog: str, head_changelog: str) -> list[str]:
    """One message per rule the pull request breaks; empty when it passes.

    ``pull`` is the event's ``pull_request`` object: ``number``, ``labels``, ``milestone``, and
    ``user``.
    """
    number = pull["number"]
    link = f"[#{number}]"
    labels = {label["name"] for label in pull["labels"]}
    found = []

    if pull["milestone"] is None and pull["user"]["login"] not in MILESTONE_EXEMPT:
        found.append("no milestone: set the release this pull request ships in")

    types = [name for name in TYPE_LABELS if name in labels]
    if len(types) != 1:
        found.append(
            f"type labels: {', '.join(types) or 'none'}; set exactly one of "
            f"{', '.join(TYPE_LABELS)}"
        )

    head = unreleased_bullets(head_changelog)
    before = {b for bullets in unreleased_bullets(base_changelog).values() for b in bullets}
    added = [b for bullets in head.values() for b in bullets if link in b and b not in before]
    if not added and (needing := [name for name in ENTRY_LABELS if name in labels]):
        found.append(
            f"{' and '.join(needing)} without a CHANGELOG entry: add a bullet under "
            f"## [Unreleased] that ends with ({link})"
        )

    for flag, section in FLAG_SECTIONS.items():
        listed = any(link in bullet for bullet in head.get(section, []))
        if flag in labels and not listed:
            found.append(f"{flag} label without a bullet ending in ({link}) under ### {section}")
        elif listed and flag not in labels:
            found.append(f"a bullet under ### {section} links {link}: add the {flag} label")
    return found


def main(argv: list[str] | None = None) -> int:
    """Check the pull request in ``--event``; return 1 when it breaks a rule."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", type=Path, required=True, help="pull_request event JSON")
    parser.add_argument("--base", type=Path, required=True, help="the base branch's CHANGELOG.md")
    parser.add_argument("--head", type=Path, required=True, help="CHANGELOG.md, merged")
    args = parser.parse_args(argv)

    pull = json.loads(args.event.read_text(encoding="utf-8"))["pull_request"]
    found = problems(
        pull, args.base.read_text(encoding="utf-8"), args.head.read_text(encoding="utf-8")
    )
    for problem in found:
        print(f"::error::{problem}")
    if not found:
        print(f"#{pull['number']}: one type label, a milestone, and the CHANGELOG bullets it needs")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
