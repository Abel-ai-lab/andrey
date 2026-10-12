"""The pull-request check flags each broken labeling rule once and passes a conforming PR."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NUMBER = 7  # the pull request under check

BASE = """\
# Changelog

## [Unreleased]

### Highlights

- An earlier highlight. ([#1])

### Breaking

### Added

- An earlier feature. ([#1])

### Fixed

## [0.1.0] - 2026-10-08

### Added

- A released feature. ([#7])

[#1]: https://github.com/Abel-ai-lab/andrey/pull/1
"""


@pytest.fixture(scope="module")
def check():
    return runpy.run_path(str(ROOT / "scripts" / "check_pr_meta.py"))


def _pull(*labels: str, milestone: str | None = "0.2.0", login: str = "someone") -> dict:
    return {
        "number": NUMBER,
        "labels": [{"name": name} for name in labels],
        "milestone": None if milestone is None else {"title": milestone},
        "user": {"login": login},
    }


def _add(section: str, bullet: str, changelog: str = BASE) -> str:
    """``changelog`` with ``bullet`` first under ``### section`` of ``## [Unreleased]``."""
    heading = f"### {section}\n\n"
    at = changelog.index(heading) + len(heading)
    return changelog[:at] + bullet + "\n" + changelog[at:]


def _count(check, pull: dict, head: str = BASE) -> int:
    return len(check["problems"](pull, BASE, head))


def test_a_chore_with_a_milestone_and_no_entry_passes(check):
    assert _count(check, _pull("chore")) == 0


def test_a_missing_milestone_fails_except_for_dependabot(check):
    assert _count(check, _pull("chore", milestone=None)) == 1
    assert _count(check, _pull("dependencies", milestone=None, login="dependabot[bot]")) == 0


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ((), 1),
        (("docs", "chore"), 1),
        (("good first issue",), 1),
        (("chore", "good first issue"), 0),
        (("dependencies", "python:uv"), 0),
    ],
)
def test_exactly_one_type_label_counts(check, labels, expected):
    assert _count(check, _pull(*labels)) == expected


@pytest.mark.parametrize(
    ("label", "section"), [("feature", "Added"), ("bug", "Fixed"), ("bug", "Highlights")]
)
def test_a_feature_or_fix_needs_its_own_unreleased_bullet(check, label, section):
    flags = ("highlight",) if section == "Highlights" else ()
    assert _count(check, _pull(label, *flags)) == 1 + len(flags)
    assert _count(check, _pull(label, *flags), _add(section, "- A change. ([#7])")) == 0


@pytest.mark.parametrize(
    "head",
    [
        _add("Added", f"- A change. ([#{NUMBER}0])"),  # a number that starts with NUMBER
        BASE + "[#7]: https://github.com/Abel-ai-lab/andrey/pull/7\n",  # a definition, no bullet
        BASE.replace("- A released feature. ([#7])", "- A released feature. ([#7]) Reworded."),
    ],
)
def test_a_link_outside_an_added_unreleased_bullet_does_not_count(check, head):
    assert _count(check, _pull("feature"), head) == 1


@pytest.mark.parametrize(
    "head",
    [
        BASE.replace("- An earlier feature. ([#1])", "- An earlier feature. ([#1], [#7])"),
        _add("Fixed", "- A change that wraps\n  onto a second line. ([#7])"),
    ],
)
def test_an_amended_or_wrapped_bullet_counts(check, head):
    assert _count(check, _pull("bug"), head) == 0


def test_a_bullet_already_on_the_base_branch_is_not_an_entry(check):
    base = _add("Added", "- A change. ([#7])")
    assert len(check["problems"](_pull("feature"), base, base)) == 1


@pytest.mark.parametrize(
    ("flag", "section"), [("highlight", "Highlights"), ("breaking", "Breaking")]
)
def test_a_flag_label_and_its_section_bullet_come_together(check, flag, section):
    in_section, in_added = _add(section, "- A change. ([#7])"), _add("Added", "- A change. ([#7])")
    assert _count(check, _pull("feature", flag), in_section) == 0
    assert _count(check, _pull("feature", flag), in_added) == 1
    assert _count(check, _pull("feature"), in_section) == 1


def test_every_broken_rule_is_reported(check):
    assert _count(check, _pull("highlight", milestone=None)) == 3
    assert _count(check, _pull("feature", "bug", "breaking", milestone=None)) == 4


def test_main_reads_the_event_and_both_changelogs(check, tmp_path, capsys):
    event, base, head = tmp_path / "event.json", tmp_path / "base.md", tmp_path / "head.md"
    base.write_text(BASE, encoding="utf-8")
    head.write_text(_add("Added", "- A change. ([#7])"), encoding="utf-8")
    argv = ["--event", str(event), "--base", str(base), "--head", str(head)]

    event.write_text(json.dumps({"pull_request": _pull("feature")}), encoding="utf-8")
    assert check["main"](argv) == 0
    event.write_text(json.dumps({"pull_request": _pull("bug", "chore")}), encoding="utf-8")
    assert check["main"](argv) == 1
    assert capsys.readouterr().out.count("::error::") == 1


def test_the_release_notes_have_a_section_for_each_label_flags_first(check):
    config = yaml.safe_load((ROOT / ".github" / "release.yml").read_text(encoding="utf-8"))
    categories = config["changelog"]["categories"]
    labels = [label for category in categories for label in category["labels"]]
    assert labels == [*check["FLAG_SECTIONS"], *check["TYPE_LABELS"], "*"]
    titles = [category["title"] for category in categories]
    assert titles[: len(check["FLAG_SECTIONS"])] == list(check["FLAG_SECTIONS"].values())
