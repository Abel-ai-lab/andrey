"""The "Why Andrey is fast" scenes: honest counts, no package named, and a pinned Motion."""

from __future__ import annotations

import hashlib
import re
import runpy
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
why_fast = runpy.run_path(str(ROOT / "site" / "why_fast.py"))
SCENES = why_fast["SCENES"]
# Motion's dist/motion.js as published on npm, below the vendored file's license banner.
MOTION_VERSION = "13.4.3"
MOTION_SHA256 = "dad54196f828ac5307b480e34f62d6a9b671770a702d19ecc21a045774f12f51"


class Tags(HTMLParser):
    """Every start tag with its attributes, in document order."""

    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def everything():
    return why_fast["dialog"]() + "".join(why_fast["fragment"](s["key"]) for s in SCENES)


def test_no_scene_names_another_package():
    words = everything().lower()
    for name in ("causal-learn", "causallearn", "gcastle", "lingam", "pgmpy", "tetrad"):
        assert name not in words, name
    # The comparison lane is "Others", and it is always the second lane.
    lanes = re.findall(r'<div class="wf-lane (\w+)"><p class="wf-who">(\w+)', everything())
    assert set(lanes) == {("andrey", "Andrey"), ("others", "Others")}


def test_counts_without_script_are_the_scenes_final_counts():
    expected = {"parallel": (1, 8), "reuse": (1, 6), "skip": (3, 10), "hardware": (3, 48)}
    for scene in SCENES:
        counts = why_fast["final_counts"](scene)
        assert counts == expected[scene["key"]]
        html = why_fast["fragment"](scene["key"])
        shown = [int(n) for n in re.findall(r"<b>(\d+)</b>", html)]
        assert shown == list(counts)
        # The script reads the same size from the markup, so its counts end where these start.
        attrs = next(a for t, a in Tags(html).tags if t == "figure")
        for name, value in scene["size"].items():
            text = " ".join(map(str, value)) if isinstance(value, tuple) else str(value)
            assert attrs[f"data-{name}"] == text
        # Andrey always does the work in fewer steps, which is the point of every scene.
        assert counts[0] < counts[1]


def test_dialog_has_one_tab_and_one_panel_per_scene():
    tags = Tags(why_fast["dialog"]()).tags
    tabs = [a for t, a in tags if a.get("role") == "tab"]
    panels = [a for t, a in tags if a.get("role") == "tabpanel"]
    assert [t["aria-controls"] for t in tabs] == [p["id"] for p in panels]
    assert [t["aria-selected"] for t in tabs] == ["true", "false", "false", "false"]
    assert ["hidden" in p for p in panels] == [False, True, True, True]
    # The opener is a link, so the scenes stay reachable without script.
    tag, opener = next((t, a) for t, a in tags if "data-wf-open" in a)
    dialog = next(a for t, a in tags if t == "dialog")
    assert tag == "a" and opener["href"]
    assert opener["data-wf-open"] == dialog["id"]


def test_the_post_shows_every_scene():
    guide = (ROOT / "docs" / "blog" / "why-andrey-is-fast.md").read_text()
    shown = re.findall(r"\.\./_generated/why-fast/(\w+)\.html", guide)
    assert shown == [s["key"] for s in SCENES]
    for i, scene in enumerate(SCENES, 1):
        assert f"## {i}. {scene['title']}" in guide


# The plain titles, as on the homepage, and the technical term that opens each guide section.
PLAIN = {
    "Do many things at once": "Batch and parallelize",
    "Never compute twice": "Cache and update incrementally",
    "Skip what can't matter": "Prune the search early",
    "Use the hardware you have": "Hardware-aware acceleration",
}


def test_every_page_names_the_ideas_plainly_and_the_guide_keeps_the_terms():
    assert [s["title"] for s in SCENES] == list(PLAIN)
    guide = (ROOT / "docs" / "blog" / "why-andrey-is-fast.md").read_text()
    for i, (title, term) in enumerate(PLAIN.items(), 1):
        assert f"## {i}. {title}\n\n**{term}.** " in guide
    dialog = why_fast["dialog"]()
    assert all(why_fast["escape"](title) in dialog for title in PLAIN)


def test_the_readme_gives_each_idea_in_one_line():
    # The text version for places without the animations: each scene's own line, in order.
    readme = " ".join((ROOT / "README.md").read_text().split())
    lines = [f"**{s['title']}**: {s['line']}" for s in SCENES]
    assert all(line in readme for line in lines)
    assert sorted(lines, key=readme.index) == lines
    assert all(s["line"].endswith(".") for s in SCENES)


def test_the_scenes_load_only_where_they_appear():
    conf = (ROOT / "docs" / "conf.py").read_text()
    pages = re.search(r"WHY_FAST_PAGES = \{(.*?)\}", conf).group(1)
    assert set(re.findall(r'"([^"]+)"', pages)) == {"index", "blog/why-andrey-is-fast"}
    # Motion loads before the script that uses it.
    assert '"motion.js", "why-fast.js"' in conf


def test_styles_and_script_name_no_color_value():
    # Every color is a palette token or derived from one, so a theme change reaches all of them.
    for name in ("why-fast.css", "why-fast.js"):
        text = (ROOT / "site" / "assets" / name).read_text()
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", text), name
        assert not re.search(r"\b(rgba?|hsla?|oklch|lab)\(", text), name
        assert not re.search(r":\s*(black|white)\b", text), name


def test_motion_is_the_pinned_build_with_its_license():
    data = (ROOT / "site" / "assets" / "motion.js").read_bytes()
    banner, body = data.split(b"*/\n", 1)
    assert f"Motion {MOTION_VERSION}".encode() in banner
    assert b"The MIT License (MIT)" in banner and b"Permission is hereby granted" in banner
    assert hashlib.sha256(body).hexdigest() == MOTION_SHA256


def test_the_replay_button_holds_its_space_before_it_shows():
    # The theme hides [hidden] with !important, so the kept space needs its own !important.
    css = (ROOT / "site" / "assets" / "why-fast.css").read_text(encoding="utf-8")
    rule = css[css.index(".wf-replay[hidden]") :]
    rule = rule[: rule.index("}")]
    assert "display: grid !important" in rule and "visibility: hidden" in rule
