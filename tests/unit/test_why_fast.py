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
    return "".join(why_fast["fragment"](s["key"]) for s in SCENES)


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
