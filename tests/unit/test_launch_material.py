"""Published data and the method registry drive the README and the launch post."""

from __future__ import annotations

import datetime
import importlib.util
import json
import re
import runpy
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import andrey
import andrey.viz

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
IMAGES = ".github/readme"
SUMMARY = ROOT / "benchmarks/published/alpha-2026-09/summary.json"
POSTS = ROOT / "docs" / "blog"
CONF = (ROOT / "docs" / "conf.py").read_text()
_spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def front_matter(path):
    """A page's front matter as its top-level ``key: value`` lines."""
    head = path.read_text().split("---\n")[1]
    return dict(re.findall(r"^(\w[\w.]*): *(.*)$", head, re.M))


def test_readme_blocks_match_summary_and_registry(monkeypatch, offline_datasets):
    monkeypatch.syspath_prepend(str(ROOT / "apps"))
    launch = runpy.run_path(str(ROOT / "apps" / "build_launch.py"))
    summary = json.loads(SUMMARY.read_text())
    launch["update_readme"](summary, check=True)  # raises on drift, including the quick start's run
    readme = README.read_text()
    start, end = launch["markers"]("PITCH")
    pitch = readme[readme.index(start) : readme.index(end)]
    # The claims sit above the fold, before any table or code.
    assert readme.index(end) < readme.index("```")
    assert summary["headline"] in pitch


def readme_targets(readme):
    """Every link, image, and source address in the README."""
    return re.findall(r"\]\(([^)]+)\)", readme) + re.findall(
        r'(?:href|src|srcset)="([^"]+)"', readme
    )


def test_readme_links_are_absolute_but_its_images():
    # PyPI renders the README away from the repository, so a relative link breaks there; the
    # images are the repository's own, which the package build points at the release tag.
    targets = readme_targets(README.read_text())
    assert targets
    relative = [t for t in targets if not t.startswith("https://")]
    assert relative and all(t.startswith(f"{IMAGES}/") for t in relative)
    assert all((ROOT / t).is_file() for t in relative)


def test_the_package_description_shows_the_images_at_the_release_tag():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    hook = config["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    assert [f["path"] for f in hook["fragments"]] == ["README.md"]
    readme = README.read_text()
    for rule in hook["substitutions"]:
        replacement = rule["replacement"].replace("$HFPR_VERSION", andrey.__version__)
        readme = re.sub(rule["pattern"], replacement, readme)
    tag = f"https://raw.githubusercontent.com/Abel-ai-lab/andrey/v{andrey.__version__}/"
    targets = readme_targets(readme)
    assert all(t.startswith("https://") for t in targets)
    images = [t.removeprefix(tag) for t in targets if t.startswith(tag)]
    assert images and all((ROOT / path).is_file() for path in images)


def test_readme_quickstart_runs_as_printed(capsys, offline_datasets):
    # Copied as is, the quick start prints exactly the output the README shows under it.
    quickstart = README.read_text().split("## Quick start", 1)[1]
    code = re.search(r"```python\n(.*?)```", quickstart, re.S)[1]
    printed = re.search(r"```text\n(.*?)```", quickstart, re.S)[1]
    exec(compile(code, "README quick start", "exec"), {})
    assert capsys.readouterr().out == printed


def test_readme_chart_is_the_homepage_panel_as_one_card_per_method():
    # The README shows both themes' charts from .github/readme/, with the numbers of the
    # homepage's speedup panel and an alt text that lists them.
    readme = README.read_text()
    summary = build.load_summary(SUMMARY)
    slides = build.speedup_slides(summary)
    for theme, palette in (("light", andrey.viz.LIGHT), ("dark", andrey.viz.DARK)):
        name = build.README_CHART.format(theme=theme)
        assert f'"{IMAGES}/{name}"' in readme
        chart = build.readme_chart(summary, palette)
        assert (ROOT / IMAGES / name).read_text() == chart
        assert ET.fromstring(chart).tag == "{http://www.w3.org/2000/svg}svg"
        assert not re.search(r"href=|url\(|@import", chart)
        labels = re.findall(r'font-weight="600"[^>]*>([^<]+x)</text>', chart)
        assert labels == [build.speedup_text(v) for _, _, bars in slides for _, v in bars]
    panel = build.speedup_panel(summary)
    for _, _, bars in slides:
        for _, speed in bars:
            assert f'class="race-time">{build.speedup_text(speed)}<' in panel
    alt = " ".join(readme.split("<img alt=\"Andrey's speedup", 1)[1].split('"', 1)[0].split())
    for name, d, bars in slides:
        assert f"{name} {build.speedup_text(dict(bars)['Andrey'])} at {d:,} variables" in alt
    assert 'https://andrey.abel.ai/docs/benchmarks.html">Benchmark report</a>' in readme


def test_the_launch_post_is_a_published_page():
    # Search indexing is the server's call (X-Robots-Tag on every host but production), so the
    # post itself carries no noindex, no preview banner, and no internal release checklist.
    post = (POSTS / "introducing-andrey.md").read_text()
    assert "noindex" not in post and "placeholder-banner" not in post
    assert "preview" not in post.lower() and "tentative" not in post.lower()
    # The blog's front page lists the posts, newest first, from ABlog.
    assert "{postlist}" in (POSTS / "index.md").read_text()


def test_every_post_is_dated_once_and_names_its_version():
    # ABlog lists a post from its front-matter date; a post without one stays a draft. A post is not
    # updated later, so it names the Andrey version it was written for.
    posts = [p for p in POSTS.glob("*.md") if front_matter(p).get("blogpost") == "true"]
    assert posts
    authors = re.search(r"blog_authors = \{(.*?)\}", CONF)[1]
    for post in posts:
        meta = front_matter(post)
        datetime.date.fromisoformat(meta["date"])
        assert re.fullmatch(r"\d+\.\d+\.\d+", meta.get("version", "")), post.name
        for author in meta["author"].split(", "):
            assert f'"{author}"' in authors, post.name
        # The page states no date of its own beside the front matter's.
        body = post.read_text().split("---\n", 2)[2]
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b|<time", body), post.name


@pytest.mark.skipif(not SUMMARY.exists(), reason="the published summary is absent")
def test_the_launch_post_states_the_summary():
    summary = build.load_summary(SUMMARY)
    text, figures = build.launch_post_context(summary)
    method, top = build.fastest(summary)
    as_good, total = build.accuracy_tally(summary)
    title, _ = build.methods(summary)
    assert text["speed_claim"] == summary["headline"]
    assert text["speed_short"] == "over 100x on PC"
    assert text["as_good"] == f"{as_good} of {total}"
    assert text["method_count"] == title.split(" in ")[0]
    assert text["slow_fit"] == (
        f"{method['method']} on {top['d']:,} variables took "
        f"{build.seconds_text(top['comparator_s'])} in {method['comparator']}"
    )
    # The one figure is the homepage's speedup panel; the post sits one level down, so its link
    # to the benchmark report gains a ../.
    panel = build.speedup_panel(summary).replace('href="docs/', 'href="../docs/')
    assert panel in figures["speedup"]
    assert 'href="../docs/benchmarks.html"' in figures["speedup"]
    assert 'href="docs/' not in figures["speedup"]
    # The post uses every number and figure the build provides, and nothing typed: its numbers
    # come from the build or from docs/conf.py's release and method tiers.
    post = (POSTS / "introducing-andrey.md").read_text()
    release = {"release", "n_supported", "n_experimental", "supported_methods"}
    assert all(f'"{key}":' in CONF for key in release)
    used = set(re.findall(r"\{\{ (\w+) \}\}", post))
    assert set(text) <= used <= set(text) | release
    included = set(re.findall(r":file: \.\./_generated/blog/(\w+)\.html", post))
    assert included == set(figures)
    body = post.split("---\n", 2)[2]
    assert not re.search(r"\b\d+(\.\d+)?x\b", body), "a typed speedup in the post"


def test_the_old_demo_address_redirects_to_the_examples():
    template = (ROOT / "docs" / "_templates" / "demo-redirect.html").read_text()
    assert 'content="0; url=../docs/demos/index.html"' in template
    assert '"demo/index": "demo-redirect.html"' in CONF


def test_the_faq_cites_what_the_readme_cites():
    # Readers copy the entry from either page, so both carry the same BibTeX block.
    def bibtex(text):
        return re.search(r"```bibtex\n(.*?)```", text, re.S)[1]

    faq = (ROOT / "docs" / "docs" / "faq.md").read_text()
    assert bibtex(faq) == bibtex(README.read_text())


def test_the_launch_post_states_the_readme_claims(monkeypatch, offline_datasets):
    # The post's four claims read as the README's, with the same numbers, and it says Andrey is in
    # alpha.
    monkeypatch.syspath_prepend(str(ROOT / "apps"))
    summary = json.loads(SUMMARY.read_text())
    launch = runpy.run_path(str(ROOT / "apps" / "build_launch.py"))
    text, _ = launch["site_builder"]().launch_post_context(summary)
    post = (ROOT / "docs" / "blog" / "introducing-andrey.md").read_text()
    post = re.sub(r"\{\{ (\w+) \}\}", lambda m: str(text.get(m.group(1), m.group(0))), post)
    post = re.sub(r"\n  (?=\S)", " ", post)  # join wrapped list items
    pitch = launch["pitch_block"](summary).splitlines()
    for claim in [line for line in pitch if line.startswith("- **")]:
        assert claim in post, claim
    assert "is in alpha" in post
