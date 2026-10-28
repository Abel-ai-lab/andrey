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

import andrey
import andrey.viz

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
IMAGES = ".github/readme"
SUMMARY = ROOT / "benchmarks/published/0.1.0/summary.json"
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
    # The claims sit above the fold, before any table or code, and are the homepage's.
    assert readme.index(end) < readme.index("```")
    summaries = build.release_summaries(summary)
    assert f"- **Fast**: {build.CLAIM}" in pitch
    lead, fewer = build.more_than(build.fewest_errors(summaries))
    claim = f"- **Accurate**: {lead.lower()} {fewer}% fewer errors than other popular packages."
    assert claim in pitch
    # The kinds of data the methods cover, which a reader and an answer engine look for first.
    for kind in ("categorical", "mixed", "time series", "nonlinear"):
        assert kind in pitch


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


def test_the_readme_package_card_counts_the_registry_in_both_themes(monkeypatch):
    # The README shows both themes' cards from .github/readme/, each self-contained, with the
    # registry's method count and every row in its alt text, and the release cards link below.
    monkeypatch.syspath_prepend(str(ROOT / "apps"))
    launch = runpy.run_path(str(ROOT / "apps" / "build_launch.py"))
    readme = README.read_text()
    count = len(andrey.spec.list_specs())
    for name, card in launch["readme_cards"]().items():
        assert f'"{IMAGES}/{name}"' in readme
        assert ET.fromstring(card).tag == "{http://www.w3.org/2000/svg}svg"
        assert not re.search(r"href=|url\(|@import", card)
        assert f">{count}</text>" in card
    alt = launch["card_alt"]()
    assert f'alt="{alt}"' in " ".join(readme.split())
    assert f"{count} methods" in alt
    assert all(f"{label}: {line}" in alt for label, line in launch["CARD_ROWS"])
    assert 'https://andrey.abel.ai/docs/benchmarks.html">Release cards</a>' in readme


def test_the_launch_post_is_a_published_page():
    # Search indexing is the server's call (X-Robots-Tag on every host but production), so the
    # post itself carries no noindex, no preview banner, and no internal release checklist.
    post = (POSTS / "20261008-introducing-andrey.md").read_text()
    assert "noindex" not in post and "placeholder-banner" not in post
    assert "preview" not in post.lower() and "tentative" not in post.lower()
    # The blog's front page lists the posts, newest first, from their front matter.
    assert "{blog-index}" in (POSTS / "index.md").read_text()


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
        # Later builds leave it as its version rendered it: the numbers are text, and every file it
        # includes is stored under blog/<version>/.
        assert "{{" not in body, post.name
        for path in re.findall(r"(?::file:|\{literalinclude\}) (\S+)", body):
            assert path.startswith(f"{meta['version']}/") and (POSTS / path).is_file(), path


def test_the_old_demo_address_redirects_to_the_examples():
    template = (ROOT / "docs" / "_templates" / "demo-redirect.html").read_text()
    assert 'content="0; url=../docs/demos/index.html"' in template
    assert '"demo/index": "demo-redirect.html"' in CONF


def test_the_benchmark_explorer_address_redirects_to_the_release_cards():
    template = (ROOT / "docs" / "_templates" / "benchmark-explorer-redirect.html").read_text()
    assert 'content="0; url=../benchmarks.html"' in template
    assert '"docs/demos/benchmark-explorer": "benchmark-explorer-redirect.html"' in CONF


def test_the_faq_cites_what_the_readme_cites():
    # Readers copy the entry from either page, so both carry the same BibTeX block.
    def bibtex(text):
        return re.search(r"```bibtex\n(.*?)```", text, re.S)[1]

    faq = (ROOT / "docs" / "docs" / "faq.md").read_text()
    assert bibtex(faq) == bibtex(README.read_text())
