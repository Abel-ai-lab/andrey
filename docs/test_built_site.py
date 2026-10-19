"""Checks on the built website that the unit tests cannot make.

Run from the repo root: ``uv run --group docs python -m pytest docs/test_built_site.py``. The first
test that needs the site builds all of it from a clean workspace with ``-W``, which takes a few
minutes; the checks then read the built pages. The template and source
checks are unit tests, which the docs workflow runs first.
"""

from __future__ import annotations

import json
import pathlib
import posixpath
import re
import shutil
import subprocess
import sys
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlsplit

import pytest

import andrey
import andrey.data.discrete
import andrey.data.mixed
import andrey.data.real
import andrey.metrics
import andrey.viz
import andrey.viz.mpl  # importable without matplotlib (it is imported lazily, inside the functions)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"  # the site's Sphinx source; the documentation pages are under docs/docs/
BUILD = DOCS / "_build"
# A fenced code block (as in scripts/check_dist.py): its addresses are examples, not links.
_FENCE = re.compile(r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL)

# Math symbols the docs write in KaTeX or ASCII, never as raw unicode. Typographic unicode
# (middle dot, em dash, prose arrows) is punctuation and allowed.
MATH_UNICODE = set(
    "≤≥≠≡≅≃∼≈≪≫∝∈∉∑∏√∀∃∞∪∩⊆⊇⊂⊃⊕⊗⊥⊤⫫∇∂∫∧∨¬∘∙⋅±×−⇒⇐⇄⇔′″²³"  # relations and operators
    "₀₁₂₃₄₅₆₇₈₉"  # subscripts
    "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"  # Greek
    "µℝℕℤℚℂ𝔼"  # the micro sign (looks like μ) and blackboard-bold sets
)
# The public surface the reference must cover, module by module, each with the prefix its
# autosummary pages carry (`andrey.viz.draw.html` -> `viz.draw`).
PUBLIC_MODULES = (
    (andrey, ""),
    (andrey.data.real, "data."),
    (andrey.data.discrete, "data."),
    (andrey.data.mixed, "data."),
    (andrey.metrics, "metrics."),
    (andrey.viz, "viz."),
    (andrey.viz.mpl, "viz.mpl."),
)
# Documented in prose instead of an autosummary page: a str, and module-level instances.
COVERAGE_SKIP = {"__version__", "config", "viz.LIGHT", "viz.DARK"}
# The longest description a page may carry: search results show about this many characters, and
# docs/conf.py cuts a description it takes from a page's first paragraph to this length.
DESCRIPTION_LENGTH = 160
# The header's links, by the text they show, and the page each must reach from any page.
HEADER = {
    "Docs": "docs/index.html",
    "Examples": "docs/examples/index.html",
    "Demos": "docs/demos/index.html",
    "Blog": "blog/index.html",
}


@pytest.fixture(scope="module")
def html() -> pathlib.Path:
    """The site, built once from a clean workspace so stale output cannot hide a regression."""
    generated = (
        BUILD,
        DOCS / "_generated",
        DOCS / "docs" / "code" / "generated",
        *(DOCS / "docs" / "examples").glob("*.ipynb"),
        DOCS / "_static" / "graph-comparison.html",
        *(DOCS / "_static").glob("graph-types-*.html"),
    )
    for p in generated:
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    out = BUILD / "html"
    # -E: never reuse a saved environment (a cached page's warnings must fire again under -W).
    cmd = [sys.executable, "-m", "sphinx", "-E", "-b", "html", str(DOCS), str(out), "-W", "-q"]
    if subprocess.run(cmd, cwd=ROOT).returncode:
        pytest.fail("sphinx-build -W failed (see the output above)")
    return out


class _Links(HTMLParser):
    """Collect every link with its classes, in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[tuple[str, str]] = []  # (classes, href)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href") is not None:
            self.found.append((values.get("class") or "", values["href"]))


def _resolve(html: pathlib.Path, page: pathlib.Path, href: str) -> pathlib.Path:
    """The built file a relative link from ``page`` leads to; a directory means its index.html."""
    path = href.split("#", 1)[0].split("?", 1)[0]
    base = page.parent.relative_to(html).as_posix()
    target = html / posixpath.normpath(posixpath.join(base, path or page.name))
    return target / "index.html" if target.is_dir() else target


def _pages(html: pathlib.Path):
    """Every built page with the site's header, as (page, its text)."""
    for page in sorted(html.rglob("*.html")):
        if "_static" in page.parts:
            continue
        text = page.read_text(encoding="utf-8")
        if 'id="pst-header"' in text:
            yield page, text


def test_the_docs_write_math_in_katex_not_unicode():
    hits = []
    for f in [*DOCS.rglob("*.md"), *(ROOT / "examples").glob("*.ipynb"), ROOT / "CONTRIBUTING.md"]:
        if {"_build", "_generated", "PR"} & set(f.parts):  # PR/: local notes, outside the site
            continue
        content = f.read_text(encoding="utf-8")
        if f.suffix == ".ipynb":
            content = "\n".join("".join(cell["source"]) for cell in json.loads(content)["cells"])
        for lineno, line in enumerate(content.splitlines(), 1):
            if MATH_UNICODE.intersection(line):
                hits.append(f"{f.relative_to(ROOT)}:{lineno}")
    assert not hits, f"unicode math symbols (use KaTeX or ASCII): {hits[:8]}"


def test_the_reference_documents_every_public_name_and_nothing_else(html):
    generated = html / "docs" / "code" / "generated"
    public = {prefix + name for module, prefix in PUBLIC_MODULES for name in module.__all__}
    documented = {p.stem.removeprefix("andrey.") for p in generated.glob("andrey.*.html")}
    missing = sorted(public - documented - COVERAGE_SKIP)
    assert not missing, f"public names without a reference page: {missing}"
    extra = sorted(documented - public)
    assert not extra, f"reference pages for names that are not public: {extra}"


def test_the_homepage_links_its_stylesheets_with_a_content_version(html):
    home = (html / "index.html").read_text(encoding="utf-8")
    for sheet in ("figures.css", "site.css"):
        assert re.search(rf'href="_static/{sheet}\?v=\w+"', home), sheet


def test_every_page_has_one_short_description_equal_to_its_preview(html):
    problems = []
    for page, text in _pages(html):
        name = page.relative_to(html)
        described = re.findall(r'<meta name="description" content="([^"]*)"', text)
        previewed = re.findall(r'<meta property="og:description" content="([^"]*)"', text)
        project = re.findall(r'<meta property="og:site_name" content="([^"]*)"', text)
        if text.count('name="description"') != 1 or len(described) != 1 or described != previewed:
            problems.append(f"{name}: no one description equal to its preview's")
        elif len(unescape(described[0])) > DESCRIPTION_LENGTH:
            problems.append(f"{name}: its description is over {DESCRIPTION_LENGTH} characters")
        elif not unescape(described[0]).strip() or described == project:
            problems.append(f"{name}: its description is empty or only the project's name")
    assert not problems, "\n".join(problems)


def test_every_homepage_tile_leads_to_details_that_are_built(html):
    home = (html / "index.html").read_text(encoding="utf-8")
    tiles = re.findall(r'<li class="tile[^"]*"><a class="tile-face" href="([^"]+)"', home)
    assert tiles and len(tiles) == len(set(tiles))  # one tile for each dialog
    problems = []
    for href in tiles:
        if href.startswith("#"):
            if f'<dialog class="wf tile-dialog" id="{href[1:]}"' not in home:
                problems.append(f"{href}: no dialog holds its details")
        elif not _resolve(html, html / "index.html", href).is_file():
            problems.append(f"{href}: not built")
    for script in ("tiles.js", "prompt.js"):
        if f"{script}" not in home or not (html / "_static" / script).is_file():
            problems.append(f"{script}: not loaded or not built")
    assert not problems, "\n".join(problems)


# The section's line icon before a header link's name.
ICON = r'<svg class="nav-icon[^"]*"[^>]*>.*?</svg>'


def test_every_header_leads_home_and_to_each_section(html):
    cards = (DOCS / "docs" / "benchmarks").glob("*.md")
    newest = max((tuple(map(int, card.stem.split("."))) for card in cards), default=())
    newest = ".".join(map(str, newest))
    problems = []
    for page, text in _pages(html):
        name = page.relative_to(html)
        header = text[text.index('id="pst-header"') : text.index("</header>")]
        links = _Links()
        links.feed(header)
        brand = [href for cls, href in links.found if "navbar-brand" in cls]
        if not brand or _resolve(html, page, brand[0]) != html / "index.html":
            problems.append(f"{name}: the wordmark does not lead home")
        for label, expected in HEADER.items():
            hrefs = re.findall(
                rf'<a class="nav-link nav-internal" href="([^"]+)"[^>]*>{ICON}{label}</a>', header
            )
            if len(hrefs) != 1 or _resolve(html, page, hrefs[0]) != html / expected:
                problems.append(f"{name}: the header's {label} link is wrong")
        # Release cards carries the dot that marks the newest release until the reader opens it.
        cards = re.findall(
            rf'<a class="nav-link nav-internal" href="([^"]+)"[^>]*>{ICON}Release cards'
            r'<span class="nav-new" data-release="([^"]+)" data-date="\d{4}-\d{2}-\d{2}" hidden>',
            header,
        )
        if len(cards) != 1 or _resolve(html, page, cards[0][0]) != html / "docs/benchmarks.html":
            problems.append(f"{name}: the header's Release cards link is wrong")
        elif cards[0][1] != newest:
            problems.append(f"{name}: the Release cards dot names {cards[0][1]}, not {newest}")
    assert not problems, "\n".join(problems)


class _Marks(HTMLParser):
    """Collect the links marked as the page being read, and whether their folds are open."""

    def __init__(self) -> None:
        super().__init__()
        self.folds: list[bool] = []
        self.marked: list[tuple[str, bool]] = []  # (href, every fold around it open)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "details":
            self.folds.append("open" in values)
        elif tag == "a" and values.get("aria-current") == "page":
            self.marked.append((values["href"], all(self.folds)))

    def handle_endtag(self, tag: str) -> None:
        if tag == "details":
            self.folds.pop()


def test_every_docs_page_finds_itself_in_its_sidebar_and_edits_a_tracked_file(html):
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    problems = []
    for page, text in _pages(html):
        name = page.relative_to(html).as_posix()
        start = text.find('<nav class="bd-docs-nav docs-nav"')
        if start < 0:
            continue
        # The tree marks the page in view, unless the page is outside it, and opens its fold.
        marks = _Marks()
        marks.feed(text[start : text.index("</nav>", start)])
        if len(marks.marked) > 1:
            problems.append(f"{name}: the sidebar marks {len(marks.marked)} pages")
        for href, shown in marks.marked:
            if _resolve(html, page, href) != page:
                problems.append(f"{name}: the sidebar marks {href}, another page")
            elif not shown:
                problems.append(f"{name}: the sidebar marks the page inside a closed fold")
        # Edit this page opens a file in the repository, never one the build writes.
        for path in re.findall(r'href="https://github\.com/[^"]+/edit/main/([^"]+)"', text):
            if path not in tracked:
                problems.append(f"{name}: Edit this page opens {path}, which git does not track")
    assert not problems, "\n".join(problems)


def test_every_page_has_a_built_favicon_and_link_preview_card(html):
    problems = []
    for page, text in _pages(html):
        name = page.relative_to(html)
        icon = re.search(r'<link rel="icon" href="([^"]+)"', text)
        if not icon or not _resolve(html, page, icon[1]).is_file():
            problems.append(f"{name}: its favicon was not built")
        card = re.search(r'<meta property="og:image" content="https://[^/]+/([^"]+)"', text)
        if not card or not (html / card[1]).is_file():
            problems.append(f"{name}: its link-preview card was not built")
    assert not problems, "\n".join(problems)


def _front_matter(path: pathlib.Path) -> dict[str, str]:
    """A page's top-level front matter keys and their values, as written."""
    head = path.read_text(encoding="utf-8").split("---\n", 2)[1]
    return dict(re.findall(r"^(\w+): (.*)$", head, re.MULTILINE))


def test_the_blog_features_the_newest_post_then_lists_the_others(html):
    # The newest dated post, by date and then version, leads with its cover and its description;
    # every other dated post follows once, in a row with its cover, newest first.
    posts = {}
    for path in (DOCS / "blog").glob("*.md"):
        meta = _front_matter(path)
        if meta.get("blogpost") == "true" and meta.get("date"):
            version = tuple(int(part) for part in meta.get("version", "").split(".") if part)
            posts[path.stem] = (meta["date"], version, meta["description"].strip('"'))
    index = html / "blog" / "index.html"
    parts = re.split(r'<div class="(release-latest|release-row)\b', index.read_text("utf-8"))[1:]
    # Each block runs to the next one or to the end of its section.
    blocks = [re.split(r"</?section\b", body, maxsplit=1)[0] for body in parts[1::2]]
    kinds = parts[::2]
    assert kinds == ["release-latest"] + ["release-row"] * (len(posts) - 1), kinds
    shown = []
    for kind, block in zip(kinds, blocks, strict=True):
        linked = set(re.findall(r'href="([^"/#]+)\.html"', block))
        assert len(linked) == 1 and linked <= posts.keys(), f"a {kind} block links {linked}"
        shown.append(linked.pop())
        cover = re.search(r'<img [^>]*src="([^"]+)"', block)
        assert cover and _resolve(html, index, cover[1]).is_file(), f"{shown[-1]} has no cover"
        assert posts[shown[-1]][2] in unescape(block), f"{shown[-1]} lacks its description"
    assert sorted(shown) == sorted(posts)
    assert shown[0] == max(posts, key=lambda post: posts[post][:2])
    dates = [posts[post][:2] for post in shown[1:]]
    assert dates == sorted(dates, reverse=True)
    # The newest post's cover is the one under its title.
    post = html / "blog" / f"{shown[0]}.html"
    own = re.search(r'<figure class="post-cover"><img [^>]*src="([^"]+)"', post.read_text("utf-8"))
    cover = re.search(r'<img [^>]*src="([^"]+)"', blocks[0])
    assert _resolve(html, index, cover[1]).read_bytes() == _resolve(html, post, own[1]).read_bytes()


def test_every_post_shows_its_cover_and_byline_under_its_title(html):
    problems = []
    for post in (p.stem for p in (DOCS / "blog").glob("*.md") if "blogpost: true" in p.read_text()):
        built = html / "blog" / f"{post}.html"
        text = built.read_text(encoding="utf-8")
        cover = re.search(r'</h1>\s*<figure class="post-cover">(<img [^>]*>)', text)
        source = cover and re.search(r'\bsrc="([^"]+)"', cover[1])
        if not source or not _resolve(html, built, source[1]).is_file():
            problems.append(f"blog/{post}.html has no built cover under its title")
        elif not re.search(r'\balt="[^"]*\S[^"]*"', cover[1]):
            problems.append(f"blog/{post}.html's cover has no alt text")
        byline = re.search(
            r'</h1>\s*(?:<figure class="post-cover">.*?</figure>\s*)?'
            r'<div class="post-byline">(.*?)</div>',
            text,
            re.S,
        )
        author = byline and re.search(r'<a href="([^"]+)">', byline[1])
        tags = _front_matter(DOCS / "blog" / f"{post}.md").get("tags", "")
        if not author or "<time datetime=" not in byline[1]:
            problems.append(f"blog/{post}.html has no author and date under its title")
        elif not _resolve(html, built, author[1]).is_file():
            problems.append(f"blog/{post}.html links its author to {author[1]}, not built")
        elif tags and 'class="blog-tags"' not in byline[1]:
            problems.append(f"blog/{post}.html shows no tags under its title")
    assert not problems, "\n".join(problems)


def test_every_site_address_in_the_readme_and_package_metadata_is_built(html):
    # PyPI shows the README and lists the project URLs; addresses inside code are examples.
    problems = []
    for source in ("README.md", "pyproject.toml", "CITATION.cff"):
        text = _FENCE.sub("", (ROOT / source).read_text(encoding="utf-8"))
        for path in sorted(set(re.findall(r"https://andrey\.abel\.ai/?([^\s\"'<>)#?]*)", text))):
            path = path.rstrip(".,;:")
            target = html / path
            if not path or path.endswith("/"):
                target = target / "index.html"
            if not target.is_file():
                problems.append(f"{source} uses https://andrey.abel.ai/{path}, which is not built")
    assert not problems, "\n".join(problems)


def _discovery_graph(text):
    scripts = re.findall(r'<script type="application/ld\+json">(.*?)</script>', text, re.S)
    assert len(scripts) == 1, "Each page must have one valid JSON-LD graph"
    data = json.loads(scripts[0])
    assert data["@context"] == "https://schema.org"
    return {node["@type"]: node for node in data["@graph"]}


def test_every_page_carries_structured_data_that_matches_its_preview(html):
    for page, text in _pages(html):
        nodes = _discovery_graph(text)
        node = nodes.get("BlogPosting", nodes.get("WebPage"))
        assert node is not None, page
        for suffix, field in (
            ("title", "name"),
            ("description", "description"),
            ("image", "image"),
        ):
            preview = re.findall(rf'<meta property="og:{suffix}" content="([^"]*)"', text)
            twitter = re.findall(rf'<meta name="twitter:{suffix}" content="([^"]*)"', text)
            assert len(preview) == 1 and preview == twitter, page
            assert node[field] == unescape(preview[0]), page
        software = nodes["SoftwareSourceCode"]
        assert software["name"] == "Andrey" and software["programmingLanguage"] == "Python"
        assert software["license"]["name"] == "Apache-2.0"
        assert software["license"]["url"].endswith("/blob/main/LICENSE")
        assert software["codeRepository"] == "https://github.com/Abel-ai-lab/andrey"
        assert software["image"] == "https://andrey.abel.ai/_static/favicon.svg"
        assert (html / "_static" / "favicon.svg").is_file()
        assert node["about"]["@id"] == software["@id"]
        assert nodes["Organization"]["url"] == "https://abel.ai/"
        assert nodes["Organization"]["logo"] == "https://andrey.abel.ai/_static/abel-symbol.png"


def test_the_organization_logo_is_a_png_search_results_can_show(html):
    from PIL import Image

    # Search engines show an organization's logo from 112 x 112 pixels up, on a light ground.
    with Image.open(html / "_static" / "abel-symbol.png") as logo:
        assert logo.format == "PNG" and min(logo.size) >= 112, logo.size


def test_a_post_carries_the_date_authors_and_release_of_its_front_matter(html):
    for source in (DOCS / "blog").glob("*.md"):
        content = source.read_text()
        if "blogpost: true" not in content:
            continue
        nodes = _discovery_graph((html / "blog" / (source.stem + ".html")).read_text())
        article = nodes["BlogPosting"]
        front = {key: re.search(rf"^{key}: (.+)$", content, re.M)[1] for key in ("date", "version")}
        assert article["datePublished"] == front["date"], source.name
        assert nodes["SoftwareSourceCode"]["version"] == front["version"], source.name
        assert "dateModified" not in article
        authors = re.search(r"^author: (.+)$", content, re.M)[1]
        names = [name.strip() for name in authors.strip("[]").split(",")]
        assert [person["name"] for person in article["author"]] == names, source.name
        for person in article["author"]:
            assert (html / urlsplit(person["url"]).path.lstrip("/")).is_file(), person["url"]


def test_llms_txt_links_public_pages_by_section_with_their_preview_text(html):
    text = (html / "llms.txt").read_text()
    assert text.startswith("# Andrey\n\n> ") and andrey.__version__ in text
    assert re.findall(r"^## (.+)$", text, re.M) == ["Docs", "Release cards", "Blog"]
    links = re.findall(r"^- \[(.*?)\]\((https://andrey\.abel\.ai/[^)]+)\): (.+)$", text, re.M)
    assert len(links) >= 12
    assert len(links) == len({url for _, url, _ in links})
    for title, url, description in links:
        path = urlsplit(url).path.lstrip("/")
        page = html / path
        assert page.is_file(), url
        assert "PR/" not in path and "_modules/" not in path and "search.html" not in path
        content = page.read_text()
        preview_title = re.search(r'<meta property="og:title" content="([^"]*)"', content)[1]
        preview_description = re.search(
            r'<meta property="og:description" content="([^"]*)"', content
        )[1]
        assert title == unescape(preview_title)
        assert description == unescape(preview_description)


def test_the_sitemap_lists_each_indexable_page_once_and_robots_txt_names_it(html):
    assert (html / "robots.txt").read_text() == (
        "User-agent: *\nAllow: /\n\nSitemap: https://andrey.abel.ai/sitemap.xml\n"
    )
    urls = re.findall(r"<loc>([^<]+)</loc>", (html / "sitemap.xml").read_text())
    assert len(urls) == len(set(urls)) > 50
    for url in urls:
        path = urlsplit(url).path.lstrip("/") or "index.html"
        content = (html / path).read_text()
        assert f'<meta property="og:url" content="{url}"' in content, url
        assert content.count('rel="canonical"') == 1, url
        assert f'<link rel="canonical" href="{url}">' in content, url
        assert 'name="robots"' not in content, url
    for required in ("", "docs/faq.html", "docs/benchmarks/0.4.0.html", "blog/index.html"):
        assert f"https://andrey.abel.ai/{required}" in urls, required
    # Left out: pages with no content of their own, and the redirects from moved addresses.
    left_out = (
        "search.html",
        "404.html",
        "genindex.html",
        "blog/drafts.html",
        "demo/index.html",
        "docs/demos/benchmark-explorer.html",
    )
    moved = (
        "blog/andrey-0-2-0.html",
        "blog/introducing-andrey.html",
        "blog/why-andrey-is-fast.html",
    )
    for address in (*left_out, *moved):
        assert f"https://andrey.abel.ai/{address}" not in urls, address


def test_only_the_search_page_and_the_404_page_are_noindex(html):
    for page, text in _pages(html):
        robots = re.findall(r'<meta name="robots" content="([^"]*)"', text)
        if page in (html / "search.html", html / "404.html"):
            assert robots == ["noindex, follow"], page
        else:
            assert not robots, page


def test_the_404_page_resolves_its_links_from_the_site_root(html):
    # The server shows 404.html at whatever address was missing, so its links and files resolve
    # from the root, whatever the depth: a <base href="/"> ahead of the first relative address.
    text = (html / "404.html").read_text()
    head = text.split("</head>", 1)[0]
    assert head.count('<base href="/">') == 1
    first = min(head.index(m) for m in ('href="_static', 'src="_static') if m in head)
    assert head.index('<base href="/">') < first
    # Its in-page links stay on the page rather than resolving to the homepage.
    assert "a.href = location.pathname + a.getAttribute('href')" in head
    assert "<h1>Page not found" in text and (html / "_static" / "missing-link.jpg").is_file()
    body = text.split("</head>", 1)[1]
    targets = {
        h.split("#")[0]
        for h in re.findall(r'href="([^"#:][^":]*)"', body)
        if not h.startswith(("/", "//")) and h.split("#")[0]
    }
    assert {"docs/index.html", "docs/benchmarks.html", "blog/index.html"} <= targets
    for target in targets:
        assert (html / target).exists(), target


def test_every_page_ends_on_the_same_footer(html):
    # Every page, the homepage included, ends on one footer without the theme's credits ("Created
    # using Sphinx", "Built with the PyData Sphinx Theme"), and credits Abel with one link that
    # holds its mark and its name.
    for page, text in _pages(html):
        footer = text[text.index('<footer class="bd-footer">') :]
        assert 'class="docs-footer"' in footer, page
        assert re.findall(r'<p class="docs-footer__heading">([^<]+)</p>', footer) == [
            "Documentation",
            "Learn",
            "Project",
        ], page
        assert "Created using" not in footer and "PyData Sphinx Theme" not in footer, page
        abel = footer[footer.index('<a class="docs-footer__abel" href="https://abel.ai/">') :]
        abel = abel[: abel.index("</a>")]
        assert '<svg class="abel-mark"' in abel and abel.endswith("Abel AI Lab"), page
        assert "Built by" not in footer, page
        for href in re.findall(r'href="([^"#:]+\.html)"', footer):
            assert (page.parent / href).resolve().is_file(), (page, href)


def test_the_brand_kit_offers_every_file_it_shows_and_zips_them(html):
    import zipfile

    from PIL import Image

    page = (html / "docs" / "brand.html").read_text()
    files = sorted(set(re.findall(r'href="\.\./_static/brand/([^"]+)" download', page)))
    assert files and "andrey-brand-kit.zip" in files
    kit = html / "_static" / "brand"
    for name in files:
        assert (kit / name).is_file(), name
    with zipfile.ZipFile(kit / "andrey-brand-kit.zip") as zipped:
        names = {pathlib.PurePosixPath(n).name for n in zipped.namelist()}
    assert set(files) - {"andrey-brand-kit.zip"} <= names and "README.txt" in names
    with Image.open(kit / "andrey-avatar.png") as avatar:
        assert avatar.size == (1024, 1024)
    with Image.open(kit / "andrey-icon-on-light.png") as icon:
        assert icon.size == (512, 512)


def test_every_local_image_address_carries_its_content_hash(html):
    # A deploy changes a file's address with its content, so no cache serves the old image.
    import hashlib

    problems = []
    for page in [html / "index.html", html / "docs" / "index.html", html / "404.html"]:
        text = page.read_text(encoding="utf-8")
        refs = re.findall(r'\bsrc="([^"#:]+\.(?:svg|jpg|png))(?:\?v=(\w+))?"', text)
        assert refs, page
        for ref, version in refs:
            target = html / ref.lstrip("/") if ref.startswith("/") else _resolve(html, page, ref)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()[:8]
            if version != digest:
                problems.append(f"{page.relative_to(html)}: {ref}?v={version} (file {digest})")
    unhashed = re.compile(r'url\("?([^")?#:]+\.(?:svg|jpg|png))(?!\?v=)"?\)')
    for sheet in (html / "_static").glob("*.css"):
        problems += [
            f"{sheet.name}: url({ref}) has no hash" for ref in unhashed.findall(sheet.read_text())
        ]
    assert not problems, "\n".join(problems)
