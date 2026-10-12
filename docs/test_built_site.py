"""Checks on the built website that the unit tests cannot make.

Run from the repo root: ``uv run --group docs python -m pytest docs/test_built_site.py``. The first
test that needs the site builds all of it from a clean workspace with ``-W``, which executes the
notebooks and takes a few minutes; the checks then read the built pages. The template and source
checks are unit tests, which the docs workflow runs first.
"""

from __future__ import annotations

import json
import pathlib
import posixpath
import re
import runpy
import shutil
import subprocess
import sys
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlsplit

import pytest

import andrey
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
    for script in ("tiles.js",):
        if f"{script}" not in home or not (html / "_static" / script).is_file():
            problems.append(f"{script}: not loaded or not built")
    assert not problems, "\n".join(problems)


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
                rf'<a class="nav-link nav-internal" href="([^"]+)"[^>]*>{label}</a>', header
            )
            if len(hrefs) != 1 or _resolve(html, page, hrefs[0]) != html / expected:
                problems.append(f"{name}: the header's {label} link is wrong")
        # Release cards carries the dot that marks the newest release until the reader opens it.
        cards = re.findall(
            r'<a class="nav-link nav-internal" href="([^"]+)"[^>]*>Release cards'
            r'<span class="nav-new" data-release="([^"]+)" hidden>',
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


def test_sharing_metadata_and_structured_data_match_the_existing_page(html):
    for page, text in _pages(html):
        nodes = _discovery_graph(text)
        node = nodes.get("BlogPosting", nodes.get("WebPage"))
        assert node is not None, page
        for suffix, field in (
            ("title", "name"),
            ("description", "description"),
            ("image", "image"),
        ):
            existing = re.findall(rf'<meta property="og:{suffix}" content="([^"]*)"', text)
            twitter = re.findall(rf'<meta name="twitter:{suffix}" content="([^"]*)"', text)
            assert len(existing) == 1 and existing == twitter, page
            assert node[field] == unescape(existing[0]), page
        software = nodes["SoftwareSourceCode"]
        assert software["name"] == "Andrey" and software["programmingLanguage"] == "Python"
        assert software["license"]["name"] == "Apache-2.0"
        assert software["license"]["url"].endswith("/blob/main/LICENSE")
        assert software["codeRepository"] == "https://github.com/Abel-ai-lab/andrey"
        assert node["about"]["@id"] == software["@id"]


def test_blog_discovery_keeps_the_original_dates_and_release_versions(html):
    for source in (DOCS / "blog").glob("*.md"):
        content = source.read_text()
        if "blogpost: true" not in content:
            continue
        text = (html / "blog" / (source.stem + ".html")).read_text()
        nodes = _discovery_graph(text)
        article = nodes["BlogPosting"]
        assert article["datePublished"] == re.search(r"^date: (.+)$", content, re.M)[1]
        assert (
            nodes["SoftwareSourceCode"]["version"]
            == re.search(r"^version: (.+)$", content, re.M)[1]
        )
        assert "dateModified" not in article
        assert article["author"][0]["name"] == "Shu Wan"
        author_path = urlsplit(article["author"][0]["url"]).path.lstrip("/")
        assert (html / author_path).is_file()


def test_discovery_text_files_link_only_existing_public_pages(html):
    assert (html / "robots.txt").read_text() == "User-agent: *\nAllow: /\n"
    text = (html / "llms.txt").read_text()
    assert text.startswith("# Andrey\n\n> ") and andrey.__version__ in text
    links = re.findall(r"^- \[(.*?)\]\((https://andrey\.abel\.ai/[^)]+)\): (.+)$", text, re.M)
    assert len(links) >= 12
    assert len(links) == len({url for _, url, _ in links})
    for title, url, description in links:
        path = urlsplit(url).path.lstrip("/")
        page = html / path
        assert page.is_file(), url
        assert "PR/" not in path and "_modules/" not in path and "search.html" not in path
        content = page.read_text()
        original_title = re.search(r'<meta property="og:title" content="([^"]*)"', content)[1]
        original_description = re.search(
            r'<meta property="og:description" content="([^"]*)"', content
        )[1]
        assert title == unescape(original_title)
        assert description == unescape(original_description)


def test_search_is_noindex_and_public_content_remains_indexable(html):
    for page, text in _pages(html):
        robots = re.findall(r'<meta name="robots" content="([^"]*)"', text)
        if page == html / "search.html":
            assert robots == ["noindex, follow"]
        else:
            assert not robots, page


SITE = "https://andrey.abel.ai/"
PAGE = {
    "title": "Benchmarks",
    "description": "Andrey benchmark timings, accuracy, and practical limits by method family.",
    "url": SITE + "docs/benchmarks.html",
    "image": SITE + "assets/preview/home.jpg",
}
SOFTWARE = {
    "name": "Andrey",
    "description": "Andrey, a very fast causal discovery package",
    "version": "0.2.0",
    "license": {
        "@type": "CreativeWork",
        "name": "Apache-2.0",
        "url": "https://github.com/Abel-ai-lab/andrey/blob/main/LICENSE",
    },
    "programmingLanguage": "Python",
    "runtimePlatform": "Python >=3.11",
    "codeRepository": "https://github.com/Abel-ai-lab/andrey",
    "sameAs": ["https://pypi.org/project/andrey-core/"],
}


@pytest.fixture
def discovery():
    source = pathlib.Path(__file__).with_name("discovery.py")
    if not source.is_file():

        def missing(*args, **kwargs):
            pytest.fail("The site has no machine-readable discovery renderer")

        return dict.fromkeys(("head_tags", "llms_index", "robots_txt"), missing)
    return runpy.run_path(str(source))


class Head(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.meta = {}
        self.scripts = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("name", attrs.get("property"))
            self.meta[key] = attrs.get("content")
        elif tag == "script":
            assert attrs.get("type") == "application/ld+json"
            self.scripts.append("")
        else:
            pytest.fail(f"Discovery metadata includes a visible element: {tag}")

    def handle_data(self, text):
        if self.scripts:
            self.scripts[-1] += text
        else:
            assert not text.strip()


def render(discovery, page=PAGE, **kwargs):
    return discovery["head_tags"](
        page, site_url=SITE, project="Andrey", author="Abel AI Lab", software=SOFTWARE, **kwargs
    )


def graph(text):
    head = Head(text)
    assert len(head.scripts) == 1
    data = json.loads(head.scripts[0])
    assert data["@context"] == "https://schema.org"
    return {node["@type"]: node for node in data["@graph"]}


def test_sharing_fields_reuse_the_existing_page_values(discovery):
    head = Head(render(discovery))
    assert head.meta["twitter:title"] == PAGE["title"]
    assert head.meta["twitter:description"] == PAGE["description"]
    assert head.meta["twitter:image"] == PAGE["image"]
    assert "description" not in head.meta  # the existing preview renderer owns this tag
    assert "robots" not in head.meta


def test_software_facts_come_from_the_package_metadata(discovery):
    nodes = graph(render(discovery))
    software = nodes["SoftwareSourceCode"]
    for key, value in SOFTWARE.items():
        assert software[key] == value
    assert nodes["Organization"]["name"] == "Abel AI Lab"
    assert nodes["WebPage"]["name"] == PAGE["title"]
    assert nodes["WebPage"]["description"] == PAGE["description"]
    assert nodes["WebPage"]["about"]["@id"] == software["@id"]
    assert "aggregateRating" not in software and "offers" not in software


def test_breadcrumbs_use_the_existing_parent_titles(discovery):
    nodes = graph(render(discovery, parents=[{"title": "Andrey", "url": SITE + "docs/index.html"}]))
    items = nodes["BreadcrumbList"]["itemListElement"]
    assert [x["item"] for x in items] == [SITE, SITE + "docs/index.html", PAGE["url"]]
    assert [x["name"] for x in items] == ["Andrey", "Andrey", PAGE["title"]]
    assert [x["position"] for x in items] == [1, 2, 3]


def test_a_historical_post_keeps_its_date_author_and_software_version(discovery):
    post = {
        "date": "2026-10-08",
        "authors": [{"name": "Shu Wan", "url": SITE + "blog/author/shu-wan.html"}],
        "version": "0.1.0",
    }
    text = render(discovery, post=post)
    nodes = graph(text)
    article = nodes["BlogPosting"]
    assert article["headline"] == PAGE["title"]
    assert article["datePublished"] == post["date"]
    assert "dateModified" not in article
    assert article["author"] == [{"@type": "Person", **post["authors"][0]}]
    assert article["mainEntityOfPage"]["@id"] == nodes["WebPage"]["@id"]
    assert "breadcrumb" not in article
    assert nodes["SoftwareSourceCode"]["version"] == "0.1.0"
    assert nodes["SoftwareSourceCode"]["@id"].endswith("#software-0.1.0")
    head = Head(text)
    assert head.meta["article:published_time"] == post["date"]
    assert head.meta["article:author"] == post["authors"][0]["url"]


def test_metadata_cannot_close_its_script_or_create_html_elements(discovery):
    title = '</script><img src=x onerror="alert(1)"> & <tag>'
    page = {**PAGE, "title": title, "description": title}
    text = render(discovery, page)
    assert text.count("</script>") == 1
    assert "<img" not in text and "<tag>" not in text
    assert graph(text)["WebPage"]["name"] == title
    assert Head(text).meta["twitter:title"] == title


def test_only_the_search_page_is_marked_noindex(discovery):
    page = {**PAGE, "indexable": False}
    assert Head(render(discovery, page)).meta["robots"] == "noindex, follow"
    assert "robots" not in Head(render(discovery)).meta


def test_llms_navigation_reuses_existing_titles_and_descriptions(discovery):
    entries = {
        "docs/faq": {**PAGE, "title": "FAQ", "description": "The existing FAQ summary."},
        "docs/code/index": {
            **PAGE,
            "title": "API reference",
            "description": "Existing API summary.",
        },
    }
    text = discovery["llms_index"](
        PAGE["description"],
        entries,
        ["docs/faq", "docs/code/index"],
        site_url=SITE,
        project="Andrey",
        version="0.2.0",
    )
    assert text.startswith("# Andrey\n\n> " + PAGE["description"])
    assert "[FAQ](https://andrey.abel.ai/docs/faq.html): The existing FAQ summary." in text
    assert (
        "[API reference](https://andrey.abel.ai/docs/code/index.html): Existing API summary."
        in text
    )
    assert "0.2.0" in text
    assert "941" not in text and "100x" not in text


def test_llms_navigation_rejects_missing_or_non_public_pages(discovery):
    render_index = discovery["llms_index"]
    kwargs = {"site_url": SITE, "project": "Andrey", "version": "0.2.0"}
    with pytest.raises(ValueError, match="missing"):
        render_index("Summary", {}, ["docs/missing"], **kwargs)
    for path in ["search", "_modules/andrey/core", "docs/PR/private", "../private"]:
        with pytest.raises(ValueError, match="public"):
            render_index("Summary", {path: PAGE}, [path], **kwargs)
    with pytest.raises(ValueError, match="indexable"):
        render_index("Summary", {"docs/faq": {**PAGE, "indexable": False}}, ["docs/faq"], **kwargs)


def test_llms_navigation_escapes_markdown_without_changing_the_label(discovery):
    text = discovery["llms_index"](
        "Original\nsummary",
        {"docs/faq": {**PAGE, "title": "FAQ [data]", "description": "One\nline"}},
        ["docs/faq", "docs/faq"],
        site_url=SITE,
        project="Andrey",
        version="0.2.0",
    )
    assert text.count("https://andrey.abel.ai/docs/faq.html") == 1
    assert "[FAQ \\[data\\]]" in text and ": One line" in text


def test_robots_allows_public_search_without_a_training_policy(discovery):
    text = discovery["robots_txt"]()
    assert text == "User-agent: *\nAllow: /\n"
    assert "Sitemap:" not in text and "GPTBot" not in text


def test_navigation_reads_the_built_page_on_incremental_builds(discovery, tmp_path):
    target = tmp_path / "docs" / "faq.html"
    target.parent.mkdir()
    target.write_text(
        '<head><meta property="og:title" content="FAQ &amp; data">'
        '<meta property="og:description" content="The original summary."></head>'
        "<body><p>Original body.</p></body>"
    )
    entries = discovery["read_pages"](tmp_path, ["docs/faq"])
    assert entries["docs/faq"]["title"] == "FAQ & data"
    assert entries["docs/faq"]["description"] == "The original summary."
    assert entries["docs/faq"]["indexable"] is True
    target.write_text(
        target.read_text().replace("</head>", '<meta name="robots" content="noindex"></head>')
    )
    assert not discovery["read_pages"](tmp_path, ["docs/faq"])["docs/faq"]["indexable"]


def test_navigation_requires_real_built_metadata(discovery, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        discovery["read_pages"](tmp_path, ["docs/faq"])
    target = tmp_path / "docs" / "faq.html"
    target.parent.mkdir()
    target.write_text("<html><head></head><body>Page</body></html>")
    with pytest.raises(ValueError, match="metadata"):
        discovery["read_pages"](tmp_path, ["docs/faq"])
