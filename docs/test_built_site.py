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
import shutil
import subprocess
import sys
from html import unescape
from html.parser import HTMLParser

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


def test_every_header_leads_home_and_to_each_section(html):
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


def test_the_blog_lists_every_post_under_its_cover_and_byline(html):
    problems = []
    listing = (html / "blog" / "index.html").read_text(encoding="utf-8")
    for post in (p.stem for p in (DOCS / "blog").glob("*.md") if "blogpost: true" in p.read_text()):
        if f'href="{post}.html"' not in listing:
            problems.append(f"blog/index.html does not list blog/{post}.html")
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
            r'<p class="post-byline">(.*?)</p>',
            text,
            re.S,
        )
        author = byline and re.search(r'<a href="([^"]+)">', byline[1])
        if not author or "<time datetime=" not in byline[1]:
            problems.append(f"blog/{post}.html has no author and date under its title")
        elif not _resolve(html, built, author[1]).is_file():
            problems.append(f"blog/{post}.html links its author to {author[1]}, not built")
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
