"""Sphinx configuration for the Andrey website: the homepage, the blog, and the documentation.

One project builds the whole site. The homepage is its own template (``site/homepage.html``), which
``site/build.py`` fills from the benchmark summary; the blog is ABlog under ``blog/``; the
documentation is MyST under ``docs/``, with autodoc from ``andrey.__all__`` (plus the ``andrey.viz``
surface) and MyST-NB notebooks. Every page shares the pydata-sphinx-theme navbar, restyled to the
``andrey.viz`` palette so the chrome and the figures the package renders share one design system.
"""

from __future__ import annotations

import html
import importlib.util
import inspect
import json
import pathlib
import posixpath
import re
import runpy
import sys
import textwrap
import tomllib
from urllib.parse import urljoin

import ablog.blog
from ablog.blog import Blog
from ablog.post import register_posts
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx import addnodes
from sphinx.errors import ConfigError

import andrey
import andrey.viz
from andrey.data import benchmarks
from andrey.spec import list_specs

if sys.version_info < (3, 11):
    raise RuntimeError("The site build needs Python 3.11 or newer, for ABlog.")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
# Build-time output that is not a page: example pages, thumbnails, the preview card.
GENERATED = HERE / "_generated"


def _module(name: str, path: pathlib.Path):  # noqa: ANN202 -- a module loaded from a file
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The homepage's content generators, and the pages of the interactive examples.
site = _module("andrey_site", ROOT / "site" / "build.py")
cards = _module("andrey_cards", ROOT / "site" / "cards.py")
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "live-discovery"))
build_assets = _module("andrey_build_assets", ROOT / "apps" / "build_assets.py")
compare = _module("andrey_compare", ROOT / "apps" / "compare.py")
discovery = _module("andrey_discovery", HERE / "discovery.py")

# ---- project ---------------------------------------------------------------------------------
project = "Andrey"
author = "Abel AI Lab"
release = andrey.__version__
version = release
copyright = "Abel AI Lab"  # noqa: A001 -- Sphinx's documented config name

# ---- extensions ------------------------------------------------------------------------------
extensions = [
    "ablog",
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.katex",
    "sphinxcontrib.typer",
]

# ---- source ----------------------------------------------------------------------------------
# autosummary writes its stubs with the first suffix here, so .rst leads: a .md stub would reach
# MyST and print its autodoc directives as text.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
# Stubs an earlier build wrote as .md would now sit beside their .rst twins and fail the build.
for _stale in (HERE / "docs" / "code" / "generated").glob("*.md"):
    _stale.unlink()
# The homepage is a template, not a document, so the documentation's index is the root document.
root_doc = "docs/index"
templates_path = ["_templates", "../site"]
# PR/ is gitignored local devflow memory (must never enter the build -> local/CI divergence).
# README.md / DEVELOPMENT.md are contributor references, not site pages (like the repo-root ones).
exclude_patterns = [
    "_build",
    "_generated",
    "PR/**",
    "**/PR/**",
    "README.md",
    "**/README.md",
    "DEVELOPMENT.md",
]

# ---- MyST ------------------------------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "substitution"]
myst_heading_anchors = 4  # mandatory: resolves the `file.md#anchor` cross-refs in the tree

# ---- autodoc / autosummary -------------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
# Types go on documented parameters only, so a class built by the package never lists its
# private constructor fields.
autodoc_typehints_description_target = "documented"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
always_use_bars_union = True

# ---- intersphinx -----------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# ---- notebooks -------------------------------------------------------------------------------
EXAMPLES = HERE / "docs" / "examples"
nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_execution_in_temp = True
nb_execution_cache_path = str(HERE / "_build" / ".jupyter_cache")
# Timings belong to the reader's machine, not the docs builder; the page ships without outputs.
nb_execution_excludepatterns = ["docs/examples/speed_on_your_machine.ipynb"]
# Warnings print local paths.
nb_output_stderr = "remove"


def _copy_notebooks(app) -> None:  # noqa: ANN001 -- Sphinx passes its application object
    """Stage the canonical notebooks beside the gallery before Sphinx discovers pages."""
    source = ROOT / "examples"
    expected = {p.name for p in source.glob("*.ipynb")}
    for stale in EXAMPLES.glob("*.ipynb"):
        if stale.name not in expected:
            stale.unlink()
    for name in expected:
        notebook = json.loads((source / name).read_text())
        # The download is the canonical notebook, without Sphinx-only markup. Its link goes to the
        # right of the Open in Colab badge.
        first = notebook["cells"][0]
        text = "".join(first["source"])
        badge = re.search(r"^\[!\[Open in Colab\]\(.+\)$", text, flags=re.MULTILINE)
        if badge is None:
            raise ConfigError(f"examples/{name} has no Open in Colab badge in its first cell")
        download = f" {{download}}`Download this notebook <../../../examples/{name}>`"
        first["source"] = text[: badge.end()] + download + text[badge.end() :]
        content = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        staged = EXAMPLES / name
        if not staged.exists() or staged.read_text() != content:
            staged.write_text(content)


# ---- copy buttons ---------------------------------------------------------------------------
# Every code block gets the same small copy button, the homepage's too (their frames and the install
# line). A console session or a doctest copies only its commands, without the prompts.
copybutton_selector = "div.highlight pre, .code-copy > pre, .install-line code"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ---- KaTeX (no MathJax) ----------------------------------------------------------------------
katex_prerender = True

# ---- HTML theme ------------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Andrey"
# site/brand/ holds the wordmark; site/assets/ the homepage's and the figures' styles and script.
html_static_path = ["_static", "_generated/static", "../site/brand", "../site/assets"]
# The wordmark's A with its right stroke in amber (`site/wordmark.py`).
html_favicon = "../site/brand/favicon.svg"
# Served from the site root as they are: the link-preview card.
html_extra_path = ["_generated/extra", "_extra"]
html_css_files = ["andrey-tokens.css", "andrey.css", "graph-guides.css"]
# The navbar's Liquid Glass, where the browser draws it (andrey.css has the frosted pill), the dot
# on Release cards until the reader has seen the latest release, and the outline's mark on the
# section being read.
html_js_files = [
    ("glass.js", {"defer": "defer"}),
    ("nav-new.js", {"defer": "defer"}),
    ("page-outline.js", {"defer": "defer"}),
]
REPO_URL = "https://github.com/Abel-ai-lab/andrey"
PYPI_URL = "https://pypi.org/project/andrey-core/"
SITE_URL = "https://andrey.abel.ai/"
ABEL_URL = "https://abel.ai/"
# Abel's logomark in Night Teal, for light backgrounds, from Abel's brand kit: the structured
# data's organization logo, drawn at the build as a PNG wide enough for search results.
ABEL_SYMBOL = ROOT / "site" / "brand" / "abel-symbol-night.svg"
ABEL_LOGO = "_static/abel-symbol.png"
ABEL_LOGO_WIDTH = 600
# Pages a search engine should not list: the search page, and the page the server shows for a
# missing address (404.md).
UNINDEXED = {"search", "404"}
# The pages llms.txt lists, by section; each release card and published post follows its section's
# pages, newest first.
LLMS_SECTIONS = (
    (
        "Docs",
        (
            "docs/index",
            "docs/guides/getting-started",
            "docs/faq",
            "docs/code/index",
            "docs/configuration",
            "docs/guides/graph-types",
            "docs/guides/graph-representation",
            "docs/examples/index",
            "docs/demos/index",
            "docs/changelog",
        ),
    ),
    ("Release cards", ("docs/benchmarks",)),
    ("Blog", ("blog/index",)),
)
PACKAGE = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
html_theme_options = {
    # One header on every page (_templates/site-*.html): the wordmark to the homepage and the
    # release to the changelog, then Docs, Examples, Blog, and FAQ, then search, theme, and GitHub.
    "navbar_start": ["site-logo", "site-version"],
    "navbar_center": ["site-nav"],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_prev_next": False,
    "icon_links": [{"name": "GitHub", "url": REPO_URL, "icon": "fa-brands fa-github"}],
    # The documentation footer (`_templates/docs-footer.html`); the theme's three areas are empty.
    "footer_start": ["docs-footer"],
    "footer_center": [],
    "footer_end": [],
    "use_edit_page_button": False,
    # A documentation page's section over its title (_templates/page-kicker.html); other pages
    # keep the breadcrumbs.
    "article_header_start": ["page-kicker"],
    # The page's outline (_templates/page-toc.html), then, on a documentation page, a link that
    # edits it and one that reports a problem with it (_templates/page-links.html).
    "secondary_sidebar_items": ["page-toc", "page-links"],
}
# The theme follows the reader's system setting until they pick one; unset, its script logs
# "invalid theme mode" on a first visit.
html_context = {
    "default_mode": "auto",
    "report_bug_url": f"{REPO_URL}/issues/new",
    "repo_url": REPO_URL,
    "pypi_url": PYPI_URL,
    # Abel's mark uses the text color in the documentation footer.
    "abel_mark": site.abel_mark(),
}
# A post's author and date sit under its title (_post_byline); the blog index links the archive.
BLOG_SIDEBAR = ["ablog/archives.html"]
html_sidebars = {
    "index": [],
    # The page's outline (phones only) and the whole documentation tree, since the navbar names
    # only the site's sections.
    "docs/**": ["page-outline.html", "docs-nav.html"],
    "blog": BLOG_SIDEBAR,
    # A post takes the full width beside its contents, so its charts have room.
    "blog/**": [],
}
# The documentation tree's groups (_docs_sidebar): each section of the docs index named here heads
# a group of its pages under its icon (_templates/nav-icons.html), and the index's other entries
# make the last group. A section folds its pages past the first NAV_SHOWN.
NAV_SECTIONS = {
    "docs/guides/index": "book",
    "docs/examples/index": "notebook",
    "docs/demos/index": "play",
}
NAV_REFERENCE = ("Reference", "code")
NAV_SHOWN = 5
html_additional_pages = {"index": "homepage.html", "demo/index": "demo-redirect.html"}

# ---- blog (ABlog) ----------------------------------------------------------------------------
# A post is a page under blog/ whose front matter sets `blogpost: true`, with its `date`. ABlog
# keeps a post without a date as a draft. A post dated in the future is published like any other:
# ABlog would keep it as a draft until then, which fails the docs checks before its day.
ablog.blog.TOMORROW = ablog.blog.FUTURE
blog_path = "blog"
blog_title = "Andrey blog"
blog_baseurl = SITE_URL
blog_authors = {"Shu Wan": ("Shu Wan", None)}
blog_feed_fulltext = True
# A post renamed after release keeps its old address: front matter `redirect` names it, and the
# page there sends the reader on at once.
post_redirect_refresh = 0
post_date_format = "%B %-d, %Y"
post_date_format_short = "%B %-d, %Y"
post_auto_image = 0
fontawesome_included = True


# ---- generated palette -----------------------------------------------------------------------
# The site palette is the package palette. `andrey.viz.theme` is the single source of truth, so the
# hexes are emitted as CSS custom properties at build time; a transcribed stylesheet would drift on
# the next palette edit. `_static/andrey.css` consumes these tokens and names no color itself. The
# generated file is gitignored.
_TOKENS_CSS = HERE / "_static" / "andrey-tokens.css"
palette_block = runpy.run_path(str(HERE / "palette_css.py"))["palette_block"]


def _write(path: pathlib.Path, text: str) -> None:
    """Write ``text`` only on a real change, staged then replaced.

    The targets live under the watched source dir, so an unconditional rewrite makes
    `sphinx-autobuild` detect its own output and rebuild forever, and a concurrent build must never
    copy a half-written file.
    """
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def _write_tokens(app) -> None:  # noqa: ANN001 -- Sphinx passes its application object
    """Emit `_static/andrey-tokens.css` from the light and dark palettes."""
    header = (
        "/* GENERATED by docs/conf.py from andrey.viz.theme -- do not edit.\n"
        "   Edit the palette in src/andrey/viz/theme.py; it propagates here on the next build.\n"
        "   pydata-sphinx-theme sets data-theme on <html>, so the theme toggle switches these. */\n"
    )
    blocks = [
        palette_block(':root, html[data-theme="light"]', andrey.viz.LIGHT),
        palette_block('html[data-theme="dark"]', andrey.viz.DARK),
        # Gallery thumbnails are drawn in the light palette in both themes, on its paper.
        f":root {{\n  --andrey-light-paper: {andrey.viz.LIGHT.paper};\n}}",
        # The stipple ground's tones (site/grain.py), the same in both themes.
        ":root {\n"
        + "".join(f"  {name}: {value};\n" for name, value in cards.ground_tokens().items())
        + "}",
    ]
    _write(_TOKENS_CSS, header + "\n" + "\n\n".join(blocks) + "\n")


def _write_diagrams(app) -> None:  # noqa: ANN001 -- Sphinx passes its application object
    """Generate guide figures from validated graphs and the package SVG renderer."""
    runpy.run_path(str(HERE / "graph_diagrams.py"), run_name="__main__")


# ---- the homepage and the blog ---------------------------------------------------------------
def _load_summary(app) -> dict:  # noqa: ANN001
    """The benchmark summary; one outside benchmarks/published/ loads only as a placeholder."""
    path = pathlib.Path(app.config.andrey_summary)
    path = path if path.is_absolute() else ROOT / path
    try:
        return site.load_summary(path, placeholder=app.config.andrey_placeholder)
    except ValueError as error:  # Sphinx reports its own errors as one line, with no traceback
        raise ConfigError(str(error)) from None


def _methods_text() -> dict[str, str]:
    """The method count and tiers, for the FAQ."""
    supported = [f"`{s.name}`" for s in list_specs() if s.status == "supported"]
    return {
        "method_count": f"{len(list_specs())} methods",
        "n_supported": str(len(supported)),
        "n_experimental": str(len(list_specs()) - len(supported)),
        "supported_methods": ", ".join(supported[:-1]) + f", and {supported[-1]}",
    }


# Brand kit colors: each color's name, role, and palette value.
BRAND_COLORS = (
    ("Ink", "Text and the wordmark on light grounds", andrey.viz.LIGHT.ink),
    ("Paper", "The light ground", andrey.viz.LIGHT.paper),
    ("Cobalt", "Links, selection, and the art's ground", cards.grain.COBALT),
    ("Navy", "The art's deep ground and the footer", cards.grain.NAVY),
    ("Night", "The dark ground", andrey.viz.DARK.paper),
    ("Pale ink", "Text and the wordmark on dark grounds", andrey.viz.DARK.ink),
    ("Amber", "The sole accent: one node or one edge", cards.grain.AMBER),
)


def _brand_colors() -> str:
    """Brand kit swatches (``docs/brand.md``), each with its palette value."""
    items = "".join(
        f'<li class="brand-color"><span class="brand-color__chip" style="background:{value}">'
        f'</span><span class="brand-color__name">{name}</span>'
        f'<code class="brand-color__value">{value.upper()}</code>'
        f'<span class="brand-color__role">{role}</span></li>'
        for name, role, value in BRAND_COLORS
    )
    return f'<ul class="brand-colors">{items}</ul>\n'


def _site_content(app, config) -> None:  # noqa: ANN001 -- Sphinx passes its application objects
    """Fill the homepage's context from the summary, and the FAQ's method counts."""
    (GENERATED / "static").mkdir(parents=True, exist_ok=True)
    summary = _load_summary(app)
    app.andrey_home = site.homepage_context(summary, placeholder=config.andrey_placeholder)
    config.myst_substitutions = {**config.myst_substitutions, **_methods_text()}
    # The link-preview card shared by the homepage and every page that is not a post.
    site.fastest(summary)
    home_card = cards.card_jpeg(cards.TAGLINE, cards.ACCENT)
    _write_bytes(GENERATED / "extra" / HOME_PREVIEW, home_card)
    # The homepage hero draws the cards' stipple from CSS gradients and these repeating tiles.
    for name, tile in cards.grain.hero_tiles():
        _write_bytes(GENERATED / "static" / name, tile)
    # A release cover or tile without its own art (_static/benchmarks.css).
    _write_bytes(GENERATED / "static" / "release-ground.jpg", cards.release_ground_jpeg())
    # Brand kit downloads (`_static/brand/`) and colors from the palette.
    for name, data in cards.brand_kit().items():
        _write_bytes(GENERATED / "static" / "brand" / name, data)
    _write(GENERATED / "brand-colors.html", _brand_colors())


# Link previews (Open Graph, Twitter card), drawn by site/cards.py. The homepage's card carries the
# slogan and each post's its title; every other page shares the first.
HOME_PREVIEW = "assets/preview/home.jpg"
# Search results show about this many characters of a description.
DESCRIPTION_LENGTH = 160
# A paragraph inside one of these is not the page's prose: a note, a comment, a parameter's entry,
# or a signature. Sphinx's API entry (`desc`) is an admonition too, but its docstring is prose.
NOT_PROSE = (nodes.Admonition, nodes.Invisible, nodes.field_list, addnodes.desc_signature)


def _prose(paragraph: nodes.paragraph) -> bool:
    """Whether ``paragraph`` is the page's own prose: inside nothing in ``NOT_PROSE``, and holding
    no image, such as a notebook's Open in Colab badge."""
    node = paragraph.parent
    while node is not None:
        if isinstance(node, NOT_PROSE) and not isinstance(node, addnodes.desc):
            return False
        node = node.parent
    return next(paragraph.findall(nodes.image), None) is None


def _first_paragraph(doctree: nodes.document) -> str:
    """The plain text of the page's first paragraph of prose, or "" when it has none. An API
    page's is its docstring's summary line."""
    for paragraph in doctree.findall(nodes.paragraph):
        if _prose(paragraph) and (text := " ".join(paragraph.astext().split())):
            return text
    return ""


def _description(app, context: dict, doctree: nodes.document | None) -> str:  # noqa: ANN001
    """The page's description: the homepage's sentence, the front matter's ``description``, or the
    first paragraph cut at a word to ``DESCRIPTION_LENGTH``. A page with neither, such as search or
    the blog archive, takes the homepage's."""
    if home := context.get("home"):
        return home["description"]
    meta = context.get("meta") or {}
    if (meta.get("description") or "").strip():  # an empty value counts as none
        return meta["description"].strip()
    paragraph = _first_paragraph(doctree) if doctree is not None else ""
    if paragraph:
        return textwrap.shorten(
            paragraph, DESCRIPTION_LENGTH, placeholder="…", break_on_hyphens=False
        )
    return app.andrey_home["description"]


def _preview_tags(app, pagename: str, context: dict, doctree) -> str:  # noqa: ANN001
    """The page's description, its Open Graph and Twitter card tags (title, summary, address, and
    image), and its structured data (``discovery.head_tags``): the organization, the website, the
    software at the release the page belongs to, the page or post, and its breadcrumbs. The pages
    in ``UNINDEXED`` are ``noindex``. The description and ``og:description`` are one value."""
    home = context.get("home")
    heading = re.sub(r"<[^>]+>", "", context.get("title") or app.config.project)
    title = home["title"] if home else html.unescape(heading)
    description = _description(app, context, doctree)
    url = SITE_URL + ("" if pagename == "index" else f"{pagename}.html")
    image = HOME_PREVIEW
    if pagename in _blog(app):  # each post's preview is its banner: its art, title, and release
        image = f"assets/preview/{pagename.removeprefix('blog/')}.jpg"
        preview = cards.preview_jpeg(_cover_kind(app, pagename), title, _cover_line(app, pagename))
        _write_bytes(pathlib.Path(app.outdir) / image, preview)
    image = SITE_URL + image
    tags = {
        "description": description,
        "og:type": "article" if pagename in _blog(app) else "website",
        "og:site_name": app.config.project,
        "og:title": title,
        "og:description": description,
        "og:url": url,
        "og:image": image,
        "og:image:type": "image/jpeg",
        "og:image:width": str(cards.W),
        "og:image:height": str(cards.H),
        "twitter:card": "summary_large_image",
    }
    preview = "".join(
        f'<meta {"property" if key.startswith("og:") else "name"}="{key}" '
        f'content="{html.escape(value, quote=True)}">\n'
        for key, value in tags.items()
    )
    page = {"title": title, "description": description, "url": url, "image": image}
    if pagename in UNINDEXED:
        page["indexable"] = False
    if match := re.fullmatch(r"docs/benchmarks/(\d+\.\d+\.\d+)", pagename):
        page["software_version"] = match[1]
    post = None
    if pagename in Blog(app):
        article = Blog(app)[pagename]
        post = {
            "date": article.date.strftime("%Y-%m-%d") if article.date else None,
            "authors": [
                {
                    "name": str(person),
                    "url": urljoin(url, app.builder.get_relative_uri(pagename, person.docname)),
                }
                for person in article.author or ()
            ],
            "version": app.env.metadata.get(pagename, {}).get("version"),
        }
    parents = [
        {
            "title": html.unescape(re.sub(r"<[^>]+>", "", parent["title"])),
            "url": urljoin(url, parent["link"]),
        }
        for parent in context.get("parents", [])
    ]
    software = {
        "name": app.config.project,
        "description": PACKAGE["description"],
        "version": release,
        "license": {
            "@type": "CreativeWork",
            "name": PACKAGE["license"],
            "url": f"{PACKAGE['urls']['Source']}/blob/main/{PACKAGE['license-files'][0]}",
        },
        "programmingLanguage": "Python",
        "runtimePlatform": f"Python {PACKAGE['requires-python']}",
        "codeRepository": PACKAGE["urls"]["Source"],
        # Andrey's own icon, the favicon; the organization's node carries Abel's logo.
        "image": SITE_URL + "_static/favicon.svg",
        "sameAs": [f"https://pypi.org/project/{PACKAGE['name']}/"],
    }
    return preview + discovery.head_tags(
        page,
        site_url=SITE_URL,
        project=app.config.project,
        author=app.config.author,
        organization_url=ABEL_URL,
        organization_logo=SITE_URL + ABEL_LOGO,
        software=software,
        language=app.config.language or "en",
        post=post,
        parents=parents,
    )


# An image keeps its address across releases, and Cloudflare and browsers cache it for hours, so a
# deploy would show the old file. After the build, each local image a stylesheet or a page loads
# (a CSS `url()`, an `src`, the favicon link) gets its content's hash as a query, so a changed file
# gets a new address; then each page's stylesheet links take the hash of the rewritten stylesheet,
# since Sphinx hashed it before. Download links keep their addresses.
ASSET_TYPES = "svg|jpg|jpeg|png|webp|gif"
ASSET_REFS = (
    (".css", re.compile(rf'(url\("?)([^")?#:]+\.(?:{ASSET_TYPES}))(?="?\))')),
    (
        ".html",
        re.compile(
            rf'((?:\bsrc|<link rel="icon"[^>]*?\bhref)=")([^"?#:]+\.(?:{ASSET_TYPES}))(?=")'
        ),
    ),
    (
        ".html",
        re.compile(r'(<link rel="stylesheet"[^>]*?\bhref=")([^"?#:]+\.css)(?:\?v=\w+)?(?=")'),
    ),
)


def _version_assets(app, exception) -> None:  # noqa: ANN001 -- Sphinx event arguments
    """Add ``?v=<hash>`` to each local image address in the built stylesheets and pages, and to
    each page's stylesheet links (``ASSET_REFS``); an address whose file is not built is left as
    it is. A stylesheet is hashed only after its own images are."""
    if exception is not None or app.builder.name != "html":
        return
    import hashlib

    outdir = pathlib.Path(app.outdir)
    digests: dict[pathlib.Path, str] = {}

    def versioned(match: re.Match, here: pathlib.Path) -> str:
        base = outdir if match[2].startswith("/") else here.parent
        target = (base / match[2].lstrip("/")).resolve()
        if not target.is_file():
            return match[0]
        if target not in digests:
            digests[target] = hashlib.sha256(target.read_bytes()).hexdigest()[:8]
        return f"{match[1]}{match[2]}?v={digests[target]}"

    for suffix, pattern in ASSET_REFS:
        for path in outdir.rglob(f"*{suffix}"):
            text = path.read_text(encoding="utf-8")
            new = pattern.sub(lambda m, here=path: versioned(m, here), text)
            if new != text:
                path.write_text(new, encoding="utf-8")


def _write_discovery(app, exception) -> None:  # noqa: ANN001 -- Sphinx event arguments
    """Write ``llms.txt`` and ``sitemap.xml`` at the site's root from the built pages
    (``docs/discovery.py``), and the organization's logo (``ABEL_LOGO``). ``llms.txt`` lists
    ``LLMS_SECTIONS``, then every release card and published post; a listed page that is missing
    or ``noindex`` fails the build."""
    if exception is not None or app.builder.name != "html":
        return
    outdir = pathlib.Path(app.outdir)
    cards = sorted(
        (n for n in app.env.found_docs if re.fullmatch(r"docs/benchmarks/\d+\.\d+\.\d+", n)),
        key=lambda n: tuple(int(part) for part in n.rsplit("/", 1)[1].split(".")),
        reverse=True,
    )
    posts = [post.docname for post in Blog(app).posts if post.published]
    later = {"Release cards": cards, "Blog": posts}
    sections = [(heading, [*pages, *later.get(heading, ())]) for heading, pages in LLMS_SECTIONS]
    try:
        entries = discovery.read_pages(outdir, [n for _, documents in sections for n in documents])
        text = discovery.llms_index(
            app.andrey_home["description"],
            entries,
            sections,
            site_url=SITE_URL,
            project=app.config.project,
            version=release,
        )
    except ValueError as error:
        raise ConfigError(str(error)) from None
    _write(outdir / "llms.txt", text)
    import resvg_py  # the docs group's renderer, as site/cards.py uses it

    logo = resvg_py.svg_to_bytes(svg_string=ABEL_SYMBOL.read_text(), width=ABEL_LOGO_WIDTH)
    _write_bytes(outdir / ABEL_LOGO, bytes(logo))
    _write(outdir / "sitemap.xml", discovery.sitemap_xml(discovery.sitemap_urls(outdir)))


# The pages that show the "Why Andrey is fast" scenes.
WHY_FAST_PAGES = {"index", "blog/20261008-why-andrey-is-fast"}


def _post_key(post: tuple[str, str, str]) -> tuple[str, tuple[int, ...]]:
    """When a post, (docname, ISO date, version), was published: its date, then its version, so a
    later version breaks a tie in date. A post without a version ("") comes before one with."""
    return post[1], tuple(int(part) for part in post[2].split(".") if part)


def _newest_post(posts: list[tuple[str, str, str]]) -> tuple[str, str]:
    """The newest of ``posts``, each (docname, ISO date, version): its docname and version, by
    ``_post_key``. Posts that tie in both date and version fail, rather than let file order pick
    one."""
    newest = max(posts, key=_post_key)
    tied = sorted(post[0] for post in posts if _post_key(post) == _post_key(newest))
    if len(tied) > 1:
        raise ValueError(f"{', '.join(tied)} are all the newest post; date one of them later")
    return newest[0], newest[2]


def _blog(app) -> Blog:  # noqa: ANN001
    """ABlog's posts, registered from the environment if ABlog has not yet done so. ABlog registers
    them only when it resolves a doctree or collects its pages, so an incremental build that
    rewrites no page reaches the index pages with none."""
    blog = Blog(app)
    if not blog:
        register_posts(app)
    return blog


def _newest_first(posts: list[tuple[str, str, str]]) -> list[str]:
    """The docnames of ``posts``, each as for ``_newest_post``: the newest post, then the others
    newest first by ``_post_key``. Others that tie in both date and version keep their order in
    ``posts``."""
    newest = _newest_post(posts)[0]
    others = sorted((post for post in posts if post[0] != newest), key=_post_key, reverse=True)
    return [newest, *(post[0] for post in others)]


def _dated_posts(app) -> list[tuple[str, str, str]]:  # noqa: ANN001
    """Every dated blog post as (docname, ISO date, version), the version "" where it has none."""
    return [
        (
            post.docname,
            f"{post.date:%Y-%m-%d}",
            str(app.env.metadata[post.docname].get("version") or ""),
        )
        for post in _blog(app).posts
        if post.date
    ]


def _release_post(app) -> tuple[str, str]:  # noqa: ANN001
    """The post about the latest release, whose version the navbar's dot on Release cards names:
    the newest dated blog post with a version, as its docname and version."""
    return _newest_post([post for post in _dated_posts(app) if post[2]])


def _page_assets(app, pagename, templatename, context, doctree) -> None:  # noqa: ANN001
    """Give each page the context and files it needs beyond the shared ones.

    Page styles load after the shared ones (priority 900), so they win a tie.
    """
    # The navbar's dot on Release cards (_templates/site-nav.html) names the latest release.
    context["release_version"] = _release_post(app)[1]
    css, js = [], []
    if pagename == "index":
        context["home"] = app.andrey_home
        context["newest_post"] = _newest_post_card(app)
        css, js = ["figures.css", "site.css"], ["figures.js", "tiles.js", "prompt.js"]
    elif pagename == "blog/index":
        # The posts in the Release cards page's blocks.
        css = ["benchmarks.css", "blog.css"]
    elif pagename == "blog" or pagename.startswith("blog/"):
        css, js = ["figures.css", "blog.css"], ["figures.js"]
    elif pagename in INTERACTIVE:
        css, js = ["examples.css"], ["example-frame.js"]
    elif pagename in ("docs/examples/index", "docs/demos/index"):
        css = ["examples.css"]
    elif pagename == "docs/benchmarks":
        css = ["benchmarks.css"]
    elif pagename.startswith("docs/benchmarks/"):
        # A release card: its cover and its figures.
        css, js = ["benchmarks.css", "figures.css"], ["figures.js"]
    elif pagename == "docs/guides/graph-representation":
        js = ["graph-guides.js"]
    # The "Why Andrey is fast" scenes, on top of the page's own files: Motion, then its script.
    if pagename in WHY_FAST_PAGES:
        css, js = [*css, "why-fast.css"], [*js, "motion.js", "why-fast.js"]
    for name in css:
        app.add_css_file(name, priority=900)
    for name in js:
        app.add_js_file(name, loading_method="defer")
    context["metatags"] = context.get("metatags", "") + _preview_tags(
        app, pagename, context, doctree
    )
    # The server answers a missing address with the 404 page at that address, at any depth, so the
    # page resolves its links and files from the site's root; its in-page links (the skip link and
    # the heading's) stay on the page.
    if pagename == "404":
        context["metatags"] = (
            '<base href="/">\n'
            + context["metatags"]
            + "<script>addEventListener('DOMContentLoaded', () => { for (const a of "
            "document.querySelectorAll('a[href^=\"#\"]')) "
            "a.href = location.pathname + a.getAttribute('href') })</script>\n"
        )


def _docs_sidebar(app, pagename, templatename, context, doctree) -> None:  # noqa: ANN001
    """Give a documentation page its sidebars' context.

    ``docs_nav`` is the documentation tree's groups (_templates/docs-nav.html): a group per entry
    of ``NAV_SECTIONS`` holding its pages, then the docs index's other entries. A row carries its
    ``pages``, itself and every page under it, so the fold that holds the current page opens.
    ``page_kicker`` names the page's section over its title: the page that lists it, or the last
    group for a page the docs index lists itself. ``edit_url`` opens the file that holds the page's
    text on GitHub: a Markdown file or notebook it includes (the changelog, a staged notebook's
    canonical copy), else its own source, and none for a page autosummary writes.
    """
    if not pagename.startswith("docs/"):
        return
    includes = app.env.toctree_includes

    def title(doc: str) -> str:
        return app.env.titles[doc].astext()

    def pages(doc: str) -> set[str]:
        return {doc}.union(*(pages(child) for child in includes.get(doc, ())))

    def row(doc: str) -> dict:
        kids = dict.fromkeys(includes.get(doc, ()))
        return {
            "doc": doc,
            "title": title(doc),
            "pages": pages(doc),
            "kids": [row(k) for k in kids],
        }

    top = list(dict.fromkeys(includes[root_doc]))
    groups = [
        {**row(doc), "icon": NAV_SECTIONS[doc], "shown": NAV_SHOWN}
        for doc in top
        if doc in NAV_SECTIONS
    ]
    rest = [row(doc) for doc in top if doc not in NAV_SECTIONS]
    name, icon = NAV_REFERENCE
    groups.append({"doc": None, "title": name, "icon": icon, "kids": rest, "shown": len(rest)})
    context["docs_nav"] = groups

    parent = next((doc for doc, kids in includes.items() if pagename in kids), None)
    if parent == root_doc:
        kicker = None if pagename in NAV_SECTIONS else {"doc": None, "title": name}
    else:
        kicker = parent and {"doc": parent, "title": title(parent)}
    context["page_kicker"] = kicker

    included = sorted(
        path
        for path in (pathlib.Path(dep).resolve() for dep in app.env.dependencies.get(pagename, ()))
        if path.suffix in {".md", ".ipynb"} and GENERATED not in path.parents
    )
    source = included[0] if included else pathlib.Path(app.env.doc2path(pagename))
    generated = "generated" in source.parts
    context["edit_url"] = None if generated else f"{REPO_URL}/edit/main/{source.relative_to(ROOT)}"


def _post_byline(app, doctree, docname) -> None:  # noqa: ANN001
    """Put a post's cover and then its authors, date, the Andrey version it was written for, and its
    tags under its title, where a reader looks for them and where they stay on a narrow screen. A
    post is not updated after it is published, so the version dates what it says. ABlog has read
    the posts' front matter by now."""
    blog = _blog(app)
    if app.builder.format != "html" or docname not in blog:
        return
    post = blog[docname]
    title = next(doctree.findall(nodes.title), None)
    if title is None:
        return
    authors = ", ".join(
        f'<a href="{app.builder.get_relative_uri(docname, author.docname)}">'
        f"{html.escape(str(author))}</a>"
        for author in post.author or ()
    )
    day = (
        f'<time datetime="{post.date:%Y-%m-%d}">'
        f"{post.date.strftime(app.config.post_date_format)}</time>"
        if post.date
        else "Draft"
    )
    version = app.env.metadata.get(docname, {}).get("version")
    written_for = f" · Andrey {html.escape(str(version))}" if version else ""
    byline = (
        f'<div class="post-byline"><span>{authors}{" · " if authors else ""}{day}{written_for}'
        f"</span>{_tag_list(app, docname, post)}</div>"
    )
    at = title.parent.index(title)
    title.parent.insert(at + 1, nodes.raw("", _post_cover(app, docname), format="html"))
    title.parent.insert(at + 2, nodes.raw("", byline, format="html"))


def _cover_kind(app, docname: str) -> str:  # noqa: ANN001
    """The post's art: its front matter's ``cover`` (``cards.COVERS``), else the stipple ground."""
    return app.env.metadata.get(docname, {}).get("cover", "ground")


def _cover_line(app, docname: str) -> str:  # noqa: ANN001
    """The banner's short line: "Release <version>" for a release post, else the post's kind (its
    first tag)."""
    meta = app.env.metadata.get(docname, {})
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip(" \"'") for tag in tags.strip("[]").split(",") if tag.strip()]
    kind = tags[0] if tags else ""
    version = str(meta.get("version") or "")
    return f"{kind} {version}" if kind == "Release" and version else kind


def _tag_list(app, page: str, post) -> str:  # noqa: ANN001
    """The post's tags as a list, each a link to ABlog's page of the posts that share it."""
    tags = "".join(
        f'<li><a href="{app.builder.get_relative_uri(page, tag.docname)}">'
        f"{html.escape(tag.name)}</a></li>"
        for tag in post.tags
    )
    return f'<ul class="blog-tags" aria-label="Tags">{tags}</ul>' if tags else ""


def _cover_file(app, docname: str) -> tuple[str, str]:  # noqa: ANN001
    """Write the post's cover, its art (site/cards.py), into the build: its path from the site's
    root, and its kind. The post and the blog's front page both show it."""
    kind = _cover_kind(app, docname)
    image = f"assets/covers/{docname.removeprefix('blog/')}.jpg"
    _write_bytes(pathlib.Path(app.outdir) / image, cards.cover_jpeg(kind))
    return image, kind


def _thumb_file(app, docname: str) -> str:  # noqa: ANN001
    """Write the post's art as a small thumbnail (site/cards.py) into the build: its path from the
    site's root."""
    image = f"assets/thumbs/{docname.removeprefix('blog/')}.jpg"
    _write_bytes(pathlib.Path(app.outdir) / image, cards.thumb_jpeg(_cover_kind(app, docname)))
    return image


def _newest_post_card(app) -> str:  # noqa: ANN001
    """The homepage's documentation band leads with the newest post, as the blog's front page
    does (``_newest_first``): its cover, its date, its title, and its description, linking to
    it. The cover is the post's art alone, so the title is in the text."""
    docname = _newest_first(_dated_posts(app))[0]
    post = next(post for post in _blog(app).posts if post.docname == docname)
    image, _ = _cover_file(app, docname)
    description = html.escape((app.env.metadata[docname].get("description") or "").strip())
    return (
        f'<a class="band-post" href="{app.builder.get_relative_uri("index", docname)}">'
        f'<img src="{image}" alt="" width="{cards.COVER_W}" height="{cards.COVER_H}" '
        'loading="lazy"><span class="band-post-body"><span class="band-post-kicker">Latest post · '
        f'<time datetime="{post.date:%Y-%m-%d}">{post.date.strftime(app.config.post_date_format)}'
        f'</time></span><span class="band-post-title">{html.escape(post.title)}</span>'
        f"<span>{description}</span></span></a>"
    )


def _post_cover(app, docname: str) -> str:  # noqa: ANN001
    """The post's cover figure (``_cover_file``); the front matter's ``cover_caption`` adds a
    caption."""
    image, kind = _cover_file(app, docname)
    caption = app.env.metadata.get(docname, {}).get("cover_caption")
    src = posixpath.relpath(image, posixpath.dirname(docname))
    return (
        f'<figure class="post-cover"><img class="dark-light" src="{src}" '
        f'alt="{html.escape(cards.cover_alt(kind), quote=True)}" '
        f'width="{cards.COVER_W}" height="{cards.COVER_H}">'
        + (f"<figcaption>{html.escape(caption)}</figcaption>" if caption else "")
        + "</figure>"
    )


# ---- the blog's front page -------------------------------------------------------------------
class _BlogIndex(Directive):
    """``blog-index``: where a page lists the posts, which ``_blog_index`` fills once ABlog has
    read every post."""

    def run(self) -> list[nodes.Node]:
        return [nodes.container(classes=["blog-index"])]


# Above the earlier posts, a button reverses their order; the page stays in order without it.
BLOG_ORDER_SCRIPT = """<script>
document.querySelectorAll(".blog-order").forEach((button) => {
  button.hidden = false;
  button.addEventListener("click", () => {
    const list = document.getElementById(button.getAttribute("aria-controls"));
    list.append(...[...list.children].reverse());
    const newest = button.dataset.order !== "newest";
    button.dataset.order = newest ? "newest" : "oldest";
    button.querySelector("span").textContent = newest ? "Newest first" : "Oldest first";
  });
});
</script>"""


def _blog_index(app, doctree, docname) -> None:  # noqa: ANN001
    """List the dated posts in place of ``blog-index``, newest first (``_newest_first``): the
    newest with its cover across the block, then its title, date, tags, and description as its
    TL;DR; then the others as rows, each with a small thumbnail of its art (``_thumb_file``),
    loaded as the reader scrolls to it, under a button that reverses their order. The Release
    cards page's blocks style both (``_static/benchmarks.css``)."""
    places = [node for node in doctree.findall(nodes.container) if "blog-index" in node["classes"]]
    if app.builder.format != "html" or not places:
        return
    meta = app.env.metadata
    posts = {post.docname: post for post in _blog(app).posts if post.date}
    order = _newest_first(_dated_posts(app))
    latest, rows = "", []
    for name in order:
        post = posts[name]
        description = html.escape((meta[name].get("description") or "").strip())
        if not description:
            raise ValueError(f"{name} sets no description, which the blog's front page shows")
        href = app.builder.get_relative_uri(docname, name)
        title = html.escape(post.title)
        day = (
            f'<time datetime="{post.date:%Y-%m-%d}">'
            f"{post.date.strftime(app.config.post_date_format)}</time>"
        )
        read = f'<a href="{href}">Read the post<span class="visually-hidden">: {title}</span></a>'
        tags = _tag_list(app, docname, post)
        # The title links to the post, so the cover's link is for the pointer only.
        link = f'<a class="blog-tile" href="{href}" tabindex="-1" aria-hidden="true">'
        if not latest:
            image, _ = _cover_file(app, name)
            cover = (
                f'{link}<img class="dark-light" '
                f'src="{posixpath.relpath(image, posixpath.dirname(docname))}" alt="" '
                f'width="{cards.COVER_W}" height="{cards.COVER_H}"></a>'
            )
            latest = (
                f'<div class="release-latest blog-latest">{cover}<div class="release-tldr">'
                f'<h2><a href="{href}">{title}</a></h2>'
                f'<div class="blog-when"><p class="blog-date">{day}</p>{tags}</div>'
                f'<p>{description}</p><div class="release-actions"><p>{read}</p></div></div></div>'
            )
            continue
        image = _thumb_file(app, name)
        width, height = cards.THUMB_W, cards.THUMB_W * cards.COVER_H // cards.COVER_W
        cover = (
            f'{link}<img class="dark-light" '
            f'src="{posixpath.relpath(image, posixpath.dirname(docname))}" alt="" '
            f'width="{width}" height="{height}" loading="lazy" decoding="async"></a>'
        )
        rows.append(
            f'<div class="release-row blog-row">{cover}<div class="release-body">'
            f'<h3><a href="{href}">{title}</a></h3><p>{description}</p>{tags}</div>'
            f'<div class="release-meta"><p>{day}</p><ul><li>{read}</li></ul></div></div>'
        )
    earlier = ""
    if rows:
        button = (
            '<button type="button" class="blog-order" aria-controls="blog-earlier" '
            'data-order="newest" hidden><span>Newest first</span></button>'
        )
        earlier = (
            f'<section><div class="blog-earlier-head"><h2>Earlier posts</h2>{button}</div>'
            f'<div id="blog-earlier">{"".join(rows)}</div></section>{BLOG_ORDER_SCRIPT}'
        )
    for place in places:
        place.parent.replace(place, nodes.raw("", latest + earlier, format="html"))


# ---- the examples and demos galleries ------------------------------------------------------------
# Each demo's page (docs/demos/), and, from the summary, the generated file that page shows at full
# width and the gallery card's thumbnail.
INTERACTIVE = {
    "docs/demos/compare": (
        "compare.html",
        lambda summary: compare.compare_page(heading=False),
        lambda summary: compare.compare_thumbnail(),
    ),
    "docs/demos/pc-replay": (
        "pc.html",
        lambda summary: build_assets.pc_page(heading=False),
        lambda summary: build_assets.pc_thumbnail(),
    ),
    "docs/demos/meek": (
        "meek.html",
        lambda summary: build_assets.meek_replay(heading=False),
        lambda summary: build_assets.meek_thumbnail(),
    ),
    "docs/demos/d-separation": (
        "d-separation.html",
        lambda summary: build_assets.dsep_document(heading=False),
        lambda summary: build_assets.dsep_thumbnail(),
    ),
    "docs/demos/benchmark-explorer": (
        "benchmarks.html",
        lambda summary: build_assets.benchmarks_page(summary, links="../../docs/", heading=False),
        build_assets.benchmarks_thumbnail,
    ),
}
THUMBS = GENERATED / "thumbs"
# The data behind a notebook whose run shows no figure: the gallery draws its true graph instead.
NOTEBOOK_DATA = {
    # The first size and seed the speed notebook times.
    "speed_on_your_machine": lambda: benchmarks.scm("linear_gauss_er", 10).sample(n=100, seed=7),
}


def _write_examples(app) -> None:  # noqa: ANN001 -- Sphinx passes its application object
    """Write each interactive example's page and thumbnail, and the script that hosts the pages."""
    summary = _load_summary(app)
    for docname, (filename, page, thumbnail) in INTERACTIVE.items():
        _write(GENERATED / "static" / "examples" / filename, page(summary))
        _write(THUMBS / (docname.rsplit("/", 1)[1] + ".svg"), thumbnail(summary))
    _write(GENERATED / "static" / "example-frame.js", build_assets.HOST_SCRIPT.strip() + "\n")
    # The live demo's card, once its Space is public.
    card = ""
    if site.DEMO_URL:
        card = (
            "## Live demo\n\n::::{grid} 1 2 2 3\n:gutter: 3\n\n"
            f":::{{grid-item-card}} Andrey on Hugging Face\n:link: {site.DEMO_URL}\n"
            ":img-top: /_generated/thumbs/compare.svg\n\n"
            "Run the methods on your own CSV.\n\n{bdg-secondary-line}`In your browser`\n:::\n::::\n"
        )
    _write(GENERATED / "examples" / "live-demo.md", card)


def _gallery_last(app, env, docnames) -> None:  # noqa: ANN001
    """Read the gallery after the notebooks, so their thumbnails exist when its cards need them."""
    if "docs/examples/index" in docnames:
        docnames.remove("docs/examples/index")
        docnames.append("docs/examples/index")


def _notebook_thumbnail(app, doctree) -> None:  # noqa: ANN001
    """A notebook's gallery thumbnail: its first figure, or else its data's true graph."""
    docname = app.env.docname
    # Sphinx 7 returns a str here, later versions a path.
    is_notebook = pathlib.Path(app.env.doc2path(docname)).suffix == ".ipynb"
    if not docname.startswith("docs/examples/") or not is_notebook:
        return
    stem = docname.rsplit("/", 1)[1]
    for node in doctree.findall():
        uri = node.get("uri") if node.tagname == "image" else None
        if uri and not uri.startswith(("http:", "https:", "data:")):
            path = pathlib.Path(uri)
            path = path if path.is_absolute() else pathlib.Path(app.srcdir) / docname / ".." / path
            if path.is_file():
                _write_bytes(THUMBS / f"{stem}{path.suffix}", path.read_bytes())
                return
        if node.tagname == "raw" and "<svg" in node.astext():
            svg = re.search(r"<svg.*?</svg>", node.astext(), re.S)[0]
            _write(THUMBS / f"{stem}.svg", svg)
            return
    if stem not in NOTEBOOK_DATA:
        raise ValueError(
            f"{docname} shows no figure; name its data in NOTEBOOK_DATA for a thumbnail"
        )
    _write(THUMBS / f"{stem}.svg", build_assets.graph_thumbnail(NOTEBOOK_DATA[stem]().graph))


def _write_bytes(path: pathlib.Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _hide_private_signature(app, what, name, obj, options, signature, return_annotation):  # noqa: ANN001, ANN202, ARG001 -- autodoc event signature
    """Drop a class signature made only of private fields: the package builds those objects."""
    if what != "class":
        return None
    try:
        params = [p for p in inspect.signature(obj).parameters.values() if p.name != "self"]
    except (TypeError, ValueError):
        return None
    if params and all(p.name.startswith("_") for p in params):
        return "", return_annotation
    return None


def setup(app) -> None:  # noqa: ANN001 -- Sphinx extension entry point
    """Generate the palette, figures, and example pages, and give each page its own files."""
    # The benchmark summary behind every number: `-D andrey_summary=<path>` with
    # `-D andrey_placeholder=1` previews a summary outside benchmarks/published/ as not for release.
    app.add_config_value("andrey_summary", str(site.SUMMARY), "env")
    app.add_config_value("andrey_placeholder", False, "env", bool)
    app.connect("config-inited", _site_content)
    app.connect("builder-inited", _write_tokens)
    app.connect("builder-inited", _copy_notebooks)
    app.connect("builder-inited", _write_diagrams)
    app.connect("builder-inited", _write_examples)
    app.connect("env-before-read-docs", _gallery_last)
    app.connect("doctree-read", _notebook_thumbnail)
    app.connect("html-page-context", _page_assets)
    app.connect("html-page-context", _docs_sidebar)
    app.connect("build-finished", _write_discovery)
    app.connect("build-finished", _version_assets)
    app.connect("doctree-resolved", _post_byline)
    app.add_directive("blog-index", _BlogIndex)
    app.connect("doctree-resolved", _blog_index)
    # Before sphinx_autodoc_typehints, which returns a signature of its own.
    app.connect("autodoc-process-signature", _hide_private_signature, priority=100)
