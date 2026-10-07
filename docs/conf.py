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

import ablog.blog
from ablog.blog import Blog
from docutils import nodes
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
# Build-time output that is not a page: blog figures, example pages, thumbnails, the preview card.
GENERATED = HERE / "_generated"


def _module(name: str, path: pathlib.Path):  # noqa: ANN202 -- a module loaded from a file
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The homepage's content generators, and the pages of the interactive examples.
site = _module("andrey_site", ROOT / "site" / "build.py")
why_fast = _module("andrey_why_fast", ROOT / "site" / "why_fast.py")
cards = _module("andrey_cards", ROOT / "site" / "cards.py")
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "live-discovery"))
build_assets = _module("andrey_build_assets", ROOT / "apps" / "build_assets.py")
compare = _module("andrey_compare", ROOT / "apps" / "compare.py")

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
# The wordmark's "A" in the accent blue: a placeholder mark until the brand has one.
html_favicon = "../site/brand/favicon.svg"
# Served from the site root as they are: the link-preview card.
html_extra_path = ["_generated/extra"]
html_css_files = ["andrey-tokens.css", "andrey.css", "graph-guides.css"]
REPO_URL = "https://github.com/Abel-ai-lab/andrey"
SITE_URL = "https://andrey.abel.ai/"
html_theme_options = {
    # One header on every page (_templates/site-*.html): the wordmark to the homepage and the
    # release to the changelog, then Docs, Examples, Blog, and FAQ, then search, theme, and GitHub.
    "navbar_start": ["site-logo", "site-version"],
    "navbar_center": ["site-nav"],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_prev_next": False,
    "icon_links": [{"name": "GitHub", "url": REPO_URL, "icon": "fa-brands fa-github"}],
    "footer_center": ["report-bug"],
    "use_edit_page_button": False,
    # The alpha status, once, above every page's header (_templates/sections/announcement.html).
    "announcement": "Andrey is in alpha: the API may change between releases.",
}
# The theme follows the reader's system setting until they pick one; unset, its script logs
# "invalid theme mode" on a first visit.
html_context = {"default_mode": "auto", "report_bug_url": f"{REPO_URL}/issues/new"}
# A post's author and date sit under its title (_post_byline); the blog index links the archive.
BLOG_SIDEBAR = ["ablog/archives.html"]
html_sidebars = {
    "index": [],
    # The whole documentation tree, since the navbar names only the site's sections.
    "docs/**": ["docs-nav.html"],
    "blog": BLOG_SIDEBAR,
    # A post takes the full width beside its contents, so its charts have room.
    "blog/**": [],
}
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
        + "".join(f"  {name}: {value};\n" for name, value in cards.grain.ground_tokens().items())
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


def _release_text() -> dict[str, str]:
    """The release and its method tiers, for the launch post and the FAQ."""
    supported = [f"`{s.name}`" for s in list_specs() if s.status == "supported"]
    return {
        "release": release,
        "n_supported": str(len(supported)),
        "n_experimental": str(len(list_specs()) - len(supported)),
        "supported_methods": ", ".join(supported[:-1]) + f", and {supported[-1]}",
    }


def _site_content(app, config) -> None:  # noqa: ANN001 -- Sphinx passes its application objects
    """Fill the homepage's context, and the launch post's numbers and figures, from the summary."""
    (GENERATED / "static").mkdir(parents=True, exist_ok=True)
    summary = _load_summary(app)
    app.andrey_home = site.homepage_context(summary, placeholder=config.andrey_placeholder)
    # The launch post prints what its first fit prints, from the homepage's run of the example.
    printed = html.unescape(app.andrey_home["example_output"])
    _write(GENERATED / "blog" / "example-output.txt", printed)
    text, figures = site.launch_post_context(summary)
    config.myst_substitutions = {**config.myst_substitutions, **text, **_release_text()}
    for name, figure in figures.items():
        _write(GENERATED / "blog" / f"{name}.html", figure + "\n")
    # The "Why Andrey is fast" scenes, shown under the post's headings.
    for scene in why_fast.SCENES:
        _write(
            GENERATED / "why-fast" / f"{scene['key']}.html", why_fast.fragment(scene["key"]) + "\n"
        )
    # The link-preview card shared by the homepage and every page that is not a post; its subtitle
    # is the headline speed claim, which a post can replace with a `card_subtitle` of its own.
    site.fastest(summary)
    app.andrey_card_subtitle = cards.subtitle(site.speed_claim(summary)[0])
    home_card = cards.card_jpeg(cards.TAGLINE, app.andrey_card_subtitle)
    _write_bytes(GENERATED / "extra" / HOME_PREVIEW, home_card)
    # The homepage hero draws the cards' stipple from CSS gradients and these repeating tiles.
    for name, tile in cards.grain.hero_tiles():
        _write_bytes(GENERATED / "static" / name, tile)


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
    """The page's description, and its Open Graph and Twitter card tags: title, summary, address,
    and image. The description and ``og:description`` are one value."""
    home = context.get("home")
    heading = re.sub(r"<[^>]+>", "", context.get("title") or app.config.project)
    title = home["title"] if home else html.unescape(heading)
    description = _description(app, context, doctree)
    url = SITE_URL + ("" if pagename == "index" else f"{pagename}.html")
    image = HOME_PREVIEW
    if pagename in Blog(app):  # each post's card is drawn from its title, once per build
        image = f"assets/preview/{pagename.removeprefix('blog/')}.jpg"
        target = pathlib.Path(app.outdir) / image
        target.parent.mkdir(parents=True, exist_ok=True)
        subtitle = (context.get("meta") or {}).get("card_subtitle", app.andrey_card_subtitle)
        target.write_bytes(cards.card_jpeg(title, subtitle))
    image = SITE_URL + image
    tags = {
        "description": description,
        "og:type": "article" if pagename in Blog(app) else "website",
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
    return "".join(
        f'<meta {"property" if key.startswith("og:") else "name"}="{key}" '
        f'content="{html.escape(value, quote=True)}">\n'
        for key, value in tags.items()
    )


# The pages that show the "Why Andrey is fast" scenes.
WHY_FAST_PAGES = {"index", "blog/why-andrey-is-fast"}


def _page_assets(app, pagename, templatename, context, doctree) -> None:  # noqa: ANN001
    """Give each page the context and files it needs beyond the shared ones.

    Page styles load after the shared ones (priority 900), so they win a tie.
    """
    css, js = [], []
    if pagename == "index":
        context["home"] = app.andrey_home
        css, js = ["figures.css", "site.css"], ["figures.js"]
    elif pagename == "blog" or pagename.startswith("blog/"):
        css, js = ["figures.css", "blog.css"], ["figures.js"]
    elif pagename in INTERACTIVE:
        css, js = ["examples.css"], ["example-frame.js"]
    elif pagename in ("docs/examples/index", "docs/demos/index"):
        css = ["examples.css"]
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


def _post_byline(app, doctree, docname) -> None:  # noqa: ANN001
    """Put a post's cover and then its authors, date, and the Andrey version it was written for
    under its title, where a reader looks for them and where they stay on a narrow screen. A post is
    not updated after it is published, so the version dates what it says. ABlog has read the posts'
    front matter by now."""
    blog = Blog(app)
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
    byline = f'<p class="post-byline">{authors}{" · " if authors else ""}{day}{written_for}</p>'
    at = title.parent.index(title)
    title.parent.insert(at + 1, nodes.raw("", _post_cover(app, docname), format="html"))
    title.parent.insert(at + 2, nodes.raw("", byline, format="html"))


def _post_cover(app, docname: str) -> str:  # noqa: ANN001
    """Draw the post's cover (site/cards.py) and return its figure. The front matter's ``cover``
    names it (``pair`` or ``graph``, the default); ``cover_caption`` adds a caption."""
    meta = app.env.metadata.get(docname, {})
    kind = meta.get("cover", "graph")
    image = f"assets/covers/{docname.removeprefix('blog/')}.jpg"
    target = pathlib.Path(app.outdir) / image
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(cards.cover_jpeg(kind))
    caption = meta.get("cover_caption")
    src = posixpath.relpath(image, posixpath.dirname(docname))
    return (
        f'<figure class="post-cover"><img class="dark-light" src="{src}" '
        f'alt="{html.escape(cards.cover_alt(kind), quote=True)}" '
        f'width="{cards.COVER_W}" height="{cards.COVER_H}">'
        + (f"<figcaption>{html.escape(caption)}</figcaption>" if caption else "")
        + "</figure>"
    )


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
    app.connect("doctree-resolved", _post_byline)
    # Before sphinx_autodoc_typehints, which returns a signature of its own.
    app.connect("autodoc-process-signature", _hide_private_signature, priority=100)
