---
name: development
description: Build, preview, and validate the website - the homepage, the blog, and the documentation.
meta:
  type: code
---

# Developing the website

`andrey.abel.ai` is one Sphinx project, this folder: the homepage at `/`, the blog at `/blog/`, and
the documentation under `/docs/`, whose sources are in [`docs/`](docs/index.md). [`conf.py`](conf.py)
configures MyST for Markdown, autodoc for `andrey.__all__`, cached MyST-NB notebooks, ABlog, and
pydata-sphinx-theme. Every page shares one navbar: the wordmark to the homepage; Docs, Examples,
Demos, Blog, Release cards, and FAQ; then search, the theme switch, and GitHub. The method API and
the spec registry are in the root [`DEVELOPMENT.md`](../DEVELOPMENT.md); the palette, the art, and
the brand are in [`DESIGN.md`](../DESIGN.md).

## Build

The unpublished `docs` dependency group (PEP 735) keeps site dependencies out of the wheel.

```shell
uv sync --group docs
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

Every link is relative, so one build serves any hostname.

After the build, `_version_assets` in `conf.py` adds each local image's content hash to its
address (`?v=<hash>`) in the pages and stylesheets, and gives each stylesheet link the hash of
the rewritten stylesheet, so a deploy never shows an image Cloudflare or a browser cached from an
earlier build. Download links keep their plain addresses.

## The homepage

The homepage is a template, [`site/homepage.html`](../site/homepage.html), on pydata's layout with
only its navbar. `conf.py` fills it with `homepage_context` from [`site/build.py`](../site/build.py):
the hero's art, the sections' claims and tiles, the charts, and the comparison table with its
sources. `site/assets/site.css` styles it, scoped to `.andrey-home` so the theme's styles and the
page's never meet. `site/assets/figures.css` and `figures.js` draw the timing comparison and the
speed charts, which the posts, the release cards, and the benchmark explorer share.

The hero is one screen: the headline, "Causal discovery: accelerated, accurate, agentic.", the
install line, and a card per section. A pill above the headline copies the prompt that installs the
agent skill (`AGENT_PROMPT`, `site/assets/prompt.js`), the one the README and the FAQ give, beside
the marks of coding agents and open models (`AGENTS`, Lobe Icons in `site/agents/`, MIT). The
sections follow the headline's words, then All-in-one (`ACTS` in `site/build.py`). Each opens on its
art at full bleed with its title and its claim (`act_covers()`), then shows its evidence: tiles,
each with a figure of its own, and a note in plain type where the evidence is a list (`bento-note`).
The hero and Accelerated share the fast lanes (`HERO_ART`, `.run` in `site.css`), which stay on the
screen while Accelerated's title comes up over them, until the tiles rise over the art.

- Accelerated: the ideas behind the speed as a carousel, one card playing at a time, then a tile per
  method family.
- Accurate: where Andrey has fewer errors, as a chart, and beside it a note: every release is
  tested against true graphs, and its release card gives every method's errors.
- Agentic: an agent's run of the command line, and the contract it reads first.
- All-in-one: the methods by family, the comparison table's features per package, and every kind
  of causal graph drawn; then "From data to graph" on its own art (`example_cover()`,
  `EXAMPLE_ART`): the Sachs example's code, output, and graph.

Each tile that leads on opens its details as a dialog (`tiles.js`) and never leaves the page;
without script the details follow the grid and the tile jumps to them. No two tiles open the same
dialog (Agentic's one tile shows the whole session, `agent_tile()`), and a tile whose dialog would
only repeat it has none and carries no plus. A tile per registry family that has a supported method
with a published comparison (`family_tiles()`) shows the family's largest ratio rounded down to one
digit ("up to 900x faster"), and none where Andrey is not at least twice as fast; a new family gets
its tile with no change to the page.

The ideas are four animated scenes from [`site/why_fast.py`](../site/why_fast.py), which the post
`docs/blog/20261008-why-andrey-is-fast.md` shows too; each compares Andrey with a lane labeled
Others and names no package. Under Accelerated's title they are a carousel (`belt()`): a card per
scene, with its title, narration, a link to its part of the post, and its stage, in a row that
scrolls sideways across the screen and snaps each card to the center. `why-fast.js` plays the
current card while its dot fills, rests on its last frame, and moves on to the next, looping; it
waits while the pointer or focus is on the carousel or it is off the screen, and the dots, arrow
keys, a swipe, or a drag pick a card. Without script the cards are a row to scroll by hand, and
under reduced motion no card plays or moves on by itself; both show the final counts. The scenes
play with Motion, vendored unmodified as `site/assets/motion.js` under its MIT license and pinned
in `tests/unit/test_why_fast.py`; `conf.py` loads both scripts only on those two pages.

The page moves with the scroll, never under reduced motion: the stylesheet drives it from scroll and
view timelines where the browser has them, and only opacity and transforms move; elsewhere the page
is still but for the tiles, which `tiles.js` brings in. Every cover's art has one amber line
(`SWEPT`), and a short trace of light runs along it while the art is on the screen (`tiles.js`):
`streak()` in `site/build.py` traces the line from the art's warm pixels, and the trace is drawn in
the art's own frame (`.cover-frame`), so it stays on the line at every width. The hero's art keeps
nearly its own size (86% of the screen's width or 60% of its height, whichever is more); a section's
cover band grows with the screen's width, to 90% of its height, so a wide screen shows its art's
full height, and there its tiles rise into the band's fading foot. The documentation section opens
on its art (`docs_cover()`, `DOCS_ART`) and leads with the newest post (`_newest_post_card` in
`conf.py`).

The art follows [`DESIGN.md`](../DESIGN.md): the files are in `site/brand/` (`HERO_ART`, `ACTS`,
`EXAMPLE_ART`, `FAMILY_ART`, and `tile_art()` in `site/build.py`), stored after `brand_tones` as
sRGB with no embedded profile, and site tests check their tones, their sizes, and that their median
color is `--andrey-ground-art` (`cards.ground_tokens()`), the ground the CSS lays around them.

The navbar on every page is a frosted glass pill (`docs/_static/andrey.css`), as is the search
dialog. In Chromium the pill is Liquid Glass: `glass.js` draws a lens map for the pill's size into
the SVG filter in `_templates/site-logo.html` and sets `data-glass`; other browsers keep the frosted
pill, and reduced transparency gets the solid paper. A dot on Release cards marks a new release
for a week: `_templates/site-nav.html` gives it the newest release's version and its post's date
(`_release_post` in `conf.py`), and `nav-new.js` shows it within seven days of that date until the
reader opens a release card, a fact it keeps in the browser's local storage, which no request
carries; a newer release shows it again.

The homepage takes every number from the release summaries `RELEASES` names
(`benchmarks/published/<label>/summary.json`) or from the registry (`andrey.spec.list_specs()`),
counting supported methods only. `comparisons()` reads every kind of summary into one layout, each
comparison's errors in the measure its summary records (`ERRORS`). The README, the live demo, and
the homepage's description state `CLAIM`, a speed claim with no number. The build fails unless the
first summary's headline is `claim_sentence()` of its numbers: the largest ratio rounded down to a
power of ten on its method, then whether the other supported methods are faster ("the other", "most
other", or nothing). The README chart shows one card per `PANEL` method, at most six.

The homepage's description names PC, GES, and FCI, counts the other methods, and states `CLAIM`.
Every other page's description is the `description` in its front matter; without one, its first
paragraph cut at a word to 160 characters (an API page's docstring summary line); a page with
neither, such as search, takes the homepage's. To preview with placeholder numbers, which replace
the first release's summary:

```shell
uv run --group docs sphinx-build -b html docs docs/_build/html \
  -D andrey_summary=tests/unit/fixtures/summary.placeholder.json -D andrey_placeholder=1
```

The page is then marked "not for release" and `noindex`; the fixture carries
`"placeholder": true`, which the build refuses without `andrey_placeholder`. The examples on the page
are [`site/example.py`](../site/example.py) and [`site/interface.py`](../site/interface.py), run at
build time: the page shows what they print and draws the graphs they return.

## The blog

ABlog builds the blog from [`blog/`](blog/index.md): the posts, an archive, author pages, and an
Atom feed at `/blog/atom.xml`. A post is a MyST page whose front matter sets `blogpost: true`, its
`author`, its `date`, and its `description`. A post without a date is a draft, which the blog
leaves out; a dated post is published in every build, even before its date.

The blog's front page lists the dated posts from their front matter (the `blog-index` directive in
`blog/index.md`, filled by `_blog_index` in `conf.py`): the newest post with its cover, title,
date, tags, and description, then the earlier ones, newest first, each with a small copy of its art
(`thumb_jpeg` in `site/cards.py`). Newest means the latest date, a later version breaking a tie. A
new post needs no change to the page.

A post's cover is its art alone; an art with a centered version (`CENTERED` in `site/cards.py`)
shows the version on its cover and thumbnail. Its link preview is the banner, the art with the
wordmark, title, and release over its left side (`preview_jpeg`). A post belongs to the release in
its `version` front matter, and later builds leave it as that release rendered it: its numbers are
typed text, and every file it includes is stored under `blog/<version>/`.

## The release cards

Each release has a release card, `docs/benchmarks/<release>.md`: its benchmarks, the methods it adds
or promotes, their known limits, and how to rerun the measurements, from the summary it links,
`benchmarks/published/<label>/summary.json`. Like a post, a released card and its summary never
change; new measurements go into the next release's card. A card's figures are finished HTML files
in `docs/benchmarks/<release>/`, included with `{raw} html`, so later code never redraws them.

[`docs/benchmarks.md`](docs/benchmarks.md), the Release cards page, leads with the latest release as
a cover beside its TL;DR (the abstract, three numbers, and a button to the card), then lists the
earlier releases, newest first. To add a release:

1. Move the current latest release to the top of "Earlier releases" (a copied row with its
   version, title, methods sentence, facts, release date, and links).
2. Fill the latest block with the new release, and link the post after the card's button once the
   post is out. The published site is built from release tags, so no reader sees a card early.
3. Add the new card to the toctree, newest first.
4. Store the cover's art as `_static/release-cards/<release>.jpg`, an 864 x 1119 JPEG made with
   `release_art` in `site/cards.py`, and add a `release-art-<release>` rule (dots as dashes) to
   `_static/benchmarks.css`. The card's cover and the Release cards page's featured cover take both
   classes, `release-art release-art-<release>`; the release's tile keeps the second when it moves
   to the earlier releases. A cover without art shows the stipple ground (`release_ground_jpeg`).
5. Open the card with the same cover, without its link, beside the opening paragraphs in a
   `release-opening` block.
6. Add the summary's folder to `RELEASES` in `site/build.py`, so the homepage and the README count
   it.

## The documentation sidebars

The left sidebar ([`_templates/docs-nav.html`](_templates/docs-nav.html)) groups the docs index's
toctree: each section in `NAV_SECTIONS` heads a group under its icon
([`_templates/nav-icons.html`](_templates/nav-icons.html)), folding pages past `NAV_SHOWN`, and the
other entries make the Reference group; `_docs_sidebar` in `conf.py` builds it from the toctrees.
The right sidebar holds the page's outline ([`_templates/page-toc.html`](_templates/page-toc.html),
marked by [`_static/page-outline.js`](_static/page-outline.js)), then links to edit the page and
report a problem ([`_templates/page-links.html`](_templates/page-links.html)); Edit this page opens
the file that holds the page's text, an included Markdown file or notebook where there is one. The
page's section sits over its title ([`_templates/page-kicker.html`](_templates/page-kicker.html)).
On a phone the outline moves into the menu
([`_templates/page-outline.html`](_templates/page-outline.html)).

## The Examples and Demos galleries

[`docs/examples/index.md`](docs/examples/index.md) is a gallery of notebook cards, and
[`docs/demos/index.md`](docs/demos/index.md) of demos that run in the browser, plus a live demo
whose card appears once `DEMO_URL` in `site/build.py` names a public Space. Each demo is a page with
a title, a paragraph, and the demo in a frame, generated into `_static/examples/` at build time: the
comparison on one dataset from [`apps/compare.py`](../apps/compare.py), and Meek's rules, the
d-separation explorer, and the benchmark explorer from
[`apps/build_assets.py`](../apps/build_assets.py). The comparison reads causal-learn's graphs from
`apps/compare-causal-learn.json`, recorded once (see [`apps/README.md`](../apps/README.md)); the
build refuses a recording made on different data.

Every choice on the site is one segmented control, `site/assets/segmented.css`: a track whose
chosen option fills with the accent. The documentation pages load it with the shared stylesheets,
and the demos and the benchmark explorer, which are standalone pages, inline the same file. It
styles the release cards' and posts' figure tabs (`race-sizes`), the comparison's method switch,
the PC replay's examples, the d-separation explorer's conditioning set, and the graph guide's
edges.

Thumbnails come from each page's own output: a demo draws its own; a notebook's is its first figure,
or its dataset's true graph (`NOTEBOOK_DATA` in `conf.py`). They are drawn in the light palette in
both themes.

## Notebooks

Canonical notebooks live in `examples/`. The build stages them in `docs/examples/` and executes
them with MyST-NB's cache, so a failed cell fails the build; the speed notebook is not executed.
Run them locally with `uv run --with jupyter jupyter lab`. Notebook dependencies stay out of
`pyproject.toml`: each notebook has an Open in Colab link and a Colab-only setup cell that installs
the pinned release from PyPI, a pin to update with every release. The rendered page's download link
points at the canonical notebook.

## Preview locally

```shell
python -m http.server 8123 --bind 127.0.0.1 --directory docs/_build/html
```

On a remote machine, keep the server on `127.0.0.1` and forward the port:
`ssh -N -L 8123:localhost:8123 <user>@<host>`, then open <http://localhost:8123>.

## Search and answer engines

[`discovery.py`](discovery.py) writes what search and AI answer engines read. Each page's head
carries its Twitter card fields and one JSON-LD graph: the organization (Abel's logomark,
`site/brand/abel-symbol-night.svg`, drawn as `_static/abel-symbol.png`), the website, the software
(Andrey's icon) at the page's release, the page, a `BlogPosting` for a post, and the breadcrumbs.
The search and 404 pages are `noindex` (`UNINDEXED`). After the build, `_write_discovery` in
`conf.py` writes `llms.txt` (the pages in `LLMS_SECTIONS`, then every release card and post) and
`sitemap.xml` (every indexable page); `robots.txt` is a plain file (`_extra/robots.txt`). A page in
`LLMS_SECTIONS` that is renamed or removed fails the build until the list follows it.

## Validation

CI runs the site's unit tests (`tests/unit/test_site.py`, `test_launch_material.py`,
`test_examples_gallery.py`, `test_why_fast.py`, `test_compare_example.py`, and `test_discovery.py`),
then [`test_built_site.py`](test_built_site.py), which builds the site once from a clean workspace:

```shell
uv run --group docs python -m pytest docs/test_built_site.py
```

It checks a strict `-W` build including notebook execution; API coverage (every public symbol of
`andrey`, `andrey.metrics`, and `andrey.viz`, and every loader of `andrey.data`, documented, no
private symbol exposed); KaTeX-ready math; one description per page, the same as its link preview's,
at most 160 characters; the navbar's links; the favicon and the link-preview cards; the blog's front
page and each post's cover and byline; every site address in the README, the project URLs, and the
citation; each page's structured data against its link preview and each post's front matter;
`llms.txt`, `sitemap.xml`, and `robots.txt`; that only the search and 404 pages are `noindex`; and
that the 404 page's links resolve from the site's root. External links: `uv run --group docs
sphinx-build -b linkcheck docs docs/_build/linkcheck`.

## How the site is wired

| Piece | Where |
|---|---|
| Sphinx config (theme, extensions, notebooks, blog, generated pages) | `docs/conf.py` |
| Homepage template, stylesheet, and content generators | `site/homepage.html`, `site/assets/site.css`, `site/build.py` |
| Timing comparison and speed charts | `site/assets/figures.css`, `site/assets/figures.js` |
| Segmented control: every choice, in the figures, the demos, and the graph guide | `site/assets/segmented.css` |
| "Why Andrey is fast" scenes: homepage carousel, blog post | `site/why_fast.py`, `site/assets/why-fast.css`, `site/assets/why-fast.js`, `site/assets/motion.js` |
| Navbar: wordmark and sections | `docs/_templates/site-logo.html`, `docs/_templates/site-nav.html` |
| Documentation footer: a navy card, three columns of links, the license, Abel's mark | `docs/_templates/docs-footer.html`, `docs/_static/andrey.css` |
| Brand kit page: the downloads (`brand_kit` and `social_svg` in `site/cards.py`, written to `_static/brand/`) and the colors (`BRAND_COLORS` in `conf.py`) | `docs/docs/brand.md`, `docs/_static/andrey.css` |
| Wordmark (Unbounded outlines in light or dark ink, the A's right stroke amber), the icon (that A), and their generator | `site/brand/`, `site/wordmark.py` |
| Palette as CSS custom properties (`--andrey-*`, from `andrey.viz`) | `docs/palette_css.py` |
| Blog posts and the front page that lists them | `docs/blog/`, `docs/_static/blog.css` |
| Release cards and the page that lists them | `docs/docs/benchmarks/`, `docs/docs/benchmarks.md`, `docs/_static/benchmarks.css` |
| Docs landing page and section index | `docs/docs/index.md` |
| API reference (autodoc from `andrey.__all__`, the Typer CLI) | `docs/docs/code/index.md` |
| Guides | `docs/docs/guides/` |
| Examples and demos galleries and pages | `docs/docs/examples/`, `docs/docs/demos/` |
| Notebook sources | `examples/*.ipynb` -> staged into `docs/docs/examples/` |
| Docs chrome over the palette tokens | `docs/_static/andrey.css` |
| Docs tables: a framed header band over a card of rows, a fade where a wide table scrolls, short values and labels on one line (`_short_cells` in `conf.py`); the API reference's summary tables get a Name and Description header (`_summary_headers`) and stack on a phone | `docs/_static/andrey.css`, `docs/conf.py` |
| The old demo address, redirected to the gallery | `docs/_templates/demo-redirect.html` |
