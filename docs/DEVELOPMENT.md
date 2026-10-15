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
pydata-sphinx-theme. Every page shares one navbar: the wordmark to the homepage, then Docs,
Examples, and Blog, then search, the theme switch, and GitHub. For the method API and spec
registry, see the root [`DEVELOPMENT.md`](../DEVELOPMENT.md).

## Build

The unpublished `docs` dependency group (PEP 735) keeps site dependencies out of the wheel.

```shell
uv sync --group docs
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

Every link is relative, so one build serves any hostname.

## The homepage

The homepage is a template, [`site/homepage.html`](../site/homepage.html), on pydata's layout with
only its navbar. `conf.py` fills it with `homepage_context` from [`site/build.py`](../site/build.py):
the hero's art, the sections' claims and tiles, the charts, and the comparison table with its
sources.
`site/assets/site.css` styles it, scoped to `.andrey-home` so the theme's styles and the page's do
not meet; `site/assets/figures.css` and `figures.js` hold the timing comparison and the speed
charts, which the launch post and the benchmark explorer show too.

The hero is one screen: the headline, "Causal discovery: accelerated, accurate, agentic.", the
install line, and a card per section below it, each with its art, its title, and its claim's
short form. The four sections follow the headline's words, then All-in-one for what the package
covers (`ACTS` in `site/build.py`). Each opens on its art at full bleed with its title and its
claim (`act_covers()`), then shows its evidence, two pieces or more: tiles, each with a figure of
its own, and a note in plain type beside a tile where the evidence is a list a card would only
frame (`bento-note`). The hero and Accelerated share one art, the fast lanes (`HERO_ART`, `.run` in
`site/assets/site.css`): it stays on the screen while the hero's type scrolls away and
Accelerated's title comes up over it right after the hero's cards, with the ideas behind the
speed, until the tiles rise over it on the page's paper. The run clips the art at its foot, so the
next section opens on its own art alone.

- Accelerated: the ideas behind the speed, one at a time as the reader scrolls, then a tile per
  method family.
- Accurate: where Andrey has fewer errors, as a chart, and beside it a note: every release is
  tested against true graphs, and its release card gives every method's errors.
- Agentic: an agent's run of the command line, and the contract it reads first.
- All-in-one: a note with the methods by family, how many of the comparison table's features each
  package has, and every kind of causal graph drawn, then "From data to graph", which opens on its
  own art (`example_cover()`, `EXAMPLE_ART`): the Sachs example's code, output, and graph.

Every claim comes from the summaries and the registry (`act_claims()`): the largest family speedup,
the largest cut in errors (`fewest_errors()`, the accuracy chart's first row; the homepage shows
only the wins, and the release cards every method's errors), and the number of methods. A tile opens
its dialog, and no two tiles open the same one (Agentic's one tile shows the whole command-line
session, `agent_tile()`); a tile whose dialog would only repeat it has none, and carries no plus.

The ideas are four animated scenes built by [`site/why_fast.py`](../site/why_fast.py) (`story()`),
each with its title and narration. Without script, and with reduced motion, they stand one after
another at their final counts. Otherwise `why-fast.js` pins Accelerated's title and the story
under the navbar (`.wf-run`, `.wf-pin`) and splits the run's scroll among the scenes: the current
one shows, the scroll plays it forward or back, and a bar per scene fills as it plays and jumps to
it. Where the title and a scene do not fit the screen together, the scenes stay stacked and each
plays as it rises through the screen. The blog post `docs/blog/20261008-why-andrey-is-fast.md`
shows the same scenes under its own headings. Each scene
compares Andrey with a lane labeled Others and names no package. `site/assets/why-fast.js` plays
them with Motion, vendored unmodified as `site/assets/motion.js` under its MIT license; the version
is pinned in the file's banner and in `tests/unit/test_why_fast.py`. `conf.py` loads both scripts
only on those two pages.

The hero, the covers, and the tiles look like the blog's covers, in both themes: the stipple ground,
art in the brand's tones with its subject on the right (`HERO_ART`, `ACTS`, `EXAMPLE_ART`,
`FAMILY_ART`, and `tile_art()` in `site/build.py`; the files in `site/brand/`), a navy fade under
the light type, and amber on one element only: the headline's words, the number in a tile, or else
the art's own key element, as on each section's cover. New art comes from the cover process and is
stored after `brand_tones`, as sRGB with no embedded profile; site tests check its tones and size,
and that its median color is `--andrey-ground-art` (`cards.ground_tokens()`), the ground the CSS
lays around it.

The hero's art keeps nearly its own size, 86% of the screen's width or 60% of its height,
whichever is more, standing on the screen's foot and held to its right edge, rather than filling
the screen's height. A section's cover band grows with the screen's width, to 90% of its height,
so a wide screen shows its art's full height.

The page moves with the scroll, and never under reduced motion. The hero's art pushes in slowly and
its stipple drifts; as the reader scrolls on, the art stays and the type and cards rise and fade.
Each section's art settles from a slight push to its own size as it reaches the middle of the screen
and drifts against the scroll, its title rises in, then its claim, and the tiles fly in. The
stylesheet drives all of it from the scroll position where the browser can (scroll and view
timelines); elsewhere the page is still, apart from the tiles, which `tiles.js` brings in with an
`IntersectionObserver`. Only opacity and transforms move. Every cover's art has one amber line
(`SWEPT`: the hero's lanes, Accurate's dotted trail, Agentic's arm, All-in-one's ray out of the
prism, From data to graph's arrow, the documentation's page edge), and a short trace of soft neon
light runs along it every few seconds while the art is on the screen (`tiles.js`), moving only its
dash and opacity: `streak()` in `site/build.py` traces the line from the art's warm pixels, a median
point in each band along its longer extent, from the bottom up, or, for a level beam, left to right,
so it follows a bend and a trail of dots, and the trace is drawn in the art's own frame
(`.cover-frame`), so it stays on the line at every width. The documentation section opens on its own
art (`docs_cover()`, `DOCS_ART`), then leads with the newest post, the one the blog's front page
leads with: its cover, date, title, and description (`_newest_post_card` in `conf.py`). No section
ends on a hard edge: the hero's art and the covers fade into the paper. The navbar on every page is
a frosted glass pill (`docs/_static/andrey.css`), and the search dialog is the same glass, larger;
the homepage's hero runs up under the navbar. In Chromium the pill is Liquid Glass: `glass.js` draws
a lens map for the pill's size into the SVG filter in `_templates/site-logo.html` and sets
`data-glass`; Safari and Firefox keep the frosted pill, and reduced transparency gets the solid
paper. The section links sit in a recessed glass track, and the current section is a raised glass
pill on it. A red dot on Release cards marks a release the reader has not opened yet:
`_templates/site-nav.html` gives it the newest release's version (`_release_post` in `conf.py`), and
`nav-new.js` shows it until the reader opens a release card page, and again when a newer release is
out. Without script or storage there is no dot.

Each tile that leads on is a link, and every one behaves alike: `tiles.js` opens its details as a
dialog, and none leaves the page; without script the details are the dialogs' content, shown in the
page after the grid (`homepage.html`'s noscript style), and the tile jumps to them. A tile per
registry family that has a supported method with a published comparison (`family_tiles()`) shows the
family's largest published ratio rounded down to one digit ("up to 900x faster"), and none where
Andrey is not at least twice as fast; its dialog holds the family's time charts. A new family gets
its tile with no change to the page.

The homepage takes every number from the release summaries that `RELEASES` in `site/build.py` names,
each `benchmarks/published/<label>/summary.json`, or from the method registry,
`andrey.spec.list_specs()`. Only methods the registry supports count. `comparisons()` reads every
kind of summary into one layout: rows at mean degree 2 where a summary measures more than one, the
observed variables on data with hidden variables, a cause-effect pair's rows as its size, a ratio
the summary flags as a lower bound printed as one, and each comparison's errors in the measure its
summary records (`ERRORS`: the structural Hamming distance, directed-edge errors on data with hidden
common causes, or reversed directions on cause-effect pairs). The speed charts share one size axis
per unit, variables or rows. The README's claim is the one `claim_sentence()` words from every
supported comparison: the largest ratio rounded down to a power of ten on its method, then whether
the other supported methods are faster ("the other", "most other", or nothing). The build
also fails unless the first summary's headline is the claim its own numbers support; the docs index,
the notebook, and the live demo repeat it. The README chart shows one card per method in `PANEL`,
a (release, method) pair each, with every package measured in the same runs at its largest size with
a paired ratio, and refuses more than six. The methods dialog's first sentence names every registry
family, and its list shows every registered method. The page's
description, which search results and link previews show, names PC, GES, and FCI, counts the other
registered methods, and states the claim's first part ("over 1,000x faster on PNL"). Every other
page's description is the `description` in its front matter; without one, the page's first
paragraph, cut at a word to 160 characters (an API page's is its docstring's summary line). A page
with neither, such as search or the blog archive, takes the homepage's. To preview with placeholder
numbers, which replace the first release's summary:

```shell
uv run --group docs sphinx-build -b html docs docs/_build/html \
  -D andrey_summary=tests/unit/fixtures/summary.placeholder.json -D andrey_placeholder=1
```

The page is then marked "not for release" and `noindex`. The fixture carries
`"placeholder": true`, which the build refuses without `andrey_placeholder` wherever the file
sits. The examples on the page are [`site/example.py`](../site/example.py) and
[`site/interface.py`](../site/interface.py), run at build time; the page shows what they print and
draws the graphs they return. `interface.py` runs three methods on one table about a lawn, and
VARMA-LiNGAM on the same lawn day by day, so the graphs tile draws all five kinds: VARMA-LiNGAM's
graph over two days (`two_days()`) and, with time collapsed, the cyclic graph.

## The blog

ABlog builds the blog from [`blog/`](blog/index.md): the posts, an archive, author pages, and an
Atom feed at `/blog/atom.xml`. A post is a MyST page whose front matter sets `blogpost: true`, its
`author`, its `date`, and its `description`. A post without a date is a draft, which the blog
leaves out. A dated post is published in every build, even before its date, so a future date does
not keep a post private.

The blog's front page, `/blog/`, lists the dated posts in the Release cards page's blocks, from
their front matter: the `blog-index` directive in `blog/index.md`, which `_blog_index` in
`conf.py` fills. The newest post leads with its cover across the block, then its title, date,
tags, its `description` as its TL;DR, and a button to the post. The earlier posts follow, newest
first, each in a row with a small copy of its art (`thumb_jpeg` in `site/cards.py`), its
description, tags, and date, under a button that reverses their order. Newest means the latest
date, a later version breaking a tie, as for the alpha banner below. Each tag links to ABlog's
page of the posts that share it. A new post needs no change to the page.

A post's cover is its art alone: the page's title and byline (authors, date, version, tags) sit
around it. Its link preview is the banner, the art with the wordmark, title, and release drawn
over its left side (`preview_jpeg`), the image a shared link shows.

A post belongs to the release in its `version` front matter, and later builds leave it as that
release rendered it: its numbers are typed text, and every file it includes (a figure, code, or
output) is stored under `blog/<version>/`. The navbar's dot on Release cards names the newest dated
post with a version, a later version breaking a tie in date (`_release_post` in `conf.py`).

## The release cards

Each release has a release card, `docs/benchmarks/<release>.md`: its benchmarks, the methods it adds
or promotes, their known limits, and how to rerun the measurements. The summary its numbers come
from is `benchmarks/published/<label>/summary.json`, which the card links. Like a post, a released
card and its summary never change: their numbers are typed text, and new measurements go into the
next release's card.

[`docs/benchmarks.md`](docs/benchmarks.md) is the Release cards page, at the address the README and
the homepage link. The latest release leads as a cover (masthead, title, version) beside its TL;DR:
the abstract, three numbers, a limit, and a button to the card.
Earlier releases follow as a list, newest first, each with a square tile showing its version. To
add a release:

1. Move the current latest release to the top of "Earlier releases": copy a row and set its
   version, title, methods sentence, facts, release date, and links.
2. Fill the latest block with the new release, and add a link to the post after the card's button
   once the post is out. The block states only what does not change: the published site is built
   from release tags, so a reader never sees a card before its release.
3. Add the new card to the toctree, newest first.
4. Store the cover's art as `_static/release-cards/<release>.jpg`: a finished JPEG, 8.5 x 11 at
   three times the cover's 288 px width, with its own glow and grain and a dark ground behind the
   title and the version: the art recolored with `brand_tones` in `site/cards.py`, so its cobalt,
   amber, and lightness match the other covers, with a soft navy shade under the masthead, the
   title, and the version. Add a `release-art-<release>` rule (dots as dashes) in
   `_static/benchmarks.css` that points at it, and give the new card's cover and the Release cards
   page's featured cover both classes, `release-art release-art-<release>`. The bare `release-art`
   rule keeps 0.2.0's art, which its released card uses. A cover without the `release-art` class
   shows the stipple ground in the same tones and shade (`release_ground_jpeg` in
   `site/cards.py`). When the release moves to the earlier releases, its tile keeps the class
   `release-art-<release>`, so the tile shows the same art.
5. Open the card with the same cover, without its link, beside the opening paragraphs in a
   `release-opening` block.
6. Add the summary's folder to `RELEASES` in `site/build.py`, so the homepage's speed and accuracy
   sections and the README count it.

A card's figures are finished HTML files in `docs/benchmarks/<release>/`, included with
`{raw} html`, so later code never redraws a released card. `_static/benchmarks.css` styles the
page and the cards' covers, and loads on those pages only; a card also loads `figures.css` and
`figures.js`, the styles and script of the posts' figures.

## The documentation sidebars

The left sidebar ([`_templates/docs-nav.html`](_templates/docs-nav.html)) groups the docs index's
toctree: each section named in `NAV_SECTIONS` in `conf.py` heads a group of its pages under its
icon ([`_templates/nav-icons.html`](_templates/nav-icons.html)), folding those past `NAV_SHOWN`,
and the index's other entries make the Reference group. `_docs_sidebar` in `conf.py` builds the
groups from the toctrees, so a page added to a toctree appears with no template change. The right
sidebar holds the page's outline ([`_templates/page-toc.html`](_templates/page-toc.html)), whose
mark [`_static/page-outline.js`](_static/page-outline.js) moves to the section being read, then
links to edit the page and report a problem
([`_templates/page-links.html`](_templates/page-links.html)). Edit this page opens the file that
holds the page's text: an included Markdown file or notebook, such as `CHANGELOG.md` or a
notebook in `examples/`, else the page's own source; an API page has none. The page's section
sits over its title ([`_templates/page-kicker.html`](_templates/page-kicker.html)). On a phone the
outline moves into the menu, above the tree
([`_templates/page-outline.html`](_templates/page-outline.html)).

## The Examples and Demos galleries

[`docs/examples/index.md`](docs/examples/index.md) is a gallery of notebook cards.
[`docs/demos/index.md`](docs/demos/index.md) is a gallery of demos, which run in the browser, and a
live demo, whose card appears once `DEMO_URL` in `site/build.py` names a public Space. The toctrees
keep every page in the sidebar.

Each demo is a page with a title, one paragraph, and the demo at full width in a frame. `conf.py`
generates the demos into `_static/examples/` at build time, so a docs-only build has them: the
comparison on one dataset from [`apps/compare.py`](../apps/compare.py), and Meek's rules, the
d-separation explorer, and the benchmark explorer from
[`apps/build_assets.py`](../apps/build_assets.py). The comparison computes Andrey's graphs at build
time and reads causal-learn's from `apps/compare-causal-learn.json`, recorded once with causal-learn
installed (see [`apps/README.md`](../apps/README.md)); the build refuses a recording made on
different data.

Thumbnails come from each page's own output at build time. A demo draws its
own; a notebook's is the first figure the executed notebook produces, or, for a notebook that shows
none, its dataset's true graph (`NOTEBOOK_DATA` in `conf.py`). They are drawn in the light palette,
on its paper, in both themes.

## Notebooks

Canonical notebooks live in `examples/`. The build stages them in `docs/examples/` beside the
gallery and executes them with MyST-NB's cache. Failed cells fail the build. Ordinary builds reuse
unchanged notebook outputs; `test_built_site.py` clears the cache for its clean build. The speed
notebook is not executed, so its page shows no builder timings; stderr is dropped from every page.
Run notebooks locally with `uv run --with jupyter jupyter lab`.

Notebook dependencies stay out of `pyproject.toml`. Every notebook has an Open in Colab link to
the public repository's default branch and a Colab-only setup cell. Outside Colab, the speed
notebook installs causal-learn with `uv pip install` in a cell; other notebooks need only Andrey.
The setup installs the pinned release from PyPI. Update that pin when publishing a new
release. The speed notebook identifies Colab timings as measurements of the
Colab runtime.

The staging step adds Sphinx's download role to the rendered page, to the right of the Open in
Colab badge, and points it at the canonical notebook in `examples/`. Downloads therefore contain
ordinary Jupyter Markdown.

## Preview locally

Serve the built HTML and open it in a browser:

```shell
python -m http.server 8123 --bind 127.0.0.1 --directory docs/_build/html
```

On a remote machine, keep the server bound to `127.0.0.1` and forward the port from your laptop:
`ssh -N -L 8123:localhost:8123 <user>@<host>`, then open <http://localhost:8123>.

## Search and answer engines

[`discovery.py`](discovery.py) writes what search engines and AI answer engines read. Each page's
head carries its Twitter card fields and one JSON-LD graph: the organization, the website, the
software at the release the page belongs to (a post's `version`, a release card's number, else the
current release), the page, a `BlogPosting` for a post with its date and authors, and the page's
breadcrumbs. The search page is `noindex`. After the build, `_write_discovery` in `conf.py` writes
three files at the site's root from the built pages: `llms.txt`, the pages in `LLMS_SECTIONS`
under Docs, Release cards, and Blog, each with its own title and description, every release card
and published post following its section's pages; `sitemap.xml`, the address of every indexable
page; and `robots.txt`, which lets every crawler read every page and names the sitemap. A page
listed in `LLMS_SECTIONS` that is renamed or removed fails the build until the list follows it.

## Validation

CI runs the site's unit tests, then [`test_built_site.py`](test_built_site.py):

```shell
uv run --group docs python -m pytest docs/test_built_site.py
```

The unit tests check the sources and the claims: `tests/unit/test_site.py` (the homepage),
`tests/unit/test_launch_material.py` (the README and the launch post),
`tests/unit/test_examples_gallery.py` (the gallery), `tests/unit/test_why_fast.py` (the speed
scenes), `tests/unit/test_compare_example.py` (the comparison on one dataset), and
`tests/unit/test_discovery.py` (the head tags, `llms.txt`, `sitemap.xml`, and `robots.txt`).
`test_built_site.py` builds the site once from a clean workspace and checks what only a built site
shows:

- A strict `-W` HTML build of the whole site, including notebook execution.
- API coverage: every public symbol of `andrey`, `andrey.metrics`, and `andrey.viz`, and every
  dataset loader of `andrey.data`, is documented, and no private symbol is exposed.
- LaTeX math for KaTeX rendering, with no raw Unicode math symbols.
- The built pages: every page has one description, the same as its link preview's, at most 160
  characters, and neither empty nor only the project's name; its wordmark leads home and Docs,
  Examples, and Blog lead to their sections; its favicon and link-preview card were built. The
  homepage's stylesheets carry a content version. The blog's front page leads with the newest post
  and its cover and lists every other post once, newest first; each post has a built cover and a
  byline with its author and date. Every site address in the README, the
  project URLs, and the citation is built.
- Search and answer engines: every page's structured data matches its link preview, and each
  post's date, authors, and release match its front matter; `llms.txt` links public pages by
  section with their own titles and descriptions; `sitemap.xml` lists each indexable page once,
  and `robots.txt` names it; only the search page is `noindex`.

To check external links: `uv run --group docs sphinx-build -b linkcheck docs docs/_build/linkcheck`.

## How the site is wired

| Piece | Where |
|---|---|
| Sphinx config (theme, extensions, notebooks, blog, generated pages) | `docs/conf.py` |
| Homepage template, stylesheet, and content generators | `site/homepage.html`, `site/assets/site.css`, `site/build.py` |
| Timing comparison and speed charts, shared by three pages | `site/assets/figures.css`, `site/assets/figures.js` |
| "Why Andrey is fast" scenes: homepage story, blog post | `site/why_fast.py`, `site/assets/why-fast.css`, `site/assets/why-fast.js`, `site/assets/motion.js` |
| Navbar: wordmark and sections | `docs/_templates/site-logo.html`, `docs/_templates/site-nav.html` |
| Wordmark (Inter outlines, light and dark ink) and its generator | `site/brand/`, `site/wordmark.py` |
| Palette as CSS custom properties | `docs/palette_css.py` |
| Blog posts and the front page that lists them | `docs/blog/`, `docs/_static/blog.css` |
| Release cards and the page that lists them | `docs/docs/benchmarks/`, `docs/docs/benchmarks.md`, `docs/_static/benchmarks.css` |
| Docs landing page and section index | `docs/docs/index.md` |
| API reference (autodoc from `andrey.__all__`, the Typer CLI) | `docs/docs/code/index.md` |
| Guides (getting started, contributing) | `docs/docs/guides/` |
| Examples gallery | `docs/docs/examples/` |
| Demos gallery and demo pages | `docs/docs/demos/` |
| Notebook sources | `examples/*.ipynb` -> staged into `docs/docs/examples/` |
| Docs chrome over the generated palette tokens | `docs/_static/andrey.css` |
| The old demo address, redirected to the gallery | `docs/_templates/demo-redirect.html` |

Every page takes its colors from `andrey.viz.LIGHT` and `andrey.viz.DARK`: `docs/palette_css.py`
renders them as `--andrey-*` custom properties, and the stylesheets name no hex.
