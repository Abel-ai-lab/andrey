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
the race, the charts, the claim cards, the comparison table, and the table's footnote.
`site/assets/site.css` styles it, scoped to `.andrey-home` so the theme's styles and the page's do
not meet; `site/assets/figures.css` and `figures.js` hold the timing comparison and the speed
charts, which the launch post and the benchmark explorer show too.

The speed section's "Why Andrey is fast" pill opens a dialog of four animated scenes, one per idea
behind the speed, built by [`site/why_fast.py`](../site/why_fast.py). The blog post
`docs/blog/why-andrey-is-fast.md` shows the same scenes under its own headings. Each scene compares
Andrey with a lane labeled Others and names no package. `site/assets/why-fast.js` plays them with
Motion, vendored unmodified as `site/assets/motion.js` under its MIT license; the version is pinned
in the file's banner and in `tests/unit/test_why_fast.py`. `conf.py` loads both scripts only on
those two pages.

The homepage takes every number from the release summaries that `RELEASES` in `site/build.py` names,
each `benchmarks/published/<label>/summary.json`, or from the method registry,
`andrey.spec.list_specs()`. Only methods the registry supports count. `comparisons()` reads every
kind of summary into one layout: rows at mean degree 2 where a summary measures more than one, the
observed variables on data with hidden variables, a ratio the summary flags as a lower bound printed
as one, and each comparison's errors in the measure its summary records (`ERRORS`: the structural
Hamming distance, or directed-edge errors on data with hidden common causes). The claim under the
heading is the one `claim_sentence()` words from every supported comparison: the largest ratio
rounded down to a power of ten on its method, then whether the other supported methods are faster
("the other", "most other", or nothing). The build also fails unless the first summary's headline is
the claim its own numbers support; the docs index, the notebook, and the live demo repeat it. The
speedup panel shows one slide per method in `PANEL`, a (release, method) pair each, with every
package measured in the same runs at its largest size with a paired ratio; the README chart draws
the same slides as cards and refuses more than six. The hero names every registry family, and the
method list shows every registered method. The page's description, which search results and link
previews show, names PC, GES, and FCI, counts the other registered methods, and states the claim's
first part ("over 100x faster on PC"). Every other page's description is the `description` in its
front matter; without one, the page's first paragraph, cut at a word to 160 characters (an API
page's is its docstring's summary line). A page with neither, such as search or the blog archive,
takes the homepage's. To preview with placeholder numbers, which replace the first release's
summary:

```shell
uv run --group docs sphinx-build -b html docs docs/_build/html \
  -D andrey_summary=tests/unit/fixtures/summary.placeholder.json -D andrey_placeholder=1
```

The page is then marked "not for release" and `noindex`. The fixture carries
`"placeholder": true`, which the build refuses without `andrey_placeholder` wherever the file
sits. The examples on the page are [`site/example.py`](../site/example.py) and
[`site/interface.py`](../site/interface.py), run at build time; the page shows what they print and
draws the graphs they return.

## The blog

ABlog builds the blog from [`blog/`](blog/index.md): the post list at `/blog/`, an archive, author
pages, and an Atom feed at `/blog/atom.xml`. A post is a MyST page whose front matter sets
`blogpost: true`, its `author`, and its `date`. A post without a date is a draft, which the post
list leaves out. A dated post is published in every build, even before its date, so a future date
does not keep a post private.

A post belongs to the release in its `version` front matter, and later builds leave it as that
release rendered it: its numbers are typed text, and every file it includes (a figure, code, or
output) is stored under `blog/<version>/`. The alpha banner above every page links the newest
dated post, a later version breaking a tie in date (`_release_post` in `conf.py`).

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
   title and the version. Add a `release-art-<release>` rule (dots as dashes) in
   `_static/benchmarks.css` that points at it, and give the new card's cover and the Release cards
   page's featured cover both classes, `release-art release-art-<release>`. The bare `release-art`
   rule keeps 0.2.0's art, which its released card uses. A cover without the `release-art` class
   shows the stipple ground instead.
5. Open the card with the same cover, without its link, beside the opening paragraphs in a
   `release-opening` block.
6. Add the summary's folder to `RELEASES` in `site/build.py`, so the homepage's speed and accuracy
   sections and the README count it.

A card's figures are finished HTML files in `docs/benchmarks/<release>/`, included with
`{raw} html`, so later code never redraws a released card. `_static/benchmarks.css` styles the
page and the cards' covers, and loads on those pages only; a card also loads `figures.css` and
`figures.js`, the styles and script of the posts' figures.

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

## Validation

CI runs the site's unit tests, then [`test_built_site.py`](test_built_site.py):

```shell
uv run --group docs python -m pytest docs/test_built_site.py
```

The unit tests check the sources and the claims: `tests/unit/test_site.py` (the homepage),
`tests/unit/test_launch_material.py` (the README and the launch post),
`tests/unit/test_examples_gallery.py` (the gallery), `tests/unit/test_why_fast.py` (the speed
scenes), and `tests/unit/test_compare_example.py` (the comparison on one dataset).
`test_built_site.py` builds the site once from a clean workspace and checks what only a built site
shows:

- A strict `-W` HTML build of the whole site, including notebook execution.
- API coverage: every public symbol of `andrey`, `andrey.metrics`, and `andrey.viz`, and every
  dataset loader of `andrey.data`, is documented, and no private symbol is exposed.
- LaTeX math for KaTeX rendering, with no raw Unicode math symbols.
- The built pages: every page has one description, the same as its link preview's, at most 160
  characters, and neither empty nor only the project's name; its wordmark leads home and Docs,
  Examples, and Blog lead to their sections; its favicon and link-preview card were built. The
  homepage's stylesheets carry a content version. The blog lists every post, and each post has a
  built cover and a byline with its author and date. Every site address in the README, the
  project URLs, and the citation is built.

To check external links: `uv run --group docs sphinx-build -b linkcheck docs docs/_build/linkcheck`.

## How the site is wired

| Piece | Where |
|---|---|
| Sphinx config (theme, extensions, notebooks, blog, generated pages) | `docs/conf.py` |
| Homepage template, stylesheet, and content generators | `site/homepage.html`, `site/assets/site.css`, `site/build.py` |
| Timing comparison and speed charts, shared by three pages | `site/assets/figures.css`, `site/assets/figures.js` |
| "Why Andrey is fast" scenes: homepage dialog, blog post | `site/why_fast.py`, `site/assets/why-fast.css`, `site/assets/why-fast.js`, `site/assets/motion.js` |
| Navbar: wordmark and sections | `docs/_templates/site-logo.html`, `docs/_templates/site-nav.html` |
| Wordmark (Inter outlines, light and dark ink) and its generator | `site/brand/`, `site/wordmark.py` |
| Palette as CSS custom properties | `docs/palette_css.py` |
| Blog posts and the post list | `docs/blog/` |
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
