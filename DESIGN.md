# Andrey design

Every visual Andrey makes uses one palette: a drawn graph, a chart, a figure in the docs, and the
website itself. The colors are the `Palette` objects `andrey.viz.LIGHT` and `andrey.viz.DARK` in
[`src/andrey/viz/theme.py`](src/andrey/viz/theme.py), and
[`docs/palette_css.py`](docs/palette_css.py) writes the website's CSS variables from the same
objects, so a figure and the page around it match. Change a color in `theme.py`; never type a hex
value into CSS or a figure. Drawn art (a post's banner art in `site/brand/art-*.jpg`, a release
card's cover in `docs/_static/release-cards/`) is stored after `brand_tones` in `site/cards.py`
recolors it to the same palette, and a site test fails on art stored in other blues or ambers.

## Brand

### Name and mark

Andrey is named after Andrey Markov and Andrey Kolmogorov. The wordmark's A carries the pair: one
letter, two strokes, the right one amber, a step ahead.

### Node and edge

The brand builds on the two parts of a causal graph, used as plain marks, never as drawn graphs:

- **Node, a dot**, marks one thing to notice: the dot before a release's version on a link preview
  and over it on a release card's cover, and the dot on Release cards in the navbar when a newer
  release is out (`nav-new`).
- **Edge, a line**, marks a relation or a direction: the A's amber stroke, the one amber stroke in
  each art's composition, and a hairline under a heading (the documentation's footer).
- **Amber is the one accent.** A composition (a cover, a card, a link preview, a page's header)
  holds one amber node or one amber edge; the rest is ink, cobalt, and navy.
- **No drawn graph in a brand mark.** A wordmark, an icon, or a marker never shows nodes joined by
  arrows, a network, or a molecule; graphs belong to figures and to the art.

### Wordmark and icon

- The wordmark is "Andrey" in Unbounded SemiBold, drawn as outlines in [`site/brand/`](site/brand/)
  by [`site/wordmark.py`](site/wordmark.py): the light or dark ink, the A's right stroke in amber.
  The A stays one letter, its apex joined and its crossbar whole; a split, gapped, or
  crossbar-less A reads as another company's mark. The word is never retyped in a font.
- The icon (`site/brand/favicon.svg`) is that A alone; its ink follows the browser's light or dark
  bar.
- Over art (a link preview, the README banner) the wordmark is white and keeps its amber stroke
  (`_placed` in `site/cards.py`).

### Type

Unbounded is the wordmark's alone. Everything else is set in Inter: interface, text, and the titles
over art (SemiBold, white). Code is in the system's monospace.

### Color

The brand's grounds are the palette's paper and ink in the light theme and navy in the dark,
with cobalt for links, selection, and the art's ground, and amber as the one accent. On the
website they are the `--andrey-ground-*` and `--andrey-*` custom properties (`site/grain.py`,
`docs/palette_css.py`); no stylesheet names a hex value.

### Art

Every cover is engraved line art: thin, even white and pale-cobalt lines on a flat cobalt ground
with a fine stipple, after the Soviet space-age posters of 1958-1963. It holds one amber element
(its node or its edge), puts its subject on the right where text sits on the left, and is stored
after `brand_tones`. [`site/brand/README.md`](site/brand/README.md) lists each art and what it
shows.

## Figures

A figure is a drawn graph, a chart, or a diagram in the docs.

### Principles

1. **A neutral backbone, with color for uncertainty.** A directed edge, which the method resolved,
   is graphite, so the structure stays calm. Color marks what is uncertain: an undirected edge
   (orientation unknown), a bidirected edge (a hidden common cause), and a circle endpoint (a PAG's
   partial mark).
2. **Marks by shape, not by color.** An endpoint is an arrowhead, a hollow circle, or a plain tail,
   so a graph reads without color and in grayscale.
3. **Cobalt is the accent, not a fill.** It marks a highlighted endpoint, and links and selection on
   the website. Nodes stay neutral.
4. **Light and dark are equal.** Every color has a light and a dark value.

### Colors

| Token | Light | Dark | Use |
|---|---|---|---|
| accent | `#3b5bdb` | `#6e86f5` | highlighted endpoints; links and selection |
| ink | `#182233` | `#e8ecf5` | text; the ring of an observed node |
| muted | `#5c6675` | `#8d97ab` | secondary text; latent nodes |
| paper / surface_2 | `#fafbfc` / `#eef1f6` | `#0b1220` / `#182236` | grounds |
| surface | `#ffffff` | `#121a2b` | the fill of an observed node |
| line | `#d9dde6` | `#25314c` | a chart's axes and grid |
| edge directed | `#3a4658` | `#98a4bd` | a resolved edge |
| edge undirected | `#e0982b` | `#f0b357` | orientation unknown (CPDAG) |
| edge bidirected | `#d6495b` | `#f0687a` | a hidden common cause (PAG) |
| edge partial | `#7c5cd8` | `#a08cf0` | a circle endpoint (PAG) |

### Charts

- **Series** take the categorical eight, in this order, safe for color blindness (light):
  `#3b5bdb #e0982b #2e9e5b #e5573f #7c5cd8 #2f9bd0 #c74e9b #c9a227`.
- **Magnitudes** (confidence, weight, density) take the cobalt ramp (light):
  `#e7ebfb #bcc8f4 #7c93ee #3b5bdb #1f338f`.
- `andrey.viz.mpl` applies both to matplotlib: `mpl.use()` or `mpl.context()` for the style,
  `mpl.cycle()` for the series colors, and `mpl.cmap()` for the ramp.

### Drawing a graph

`andrey.viz.draw` follows these rules for every kind of graph (DAG, CPDAG, PAG, digraph):

- A latent node is hollow and dashed.
- A 2-cycle is two parallel curved edges, distinct from one bidirected edge.
- An edge that would cross a node it does not join bends around it, so a drawing never suggests an
  edge the graph lacks.
- The layout is deterministic. With Graphviz (the `viz` extra), `dot` ranks the nodes along the
  directed edges, causes on the left; without it, nodes sit in layers, or on a circle when no edge
  is directed.
- Labels sit inside the nodes when every label has at most two characters, and below them
  otherwise, the same for the whole figure.
- A highlighted endpoint gets an accent ring.
