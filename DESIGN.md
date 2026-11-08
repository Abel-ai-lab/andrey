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

Andrey is named after Andrey Markov and Andrey Kolmogorov. The wordmark's A represents the pair:
one letter with two strokes, the right one amber and a step ahead.

### Node and edge

The brand uses the two parts of a causal graph as plain marks, never as drawn graphs:

- **Node, a dot**, marks one thing to notice: the dot before a release's version on a link preview,
  above the version on a release card's cover, and on Release cards in the navbar for a release
  the reader has not seen, for a week after the release (`nav-new`).
- **Edge, a line**, marks a relation or direction: the A's amber stroke, the single amber stroke
  in each artwork, and the hairline under a heading in the documentation footer.
- **Amber is the one accent.** A composition (a cover, a card, a link preview, a page's header)
  holds one amber node or one amber edge; the rest is ink, cobalt, and navy.
- **No drawn graph in a brand mark.** A wordmark, an icon, or a marker never shows nodes joined by
  arrows, a network, or a molecule; graphs belong to figures and to the art.

### Wordmark and icon

- The wordmark is "Andrey" in Unbounded SemiBold, drawn as outlines in [`site/brand/`](site/brand/)
  by [`site/wordmark.py`](site/wordmark.py). It uses light or dark ink with the A's right stroke in
  amber. The A stays one letter, with a joined apex and an intact crossbar; a split, gapped, or
  crossbar-less A resembles another company's mark. The word is never retyped in a font.
- The icon (`site/brand/favicon.svg`) is that A alone; its ink follows the browser's light or dark
  bar.
- Over art (a link preview, the README banner) the wordmark is white and keeps its amber stroke
  (`_placed` in `site/cards.py`).

### Type

Unbounded is reserved for the wordmark. Everything else uses Inter: interface, text, and titles
over art (SemiBold, white). Code uses the system's monospace font.

### Color

The light theme is ink on paper and the dark theme pale ink on night. Cobalt marks links, selection,
and the art's ground; navy is the art's deep ground and the footer's; amber is the sole accent. The
website uses the `--andrey-ground-*` and `--andrey-*` custom properties (`site/grain.py`,
`docs/palette_css.py`); no stylesheet specifies a hex value.

### Art

Every cover uses engraved line art: thin, even white and pale-cobalt lines on a flat cobalt ground
with fine stippling, modeled on Soviet space-age posters of 1958-1963. Each has one amber element
(a node or an edge), with the subject on the right and text on the left, and is stored after
`brand_tones` processing. [`site/brand/README.md`](site/brand/README.md) lists each artwork and its
subject.

## Figures

A figure is a drawn graph, a chart, or a diagram in the docs.

### Principles

1. **A neutral backbone, with color for uncertainty.** A directed edge, which the method resolved,
   is graphite, so the structure stays calm. Color marks what is uncertain: an undirected edge
   (orientation unknown), a bidirected edge (a hidden common cause), and a circle endpoint (a PAG's
   partial mark).
2. **Marks by shape, not by color.** An endpoint is an arrowhead, a hollow circle, or a plain tail,
   so a graph reads without color and in grayscale.
3. **Cobalt is the accent, not a fill.** It lights a highlighted endpoint, and marks links and
   selection on the website. Nodes stay neutral.
4. **Light and dark are equal.** Every color has a light and a dark value.

### Colors

Each token is a field of `Palette`, with one value in `andrey.viz.LIGHT` and one in
`andrey.viz.DARK` ([`src/andrey/viz/theme.py`](src/andrey/viz/theme.py)); the
[brand kit](docs/docs/brand.md#colors) shows the brand colors with their values.

| Token | Use |
|---|---|
| `accent` | highlighted endpoints; links and selection |
| `ink` | text; the ring of an observed node |
| `muted` | secondary text; latent nodes |
| `paper`, `surface_2` | grounds |
| `surface` | the fill of an observed node |
| `line` | a chart's axes and grid |
| `edge_directed` | a resolved edge |
| `edge_undirected` | orientation unknown (CPDAG) |
| `edge_bidirected` | a hidden common cause (PAG) |
| `edge_partial` | a circle endpoint (PAG) |

### Charts

- **Series** take the eight `categorical` colors, in their order, safe for color blindness.
- **Magnitudes** (confidence, weight, density) take the cobalt `ramp`.
- `andrey.viz.mpl` applies both to matplotlib: `mpl.use()` or `mpl.context()` for the style,
  `mpl.cycle()` for the series colors, and `mpl.cmap()` for the ramp.

### Tables

A table that compares Andrey with other packages or builds reads one way everywhere:

- Andrey's column comes first and is tinted in the accent, its values in the accent color.
- In each row, the best value of each measure (the lower time, the fewer errors, the higher F1) is
  bold, and shaded neutral when another column holds it. The empty graph is a reference, never the
  best.
- Rows group by method family, the family's label set on the table's side.
- A share is a percentage, its denominator in the text or the row's label, not in every cell.

The 0.7.0 post's tables come from `table()` in `docs/blog/0.7.0/figures.py`; a Markdown table on a
post or a release card gets the tint and the bold at build time (`_comparison_tables`,
`docs/conf.py`).

### Drawing a graph

`andrey.viz.draw` follows these rules for every kind of graph (DAG, CPDAG, PAG, digraph):

- A latent node is hollow and dashed.
- An intervened node (`targets=`) is a square, the shape of a regime or decision node in
  influence diagrams, and a bar crosses each arrowhead into it: the edges an intervention cuts. A
  target no arrowhead points into is marked by its square alone. Shape alone carries the mark, so
  it reads without color, and a latent target keeps its dashed ring. `to_graphviz` writes the
  square and the bars in DOT: `shape=square` and `normaltee` arrowheads.
- A 2-cycle is two parallel curved edges, distinct from one bidirected edge.
- An edge that would cross a node it does not join bends around it, so a drawing never suggests an
  edge the graph lacks.
- The layout is deterministic. With Graphviz (the `viz` extra), `dot` ranks the nodes along the
  directed edges, causes on the left; without it, nodes sit in layers, or on a circle when no edge
  is directed.
- Labels sit inside the nodes when every label has at most two characters, and below them
  otherwise, the same for the whole figure. A label below its node is outlined in the ground color
  (`paper`), so an edge that passes under it never crosses its letters.
- A highlighted endpoint lights up: its mark and its half of the edge turn accent, a little
  heavier, over a soft accent glow. The highlight is drawn in strokes, so it keeps its proportions
  at any size.
