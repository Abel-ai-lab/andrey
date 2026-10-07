# Andrey design

Every visual Andrey makes uses one palette: a drawn graph, a chart, a figure in the docs, and the
website itself. The colors are the `Palette` objects `andrey.viz.LIGHT` and `andrey.viz.DARK` in
[`src/andrey/viz/theme.py`](src/andrey/viz/theme.py), and
[`docs/palette_css.py`](docs/palette_css.py) writes the website's CSS variables from the same
objects, so a figure and the page around it match. Change a color in `theme.py`; never type a hex
value into CSS or a figure.

## Principles

1. **A neutral backbone, with color for uncertainty.** A directed edge, which the method resolved,
   is graphite, so the structure stays calm. Color marks what is uncertain: an undirected edge
   (orientation unknown), a bidirected edge (a hidden common cause), and a circle endpoint (a PAG's
   partial mark).
2. **Marks by shape, not by color.** An endpoint is an arrowhead, a hollow circle, or a plain tail,
   so a graph reads without color and in grayscale.
3. **Cobalt is the accent, not a fill.** It marks a highlighted endpoint, and links and selection on
   the website. Nodes stay neutral.
4. **Light and dark are equal.** Every color has a light and a dark value.

## Colors

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

## Charts

- **Series** take the categorical eight, in this order, safe for color blindness (light):
  `#3b5bdb #e0982b #2e9e5b #e5573f #7c5cd8 #2f9bd0 #c74e9b #c9a227`.
- **Magnitudes** (confidence, weight, density) take the cobalt ramp (light):
  `#e7ebfb #bcc8f4 #7c93ee #3b5bdb #1f338f`.
- `andrey.viz.mpl` applies both to matplotlib: `mpl.use()` or `mpl.context()` for the style,
  `mpl.cycle()` for the series colors, and `mpl.cmap()` for the ramp.

## Drawing a graph

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

## Type

The wordmark is Inter, drawn as outlines in [`site/brand/`](site/brand/) by
[`site/wordmark.py`](site/wordmark.py), in a light and a dark ink.
