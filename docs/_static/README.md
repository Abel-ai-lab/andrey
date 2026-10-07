# `docs/_static/`

The site palette is the package palette: `src/andrey/viz/theme.py` is the single source of truth,
so no color is transcribed here.

- **`andrey-tokens.css`** - **generated** by `conf.py` at build time from `andrey.viz.LIGHT` and
  `andrey.viz.DARK`, one `--andrey-*` custom property per palette field, per theme, through
  `docs/palette_css.py`, plus `--andrey-light-paper` for the gallery's thumbnails. Gitignored;
  never hand-edit. To change a color, edit `theme.py`.
- **`andrey.css`** - the chrome (hand-edited): maps pydata-sphinx-theme's `--pst-*` variables onto
  the generated tokens and sets what is not a color (type, radii, motion). It names no hex.

- **`graph-guides.css`**, **`graph-guides.js`** - graph-guide layout and progressive controls.
  The comparison works with keyboard or pointer input; all examples remain visible without JS.
  The script loads only on the graph-representation page.
- **`graph-comparison.html`**, **`graph-types-*.html`** - generated
  by `docs/graph_diagrams.py` at build time. Graphs are validated `GraphStructure` objects rendered
  with `andrey.viz.draw`; tables come from their `to_numpy()` arrays. SVGs consume palette tokens
  directly. Gitignored.

- **`blog.css`** - the blog's post list and post card, and a post's cover and figures.
  Loaded on blog pages only.
- **`examples.css`** - the Examples gallery's cards and the frame an interactive example fills.
  Loaded on the gallery and the example pages only.

Load order in `conf.py` is `andrey-tokens.css`, `andrey.css`, then `graph-guides.css`; a page's own
stylesheets load after them. The homepage's stylesheets, and the timing comparison and charts it
shares with the blog, are in `site/assets/`. Build-time output that is not a stylesheet here (the
interactive examples, their frame script, thumbnails, the blog's figures) goes to the gitignored
`docs/_generated/`.
