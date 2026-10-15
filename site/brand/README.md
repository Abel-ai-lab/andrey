# Brand files

- `art-<cover>.jpg`: each blog post's art, 1600 x 640, its subject on the right where the banner
  leaves room for the text, stored after `brand_tones` in `site/cards.py` recolors it to the
  brand's tones (a site test checks). The build adds the wordmark, the title, and the release.
  - `art-pair.jpg`: the launch post's, portraits of Andrey Markov and Andrey Kolmogorov drawn as
    navy line screens, overlapping, Kolmogorov mirrored to face left and scaled so both heads read
    the same size (Markov from his 1886 portrait, public domain, Wikimedia Commons
    `File:Andrei_Markov.jpg`).
  - `art-lanes.jpg`: "Why Andrey is fast" and the homepage's hero, lanes rising like a bridge into
    a starry sky, the others moving in short steps while one amber streak runs far ahead.
  - `art-iceberg.jpg`: 0.2.0, an iceberg whose two peaks rise out of the frame and join into one
    faint mass under the waterline, a dim amber light at its center.
  - `art-contour.jpg`: 0.3.0, two summits drawn in navy contour lines: from one trailhead, a white
    dotted trail ends on the lower summit and an amber one climbs to a glowing star on the higher.
  - `art-hourglass.jpg`: 0.4.0, an hourglass in navy lines, cropped by the frame's top and bottom,
    its amber sand falling in a fine stream from the upper bulb to the lower: one way only.
  - `art-agentic.jpg`: the homepage's "agentic", an orrery in pale engraved rings tilted across
    the frame, a small directed graph on its rings, and one amber arm from its hidden center
    steering one node.
  - `art-start.jpg` and `art-docs.jpg`: the homepage's "From data to graph" (data points
    gathering into a small graph, one arrow amber) and its documentation (an atlas's pages fanning
    open, one edge amber).
  - `art-methods.jpg`: the homepage's All-in-one, the prism of `bento/methods.jpg`, one beam
    fanning into many rays, the level ray out of the prism drawn amber. It is scaled from that
    crop to the cover's size, so it is softer than the others until the art is drawn again at
    full size.
- `bento/<art>.jpg`: the homepage's bento tiles, each a crop of an art at twice its tile's size
  (1168 x 416 for a wide tile, 768 x 416 for a narrow one), in the same tones. The crops of the
  covers above keep their names; the others are drawn for their tile: `hardware` (a honeycomb of
  cells, one lit), `constraint` (a web of threads, most cut), `permutation` (a row of slabs, one
  lifted out), `lingam` (dunes leaning one way), `accuracy` (a sighting reticle over a graph),
  `graphs` (many streams braided into one), `compare` (a level balance), and `methods` (a prism
  fanning one beam into many rays). A narrow tile's art, and any tile that shows a number, is
  cobalt only, so the number is its one amber.
- The release cards' covers (`docs/_static/release-cards/<release>.jpg`) are tall versions of the
  same art, in the same tones, with calm areas where the card's text sits.
- `abel-logo-white.svg`: the Abel AI Lab logotype, from Abel's brand kit. Its last path is the
  mark, which the homepage's footer draws beside Abel's name.
- `abel-symbol-night.svg`: Abel's logomark in Night Teal, for light backgrounds, unchanged from
  the brand kit (`Abel-design-branding`, `assets/logos/symbol-night.svg`). The build draws it as
  `_static/abel-symbol.png`, the organization's logo in the site's structured data.
- `wordmark-light.svg`, `wordmark-dark.svg`, `favicon.svg`: Andrey's wordmark and icon.
- `fonts/`: Inter, under the SIL Open Font License (`fonts/OFL.txt`).
