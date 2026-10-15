---
name: brand
description: Andrey's brand kit - the wordmark, the icon, and an avatar to download, with the colors, type, and how to use them.
meta:
  type: guide
  tags: [brand]
---

# Brand kit

Andrey's wordmark and icon, for writing about Andrey or linking to it. Download what you need, or
everything at once.

```{raw} html
<p class="brand-all"><a class="sd-btn sd-btn-primary" href="../_static/brand/andrey-brand-kit.zip" download>Download the brand kit (ZIP)</a></p>
```

## Wordmark

The wordmark is "Andrey" in Unbounded, its A's right stroke in amber. Use the ink version on light
grounds and the pale one on dark grounds.

```{raw} html
<ul class="brand-assets">
  <li class="brand-asset">
    <div class="brand-asset__view brand-asset__view--light"><img class="dark-light" src="../_static/brand/andrey-wordmark-on-light.svg" alt="Andrey wordmark in ink" width="240" height="51"></div>
    <p class="brand-asset__name">On light grounds</p>
    <p class="brand-asset__files"><a href="../_static/brand/andrey-wordmark-on-light.svg" download>SVG</a><a href="../_static/brand/andrey-wordmark-on-light.png" download>PNG</a></p>
  </li>
  <li class="brand-asset">
    <div class="brand-asset__view brand-asset__view--dark"><img class="dark-light" src="../_static/brand/andrey-wordmark-on-dark.svg" alt="Andrey wordmark in pale ink" width="240" height="51"></div>
    <p class="brand-asset__name">On dark grounds</p>
    <p class="brand-asset__files"><a href="../_static/brand/andrey-wordmark-on-dark.svg" download>SVG</a><a href="../_static/brand/andrey-wordmark-on-dark.png" download>PNG</a></p>
  </li>
</ul>
```

## Icon and avatar

The icon is the wordmark's A alone, for small spaces: a favicon, an app tile, a slide's corner. The
avatar sets it on cobalt for profile pictures.

```{raw} html
<ul class="brand-assets brand-assets--three">
  <li class="brand-asset">
    <div class="brand-asset__view brand-asset__view--light"><img class="dark-light" src="../_static/brand/andrey-icon-on-light.svg" alt="Andrey icon in ink" width="96" height="96"></div>
    <p class="brand-asset__name">Icon, light grounds</p>
    <p class="brand-asset__files"><a href="../_static/brand/andrey-icon-on-light.svg" download>SVG</a><a href="../_static/brand/andrey-icon-on-light.png" download>PNG</a></p>
  </li>
  <li class="brand-asset">
    <div class="brand-asset__view brand-asset__view--dark"><img class="dark-light" src="../_static/brand/andrey-icon-on-dark.svg" alt="Andrey icon in pale ink" width="96" height="96"></div>
    <p class="brand-asset__name">Icon, dark grounds</p>
    <p class="brand-asset__files"><a href="../_static/brand/andrey-icon-on-dark.svg" download>SVG</a><a href="../_static/brand/andrey-icon-on-dark.png" download>PNG</a></p>
  </li>
  <li class="brand-asset">
    <div class="brand-asset__view brand-asset__view--plain"><img class="brand-asset__avatar dark-light" src="../_static/brand/andrey-avatar.png" alt="Andrey avatar: the icon on cobalt" width="112" height="112"></div>
    <p class="brand-asset__name">Avatar</p>
    <p class="brand-asset__files"><a href="../_static/brand/andrey-avatar.png" download>PNG</a></p>
  </li>
</ul>
```

## The A

Andrey is named after two mathematicians, Andrey Markov and Andrey Kolmogorov. The A carries the
pair: one letter, two strokes, the right one amber, a step ahead. The letter always stays whole,
its apex joined and its crossbar intact.

## Node and edge

The brand builds on the two parts of a causal graph, used as plain marks.

```{raw} html
<ul class="brand-marks">
  <li><span class="brand-mark brand-mark--node" aria-hidden="true"></span><p><strong>Node, a dot,</strong> marks one thing to notice: the dot before a release on a link preview, over the version on a release card, on Release cards in the navigation when a release is new.</p></li>
  <li><span class="brand-mark brand-mark--edge" aria-hidden="true"></span><p><strong>Edge, a line,</strong> marks a relation or a direction: the A's amber stroke, the one amber stroke in each cover's art, the hairline under a heading.</p></li>
</ul>
```

Each composition holds one amber node or one amber edge; everything else is ink, cobalt, and
navy.

## Colors

```{raw} html
:file: ../_generated/brand-colors.html
```

## Type

- **Unbounded** sets the wordmark, and nothing else; the wordmark files are outlines, so no font
  is needed to use them.
- **Inter** sets everything else: text, interface, and titles over art.

Both are free under the SIL Open Font License, on [Google Fonts](https://fonts.google.com/).

## Using the logo

Do:

- Use the files as they are, with room around them.
- Put the ink version on light grounds and the pale version on dark grounds.
- Use the icon or the avatar where the wordmark would be too small to read.

Don't:

- Recolor, stretch, outline, or add effects to the wordmark or the icon.
- Split the A, separate its strokes, or drop its crossbar.
- Retype "Andrey" in a font in place of the wordmark.
- Use the marks in a way that suggests Andrey endorses you or your product.

The Apache License 2.0 covers Andrey's code, not its name or marks. Questions:
[open an issue](https://github.com/Abel-ai-lab/andrey/issues/new).
