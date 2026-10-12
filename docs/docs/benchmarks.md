---
name: benchmarks
description: Andrey's release cards, one per release - its benchmarks, the methods it adds or promotes, their known limits, and how to rerun the measurements.
meta:
  type: index
html_theme.sidebar_secondary.remove: true
---

# Release cards

Every release comes with a post that announces it and a release card that keeps the record: the
benchmarks first, then the methods the release adds or promotes, their known limits, and how to
rerun the measurements. A card keeps the numbers its release published.

% The latest release: its cover, then its TL;DR. The next release replaces this block, and this
% release moves to the top of "Earlier releases" (docs/DEVELOPMENT.md, "The release cards").
:::::{div} release-latest

:::{card}
:link: benchmarks/0.2.0
:link-type: doc
:link-alt: The 0.2.0 release card: hidden common causes
:class-card: release-cover
:shadow: none
:margin: 0

Release card

Hidden common causes

0.2.0
:::

::::{div} release-tldr

<p class="release-status"><span class="release-tag">Latest</span>
<span class="release-badge">Pre-release</span> Measured 11 October 2026</p>

RCD and BottomUpParceLiNGAM, two methods for data with hidden common causes, compared with lingam,
which implements both. Both are supported from 0.2.0.

:::{div} release-stats

- **25 of 25** datasets with the same graph as lingam
- **1.7-2.9x** RCD speedup over lingam
- **2.5-10.8x** BottomUpParceLiNGAM speedup over lingam
:::

:::{div} release-limit
**Limit** BottomUpParceLiNGAM's directed edges have more errors than the empty graph's from 8
observed variables, in both packages.
:::

:::{div} release-sections

- [Benchmarks](benchmarks/0.2.0.md#summary)
- [Methods](benchmarks/0.2.0.md#hidden-confounder-methods)
- [Known limits](benchmarks/0.2.0.md#known-limits)
- [Reproduce](benchmarks/0.2.0.md#reproduce)
:::

:::{div} release-actions
[Read the release card](benchmarks/0.2.0.md)
<span class="release-pending">Post: with the release</span>
:::
::::
:::::

## Earlier releases

::::{div} release-row

:::{card}
:link: benchmarks/0.1.0
:link-type: doc
:link-alt: The 0.1.0 release card: the alpha benchmarks
:class-card: release-tile
:shadow: none
:margin: 0

0.1.0
:::

:::{div} release-body
<h3><a href="benchmarks/0.1.0.html">The alpha benchmarks</a></h3>

PC, FCI, GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM, compared with causal-learn, gCastle, and
lingam.

- **7** methods
- PC **941x** faster than causal-learn at 400 variables
:::

:::{div} release-meta
Released 8 October 2026

- [Release card<span class="visually-hidden"> 0.1.0</span>](benchmarks/0.1.0.md)
- [Post: Introducing Andrey](../blog/introducing-andrey.md)
:::
::::

The homepage's speed claim and charts come from the 0.1.0 release card.

```{toctree}
:hidden:

0.2.0: hidden common causes <benchmarks/0.2.0>
0.1.0: the alpha benchmarks <benchmarks/0.1.0>
```
