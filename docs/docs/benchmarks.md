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
:link: benchmarks/0.3.0
:link-type: doc
:link-alt: The 0.3.0 release card: score-based search
:class-card: release-cover release-art release-art-0-3-0
:shadow: none
:margin: 0

Release card

Score-based search

0.3.0
:::

::::{div} release-tldr

0.3.0 is here: it adds three score-based searches, XGES, Tabu search, and hill climbing, to
Andrey's supported methods, now 12 in all. Tested against xges and bnlearn on the same data, they
return the same graphs on most datasets and run up to 906x faster.

:::{div} release-stats

- **40 of 40** datasets with the same graph as xges
- **Up to 140x** XGES speedup over xges
- **Up to 561x** Tabu search speedup over bnlearn
:::

:::{div} release-actions
[Read the release card](benchmarks/0.3.0.md)
[Read the post](../blog/20261012-andrey-0-3-0.md)
:::
::::
:::::

## Earlier releases

::::{div} release-row

:::{card}
:link: benchmarks/0.2.0
:link-type: doc
:link-alt: The 0.2.0 release card: hidden common causes
:class-card: release-tile
:shadow: none
:margin: 0

0.2.0
:::

:::{div} release-body
<h3>Hidden common causes</h3>

RCD and BottomUpParceLiNGAM, two methods for data with hidden common causes, compared with lingam.

- **2** methods
- BottomUpParceLiNGAM **10.8x** faster than lingam at 24 observed variables
:::

:::{div} release-meta
Released October 12, 2026

- [Release card<span class="visually-hidden"> 0.2.0</span>](benchmarks/0.2.0.md)
- [Post: Hidden common causes](../blog/20261012-andrey-0-2-0.md)
:::
::::

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
<h3>The alpha benchmarks</h3>

PC, FCI, GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM, compared with causal-learn, gCastle, and
lingam.

- **7** methods
- PC **941x** faster than causal-learn at 400 variables
:::

:::{div} release-meta
Released October 8, 2026

- [Release card<span class="visually-hidden"> 0.1.0</span>](benchmarks/0.1.0.md)
- [Post: Introducing Andrey](../blog/20261008-introducing-andrey.md)
:::
::::

The homepage's speed claim and charts come from these release cards.

```{toctree}
:hidden:

0.3.0: score-based search <benchmarks/0.3.0>
0.2.0: hidden common causes <benchmarks/0.2.0>
0.1.0: the alpha benchmarks <benchmarks/0.1.0>
```
