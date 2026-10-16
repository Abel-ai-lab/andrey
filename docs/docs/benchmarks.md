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
:link: benchmarks/0.5.0
:link-type: doc
:link-alt: The 0.5.0 release card: time series
:class-card: release-cover release-art release-art-0-5-0
:shadow: none
:margin: 0

Release card

Time series

0.5.0
:::

::::{div} release-tldr

0.5.0 is here: PCMCI, PCMCI+, LPCMCI, lasso Granger causality, VARMA-LiNGAM, and LongitudinalLiNGAM,
methods for data ordered in time, are now supported. Tested against tigramite on the same series,
PCMCI+ returns the same graphs and runs more than 50x faster at every size, and more than 100x at 80
variables and 2,000 time steps.

:::{div} release-stats

- **110 of 110** datasets with the same PCMCI+ graph as tigramite
- **More than 50x** PCMCI+ speedup over tigramite at every size
- **More than 20x** PCMCI speedup over tigramite at 10 variables
:::

:::{div} release-actions
[Read the release card](benchmarks/0.5.0.md)
[Read the post](../blog/20261016-andrey-0-5-0.md)
:::
::::
:::::

## Earlier releases

::::{div} release-row

:::{card}
:link: benchmarks/0.4.0
:link-type: doc
:link-alt: The 0.4.0 release card: pairwise direction
:class-card: release-tile release-art-0-4-0
:shadow: none
:margin: 0

0.4.0
:::

:::{div} release-body
<h3>Pairwise direction</h3>

ANM and PNL, two tests of which of two variables causes the other, compared with causal-learn.

- **2** methods
- PNL **1,470x** faster than causal-learn at 200 rows
:::

:::{div} release-meta
Released October 15, 2026

- [Release card<span class="visually-hidden"> 0.4.0</span>](benchmarks/0.4.0.md)
- [Post: Pairwise direction](../blog/20261015-andrey-0-4-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.3.0
:link-type: doc
:link-alt: The 0.3.0 release card: score-based search
:class-card: release-tile release-art-0-3-0
:shadow: none
:margin: 0

0.3.0
:::

:::{div} release-body
<h3>Score-based search</h3>

XGES, compared with xges, and Tabu search and hill climbing, compared with bnlearn, on
linear-Gaussian data.

- **3** methods
- Hill climbing **906x** faster than bnlearn at 800 variables
:::

:::{div} release-meta
Released October 13, 2026

- [Release card<span class="visually-hidden"> 0.3.0</span>](benchmarks/0.3.0.md)
- [Post: Score-based search](../blog/20261012-andrey-0-3-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.2.0
:link-type: doc
:link-alt: The 0.2.0 release card: hidden common causes
:class-card: release-tile release-art-0-2-0
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
:link-alt: The 0.1.0 release card: the alpha release
:class-card: release-tile
:shadow: none
:margin: 0

0.1.0
:::

:::{div} release-body
<h3>The alpha release</h3>

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

0.5.0: time series <benchmarks/0.5.0>
0.4.0: pairwise direction <benchmarks/0.4.0>
0.3.0: score-based search <benchmarks/0.3.0>
0.2.0: hidden common causes <benchmarks/0.2.0>
0.1.0: the alpha release <benchmarks/0.1.0>
```
