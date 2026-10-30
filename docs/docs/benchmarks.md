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
:link: benchmarks/0.12.0
:link-type: doc
:link-alt: The 0.12.0 release card: calibration
:class-card: release-cover release-art release-art-0-12-0
:shadow: none
:margin: 0

Release card

Calibration

0.12.0
:::

::::{div} release-tldr

0.12.0 is here: calibration. The card measures the three methods it makes supported, Inter-IAMB,
MMHC, and RSMAX2, against bnlearn on the CPDAG, the skeleton, and the moral graph each returns.

:::{div} release-stats

- **3 methods** newly supported: Inter-IAMB, MMHC, and RSMAX2
- **0 of 72** comparisons where bnlearn makes significantly fewer errors
- **7.8x** RSMAX2's median speed over bnlearn's, and 6.6x for MMHC
:::

:::{div} release-actions
[Read the release card](benchmarks/0.12.0.md)
:::
::::
:::::

## Earlier releases

::::{div} release-row

:::{card}
:link: benchmarks/0.11.0
:link-type: doc
:link-alt: The 0.11.0 release card: ready for real data
:class-card: release-tile release-art-0-11-0
:shadow: none
:margin: 0

0.11.0
:::

:::{div} release-body
<h3>Ready for real data</h3>

The extended BIC as the default penalty of ten score searches, test-wise deletion under missing
values, and how often the bootstrap's frequent edges are true.

- GES's median errors from **87 to 25** with the extended BIC, over 299 simulated datasets
- **11 methods** accept missing values
:::

:::{div} release-meta
Released October 29, 2026

- [Release card<span class="visually-hidden"> 0.11.0</span>](benchmarks/0.11.0.md)
- [Post: Ready for real data](../blog/20261029-andrey-0-11-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.10.0
:link-type: doc
:link-alt: The 0.10.0 release card: the boundary
:class-card: release-tile release-art-0-10-0
:shadow: none
:margin: 0

0.10.0
:::

:::{div} release-body
<h3>The boundary</h3>

How far each method goes: for 26 methods, the problems each finishes within the time and memory
caps with fewer errors than the empty graph, by variables, samples, and density.

- BOSS more accurate than the empty graph up to **1,600 variables**, at 2,000 samples
- GRaSP up to **800 variables**, PC up to **400**
:::

:::{div} release-meta
Released October 27, 2026

- [Release card<span class="visually-hidden"> 0.10.0</span>](benchmarks/0.10.0.md)
- [Post: The boundary](../blog/20261027-andrey-0-10-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.9.0
:link-type: doc
:link-alt: The 0.9.0 release card: acceleration
:class-card: release-tile release-art-0-9-0
:shadow: none
:margin: 0

0.9.0
:::

:::{div} release-body
<h3>Acceleration</h3>

The same answers as 0.8.0, in less time, with each sped-up method checked for the same output, and
GES compared with pcalg.

- BOSS up to **8.19x** faster than 0.8.0 with one worker
- GES **more than 5x** faster than pcalg at 400 variables
:::

:::{div} release-meta
Released October 23, 2026

- [Release card<span class="visually-hidden"> 0.9.0</span>](benchmarks/0.9.0.md)
- [Post: Acceleration](../blog/20261023-andrey-0-9-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.8.0
:link-type: doc
:link-alt: The 0.8.0 release card: nonlinear tests
:class-card: release-tile release-art-0-8-0
:shadow: none
:margin: 0

0.8.0
:::

:::{div} release-body
<h3>Nonlinear tests</h3>

PC and FCI with four tests for nonlinear dependence, KCI, RCoT, CMIknn, and GPDC, and BOSS and
GRaSP on categorical data, compared with causal-learn, pcalg, and tigramite.

- **4** nonlinear tests supported
- PC with RCoT **more than 10x** faster than causal-learn at 50 variables
:::

:::{div} release-meta
Released October 23, 2026

- [Release card<span class="visually-hidden"> 0.8.0</span>](benchmarks/0.8.0.md)
- [Post: Nonlinear dependence](../blog/20261023-andrey-0-8-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.7.0
:link-type: doc
:link-alt: The 0.7.0 release card: permutation invariance
:class-card: release-tile release-art-0-7-0
:shadow: none
:margin: 0

0.7.0
:::

:::{div} release-body
<h3>Permutation invariance</h3>

GES and seven newly supported methods, from GIES to CALM, on data whose columns are in a random
order, compared with causal-learn, lingam, pcalg, and Tetrad.

- **7** methods supported
- GES **more than 60x** faster than causal-learn from 50 to 150 variables
:::

:::{div} release-meta
Released October 19, 2026

- [Release card<span class="visually-hidden"> 0.7.0</span>](benchmarks/0.7.0.md)
- [Post: Permutation invariance](../blog/20261019-andrey-0-7-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.6.0
:link-type: doc
:link-alt: The 0.6.0 release card: mixed and categorical data
:class-card: release-tile release-art-0-6-0
:shadow: none
:margin: 0

0.6.0
:::

:::{div} release-body
<h3>Mixed and categorical data</h3>

PC, GES, hill climbing, and seven more methods for categorical columns and for tables that mix them
with numbers, compared with causal-learn, pgmpy, bnlearn, and Tetrad.

- **10** methods
- GES **more than 50x** faster than causal-learn at every size
:::

:::{div} release-meta
Released October 17, 2026

- [Release card<span class="visually-hidden"> 0.6.0</span>](benchmarks/0.6.0.md)
- [Post: Mixed and categorical data](../blog/20261017-andrey-0-6-0.md)
:::
::::

::::{div} release-row

:::{card}
:link: benchmarks/0.5.0
:link-type: doc
:link-alt: The 0.5.0 release card: temporal causal discovery
:class-card: release-tile release-art-0-5-0
:shadow: none
:margin: 0

0.5.0
:::

:::{div} release-body
<h3>Temporal causal discovery</h3>

PCMCI, PCMCI+, LPCMCI, and three more methods for data ordered in time, compared with tigramite,
causal-learn, and lingam.

- **6** methods
- PCMCI+ **more than 50x** faster than tigramite at every size
:::

:::{div} release-meta
Released October 16, 2026

- [Release card<span class="visually-hidden"> 0.5.0</span>](benchmarks/0.5.0.md)
- [Post: Temporal causal discovery](../blog/20261016-andrey-0-5-0.md)
:::
::::

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

0.12.0: calibration <benchmarks/0.12.0>
0.11.0: ready for real data <benchmarks/0.11.0>
0.10.0: the boundary <benchmarks/0.10.0>
0.9.0: acceleration <benchmarks/0.9.0>
0.8.0: nonlinear tests <benchmarks/0.8.0>
0.7.0: permutation invariance <benchmarks/0.7.0>
0.6.0: mixed and categorical data <benchmarks/0.6.0>
0.5.0: temporal causal discovery <benchmarks/0.5.0>
0.4.0: pairwise direction <benchmarks/0.4.0>
0.3.0: score-based search <benchmarks/0.3.0>
0.2.0: hidden common causes <benchmarks/0.2.0>
0.1.0: the alpha release <benchmarks/0.1.0>
```
