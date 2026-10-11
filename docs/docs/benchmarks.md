---
name: benchmarks
description: Andrey's benchmark reports, one per release - timings, accuracy, and how to rerun each.
meta:
  type: index
html_theme.sidebar_secondary.remove: true
---

# Benchmarks

Each release that brings new benchmark evidence has its own report: the campaigns it ran, the
packages it compared, how to read the results, and how to rerun them. A report keeps the numbers
its release published, and each comes with a summary file of those numbers. A later release adds
a report and leaves the earlier ones as they were.

::::{grid} 1 1 2 2
:gutter: 3
:class-container: benchmark-reports

:::{grid-item-card} Hidden common causes
:link: benchmarks/0.2.0
:link-type: doc
:link-alt: The 0.2.0 benchmark report: hidden common causes
:class-header: report-cover

Benchmark report

0.2.0
^^^
RCD and BottomUpParceLiNGAM, compared with lingam.

{bdg-secondary-line}`Measured 11 October 2026`
:::

:::{grid-item-card} The alpha benchmarks
:link: benchmarks/0.1.0
:link-type: doc
:link-alt: The 0.1.0 benchmark report: the alpha benchmarks
:class-header: report-cover

Benchmark report

0.1.0
^^^
PC, FCI, GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM, compared with causal-learn, gCastle,
and lingam.

{bdg-secondary-line}`Measured 21 August to 15 September 2026`
:::
::::

The homepage's speed claim and charts come from the 0.1.0 report.

```{toctree}
:hidden:

0.2.0: hidden common causes <benchmarks/0.2.0>
0.1.0: the alpha benchmarks <benchmarks/0.1.0>
```
