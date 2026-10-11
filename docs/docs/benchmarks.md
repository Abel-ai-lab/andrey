---
name: benchmarks
description: Andrey's benchmark reports, one per release - timings, accuracy, and how to rerun each.
meta:
  type: index
---

# Benchmarks

Each release that brings new benchmark evidence has its own report: the campaigns it ran, the
packages it compared, how to read the results, and how to rerun them. A report keeps the numbers
its release published, and each comes with a summary file of those numbers. A later release adds
a report and leaves the earlier ones as they were. Newest first:

| Release | Report | Methods |
|---|---|---|
| 0.2.0 | [Hidden common causes](benchmarks/0.2.0.md) | RCD and BottomUpParceLiNGAM, compared with lingam |
| 0.1.0 | [The alpha benchmarks](benchmarks/0.1.0.md) | PC, FCI, GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM, compared with causal-learn, gCastle, and lingam |

The homepage's speed claim and charts come from the 0.1.0 report.

```{toctree}
:hidden:

0.2.0: hidden common causes <benchmarks/0.2.0>
0.1.0: the alpha benchmarks <benchmarks/0.1.0>
```
