---
name: andrey-0-2-0
description: "Andrey 0.2.0 supports two methods for data with hidden common causes, and runs small fits several times faster."
meta:
  type: post
blogpost: true
date: 2026-10-16
author: Shu Wan
version: 0.2.0
cover: graph
---

# Andrey 0.2.0: hidden common causes, and faster small fits

Andrey 0.2.0 supports two methods for data with hidden common causes, RCD and BottomUpParceLiNGAM.
It also runs small fits several times faster, with the same graphs as before.

## Hidden common causes

Most causal discovery methods assume that every cause is in your data. When two variables share a
cause you did not measure, they depend on each other with no edge between them, and such a method
can draw a wrong edge. RCD and BottomUpParceLiNGAM allow for these hidden common causes. Both assume
linear relations and non-Gaussian noise, and both return a PAG over the variables you measured:

- `andrey.rcd` finds each variable's direct causes and marks a pair that shares a hidden cause
  `<->`.
- `andrey.bottom_up_parce_lingam` orders the variables from the last effect up, and leaves the
  variables a hidden cause keeps it from ordering in one block, joined by `o-o`.

Use them when you suspect a hidden cause and your noise is not Gaussian. RCD needs many rows: a
weak hidden cause leaves only a small trace in the residuals it tests. Without hidden causes,
DirectLiNGAM returns a full DAG; with Gaussian noise, FCI returns the PAG that the data's
independences support.

Here `h` drives `x0` and `x1`, and the data leave it out:

```{literalinclude} 0.2.0/example.py
:language: python
:caption: example.py
```

```{literalinclude} 0.2.0/example-output.txt
:language: text
:caption: example.py
```

RCD marks `x0 <-> x1`, the pair that `h` links, and finds both direct edges with their weights.

We compared both methods with lingam, which implements both, on 25 datasets with hidden common
causes: 4 to 24 observed variables and 2,000 rows each, on 2 cores. Both return the same graph as
lingam on 25 of 25 datasets. RCD runs 1.7-2.9x faster, and BottomUpParceLiNGAM 2.5-10.8x faster.

The same graph is not always an accurate one. From 8 observed variables, BottomUpParceLiNGAM's
directed edges have more errors than the empty graph's, in both packages. The
[0.2.0 release card](https://andrey.abel.ai/docs/benchmarks/0.2.0.html) gives the numbers for each
size.

A third method for hidden common causes, CAM-UV (`andrey.camuv`), allows nonlinear relations. It
ships experimental.

## Faster small fits

PC, FCI, GES, and hill climbing limit BLAS to one thread during a fit. Andrey now finds the BLAS
libraries once per process, instead of on every fit. At 10 variables, PC and FCI take about a
twelfth of the time they took before, hill climbing an eighth, and GES two fifths. Larger fits are
no slower, and the graphs are unchanged. You don't need to change any code.

## Also in 0.2.0

- Every method checks your data and settings before the fit. Data with `NaN` or infinity, a
  constant column, or fewer than 2 rows raises a `ValueError` that names the argument.
- The [benchmarks](../docs/benchmarks.md) page holds a release card for each release: its
  benchmarks, the methods it adds or promotes, their known limits, and how to rerun them.
- Every supported method's docstring says where it was measured to work.

The [changelog](../docs/changelog.md) lists every change.

## Try it

```{code-block} shell
:class: window

pip install -U andrey-core
```

Andrey is in alpha, and the API may change between releases. When you try it on your own data,
[tell us what breaks](https://github.com/Abel-ai-lab/andrey/issues/new) or which method you need.

---

- [Getting started](../docs/guides/getting-started.md)
- [Examples](../docs/examples/index.md)
- [Benchmarks](../docs/benchmarks.md)
- [Changelog](../docs/changelog.md)
- [GitHub](https://github.com/Abel-ai-lab/andrey)
