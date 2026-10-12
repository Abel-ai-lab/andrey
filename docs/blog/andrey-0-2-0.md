---
name: andrey-0-2-0
description: "Andrey's six methods for data with hidden common causes, with RCD and BottomUpParceLiNGAM new and supported, and small fits several times faster."
meta:
  type: post
blogpost: true
date: 2026-10-16
author: Shu Wan
version: 0.2.0
cover: iceberg
---

# Hidden common causes

This release is about data with hidden common causes. It adds RCD and BottomUpParceLiNGAM as
supported methods and CAM-UV as an experimental one, which brings Andrey to six methods for this
kind of data. Below, we run five of them on one small dataset and give their benchmark
results. Small fits also run several times faster.

## Why hidden causes matter

Real data often leave out variables that matter. Many causal discovery methods, PC, GES, and
DirectLiNGAM among them, assume causal sufficiency: every common cause of two measured variables
is measured too, which often does not hold. When two variables share a cause you did not measure,
they depend on each other without an edge between them, and a method that assumes sufficiency can
draw one. Methods for hidden common causes mark such a pair, or leave its ends open, instead.

## Six methods

Supported methods are benchmarked against another package and checked against known graphs before
each release. Experimental methods pass recovery tests but have no published benchmark yet, and
warn on their first call. RCD, BottomUpParceLiNGAM, and CAM-UV are new in this release.

| Method | Status | Assumes |
|---|---|---|
| [FCI](../docs/code/generated/andrey.fci.rst) | Supported | Linear relations with roughly Gaussian noise, through its Fisher-z test |
| [GFCI](../docs/code/generated/andrey.gfci.rst) | Experimental | Linear relations with Gaussian noise, through its BIC score and Fisher-z test |
| [RCD](../docs/code/generated/andrey.rcd.rst) | Supported | Linear relations with non-Gaussian noise |
| [BottomUpParceLiNGAM](../docs/code/generated/andrey.bottom_up_parce_lingam.rst) | Supported | Linear relations with non-Gaussian noise |
| [CAM-UV](../docs/code/generated/andrey.camuv.rst) | Experimental | Each variable a sum of nonlinear functions of its causes, plus noise |
| [GIN](../docs/code/generated/andrey.gin.rst) | Experimental | Linear relations with non-Gaussian noise; the measured variables are indicators of a few hidden ones |

## Reading a PAG

All but GIN return a PAG, a partial ancestral graph over the variables you measured. Each end of
an edge carries a mark: a tail, an arrowhead, or a circle, which leaves that end open.

- `x -> y`: `x` causes `y`.
- `x <-> y`: a hidden variable links `x` and `y`, such as a common cause.
- `x o-> y`: `y` does not cause `x`; whether `x` causes `y` is open.
- `x o-o y`: both ends are open.

FCI and GFCI mark an end with a tail or an arrowhead only when every graph with the data's
independences agrees on it. RCD, BottomUpParceLiNGAM, and CAM-UV use the same marks for what they
find: `->` for a direct cause, `<->` for a pair a hidden variable links, and, in
BottomUpParceLiNGAM, `o-o` between the variables it cannot order.
[Graph types](../docs/guides/graph-types.md#pag-uncertainty-with-hidden-variables) has the
details.

## One dataset, five methods

Here `h` drives `x0` and `x1` but is not in the data, `x0` causes `x3`, and `x1` causes `x2`. The
noise is uniform, so it is not Gaussian. GFCI and CAM-UV warn on their first call that they are
experimental.

```{literalinclude} 0.2.0/example.py
:language: python
:caption: example.py
```

```{literalinclude} 0.2.0/example-output.txt
:language: text
:caption: example.py
```

- RCD and CAM-UV return the true graph: `x0 <-> x1` for the pair `h` links, and both direct
  edges.
- BottomUpParceLiNGAM finds both direct edges. It cannot order `x0` and `x1`, which share `h`, so
  it leaves them in one block, joined by `o-o`; `causal_order` nests the block as `[0, 1]`.
- FCI and GFCI find the three edges and leave every end a circle. They use only conditional
  independences, and these four variables have the same independences as a graph with no
  collider and no hidden cause, such as `x3 <- x0 -> x1 -> x2`, so no mark can be settled. Given
  an observed cause of `x0` and another of `x1`, FCI can mark the pair `x0 <-> x1`.

RCD, BottomUpParceLiNGAM, and CAM-UV also test whether regression residuals are independent,
which, with non-Gaussian noise, settles the directions. RCD and BottomUpParceLiNGAM estimate the
edge weights too: 0.598 and 0.61, against the true 0.6. For comparison, DirectLiNGAM, which
assumes every common cause is measured, returns `x0 -> x1` on the same data, an edge the true
graph does not have.

GIN answers a different question. It takes data whose measured variables are noisy indicators of
a few hidden variables, groups the indicators by hidden variable, and orders the hidden variables.
Its result is a DAG with the hidden variables as nodes; its
[API page](../docs/code/generated/andrey.gin.rst) has an example.

## Benchmarks

We compared the three supported methods with packages that implement them.

| Method | Compared with | Variables | Andrey's speedup | Same graph |
|---|---|---|---|---|
| RCD | lingam | 4-24 observed | 1.7-2.9x | 25 of 25 datasets |
| BottomUpParceLiNGAM | lingam | 4-24 observed | 2.5-10.8x | 25 of 25 datasets |
| FCI | causal-learn | 20-800, hidden ones included | 3.4-16.5x at 50 to 800; 0.62x at 20 | 13 of 18 datasets |

The [0.2.0 release card](../docs/benchmarks/0.2.0.md) measures RCD and BottomUpParceLiNGAM on
linear data with uniform noise, one node in five hidden, and 2,000 rows, on 2 cores. The same
graph is not always an accurate one: from 8 observed variables, BottomUpParceLiNGAM's directed
edges have more errors than the empty graph's, in both packages.

FCI's numbers come from the [0.1.0 release card](../docs/benchmarks/0.1.0.md): linear-Gaussian
data with one variable in ten hidden and ten rows per variable, on 16 cores. At 20 variables,
where Andrey is slower, both packages take under 0.1 s. Their median endpoint errors are within
two of each other, and at 800 variables both are above the empty graph's.

GFCI, CAM-UV, and GIN are not benchmarked yet. Speed and accuracy depend on the data, the
settings, and the hardware.

## Other highlights

- **Faster small fits.** PC, FCI, GES, and hill climbing now find the BLAS libraries once per
  process instead of on every fit. At 10 variables, PC and FCI take about a twelfth of the time
  they took before, hill climbing an eighth, and GES two fifths. Larger fits are no slower, and
  the graphs are unchanged.
- **Release cards.** The [Release cards](../docs/benchmarks.md) page holds one card per release:
  its benchmarks, the methods it adds or promotes, their known limits, and how to rerun them.
- **Limitations.** Every method's docstring has a Limitations paragraph: where it was measured to
  work, or that it is not benchmarked yet.
- **Checks before the fit.** Every method checks your data and settings before it fits. Data with
  `NaN` or infinity, a constant column, or fewer than 2 rows raises a `ValueError` that names the
  argument, and so does a setting out of its range.

The [changelog](../docs/changelog.md) lists every change.

## Try it

```{code-block} shell
:class: window

pip install -U andrey-core
```

Two new notebooks fit [RCD](../docs/examples/rcd) and
[BottomUpParceLiNGAM](../docs/examples/bottom_up_parce_lingam) on the example's model, read the
edge weights, and check the result over ten datasets. Each opens in Colab. Then try the methods on
your own data.

## Help us build it

Andrey is in alpha, and the API may change between releases. When you try it on your own data,
[tell us what breaks](https://github.com/Abel-ai-lab/andrey/issues/new) or which method you need.
We'd welcome a pull request too. The
[contributing guide](https://github.com/Abel-ai-lab/andrey/blob/main/CONTRIBUTING.md) explains how
to set up.

---

- [Getting started](../docs/guides/getting-started.md)
- [Examples](../docs/examples/index.md)
- [Release cards](../docs/benchmarks.md)
- [Changelog](../docs/changelog.md)
- [GitHub](https://github.com/Abel-ai-lab/andrey)
