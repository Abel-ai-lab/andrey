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

A hidden common cause, or latent confounder, is an unmeasured variable that affects two measured
variables. This release adds RCD and BottomUpParceLiNGAM as supported methods and CAM-UV as an
experimental one, bringing Andrey to six methods for data with hidden common causes. Below, we
run five of them on one small dataset and report benchmark results. Other updates make small
fits several times faster.

## Why hidden causes matter

Many causal discovery methods, including PC, GES, and DirectLiNGAM, assume causal sufficiency:
every common cause of two measured variables is measured too. Real data often violate this
assumption. A hidden common cause can make two variables dependent even without an edge between
them. A method that assumes causal sufficiency can then draw an edge that is not there. Methods
that allow hidden common causes can mark the pair or leave the edge's ends unresolved.

## Six methods

We benchmark each supported method against another package and check its results on known graphs
before each release. Experimental methods pass recovery tests but have no published
benchmark yet. They warn on their first call. RCD, BottomUpParceLiNGAM, and CAM-UV are new in
this release.

| Method | Status | Assumes |
|---|---|---|
| [FCI](../docs/code/generated/andrey.fci.rst) | Supported | Linear relations with roughly Gaussian noise, through its Fisher-z test |
| [GFCI](../docs/code/generated/andrey.gfci.rst) | Experimental | Linear relations with Gaussian noise, through its BIC score and Fisher-z test |
| [RCD](../docs/code/generated/andrey.rcd.rst) | Supported | Linear relations with non-Gaussian noise |
| [BottomUpParceLiNGAM](../docs/code/generated/andrey.bottom_up_parce_lingam.rst) | Supported | Linear relations with non-Gaussian noise |
| [CAM-UV](../docs/code/generated/andrey.camuv.rst) | Experimental | Each variable a sum of nonlinear functions of its causes, plus noise |
| [GIN](../docs/code/generated/andrey.gin.rst) | Experimental | Linear relations with non-Gaussian noise; the measured variables are indicators of a few hidden ones |

## Reading a PAG

All but GIN return a partial ancestral graph (PAG) over the variables you measured. Each end of
an edge carries a tail, an arrowhead, or a circle. A circle means that end is unresolved.

- `x -> y`: `x` causes `y`.
- `x <-> y`: a hidden variable links `x` and `y`, such as a common cause.
- `x o-> y`: `y` does not cause `x`; whether `x` causes `y` is unresolved.
- `x o-o y`: both ends are unresolved.

FCI and GFCI mark an end with a tail or an arrowhead only when every graph consistent with the
data's conditional independences agrees on that mark. RCD, BottomUpParceLiNGAM, and CAM-UV use
the same marks for their findings: `->` for a direct cause and `<->` for a pair linked by a hidden
variable. BottomUpParceLiNGAM also uses `o-o` between variables it cannot order.
See [Graph types](../docs/guides/graph-types.md#pag-uncertainty-with-hidden-variables) for details.

## One dataset, five methods

Here `h` causes `x0` and `x1` but is not in the data. Also, `x0` causes `x3`, and `x1` causes
`x2`. The noise is uniform, so it is not Gaussian. GFCI and CAM-UV issue the experimental warning
on their first call.

```{literalinclude} 0.2.0/example.py
:language: python
:caption: example.py
```

```{literalinclude} 0.2.0/example-output.txt
:language: text
:caption: example.py
```

- RCD and CAM-UV return the true graph: `x0 <-> x1` for the pair linked by `h`, and both direct
  edges.
- BottomUpParceLiNGAM finds both direct edges. It cannot order `x0` and `x1`, which share `h`,
  so it groups them in one block joined by `o-o`. In `causal_order`, this block appears as the
  nested list `[0, 1]`.
- FCI and GFCI find the three edges but leave every end a circle. Their orientations rely on
  conditional independences. These four variables have the same independences as a graph with
  no hidden cause and no collider (a variable that two arrows point into), such as
  `x3 <- x0 -> x1 -> x2`, so the independences cannot settle any mark. Given an observed cause
  of `x0` and another of `x1`, FCI can mark the pair `x0 <-> x1`.

RCD, BottomUpParceLiNGAM, and CAM-UV also test whether regression residuals are independent.
With non-Gaussian noise, this test settles the directions in this example. RCD and
BottomUpParceLiNGAM also estimate the edge weights: 0.598 and 0.61, compared with the true 0.6.
DirectLiNGAM assumes causal sufficiency. On the same data, it returns `x0 -> x1` (the last result
above), an edge absent from the true graph.

GIN answers a different question. If your measured variables are noisy indicators of a few
hidden variables, GIN groups the indicators by hidden variable and orders the hidden variables.
It returns a directed acyclic graph (DAG) whose nodes are the hidden variables. See its
[API page](../docs/code/generated/andrey.gin.rst) for an example.

## Benchmarks

We compared the three supported methods with packages that implement them.

| Method | Compared with | Variables | Andrey's speedup | Same graph |
|---|---|---|---|---|
| RCD | lingam | 4-24 observed | 1.7-2.9x | 25 of 25 datasets |
| BottomUpParceLiNGAM | lingam | 4-24 observed | 2.5-10.8x | 25 of 25 datasets |
| FCI | causal-learn | 20-800, hidden ones included | 3.4-16.5x at 50 to 800; 0.62x at 20 | 13 of 18 datasets |

The [0.2.0 release card](../docs/benchmarks/0.2.0.md) reports measurements for RCD and
BottomUpParceLiNGAM on linear data with uniform noise, one node in five hidden, and 2,000 rows,
on 2 cores. Agreement between packages does not establish accuracy: from 8 observed variables,
BottomUpParceLiNGAM's directed edges have more errors than the empty graph's, in both packages.

FCI's numbers come from the [0.1.0 release card](../docs/benchmarks/0.1.0.md): linear-Gaussian
data with one variable in ten hidden and ten rows per variable, on 16 cores. At 20 variables,
Andrey is slower, but both packages take under 0.1 s. Their median endpoint errors are within
two of each other. At 800 variables, both have more endpoint errors than the empty graph.

GFCI, CAM-UV, and GIN have no published benchmarks yet. Speed and accuracy depend on your data,
settings, and hardware.

## Other highlights

- **Faster small fits.** PC, FCI, GES, and hill climbing now find the Basic Linear Algebra
  Subprograms (BLAS) libraries once per process instead of on every fit. At 10 variables, PC
  and FCI take about a twelfth of the time they took before, hill climbing an eighth, and GES
  two fifths. Larger fits are no slower, and the graphs are unchanged.
- **Release cards.** The [Release cards](../docs/benchmarks.md) page has one card per release:
  its benchmarks, the methods it adds or promotes, their known limits, and how to rerun the
  benchmarks.
- **Limitations.** Every method's docstring has a Limitations paragraph stating where the method
  was measured to work or that it is not benchmarked yet.
- **Checks before the fit.** Every method checks your data and settings before fitting. Data
  with `NaN` or infinity, a constant column, or fewer than 2 rows raises a `ValueError` that
  names the argument. An out-of-range setting raises the same error.

The [changelog](../docs/changelog.md) lists every change.

## Try it

```{code-block} shell
:class: window

pip install -U andrey-core
```

Two new notebooks fit [RCD](../docs/examples/rcd) and
[BottomUpParceLiNGAM](../docs/examples/bottom_up_parce_lingam) on the example's model, read the
edge weights, and check the result over ten datasets. You can open each notebook in Colab, then
try the methods on your own data.

## Help us build it

Andrey is in alpha, and the API may change between releases. When you try it on your own data,
[tell us what breaks](https://github.com/Abel-ai-lab/andrey/issues/new) or which method you need.
We also welcome pull requests. The
[contributing guide](https://github.com/Abel-ai-lab/andrey/blob/main/CONTRIBUTING.md) explains how
to set up.

---

- [Getting started](../docs/guides/getting-started.md)
- [Examples](../docs/examples/index.md)
- [Release cards](../docs/benchmarks.md)
- [Changelog](../docs/changelog.md)
- [GitHub](https://github.com/Abel-ai-lab/andrey)
