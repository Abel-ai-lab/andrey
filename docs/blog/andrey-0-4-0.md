---
name: andrey-0-4-0
description: "Pairwise direction in Andrey 0.4.0: ANM and PNL test which of two variables causes the other, and both are now supported."
meta:
  type: post
blogpost: true
date: 2026-10-13
author: Shu Wan
version: 0.4.0
cover: graph
---

% TODO(0.4.0): `date` is a placeholder for the release date, and `cover: graph` is the generic
% cover until the 0.4.0 art is approved.

# Pairwise direction

Does `x` cause `y`, or does `y` cause `x`? With only two variables, independence tests cannot
tell: both directions fit the data equally well. ANM and PNL answer from the shape of the relation
instead. This release adds ANM, and from 0.4.0 both are supported.

Below, we explain how two variables can reveal their direction. We run both tests on one pair, then
report benchmark results.

## Direction from two variables

PC and GES can orient an edge only through a third variable, such as a common effect. A pair has
none, so `x -> y` and `y -> x` explain the data equally well, and these methods leave the edge
undirected.

A model of the mechanism can break the tie. Suppose `y = f(x) + e`, where the noise `e` is
independent of `x`. Regress `y` on `x`, and the residual looks independent of `x`. Regress `x` on
`y`, and in general no residual is independent of `y`, unless `f` is linear and the noise is
Gaussian. So a test fits both directions and checks whether each residual is independent of the
proposed cause. It uses HSIC, a kernel test that detects nonlinear dependence, not only
correlation. The direction whose residual looks independent is the causal one.

The two methods fit different models:

- **ANM** (additive noise model) fits `y = f(x) + e` by Gaussian-process regression. The direction
  is identifiable when `f` is nonlinear or the noise is non-Gaussian.
- **PNL** (post-nonlinear model) fits `y = f2(f1(x) + e)`, where `f2` is invertible: an additive
  noise model seen through a distortion, such as a sensor that saturates. ANM is the case where
  `f2` changes nothing. PNL fits each transform as a sum of smooth basis functions.

## Methods

Before each release, we benchmark each supported method against another package and check its
results on known data. ANM is new in this release, and PNL was experimental until now.

| Method | Status | Model | Returns |
|---|---|---|---|
| [ANM](../docs/code/generated/andrey.anm.rst) | Supported | `y = f(x) + e` | A two-node DAG, both p-values, and both test statistics |
| [PNL](../docs/code/generated/andrey.pnl.rst) | Supported | `y = f2(f1(x) + e)` | A two-node DAG, both p-values, and both test statistics |

Choose a method for your data:

- **ANM** when the effect is a smooth function of the cause plus independent noise. Its regression
  time grows with the cube of the rows, so it suits up to a few thousand rows.
- **PNL** when the effect may pass through a distortion, or for more rows: in our benchmarks, its
  median fit took under 2 s at 5,000 rows on 4 cores.

Both assume that no hidden common cause links the pair. When one is plausible, use the methods in
the [0.2.0 post](andrey-0-2-0.md).

## Reading the result

Each test returns a two-node DAG and four values in `metadata`:

- `pval_forward` and `pval_backward`: the p-values of the independence tests for `x -> y` and
  `y -> x`. A large p-value means the residual looks independent of the proposed cause, so that
  direction fits.
- `stat_forward` and `stat_backward`: the two HSIC statistics, each divided by its mean under
  independence. Smaller means weaker dependence.

By default, the edge points in the direction with the larger p-value. On a large sample, both
p-values can round to zero; then the smaller statistic decides. This rule always names a direction,
even for unrelated variables. With a significance level such as `alpha=0.05`, a direction fits
when its p-value exceeds the level, and the test draws an edge only when exactly one direction
fits.

## Example

Here `x` causes `y`, which a saturating sensor reads: `y = tanh(x + e)`, with Laplace noise. This
is a post-nonlinear pair, and not an additive-noise one.

```{literalinclude} 0.4.0/example.py
:language: python
:caption: example.py
```

```{raw} html
:file: 0.4.0/pair.html
```

- ANM rejects both directions: both of its p-values are near zero, because the saturation is not
  additive noise. Its default rule still names a direction, here the reversed one, `y -> x`. With
  `alpha=0.05`, ANM draws no edge (the last result below).
- PNL models the saturation as its outer transform. Its p-value for `x -> y` is large and its
  p-value for `y -> x` is near zero, so it names the true direction.

The script prints each result in full:

```{literalinclude} 0.4.0/example-output.txt
:language: text
:caption: example.py
```

## Benchmarks

We compared both methods with causal-learn, a Python package that implements them, on simulated
cause-effect pairs: ANM on additive-noise pairs and PNL on post-nonlinear pairs.

| Method | Compared with | Andrey's speedup | Direction |
|---|---|---|---|
| ANM | causal-learn | about 2x | the same as causal-learn's on every pair |
| PNL | causal-learn | up to 1,470x | true on more pairs than causal-learn's |

The [0.4.0 release card](../docs/benchmarks/0.4.0.md) reports every measurement, with the sizes,
machine, and settings.

```{raw} html
:file: 0.4.0/speedup.html
```

ANM fits the same Gaussian process and the same independence test as causal-learn, so the two
return the same graph on every pair. PNL differs more. causal-learn trains neural networks for each
direction and compares means with a t-test, while Andrey fits smooth transforms and tests
independence. Andrey's PNL names the true direction on every pair from 1,000 rows; causal-learn's
names it on at most half the pairs at any size.

```{raw} html
:file: 0.4.0/accuracy.html
```

At 200 rows, both methods name the reversed direction on a quarter of the pairs. A direction test
names a direction on every pair, right or wrong, so check both p-values before you trust it.

Speed and accuracy depend on your data, settings, and hardware.

## Other highlights

- **Ties broken by the statistics.** On a large sample, both p-values can round to zero. PNL now
  names the direction with the smaller HSIC statistic, relative to its mean under independence,
  instead of drawing no edge, and ANM uses the same rule. Both statistics are in `metadata`.
- **ANM in any units.** ANM standardizes each column first, so its answer does not depend on either
  variable's units.

For more details, read the [changelog](../docs/changelog.md).

## Try it

```{code-block} shell
:class: window

pip install -U andrey-core
```

The [pairwise notebook](../docs/examples/pairwise) runs both tests on simulated pairs. You can open
it in Colab, then pass your own two-column table to `andrey.anm` or `andrey.pnl`.

## Help us build it

Andrey is in alpha, and the API may change between releases. When you try it on your own data,
[tell us what breaks](https://github.com/Abel-ai-lab/andrey/issues/new) or which method you need.
We also welcome pull requests. The
[contributing guide](https://github.com/Abel-ai-lab/andrey/blob/main/CONTRIBUTING.md) explains how
you can set up.

---

- [Getting started](../docs/guides/getting-started.md)
- [Examples](../docs/examples/index.md)
- [Release cards](../docs/benchmarks.md)
- [Changelog](../docs/changelog.md)
- [GitHub](https://github.com/Abel-ai-lab/andrey)
