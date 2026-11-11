---
name: examples-index
description: Example notebooks, each running one method or task on your machine or in Colab, and
  interactive demos that run in your browser.
meta:
  type: index
html_theme.sidebar_secondary.remove: true
---

# Examples

Each notebook runs one method or task, on your own machine or in Colab; each has **Open in Colab**
and **Download this notebook** links. The [interactive demos](#interactive-demos) run in your
browser instead. From a source checkout, start Jupyter with `uv run --with jupyter jupyter lab`. A
notebook that needs another package installs it in a cell with `uv pip install`.

To run one in Colab, choose **Open in Colab**, select a CPU runtime, and run the setup cell before
the example. The setup cell installs Andrey in the Colab runtime only; the speed notebook labels
that runtime in its output.

## Quick start

::::{grid} 1 2 2 3
:gutter: 3
:class-container: example-gallery

:::{grid-item-card} Quick start
:link: quick_start
:link-type: doc
:img-top: /_generated/thumbs/quick_start.svg
:img-alt: The graph PC learns from the Sachs protein measurements

Load the Sachs protein measurements, fit PC, GES, FCI, and DirectLiNGAM, read each result, and
draw each graph.

{bdg-secondary-line}`Notebook · Colab`
:::
::::

```{toctree}
:hidden:
:caption: Quick start

quick_start
```

## Methods

::::{grid} 1 2 2 3
:gutter: 3
:class-container: example-gallery

:::{grid-item-card} Constraint-based discovery
:link: constraint_based
:link-type: doc
:img-top: /_generated/thumbs/constraint_based.svg
:img-alt: The graph PC or FCI recovers in this notebook

PC's CPDAG and FCI's and GFCI's PAGs from conditional-independence tests; IAMB, Inter-IAMB, and
MMPC, which join each variable's local structure into a graph; CD-NOD and FCI-JCI on data from
several domains; and ICP's parents of one variable from several environments.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Score-based search
:link: score_search
:link-type: doc
:img-top: /_generated/thumbs/score_search.svg
:img-alt: The true graph of this notebook's data

XGES, hill climbing, tabu search, and exact search look for the graph with the best score, by
default the extended BIC; MMHC and RSMAX2 restrict the search with tests first; GIES also learns
from interventions, and GnIES from environments whose changes it is not told.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Continuous optimization
:link: calm
:link-type: doc
:img-top: /_generated/thumbs/calm.svg
:img-alt: The true graph of this notebook's data

CALM learns a weighted DAG by gradient descent on a penalized likelihood; it runs on torch.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Permutation-based search
:link: permutation_based
:link-type: doc
:img-top: /_generated/thumbs/permutation_based.svg
:img-alt: The graph BOSS or GRaSP recovers in this notebook

BOSS and GRaSP search over orderings of the variables for the best-scoring graph, and BOSS-FCI
and GRaSP-FCI start from that graph and allow hidden common causes; IGSP searches orderings too,
on interventional settings, one per environment, whose changed variables it is told, and UT-IGSP
on settings whose changed variables it is told in part or not at all.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Linear non-Gaussian models
:link: lingam
:link-type: doc
:img-top: /_generated/thumbs/lingam.svg
:img-alt: The DAG DirectLiNGAM or ICA-LiNGAM recovers in this notebook

DirectLiNGAM and ICA-LiNGAM orient every edge when the noise is non-Gaussian; MultiGroupDirectLiNGAM
fits several datasets, and GIN finds latent variables.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Nonlinear additive-noise models
:link: functional
:link-type: doc
:img-top: /_generated/thumbs/functional.svg
:img-alt: The DAG CAM recovers in this notebook

CAM, RESIT, SCORE, and NoGAM learn a DAG, with every edge oriented, when each variable is a
nonlinear function of its parents plus independent noise; CAM-UV allows hidden causes and marks the
pairs they join.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Time series
:link: time_series
:link-type: doc
:img-top: /_generated/thumbs/time_series.svg
:img-alt: The true graph of this notebook's series, its lags merged

PCMCI, PCMCI+, LPCMCI, Granger, VARMA-LiNGAM, DYNOTEARS, Longitudinal LiNGAM, and J-PCMCI+ on data
over time.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Pairs that share a hidden cause
:link: rcd
:link-type: doc
:img-top: /_generated/thumbs/rcd.svg
:img-alt: The true graph of this notebook's data, with one pair that shares a hidden cause

RCD finds the direct causes and marks the pairs that share a hidden cause.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Causal order with hidden causes
:link: bottom_up_parce_lingam
:link-type: doc
:img-top: /_generated/thumbs/bottom_up_parce_lingam.svg
:img-alt: The true graph of this notebook's data, with one pair that shares a hidden cause

BottomUpParceLiNGAM orders the variables and leaves those a hidden cause confounds unordered.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Pairwise direction
:link: pairwise
:link-type: doc
:img-top: /_generated/thumbs/pairwise.svg
:img-alt: The true graph of this notebook's pair

ANM, PNL, and LOCI test which of two variables causes the other.

{bdg-secondary-line}`Notebook · Colab`
:::
::::

```{toctree}
:hidden:
:caption: Methods

constraint_based
score_search
calm
permutation_based
lingam
functional
time_series
rcd
bottom_up_parce_lingam
pairwise
```

## Advanced

::::{grid} 1 2 2 3
:gutter: 3
:class-container: example-gallery

:::{grid-item-card} Bootstrap: how stable is each edge?
:link: bootstrap
:link-type: doc
:img-top: /_generated/thumbs/bootstrap.svg
:img-alt: The averaged graph of PC over 50 subsamples of data on the ALARM network

Refit a method on resamples of the rows: edge frequencies, the averaged graph, and which edges to
trust.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Markov check: do the data contradict a graph?
:link: markov_check
:link-type: doc
:img-top: /_generated/thumbs/markov_check.svg
:img-alt: The sparsest GES graph on ALARM data that the Markov check does not reject

Test the independences a graph implies against held-out rows: a true graph passes, a damaged one is
rejected, and the sparsest passing graph is not always the most accurate.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Background knowledge
:link: knowledge
:link-type: doc
:img-top: /_generated/thumbs/knowledge.svg
:img-alt: The graph PC recovers with tiers in this notebook

Pass what you already know to a search: tiers in time, and required or forbidden edges.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Ensemble of methods
:link: ensemble
:link-type: doc
:img-top: /_generated/thumbs/ensemble.svg
:img-alt: The true graph of this notebook's simulated data

Fit four methods on the same data, combine their edges by a vote, weigh each vote by how stable it
is, and measure the result against each method.

{bdg-secondary-line}`Notebook · Colab`
:::
::::

```{toctree}
:hidden:
:caption: Advanced

bootstrap
markov_check
knowledge
ensemble
```

## Interactive demos

The demos run in your browser, with nothing to install. Step through PC and Meek's rules,
explore d-separation, or compare methods on one dataset.

::::{grid} 1 2 2 3
:gutter: 3
:class-container: example-gallery

:::{grid-item-card} Compare on one dataset
:link: compare
:link-type: doc
:img-top: /_generated/thumbs/compare.svg
:img-alt: Andrey's graph with each edge colored against the true graph

The true graph, Andrey's, and causal-learn's, side by side and scored.

{bdg-secondary-line}`In your browser`
:::

:::{grid-item-card} PC, step by step
:link: pc-replay
:link-type: doc
:img-top: /_generated/thumbs/pc-replay.svg
:img-alt: The graph PC recovers from the rain and sprinkler example

Every independence test, collider, and Meek rule PC recorded, beside its pseudocode.

{bdg-secondary-line}`In your browser`
:::

:::{grid-item-card} Meek's rules
:link: meek
:link-type: doc
:img-top: /_generated/thumbs/meek.svg
:img-alt: A partially directed graph after one of Meek's rules

Each orientation rule, one step at a time, from the orientation engine.

{bdg-secondary-line}`In your browser`
:::

:::{grid-item-card} d-separation explorer
:link: d-separation
:link-type: doc
:img-top: /_generated/thumbs/d-separation.svg
:img-alt: A graph with a fork, a collider, and a descendant of the collider

Choose two variables and a conditioning set; see which paths stay open.

{bdg-secondary-line}`In your browser`
:::
::::

```{include} ../../_generated/examples/live-demo.md
```

```{toctree}
:hidden:
:caption: Interactive demos

compare
pc-replay
meek
d-separation
```

## Misc

::::{grid} 1 2 2 3
:gutter: 3
:class-container: example-gallery

:::{grid-item-card} Visualize a causal graph
:link: viz
:link-type: doc
:img-top: /_generated/thumbs/viz.svg
:img-alt: A structure drawn with andrey.viz.draw

Draw a recovered graph as a self-contained SVG.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Speed on your machine
:link: speed_on_your_machine
:link-type: doc
:img-top: /_generated/thumbs/speed_on_your_machine.svg
:img-alt: The true graph of the notebook's smallest dataset

Time PC in Andrey and causal-learn on your own hardware.

{bdg-secondary-line}`Notebook · Colab`
:::
::::

```{toctree}
:hidden:
:caption: Misc

viz
speed_on_your_machine
```
