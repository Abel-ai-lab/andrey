---
name: examples-index
description: Example notebooks - each runs one method or task on your machine or in Colab.
meta:
  type: index
html_theme.sidebar_secondary.remove: true
---

# Examples

Each notebook runs one method or task, on your own machine or in Colab; each has **Open in Colab**
and **Download this notebook** links. The [demos](../demos/index) run in your browser instead. From
a source checkout, start Jupyter with `uv run --with jupyter jupyter lab`. A notebook that needs another
package installs it in a cell with `uv pip install`.

To run one in Colab, choose **Open in Colab**, select a CPU runtime, and run the setup cell before
the example. The setup cell installs Andrey in the Colab runtime only; the speed notebook labels
that runtime in its output.

::::{grid} 1 2 2 3
:gutter: 3
:class-container: example-gallery

:::{grid-item-card} Speed on your machine
:link: speed_on_your_machine
:link-type: doc
:img-top: /_generated/thumbs/speed_on_your_machine.svg
:img-alt: The true graph of the notebook's smallest dataset

Time PC in Andrey and causal-learn on your own hardware.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} GES quickstart
:link: ges_quickstart
:link-type: doc
:img-top: /_generated/thumbs/ges_quickstart.svg
:img-alt: The graph GES recovers in this notebook

Recover a three-variable chain with GES.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Constraint-based discovery
:link: constraint_based
:link-type: doc
:img-top: /_generated/thumbs/constraint_based.svg
:img-alt: The graph PC or FCI recovers in this notebook

PC's CPDAG and FCI's PAG from conditional-independence tests.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Permutation-based search
:link: permutation_based
:link-type: doc
:img-top: /_generated/thumbs/permutation_based.svg
:img-alt: The graph BOSS or GRaSP recovers in this notebook

BOSS and GRaSP search over orderings of the variables for the best-scoring graph.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Score-based search
:link: score_search
:link-type: doc
:img-top: /_generated/thumbs/score_search.svg
:img-alt: The true graph of this notebook's data

XGES, hill climbing, and Tabu search look for the graph with the best BIC score.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Linear non-Gaussian models
:link: lingam
:link-type: doc
:img-top: /_generated/thumbs/lingam.svg
:img-alt: The DAG DirectLiNGAM or ICA-LiNGAM recovers in this notebook

DirectLiNGAM and ICA-LiNGAM orient every edge when the noise is non-Gaussian.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Hidden common causes: RCD
:link: rcd
:link-type: doc
:img-top: /_generated/thumbs/rcd.svg
:img-alt: The true graph of this notebook's data, with one pair that shares a hidden cause

RCD finds the direct causes and marks the pairs that share a hidden cause.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Hidden common causes: BottomUpParceLiNGAM
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

ANM and PNL test which of two variables causes the other.

{bdg-secondary-line}`Notebook · Colab`
:::

:::{grid-item-card} Drawing a recovered structure
:link: viz
:link-type: doc
:img-top: /_generated/thumbs/viz.svg
:img-alt: A structure drawn with andrey.viz.draw

Draw a recovered graph as a self-contained SVG.

{bdg-secondary-line}`Notebook · Colab`
:::
::::

```{toctree}
:hidden:
:caption: Notebooks

speed_on_your_machine
ges_quickstart
constraint_based
permutation_based
score_search
lingam
rcd
bottom_up_parce_lingam
pairwise
viz
```
