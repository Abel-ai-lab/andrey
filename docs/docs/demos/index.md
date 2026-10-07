---
name: demos-index
description: Demos that run in your browser - step through PC and Meek's rules, explore
  d-separation, compare methods on one dataset, and browse the benchmarks.
meta:
  type: index
html_theme.sidebar_secondary.remove: true
---

# Demos

Demos run in your browser, with nothing to install. Step through PC and Meek's rules, explore
d-separation, compare methods on one dataset, or browse the benchmark results. The
[examples](../examples/index) are notebooks you run yourself.

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

:::{grid-item-card} Benchmark explorer
:link: benchmark-explorer
:link-type: doc
:img-top: /_generated/thumbs/benchmark-explorer.svg
:img-alt: Fit time against the number of variables for Andrey and other packages

Every recorded timing and error count, as charts and a table.

{bdg-secondary-line}`In your browser`
:::
::::

```{include} ../../_generated/examples/live-demo.md
```

```{toctree}
:hidden:
:caption: Demos

compare
pc-replay
meek
d-separation
benchmark-explorer
```
