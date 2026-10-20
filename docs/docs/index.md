---
name: docs-index
description: Andrey - fast causal discovery on CPU and GPU. Documentation home.
meta:
  type: index
---

# Andrey

Andrey is a Python package for causal discovery: from observational data, it learns a graph of
which variables drive which. It runs PC, FCI, GES, BOSS, GRaSP, the LiNGAM family, and more, each
as one function call, on CPU cores or a GPU. It is for people who use these methods on more
variables than the usual packages handle quickly, or from a script or a coding agent.

**The fastest causal discovery package.**
The [release cards](benchmarks.md) give each measured speedup compared with other packages,
including where Andrey is not faster.

```shell
uv add andrey-core
```

```shell
pip install andrey-core
```

```{literalinclude} ../../site/example.py
:language: python
```

The first call downloads 853 cells' measurements of 11 proteins, and PC learns a graph over them;
[Getting started](guides/getting-started.md#learn-a-graph) reads the result. Andrey needs Python
3.11 or later. Linux is fully supported; macOS and Windows install and run on the CPU, with no
guarantee yet.

::::{grid} 1 2 2 4
:::{grid-item-card} Get started
:link: guides/getting-started
:link-type: doc
Install Andrey and learn your first graph.
:::
:::{grid-item-card} Examples
:link: examples/index
:link-type: doc
Notebooks that run one method or task each.
:::
:::{grid-item-card} Demos
:link: demos/index
:link-type: doc
Step through methods and explore results in your browser.
:::
:::{grid-item-card} FAQ
:link: faq
:link-type: doc
Install, methods, speed, data, and the project.
:::
:::{grid-item-card} Changelog
:link: changelog
:link-type: doc
What changed in each release.
:::
::::

## Guides and reference

- [Install](guides/getting-started.md#install): the package and optional GPU acceleration.
- [Get started](guides/getting-started.md#learn-a-graph): learn a causal graph from real data.
- [Examples](examples/index): notebooks that run one method or task each.
- [Demos](demos/index): methods to step through in your browser.
- [API reference](api.md): methods, graph structures, and visualization.
- [CLI reference](api.md#command-line-interface): run methods from the shell.
- [Graph types](guides/graph-types.md): DAGs, CPDAGs, PAGs, and temporal graphs, with diagrams.
- [Graph representation](guides/graph-representation.md): Andrey's graph model compared with other
  causal discovery packages.
- [Release cards](benchmarks.md): each release's timings, accuracy, and known limits.

## More

- [Configuration](configuration.md): the environment variables Andrey reads.
- [Testing](ci.md): what each release is tested on, and how to run the GPU checks.
- [Brand kit](brand.md): wordmark, icon, and avatar downloads, with colors and usage guidelines.

```{toctree}
:hidden:
:maxdepth: 2

guides/index
examples/index
demos/index
api
configuration
benchmarks
faq
changelog
roadmap
brand
```
