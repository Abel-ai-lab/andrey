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

**Over 100x faster on PC, and faster on most other supported methods.**
The [benchmarks](benchmarks.md) give each measured speedup compared with causal-learn,
gCastle, and lingam, including where Andrey is not faster.

```shell
uv add andrey-core
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

## Documentation

- [Install](guides/getting-started.md#install) Andrey and choose optional CPU or GPU acceleration.
- [Get started](guides/getting-started.md#learn-a-graph) by learning a causal graph from real data.
- [Examples](examples/index) are notebooks; [demos](demos/index) run in your browser.
- [API reference](code/index.md) covers methods, graph structures, and visualization.
- [Graph representation](guides/graph-representation.md) compares Andrey's graph model with other
  causal discovery packages.
- [Graph types](guides/graph-types.md) explains DAGs, CPDAGs, PAGs, and temporal graphs with
  diagrams.
- [CLI reference](code/index.md#command-line-interface) covers running methods from the shell.
- [Configuration](configuration.md) lists the environment variables Andrey reads.
- [Benchmarks](benchmarks.md) report timings, accuracy, and practical limits by method family.
- [Testing](ci.md) says what each release is tested on, and how to run the GPU checks.

```{toctree}
:hidden:
:maxdepth: 2

guides/index
examples/index
demos/index
code/index
configuration
benchmarks
faq
changelog
roadmap
```
