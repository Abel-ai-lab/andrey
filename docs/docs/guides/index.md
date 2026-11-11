---
name: guides-index
description: The causal discovery problem stated formally, then practical guides - install
  Andrey, recover your first structure, read its graph types and representations.
meta:
  type: index
---

# Guides

Practical guides for using Andrey.

- [Concepts](concepts.md): the causal discovery problem stated formally, what the data identify,
  what each graph a method returns claims, and which methods fit which assumptions.
- [Getting started](getting-started.md): install Andrey, learn a graph from real measurements,
  and check how far to trust it.
- [Graph types](graph-types.md): what a DAG, a CPDAG, a PAG, and a temporal graph each claim, with
  diagrams, which methods return each, and a legend of the marks in a drawing.
- [Graph representation](graph-representation.md): how Andrey stores a graph, how to read and
  convert it, how other packages' matrices differ, and how to hand it to DoWhy.
- [Column order and machines](column-order.md): what each method keeps when you reorder the columns
  or change machines, and the `tie_break` setting.
- [Upgrading](upgrading.md): what to change in your code for a new release, with the common calls
  before and after.
- [Contributing](contributing.md): ways to help, and the steps from a fork to a merged pull
  request.

```{toctree}
:hidden:

concepts
getting-started
graph-types
graph-representation
column-order
upgrading
contributing
```
