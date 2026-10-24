---
name: guides-index
description: Practical guides - install Andrey, recover your first structure, read its graph
  types and representations.
meta:
  type: index
---

# Guides

Practical guides for using Andrey.

- [Getting started](getting-started.md): install Andrey, learn a graph from real measurements,
  and check how far to trust it.
- [Graph types](graph-types.md): what a DAG, a CPDAG, a PAG, and a temporal graph each claim, with
  diagrams, and which methods return each.
- [Graph representation](graph-representation.md): how Andrey stores a graph, how to read and
  convert it, how other packages' matrices differ, and how to hand it to DoWhy.
- [Column order and machines](column-order.md): what each method keeps when you reorder the columns
  or change machines, and the `tie_break` setting.
- [Contributing](contributing.md): set up a development environment, run the checks, and add a
  method.

```{toctree}
:hidden:

getting-started
graph-types
graph-representation
column-order
contributing
```
