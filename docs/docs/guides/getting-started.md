---
name: getting-started
description: Learn a causal graph from real protein measurements and check how far to trust it.
meta:
  type: guide
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

This guide learns a causal graph from real measurements, checks it against what biologists already
know, and then tests how far to trust it. Every output on this page is computed when the site is
built.

## Install

Andrey installs from PyPI as `andrey-core` (the import root is `andrey`), on Python 3.11 or later.
With [uv](https://docs.astral.sh/uv/), add it to your project (`uv init` creates one):

```shell
uv add andrey-core            # base (pure NumPy)
uv add "andrey-core[numba]"   # CPU JIT for GES, up to 64 variables
uv add "andrey-core[torch]"   # torch backend: GPU primitives
uv add "andrey-core[viz]"     # Graphviz layouts and matplotlib styling
uv add "andrey-core[all]"     # every extra
```

With pip, use the same names: `pip install andrey-core`, `pip install "andrey-core[numba]"`, and so
on.

### PyTorch for a GPU

The `torch` extra needs the PyTorch build that matches your GPU, which an install cannot detect on
its own. uv can: it reads the installed driver and fetches the matching build. With pip, install
PyTorch from its own index first, choosing the build for your CUDA version or `/cpu`:

```shell
UV_TORCH_BACKEND=auto uv pip install "andrey-core[torch]"

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "andrey-core[torch]"
```

For ROCm, install the matching PyTorch build yourself, then `andrey-core`. Andrey has no backend
for Intel GPUs.

### The development version

The latest `main` from GitHub, ahead of the next release:

```shell
uv add git+https://github.com/Abel-ai-lab/andrey
pip install git+https://github.com/Abel-ai-lab/andrey
```

### From source

Build the wheel and the source distribution yourself, then install the wheel:

```shell
git clone https://github.com/Abel-ai-lab/andrey
cd andrey
uv build                                  # writes both to dist/
uv pip install dist/andrey_core-*.whl     # into the active environment
```

To work on Andrey itself, `uv sync` in the clone sets up a development environment instead; see
[CONTRIBUTING](https://github.com/Abel-ai-lab/andrey/blob/main/CONTRIBUTING.md).

## The data

Sachs and colleagues (2005) measured 11 proteins and phospholipids in 853 single human immune
cells. Biologists had already mapped how these molecules signal to one another, so a graph learned
from the measurements can be checked against a known network of 17 links. `load_dataset` downloads
the measurements on first use and caches them.

```{code-cell} ipython3
import numpy as np
import pandas as pd

import andrey
from andrey.data import load_dataset

sachs = load_dataset("sachs")
sachs.data.head()
```

Each row is one cell and each column one molecule.

## Learn a graph

```{code-cell} ipython3
out = andrey.pc(sachs.data)
print(out)
```

PC starts with every pair of molecules linked. It removes a link when a statistical test finds the
two independent, either alone or given some of their other neighbors. The links that survive form
the graph's skeleton.

Directions are harder to learn. To an independence test, `raf -> mek` and `mek -> raf` usually
look the same, so PC leaves such a link undirected (`--`). The result is a CPDAG: it stands for
every causal graph that independence tests cannot tell apart.

PC orients a link only when a pattern forces it. Here p38 and jnk are not linked, and pkc was not
needed to make them independent. Of the graphs that fit, only one explains that: p38 -> pkc <- jnk,
a collider.

## Use the result

`out.structure` holds the graph. `oriented_edges()` lists each link as (source, target, type), with
the column names:

```{code-cell} ipython3
out.structure.oriented_edges()
```

`to_networkx()` hands the graph to NetworkX, and `to_numpy()` returns its endpoint marks; see
{doc}`graph-representation`.

## Check it against the known network

`andrey.metrics.score` compares a learned graph with a reference graph:

```{code-cell} ipython3
from andrey.metrics import score

scores = score(out.structure, sachs.graph)
shown = ("shd", "skeleton_precision", "skeleton_recall")
{k: round(scores[k], 2) for k in shown}
```

- **Skeleton precision** is the share of learned links that are in the known network.
- **Skeleton recall** is the share of the known network's links that were learned.
- **SHD**, the structural Hamming distance, counts the links to add, remove, or reorient to reach
  the known network. Lower is better.

PC finds 8 of the 17 links in the known signalling network, and none outside it.

A score means little on its own. The graph with no links is the baseline: a graph whose SHD is no
lower has not improved on it by this measure, even if some of its links are right.

```{code-cell} ipython3
empty = andrey.GraphStructure.from_numpy(
    np.zeros((11, 11), dtype=np.int8),
    kind="cpdag",
    labels=sachs.feature_names,
)
score(empty, sachs.graph)["shd"]
```

The directions need more care. These cells saw no intervention, and independence tests on
observational data cannot orient any of the known network's 17 links. Its CPDAG, the most PC or GES
could recover from these cells, leaves every link undirected:

```{code-cell} ipython3
from andrey.metrics import to_cpdag

{kind for _, _, kind in to_cpdag(sachs.graph).oriented_edges()}
```

So PC's collider at pkc disagrees with the known network. If the known network is right, at least
one of PC's independence decisions was wrong. A direction learned from observational data can rest
on a single test. Methods that assume more, such as LiNGAM's non-Gaussian noise, can orient links
that independence tests leave open; {doc}`graph-types` lists the methods that return a fully
directed graph.

## How far to trust it

### The threshold

`alpha` is the significance level of each independence test. PC removes a link when the test
cannot reject independence at that level, so a larger `alpha` removes fewer links:

```{code-cell} ipython3
def summary(graph):
    """Links, precision, recall, and SHD against the known network."""
    s = score(graph, sachs.graph)
    return {
        "links": len(graph.oriented_edges()),
        "precision": s["skeleton_precision"],
        "recall": s["skeleton_recall"],
        "shd": s["shd"],
    }


sweep = {}
for alpha in (0.01, 0.05, 0.1, 0.2):
    sweep[alpha] = summary(andrey.pc(sachs.data, alpha=alpha).structure)
pd.DataFrame.from_dict(sweep, orient="index").rename_axis("alpha").round(2)
```

```{code-cell} ipython3
:tags: [remove-cell]

# The prose below holds only while these do.
assert sweep[0.2]["links"] > sweep[0.05]["links"]
assert sweep[0.2]["recall"] > sweep[0.05]["recall"]
assert sweep[0.2]["precision"] < sweep[0.05]["precision"] == 1.0
```

The extra links a larger `alpha` keeps are partly right and partly wrong: recall rises and precision
falls. No setting is correct in general. Pick `alpha` for the cost of a missed link against the cost
of a false one, and report how the graph changes with it.

### The test's assumptions

PC's test, Fisher's z, assumes each variable is a linear function of its causes plus Gaussian
noise. These measurements are far from Gaussian: most are strongly skewed to the right.

```{code-cell} ipython3
sachs.data.skew().round(1)
```

A log transform makes them closer to Gaussian. Learn the graph again from the logged values:

```{code-cell} ipython3
logged = andrey.pc(np.log(sachs.data))
print(logged)
```

```{code-cell} ipython3
:tags: [remove-cell]

# The prose below holds only while these do.
def links(graph):
    return {frozenset(e[:2]) for e in graph.oriented_edges()}


def directed(graph):
    return {e[:2] for e in graph.oriented_edges() if e[2] == "directed"}


assert links(logged.structure) < links(out.structure) <= links(sachs.graph)
assert len(links(out.structure) - links(logged.structure)) == 1
assert directed(out.structure) == {("p38", "pkc"), ("jnk", "pkc")}
assert directed(logged.structure) == {("plc", "pip3"), ("pip2", "pip3")}
```

PC keeps 7 of the 8 links it found before and adds none. The directions changed: the collider at
pkc is gone, and another appears at pip3. The skeleton is the more stable part of the answer.
Treat directions learned from observational data as hypotheses to test.

### A second method

GES reaches a graph a different way: it searches for the graph with the best score, instead of
testing independence.

```{code-cell} ipython3
ges = andrey.ges(sachs.data)
ges.structure.oriented_edges() == out.structure.oriented_edges()
```

On these data GES returns the same graph as PC. Agreement between methods is reassuring but not
proof: GES's score assumes linear-Gaussian data too. The {doc}`../examples/index` compare more
methods, and draw the learned graph beside the known one.

## Your own data

Pass a pandas DataFrame, whose column names become the node names, or a NumPy array with one row
per sample and one column per variable. Most methods take that one argument and return the same
kind of result; `cdnod` also takes a domain index, and `multi_group_direct_lingam` and
`longitudinal_lingam` take a list of datasets. `andrey list` names the methods, and
{doc}`graph-types` explains which graph each one returns.

## Backends and parallelism

Andrey reads its numeric backend and its worker count from `andrey.config`, or from the
`ANDREY_BACKEND` and `ANDREY_NUM_WORKERS` environment variables;
[Configuration](../configuration.md) lists every setting. With more than one
worker, GES, GIES, GFCI, and hill climbing scan each pass in worker processes, and GES returns the
same graph as it does serially.

Starting the worker processes re-imports the script that started them. A script that runs with
more than one worker (`ANDREY_NUM_WORKERS` above 1, or `-1` for every core) must put its entry
point under the `__main__` guard:

```python
import andrey
from andrey.data import load_dataset


def main():
    X = load_dataset("sachs").data
    andrey.ges(X)


if __name__ == "__main__":
    main()
```

Without the guard the re-imported script starts the search again, the workers fail to start, and
the fit stops with `RuntimeError: parallel-search workers failed to start`.

## Known limits

FCI and GFCI depend on column order, GES was more accurate with a stronger penalty on the large
sparse benchmark graphs, and PC and FCI are practical to roughly 1,000 variables. See
[Known limits](../benchmarks.md#known-limits).

## References

- [Sachs et al. (2005), Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell
  Data][sachs].
- [Spirtes, Glymour, and Scheines (2000), Causation, Prediction, and Search][pc]: the PC algorithm.
- [Chickering (2002), Optimal Structure Identification With Greedy Search][ges]: GES.
- [Shimizu et al. (2006), A Linear Non-Gaussian Acyclic Model for Causal Discovery][lingam].

[sachs]: https://doi.org/10.1126/science.1105809
[pc]: https://doi.org/10.7551/mitpress/1754.001.0001
[ges]: https://jmlr.org/papers/v3/chickering02b.html
[lingam]: https://www.jmlr.org/papers/v7/shimizu06a.html
