---
name: andrey-0-3-0
description: "Score-based search in Andrey 0.3.0: Tabu search and XGES are new and supported, as is hill climbing, which now returns the same graphs faster."
meta:
  type: post
blogpost: true
date: 2026-10-16
author: Shu Wan
version: 0.3.0
cover: contour
---

# Score-based search

A score-based method searches for the graph that best explains the data according to a score.
This release adds Tabu search and XGES. From 0.3.0, both are supported, as is hill climbing.
Hill climbing also returns the same graphs in less time.

Below, we explain why a search can stop at the wrong graph. We run GES, hill climbing, Tabu search,
and XGES on one small dataset, then report benchmark results.

## Searching by a score

Andrey's score-based methods rate a graph by its BIC (Bayesian information criterion). For each
variable, the BIC measures how much variance its parents in the graph leave unexplained.
It adds a penalty for each parent. A lower score is better. This BIC assumes linear relations
with Gaussian noise. The methods also assume no hidden common causes.

The number of graphs grows faster than exponentially with the number of variables, so a search
cannot score them all. A greedy search starts from the empty graph. It takes the move that
improves the score most, such as adding, removing, or reversing one edge. It stops when no move
improves the score. That graph is a local optimum: no single move improves it. A better graph
can lie a few moves away, past graphs that score worse. A search that stops there can miss edges
or point them the wrong way.

The four methods handle local optima differently:

- **Hill climbing** searches over DAGs (directed acyclic graphs) and stops at the first local
  optimum.
- **Tabu search** takes the same moves. At a local optimum, it takes the best move even when the
  score gets worse, and keeps going. It never returns to a recently visited graph, so it cannot
  step straight back. It returns the best graph it visited, so with the same settings its score is
  never worse than hill climbing's.
- **GES** (greedy equivalence search) searches over equivalence classes: sets of graphs that fit
  any data equally well. It adds edges while the score improves, then removes edges while the
  score improves.
- **XGES** (extremely greedy equivalence search) searches the same classes with GES's moves and
  edge reversals, in a different order. Each step deletes an edge if that improves the score.
  Otherwise, it tries reversing an edge, then adding one. Its extended search then deletes each
  edge of the result in turn, forbids adding it back, and searches again. It keeps any result
  with a better score, which lets it leave a local optimum.

## Methods

Before each release, we benchmark each supported method against another package and check its
results on known graphs. Tabu search and XGES are new in this release.

| Method | Status | Searches over | Returns |
|---|---|---|---|
| [GES](../docs/code/generated/andrey.ges.rst) | Supported | Equivalence classes | A CPDAG and its BIC score |
| [XGES](../docs/code/generated/andrey.xges.rst) | Supported | Equivalence classes | A CPDAG and its BIC score |
| [Hill climbing](../docs/code/generated/andrey.hc.rst) | Supported | DAGs | The CPDAG of the final DAG, and its score |
| [Tabu search](../docs/code/generated/andrey.tabu.rst) | Supported | DAGs | The CPDAG of the best DAG visited, and its score |
| [GIES](../docs/code/generated/andrey.gies.rst) | Experimental | Equivalence classes | A CPDAG and its BIC score; without interventions, the same search as GES |
| [Exact search](../docs/code/generated/andrey.exact_search.rst) | Experimental | Every DAG, so only a few variables | One DAG with the best BIC score |
| [CALM](../docs/code/generated/andrey.calm.rst) | Experimental | A penalized likelihood, by continuous optimization | A weighted DAG, without a score |

Choose a method for your data:

- **XGES** for continuous data that roughly fit the BIC's assumptions. In our benchmarks, its
  graphs scored better and had fewer errors than GES's (see [Benchmarks](#benchmarks)). Set
  `extended_search=False` for a faster fit that can stop at a local optimum.
- **GES** for the same data, as a well-studied reference.
- **Hill climbing** as a fast baseline over DAGs, and **Tabu search** when hill climbing may stop
  early. Both take `max_iter`, a move limit.
- **Exact search** to find the best-scoring graph on a few variables.
- **CALM** to try continuous optimization that returns edge weights. You need the `torch` extra.

BOSS and GRaSP also minimize the BIC but search over orders of the variables. These are Andrey's
permutation-based methods. When hidden common causes are plausible, use the methods in the
[0.2.0 post](andrey-0-2-0.md).

## Reading a CPDAG

GES, XGES, hill climbing, Tabu search, and GIES all return a CPDAG (completed partially directed
acyclic graph). You can find its BIC score in `metadata["score"]`.
A CPDAG represents an equivalence class:

- `x -> y`: every graph in the class has `x` causing `y`.
- `x -- y`: the class has graphs with `x -> y` and graphs with `x <- y`. These fit the data
  equally well, so the data cannot orient this edge.

See [Graph types](../docs/guides/graph-types.md#cpdag-an-equivalence-class) for details.

## Example

Here `x0` causes `x1`, and `x0`, `x1`, and `x2` cause `x3`. The relations are linear.
The noise is Gaussian. We chose this model because GES and hill climbing stop early on it.

```{literalinclude} 0.3.0/example.py
:language: python
:caption: example.py
```

```{raw} html
:file: 0.3.0/graphs.html
```

- GES and hill climbing stop at the same graph, with a score of 18.22. It has `x2 -> x0`, an edge
  absent from the true graph. It also has `x3 -> x0`, which points the wrong way.
  It misses `x0 -- x1`.
- Tabu search and XGES return the true graph's equivalence class, with a lower score of 15.11.
  The edge `x0 -- x1` stays undirected. The true graph and the same graph with `x1 -> x0` are
  equivalent, so they fit any data equally well.
- XGES without its extended search returns GES's graph (the last result below). Here, the
  extended search finds the better graph.

The script prints each result in full:

```{literalinclude} 0.3.0/example-output.txt
:language: text
:caption: example.py
```

## Benchmarks

We compared XGES with xges, a Python package that implements it. We compared Tabu search and
hill climbing with bnlearn, an R package. We simulated linear-Gaussian data from random graphs,
with ten rows per variable.

| Method | Compared with | Variables | Andrey's speedup | Same graph |
|---|---|---|---|---|
| XGES | xges | 25-200 | 3.95-140x | 40 of 40 datasets |
| Tabu search | bnlearn | 10-800 | 2.56-617x | 31 of 35 datasets |
| Hill climbing | bnlearn | 10-800 | 3.48x to at least 933x | 30 of 34 datasets |

The [0.3.0 release card](../docs/benchmarks/0.3.0.md) reports every measurement.
XGES ran on graphs with a mean degree of 2 and 4, on 1 core per fit.
Tabu search and hill climbing ran on graphs with a mean degree of 2, on 4 cores per fit.
All fits ran on an AMD EPYC 7713. Hill climbing and Tabu search used a move limit, `max_iter`,
high enough never to stop them. Their speedups are approximate because Andrey's times vary
between datasets of the same size. The release card has the details.

At 800 variables, one bnlearn hill-climbing fit hit its time limit, so that speedup is a lower
bound. At 1,600 variables, every bnlearn fit hit its 1-hour limit. Andrey's speedup is therefore
at least 474x for Tabu search and at least 609x for hill climbing.
Andrey alone also ran both at 3,200 variables.
Without the extended search, XGES returns the same graph as xges on every dataset.
At a mean degree of 2, it runs 1.34-6.35x faster.

```{raw} html
:file: 0.3.0/speedup.html
```

On these data, XGES's graphs had a lower median BIC than GES's at every size and both mean
degrees. They also had fewer errors at the median. At a mean degree of 4, the extended search
left fewer errors than XGES without it at every size: 45 against 88 at 50 variables.

Agreement between packages does not establish accuracy. From 50 variables, Tabu search's and
hill climbing's median errors exceed the empty graph's, in both packages. Tabu search's BIC was
lower than hill climbing's at every size, by a median of 15.4 at 50 variables and 53.9 at 400.
Yet its errors were not lower: 615 against 605 at 400 variables.
A better score is not always a better graph.

At a mean degree of 2, XGES's graphs have a lower median BIC than the true graph at every size.
With ten rows per variable, the best-scoring graph is not the true one.
A larger `lambda_value` penalizes each parent more and favors sparser graphs.

```{raw} html
:file: 0.3.0/errors.html
```

Andrey's XGES alone also ran at 500 and 1,000 variables. At a mean degree of 2, its graphs at
these sizes have more errors than the empty graph's. Tabu search's and hill climbing's graphs
also have more errors than the empty graph's at 1,600 and 3,200 variables.

Speed and accuracy depend on your data, settings, and hardware.

## Other highlights

- **Faster hill climbing.** Hill climbing caches the score change for each candidate move
  between steps. It recomputes only the moves a step changes.
  It returns the same graphs as before in less time.
- **Hill climbing runs in one process.** Its worker processes were slower at every measured size.
  We removed them, along with `ANDREY_HC_PARALLEL_MIN_WORK` and `andrey calibrate --method hc`.
  This change breaks scripts that set either one.
  `ANDREY_NUM_WORKERS` still sets the workers for GES, GIES, and GFCI.
- **XGES without the extended search.** `andrey.xges(data, extended_search=False)` returns the
  graph where the first search stops. At 200 variables and a mean degree of 2, it took 0.164 s,
  against 1.64 s with the extended search. These are medians over five datasets on 1 core.
- **Shorter summaries.** Since 0.2.1, a result's summary shows the length of a long metadata list
  instead of printing it in full.

For more details, read the [changelog](../docs/changelog.md).

## Try it

```{code-block} shell
:class: window

pip install -U andrey-core
```

The [GES quickstart](../docs/examples/ges_quickstart) notebook fits GES on a small chain.
You can open it in Colab and replace `andrey.ges` with `andrey.xges`, `andrey.tabu`, or `andrey.hc`.
Then try the methods on your own data.

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
