---
name: graph-types
description: DAG, CPDAG, PAG, and temporal graphs - diagrams, meaning, and the drawing legend.
meta:
  type: guide
---

# Graph types

The graph type says what a discovery result identifies: a directed model, a class of equivalent
models, or relations across time. Andrey records this in `GraphStructure.kind` or in a
`TemporalStructure`.

Every matrix below comes from the pictured graph's `to_numpy()` method. A cell `M[i, j]` is the
mark at node `i` on its edge with `j`: `0` absent, `1` tail, `2` arrowhead, `3` circle. See
[Graph representation](graph-representation.md) to compare this convention with other packages.

## DAG: a directed model

A **directed acyclic graph** has a direction on every edge and no directed cycles. In a causal
model, `X -> Y` represents a direct effect relative to the variables in that model. Interpreting
a learned DAG causally depends on the method's assumptions and the data.

```{raw} html
:file: ../../_static/graph-types-dag.html
```

The chain `X -> Y -> Z` has marks `(TAIL, ARROW)` on each edge. Andrey stores it with `kind="dag"`:

```python
import numpy as np
from andrey import GraphStructure

dag = GraphStructure.from_numpy(
    np.array([[0, 1, 0], [2, 0, 1], [0, 2, 0]], dtype=np.int8),
    kind="dag",
    labels=("X", "Y", "Z"),
)
assert dag.endpoints(0, 1) == (1, 2)
```

## CPDAG: an equivalence class

A **completed partially directed acyclic graph** represents DAGs with the same conditional
independence relationships. Directed edges agree across that class; an undirected edge can
point either way in different members. [Chickering (2002)](#references) describes this
representation.

```{raw} html
:file: ../../_static/graph-types-cpdag.html
```

The chain `X -- Y -- Z` represents three DAGs: `X -> Y -> Z`, `X <- Y <- Z`, and
`X <- Y -> Z`. It excludes `X -> Y <- Z`, which has a collider at `Y` and different independence
relationships. Undirected edges cannot always be oriented independently.

Andrey uses `kind="cpdag"`. The pair `(TAIL, TAIL)` means an unresolved direction, not two
opposite causal effects. PC and GES, among other methods, return this kind. GIES returns it too
for data with interventions, where the class holds only the DAGs that also agree on the data under
each set of intervened variables, so it can direct more edges
([Hauser and Buhlmann (2012)](#references)). GnIES returns the same kind of class for the targets
it learns, with every edge at a target directed.

```python
cpdag = GraphStructure.from_numpy(
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8),
    kind="cpdag",
    labels=("X", "Y", "Z"),
)
assert cpdag.endpoints(0, 1) == (1, 1)
```

## PAG: uncertainty with hidden variables

A **partial ancestral graph** represents an equivalence class of ancestral graphs when variables
may be unobserved. Arrowheads and tails record shared endpoint information; circles mark
unresolved endpoints. [Zhang (2008)](#references) gives the causal interpretation.

```{raw} html
:file: ../../_static/graph-types-pag.html
```

In `X o-> Y`, the arrowhead rules out `Y` being an ancestor of `X` under the ancestral-graph
assumptions. The circle at `X` leaves that endpoint unresolved. This does not establish a direct
effect from `X` to `Y`. A bidirected edge `X <-> Y` indicates a relation through hidden causes;
it is not a directed feedback loop and does not identify a particular hidden variable.

FCI, GFCI, BOSS-FCI, GRaSP-FCI, and FCI-JCI return `kind="pag"`, and MMB-by-MMB returns the edges at
its target as the PAG holds them. RCD, BottomUpParceLiNGAM, and CAM-UV use the same marks without
the equivalence-class reading: `X -> Y` is an edge they found, `X <-> Y` a pair that RCD links
through a hidden variable, and `X o-o Y` a pair in the block BottomUpParceLiNGAM leaves unordered,
which holds the variables a hidden common cause confounds and their observed ancestors, or a pair
that CAM-UV finds joined through a hidden variable, a common cause or a causal path, whose relation
it cannot determine. A circle is `CIRCLE=3`, so `X o-> Y` has marks `(3, 2)`:

```python
pag = GraphStructure.from_numpy(
    np.array([[0, 3, 0], [2, 0, 2], [0, 3, 0]], dtype=np.int8),
    kind="pag",
    labels=("X", "Y", "Z"),
)
assert pag.endpoints(0, 1) == (3, 2)
```

The examples illustrate graph semantics. Assigning a `kind` and passing `validate()` checks the
endpoint encoding; it does not prove that a hand-built graph is a completed equivalence class
or that a learned graph is causally correct.

## Directed cycles: the kind matters

Andrey's `kind="digraph"` allows directed cycles and self-loops. The same endpoint pair `(2, 2)`
has different meanings in `pag` and `digraph`, and the renderer draws them differently:

```{raw} html
:file: ../../_static/graph-types-feedback.html
```

In the PAG fragment, one edge has two arrowheads. In the directed graph, `X -> Y` and `Y -> X`
are separate directed edges. The endpoint matrix alone cannot distinguish them; retain `kind`
when saving or converting a graph.

`local_isa_ling` returns a `digraph`: the edges into one variable and into each of its children in
a linear model whose graph may hold cycles, where a 2-cycle at the variable is two directed edges.

## Undirected graphs: adjacency without direction

`kind="undirected"` holds `(TAIL, TAIL)` pairs only. Its edges claim no direction, and a moral
graph's edges also join spouses, two parents of one child that need not be adjacent. `mmpc` and
`hiton_pc` return the skeleton of their parents-and-children sets by default, an undirected graph,
and a CPDAG with `structure="cpdag"`. `iamb`, `inter_iamb`, `gs`, `fast_iamb`, `hiton_mb`, `mmmb`,
and `pcmb` return a CPDAG by default; with `structure="skeleton"` they return its adjacencies as an
undirected graph, and with `structure="moral"` the moral graph of their Markov blankets. With a
`target`, the nine return that variable's set as undirected edges. An undirected graph has no CPDAG
or DAG, so `andrey.metrics.score` scores it on its skeleton alone: `shd` counts the pairs adjacent
in one graph and not in the other, and no arrowhead metric is reported.

## Temporal graphs

A `TemporalStructure` stores a graph for each lag. An edge `X -> Y` in `lag(k)` means
`X(t-k) -> Y(t)`. Lag zero describes within-time relations. A self-loop at a positive lag means
a variable affects its own future value.

```{raw} html
:file: ../../_static/graph-types-temporal.html
```

Here, `X(t-1) -> Y(t)` occurs at lag 1 and `Y(t-2) -> X(t)` occurs at lag 2. Collapsing time
produces a directed cycle in `summary_graph()`, even though both original arrows point forward
in time. The summary retains each direction's lags:

```python
from andrey import TemporalStructure

lag1 = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]]), kind="dag")
lag2 = GraphStructure.from_numpy(np.array([[0, 2], [1, 0]]), kind="dag")
temporal = TemporalStructure.from_lag_graphs([lag1, lag2], lags=[1, 2], labels=("X", "Y"))

summary = temporal.summary_graph()
assert summary.kind == "digraph"
assert summary.lags_of(0, 1) == (1,)
assert summary.lags_of(1, 0) == (2,)
```

`lags` lists lag values, so `lag(2)` selects lag 2, not the third array position. Granger
causality finds no effects within a time step, so its lags start at 1. `lag_weights` holds
autoregressive coefficients; VARMA can also provide `lag_weights_ma` for moving-average
coefficients.

Longitudinal results add an occasion axis. Use `at(time, lag)` and `time_weights` to inspect the
full panel. **`lag(k)`, `lag_weights`, and `summary_graph()` refer to the final occasion**, not a
union over all occasions. `NaN` in `time_weights` marks a block that could not be computed.

## Which methods return each type?

| Structure | Methods |
|---|---|
| `dag` | `direct_lingam`, `ica_lingam`, `multi_group_direct_lingam`, `calm`, `notears`, `dagma`, `cam`, `resit`, `score_matching`, `nogam`, `gin`, `anm`, `pnl`, `loci`, `icp`, `y_structures`, `ldp`, `ld3` |
| `cpdag` | `pc`, `cdnod`, `ges`, `gies`, `gnies`, `xges`, `ages`, `hc`, `tabu`, `exact_search`, `mmhc`, `rsmax2`, `boss`, `grasp`, `igsp`, `ut_igsp`, `iamb`, `inter_iamb`, `gs`, `fast_iamb`, `hiton_mb`, `mmmb`, `pcmb`, `mb_by_mb`, `ldecc`, `locpc_cde` |
| `undirected` | `mmpc`, `hiton_pc` |
| `pag` | `fci`, `gfci`, `boss_fci`, `grasp_fci`, `fci_jci`, `mmb_by_mmb`, `rcd`, `bottom_up_parce_lingam`, `camuv` |
| `digraph` | `local_isa_ling` |
| `TemporalStructure` | `varma_lingam`, `longitudinal_lingam`, `pcmci`, `pcmciplus`, `jpcmciplus`, `lpcmci`, `granger`, `dynotears` |

`cdnod` returns kind `cpdag`, but its graph is only partially directed: the domain index orients
edges that the CPDAG of observational data leaves undirected, so score it against the graph CD-NOD
can identify rather than the true DAG's CPDAG. The index itself is not a node of the graph: its
edges, each `env -> variable`, are in `metadata["index_edges"]`, and
`metadata["changing_variables"]` lists the same variables. They are the variables whose causal
module, the distribution given their causes, changes across the environments; their descendants
change through them and are not listed.

With `env`, `ges`, `hc`, `tabu`, `boss`, and `grasp` return the same kind of graph: kind `cpdag`,
partially directed by the index, with the same `index_edges` and `changing_variables`.

`ages` returns kind `cpdag`, but its graph can direct edges that the CPDAG of its skeleton leaves
undirected, as a CPDAG narrowed by background knowledge does: it takes the directions from the
sparser CPDAGs that GES returns at higher penalties, so score it against the graph AGES can
identify rather than the true DAG's CPDAG. `metadata["path_penalties"]` lists the penalties of
those CPDAGs and `metadata["path_aggregated"]` whether each one's directions were taken.

`y_structures` returns a `dag` with no edges: it claims that one variable is a causal ancestor of
another, which is not an edge, so its claims are records in `metadata["effects"]`, and a pair
without a record is not claimed either way.

`ldp` returns a `dag` with no edges: its labels say how each column relates to the pair, which is
not an edge, so its answer, an adjustment set or none, is a record in `metadata["effects"]`. `ld3`
returns a `dag` whose only edges point into its `outcome`, from the parents it finds, the
`treatment` among them when the structural direct criterion holds. `locpc_cde` returns a `cpdag`,
the local essential graph it learned around its `outcome`: the edges at the variables it visited,
directed where its tests decide, and its answer is a record in `metadata["effects"]`.

`ldecc` returns a `cpdag` that holds only the edges at its `treatment`, as `mb_by_mb`'s does at its
`target`: an absent edge away from the treatment means nothing. Its answer, the locally valid parent
sets for the effect on its `outcome`, is a record in `metadata["effects"]`.

`fci_jci` returns a `pag` over the data variables alone. The environment index is not a node:
`metadata["index_edges"]` lists its edges as `[variable, kind]` pairs, `"directed"` for
`index -> variable`, `"partially_directed"` for `index o-> variable`, and `"bidirected"` for
`index <-> variable`. A variable keeps an edge to the index when its distribution changes across the
environments and no other variable explains the change; under hidden common causes that does not
make it a target of the change, so `fci_jci` lists no `changing_variables`. With `env`, `gfci`,
`boss_fci`, and `grasp_fci` return their PAG and `index_edges` the same way.

`jpcmciplus` returns a `TemporalStructure` over the system variables alone, and lists no
`changing_variables` either: the links of its dummies, in `metadata["index_edges"]` and
`metadata["time_index_edges"]`, stand in for the hidden contexts, and the links of the measured
contexts are in `metadata["context_links"]`.

`icp` returns a `dag` whose only edges point into its `target`, from the parents it estimates; an
absent edge says nothing about the pair.

`multi_group_direct_lingam` returns a list of results, one per group. GIN includes explicit latent
nodes, marked by `node_types` (`OBSERVED=0`, `LATENT=1`). ANM and PNL are direction tests on two
variables: each returns a two-node `dag` with the direction its tests decide, or no edge, and
both p-values and test statistics in `metadata`.

## How to read a causal graph

Draw any `GraphStructure`, including a temporal lag or summary, with `andrey.viz.draw`:

```python
from pathlib import Path
from andrey.viz import draw

Path("graph.svg").write_text(draw(pag), encoding="utf-8")
```

We use the same marks for every graph kind. Each mark is a shape, so you can read drawings in
print and without color. Edge colors only repeat what the marks say.

```{raw} html
:file: ../../_static/graph-types-legend.html
```

`draw(structure, targets=["Y"])` draws each intervention target as a square and puts a cross in
the middle of each edge with an arrowhead into it. A hard intervention cuts the edges into its
targets, and a soft one keeps them; both get the same marks, so a cross does not mean the edge was
removed. If any label is longer than two characters, every label sits below its node.
`andrey.viz.to_graphviz` writes the same nodes and edge marks in DOT, for Graphviz and other tools,
with squares for the targets. It leaves out the cross because Graphviz marks only an edge's ends.

## References

- [Chickering (2002), Learning Equivalence Classes of Bayesian-Network Structures][chickering].
- [Zhang (2008), Causal Reasoning with Ancestral Graphs][zhang].
- [Hauser and Buhlmann (2012), Characterization and Greedy Learning of Interventional Markov
  Equivalence Classes of Directed Acyclic Graphs][hauser].

[chickering]: https://jmlr.org/papers/v2/chickering02a.html
[zhang]: https://www.jmlr.org/papers/v9/zhang08a.html
[hauser]: https://jmlr.org/papers/v13/hauser12a.html
