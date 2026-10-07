---
name: graph-types
description: DAG, CPDAG, PAG, and temporal graphs - diagrams, interpretation, and Andrey structures.
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
opposite causal effects. PC and GES, among other methods, return this kind.

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

FCI and GFCI return `kind="pag"`. A circle is `CIRCLE=3`, so `X o-> Y` has marks `(3, 2)`:

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

`lags` lists lag values, so `lag(2)` selects lag 2, not the third array position. `lag_weights`
holds autoregressive coefficients; VARMA can also provide `lag_weights_ma` for moving-average
coefficients.

Longitudinal results add an occasion axis. Use `at(time, lag)` and `time_weights` to inspect the
full panel. **`lag(k)`, `lag_weights`, and `summary_graph()` refer to the final occasion**, not a
union over all occasions. `NaN` in `time_weights` marks a block that could not be computed.

## Which methods return each type?

| Structure | Methods |
|---|---|
| `dag` | `direct_lingam`, `ica_lingam`, `multi_group_direct_lingam`, `exact_search`, `calm`, `gin` |
| `cpdag` | `pc`, `cdnod`, `ges`, `gies`, `hc`, `boss`, `grasp` |
| `pag` | `fci`, `gfci` |
| `TemporalStructure` | `varma_lingam`, `longitudinal_lingam` |

`multi_group_direct_lingam` returns a list of results, one per group. GIN includes explicit latent
nodes, marked by `node_types` (`OBSERVED=0`, `LATENT=1`). PNL is a direction test on two variables:
it returns a two-node `dag` with the direction its p-values decide, or no edge, and both p-values
in `metadata`.

Draw any `GraphStructure`, including a temporal lag or summary, with `andrey.viz.draw`:

```python
from pathlib import Path
from andrey.viz import draw

Path("graph.svg").write_text(draw(pag), encoding="utf-8")
```

## References

- [Chickering (2002), Learning Equivalence Classes of Bayesian-Network Structures][chickering].
- [Zhang (2008), Causal Reasoning with Ancestral Graphs][zhang].

[chickering]: https://jmlr.org/papers/v2/chickering02a.html
[zhang]: https://www.jmlr.org/papers/v9/zhang08a.html
