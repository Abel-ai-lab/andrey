---
name: graph-representation
description: How Andrey stores a causal graph, compared with other causal discovery packages.
meta:
  type: guide
---

# Graph representation

A graph drawing and its matrix are two views of the same result. The numbers in the matrix can
mean endpoint marks, directed-edge presence, or coefficients, depending on the package. The row
can identify a source, an effect, or the endpoint being marked.

Andrey uses endpoint marks for topology and a separate array for weights. A `StructureOutput`
holds the graph, optional causal order and weights, and method-specific metadata.
`multi_group_direct_lingam` returns one such result per group.

For the meaning of DAGs, CPDAGs, PAGs, and temporal results, see
[Graph types](graph-types.md).

## Same graph, different numbers

Choose an edge to see its encoding. Focus or hover over a matrix cell to highlight the same node
pair across representations. The meanings of those cells depend on the matrix convention. The
coefficient `0.8` is an illustration, not an estimate from data.

```{raw} html
:file: ../../_static/graph-comparison.html
```

## At a glance

| Representation | Object or array | Meaning of cell `[i, j]` for `i -> j` |
|---|---|---|
| Andrey endpoints | `GraphStructure.to_numpy()` | `1`: tail at `i`; reverse cell is `2`: arrowhead at `j` |
| causal-learn endpoints | `GeneralGraph.graph` | `-1`: tail at `i`; reverse cell is `1`: arrowhead at `j` |
| gCastle directed adjacency | `causal_matrix` | `1`: edge from row `i` to column `j` |
| lingam coefficients | `adjacency_matrix_` | Effect of column `j` on row `i`; the coefficient for `i -> j` is in `[j, i]` |

A binary directed adjacency matrix needs extra conventions to distinguish an undirected edge,
two directed edges, and uncertain endpoint marks. A coefficient matrix describes a fitted
directed model. Andrey retains endpoint uncertainty in the graph and stores coefficients
separately. The [package references](#references) document the compared conventions.

## Endpoint marks

A `GraphStructure` stores one mark at each end of each edge. `M[i, j]` is the mark at node `i` on
the edge between `i` and `j`, so every edge fills two cells. The codes are exported from
`andrey.core`:

| Mark | Code | Symbol |
|---|---:|---|
| `NULL` | 0 | no edge |
| `TAIL` | 1 | `-` |
| `ARROW` | 2 | `>` |
| `CIRCLE` | 3 | `o` |

| Edge | `M[i, j]` | `M[j, i]` | Valid in |
|---|---|---|---|
| `i -> j` | `TAIL` | `ARROW` | every kind |
| `i -- j` | `TAIL` | `TAIL` | `cpdag`, `pag` |
| `i <-> j` | `ARROW` | `ARROW` | `pag` (bidirected edge); the same codes mean a 2-cycle in `digraph` |
| `i o-> j` | `CIRCLE` | `ARROW` | `pag` |
| `i -o j` | `TAIL` | `CIRCLE` | `pag` |
| `i o-o j` | `CIRCLE` | `CIRCLE` | `pag` |

The codes are nonnegative, with `0` as the only empty value. The store is a CSR matrix of the
nonzero marks (`int64` row pointers, `int32` columns, `int8` marks), so memory grows with the number
of edges, not with $n^2$. The graph's `kind` states which edge types are valid; `validate()`
rejects a `CIRCLE` outside a `pag`, a double arrowhead in a `dag` or `cpdag`, and an undirected edge
in a `dag` or `digraph`.

## Run extras

- `ordering` - the causal order, from `direct_lingam`, `ica_lingam`, `multi_group_direct_lingam`,
  and `varma_lingam`.
- `weighted_adjacency` - dense `float64` weights from `calm` and the static LiNGAM methods.
  `W[i, j]` is the weight of `i -> j`, so rows are causes. This is the transpose of lingam's
  `adjacency_matrix_`, where `B[i, j]` is the effect of $x_j$ on $x_i$ in $x = Bx + e$.
- `metadata` - JSON-safe, method-specific values. GES, GIES, HC, BOSS, and GRaSP report `score`;
  CALM reports `objective`. ExactSearch does not report a score. PNL reports its two directional
  p-values; its two-node graph holds the direction they decide, by the rule its `alpha` parameter
  selects.
- `save(path, fmt="json")` and `StructureOutput.load(path)`; `fmt="npz"` supports graphs only.

## Temporal results

`TemporalStructure` is a stack of `GraphStructure` objects, one per lag. `lags` lists the lag
values, `lag(k)` selects a graph by lag value, and `lag_weights` holds the `(n_lags, n, n)`
coefficients (`lag_weights_ma` adds the moving-average stack for VARMA). An autoregressive edge
`X(t-k) -> X(t)` is a self-loop in its lag graph. A panel result from `longitudinal_lingam` adds a
time axis: `times`, `at(time, lag)`, and `time_weights`, where `NaN` marks a block that cannot be
computed. `summary_graph()` merges the lags into one `digraph`, and its `lags_of(i, j)` returns the
lags at which `i -> j` appears. For a panel, `lag(k)`, `lag_weights`, and `summary_graph()` refer
to the **final occasion**; use `at(time, lag)` and `time_weights` to inspect earlier occasions.
The [temporal diagrams](graph-types.md#temporal-graphs) show how collapsing time can create cycles.

## Reading and converting

`print(out)`, or `out` alone in a notebook, shows the algorithm, the graph kind, the node and
edge counts, and the first 30 edges, then the causal order and metadata values: the text
`andrey run` prints in a terminal. `print(out.structure)` shows the graph part alone.

```python
import numpy as np
import andrey

rng = np.random.default_rng(0)
X = rng.uniform(size=(500, 3))
X[:, 1] += 2 * X[:, 0]
X[:, 2] += 1.5 * X[:, 1]

out = andrey.direct_lingam(X)
g = out.structure
g.kind                  # "dag"
g.endpoints(0, 1)       # (1, 2): TAIL at 0, ARROW at 1, that is, 0 -> 1
g.oriented_edges()      # [(0, 1, "directed"), (1, 2, "directed")]
g.to_edges()            # one row per edge: i, j, mark_i, mark_j
out.ordering            # (0, 1, 2)
```

`oriented_edges()` returns one `(source, target, type)` row per edge. A directed edge runs from
`source` to `target` whatever the node order. The types are `directed`, `undirected`, `bidirected`,
`circle` (`o-o`), `partially_directed` (`o->`), and `partially_undirected` (`-o`, a tail at
`source` and a circle at `target`). The three symmetric types list the lower node index first.
In a `digraph`, an `ARROW`/`ARROW` pair produces two `directed` rows, lower-to-higher node index
first, then the reverse. Printed summaries count and list both directions.

Nodes are numbered from 0 for an array. For a pandas DataFrame the column names become the
structure's `labels`, and `oriented_edges()` returns names instead of indices:

```python
import pandas as pd

df = pd.DataFrame(X, columns=["rain", "wet", "slippery"])
andrey.direct_lingam(df).structure.oriented_edges()
# [("rain", "wet", "directed"), ("wet", "slippery", "directed")]
```

`andrey run` reads a CSV or TSV header row the same way. Its JSON result lists the names in
`labels`, and each edge's `source` and `target` are indices into that list.

| Call | Returns |
|---|---|
| `to_numpy()`, `adjacency` | dense `int8` `(n, n)` mark matrix |
| `to_scipy_sparse()` | `scipy.sparse.csr_array` of marks |
| `oriented_edges()` | list of `(source, target, type)`, named by `labels` when set |
| `to_edges()` | structured array with fields `i`, `j`, `mark_i`, `mark_j` |
| `to_networkx()` | `networkx.DiGraph` or `networkx.MultiDiGraph` |
| `endpoints(i, j)`, `neighbors(i)` | the marks on one edge; the adjacent nodes |

`GraphStructure.from_numpy`, `from_edges`, and `from_networkx` build a graph back from each. Pass
the graph's `kind`: when it is omitted, they infer it from the marks, and the marks alone do not
always decide it. A `digraph` 2-cycle, for one, comes back as a `pag` with a bidirected edge.

`to_networkx()` returns a `DiGraph` for `dag`, `cpdag`, and `digraph`, and a `MultiDiGraph` for
`pag` or `multigraph=True`. Every edge carries `endpoints=(mark_i, mark_j)`. For `dag`, `cpdag`,
and `pag`, each edge becomes one arc from the lower to the higher node index, whatever its
direction, so read `endpoints` or `oriented_edges()` rather than the arc. A `MultiDiGraph` adds the
reverse arc for `<->` and `o-o`. A `digraph` emits each arc from tail to arrowhead, so NetworkX's
cycle functions see its cycles. `from_networkx` reads `endpoints`; an edge without it is read as
`u -> v`.

Andrey has no converters for other packages' objects. Build their matrices into a `GraphStructure`
with `from_numpy`:

```python
import numpy as np
from andrey import GraphStructure
from andrey.core import ARROW, CIRCLE, TAIL


def from_signed(G, kind):
    """Signed endpoint matrix (-1 tail, 1 arrowhead, 2 circle); G[i, j] is the mark at i."""
    G = np.asarray(G)
    if not np.isin(G, (-1, 0, 1, 2)).all():
        raise ValueError("only the codes -1, 0, 1, 2 have an Andrey equivalent")
    M = np.zeros(G.shape, dtype=np.int8)
    M[G == -1], M[G == 1], M[G == 2] = TAIL, ARROW, CIRCLE
    return GraphStructure.from_numpy(M, kind=kind)


def from_binary_dag(A):
    """Binary matrix with A[i, j] = 1 for i -> j."""
    A = np.asarray(A) != 0
    M = np.zeros(A.shape, dtype=np.int8)
    M[A], M[A.T] = TAIL, ARROW
    return GraphStructure.from_numpy(M, kind="dag")
```

- **causal-learn** - `G.graph[i, j]` is also the mark at `i`, so `from_signed(G.graph, kind)`
  needs no transpose. Its wildcard and compound codes (3 and above) have no Andrey equivalent.
- **gCastle DAG estimators** - `causal_matrix` sets `A[i, j] = 1` for `i -> j`; use
  `from_binary_dag` for a directed acyclic result.
- **lingam** - pass `adjacency_matrix_.T` to `from_binary_dag`. Some estimators write `NaN` for
  an unobserved common cause; decide how to encode those pairs first.

## References

- [causal-learn graph implementation][signed-graph]: signed endpoint marks.
- [gCastle NOTEARS implementation][notears]: binary directed adjacency and separate weights.
- [LiNGAM model definition](https://lingam.readthedocs.io/en/latest/tutorial/lingam.html):
  the coefficient convention in $x = Bx + e$.

[signed-graph]:
  https://github.com/py-why/causal-learn/blob/main/causallearn/graph/GeneralGraph.py
[notears]:
  https://gcastle.readthedocs.io/en/latest/_modules/castle/algorithms/gradient/notears/linear.html
