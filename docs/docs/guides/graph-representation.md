---
name: graph-representation
description: How Andrey stores a causal graph, converts it between kinds, and hands it to other
  packages such as DoWhy.
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

## Compared with other packages

| Representation | Object or array | Meaning of cell `[i, j]` for `i -> j` |
|---|---|---|
| Andrey endpoints | `GraphStructure.to_numpy()` | `1`: tail at `i`; reverse cell is `2`: arrowhead at `j` |
| causal-learn endpoints | `GeneralGraph.graph` | `-1`: tail at `i`; reverse cell is `1`: arrowhead at `j` |
| pcalg `amat.cpdag` | `as(fit, "amat")` | `0`: tail at `i`; reverse cell is `1`: arrowhead at `j` |
| pcalg `amat.pag` | `fit@amat` | `2`: arrowhead at `j`; reverse cell is `3`: tail at `i` |
| Binary directed adjacency | `A` | `1`: edge from row `i` to column `j` |
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
| `i -- j` | `TAIL` | `TAIL` | `cpdag`, `pag`, `mag` |
| `i <-> j` | `ARROW` | `ARROW` | `pag`, `mag` (bidirected edge); the same codes mean a 2-cycle in `digraph` |
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
- `metadata` - JSON-safe, method-specific values. GES, GIES, XGES, HC, BOSS, and GRaSP report
  `score`; CALM reports `objective`. ExactSearch does not report a score. ANM and PNL report their
  two directional p-values and HSIC statistics; each two-node graph holds the direction they
  decide, by the rule the method's `alpha` parameter selects. PCMCI reports each lagged link's MCI
  `p_values` and `statistics`, and the `parents` its first step selected; PCMCI+ reports the same
  for every lag from 0, plus the same-step `conflicts` and `ambiguous_triples`. PC, CD-NOD, FCI,
  and GFCI with `collider_rule="conservative"` or `"majority"` report `ambiguous_triples`, the
  `[x, z, y]` triples, by column position, whose middle node the vote left undecided. LPCMCI reports
  every link with all its marks as `links`, plus `p_values`, `statistics`, its preliminary
  `parents`, and the links with a conflicting mark as `conflicts`.
- `save(path, fmt="json")` and `StructureOutput.load(path)`; `fmt="npz"` supports graphs only.

## Temporal results

`TemporalStructure` is a stack of `GraphStructure` objects, one per lag. `lags` lists the lag
values, `lag(k)` selects a graph by lag value, and `lag_weights` holds the `(n_lags, n, n)`
coefficients (`lag_weights_ma` adds the moving-average stack for VARMA); a `pcmci` result has lags 1
to `tau_max` and no weights, and a `pcmciplus` result adds lag 0 as a `cpdag`. An `lpcmci` result
has lag 0 as a `pag` and holds only the lagged links whose earlier end is a tail; a lagged link
with a circle or an arrowhead at its earlier end is in `metadata["links"]`. An autoregressive
edge `X(t-k) -> X(t)` is a self-loop in its lag graph. A panel result from `longitudinal_lingam`
adds a time axis: `times`, `at(time, lag)`, and `time_weights`, where `NaN` marks a block that
cannot be computed. `summary_graph()` merges the lags into one `digraph`, and its `lags_of(i, j)`
returns the lags at which `i -> j` appears. For a panel, `lag(k)`, `lag_weights`, and
`summary_graph()` refer to the **final occasion**; use `at(time, lag)` and `time_weights` to inspect
earlier occasions. The [temporal diagrams](graph-types.md#temporal-graphs) show how collapsing time
can create cycles.

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

Nodes are numbered from 0 for an array. For a pandas DataFrame the column names, as text, become
the structure's `labels`, and `oriented_edges()` returns names instead of indices:

```python
import pandas as pd

df = pd.DataFrame(X, columns=["rain", "wet", "slippery"])
andrey.direct_lingam(df).structure.oriented_edges()
# [("rain", "wet", "directed"), ("wet", "slippery", "directed")]
```

`andrey run` reads a CSV or TSV header row the same way. Its JSON result lists the names in
`labels`, and each edge's `source` and `target` are indices into that list. A first row of numbers
is data, not a header, so quote a name that is a number, as in `"1","0","2"`; a first row that
mixes numbers with text is an error.

| Call | Returns |
|---|---|
| `to_numpy()`, `adjacency` | dense `int8` `(n, n)` mark matrix |
| `to_scipy_sparse()` | `scipy.sparse.csr_array` of marks |
| `oriented_edges()` | list of `(source, target, type)`, named by `labels` when set |
| `to_edges()` | structured array with fields `i`, `j`, `mark_i`, `mark_j` |
| `to_networkx()` | `networkx.DiGraph` or `networkx.MultiDiGraph` |
| `to_pcalg()` | dense `int8` `(n, n)` pcalg `amat`, for a `dag`, `cpdag`, or `pag` |
| `to_dot()` | Graphviz DOT text, named by `labels` when set |
| `endpoints(i, j)`, `neighbors(i)` | the marks on one edge; the adjacent nodes |

`GraphStructure.from_numpy`, `from_edges`, and `from_networkx` build a graph back from each. They
take each label or NetworkX node as text, `str(label)`, and raise a `ValueError` when two are the
same text, such as `1` and `"1"`. Pass the graph's `kind`: when it is omitted, they infer it from
the marks, and the marks alone do not always decide it. A `digraph` 2-cycle, for one, comes back
as a `pag` with a bidirected edge.

`to_networkx()` returns a `DiGraph` for `dag`, `cpdag`, and `digraph`, and a `MultiDiGraph` for
`pag` or `multigraph=True`. Every edge carries `endpoints=(mark_i, mark_j)`. For `dag`, `cpdag`,
and `pag`, each edge becomes one arc from the lower to the higher node index, whatever its
direction, so read `endpoints` or `oriented_edges()` rather than the arc. A `MultiDiGraph` adds the
reverse arc for `<->` and `o-o`. A `digraph` emits each arc from tail to arrowhead, so NetworkX's
cycle functions see its cycles. `from_networkx` reads `endpoints`; an edge without it is read as
`u -> v`.

For pcalg, `GraphStructure.from_pcalg(amat, kind)` reads its `amat` and `to_pcalg()` writes one for
the graph's kind. `amat.pag` (`fit@amat` from `fci` or `rfci`; `kind="pag"`, a MAG too) holds the
mark at the column node, coded `1` circle, `2` arrowhead, `3` tail, so the methods transpose it and
swap pcalg's codes for circle and tail. `amat.cpdag` (`as(fit, "amat")` from `pc`; `kind="cpdag"`
or `"dag"`) writes `i -> j` at `[j, i]`, the child's row, and an undirected edge as `1` in both
cells, so the methods transpose it too. A `digraph` has no pcalg matrix.

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
- **lingam** - pass `adjacency_matrix_.T` to `from_binary_dag`. Some estimators write `NaN` for
  an unobserved common cause; decide how to encode those pairs first.

`to_dot()` writes every edge with its endpoint marks as Graphviz arrow shapes: a directed edge runs
from tail to arrowhead, an undirected edge has `dir=none`, and any other edge has `dir=both`, with
`arrowtail` and `arrowhead` set to `normal` for an arrowhead, `odot` for a circle, and `none` for a
tail. Writing it needs no other package; Graphviz renders it.

`GraphStructure` also converts between graph kinds; a result reaches these through its graph, as
in `out.structure.to_dag(tie_break="label")`:

| Call | Returns |
|---|---|
| `cpdag.to_dag(tie_break="label")` or `cpdag.to_dag(seed=0)` | a `dag` of the CPDAG's equivalence class |
| `dag.to_mag(latent=..., selection=...)` | the `mag` over the other nodes |
| `graph.to_pag(latent=..., selection=...)` | the `pag` of a DAG or of a MAG over the other nodes |

A CPDAG with an undirected edge holds more than one DAG, so `to_dag` raises a `ValueError` that
names those edges unless you say how to pick one: `tie_break="label"` takes the DAG the node labels
decide, the same whatever the column order, and `seed` takes a reproducible random DAG of the
class, though not every DAG is equally likely. A CPDAG without an undirected edge, or a DAG, needs
neither.

A `mag`, a maximal ancestral graph, is one member of a PAG's class: it has a PAG's marks without
circles, and the metrics score it as they score a PAG. `latent` and `selection` name nodes by
label or 0-based position; `latent` defaults to the nodes that `node_types` flags. A hidden common
cause becomes `<->` in the MAG, and two ancestors of a selection variable become `--`. `to_pag`
gives the PAG that `fci` returns with an exact conditional-independence test, the fair target to
score `fci` against. `andrey.metrics.to_cpdag` returns a DAG's CPDAG. `from_numpy` never infers
`mag`, so pass `kind="mag"` for one; `to_pcalg` writes a MAG as `amat.pag`, and
`from_pcalg(amat, "mag")` reads it back.

## Hand a graph to DoWhy

DoWhy estimates a causal effect from data and a DAG. Pass it the DAG as DOT text, with the data
whose columns name the nodes:

```python
import andrey
from dowhy import CausalModel

out = andrey.pc(df)                                # a CPDAG over the columns of df
dag = out.structure.to_dag(tie_break="label")      # the DAG of its class the labels decide
model = CausalModel(data=df, treatment="x", outcome="y", graph=dag.to_dot())
estimand = model.identify_effect()
estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
estimate.value
```

DoWhy reads DOT text through pydot or pygraphviz, so install one of them with DoWhy. It reads every
edge as directed, so pass a DAG. The DAGs of one CPDAG's class can disagree on the direction of an
edge near the treatment, and the estimate follows that direction: a common cause of the treatment
and the outcome in one DAG can be a step between them in another, which changes what DoWhy adjusts
for. That is why `to_dag` asks you to pick, with `tie_break="label"` or `seed`. Settle such an
edge with `knowledge` before you estimate, or compare the estimates over several seeds. A PAG has
no single DAG to hand over.

## References

- [causal-learn graph implementation][signed-graph]: signed endpoint marks.
- [LiNGAM model definition](https://lingam.readthedocs.io/en/latest/tutorial/lingam.html):
  the coefficient convention in $x = Bx + e$.
- [pcalg `amatType`][pcalg-amat-type]: the `amat.cpdag` and `amat.pag` codes, and the row or
  column the mark belongs to.

[signed-graph]:
  https://github.com/py-why/causal-learn/blob/main/causallearn/graph/GeneralGraph.py
[pcalg-amat-type]:
  https://search.r-project.org/CRAN/refmans/pcalg/html/amatType.html
