---
name: concepts
description: Causal discovery stated formally - the model, what the data identify, what each graph
  Andrey returns claims, and which methods fit which assumptions.
meta:
  type: guide
---

# Concepts

This page states the problem that Andrey's {{ method_count }} solve, what data can and cannot
settle about it, and what the graph a method returns claims. It assumes you know probability and
regression, not causal graphs. [Graph types](graph-types.md) draws each kind of graph, and
[Getting started](getting-started.md) runs the methods on real data.

## The data and the model

The data are a sample $\mathcal{D} = \{x^{(1)}, \dots, x^{(n)}\}$ of $n$ rows, drawn i.i.d. from a
distribution $P$ over the variables $V = \{X_1, \dots, X_d\}$, one column each. A column may be
continuous, categorical, or mixed. For a set of indices $S \subseteq \{1, \dots, d\}$, $X_S$ is the
vector of the variables $X_i$ with $i \in S$.

The causal model is a directed acyclic graph (DAG) $G = (V, E)$ with one structural equation per
variable:

$$X_j = f_j(X_{\mathrm{pa}_G(j)}, N_j), \qquad j = 1, \dots, d.$$

$\mathrm{pa}_G(j)$ holds the parents of $X_j$, the variables with an edge into it, and an edge
$X_i \to X_j$ says that $X_i$ is a direct cause of $X_j$ relative to $V$. The noise terms $N_j$ are
jointly independent, which is causal sufficiency: no hidden variable causes two variables of $V$.
Causal discovery estimates $G$, or the part of $G$ the data identify, from $\mathcal{D}$.

## What the data identify

$X_S$ d-separates $X_i$ from $X_k$ in $G$, written $X_i \perp_G X_k \mid X_S$, when it blocks every
path between them; the {doc}`d-separation demo <../examples/d-separation>` steps through the rule.
We write $X_i \perp\!\!\!\perp X_k \mid X_S$ when the two are independent given $X_S$ under $P$.
Two assumptions tie the graph to the distribution:

- **Causal Markov condition.** $X_i \perp_G X_k \mid X_S$ implies
  $X_i \perp\!\!\!\perp X_k \mid X_S$. Every model of the form above satisfies it.
- **Faithfulness.** The converse: every conditional independence in $P$ comes from a
  d-separation in $G$; for example, no two paths of effects cancel exactly.

Under both, the conditional independences of $P$ are exactly the d-separations of $G$, and DAGs
with the same d-separations give the same independences. So $P$ identifies $G$ only up to its
Markov equivalence class $[G]$: the DAGs with the same skeleton, the edges with their directions
dropped, and the same v-structures, the triples $X_i \to X_k \leftarrow X_l$ with $X_i$ and $X_l$
not adjacent (Verma and Pearl, 1990).

A CPDAG (completed partially directed acyclic graph) represents $[G]$. A directed edge points the
same way in every member of $[G]$; an undirected edge does not (Meek, 1995; Chickering, 2002).
Constraint-based methods such as PC test conditional independences, and score-based and
permutation-based methods such as GES and BOSS search for the graph with the best score; with no
other information, both kinds return a CPDAG. Background knowledge narrows $[G]$: required and
forbidden edges, or tiers $T_1, \dots, T_m$, a partition of $V$ in which no variable causes one in
an earlier tier.

### A worked example

Take three variables. Four DAGs share the skeleton $X_1 - X_2 - X_3$:

| The DAGs of one class $[G]$ | Their independence | Their CPDAG |
|---|---|---|
| ${X_1 \to X_2 \to X_3,}$ ${X_1 \leftarrow X_2 \leftarrow X_3,}$ and ${X_1 \leftarrow X_2 \to X_3}$ | ${X_1 \perp\!\!\!\perp X_3 \mid X_2}$ | ${X_1 - X_2 - X_3}$ |
| ${X_1 \to X_2 \leftarrow X_3}$ | ${X_1 \perp\!\!\!\perp X_3}$ | ${X_1 \to X_2 \leftarrow X_3}$ |

The chain, its reverse, and the fork in the first row imply the same independence and have no
v-structure, so observational data alone cannot orient their edges: on data from any of them, PC
returns `X1 -- X2 -- X3`. The fourth DAG, in the second row, is a v-structure. $X_1$ and $X_3$ are
independent until you condition on $X_2$, a pattern no other DAG on this skeleton gives, so PC
orients both edges.

Now let the data come from $X_1 \to X_2 \leftarrow H \to X_3$, where $H$ is not measured. Over
$X_1, X_2, X_3$ the independences are those of the v-structure, so PC, which assumes causal
sufficiency, returns `X1 -> X2 <- X3`. Its edge `X3 -> X2` is false: $X_3$ does not cause $X_2$;
$H$ causes both. FCI, which allows hidden common causes, returns `X1 o-> X2 <-o X3` on the same
data, and that claims only what holds: $X_2$ causes neither $X_1$ nor $X_3$.

## Restricted function classes

$[G]$ is as far as the data go when the $f_j$ and the distributions of the $N_j$ are left free.
Restricting them identifies $G$ itself, apart from exceptions each paper lists:

- **Linear non-Gaussian:** $X = BX + N$, where $B_{jk} \neq 0$ only if $k \in \mathrm{pa}_G(j)$,
  with non-Gaussian noise (Shimizu et al., 2006).
- **Additive noise:** $X_j = f_j(X_{\mathrm{pa}(j)}) + N_j$ (Hoyer et al., 2008; Peters et al.,
  2014). A linear $f_j$ with Gaussian noise is the main exception.
- **Post-nonlinear:** $X_j = g_j(f_j(X_{\mathrm{pa}(j)}) + N_j)$, with $g_j$ invertible (Zhang and
  Hyvärinen, 2009).

In each, the noise is independent of the causes in the causal direction and not in the reverse
one, and that asymmetry orients every edge. These methods return a DAG, many with a causal order.
Methods that fit a whole DAG to a score by continuous optimization, such as NOTEARS, return a DAG
too, but under a linear model with Gaussian noise the data identify only $[G]$: an edge the class
leaves undirected gets a direction the data do not determine.

## Hidden common causes

Without causal sufficiency, some causes are not measured, and the noise terms of the measured
variables can depend on each other. $X_i$ is an ancestor of $X_j$ when $G$ has a directed path
from $X_i$ to $X_j$. The data now identify a class of graphs over the measured variables alone, and
a PAG (partial ancestral graph) represents it (Richardson and Spirtes, 2002; Zhang, 2008). On the
edge between $X_i$ and $X_j$, the mark at $X_j$ means:

- **arrowhead:** $X_j$ is not an ancestor of $X_i$, in every member;
- **tail:** $X_j$ is an ancestor of $X_i$, in every member;
- **circle:** the members differ.

So $X_i \leftrightarrow X_j$ says that neither causes the other and that a hidden common cause
joins them. $X_i \to X_j$ says that $X_i$ causes $X_j$, perhaps through unmeasured variables, and
does not rule out a hidden common cause as well. In the example, `X1 o-> X2 <-o X3` leaves open
whether $X_3$ causes $X_2$ or a hidden variable causes both; the arrowheads at $X_2$ hold either
way. These readings assume no selection bias: no row is kept or dropped by the values of the
variables.

## What a returned graph claims

Every Andrey method returns one result type: a graph over the column labels whose edge ends carry
marks, tail, arrowhead, or circle, of the kind its method identifies. Each mark is a claim about
$G$ that holds under the method's assumptions:

- **DAG.** $X_i \to X_j$: $X_i$ is a parent of $X_j$. No edge: neither is a parent of the other.
- **CPDAG.** $X_i \to X_j$: every member of $[G]$ has this edge. $X_i - X_j$: some members point
  it each way. No edge: some $X_S$ d-separates the two, in every member.
- **PAG.** Each mark as above. No edge: some set of measured variables makes the two
  independent, so no member joins them.
- **Time-series graph.** The marks of a DAG, a CPDAG, or a PAG, on edges between variables at
  given lags (below).
- **Undirected skeleton.** $X_i - X_j$: the two are adjacent, with no claim about direction.

[Graph types](graph-types.md#which-methods-return-each-type) draws each kind and lists the methods
whose graph reads differently, such as ICP's, whose only edges point into its target.

Each claim rests on assumptions that can fail. Faithfulness fails when effects nearly cancel, and
with $n$ rows a test can miss a weak dependence. Causal sufficiency fails when a common cause is not
measured, as in the example. A function class fails when, say, the noise is close to Gaussian. A
returned graph is an estimate under those assumptions, not the causal truth. Two tools check a
result: `andrey.markov_check` tests the independences a graph implies on rows it was not learned
from, and can reject the graph but not confirm it; `andrey.bootstrap` refits on resamples and
counts how often each edge appears, a measure of stability, not of correctness
([Getting started](getting-started.md#how-far-to-trust-it)). [Standards](../standards.md) describes
how each method is tested.

## Time series

In a time series, $X_{i,t}$ is variable $i$ at step $t$. An edge $X_{i,t-\tau} \to X_{j,t}$ has lag
$\tau \ge 0$, where $\tau = 0$ is a same-step edge, and the structure is the same at every $t$
(Runge et al., 2019; Runge, 2020). Time orients every edge with $\tau \ge 1$, since a cause comes
before its effect; same-step edges are oriented as in a DAG. PCMCI+ leaves the same-step edges the
data cannot orient undirected, a CPDAG at lag 0; LPCMCI allows hidden variables and uses the marks
of a PAG. Granger causality and PCMCI assume every effect takes at least one step, so their lags
start at 1, and Longitudinal LiNGAM lets the structure change from one time point to the next.
Andrey returns a `TemporalStructure`, one graph per lag
([Graph types](graph-types.md#temporal-graphs)).

## Environments and interventions

Data can come from several environments $e = 0, \dots, K$. An intervention in environment $e$
changes the equations of its targets $I_e \subseteq V$, setting each target or changing its $f_j$,
and leaves the others as they are; an observational environment has $I_e = \emptyset$. The targets
can be known or unknown. DAGs that also agree on every environment's distribution form an
interventional equivalence class, which refines $[G]$ and directs more edges (Hauser and Bühlmann,
2012). The methods that search that class return a CPDAG of it, and those that learn unknown
targets report them.

ICP asks about one target variable $X_j$: it accepts each set $S$ for which $P(X_j \mid X_S)$ stays
the same across the environments (Peters et al., 2016). When no environment intervenes on $X_j$, its
parents form one such set. ICP returns the variables in every accepted set; at test level $\alpha$,
they are all parents of $X_j$ with probability at least $1 - \alpha$. Other methods read the
environment as an index, one more variable that no measured variable causes; its edges show which
mechanisms change.
[Environments and interventions](getting-started.md#environments-and-interventions) shows the calls.

## Which method for which assumptions

Start from what you are willing to assume about your data, and read the method family and the graph
across the row. Each method's docstring states its setting in a **When to use.** paragraph and
where it stops working in a **Limitations.** paragraph.

| You assume | Family and methods | Start with | Graph returned |
|---|---|---|---|
| One table, no hidden common causes, Markov and faithfulness, a test or score that fits the data | Constraint-, score-, and permutation-based: {{ class_cpdag_methods }} | `pc`, `ges`, `boss` | CPDAG; {{ skeleton_methods }} returns the skeleton |
| One table, hidden common causes allowed, Markov and faithfulness | Constraint- and permutation-based: {{ class_pag_methods }} | `fci`, `gfci` | PAG |
| No hidden common causes, a restricted function class | Functional and pairwise: {{ identified_dag_methods }} | `direct_lingam`, `cam`, `resit` | DAG |
| No hidden common causes, the model the method's score assumes | Continuous optimization: {{ optimized_dag_methods }} | `calm`, `notears` | DAG, one member of $[G]$ when only $[G]$ is identified |
| Hidden common causes, a restricted function class | Hidden variables: {{ latent_methods }} | `rcd`, `camuv` | PAG marks, read as [Graph types](graph-types.md#pag-uncertainty-with-hidden-variables) explains; {{ latent_nodes_methods }} returns a DAG with latent nodes |
| A time series with the same structure at every step | Time series: {{ temporal_methods }} | `pcmciplus`, `lpcmci`, `varma_lingam` | Time-series graph |
| Several environments with known or unknown targets | Interventional: {{ targets_methods }} | `gies`, `ut_igsp`, `icp` | CPDAG of the interventional class; ICP the parents of one target |
| Several environments, read as an index no variable causes | Environment index: {{ env_index_methods }} | `cdnod`, `fci_jci` | CPDAG or PAG, with the index's edges in `metadata` |

## References

- [Verma and Pearl (1990), Equivalence and Synthesis of Causal Models][verma], Proceedings of the
  Sixth Conference on Uncertainty in Artificial Intelligence.
- [Meek (1995), Causal Inference and Causal Explanation with Background Knowledge][meek],
  Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence.
- [Chickering (2002), Learning Equivalence Classes of Bayesian-Network Structures][chickering],
  Journal of Machine Learning Research 2.
- [Richardson and Spirtes (2002), Ancestral Graph Markov Models][richardson], The Annals of
  Statistics 30(4).
- [Zhang (2008), On the Completeness of Orientation Rules for Causal Discovery in the Presence of
  Latent Confounders and Selection Bias][zhang], Artificial Intelligence 172(16-17).
- [Shimizu, Hoyer, Hyvärinen, and Kerminen (2006), A Linear Non-Gaussian Acyclic Model for Causal
  Discovery][shimizu], Journal of Machine Learning Research 7.
- [Hoyer, Janzing, Mooij, Peters, and Schölkopf (2008), Nonlinear Causal Discovery with Additive
  Noise Models][hoyer], Advances in Neural Information Processing Systems 21.
- [Peters, Mooij, Janzing, and Schölkopf (2014), Causal Discovery with Continuous Additive Noise
  Models][peters2014], Journal of Machine Learning Research 15.
- [Zhang and Hyvärinen (2009), On the Identifiability of the Post-Nonlinear Causal Model][pnl],
  Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence.
- [Runge, Nowack, Kretschmer, Flaxman, and Sejdinovic (2019), Detecting and Quantifying Causal
  Associations in Large Nonlinear Time Series Datasets][runge2019], Science Advances 5(11).
- [Runge (2020), Discovering Contemporaneous and Lagged Causal Relations in Autocorrelated
  Nonlinear Time Series Datasets][runge2020], Proceedings of the 36th Conference on Uncertainty in
  Artificial Intelligence, PMLR 124.
- [Hauser and Bühlmann (2012), Characterization and Greedy Learning of Interventional Markov
  Equivalence Classes of Directed Acyclic Graphs][hauser], Journal of Machine Learning Research 13.
- [Peters, Bühlmann, and Meinshausen (2016), Causal Inference by Using Invariant Prediction:
  Identification and Confidence Intervals][icp], Journal of the Royal Statistical Society Series B
  78(5).

[verma]: https://mlanthology.org/uai/1990/verma1990uai-equivalence
[meek]: https://arxiv.org/abs/1302.4972
[chickering]: https://jmlr.org/papers/v2/chickering02a.html
[richardson]: https://doi.org/10.1214/aos/1031689015
[zhang]: https://doi.org/10.1016/j.artint.2008.08.001
[shimizu]: https://www.jmlr.org/papers/v7/shimizu06a.html
[hoyer]: https://papers.nips.cc/paper/3548-nonlinear-causal-discovery-with-additive-noise-models
[peters2014]: https://jmlr.org/papers/v15/peters14a.html
[pnl]: https://arxiv.org/abs/1205.2599
[runge2019]: https://doi.org/10.1126/sciadv.aau4996
[runge2020]: https://proceedings.mlr.press/v124/runge20a.html
[hauser]: https://jmlr.org/papers/v13/hauser12a.html
[icp]: https://doi.org/10.1111/rssb.12167
