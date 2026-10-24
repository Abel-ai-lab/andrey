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
uv add "andrey-core[torch]"   # torch backend: GPU primitives
uv add "andrey-core[viz]"     # Graphviz layouts and matplotlib styling
uv add "andrey-core[all]"     # every extra
```

With pip:

```shell
pip install andrey-core            # base (pure NumPy)
pip install "andrey-core[torch]"   # torch backend: GPU primitives
pip install "andrey-core[viz]"     # Graphviz layouts and matplotlib styling
pip install "andrey-core[all]"     # every extra
```

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
uv add "andrey-core @ git+https://github.com/Abel-ai-lab/andrey@main"
pip install "andrey-core @ git+https://github.com/Abel-ai-lab/andrey@main"
```

The distribution is `andrey-core`. After the `@`, name any branch, tag, or commit to install that
one.

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

PC finds 8 of the 17 links in the known signaling network, and none outside it.

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

PC's default test, Fisher's z, assumes each variable is a linear function of its causes plus
Gaussian noise. (For effects that are not linear, `indep_test` also takes nonlinear tests; the
[API reference](../api.md#conditional-independence-tests) lists them.) These measurements are far
from Gaussian: most are strongly skewed to the right.

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

### Resampling

One run gives one graph, and a different sample of cells could give another. `andrey.bootstrap`
refits PC on resamples of the cells, drawn with replacement, and counts how often each link
appears. `seed` fixes the resamples, so a rerun gives the same counts:

```{code-cell} ipython3
boot = andrey.bootstrap("pc", sachs.data, n_boot=100, seed=0)
pd.DataFrame(boot.table()[:8]).round(2)
```

`table()` lists every link that appeared in some resample, with the direction it had there, most
frequent first. The four undirected links appear in 79 to 96 of the 100 resamples. Directions are
less stable: each arrow of the collider at pkc appears in 56 resamples, and `plc -> pip3` in 41.

`boot.structure` is the averaged graph. A pair is in it when it is adjacent in at least half of the
resamples (`threshold=0.5`), with the edge it has most often. It holds the same eight links as
`out`. A stricter threshold keeps the links the resamples agree on, without refitting:

```{code-cell} ipython3
print(boot.average(0.9))
```

```{code-cell} ipython3
:tags: [remove-cell]

# The prose below holds only while these do.
table = pd.DataFrame(boot.table())
assert table.iloc[:4]["type"].eq("undirected").all()
assert table.iloc[:4]["count"].between(75, 100).all()
arrows = table[table["target"].eq("pkc") & table["type"].eq("directed")]
assert set(arrows["source"]) == {"p38", "jnk"} and arrows["count"].between(50, 65).all()
assert 70 <= table[table[["source", "target"]].isin(["plc", "pip3"]).all(axis=1)]["count"].sum() < 90
assert links(boot.structure) == links(out.structure)
assert links(boot.average(0.9)) < links(out.structure)
assert len(links(out.structure) - links(boot.average(0.9))) == 1
```

That drops the link between plc and pip3, which is in 81 of the 100 resamples. Treat the counts as a
ranking of the links, not as probabilities. A resample repeats some cells, so PC's tests see more
evidence than it holds and keep weak links more often than they do on the data.

## Your own data

Pass a pandas DataFrame, whose column names become the node names, or a NumPy array with one row
per sample and one column per variable. Most methods take that one argument and return the same
kind of result; `cdnod` also takes a domain index, `gies` also takes which variables were
intervened on in each row, and `multi_group_direct_lingam` and `longitudinal_lingam` take a list of
datasets. `andrey list` names the methods, and {doc}`graph-types` explains which graph each one
returns.

## Background knowledge

What you already know about the system narrows the search. Pass it as an `andrey.Knowledge`,
naming each column by its label, a string, or by its 0-based position, an integer:

```python
from andrey import Knowledge, pc

knowledge = Knowledge(
    required=[("smoking", "tar")],          # edges the graph must have
    forbidden=[("cancer", "smoking")],      # edges it must not have
    tiers=[["age", "sex"], ["smoking"], ["tar", "cancer"]],  # no edge into an earlier tier
)
result = pc(data, knowledge=knowledge)
```

`allowed` lists the only pairs that may be adjacent; listing a pair permits an edge but does not
require one. A short list narrows the search, which can make it faster. PC, FCI, and CD-NOD start
their adjacency search from the allowed pairs, and GFCI's GES phase searches only inside them, so
you can screen first and then search inside the pairs that survived:
`Knowledge.from_skeleton(pc(data, max_depth=1))`. `max_depth` is the maximum conditioning-set size
of those four methods' tests, the cap pcalg names `m.max` and causal-learn's FCI names `depth`.

The result honors every part: a required edge is there and directed, a forbidden one is not, and a
pair outside `allowed` is not adjacent. In a CPDAG, the edges the knowledge decides are directed
and Meek's rules direct the edges that follow, which gives the maximally oriented PDAG (MPDAG): the
DAGs of the equivalence class that satisfy the knowledge. In a PAG, a forbidden `a -> b` puts an
arrowhead at `a` and can leave a circle at `b`. Knowledge that contradicts itself, such as a
required edge into an earlier tier, raises a `ValueError` before the search runs. In PC, CD-NOD,
and FCI, knowledge whose directions close a cycle over the edges the search kept raises one after
the adjacency search. Labels stay attached to their columns, so the same knowledge fits the table
in any column order. A label is the column name as text: name a column called `1` with `"1"`, as
the integer `1` is the second column. On the command line, give a JSON object inline or as a file:
`andrey run pc --data data.csv knowledge=knowledge.json`.

PCMCI, PCMCI+, LPCMCI, VARMA-LiNGAM, and Longitudinal LiNGAM read the knowledge as links between
lagged variables. A third entry
gives an edge its lag: `("x", "y", 1)` is the link from `x` at time `t - 1` to `y` at time `t`,
and lag `0` is the same step. A forbidden pair without a lag holds at every lag, and a required
link names its lag. Tiers and `allowed` hold at every lag, and a variable's links to its own past
stay allowed:

```python
from andrey import Knowledge, pcmciplus

knowledge = Knowledge(
    forbidden=[("y", "x", 1)],     # no link from y at t - 1 to x at t
    required=[("x", "y", 0)],      # x drives y within a step
    tiers=[["temperature"], ["x", "y"]],
)
result = pcmciplus(series, tau_max=2, knowledge=knowledge)
```

## Data types

Each test and score reads the columns as one kind of data:

| Data | `indep_test` | `score_func` |
|---|---|---|
| Continuous | `fisherz` (the default), `student_t`; for nonlinear effects, `kci`, `rcot`, `cmiknn`, and `gpdc` | `local_score_BIC` (the default when no column is categorical) |
| Categorical | `chisq`, `gsq` | `local_score_BDeu` (the default when every column is categorical), `local_score_BIC_categorical` |
| Mixed | `degen_gauss`, `cond_gauss` | `local_score_cond_gauss` (the default when some columns are categorical), `local_score_degen_gauss` |

`pc`, `fci`, `gfci`, `cdnod`, `iamb`, `inter_iamb`, `mmpc`, `mmhc`, and `rsmax2` take the tests;
`ges`, `gfci`, `hc`, `mmhc`, `rsmax2`, `boss`, `grasp`, and `exact_search` take every score, and
`tabu` the continuous and categorical ones. `gfci`, `mmhc`, and `rsmax2` take a score and a test
that read the same kind of data, such as
`andrey.gfci(data, score_func="local_score_BDeu", indep_test="chisq")`.

**Categorical data.** Each distinct value of a column is one category. A column may hold whole
numbers, which need not start at 0, or, in a DataFrame, text, booleans, or a pandas `category`.

**The default score.** `ges`, `hc`, `tabu`, `boss`, `grasp`, and `exact_search` pick their score
from the columns: a DataFrame's `category`, `bool`, or text column, or a column of at most
`max_categories` distinct whole numbers (10 by default), is categorical. When every column is
categorical, they score with `local_score_BDeu`; when some are, with `local_score_cond_gauss`, the
conditional Gaussian score. `tabu` has no mixed-data score and keeps the linear-Gaussian BIC on
mixed data. Either switch says so with a one-time `DataTypeWarning`, which names the categorical
columns of mixed data. A column of counts or ratings with that few values counts as categorical too;
to read such columns as numbers, pass the linear-Gaussian BIC, `score_func="local_score_BIC"` or
`"local_score_BIC_from_cov"`; every one of these searches takes both names. A `score_func` you pass
is always the one used. To name the categorical columns yourself, pass `categorical=[...]`
instead of the inference: naming every column picks `local_score_BDeu`, and naming some picks
`local_score_cond_gauss`, each with the same warning. Data with no categorical column keep the
linear-Gaussian BIC. The tests do not switch: `pc`, `fci`, `gfci`, and `cdnod` take
`indep_test="chisq"` or `"gsq"` for categorical data and a mixed-data test for mixed data.

**Mixed data.** With continuous and categorical columns, choose a mixed-data test or score, such as
`andrey.pc(data, indep_test="cond_gauss")` or
`andrey.ges(data, score_func="local_score_cond_gauss")`. The degenerate Gaussian (`degen_gauss`)
turns each categorical column into indicator columns and treats every column as Gaussian, so it
suits effects that are linear and add up across categories. The conditional Gaussian
(`cond_gauss`) gives each combination of categories its own Gaussian, so a category can change a
variable's spread as well as its mean; it needs several rows for each combination of categories it
conditions on. Both are experimental. With either score, `hc` gives no categorical column a
continuous parent.

**Marking the categorical columns.** Name them with `categorical`, by label (the column name as
text) or 0-based position:
`andrey.pc(data, indep_test="cond_gauss", categorical=["smoker", "region"])`. Without
`categorical`, a mixed-data setting infers them: a DataFrame's `category`, `bool`, and text columns,
and every column whose values, over all its rows, are at most `max_categories` distinct whole
numbers (10 by default). pandas does not mark a column as categorical by itself, so this follows
the rule Tetrad applies when it loads mixed data. A `DataTypeWarning` names the inferred columns and
why; check them, and pass `categorical=[...]` to choose the columns yourself or `categorical=[]`
for none. The result's `metadata["categorical"]` lists the categorical columns it read.

**No categorical column.** A mixed-data setting that finds no categorical column, or gets
`categorical=[]`, warns and runs as its linear-Gaussian form: both tests become the likelihood-ratio
form of the partial-correlation test, close to `fisherz`, and the degenerate Gaussian score becomes
the BIC.

**Integer-coded columns.** The default test and the linear-Gaussian BIC read a column of codes
such as 0, 1, 2 as numbers, which assumes the categories are ordered and evenly spaced. When columns
of at most `max_categories` distinct whole numbers reach them, a one-time `DataTypeWarning` names
those columns; if they are categories, choose a mixed-data setting.

From the command line, the same settings are `key=value` arguments, such as
`andrey run pc --data data.csv indep_test=cond_gauss 'categorical=[smoker,region]'` or
`max_categories=5`. The data file holds numbers only, so code each category as a whole number. The
JSON result lists the categorical columns under `metadata.categorical` and the warnings under
`run.warnings`.

## Missing values

Mark a missing value with `NaN` or pandas' `NA`, or, for `andrey run`, leave its field empty in the
data file. PC, FCI, CD-NOD, IAMB, Inter-IAMB, MMPC, PCMCI, PCMCI+, and LPCMCI accept missing values:
each conditional-independence test reads only the rows where all of its variables are observed
(test-wise deletion), so it keeps the rows that deleting every incomplete row first would drop. Here
a tenth of the Sachs measurements go missing at random:

```{code-cell} ipython3
gappy = sachs.data.mask(np.random.default_rng(0).random(sachs.data.shape) < 0.1)
out = andrey.pc(gappy)
out.metadata["rows_tested"]
```

`metadata["rows_tested"]` gives the fewest (`min`) and the median rows a test read; a fit on
complete data has no such entry. A time series keeps every step: each row of the lagged design that
reads a missing value holds it, and each test of PCMCI, PCMCI+, or LPCMCI reads the design rows
where all of its lagged variables are observed.

Deleting rows per test is sound when values are missing completely at random: the rows a test keeps
are then a random sample of all rows. When whether a value is missing depends on the variables, as
when high incomes go unreported, the rows a test keeps are a selected sample. PC's tests can then
find dependence where the population has none and keep extra edges ([Tu et al., 2019][tu]); FCI
allows for selection and stays sound when no missingness mechanism causally affects another
([Strobl, Visweswaran, and Spirtes, 2018][strobl]).

A test needs enough rows where its variables are observed: `|S| + 3` for `fisherz` and `student_t`,
with `|S|` the size of its conditioning set, and 2 for the other tests. A test with fewer raises a
`ValueError` that says so; drop the columns that miss the most values, or cap the conditioning sets
with `max_depth` in PC, FCI, and CD-NOD. A column with no observed value, or with one value
throughout, raises before the fit, and so does infinity, which is not a missing value. CD-NOD needs
a complete domain index.

Every other method raises a `ValueError` that names it: its score or loss reads every variable over
the same rows, so it cannot delete rows per test. Drop the incomplete rows yourself, with
`data.dropna()` for a DataFrame, which suits values missing completely at random, or use one of the
methods above. A time series cannot lose a step without joining the steps around it, so for
VARMA-LiNGAM and Granger, fill in the missing steps first.

## Backends and parallelism

Andrey reads its numeric backend and its worker count from `andrey.config`, or from the
`ANDREY_BACKEND` and `ANDREY_NUM_WORKERS` environment variables;
[Configuration](../configuration.md) lists every setting. With more than one
worker, GES and GFCI scan each pass in worker processes, and GES returns the same graph as it does
serially. PC, FCI, GFCI, and CD-NOD run the larger passes of their skeleton search in worker
processes with the `fisherz` and `student_t` tests, and return the same graph as with one worker.
`granger` fits its variables on worker threads, with the same coefficients, and needs none of the
guard below.

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

GES was more accurate with a stronger penalty on the large sparse benchmark graphs. PC is practical
to roughly 1,000 variables; FCI's errors passed the empty graph's at 800
([Large graphs](../benchmarks/0.1.0.md#large-graphs)). Each supported method's API page states the
data and sizes it was measured on. See [Known limits](../benchmarks/0.1.0.md#known-limits).

## References

- [Sachs et al. (2005), Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell
  Data][sachs].
- [Spirtes, Glymour, and Scheines (2000), Causation, Prediction, and Search][pc]: the PC algorithm.
- [Chickering (2002), Optimal Structure Identification With Greedy Search][ges]: GES.
- [Shimizu et al. (2006), A Linear Non-Gaussian Acyclic Model for Causal Discovery][lingam].
- [Meek (1995), Causal Inference and Causal Explanation with Background Knowledge][meek]: Meek's
  rules with background knowledge.
- [Perkovic, Kalisch, and Maathuis (2017), Interpreting and Using CPDAGs with Background
  Knowledge][mpdag]: the MPDAG.
- [Strobl, Visweswaran, and Spirtes (2018), Fast Causal Inference with Non-Random Missingness by
  Test-Wise Deletion][strobl]: FCI with test-wise deletion.
- [Tu et al. (2019), Causal Discovery in the Presence of Missing Data][tu]: PC with test-wise
  deletion.

[sachs]: https://doi.org/10.1126/science.1105809
[pc]: https://doi.org/10.7551/mitpress/1754.001.0001
[ges]: https://jmlr.org/papers/v3/chickering02b.html
[lingam]: https://www.jmlr.org/papers/v7/shimizu06a.html
[meek]: https://arxiv.org/abs/1302.4972
[mpdag]: https://arxiv.org/abs/1707.02171
[strobl]: https://doi.org/10.1007/s41060-017-0094-6
[tu]: https://proceedings.mlr.press/v89/tu19a.html
