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
uv add "andrey-core[torch]"   # torch backend: GPU primitives and CALM
uv add "andrey-core[viz]"     # Graphviz layouts and matplotlib styling
uv add "andrey-core[all]"     # every extra
```

With pip:

```shell
pip install andrey-core            # base (pure NumPy)
pip install "andrey-core[torch]"   # torch backend: GPU primitives and CALM
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

### From GitHub

The `main` branch on GitHub holds the latest release:

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
refits PC on subsamples, each a random half of the cells, and counts how often each link appears.
`seed` fixes the subsamples, so a rerun gives the same counts:

```{code-cell} ipython3
boot = andrey.bootstrap("pc", sachs.data, n_resamples=100, seed=0)
pd.DataFrame(boot.table()[:8]).round(2)
```

`table()` lists every link that appeared in some resample, with the direction it had there, most
frequent first. The four undirected links appear in 91 to 100 of the 100 resamples. Directions are
less stable: each arrow of the collider at pkc appears in 86 resamples, and `plc -> pip3` in 31.

`boot.structure` is the averaged graph. Each pair takes the state it has most often, with no edge
counted as one of the states, and the result is then made a valid CPDAG. It holds seven of `out`'s
eight links: plc and pip3 are adjacent in 54 of the 100 resamples, but no one edge between them is
as frequent as no edge (`plc -> pip3` in 31, no edge in 46). A stricter threshold asks a link to be
adjacent in more of the resamples, without refitting:

```{code-cell} ipython3
print(boot.average(0.95))
```

```{code-cell} ipython3
:tags: [remove-cell]

# The prose below holds only while these do.
table = pd.DataFrame(boot.table())
assert table.iloc[:4]["type"].eq("undirected").all()
assert table.iloc[:4]["count"].between(75, 100).all()
arrows = table[table["target"].eq("pkc") & table["type"].eq("directed")]
assert set(arrows["source"]) == {"p38", "jnk"} and arrows["count"].between(80, 90).all()
assert 50 <= table[table[["source", "target"]].isin(["plc", "pip3"]).all(axis=1)]["count"].sum() < 60
plc_pip3 = table[table[["source", "target"]].isin(["plc", "pip3"]).all(axis=1)]
assert plc_pip3["count"].max() < 100 - plc_pip3["count"].sum()
assert links(boot.structure) == links(out.structure) - {frozenset({"plc", "pip3"})}
assert links(out.structure) - links(boot.average(0.95)) == {
    frozenset({"plc", "pip3"}),
    frozenset({"erk", "pka"}),
}
adjacent = 1 - boot.frequency("absent")
columns = list(sachs.data.columns)
assert 0.9 < adjacent[columns.index("erk"), columns.index("pka")] <= 0.95
```

That drops the link between erk and pka, which is in 93 of the 100 resamples. Treat the counts as a
ranking of the links, not as probabilities. `resampling="bootstrap"` draws all the cells with
replacement instead; a resample then repeats some cells, so PC's tests see more evidence than it
holds and keep weak links more often than they do on the data.

### Checking a result against the data

A graph implies independences: PC's graph has no link between p38 and jnk, so given the right
variables the two must look independent in the data. `andrey.markov_check` tests the ordered local
Markov independences, each variable and each earlier non-parent given its parents, and asks whether
the p-values are uniform, as they are when all of them hold. The cells that learned the graph would
flatter it, so learn on a random half of the cells and check on the other half:

```{code-cell} ipython3
cells = np.random.default_rng(0).permutation(len(sachs.data))
learn, held = sachs.data.iloc[cells[: len(cells) // 2]], sachs.data.iloc[cells[len(cells) // 2 :]]
check = andrey.markov_check(andrey.pc(learn), held, seed=0)
print(check)
```

```{code-cell} ipython3
:tags: [remove-cell]

# The prose below holds only while these do.
assert check.passed and check.n_tests == 47
```

The data do not reject the 47 independences of PC's graph. That does not show the graph is right:
a graph with extra links implies fewer independences and passes too, so read a pass with the number
of links. A rejection is the informative result, because it means a link is missing or an arrow is
wrong. `check.table()` lists each independence with its p-value, `check.dag` is the DAG whose
independences were tested, and `andrey markov-check` runs the same from the shell. The
[Markov check](../api.md#markov-check) section gives the measured error rates.

## Your own data

Pass a pandas DataFrame, whose column names become the node names, or a NumPy array with one row per
sample and one column per variable. Most methods take that one argument and return the same kind of
result; `cdnod` and `fci_jci` also take `env`, each row's environment, and `ges`, `hc`, `tabu`,
`boss`, `grasp`, `gfci`, `boss_fci`, and `grasp_fci` take it optionally, `gies` and `igsp` also take
`env` with `targets`, the variables intervened on in each environment, `gnies` takes `env` and
learns which variables each environment's interventions changed, `icp` takes `env` and `target`, the
variable whose parents it finds, and `multi_group_direct_lingam`, `longitudinal_lingam`, and
`jpcmciplus` take a list of datasets. `andrey list` names the methods, and {doc}`graph-types`
explains which graph each one returns.

`ut_igsp` takes `env` and `targets` too, and learns the targets you do not know. It calls the rows
of one environment of `env` a setting. Map an observational environment to `[]`, a setting to the
targets you know, and a setting whose targets you do not know to `None`:

```python
from andrey import ut_igsp

result = ut_igsp(data, "condition", targets={"control": [], "drug": ["dose"], "knockout": None})
result.metadata["targets"]  # each environment's targets, known and learned, by its label
```

When each environment's rows come as their own table, pass them as a dict, or a list labeled by
position, in place of `env`:

```python
result = ut_igsp(
    {"control": control, "drug": drug, "knockout": knockout},
    targets={"control": [], "drug": ["dose"], "knockout": None},
)
```

## Background knowledge

What you already know about the system narrows the search. Pass it as an `andrey.Knowledge`,
naming each column by its label, a string, or by its 0-based position, an integer. The
[background knowledge notebook](../examples/knowledge) shows tiers orienting a graph that the data
alone leave undirected.

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
pair outside `allowed` is not adjacent. In a CPDAG, the edges the knowledge decides are directed and
Meek's rules direct the edges that follow, which gives the maximally oriented PDAG (MPDAG): the DAGs
of the equivalence class that satisfy the knowledge. IAMB, Inter-IAMB, and MMPC return undirected
graphs with a `target`, with `structure="skeleton"`, the default of `mmpc`, and with
`structure="moral"`, a choice of `iamb` and `inter_iamb`: a required edge is there without a
direction, and tiers and edges forbidden one way, which constrain orientation, do not change them.
With `structure="cpdag"`, the default of `iamb` and `inter_iamb`, they direct the edges the
knowledge decides, as in any CPDAG. In a PAG, a forbidden `a -> b` puts an arrowhead at `a`
and can leave a circle at `b`. Knowledge that contradicts itself, such as a required edge into an
earlier tier, raises a `ValueError` before the search runs. In PC, CD-NOD, and FCI, knowledge whose
directions close a cycle over the edges the search kept raises one after the adjacency search.
Labels stay attached to their columns, so the same knowledge fits the table in any column order. A
label is the column name as text: name a column called `1` with `"1"`, as the integer `1` is the
second column. On the command line, give a JSON object inline or as a file:
`andrey run pc --data data.csv knowledge=knowledge.json`.

Knowledge can also state causal paths rather than edges. `required_paths` lists the pairs `(a, b)`
in which `a` causes `b`, directly or through other variables, and `forbidden_paths` the pairs in
which it does not:

```python
knowledge = Knowledge(
    required_paths=[("smoking", "cancer")],  # smoking causes cancer, directly or not
    forbidden_paths=[("tar", "age")],        # tar causes age by no path
)
result = pc(data, knowledge=knowledge)
result.metadata["ancestor_clauses"]
```

The methods whose result is a CPDAG (PC, CD-NOD, GES, XGES, HC, Tabu, MMHC, RSMAX2, BOSS, GRaSP,
exact search, and IAMB, Inter-IAMB, and MMPC with `structure="cpdag"` and no `target`) honor them:
of the DAGs the graph they found stands for, they keep those that meet every claim and return their
MPDAG (Fang, Zhao, Liu, and He, 2025), so an edge is directed when all of those DAGs direct it
alike. With an undirected result, which the claims cannot restrict, IAMB, Inter-IAMB, and MMPC raise
`NotImplementedError`. Some claims cannot be shown by directing edges. If `smoking` causes `cancer`
exactly when it is a direct cause of `tar` or of `gene`, and the data leave both edges undirected,
`metadata["ancestor_clauses"]` lists the claim as `["smoking", ["gene", "tar"]]`. The result is the
class the search found, restricted, not a new search for the best class that meets the claims. A
claim directs edges of the graph the search found, and where that graph is wrong, a true claim can
direct an edge the wrong way: in our measurements, 1 to 11% of the edges true claims directed.

When hidden common causes are possible, `confounded` lists the pairs that share one, a variable
that is not in the data; a common cause that is in the data does not make the pair adjacent. The
PAG methods (FCI, FCI-JCI, GFCI, BOSS-FCI, GRaSP-FCI, and LPCMCI over lagged links) honor all three
parts:

```python
from andrey import fci

knowledge = Knowledge(
    required_paths=[("smoking", "cancer")],
    confounded=[("diet", "exercise")],  # a hidden common cause
    forbidden_paths=[("diet", "exercise"), ("exercise", "diet")],  # neither causes the other
)
result = fci(data, knowledge=knowledge)  # diet <-> exercise
```

On a pair the tests keep, a required path makes the edge `smoking -> cancer` and a forbidden path
puts an arrowhead at its cause; no test removes a `confounded` pair. These marks and checks assume
no selection bias: the samples were not kept or dropped by the value of a variable the measured ones
cause. DirectLiNGAM and multi-group DirectLiNGAM read `required_paths` and `forbidden_paths` to
narrow the causal order, as their paper describes (Shimizu et al., 2011). Every other method raises
`NotImplementedError` for a part it does not honor, naming the methods that do. The parts are
supported in PC, GES, BOSS, FCI, GFCI, BOSS-FCI, LPCMCI, and DirectLiNGAM: with true claims, each
measured no worse than the same fit without them and than reference packages given the same
knowledge, on the [0.16.0 release card](../benchmarks/0.16.0.md#path-and-confounder-claims). In the
other methods, which we have not measured, they are experimental. A path claim says less than an
edge and so directs less: a required path cannot restore an edge the search left out. When you know
the edge itself, state it as `required`.

Knowledge that no graph meets, such as `required_paths` that lead in a cycle, raises a `ValueError`
before the search. A true claim can still be one the graph a fit finds does not meet: on finite
data the tests can remove an edge on the only path from `smoking` to `cancer`. The fit then keeps
its graph, drops the claim, warns with `KnowledgeWarning`, and lists it:

```python
result.metadata["unmet_knowledge"]  # [] when every claim is met
# [["required_paths", "smoking", "cancer"]]
```

When only some claims hold together, a CPDAG method keeps them in order, `required_paths` and then
`forbidden_paths`, each by column position.

The time-series methods read the knowledge as links between lagged variables. A third entry gives an
edge its lag: `("x", "y", 1)` is the link from `x` at time `t - 1` to `y` at time `t`, and lag `0`
is the same step, except in PCMCI and Granger, which learn no same-step links. A forbidden pair
without a lag holds at every lag, and a required link names its lag. Tiers and `allowed` hold at
every lag, and a variable's links to its own past stay allowed:

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
| Continuous | `fisherz` (the default when no column is categorical), `student_t`, `normal_scores` (monotone effects, any marginals); for nonlinear effects, `kci`, `rcot`, `cmiknn`, and `gpdc` | `bic` (the default when no column is categorical) |
| Categorical | `chisq`, `gsq` (the default when every column is categorical) | `bdeu` (the default when every column is categorical), `discrete_bic` |
| Mixed | `degen_gauss`, `cond_gauss` (the default when some columns are categorical) | `cond_gauss` (the default when some columns are categorical), `degen_gauss` |

{{ indep_test_names }} take the tests; {{ score_func_names }} take every score; `gies` scores each
variable on the rows without an intervention on it. `gfci`, `boss_fci`, `grasp_fci`, `mmhc`, and
`rsmax2` take a score and a test that read the same kind of data, such as
`andrey.gfci(data, score_func="bdeu", indep_test="chisq")`.

### Categorical data

Each distinct value of a column is one category. A column may hold whole numbers, which need not
start at 0, or, in a DataFrame, text, booleans, or a pandas `category`.

### Default test and score

`indep_test=None` and `score_func=None`, the defaults, pick from the column types you declare: a
column is categorical when `categorical=[...]` names it, by label or 0-based position, or when it is
a DataFrame's `category`, `bool`, or text column. When every column is categorical, the tests
default to `gsq` and the scores to `bdeu`; when some are, both default to `cond_gauss`, the
conditional Gaussian; otherwise to `fisherz` and `bic`. Values never decide: a column of counts or
ratings stays numbers until you declare it. An `indep_test` or `score_func` you pass is always the
one used. The result's `metadata["indep_test"]` and `metadata["score_func"]` name the test and the
score the fit used, the one picked for `None` included.

### Mixed data

With continuous and categorical columns, choose a mixed-data test or score, such as
`andrey.pc(data, indep_test="cond_gauss")` or `andrey.ges(data, score_func="cond_gauss")`. The
degenerate Gaussian (`degen_gauss`) turns each categorical column into indicator columns and treats
every column as Gaussian, so it suits effects that are linear and add up across categories. The
conditional Gaussian (`cond_gauss`) gives each combination of categories its own Gaussian, so a
category can change a variable's spread as well as its mean; it needs several rows for each
combination of categories it conditions on. Both are experimental. With either score, `hc` and
`tabu`, and so the score phase of `mmhc` and `rsmax2`, give no categorical column a continuous
parent.

### Marking the categorical columns

Name them with `categorical`, by label (the column name as text) or 0-based position:
`andrey.pc(data, indep_test="cond_gauss", categorical=["smoker", "region"])`. Without `categorical`,
a mixed-data setting reads a DataFrame's `category`, `bool`, and text columns as categorical, since
their type declares them. Values never declare a column: a column of whole numbers that you have not
declared is continuous, so `andrey.pc(df, indep_test="cond_gauss")` reads the same columns as
`andrey.pc(df)` does when it picks `cond_gauss`. Pass `categorical=[]` for no categorical column.
The result's `metadata["categorical"]` lists the categorical columns it read.

A declared column needs a test or score that reads categories. A test or score for continuous
data, such as `fisherz` or `bic`, raises `ValueError` on a column you declare categorical, whether
`categorical` names it or its type does, since it would read the categories as numbers. Leave the
setting as `None` to pick a categorical or mixed-data one, or pass `categorical=[]` to read every
column as numbers. A categorical test or score, such as `chisq` or `bdeu`, reads every column as
categorical, declared or not.

A mixed-data setting that finds no categorical column, or gets `categorical=[]`, warns and runs as
its linear-Gaussian form: both tests become the likelihood-ratio form of the partial-correlation
test, close to `fisherz`, and the degenerate Gaussian score becomes the BIC.

### Integer-coded columns

Every method that reads columns as numbers, the default test and the linear-Gaussian BIC included,
reads a column of codes such as 0, 1, 2 as numbers, which assumes the categories are ordered and
evenly spaced. When columns of at most 10 distinct whole numbers reach such a method, or the
continuous part of a mixed-data setting, a one-time `DataTypeWarning` names those columns; if they
are categories, declare them with `categorical=[...]` or a pandas `category`, and the default test
or score switches to a categorical or mixed-data one.

### From the command line

The same settings are `key=value` arguments, such as
`andrey run pc --data data.csv indep_test=cond_gauss 'categorical=[smoker,region]'`. The data file
holds numbers only, so code each category as a whole number and name those columns in `categorical`.
The JSON result lists the categorical columns under `metadata.categorical` and the warnings under
`run.warnings`.

## Missing values

Mark a missing value with `NaN` or pandas' `NA`, or, for `andrey run`, leave its field empty or
write `NA` or `N/A` in the data file. {{ missing_methods }} accept missing values: each
conditional-independence test reads only the rows where all of its variables are observed
(test-wise deletion). This keeps rows that would be dropped if every incomplete row were deleted
first. The pair tests of ANM and PNL read the rows where both columns are observed, which for a pair
is the same deletion. Here a tenth of the Sachs measurements go missing at random:

```{code-cell} ipython3
gappy = sachs.data.mask(np.random.default_rng(0).random(sachs.data.shape) < 0.1)
out = andrey.pc(gappy)
out.metadata["rows_tested"]
```

`metadata["rows_tested"]` gives the minimum (`min`) and median number of rows read by a test.
`metadata["missing"]` names the handling the fit used; a fit on complete data has neither entry.
The argument `missing` names the handling. `missing="testwise"` is test-wise deletion.
`None`, the default, picks the method's default handling, which is `"testwise"` for every method
that takes the argument. A value a method does not list raises a `ValueError` that names the values
it takes:

```{code-cell} ipython3
andrey.pc(gappy, missing="testwise").structure == out.structure
```

With `fisherz` and `student_t`, each test computes its correlations directly from its rows, for many
tests at once. The other tests are built anew on each test's rows, which costs more. A time series
keeps every step: each row of the lagged design that reads a missing value holds it. Each test of
PCMCI, PCMCI+, or LPCMCI reads the design rows where all of its lagged variables are observed.

Deleting rows per test is sound when values are missing completely at random: the rows a test keeps
are then a random sample of all rows. When whether a value is missing depends on the variables, as
when high incomes go unreported, the rows a test keeps are a selected sample. PC's tests can then
find dependence where the population has none and keep extra edges ([Tu et al., 2019][tu]); FCI
allows for selection and stays sound when no missingness mechanism causally affects another
([Strobl, Visweswaran, and Spirtes, 2018][strobl]).

`andrey.pc(data, missing="mvpc")` corrects PC for those extra edges with Missing-Value PC
([Tu et al., 2019][tu]). It finds the variables each missingness indicator depends on, then tests
again the edges such an indicator could have kept. These tests use virtual data rebuilt from the
rows where those variables are observed. It takes `fisherz` or `student_t`;
`metadata["missingness_parents"]` names the parents it found and `metadata["corrected_edges"]` the
edges it tested again and removed. A corrected test reads the rows where its pair, its conditioning
set, and those parents are all observed. When too few rows are available, it uses the test-wise
test instead. The pair then keeps its test-wise result for that conditioning set, and
`metadata["corrected_edges"]["uncorrected"]` lists the pair. On complete data, MVPC gives the
test-wise result.

The correction holds when no indicator causes a variable, every independence in the observed rows
also holds in the missing ones, no indicator is a deterministic function of others, and no value
causes its own missingness. The data cannot check these conditions. The permutation correction also
needs linear-Gaussian variables and the paper's two conditions: given `W`, the parents of the
indicators involved, those indicators are independent of the tested variables, and conditioning on
the indicators of `W` leaves the tested relation unchanged. The second requires the rows used to
rebuild the data to preserve the independence of each pair tested. When a variable used to rebuild
the data goes missing because of a common effect of that pair, the extra edge can stay. This occurs
in the paper's Figure 1c and when two effects of the same causes each cause the other's missingness.
The paper's density-ratio correction for those cases is not implemented. `andrey.data.missingness`
simulates data whose missingness graph you know, to see where the correction helps.

On simulated data in the paper's setting, MVPC made fewer errors than test-wise deletion in 10 of
16 settings where values go missing for a reason and tied in the rest. It also tied where values
are missing completely at random. At 50 variables and 10,000 rows it made 5.1 errors against 14.3.
It does not help when every missingness indicator depends weakly on many variables. In that case,
its search for those variables can take a long time; `max_depth` bounds it. On larger graphs, 50 or
100 variables with 20 or 30% of the values missing, it made 0.7 to 3.6 more errors than test-wise
deletion in 12 settings, on two sets of datasets each. The
[0.16.0 release card](../benchmarks/0.16.0.md#what-we-measured-on-missing-data) has the numbers.

A test needs enough rows where its variables are observed: `|S| + 3` for `fisherz` and `student_t`,
with `|S|` the size of its conditioning set, and 2 for the other tests. A test with fewer rows
raises a `ValueError` that says so. Drop the columns with the most missing values, or cap the
conditioning sets with `max_depth` (`max_p_global` in LPCMCI and the same-step phases of PCMCI+). A
column with no observed value, or with one value throughout, raises before the fit, and so does
infinity, which is not a missing value. CD-NOD needs a complete `env`.

The searches whose fit reads the data only through their covariance, {{ em_methods }}, cannot delete
rows per test. They refuse missing values by default, with a message that names `missing="em"`.
They take `missing="em"` instead: structural EM (Friedman, 1997; [Gao et al., 2022][missdag])
alternates two steps. Under a linear-Gaussian model of the current graph, it replaces each row's
missing values by their expected values given the row's observed ones and adds their uncertainty.
This gives the expected covariance of the data. The search then runs on that covariance over every
row, and the model is fit again to the graph it returns. It runs 10 rounds, and `metadata["em"]`
records the observed-data log-likelihood after each:

```{code-cell} ipython3
em = andrey.ges(gappy, missing="em")
em.metadata["em"]["log_likelihood"][-1]
```

EM is sound when values are missing completely at random or at random given observed values. When
whether a value is missing depends on the missing value itself, it has no guarantee. Each fit runs
the search 10 times, and `missing="em"` takes only `score_func="bic"` and no `env`.

`missing="em"` is supported in `notears`, `hc`, and `tabu`, where it was no less accurate than a
reference implementation in any setting we measured. With about 100 rows, `hc` and `tabu` are often
no better than the empty graph, both with EM and on complete data. From 500 rows on they were better
in every setting. In the other searches, `ges` among them, `missing="em"` is experimental: no
reference implementation exists to compare them with, so the
[release card](../benchmarks/0.16.0.md#what-we-measured-on-missing-data) reports them beside
test-wise deletion, dropping rows, and mean imputation.

Every other method raises a `ValueError` that names it: its score or loss reads every variable over
the same rows, so it cannot delete rows per test. Drop the incomplete rows yourself, with
`data.dropna()` for a DataFrame, which suits values missing completely at random, or use one of the
methods above. A time series cannot lose a step without joining the steps around it, so for
VARMA-LiNGAM, Granger, DYNOTEARS, and J-PCMCI+, fill in the missing steps first.

## Backends and parallelism

Andrey reads its numeric backend and its worker count from `andrey.config`, or from the
`ANDREY_BACKEND` and `ANDREY_N_JOBS` environment variables;
[Configuration](../configuration.md) lists every setting. With more than one
worker, GES, GFCI, the skeleton search of PC and the methods built on it, and the
possible-d-separation step of FCI, FCI-JCI, and GFCI run their larger passes in worker processes
and return the same graph as with one worker;
[Parallel search](../configuration.md#parallel-search) lists the methods and tests that use them.

Starting the worker processes re-imports the script that started them. A script that runs with
more than one worker (`ANDREY_N_JOBS` above 1, or `-1` for every core) must put its entry
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

On the large sparse benchmark graphs, GES with the standard BIC (`ebic_gamma=0.0`) kept many false
edges. The default extended BIC, `ebic_gamma=1.0`, adds a penalty that grows with the number of
variables, as the {py:func}`~andrey.ges` page describes. On linear-Gaussian data from Erdos-Renyi
graphs with mean degree 2, PC's median errors stayed below the empty graph's through 400 variables
with 2,000 rows, and its fits finished within 1 h on 4 cores through 1,600; with 10 rows per
variable, FCI's stayed below the empty graph's through 200 variables. Each method's API page states
the data and sizes it was measured on, the largest sizes where its errors stayed below the empty
graph's, and what ended each measurement. See [Known limits](../benchmarks/0.1.0.md#known-limits).

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
- Shimizu et al. (2011), DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian
  Structural Equation Model, Journal of Machine Learning Research 12: DirectLiNGAM with prior
  knowledge.
- Fang, Zhao, Liu, and He (2025), On the Representation of Pairwise Causal Background Knowledge
  and Its Applications in Causal Inference, Journal of Machine Learning Research 26(229): path
  claims in a CPDAG.
- Friedman (1997), Learning Belief Networks in the Presence of Missing Values and Hidden
  Variables, Proceedings of the 14th International Conference on Machine Learning: structural EM.
- [Gao et al. (2022), MissDAG: Causal Discovery in the Presence of Missing Data with Continuous
  Additive Noise Models][missdag]: structural EM with a continuous search.

[sachs]: https://doi.org/10.1126/science.1105809
[pc]: https://doi.org/10.7551/mitpress/1754.001.0001
[ges]: https://jmlr.org/papers/v3/chickering02b.html
[lingam]: https://www.jmlr.org/papers/v7/shimizu06a.html
[meek]: https://arxiv.org/abs/1302.4972
[mpdag]: https://arxiv.org/abs/1707.02171
[strobl]: https://doi.org/10.1007/s41060-017-0094-6
[tu]: https://proceedings.mlr.press/v89/tu19a.html
[missdag]: https://arxiv.org/abs/2205.13869
