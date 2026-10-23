---
name: api
description: The Andrey public API reference, generated from the source docstrings.
meta:
  type: code
---

# API reference

The public API of Andrey - everything exported by `andrey`, `andrey.metrics`, and `andrey.viz`,
the dataset loaders and simulators of `andrey.data`, and the conditional-independence tests of
`andrey.core` - generated from the source docstrings.
`andrey.__version__` gives the installed version.

## Structure-learning methods

Each method takes a data matrix and returns a {py:class}`~andrey.StructureOutput`, with four
exceptions: `cdnod` also takes a domain index, `gies` also takes which variables were intervened on
in each row, `longitudinal_lingam` takes a list of datasets, and `multi_group_direct_lingam` takes
a list of datasets and returns one result per dataset. Each method is either supported or
experimental; `andrey list` prints the status.

Every method checks its data before the fit: a numeric `(n_samples, n_variables)` matrix with at
least 2 rows, no `NaN` or infinity, no constant column, and, for a DataFrame, column names that
differ as text. Each column name becomes its node's label as text, `str(name)`, so columns named
`1` and `"1"` would share the label `"1"`, and the `ValueError` names it with both positions.
Where an argument names columns, as `Knowledge` and `categorical` do, a string is a label and an
integer is a 0-based position: the column named `1` is `"1"`, and `1` is the second column.
A constant column holds no information about the other variables, so one `ValueError` names every
constant column and asks you to drop them; resampling or subsetting rows can make a column
constant. `direct_lingam`, `ica_lingam`, and `multi_group_direct_lingam` also need at least
`n_variables + 1` rows in each dataset, the fewest with which their pruning step can estimate the
edge weights.
The categorical settings, `indep_test="chisq"` or `"gsq"` and `score_func="local_score_BDeu"` or
`"local_score_BIC_categorical"`, read text, boolean, and `category` columns as categories and need
whole numbers in a column of numbers, each distinct value of a column one category. `ges`, `hc`,
`tabu`, `boss`, `grasp`, and `exact_search` default to `score_func=None`, which picks the
linear-Gaussian BIC, `"local_score_BDeu"` when every column is categorical, or
`"local_score_cond_gauss"` when some are (not for `tabu`), the last two with a `DataTypeWarning`, as
[Data types](guides/getting-started.md#data-types) describes. The mixed-data
settings, `indep_test="degen_gauss"` or `"cond_gauss"` and `score_func="local_score_degen_gauss"` or
`"local_score_cond_gauss"`, read the categorical columns that `categorical` lists, or infer them, as
[Data types](guides/getting-started.md#data-types) describes; a categorical column of numbers
needs whole numbers. Methods that score with BIC also reject a column that repeats another up to
scale and shift or, given more rows than columns, is a linear function of other columns. Copies of
one variable cannot be told apart, so one `ValueError` names every group of copies, such as `x2`
and `x7` repeating `x1`, and asks you to keep one column per group and read its edges as the
group's; no option drops or merges the copies for you. `pc`, `fci`, `gfci`, and `cdnod` with a
partial-correlation test, `"fisherz"` or `"student_t"`, reject copies the same way before any test
runs. So do `pcmci`, `pcmciplus`, and `lpcmci`, which also reject a series that repeats another up
to `2 * tau_max` steps back and name its lagged value, such as `'a' at lag 1 repeats 'd'`. Other
data raises a `ValueError` that names the argument and, for a bad cell, its row and column. A
single column gives a one-node graph.

`pc`, `fci`, `gfci`, and `cdnod` take the same `indep_test` choices: `"fisherz"`, the default, and
`"student_t"` for continuous data, `"chisq"` and `"gsq"` for categorical data, `"degen_gauss"` and
`"cond_gauss"` for mixed data, and the nonlinear tests `"kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"`
([Conditional-independence tests](#conditional-independence-tests)).

`pc`, `fci`, `gfci`, `cdnod`, `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp`,
`exact_search`, `direct_lingam`, `ica_lingam`, `multi_group_direct_lingam`, `varma_lingam`,
`longitudinal_lingam`, `pcmci`, `pcmciplus`, `lpcmci`, `rcd`, and `bottom_up_parce_lingam` take
`knowledge`, a {py:class}`~andrey.Knowledge` of required and forbidden edges, the pairs allowed to
be adjacent, and temporal tiers; each method's `knowledge` parameter says how it honors them.
`bottom_up_parce_lingam` honors every part but required edges. For the time-series methods, an
edge `(cause, effect, lag)` is the link from `cause` at time `t - lag` to `effect` at time `t`. `pc`, `fci`, `gfci`, and `cdnod` also take `max_depth`,
the maximum conditioning-set size of their conditional-independence tests.
[Background knowledge](guides/getting-started.md#background-knowledge) shows them in use.

`ges` takes `max_parents`, a cap on every node's parents: no graph the returned CPDAG represents
gives a node more than that many parents, and some graph reaches each node's count.

```{eval-rst}
.. currentmodule:: andrey

.. autosummary::
   :toctree: api

   Knowledge
```

### Supported methods

The supported methods are benchmarked compared with other packages, and a quality gate checks
them against known graphs before each release. Supported means measured, not accurate on every
workload: the [0.1.0](benchmarks/0.1.0.md#large-graphs),
[0.2.0](benchmarks/0.2.0.md#known-limits), [0.3.0](benchmarks/0.3.0.md#known-limits),
[0.5.0](benchmarks/0.5.0.md#known-limits), and [0.7.0](benchmarks/0.7.0.md#limits) release
cards show where a method scores worse than the empty graph, or about as well, as ICA-LiNGAM does
at the 0.1.0 card's sample size, BottomUpParceLiNGAM does from 8 observed variables, hill climbing
and Tabu search do from 50 variables with ten rows per variable, PCMCI does from 40 series, Granger
does from 20 series at 1,000 time steps, and GES does at 200 variables with ten rows per variable.

```{eval-rst}
.. currentmodule:: andrey

.. autosummary::
   :toctree: api
   :nosignatures:

   pc
   fci
   gfci
   cdnod
   ges
   gies
   xges
   hc
   tabu
   exact_search
   calm
   boss
   grasp
   direct_lingam
   ica_lingam
   multi_group_direct_lingam
   anm
   pnl
   rcd
   bottom_up_parce_lingam
   gin
   varma_lingam
   longitudinal_lingam
   pcmci
   pcmciplus
   lpcmci
   granger
```

FCI and GFCI use uncovered paths for their R5, R9, and R10 tail orientations: nodes two steps
apart on each path must be non-adjacent. R5 considers every circle path, up to a fixed search
bound. R8 checks each pair of endpoints in both orders, so it does not depend on the node labels.
The rules run in rounds, each setting together the marks every rule implies on the graph as it
stands and leaving a circle where two of them disagree, so they do not depend on the node labels.
The same rules determine the oracle PAG returned by `andrey.data.latent.marginal(..., "pag")`.
Their possible-d-separation step visits the pairs and nodes by a rank computed from each column's
own data, so with `possible_dsep="order_free"`, the default, their result does not depend on the
column order unless two columns are identical; `"fast"` visits them in column order and is the
default under `andrey.config(tie_break="position")`.

### Experimental methods

Every method is supported. A new method ships as experimental: it passes recovery tests but has no
published benchmark, its API may change without deprecation, and it warns with
{py:class}`~andrey.ExperimentalWarning` on its first call in a process. It becomes supported once it
has:

- a benchmark campaign compared with another package, in a published summary and with a section
  in its release's [release card](benchmarks.md);
- recovery baselines in `tests/recovery/baselines/`;
- a place in the release quality gate, `qa/quality_gate.py`;
- an example notebook.

`tests/unit/test_spec_registry.py` checks each criterion for every supported method.

### Conditional-independence tests

PC, FCI, GFCI, and CD-NOD take their conditional-independence test as `indep_test`:

| `indep_test` | Test | Status |
|---|---|---|
| `"fisherz"` | Partial correlation with Fisher's z-transform: linear effects, Gaussian noise | Supported |
| `"student_t"` | Partial correlation with a Student-t p-value, exact for Gaussian data at any size | Supported |
| `"chisq"`, `"gsq"` | Pearson's chi-square and G-square on category codes | Supported |
| `"degen_gauss"`, `"cond_gauss"` | Degenerate and conditional Gaussian likelihood ratios on mixed data | Experimental |
| `"kci"` | Kernel conditional-independence test, with a gamma approximation of its null | Supported |
| `"rcot"` | Random-feature approximation of the kernel test; time linear in the rows | Supported |
| `"cmiknn"` | Nearest-neighbor conditional mutual information, with a local permutation null | Supported |
| `"gpdc"` | Gaussian-process regression, then a distance-correlation test of the residuals | Supported |

The nonlinear tests, the last four, detect nonlinear dependence that partial correlation misses,
at a far higher cost per test. They are benchmarked and checked in the release quality gate. Inside
the search, KCI uses every BLAS thread from 2,000 rows and the other tests use one. On every thread,
a p-value repeats for a fixed thread count and can differ in the last digits between thread counts.
By name, a test runs with its default settings.
{py:func}`~andrey.core.ci.make_indep_test` passes keyword settings to the test it builds; to run
other settings in a search, register a variant with {py:func}`~andrey.core.ci.register_indep_test`
and select it by its new name:

```python
from functools import partial

import andrey
from andrey.core.ci import make_indep_test, register_indep_test

test = make_indep_test("cmiknn", data, k_cmi=5)  # five neighbors
p = test(0, 1, [2])
cmi = test.statistic(0, 1, [2])  # the estimate itself, in nats

register_indep_test("cmiknn_k5", partial(make_indep_test, "cmiknn", k_cmi=5))
andrey.pc(data, indep_test="cmiknn_k5")
```

{py:meth}`CMIknn.statistic <andrey.core.ci_cmiknn.CMIknn.statistic>` returns the conditional
mutual information estimate the test's p-value is built on, a dependence measure of its own.

```{eval-rst}
.. currentmodule:: andrey.core

.. autosummary::
   :toctree: api
   :nosignatures:

   ci.FisherZ
   ci.StudentT
   ci_kernel.KCI
   ci_kernel.RCoT
   ci_cmiknn.CMIknn
   ci_gpdc.GPDC
   ci.CITest
   ci.register_indep_test
   ci.available_indep_tests
   ci.make_indep_test

.. currentmodule:: andrey
```

## Graph structures

`GraphStructure` also reads and writes pcalg's `amat`, with `from_pcalg` and `to_pcalg`;
[Graph representation](guides/graph-representation.md) gives the conventions.

```{eval-rst}
.. autosummary::
   :toctree: api

   GraphStructure
   Structure
   StructureOutput
   SummaryGraph
   TemporalStructure
```

## Bootstrap

`andrey.bootstrap` refits a method on resamples of the rows and reports how often each edge appears,
and the graph that keeps the pairs that are adjacent often enough. It takes a method's name, such
as `"pc"`, or the method itself, and the method's own arguments. Each resample draws the rows
with replacement from a generator that depends on `seed` and the resample's number alone, so a seed
gives the same result at any `num_workers`, and a longer run begins with the shorter run's fits.
A method that takes a row-aligned input (`cdnod`'s `domain_index`, `gies`'s `interventions`)
resamples it with the rows; `multi_group_direct_lingam` resamples each dataset on its own and
returns one result per dataset; `longitudinal_lingam` resamples the units, the same rows of every
dataset. A time-series method on one series resamples blocks of `block_length` consecutive rows.
`gin` is the one method it does not take, since its number of latent nodes differs from fit to fit.

{py:class}`~andrey.BootstrapOutput` holds the counts. `frequency("->")` is an `(n, n)` array whose
cell `[i, j]` is the share of fits with the edge `i -> j`, and the same for `"<-"`, `"--"`,
`"<->"`, `"o->"`, `"<-o"`, `"o-o"`, `"-o"`, `"o-"`, and `"absent"`; `table()` lists every edge that
appeared; `structure` is the averaged graph, and `average(threshold)` builds it at another
threshold without refitting. A pair is in the averaged graph when it is adjacent in at least
`threshold` of the fits, and it takes the edge it has most often; two edges that tie give `--`
(`o-o` for a PAG). Because a resample repeats rows, a conditional-independence test sees more
evidence than the resample holds, so read a frequency as a ranking of the edges, not as a
probability. `andrey bootstrap` runs the same from the shell
([Command-line interface](#command-line-interface)).

```{eval-rst}
.. autosummary::
   :toctree: api

   bootstrap
   BootstrapOutput
```

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   describe
   seed_all
```

`andrey.config` holds the process-wide backend preference, worker count, and tie-break rule. Assign
`andrey.config.backend`, `andrey.config.num_workers`, or `andrey.config.tie_break` to change the
default for every later call, or use `with andrey.config(backend=..., num_workers=...,
tie_break=...):` for one block. [Configuration](configuration.md) lists the `ANDREY_*`
environment variables and the order in which the settings apply, and
[Column order and machines](guides/column-order.md) describes `tie_break`.

## Warnings

Every warning Andrey raises is an {py:class}`~andrey.AndreyWarning`, a `UserWarning`. A warning
never changes a result. Each distinct message warns once per process, at the line in your code that
made the call. A warning from a dependency keeps its own category, such as scikit-learn's
`ConvergenceWarning` from `ica_lingam`.

| Category | When |
|---|---|
| {py:class}`~andrey.ExperimentalWarning` | The first call of an experimental method. |
| {py:class}`~andrey.PerformanceWarning` | A call that will run correctly but slowly: a PC conditioning pass of at least ten million CI tests. |
| {py:class}`~andrey.BackendFallbackWarning` | A requested backend is unavailable, so the call runs on another one. |
| {py:class}`~andrey.SearchLimitWarning` | A search stopped at its move limit: `hc` or `tabu` took `max_iter` moves. |
| {py:class}`~andrey.SampleSizeWarning` | A test runs below its minimum rows, so it may reject a true independence more often than `alpha`: RCoT conditions on fewer than 500 rows. |
| {py:class}`~andrey.DataTypeWarning` | A score search's default picked BDeu because every column is categorical or the conditional Gaussian score because some are, a mixed-data setting inferred its categorical columns or found none, or a continuous one met columns that look like category codes. |

Filter the base class for all of them, or one category. A later filter takes precedence, so
the broad one goes first:

```python
import warnings
import andrey

warnings.filterwarnings("ignore", category=andrey.AndreyWarning)
warnings.filterwarnings("error", category=andrey.BackendFallbackWarning)
```

In a test suite, `pytest -W error::andrey.ExperimentalWarning` fails every test that triggers the
warning: a warning raised as an error does not count as the message's one warning. A warning that
an earlier filter ignored or recorded does count, so a later test that triggers the same message
passes. Python's own `-W` option and `PYTHONWARNINGS` cannot name Andrey's categories: Python reads
them before it can import `andrey`, and ignores the option as an invalid module name. A module field
such as `error::UserWarning:andrey` does not match, because a warning is attributed to the calling
module.
From the command line, `-W error::UserWarning` escalates Andrey's warnings along with every other
`UserWarning`; for one category, use `warnings.filterwarnings` or pytest's flag. `andrey run`
returns a fit's warnings in the `run.warnings` field of its JSON output, each with its category
name.

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   AndreyWarning
   ExperimentalWarning
   PerformanceWarning
   BackendFallbackWarning
   SearchLimitWarning
   SampleSizeWarning
   DataTypeWarning
```

## Datasets

`andrey.data.load_dataset` loads a dataset with its reference graph:

| Name | Variables | Edges | Data |
|---|---|---|---|
| `sachs` | 11 | 17 | Measured: 853 observational cells (Sachs et al., 2005) |
| `asia` | 8 | 8 | Simulated on the published network (bnlearn) |
| `alarm` | 37 | 46 | Simulated on the published network (bnlearn) |
| `hepar2` | 70 | 123 | Simulated on the published network (bnlearn) |
| `andes` | 223 | 338 | Simulated on the published network (bnlearn) |
| `diabetes` | 413 | 602 | Simulated on the published network (bnlearn) |

A published network's data come from a linear-Gaussian model on its graph, standardized, or, with
`discrete=True`, from its own conditional probability tables, each column a pandas `category` of the
network's category names; `n` (default 1000) and `seed` (default 0) set the rows. The columns come
in an order drawn from the seed, since a network file often lists its variables in causal order, and
`feature_names` and the graph's labels follow it; `shuffle=False` keeps the file's order. Sachs
keeps its file's order, which does not follow its network. The first load downloads the file, checks
its SHA-256, and caches it; later loads need no network. The cache directory is the `data_home`
argument, else `ANDREY_DATA_DIR`, else `$XDG_CACHE_HOME/andrey` (`~/.cache/andrey` when
`XDG_CACHE_HOME` is unset). Without network access, the error names the path to put the file at.
[Getting started](guides/getting-started.md) learns and scores the Sachs data end to end.

```{eval-rst}
.. currentmodule:: andrey.data

.. autosummary::
   :toctree: api
   :nosignatures:

   load_dataset
   list_datasets
   RealDataset
```

### Simulators

`andrey.data.sample_discrete` samples a discrete network, the data for the categorical tests and
scores, and `andrey.data.sample_mixed` a network of categorical and continuous variables, the data
for the mixed-data tests and scores. Each returns the data with the true graph. The columns come in
an order shuffled from the seed, with the graph relabeled to match, so the order says nothing about
the causal order; `shuffle=False` keeps the order in which the graph was drawn.

```{eval-rst}
.. currentmodule:: andrey.data

.. autosummary::
   :toctree: api
   :nosignatures:

   sample_discrete
   sample_mixed
```

## Metrics

`andrey.metrics` scores a result against a known graph. `score(output, truth)` chooses the metrics
from what the two structures carry (their kind, latent nodes, lags, an ordering, weights) and
returns them in one dictionary, with the choice under `family`. The other functions compute one
metric each. Both arguments can be a `StructureOutput`, a structure, or a directed adjacency
matrix, where a nonzero `A[i, j]` means `i -> j`, over the same variables in the same order. Wrap
a matrix of endpoint marks with `GraphStructure.from_numpy(M, kind=...)` first.

| `family` | When | `score` returns |
|---|---|---|
| `dag` | both graphs are DAGs | skeleton and arrowhead precision, recall, and F1; `shd`; `mec_shd` and `mec_arrowhead_f1`, scored through the CPDAGs; the causal-order scores when the result has an ordering; `coefficient_mae` when both carry weights |
| `cpdag` | either graph is a CPDAG | the skeleton and arrowhead scores and `shd`, with a DAG converted to its CPDAG first |
| `pag` | either graph is a PAG | the skeleton and arrowhead scores and `shd`, plus `shd_endpoint` and the scores for bidirected pairs |
| `digraph` | either graph is a `digraph` | the skeleton and arrowhead scores and `shd`, self-loops included |
| `latent` | either graph has latent nodes | `latent_cluster_ari` and the two latent counts |
| `temporal` | both are a `TemporalStructure` | scores per lag, for the lagged and contemporaneous edges, and for the summary graph |

```{eval-rst}
.. currentmodule:: andrey.metrics

.. autosummary::
   :toctree: api
   :nosignatures:

   score
   shd
   skeleton_scores
   orientation_scores
   confounder_pair_scores
   markov_equivalent
   to_cpdag
   causal_order_accuracy
   causal_order_kendall_tau
   causal_order_scores
   coefficient_mae
   direction_accuracy
   decision_rate
   temporal_scores
   latent_cluster_ari
   adjusted_rand_index
   number_of_latents
```

## Visualization

`andrey.viz` renders a `GraphStructure` to a self-contained SVG on the package palette; for a
temporal result, draw one lag's graph or its `summary_graph()`. Graphviz places
the nodes when pygraphviz, the `viz` extra, is installed, and a built-in layout otherwise.

```{eval-rst}
.. currentmodule:: andrey.viz

.. autosummary::
   :toctree: api
   :nosignatures:

   draw
   layout
   to_graphviz
   classify_edge
   SVG
   Palette
```

`andrey.viz.LIGHT` and `andrey.viz.DARK` are the built-in {py:class}`~andrey.viz.Palette`
instances; pass either as `palette=` to override the `dark` flag.

### Chart styling

`andrey.viz.mpl` applies the same palette to matplotlib, so charts and structure diagrams speak one
visual language. matplotlib is the optional `viz` extra (`uv add 'andrey-core[viz]'`); `cycle`
and `rc_params` return plain data and need no import.

```{eval-rst}
.. currentmodule:: andrey.viz.mpl

.. autosummary::
   :toctree: api
   :nosignatures:

   cycle
   rc_params
   use
   context
   cmap
```

## Command-line interface

`andrey run <method>` exposes the same methods from the shell.

| Command | Output |
|---|---|
| `andrey list` | Available methods and their status |
| `andrey run hc --help` | Readable inputs, parameters, defaults, and examples |
| `andrey run hc --help --json` | One versioned method contract, without execution |
| `andrey bootstrap hc --data data.csv` | Edge frequencies over bootstrap refits, and the averaged graph |
| `andrey --help --json` | Every method contract and the shared CLI contract |
| `andrey config` | Current settings, backend availability, the platform, and the optional extras |
| `andrey config --help` | Environment variables, defaults, effects, and precedence |
| `andrey calibrate` | Measured GES and skeleton worker-pool cutoffs for this machine, as settings to export |
| `andrey --skill` | The agent guide (`SKILL.md`): methods, settings, how to run and read results |

`--help` selects introspection and `--json` selects the representation: help reads no data or
configuration file and never executes the method, while `andrey run hc --data data.csv --json` runs
HC. Configuration reports list detected backends, not the backend an operation will use.

`andrey bootstrap <method>` takes the inputs, parameters, and options of `andrey run`, and
`--n-boot`, `--seed`, `--threshold`, and `--block-length`. Its result is a run's, with the averaged
graph as `structure`, plus `bootstrap` (the settings) and `frequencies` (each edge that appeared in
a fit, most frequent first); see [Bootstrap](#bootstrap).

Structured help, configuration and calibration reports, and run results carry
`schema_version: "2"`, which versions the command line's JSON separately from the `format_version`
of a saved result. Help and configuration print readable text unless `--json` is given; runs,
listings, and calibration print JSON when piped and accept `--json` on a terminal.

The command reference below is generated from the Typer app.

```{eval-rst}
.. typer:: andrey.cli:app
   :prog: andrey
   :show-nested:
```
