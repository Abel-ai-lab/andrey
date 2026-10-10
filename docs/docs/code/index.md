---
name: code-index
description: The Andrey public API reference, generated from the source docstrings.
meta:
  type: code
---

# API reference

The public API of Andrey - everything exported by `andrey`, `andrey.metrics`, and `andrey.viz`,
and the dataset loaders of `andrey.data` - generated from the source docstrings.
`andrey.__version__` gives the installed version.

## Structure-learning methods

Each method takes a data matrix and returns a {py:class}`~andrey.StructureOutput`, with three
exceptions: `cdnod` also takes a domain index, `longitudinal_lingam` takes a list of datasets, and
`multi_group_direct_lingam` takes a list of datasets and returns one result per dataset. Each
method is either supported or experimental; `andrey list` prints the status.

Every method checks its data before the fit: a numeric `(n_samples, n_variables)` matrix with at
least 2 rows, no `NaN` or infinity, no constant column, and, for a DataFrame, distinct column
names. Other data raises a `ValueError` that names the argument and, for a bad cell, its row and
column. A single column gives a one-node graph.

### Supported methods

The supported methods are benchmarked compared with other packages, and a quality gate checks
them against known graphs before each release. Supported means measured, not accurate on every
workload: the [benchmarks page](../benchmarks.md) shows where a method scores worse than the empty
graph, as ICA-LiNGAM does at the benchmarks' sample size.

```{eval-rst}
.. currentmodule:: andrey

.. autosummary::
   :toctree: generated
   :nosignatures:

   pc
   fci
   ges
   boss
   grasp
   direct_lingam
   ica_lingam
```

FCI and GFCI use uncovered possibly-directed paths for their R9 and R10 tail orientations:
nodes two steps apart on each path must be non-adjacent. The same rules determine the oracle PAG
returned by `andrey.data.latent.marginal(..., "pag")`.

### Experimental methods

These methods ship and pass recovery tests, but have no published benchmark. Their API may change
without deprecation, and each warns with {py:class}`~andrey.ExperimentalWarning` on its first call
in a process.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   gfci
   cdnod
   gies
   hc
   exact_search
   calm
   multi_group_direct_lingam
   varma_lingam
   longitudinal_lingam
   pnl
   gin
```

A method becomes supported once it has:

- a benchmark campaign compared with another package, in the published summary and with a section
  on the benchmarks page;
- recovery baselines in `tests/recovery/baselines/`;
- a place in the release quality gate, `qa/quality_gate.py`;
- an example notebook.

`tests/unit/test_spec_registry.py` checks each criterion for every supported method.

## Graph structures

```{eval-rst}
.. autosummary::
   :toctree: generated

   GraphStructure
   Structure
   StructureOutput
   SummaryGraph
   TemporalStructure
```

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   describe
   seed_all
```

`andrey.config` holds the process-wide backend preference and worker count. Assign
`andrey.config.backend` or `andrey.config.num_workers` to change the default for every later call,
or use `with andrey.config(backend=..., num_workers=...):` for one block.
[Configuration](../configuration.md) lists the `ANDREY_*` environment variables and the order in
which the settings apply.

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
| {py:class}`~andrey.SearchLimitWarning` | A search stopped at its move limit: `hc` took `max_iter` moves. |

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
   :toctree: generated
   :nosignatures:

   AndreyWarning
   ExperimentalWarning
   PerformanceWarning
   BackendFallbackWarning
   SearchLimitWarning
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

A published network's data come from a linear-Gaussian model on its graph, standardized, since its
own tables are discrete; `n` (default 1000) and `seed` (default 0) set the rows. The first load
downloads the file, checks its SHA-256, and caches it; later loads need no network. The
cache directory is the `data_home` argument, else `ANDREY_DATA_DIR`, else `$XDG_CACHE_HOME/andrey`
(`~/.cache/andrey` when `XDG_CACHE_HOME` is unset). Without network access, the error names the
path to put the file at. [Getting started](../guides/getting-started.md) learns and scores the
Sachs data end to end.

```{eval-rst}
.. currentmodule:: andrey.data

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_dataset
   list_datasets
   RealDataset
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
   :toctree: generated
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
   :toctree: generated
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
   :toctree: generated
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
| `andrey --help --json` | Every method contract and the shared CLI contract |
| `andrey config` | Current settings, backend availability, the platform, and the optional extras |
| `andrey config --help` | Environment variables, defaults, effects, and precedence |
| `andrey calibrate` | Measured GES and HC worker-pool cutoffs for this machine, as settings to export |
| `andrey --skill` / `andrey --llms` | Generated package guides |

`--help` selects introspection and `--json` selects the representation: help reads no data or
configuration file and never executes the method, while `andrey run hc --data data.csv --json` runs
HC. Configuration reports list detected backends, not the backend an operation will use.

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
