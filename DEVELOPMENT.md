# Development

How the code fits together, the common changes, and how a method is tested.
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup and the checks before a pull request,
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) the website, and [`AGENTS.md`](AGENTS.md) the
repository's conventions.

## Repository layout

- `src/andrey/`: the library and its public API.
- `tests/`: correctness checks that run on every pull request.
- `qa/`: the pre-release quality gate, the identity check, baseline capture, and calibration.
- `benchmarks/`: comparisons with other packages and older Andrey builds.
- `examples/`: the notebooks.
- `apps/`: the interactive examples.
- `docs/` and `site/`: the website, one Sphinx project in `docs/` (the documentation in
  `docs/docs/`), with the homepage's template and content generators in `site/`.

## One source of truth

Each public method (`andrey.pc`, `andrey.ges`, ...) is a facade in `src/andrey/api/` with one
typed [`MethodSpec`](src/andrey/spec/registry.py) per name in `andrey.api.__all__`.
`src/andrey/spec/generate.py` builds every agent interface from the registry: the JSON tool schema
(`--help --json`), the agent skill (`andrey --skill`), and the flags of `andrey run`, `list`,
`config`, and `calibrate`. Every `ANDREY_*` variable is described once, in
[`src/andrey/spec/environment.py`](src/andrey/spec/environment.py), and reaches the same surfaces.
Edit the spec, not the surfaces.

| File | Role |
|---|---|
| `src/andrey/spec/models.py` | The frozen Pydantic models: `MethodSpec`, `DataSpec`, `InputSpec`, `ParamSpec`, `OutputSpec`, `EnvVarSpec`. |
| `src/andrey/spec/registry.py` | One `MethodSpec` per public method; parameter defaults are the source of truth. |
| `src/andrey/spec/concepts.py` | The package-wide parameter rules: one `Concept` per shared parameter or behavior, with its template `ParamSpec`, the suitable and exempt methods, and each cell that still differs. |
| `src/andrey/spec/environment.py` | Every `ANDREY_*` variable, and the precedence rules (`PRECEDENCE`). |
| `src/andrey/core/env.py` | One parser per `ANDREY_*` variable, used by the package, `andrey config`, and `andrey run`. |
| `src/andrey/spec/generate.py` | `to_json_schema` / `help_schema` / `render_skill`; `SCHEMA_VERSION`, `EXIT_CODES`. |
| `src/andrey/cli.py` | The `andrey` console script (Typer): `list`, `config`, `calibrate`, `run`, `bootstrap`, and `markov-check`; `run` and `bootstrap` validate `<method>` and its `key=value` overrides (and `--config`) against the registry. |
| `src/andrey/resample.py` | `andrey.bootstrap` and `BootstrapOutput`: the refits on resamples, the counts of each edge by endpoint marks, and the averaged graph, a valid graph of the method's kind. |
| `src/andrey/markov.py` | `andrey.markov_check` and `MarkovCheckOutput`: the independences a DAG or CPDAG implies, tested against the data. |

### Counts and lists on the pages

A page that states a fact about the methods, such as how many there are, how many are supported,
or which ones a family holds, computes it from the registry when it is built; no page types it in.
A typed count goes stale without failing anything when a method is added or promoted.

- `docs/conf.py` gives the documentation pages substitutions (`{{ method_count }}`,
  `{{ n_supported }}`) from `_methods_text`.
- `site/build.py` computes the homepage's counts, including the comparison table's distinct
  methods (`COUNTED_WITH` names the methods that count with the one they extend).
- `apps/build_launch.py` writes the README's and `ROADMAP.md`'s generated blocks between their
  markers; `--check-readme` and `tests/unit/test_launch_material.py` fail when a block drifts.

A number that cannot come from the registry, such as another package's method count or a planned
method, sits in one named constant beside the code that prints it, with a comment saying where it
was read.

### The agent skill

`andrey --skill` prints a `SKILL.md`, with the front matter an agent needs to install it, covering
the command-line loop, reading a result (every edge type and its symbol), choosing a method (one
line per method), the `ANDREY_*` variables, and the exit codes. `render_skill` builds it from the
specs, so a new method or variable reaches it without an edit. To install it for an agent:
`andrey --skill > ~/.claude/skills/andrey/SKILL.md`.

An agent that has only run `pip install andrey-core` finds it from `help(andrey)`, whose docstring
(`src/andrey/__init__.py`) holds a quick start and names `andrey --skill` and
`andrey --help --json`, and from the PyPI summary (`pip show andrey-core`), which gives the import
name. `tests/unit/test_cli.py` runs the quick start and every `andrey ...` command the docstring
names.

`tests/unit/test_spec_generate.py` checks that it lists every method and variable, opens with the
front matter, and explains exactly the edge types a result can carry; `tests/unit/test_cli.py`
checks that `andrey --skill` prints it. Before a release that changes it, give a coding agent a
dataset and a question (Gaussian data, a hidden-confounder worry, non-Gaussian data, a time
series) with and without the skill, and check that it picks a fitting method, runs it through
`andrey run`, and reads the result correctly. The prompts and datasets stay outside the
repository.

## Drift guards

`tests/unit/test_spec_registry.py` keeps the registry and the live API in agreement:

- The registry and `andrey.api.__all__` name the same methods.
- Each spec's data inputs, keyword parameters, and **defaults** match `inspect.signature`
  (positional-or-keyword -> a `DataSpec` input, keyword-only -> a `ParamSpec`; no facade takes
  `**params`).
- Shared concepts stay consistent: `alpha` is a float; seeds are nullable ints.
- `tests/unit/test_concepts.py` checks every concept in `src/andrey/spec/concepts.py` against
  every method: a suitable method has the template's name, type, default, and bounds, or the
  deviation recorded for it, and its docstring entry states the concept; an exempt method lacks
  it; each recorded deviation still holds; and every paragraph of `docs/docs/api.md` marked
  `% concept: <key>` names exactly the methods that take the concept.
- Every `ANDREY_*` variable the source reads has an `EnvVarSpec` and one parser in
  `andrey.core.env`.
- Every method has a status and a **Limitations.** paragraph; the status checks are listed under
  [Test a method](#test-a-method).

`tests/unit/test_spec_generate.py` keeps the generated surfaces consistent with the specs.

## Docstring standard

Public `andrey.api` facades carry a NumPy-style docstring (`ruff.toml` pins `convention = "numpy"`
and enforces the `D` rules on `src/andrey/api/**`). Autodoc renders it; machine-readable metadata
lives in the registry. Every facade docstring has, in order:

- **Summary**: one imperative line naming what it learns and the graph kind ("Learn a CPDAG ...").
- **Extended summary**: a sentence or two on the method, where it helps.
- **Parameters**: every argument with its type or shape and, for enums, the `{"a", "b"}` choices,
  named as in the signature (the drift guard checks them).
- **Returns**: always `StructureOutput`, naming the wrapped kind (`cpdag`, `dag`, `pag`, temporal)
  and the contents of `.metadata`, `.ordering`, and `.weighted_adjacency`.
- **Raises**: the error taxonomy, noting which `ValueError`s enumerate valid options.
- **Examples**: a runnable doctest on tiny synthetic data that asserts a stable property (`kind`, a
  shape, `ordering`), never a repr. Stochastic methods are seeded (`seed=0`); `[torch]`-only ones
  carry `# doctest: +SKIP`.

`andrey.pc` is the model. The public core classes autodoc renders (`GraphStructure`,
`TemporalStructure`, `SummaryGraph`, `StructureOutput`) take the same sections on their
argument-taking methods; `ruff`'s `D` rules are off there, so
`test_public_core_methods_document_their_parameters` pins their `Parameters` to the signatures.

## Common changes

### Add a method

1. Implement the facade in `src/andrey/api/<family>.py`, decorated with `@method_spec("<name>")`,
   and add its name to `andrey.api.__all__`. The wrapper (`src/andrey/api/_method.py`) checks every
   argument against the method's `ParamSpec`, resolves a seeded method's seed, gives the
   experimental warning, and sets `metadata["algorithm"]` and `metadata["seed"]`, so the facade
   holds only the method's own logic.
   End the docstring's Notes with a **Limitations.** paragraph: "Not benchmarked yet.", the largest
   problem its tests run it on (variables, and rows where they matter), and any known structural
   limit, such as an assumption on the data or a cost that grows fast with the size.
2. Add one `MethodSpec` to `src/andrey/spec/registry.py` (copy the nearest sibling; keep defaults
   identical to the signature), with `status="experimental"` until the method is supported. A
   method whose tests delete rows per test sets `DataSpec(missing=True)` and passes `missing=True`
   to its data check. Every data check passes the method's name as `method=`, so a refusal of
   `NaN` names it and data past the measured boundary warn.
   `tests/unit/test_missing_values.py` checks both against the spec. Its
   `column_order`, `tie_break`, and `order_note` fields fill the method's row in the
   [column-order guide](docs/docs/guides/column-order.md): break ties with
   `andrey.search.hc.tie_ranks`, which follows `andrey.config.tie_break`, and set `tie_break=True`;
   `tests/unit/test_column_labels.py` checks both fields on every method.
   List the method in `src/andrey/spec/concepts.py` under each concept, as suitable (taking the
   template with `concept_param`, or a recorded deviation) or exempt with the reason;
   `tests/unit/test_concepts.py` names each concept that does not list it.
3. Run `uv run pytest tests/unit/test_spec_registry.py`, which names whatever is missing. The CLI,
   the schema, and the skill pick up the method on their own.
4. Test it as [Test a method](#test-a-method) describes.

### Speed up a method

Dispatch numeric kernels through `andrey.core.backend` (`docs/docs/configuration.md` gives the
selection rules) and, for the greedy score-based searches, use the shared worker pool in
`src/andrey/search/_parallel.py`. Keep the recovery tests passing and the outputs equal across
Andrey's execution paths, and measure the speedup with a benchmark campaign. `qa/identity.py`
shows the change returns the same bits and makes no more queries: write its manifest on the base
commit and on the change, and compare the two ([`qa/README.md`](qa/README.md#identity)).

### Change a parameter or default

Change the facade **and** its `ParamSpec` in the same commit; the signature-agreement test fails
when they differ. A new enumerated value goes in `ParamSpec.choices`, so the schema, the CLI's
`key=value` coercion, and `andrey run <method> --help` all learn it at once. A parameter that
implements a concept changes in `src/andrey/spec/concepts.py`: bringing a method to the template
deletes its deviation, and `tests/unit/test_concepts.py` fails until it does. A parameter or a
choice renamed after a release keeps its former name in `ParamSpec.aliases` or
`ParamSpec.choice_aliases` for one release, which `method_spec` maps with a
`DeprecatedArgumentWarning`; the deviation's `aliases` lists the names that shipped. A parameter
removed after a release goes in `MethodSpec.dropped_args` for one release: the wrapper drops it
with the same warning. Bounds go in the `ParamSpec`, which the wrapper checks; a facade checks only
what depends on the data or on another parameter.

A method that draws random numbers sets `seeded=True` and takes `seed` (`concept_param("seed")`),
and the wrapper resolves it and records `metadata["seed"]`. A search that tests independence
passes the seed to `make_indep_test`, which gives it to the tests in `STOCHASTIC_INDEP_TESTS` and
only to them, at every place the search builds a test. A method takes no execution setting such as
`n_jobs`; it reads them from `andrey.config`.

### Publish a boundary lane

A release summary with `boundary.ladders` sets where each method's `BoundaryWarning` starts:
regenerate its table with `uv run python scripts/boundary_table.py`, which writes
`src/andrey/api/_boundary_table.py` from the newest such summary. `tests/unit/test_boundary.py`
fails until the table matches it. Every method's data check applies the table, after the input
checks and before the fit.

### Add an environment variable

Read it in the code, then add an `EnvVarSpec` to `environment.py` (and to `PRECEDENCE` if it takes
part in layered selection) and a line to `docs/docs/configuration.md`. The coverage test fails until
the spec exists.

## Test a method

A new method ships as experimental and becomes supported once it passes every step below.
`tests/unit/test_spec_registry.py` checks most of them and names what is missing: a recovery test
for an experimental method; a **Limitations.** paragraph for every method; and for a supported
one, a recorded output, a place in the quality gate, a section in a release card, rows in a
published summary, an example notebook, and an entry under "Supported methods" in the API
reference. Review checks device parity and the card's "Reproduce" table.

| Step | Where | When it runs | Required for |
|---|---|---|---|
| 1. Recovery tests | `tests/recovery/` | every pull request | experimental |
| 2. Recorded outputs | `tests/recovery/baselines/` | every pull request | supported |
| 3. Device parity | `tests/gpu/` | on a GPU machine | a method with a torch path |
| 4. Quality gate | `qa/quality_gate.py` | before a release | supported |
| 5. Benchmark campaign | `benchmarks/` | on the benchmark machine | supported |
| 6. Mark it supported | registry, docs, published summary | once, in one pull request | supported |

`tests/` checks on every pull request that the API does what it documents: output contracts, input
validation, determinism, fast paths equal to a reference path, and recorded outputs. How well a
method recovers a graph is measured in `qa/` (against itself or known graphs, before a release) and
`benchmarks/` (against other packages and older builds). A pull-request test earns its place by
reaching a code path no other test reaches, on the smallest input that reaches it:

- Parametrize over cases that take different branches (both sides of a size cutoff, each failure
  mode, the degenerate input); another size or seed on the same branches adds time, not coverage.
- Two entry points that share a code path need one test: GES and GFCI both call `ges()`.
- A fixed input with one certain answer is an integrity check; a threshold on a median over random
  draws is a quality measurement.
- A fit made once per module is shared by the tests that inspect it.
- Nothing runs on a schedule: with dependencies locked, an unchanged commit gives the same result.

### 1. Recovery tests

A test module in `tests/recovery/` calls the method and shows it returns the right graph:

- **Fixed cases:** small graphs whose answer can be checked by hand, including the degenerate one,
  such as independent columns returning no edges.
- **Ground truth:** seeded data from `andrey.data` that meets the method's assumptions, scored with
  `andrey.metrics` against the true graph, with a margin a correct build clears every time.
- **What the method promises:** the same result on every run, and with more worker processes where
  it runs them.

```shell
uv run pytest tests/recovery -q
```

A failure means a wrong answer: fix the method, not the test.

### 2. Recorded outputs

Declare in `SPEC` in `tests/recovery/helpers/tolerances.py` which output fields must match exactly
and which within a tolerance, add the method to `ALGORITHMS` in `qa/capture_baselines.py`, record
its output on the fixed inputs, and review the diff before committing it:

```shell
uv run python -m qa.capture_baselines --algorithms <NAME>
```

`tests/recovery/test_regression_baselines.py` then fails whenever the output changes: a change to
review, not necessarily a wrong one; recapture when it is intended.
[`tests/recovery/README.md`](tests/recovery/README.md) describes the format and tolerances.

### 3. Device parity

For a method with a torch path, add a case to `tests/gpu/test_device_parity.py` and run the suite
on a machine with CUDA or MPS and a device torch build
([PyTorch for a GPU](docs/docs/guides/getting-started.md#pytorch-for-a-gpu)):

```shell
ANDREY_REQUIRE_GPU=1 uv run --no-sync pytest tests/gpu -ra
```

It compares each device with the numpy result within per-device tolerances;
`ANDREY_REQUIRE_GPU=1` fails, rather than skips, without an accelerator. No hosted runner has a
GPU, so CI does not run it.

### 4. Quality gate

Add the method to `SUPPORTED` in `qa/quality_gate.py`, with the datasets whose assumptions it
needs, then record its baselines:

```shell
uv run python -m qa.quality_gate --capture
uv run python -m qa.quality_gate <NAME>
```

On each dataset the method passes when its median SHD is below the empty graph's and within its
tolerance of the recorded baseline in either direction. The gate runs once per release, on the
release pull request into `main`, which `main` requires to pass; a push that changes only `docs/`,
`site/`, or top-level Markdown files reuses an earlier passing run, and the publish reuses that run
when the tagged commit has the same tree and runs the gate itself otherwise. `--jobs N` fits N cells
at a time.
[`qa/README.md`](qa/README.md#quality-gate) lists the datasets and tolerances. Reference parity is a
soft gate: the quality gate also flags any published benchmark row where Andrey makes more errors
than the reference package (from 0.8.0, when the paired bootstrap 95% interval of the difference
over at least 10 datasets lies above 0), unless `benchmarks/published/parity-allowlist.json` cites
the cause, but does not fail on it. `uv run python -m qa.reference_parity` runs the same check
offline and exits 1 on a flag; the release checklist reviews every flag
([reference parity](qa/README.md#reference-parity)).

### 5. Benchmark campaign

Measure speed and accuracy against every package that implements the method: copy
`benchmarks/benchmark.py`, set its `SOLUTIONS` and `SLICE`, add an adapter for a new package under
`benchmarks/andrey_bench/adapters/`, and run the campaign on the benchmark machine
([`benchmarks/README.md`](benchmarks/README.md)). A reference package's answer is measured, not a
target, but every difference needs a cause. Both packages are deterministic on the same data, so
each differing series has a first step where they diverge: fix an Andrey defect, with a test that
fails before the fix; make an arbitrary tie-break independent of column order, or, until it is, add
a `tie-break` entry to the allowlist with the row's result on the reversed column order; and where
Andrey is correct by the method's paper or the reference has a defect, add a cited entry to the
allowlist.
The fit records stay outside the repository.

### 6. Mark it supported

One pull request switches the method to supported, since the checks tie these changes together
(the quality gate covers exactly the supported methods, and the published summaries list exactly
them):

- `status="supported"` in its `MethodSpec`, with the docstring's `ExperimentalWarning` entry
  removed;
- a **Limitations.** paragraph from the campaign: the data and sizes measured, the sizes where its
  median errors reach the empty graph's, its fit times with the core count, and the size where it
  stops being practical, if the report shows one;
- the method under "Supported methods" in `docs/docs/api.md`;
- its rows in `benchmarks/published/<label>/summary.json` for the release that publishes it,
  generated from the campaign's records and passing `uv run python -m qa.reference_parity`;
- a section in that release's card, `docs/docs/benchmarks/<release>.md`, with its driver in the
  "Reproduce" table ("The release cards" in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) adds a
  card);
- an example notebook in `examples/` that calls it;
- the entries from steps 2 and 4.

## Command-line interface

The [API reference](docs/docs/api.md#command-line-interface) lists the commands, generated
from the Typer app. A change keeps this contract:

- `--help` selects introspection and `--json` the representation: method help reads no data or
  configuration file and never runs the method, while `andrey run hc --json` still runs HC and needs
  data. `andrey --help --json` returns every method contract.
- Structured help, configuration reports, and run results carry `schema_version: "2"`
  (`SCHEMA_VERSION`), versioned apart from a saved graph's `format_version`; changes within a
  version are additive, and consumers check it first. Help and configuration reports carry
  `andrey_version`; results carry it under `run.versions.andrey`.
- A non-interactive stdout (a pipe, an agent, CI) gets JSON on success, a terminal a human summary,
  and `--json` forces JSON; diagnostics and the error envelope go to stderr.
- A large graph's inline `edges` are capped (`edges_truncated`); `n_edges` and the `edge_types`
  breakdown stay complete, and `--full` or `-o <file>` keep every edge.
- Result envelope: `{schema_version, method, status, structure|results, metadata, run}`, where `run`
  carries the resolved params, seed, versions, and sizes.
- `andrey bootstrap` returns that envelope with `structure` the averaged graph, plus `bootstrap`
  (the resample settings) and `frequencies` (one row per edge that appeared in a fit); the fields
  are additive, so `schema_version` stays the same.
- `andrey markov-check` returns `{schema_version, status, markov_check, run}`, where `status` is
  the run's and `markov_check` holds the verdict (`passed`), the uniformity test, and `tests`
  (one row per tested independence, capped at the 1,000 smallest p-values with
  `tests_truncated`); the fields are additive, so `schema_version` stays the same.
- Error envelope: `{error: {code, message, valid_options?, next?}}`, where `next` lists runnable
  follow-up commands when the fix is obvious.
- Exit codes (`EXIT_CODES`): `0` success, `2` usage or validation, `3` data, `4` algorithm failure,
  `1` internal.
