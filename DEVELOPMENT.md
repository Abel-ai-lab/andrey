# Development

How the code fits together, the common changes, and how a method is tested.
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup and the checks before a pull request;
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) covers the website. [`AGENTS.md`](AGENTS.md) holds
the repository's conventions.

## Repository layout

- `src/andrey/`: the library and its public API.
- `tests/`: correctness checks that run on every pull request.
- `qa/`: the pre-release quality gate, baseline capture, and calibration.
- `benchmarks/`: comparisons with other packages and older Andrey builds.
- `examples/`: the notebooks.
- `apps/`: the interactive examples.
- `docs/` and `site/`: the website, one Sphinx project in `docs/`, with the documentation in
  `docs/docs/` and the homepage's template and content generators in `site/`.

## One source of truth

Each public method (`andrey.pc`, `andrey.ges`, ...) has a typed
[`MethodSpec`](src/andrey/spec/registry.py). The registry generates these agent interfaces; edit
the spec to change them. Here, `SKILL.md` means the `andrey --skill` output:

```text
                       src/andrey/api/*.py          (the facade: the real callable + docstring)
                              │  signature
                              ▼
        src/andrey/spec/registry.py   ── one MethodSpec per andrey.api.__all__
                              │
        ┌─────────────┬───────┴────────┬─────────────────┐
        ▼             ▼                ▼                 ▼
   JSON tool-schema  llms.txt      SKILL.md         andrey CLI flags
   (--help --json)   (index)     (agent guide)    (run / list / config / calibrate / help)
        └───────────── all from src/andrey/spec/generate.py ──────────┘
```

The `ANDREY_*` environment system is described the same way, once, in
[`src/andrey/spec/environment.py`](src/andrey/spec/environment.py), and flows into `--help --json`,
the `--skill` guide, and `llms.txt` alongside the methods.

### The agent skill

`andrey --skill` prints the agent skill: a `SKILL.md`, with the front matter an agent needs to
install it, that covers the command-line loop, reading a result (every edge type and its symbol),
choosing a method (one line per registered method), the `ANDREY_*` variables, and the exit codes.
`render_skill` in `src/andrey/spec/generate.py` builds it from the registry and the environment
specs, so a new method or variable reaches it without an edit. To install it for an agent, write
the output where the agent reads skills, for example
`andrey --skill > ~/.claude/skills/andrey/SKILL.md`.

It is tested at two levels:

- **On every pull request**, `tests/unit/test_spec_generate.py` checks that it lists every method
  and variable, opens with the skill front matter, and explains exactly the edge types a result can
  carry; `tests/unit/test_cli.py` checks that `andrey --skill` prints it.
- **Before a release that changes it**, run agent tasks with and without the skill: give a coding
  agent a dataset and a question (Gaussian data, a hidden-confounder worry, non-Gaussian data, a
  time series), with the `andrey` command on its path, and check that it chooses a fitting method,
  runs it through `andrey run`, and reads the result correctly. The task prompts and datasets are
  kept outside the repository.

### Where each piece lives

| File | Role |
|---|---|
| `src/andrey/spec/models.py` | The frozen Pydantic models: `MethodSpec`, `DataSpec`, `InputSpec`, `ParamSpec`, `OutputSpec`, `EnvVarSpec`. |
| `src/andrey/spec/registry.py` | One `MethodSpec` per public method; parameter defaults are the source of truth. |
| `src/andrey/spec/environment.py` | Every `ANDREY_*` variable, and the precedence rules. |
| `src/andrey/core/env.py` | One parser per `ANDREY_*` variable, used by the package, `andrey config`, and `andrey run`. |
| `src/andrey/spec/generate.py` | `to_json_schema` / `help_schema` / `render_llms_txt` / `render_skill`; `SCHEMA_VERSION`, `EXIT_CODES`. |
| `src/andrey/cli.py` | The `andrey` console script (Typer): `list` / `config` / `calibrate` / one `run` command that validates `<method>` + `key=value` overrides (and `--config`) against the registry. |

## Drift guards

`tests/unit/test_spec_registry.py` checks agreement between the registry and live API:

- The registry and `andrey.api.__all__` name the same methods, both directions.
- Each spec's data inputs, keyword parameters, and **defaults** match the live `inspect.signature`
  (positional-or-keyword arg -> a `DataSpec` input; keyword-only arg -> a `ParamSpec`; `**params` ->
  `passthrough=True`).
- Shared concepts stay consistent (`alpha` is always a float; seeds are always nullable ints).
- Every `ANDREY_*` variable the source reads has an `EnvVarSpec` (add a variable, add a spec),
  and each one has one setting in `andrey.core.env` that code outside it reads.
- Every method has a status. A supported method meets each criterion listed under "Experimental
  methods" in `docs/docs/code/index.md`, and the published summary names exactly the supported
  methods. An experimental method has a recovery test, warns before it reads its inputs, and lists
  `ExperimentalWarning` under `Warns`.

`tests/unit/test_spec_generate.py` then asserts the generated surfaces stay consistent with the
specs.

## Docstring standard

Public `andrey.api` facades carry a NumPy-style docstring (`ruff.toml` pins `convention = "numpy"`,
and the `D` rules enforce it on `src/andrey/api/**`). Sphinx autodoc renders the docstring;
machine-readable metadata lives in the typed registry.

Every facade docstring has, in order:

- **Summary** - one imperative line: what it learns and the graph kind it returns ("Learn a CPDAG
  ...").
- **Extended summary** - a sentence or two on the method, where it helps.
- **Parameters** - every argument with type/shape and, for enums, the `{"a", "b"}` choices. The
  names must match the signature; the drift guard checks them against the registry.
- **Returns** - always `StructureOutput`; name the wrapped kind (`cpdag`, `dag`, `pag`, temporal)
  and the contents of `.metadata`, `.ordering`, and `.weighted_adjacency`.
- **Raises** - the error taxonomy; note which `ValueError`s enumerate valid options.
- **Examples** - a runnable doctest on tiny synthetic data that asserts a stable property (`kind`,
  a shape, `ordering`), never a repr. Seed stochastic methods (`seed=0`); skip the `[torch]`-only
  ones with `# doctest: +SKIP`.

Use `andrey.pc` as the example and follow the
[`AGENTS.md` documentation style](AGENTS.md#documentation-style).

The same sections carry over to the **public core API** that autodoc renders - the
`andrey.core` classes (`GraphStructure`, `TemporalStructure`, `SummaryGraph`, `StructureOutput`)
and their public methods. There, `Parameters`/`Returns`/`Raises` bring each argument-taking method
up to the facade shape (properties stay concise one-liners). `ruff D`
is left off the core files (they hold private helpers with a noun-phrase voice), so a pytest drift
guard pins every public core method's `Parameters` to its signature instead.

## Common changes

### Add a method

1. Implement the facade in `src/andrey/api/<family>.py` and add its name to `andrey.api.__all__`.
2. Add one `MethodSpec` to `src/andrey/spec/registry.py` (copy the nearest sibling; keep defaults
   identical to the signature). Set `status="experimental"` and call `warn_experimental("<name>")`
   first in the facade, until the method meets the criteria for supported.
3. Run the guards: `pytest tests/unit/test_spec_registry.py`. They tell you exactly what is missing
   or mismatched. The CLI, schema, llms.txt, and SKILL.md pick up the new method automatically.
4. Test it as [Test a method](#test-a-method) describes.

### Speed up a method

Dispatch numeric kernels through `andrey.core.backend` (`docs/docs/configuration.md` gives the
selection rules) and, for the greedy score-based searches, use the shared worker pool in
`src/andrey/search/_parallel.py`. Keep the recovery tests passing and the outputs equal across
Andrey's execution paths, and measure the speedup with a benchmark campaign.

### Change a parameter or default

Change it in the facade **and** its `ParamSpec` in the same commit. If they disagree, the
signature-agreement test fails. Adding an enumerated value? Put it in `ParamSpec.choices` so the
schema, the CLI's `key=value` coercion, and `andrey run <method> --help` all learn it at once.

### Add an environment variable

Read it in the code, then add an `EnvVarSpec` to `environment.py` and a line to
`docs/docs/configuration.md`. The coverage test fails until you add the spec. Update the
precedence tuple if it participates in layered selection.

## Test a method

A new method ships as experimental. It becomes supported once it passes every step below.
`tests/unit/test_spec_registry.py` checks most of them and names what is missing: a recovery test
for an experimental method; for a supported one, a recorded output, a place in the quality gate, a
section on the benchmarks page, a row in the published summary, an example notebook, and its entry
under "Supported methods" in the API reference. Review checks the rest: device parity and the
benchmarks page's "Reproduce" table.

| Step | Where | When it runs | Required for |
|---|---|---|---|
| 1. Recovery tests | `tests/recovery/` | every pull request | experimental |
| 2. Recorded outputs | `tests/recovery/baselines/` | every pull request | supported |
| 3. Device parity | `tests/gpu/` | on a GPU machine | a method with a torch path |
| 4. Quality gate | `qa/quality_gate.py` | before a release | supported |
| 5. Benchmark campaign | `benchmarks/` | on the benchmark machine | supported |
| 6. Mark it supported | registry, docs, published summary | once, in one pull request | supported |

The three directories split the work: `tests/` runs on every pull request; `qa/` checks Andrey
against itself or against known graphs, on demand or before a release; `benchmarks/` compares
Andrey with other packages and with older Andrey builds.

A pull request's tests check that the API still does what it documents: output contracts, input
validation, determinism, fast paths that must equal a reference path, and recorded outputs. How
well a method recovers a graph is a measurement for `qa/` and `benchmarks/`, not a pull-request
test. A test belongs on pull requests when it reaches a code path no other test reaches, on the
smallest input that reaches it:

- Parametrize over cases that take different branches (both sides of a size cutoff, each failure
  mode, the degenerate input); another size or seed on the same branches adds time, not coverage.
- Two entry points that share a code path need one test: GIES and GFCI call `ges()`.
- A fixed input with one certain answer is an integrity check; a threshold on a median over random
  draws is a quality measurement.
- Fit a model once per module and share the fit across the tests that inspect it.
- Nothing runs on a schedule: with dependencies locked, an unchanged commit gives the same result.

### 1. Recovery tests

Add a test module to `tests/recovery/` that calls the method. It shows the method returns the right
graph:

- **Fixed cases:** small graphs whose answer can be checked by hand, including the degenerate one,
  such as independent columns returning no edges.
- **Ground truth:** seeded data from `andrey.data` that meets the method's assumptions, scored with
  `andrey.metrics` against the true graph, with a margin a correct build clears every time.
- **What the method promises:** the same result on every run, and with more worker processes where
  it runs them.

```shell
uv run pytest tests/recovery -q
```

A failure means a wrong answer. Fix the method, not the test.

### 2. Recorded outputs

Declare in `SPEC` in `tests/recovery/helpers/tolerances.py` which of the method's output fields
must match exactly and which within a tolerance, and how much. Then add the method to `ALGORITHMS`
in `qa/capture_baselines.py`, record its output on the fixed inputs, and review the diff before
committing it:

```shell
uv run python -m qa.capture_baselines --algorithms <NAME>
```

`tests/recovery/test_regression_baselines.py` then fails whenever the output changes. A failure
means a change to review, not necessarily a wrong one; recapture when the change is intended.
[`tests/recovery/README.md`](tests/recovery/README.md) describes the format and tolerances.

### 3. Device parity

For a method with a torch path, add a case to `tests/gpu/test_device_parity.py` and run the suite
on a machine with CUDA or MPS and a device torch build
([PyTorch for a GPU](docs/docs/guides/getting-started.md#pytorch-for-a-gpu)):

```shell
ANDREY_REQUIRE_GPU=1 uv run --no-sync pytest tests/gpu -ra
```

It compares each device with the numpy result within per-device tolerances.
`ANDREY_REQUIRE_GPU=1` makes the suite fail, not skip, when no accelerator is present. No hosted
runner has a GPU, so CI does not run it: run it on your own GPU machine.

### 4. Quality gate

Add the method to `SUPPORTED` in `qa/quality_gate.py`, with the datasets whose assumptions it needs,
then record its baselines:

```shell
uv run python -m qa.quality_gate --capture
uv run python -m qa.quality_gate <NAME>
```

On each dataset the method passes when its median SHD is below the empty graph's, and within its
tolerance of the recorded baseline in either direction. The gate runs before every release and
fails the publish. [`qa/README.md`](qa/README.md#quality-gate) lists the datasets and tolerances.

### 5. Benchmark campaign

Measure the method's speed and accuracy against every package that implements it: copy
`benchmarks/benchmark.py`, set its `SOLUTIONS` and `SLICE`, add an adapter for any new package
under `benchmarks/andrey_bench/adapters/`, and run the campaign on the benchmark machine.
[`benchmarks/README.md`](benchmarks/README.md) covers environments, runs, and reading a result.

A reference package's answer is measured, not a target: where the graphs differ, report the
difference and its cause. The fit records stay outside the repository.

### 6. Mark it supported

One pull request switches the method from experimental to supported, because the checks tie these
changes together: the quality gate must cover exactly the supported methods, and the published
summary must list exactly them, so any one of the changes below without the others fails. The pull
request makes:

- `status="supported"` in its `MethodSpec`, with the `warn_experimental` call and the docstring's
  `ExperimentalWarning` entry removed;
- the method under "Supported methods" in `docs/docs/code/index.md`;
- its row in the published summary, `benchmarks/published/<release>/summary.json`, generated from
  the campaign's records;
- a section for it on `docs/docs/benchmarks.md`, with its driver in the "Reproduce" table;
- an example notebook in `examples/` that calls it;
- the quality-gate and recorded-output entries from steps 2 and 4.

```shell
uv run pytest tests/unit/test_spec_registry.py
```

## Command-line interface

Discover and run methods with:

1. `andrey list` - every method with a one-line summary.
2. `andrey run <method> --help --json` - the machine schema: data inputs, params (a JSON-schema),
   output shape, the `ANDREY_*` variables, and the CLI contract (exit codes, envelope shapes).
3. `andrey run <method> --data <path|-> [key=value ...]` - execute; a JSON envelope prints to stdout
   (hyperparameters are `key=value` or a `--config` YAML/JSON file; data arrays stay `--flags`).

`andrey config` reports the backend preference, worker setting, detected backend availability, the
machine (platform and support tier, usable cores, each optional extra with its install command, why
each unavailable backend is off), and raw environment values. It is read-only.
`andrey config --help` explains defaults, effects, and precedence from the shared specs. Both
accept `--json`; help and configuration remain readable when piped unless `--json` is given.

`--help` selects introspection and `--json` selects representation. Method help performs no data or
config-file I/O and never executes a method. `andrey run hc --json` still requests execution and
requires data. At the root, `andrey --help --json` returns every method contract.

Structured help, configuration reports, and run results carry `schema_version: "2"`. Help and
configuration reports carry `andrey_version`; results carry the package version under
`run.versions.andrey`. `SCHEMA_VERSION` versions these CLI contracts independently of graph
serialization's `format_version`; consumers must check the version before interpreting fields.
Changes within a schema version are additive. Contract details:

- **Shell-aware run and list output.** A non-interactive stdout (piped, an agent, CI) gets JSON on
  success; an interactive terminal gets a human summary, and `--json` forces JSON. Diagnostics and a
  structured error envelope always go to stderr.
- **Bounded output.** A large graph's inline `edges` are capped (`edges_truncated`); `n_edges` and
  the `edge_types` breakdown stay complete. `--full` or `-o <file>` keep the entire edge list.
- **Result envelope:** `{schema_version, method, status, structure|results, metadata, run}`, where
  `run` carries the resolved params, seed, versions, and sizes for reproducibility.
- **Error envelope:** `{error: {code, message, valid_options?, next?}}` - `next` is a list of
  runnable follow-up commands when there is an obvious fix.
- **Exit codes:** `0` ok · `2` usage/validation · `3` data error · `4` algorithm failure ·
  `1` internal (see `EXIT_CODES`).

`SKILL.md` and `llms.txt` are the human/agent onboarding text; regenerate and inspect them with
`andrey --skill` and `andrey --llms`.
