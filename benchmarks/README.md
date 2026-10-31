# andrey-bench

Compare Andrey with other causal discovery packages and older Andrey builds.

## Comparisons and published measurements

[`CATALOG.md`](CATALOG.md) lists every comparison the harness runs, with its driver, its solutions,
and a page in [`comparisons/`](comparisons/) on how it is run, and every release card with the
summary under `published/` that its numbers come from. A summary is generated from the recorded
fits; those records and the script that writes it are kept outside this repository. A released card
and its folder do not change: new measurements go into the next release's card and folder. To rerun
a comparison, use its driver with the builds, versions, and settings its card lists under Reproduce.

[`published/parity-allowlist.json`](published/parity-allowlist.json) lists the rows where a
reference package makes fewer errors than Andrey by the parity rule, each with its cited cause; any
other such row fails the quality gate before a release
([reference parity](../qa/README.md#reference-parity)).

## Comparison packages

Packages:

- [causal-learn](https://github.com/py-why/causal-learn)
- [lingam](https://github.com/cdt15/lingam)
- [pcalg](https://cran.r-project.org/package=pcalg) (R)
- [bnlearn](https://www.bnlearn.com/) (R)
- [pgmpy](https://github.com/pgmpy/pgmpy) - see [Categorical data](comparisons/categorical.md)
- [Tetrad](https://github.com/cmu-phil/tetrad), through its causal-cmd jar and tetrad-lib's - see
  [Mixed data](comparisons/mixed.md), [GFCI](comparisons/gfci.md), and
  [Tetrad on continuous data](comparisons/tetrad-continuous.md)
- [xges](https://github.com/ANazaret/XGES) - uses a separate environment; see below
- [tigramite](https://github.com/jakobrunge/tigramite) - GPL-licensed; uses a separate environment
  and is compared by its outputs only; see below

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```shell
benchmarks/start.sh
export ANDREY_BENCH_PYTHON=$PWD/benchmarks/.venv-bench/bin/python
```

The runner starts each fit in `.venv-bench`. Set `$ANDREY_BENCH_PYTHON` to benchmark another
checkout.

### The R lane

R solutions require an R module and an additional setup step:

```shell
benchmarks/start_r.sh          # R package library beside .venv-bench, then rpy2
eval "$(benchmarks/start_r.sh --env)"
```

`start_r.sh --env` prints the module load and the `R_LIBS_USER` export that every process fitting an
R solution needs, including batch jobs. `$ANDREY_BENCH_R_MODULE` names the modules (an R built
`--enable-R-shlib`, plus the compiler it was built with) and `$ANDREY_BENCH_R_LIBS` the library
directory.

`rpy2` embeds R in the benchmark worker. Memory and CPU accounting and the wall cap cover R. Startup
and `library()` run in the adapter's `setup()`, before warm-up fits and outside fit timing. See
[Where the clock starts](#where-the-clock-starts).

### Separate solution environments

xges pins `numpy<2` and `numba<0.60`, which bench-env cannot satisfy. Its fits use a second
environment:

```shell
benchmarks/start_xges.sh
export ANDREY_BENCH_XGES_PYTHON=$PWD/benchmarks/.venv-xges/bin/python
```

tigramite (GPL-3.0), the reference for the CMIknn and GPDC conditional-independence tests, also
fits in an environment of its own, so it and its dependencies stay out of bench-env:

```shell
benchmarks/start_tigramite.sh
export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
```

Its adapter calls the package's public API only.

lingam 1.13.0 pins scipy 1.13.1 or older, and pygam, which lingam imports, pins scipy below 1.17.
lingam's MultiGroupDirectLiNGAM runs in a third environment, on Python 3.12:

```shell
benchmarks/start_lingam.sh
export ANDREY_BENCH_LINGAM_PYTHON=$PWD/benchmarks/.venv-lingam/bin/python
```

An adapter selects another environment by setting `python_env` to the name of the variable that
holds its interpreter. The runner starts that solution's `fit` calls there. `to_structure` and
scoring remain in `.venv-bench`, so every solution uses the same graph model and scorer. An unset
declared variable stops the run at launch.

tigramite is GPL-licensed, so it is installed only in an environment of its own, and the harness
calls its public API there and compares the graphs it returns:

```shell
benchmarks/start_tigramite.sh
export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
```

Drivers pass each additional environment to `rundir.provenance_once(..., extra_venvs={...})`.
`run_id` excludes interpreter identity; `provenance.json` records every environment used, and
`aggregate.py` exposes additional environments in `extra_environments`.

### Older Andrey builds

An older Andrey build is a benchmark contestant. Give its adapter a distinct solution name and a
`python_env` pointing to the environment containing that build. Keep the adapter in an importable
module so the worker can reconstruct it, and record that environment in `extra_venvs`. Keep a
checkout's `src/` off `PYTHONPATH` so it cannot override the selected installation. Use the same
dataset, parameters, and resource limits for both builds.

For separate campaigns, `ANDREY_BENCH_PYTHON` selects the shared fit environment. Give each build
its own output directory and retain `provenance.json`: `run_id` excludes the commit, so a new build
must not resume into an older build's results. Compare paired campaign measurements; PR checks
do not measure speed beyond one direction-only test.

## Run

```shell
$ANDREY_BENCH_PYTHON benchmarks/benchmark.py --out outputs/smoke --ladder 20 --seeds 1 --repeats 1
cd benchmarks && .venv-bench/bin/python -m pytest tests/ -q
```

The first command should end with `table: 4 rows` and `errors: 0`. The second checks the harness
against known answers.

## Writing a benchmark

Copy `benchmark.py`, then edit its `SOLUTIONS` and `SLICE`. Each adapter in
`andrey_bench/adapters/` represents one package, method, backend, and mode. Include
`baseline.empty` to bound the solutions run on the same datasets, and keep one method per file. The
harness does not submit jobs: one file is one job.

A new comparison adds a row to [`CATALOG.md`](CATALOG.md) and, when its data, settings, or outputs
go beyond what this README describes, a page in [`comparisons/`](comparisons/).

An adapter's `params` name the BIC penalty `lambda_value` and the scores `local_score_*` for every
package, the harness's recorded vocabulary; the Andrey adapters pass Andrey's `penalty_discount` and
score names (`bic`, `bdeu`, `discrete_bic`, `cond_gauss`, `degen_gauss`) to it.

### Column order

A generated dataset lists its columns in an order drawn from its seed, with its truth permuted the
same way, so a method's accuracy cannot depend on the column order the simulation used. The shuffle
is part of the generator call: a shuffled dataset has its own `dataset_id` and a label ending in
`-shuffled`, the registry records its `column_order`, and `DatasetKey(shuffle=False)` keeps the
generated order under the `dataset_id` its records carry. The PCMCI, PCMCI+, LPCMCI, and
longitudinal LiNGAM simulations, GIN's `latent_measurement`, and the GIES lane's own simulation draw
an order that carries no information about the graph and take no shuffle step. The summaries
published up to 0.6.0 were measured on the generated order and stay as recorded.

`public_datasets.py` loads the bnlearn networks with `load_dataset`, which lists their columns in an
order drawn from the seed rather than the file's order, often a causal one; each record carries
`shuffle`. Sachs keeps its file's order, which does not follow its network. The public-datasets
rows published with 0.1.0 were measured on the file's order and stay as recorded.

### Scale

A key's `standardize` coordinate sets how `andrey.data` scales the values, and it enters the
generator call, so each scale is a dataset of its own. The linear and nonlinear regimes are
`standardized`: every column is z-scored, so no solution can order the variables by their variances
(var-sortability, Reisach et al. 2021). Z-scoring leaves each variable's R², the share of its
variance the other variables explain, as it was, and in linear data that share still grows down the
causal order (R²-sortability, Reisach et al. 2023).

The `_unitless` regimes, `linear_gauss_er_unitless`, `lingam_sf_unitless`,
`latent_gauss_er_unitless`, and `latent_lingam_er_unitless`, are the four linear regimes with the
weights and noise of Herman et al. (2025): every variable has variance 1 in the population, and a
variable's weights depend only on its number of parents. Their R²-sortability is close to chance as
well. Over 20 seeds at `d = 10, 20, 50` and `n = 10 d`, its mean is 0.46 to 0.62 against 0.65 to
0.76 for the standardized regimes; the hidden-confounder ones sit at the top of that range, since
hiding a node changes what the others explain. The generator keeps the variables' dense correlation
matrix, so a unitless key takes at most 1,000 variables and raises above that; run these regimes
at small sizes. A unitless key's data are not z-scored, so its sample variances differ from 1 by
sampling error.

Every record carries its dataset's `varsortability` and `r2_sortability` under the truth, over all
directed paths between observed variables, computed once per dataset in the parent; a time series
has neither. A solution's fit receives the data matrix alone.

### Hidden confounders

`latent_gauss_er` hides one node in ten, at least one, from an Erdos-Renyi linear-Gaussian graph.
Hidden nodes are chosen among nodes with two or more children, so hiding one confounds its children.
Set `latents` on the `DatasetKey` with `datasets.latents_for`, as `fci_latent.py` does. `d` includes
hidden nodes: the data has `d - latents` columns; the stored truth keeps all `d` nodes.

Every estimate is scored against the observed PAG, including the empty graph's CPDAG. The PAG is
built untimed in the parent, once per dataset. Its cost grows about 8x per doubling of `d`. A DAG or
CPDAG estimate is scored as a PAG without canonicalization. Its tails and undirected edges count
against the PAG's circles. Use PAG methods and the empty graph on these datasets.

### Where the clock starts

Only `fit` is timed. Package imports, interpreter startup, and library loading belong in an optional
`setup()`, called once before warm-up fits.

At `--warmup 0`, the reach setting, the first fit includes any lazy imports inside `fit`.
Every adapter that imports a package declares a `setup()` except the xges package's, which imports
numba and XGES inside `fit`. Account for those imports when comparing fit times.

`setup_s` records setup duration and is included in `warmup_s`. At `--warmup 0`, they are equal and
provide no fit-time bound.

A package whose search runs in a process `fit` starts, such as Tetrad's JVM in every lane that
runs it, times that search itself: its adapter declares `search_s` and sets it in `fit` to the
seconds the process measured around the search call alone. The record keeps that time as `wall_s`
and the whole `fit` call as `process_s`, which is null for every other adapter. Its warm-up
searches run in the timed fit's process before its clock, so `warmup_s` holds only `setup_s`. See
`contracts.SolutionAdapter`.

## Flags

| flag | |
|---|---|
| `--out` | output directory, one per run. Required |
| `--ladder` | the `d` sizes, in the order given. Required |
| `--seeds` | datasets per rung. Default 3 |
| `--repeats` | timed repeats per (dataset, solution). Default 5 |
| `--warmup` | untimed fits per repeat. Default 1; use 0 for reach runs |
| `--cap-wall-s` | wall seconds one repeat may spend, imports and warm-up included. Default 1800 |
| `--cap-mem-mb` | memory one repeat may use. Default uncapped |
| `--mem-route` | cap enforcement: `poll` (default) or `slurm`. `none` with a cap is an error |
| `--cores` | core budget per fit. Defaults to the scheduler affinity mask |
| `--machine-id` | the machine name stamped on every record |
| `--resume` | continue the run in `--out`, skipping units already measured |

Every rung runs every solution. A rung that no solution finishes costs `cap_wall_s` per fit. Prefer
seeds over repeats: each seed creates a dataset, while repeats mostly measure startup again.

The cap covers imports, warmups, and the timed fit. `wall_s` covers only the timed fit, so a
solution can be killed having never been timed. Set `--cap-wall-s` for the total child runtime.
With `WARMUP = 1`, this is about twice the fit time plus import time. A child, and any worker pool
it opened, dies with the driver that started it, so a stopped run leaves nothing running. The
supervisor also kills the pool when the child exits, preserving the child's exit status in the
record.

Cost is `spawns * startup + fit time`, where
`spawns = solutions * rungs * seeds * repeats`. Each repeat starts one child, warms up, and fits.
Startup dominates below `d` ~ 200 and varies with cache state and machine load. The smoke run
uses four children, so measure `smoke_wall / 4`.

## Where output lives

Output has three levels. The harness manages only runs.

| | what it is |
|---|---|
| **results root** | directory holding many batches, named by `ANDREY_BENCH_OUTPUTS` |
| **batch** | one issue's worth of work, with its scripts and conditions. See [Batches](#batches) |
| **run** | one `--out`, driver, ladder, and `runs.parquet`; the unit managed by the harness |

```text
<results root>/
    <batch>/                        one batch
        campaign/<run>/             publishable seeds and repeats from one held node
        probes/<run>/               one seed, one repeat: ceilings and sizing only
        data/                       the shared dataset store (ANDREY_BENCH_DATA)
        scratch/                    ad-hoc scripts, submission files, and job logs
```

See [Output](#output) for the files within a run.

No code reads `ANDREY_BENCH_OUTPUTS`; it names the results root in the commands below. Drivers
write where `--out` points, and git ignores `outputs/` for local runs. Store real campaigns outside
the repository because a batch may use gigabytes.

## Batches

Batch scripts accompany the results:

```text
$ANDREY_BENCH_OUTPUTS/<batch>/
    run.sh            what was submitted
    CONDITIONS.md     what was measured, and the conditions the numbers depend on
    report.py         this batch's tables
    <driver>.py       any batch-specific driver
    campaign/  probes/  data/  scratch/
```

To start a batch:

1. Create `$ANDREY_BENCH_OUTPUTS/<batch>/`, named `<issue>-<slug>`.
2. Write `run.sh`. It points `ANDREY_BENCH_DATA` at the absolute path of `<batch>/data`, so every
   run reads the same datasets wherever it starts, and gives each driver an `--out` under
   `campaign/` or `probes/`. Job logs go to `scratch/`.
3. Record the core count, caps, isolation, ladder, and solution list in `CONDITIONS.md`.
4. Write `report.py`. It pools the batch with `andrey_bench.aggregate.pool` and keeps the rows
   whose `run_dir` starts with `campaign/` or `probes/`: `pool` reads every `records/` below the
   path it is given, so a run left in `scratch/` would join the probe rows. It checks that each
   dimension a table holds fixed - machine, build, cap, cores - takes one value, and prints the
   tables, with ratios from `andrey_bench.pairing`. It exits non-zero when a check fails.

From the repository root, run and report a batch with:

```shell
export ANDREY_BENCH_OUTPUTS=/path/to/results   # the root, above every batch
$ANDREY_BENCH_OUTPUTS/<batch>/run.sh
PYTHONPATH=benchmarks $ANDREY_BENCH_PYTHON $ANDREY_BENCH_OUTPUTS/<batch>/report.py \
    $ANDREY_BENCH_OUTPUTS/<batch>
```

Commit reusable changes such as harness fixes, comparison rules in `andrey_bench/pairing.py`, tests,
and guidance in [Reading a result](#reading-a-result).

## Output

| file | |
|---|---|
| `runs.parquet` | one row per timed fit, assembled from `records/` when the ladder finishes |
| `STATUS.json` | job, node, and units done against expected. `complete` stays false until the |
| | table assembles, so an interrupted run still shows how far it got |
| `records/` | one parquet per (dataset, solution), written as each finishes |
| `provenance.json` | package versions and BLAS of the bench venv (`$ANDREY_BENCH_PYTHON`'s when |
| | set), CPU, GPU, git sha. Captured once, so a resume keeps the recorded build |
| `run_meta.json` | flags and literals, written before the first fit |
| `data/registry.json` | each dataset's generator call and column order; under |
| | `$ANDREY_BENCH_DATA` when set |

Each fit uses a temporary compute-node directory for its pickled job, result, and start marker. The
runner removes the directory after the fit.

A record stores its data (`d`, `n`, `latents`, regime coordinates, `data_seed`, `repeat`), run
settings (`cap_wall_s`, `cap_mem_mb`, `cores`, `device`, `dtype`, `backend`, `mode`, `machine_id`, `params`),
and results (`status`, `wall_s`, `process_s`, `cpu_s`, `warmup_s`, `setup_s`, `peak_rss_mb`,
`output_hash`, `shd`, `skeleton_f1`, `arrowhead_f1`). A temporal solution's record also carries
`tau_max`, and its scores are the lag columns of [PCMCI on lagged series](comparisons/pcmci.md),
or the link columns of [PCMCI+ and LPCMCI](comparisons/pcmciplus-lpcmci.md) for a
window.

`device_id` records `cpu` or the CUDA device index and name. The backend determines the device, so a
pin inside `fit` outranks `ANDREY_DEVICE`; the record keeps the request separately as `device`.
`gpu_peak_alloc_mb` is null for CPU backends. For CUDA, it records peak tensor memory in PyTorch's
allocator during the timed fit. It excludes the reserved cache, CUDA context, and library
workspaces, so it understates the free card memory required.

Without `ANDREY_BENCH_DATA`, each run creates its own `data/` and copies every dataset it uses.
The shared store configured in [Batches](#batches) avoids those copies and gives every run identical
bytes.

`dataset_id` identifies a dataset. It is a content hash of the generator call and the `.npz`
filename. `run_id` identifies one measurement from the dataset, solution, repeat, caps, dtype,
device, backend, mode, and thread split.

`run_id` excludes the commit, so two builds in one environment have the same ID. An editable install
can also change under an ID. `provenance.json` identifies the build for each output directory. Keep
that file with records combined from several directories.

## Resume

Use `--resume` to continue an interrupted run from its completed dataset/solution units:

```shell
$ANDREY_BENCH_PYTHON benchmarks/benchmark.py --out outputs/smoke --ladder 20 \
    --seeds 1 --repeats 1 --resume
```

A part filename includes its dataset, solution, and environment. Resume skips existing part files.
A different cap, core count, or machine creates new measurements. Set `--machine-id` when the
scheduler may choose another node; the default hostname would rerun every measurement.

`--resume` compares `run_meta.json` and accepts only the same run settings. It prints any
difference. Starting without `--resume` fails when `records/` contains files.

Resume cannot detect solution edits because part filenames exclude `params` and no digest includes
the body of `fit`. Freeze the code during a run.

The run declares `cores` and `device`, so timeout records include them. The child reports `threads`,
`n_jobs`, and `device_id`; these fields are empty when the fit does not return.

Each failure produces a record. A stopped fit includes `status=timeout` or `oom` and its cap.
`error_text` describes other failures.

`status=blocked` means the step was queued and dropped. With `--mem-route slurm`, a step that asks
for more memory than the job owns waits until the wall cap. Treat that size as a coverage gap. Give
the job more memory than the cap.

Read `params` before comparing solutions because their defaults may optimize different objectives.

## Reading a result

### Before the run

- **Write down the expected result before running.**
- **Run a pilot at one seed and one repeat**, and keep its rows.
- **Include the sizes where the favorite might lose.**
- **Put `baseline.empty` in every `SOLUTIONS` list.**
- **Choose every hyperparameter, and record it in `run_id`.**
- **Use one `--cap-wall-s` everywhere.**
- **Set `--warmup 0` when measuring the largest size a method completes.**

### Conditions

- **Time on an uncontended node.**
- **Pin the node with `-w`** across whatever is compared.
- **Repeat anything surprising.**
- **Don't run a method past the size where it times out.** Take larger sizes in a separate run.

### Comparing fairly

- **Match the quantity measured by each parameter.** A penalty in deviance units is twice the same
  penalty in log-likelihood units. `adapters/bic_penalty.py` derives both BIC coefficients from one
  value. A cap on parents is not a cap on degree.
- **Feed every method the same data through the same scorer.**
- **Read `output_type` before comparing metrics.** DAG and CPDAG scores use different units. A DAG
  search returns one member of the equivalence class and is charged for unidentifiable orientations.
  A `dag` row's `mec_shd` and `mec_arrowhead_f1` score it through its essential graph, in the units
  of a `cpdag` row's `shd` and `arrowhead_f1`; a `cpdag` row has no `mec_*` columns. `bnlearn.hc`,
  `bnlearn.tabu`, and the DirectLiNGAM and ICA-LiNGAM solutions return DAGs; the other
  structure-learning solutions return a CPDAG or PAG.
- **Score a PAG against the truth's PAG.** `scoring.score_result` projects the DAG truth with
  `andrey.data.latent.marginal` before scoring a `pag` solution, so correct FCI circles incur no
  penalty. Scoring against the DAG penalizes even the true PAG. `shd` counts one per pair with
  differing marks; `shd_endpoint` counts each differing endpoint.
- **Compare graphs with `scoring.structure_hash`.** Equal metrics can come from different graphs.
  Between families, hash the canonicalized graph instead: a DAG's hash never equals its CPDAG's.
- **Save the estimated graph with the metrics.**
- **Report the size of a difference**: how large, against what spread, over how many measurements.

### Reading the table

- **Pair a ratio within one run** with `pairing.paired_ratios`.
- **Group on `dataset_id`.**
- **Name the method a speedup is against.**
- **Read `status` before `wall_s`.** `timeout` means the fit ran and hit the cap; `blocked` means
  the step was dropped before it ran.
- **Do not infer a ceiling from which method stopped first.** Each method has its own per-fit clock.
- **Do not call a timeout a cap artifact from an extrapolation.** These curves steepen, so a fitted
  exponent gives a lower bound.
- **Read `1/speedup` as a serial fraction only when the gate sits upstream of the pinned
  config.**

### Guarding yourself

- **Assert the grouping key in a test.**
- **Check the harness, the estimator and the grouping key.**
- **Support a retraction with evidence.**

## Layout

- `benchmark.py` - defines solutions, slice, and ladder, then writes one `runs.parquet`. Copy it.
- The drivers, one per comparison, and the `start_*.sh` scripts that build their environments:
  [`CATALOG.md`](CATALOG.md).
- `andrey_bench/aggregate.py` - pools a results root and joins each run with its provenance.
- `andrey_bench/pairing.py` - defines valid comparisons, including run pairing, dataset grouping,
  duration skew, and timeout meaning.
- `andrey_bench/contracts.py` - adapter protocol, record schema, dataset and run identity.
- `andrey_bench/adapters/` - one adapter per package, method, backend, and mode.
- `andrey_bench/rsession.py` - embedded R interpreter, data transfer, and result retrieval within
  the R adapters' timed fits.
- `andrey_bench/runner.py` - one subprocess per fit, fit-only timing, caps. Each fit runs under
  a supervisor that kills its process group when the driver dies.
- `andrey_bench/rundir.py` - manages record parts, resume, and assembly in the output directory.
- `andrey_bench/memcap.py` - caps process-tree memory or uses `srun --mem=`.
- `andrey_bench/datasets.py` - dataset store and regime vocabulary; every solution reads identical
  bytes.
- `andrey_bench/regime_models.py` - the data of the domain, group, and latent-indicator regimes.
- `andrey_bench/integration.py` - runs dataset key -> data -> fit -> structure -> score -> record.
- `andrey_bench/scoring.py` - computes metrics and the cross-package structure hash.
- `andrey_bench/oracle.py` - a solution returning the true graph; a correct pipeline scores SHD 0.
- `andrey_bench/provenance.py` - reads versions, BLAS, CPU, and GPU from the benchmark environment.
- `tests/test_validation.py`, `tests/test_memory_cap.py` - the validation suite.
- `tests/test_contract.py` - dataset digest, record schema, environment fields, one unstubbed run.
