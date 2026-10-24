# QA tools

On-demand tools check Andrey against itself or known ground truth. They run by hand on any machine
with the required package extras. Correctness tests and recorded outputs live in `tests/`;
comparisons with other packages and older Andrey builds live in `benchmarks/`.

| Tool | Purpose |
|---|---|
| `quality_gate.py` | Pre-publish gate for the supported methods. |
| `reference_parity.py` | Flag a published row where Andrey makes more errors than the reference package. |
| `capture_baselines.py` | Record Andrey outputs in `tests/recovery/baselines/`. |
| `identity.py` | Hash every supported method's full output, to show a change returns the same bits. |
| `profiler.py` | Time one method's fit and show where it spends its time. |
| `column_order.py` | Count how often a column relabeling changes FCI's and GFCI's marks. |
| `ci_calibration.py` | Measure how often each nonlinear CI test rejects, with and without a dependence. |
| `gpu_calibration_validate.py` | Check accelerator crossovers, calibration cache reuse, and dispatch. |

## Identity

A speed change must return the same answers. `identity.py` fits every supported method on the
first two draws of each of its quality-gate datasets and of one larger dataset, on the numpy
backend with one worker and one BLAS thread, and hashes each fit's full output: the graph, the edge
weights, the causal order, and every metadata value, such as a score or a p-value. Floats enter as
their bytes, so equal hashes mean equal bits; a second hash covers only the graphs' endpoint marks.
The manifest has one row per fit, keyed by method, dataset, seed, permutation, and worker count,
under a header naming the commit, the Python, NumPy, and SciPy versions, the loaded BLAS, the CPU,
and the thread settings. It also names the directory of the Andrey package the run imported and,
when that directory is in a git work tree, that tree's commit, so a manifest made with Andrey
imported from another tree through `PYTHONPATH` names the tree it ran.

A fit that queries a CI test or a local score also records two counts, which are deterministic, so
a change can be held to not raising them: the CI-test queries of a pair made after a query of that
pair first returned `p > alpha`, and the distinct conditioning sets of each size, a CI test's
conditioning set or a local score's parent set.

```shell
uv run python -m qa.identity -o before.json   # on the base commit
uv run python -m qa.identity -o after.json    # on the change
uv run python -m qa.identity --compare before.json after.json
```

The compare counts the rows whose graph differs and the rows where only bits differ, and lists them,
the rows in one manifest only, and every count that changed; it exits 1 when a row differs, is in
one manifest only, or has a count that rose. Two more options refit each draw and list every refit
whose answer differs from the first fit, with the parts that differ, exiting 1 if there is one:

- `--shuffle K` refits on `K` seeded column orders and maps each answer back to the original
  columns. The metadata lists of node indices, such as RCD's confounded pairs, are left out.
- `--workers N` refits PC, FCI, GFCI, CD-NOD, GES, and Granger on `N` workers, with every GES pass
  and every skeleton pass of conditioning size two or more sent to the worker pool.

Method names select methods, as in `uv run python -m qa.identity PC GES --workers 4`, and `--seeds`
sets the draws per dataset.

## Profile

A speed change starts from where a fit spends its time. `profiler.py` fits one method on one draw
of one of its identity datasets, with the identity check's call, on the numpy backend with one
worker and one BLAS thread. After the warm-up fits, it prints the fastest of the timed plain fits
with the host: the CPU, the Python, NumPy, and SciPy versions, the loaded BLAS, and the thread
settings. One more fit runs under cProfile, whose bookkeeping adds a cost to every call and so
inflates the functions that make many small calls. The tool measures that cost per call on the
host, from a loop of calls to a function that does nothing, and takes the call count times the cost
out of each function's own time. The table lists each function as `module:qualname:line` with its
share of the corrected total, which sums to 1 over all functions, its cumulative share with the
functions it calls, and its call count, which repeats from fit to fit.

```shell
uv run python -m qa.profiler PC --list                 # the dataset keys PC runs on
uv run python -m qa.profiler PC linear_gauss_er_d20    # wall time and the top 25 functions
uv run python -m qa.profiler BOSS linear_gauss_er_d100 --seed 1 --rows 40 -o boss.json
```

`--fits` and `--warmup` set the timed and warm-up fits (3 and 1). `-o` writes every function's
row, shares and call counts only, to a JSON file. Times depend on the host and its load, so no
file records them.

## FCI column order

`column_order.py` measures how often reordering the columns changes FCI's and GFCI's endpoint
marks, for the column-order sentence in the 0.1.0 benchmark report. It fits both methods on 237
latent-confounded models (Erdos-Renyi and scale-free graphs with one hidden confounder, 6-12 nodes,
500-5000 samples, seeds 0-9), refits each on three seeded column orders, maps the answers back, and
counts the relabelings that change any endpoint mark. With the default
`possible_dsep="order_free"`, a reordering relabels the marks and changes nothing else, unless two
columns are identical; with `"fast"`, the possible-d-separation step can change them. It is a
measurement with no passing threshold.

```shell
uv run python -m qa.column_order
```

## CI test calibration

A conditional-independence test is calibrated when it rejects a true independence at the rate
`alpha`. `ci_calibration.py` measures that rate, and the rate with a dependence (the power), for
KCI, RCoT, CMIknn, and GPDC at their default settings. Each draw makes `x` and `y` different
nonlinear functions of 0 to 3 conditioning variables plus noise; a noise term shared by `x` and
`y` makes them dependent. It is a measurement with no passing threshold: a rate is a property of
the test on this data, not a pass or fail of the build.

```shell
uv run python -m qa.ci_calibration                 # every test at its default row counts
uv run python -m qa.ci_calibration kci --rows 200  # one test, one row count
```

## Quality gate

One command checks the current commit before a release, and exits 1 if any check fails:

```shell
uv run python -m qa.quality_gate
```

The `quality-gate` workflow runs the same command once per release, on the head of the release pull
request into `main`, and `main`'s ruleset requires it. The publish workflow skips its own run when a
passing run has the tree of the commit it publishes (`scripts/gate_reuse.py`; a squash merge keeps
the tree), so the published files are the ones the gate passed. A change after the gate, or a lookup
that fails, runs the gate again. Every job that publishes to PyPI must need a job that calls the
workflow; `tests/unit/test_release_gates.py` fails otherwise. Other pull requests do not run the
gate. It checks PC, FCI, GFCI, CD-NOD, GES, GIES, XGES, HC, Tabu, exact search, CALM, BOSS, GRaSP,
DirectLiNGAM, ICA-LiNGAM, MultiGroupDirectLiNGAM, RCD, BottomUpParceLiNGAM, GIN, ANM, PNL, PCMCI,
PCMCI+, LPCMCI, VARMA-LiNGAM, LongitudinalLiNGAM, and Granger on the datasets whose assumptions each
method needs, and PC with each supported nonlinear conditional-independence test, KCI, RCoT,
CMIknn, and GPDC, on nonlinear data:

| Dataset | Draws | Methods |
|---|---|---|
| `linear_gauss_er` at `d = 20, 50` | 10 seeds each | PC, GES, XGES, HC, Tabu, BOSS, GRaSP |
| `linear_gauss_er` at `d = 10, 15` | 10 seeds each | exact search; CALM at `d = 10` |
| `latent_gauss_er` at `d = 20, 50` | 10 seeds each | FCI, GFCI |
| `anm_er`, `nonmonotone_er` at `d = 5, 10`, `n = 50d` | 10 seeds each | PC with KCI; with RCoT at `d = 10`; with CMIknn and GPDC at `d = 5` |
| `interventional_gauss_er` at `d = 20, 50` | 10 seeds each | GIES |
| `domain_shift_er` at `d = 20, 50`, `n = 2000` | 10 seeds each | CD-NOD |
| `multi_group_er` at `d = 10, 20`, three groups of `n = 1000` | 10 seeds each | MultiGroupDirectLiNGAM |
| `latent_measurement` at `d = 8, 12`, `n = 1000` | 10 seeds each | GIN |
| `lingam_sf` at `d = 20, 50` | 10 seeds each | DirectLiNGAM, ICA-LiNGAM |
| `latent_lingam_er` at `d = 10, 15` | 10 seeds each | RCD, BottomUpParceLiNGAM |
| `anm_pair` at `n = 1000` | 10 seeds | ANM |
| `pnl_pair` at `n = 1000` | 10 seeds | PNL |
| `pcmci_paper` at `d = 10`, `n = 500, 1000` | 10 seeds each | PCMCI |
| `pcmciplus_paper` at `(d, n) = (10, 500), (20, 1000)` | 10 seeds each | PCMCI+ |
| `lpcmci_paper` at `(d, n) = (5, 500), (10, 1000)` | 10 seeds each | LPCMCI |
| `varma_lingam_paper` at `d = 3, 5`, `n = 300` | 10 seeds each | VARMA-LiNGAM |
| `longitudinal_lingam_paper` at `(d, n, m) = (3, 20, 200), (5, 20, 500)` | 10 seeds each | LongitudinalLiNGAM |
| `var2_lagged` at `d = 5`, `n = 1000, 2000` | 10 seeds each | Granger |
| ASIA, linear-Gaussian | 10 seeds | PC, GES, XGES, HC, Tabu, BOSS, GRaSP, exact search, CALM, FCI, GFCI |
| ASIA, linear with uniform noise | 10 seeds | DirectLiNGAM, ICA-LiNGAM |
| Sachs, measured | 1 | PC, FCI, GFCI, GES, XGES, HC, Tabu, BOSS, GRaSP, exact search, DirectLiNGAM, ICA-LiNGAM |

- **Regimes.** `andrey.data` generates the published benchmark regimes with the benchmarks'
  arguments, at `n = 50d`; a test keeps the two in step. At the benchmarks' `n = 10d`, ICA-LiNGAM
  scores worse than the empty graph, so the check could not tell a broken build from a correct one.
  `latent_lingam_er` keeps its benchmark's 2,000 rows, with one node in five hidden; `d` counts
  the hidden nodes. From `d = 20`, BottomUpParceLiNGAM's median SHD is no lower than the empty
  graph's, and on ASIA's 1,000 rows RCD's is higher, so RCD and BottomUpParceLiNGAM run on this
  regime only.
- **Nonlinear regimes.** `anm_er` adds to each variable a weighted sum of `z + sin(z)` over its
  parents, and `nonmonotone_er` of `cos(z)`, where a child is uncorrelated with its parent, so
  partial correlation misses the edge; both have Gaussian noise. PC with a nonlinear test costs far
  more per test, so they run at `d = 5` and `10`, `n = 50d`, and CMIknn and GPDC, at over half a
  minute per fit from `d = 10` on one core, at `d = 5` only. RCoT warns under 500 rows, so it runs
  at `d = 10` only. A test keeps both in step with the benchmark harness.
- **Smaller graphs.** Exact search scores every parent set, so its run time at least doubles with
  each variable, and the cost of a CALM optimizer step grows with the cube of the variable count;
  they run on `linear_gauss_er` at `d = 10` and `15`, and CALM on the smaller one only.
- **Interventions, domains, groups, and indicators.** The published regimes of GIES, CD-NOD,
  MultiGroupDirectLiNGAM, and GIN, drawn by the benchmark harness (`benchmarks/andrey_bench` and
  GIES's benchmark driver): `interventional_gauss_er` at `n = 50d`, with intervention targets on
  four variables in ten and the rows split evenly between no intervention and each target;
  `domain_shift_er`, 2,000 rows from four domains, in which one variable in three shifts its mean;
  `multi_group_er`, three groups of 1,000 rows over one graph with weights drawn for each group;
  and `latent_measurement`, latents that each cause every later one, with three indicators each,
  where `d` counts the latents and the indicators. CD-NOD and MultiGroupDirectLiNGAM run through
  their benchmark adapters, which build the domain index and split the groups. GIN's number of
  tests grows combinatorially with the indicator count, so it runs at the two smaller sizes. A test
  keeps the domain, group, and measurement draws in step with the harness.
- **Pairs.** `anm_pair` and `pnl_pair` are the published cause-effect pairs: two variables and one
  edge, in a direction the seed draws, at the benchmarks' 1,000-row size; a test keeps them in step
  with the benchmark drivers. The empty graph's SHD on a pair is 1, so ANM and PNL pass only when
  they name the true direction on most draws.
- **Time series.** The published time-series regimes, drawn by the benchmark harness
  (`benchmarks/andrey_bench`), which the gate imports; each method runs through its benchmark
  adapter with the published settings. The sizes are published sizes at which the method's median
  error is below the empty graph's: PCMCI's only at `d = 10`, and VARMA-LiNGAM's the two smallest,
  as its fit takes seconds. `longitudinal_lingam_paper` draws `m` units at `n` time points.
  Granger's published median error is above the empty graph's at every published size, so its
  gate draws 5 variables, where it is below. A test keeps the Granger draws in step with the
  benchmark driver and the regime names with the harness.
- **ASIA.** The 8-arc network of Lauritzen and Spiegelhalter (1988), sampled as a linear SCM with
  1,000 rows per seed. Its arcs are bnlearn's (`andrey.data.load_dataset("asia")`); the gate lists
  them in its own order, which fixes each seed's weights, and a test checks the two agree.
- **Sachs.** The 853 observational rows of Sachs et al. (2005), scored against the 17-arc consensus
  network, from `andrey.data.load_dataset("sachs")`. It downloads
  [`sachs.2005.continuous.txt`](https://github.com/cmu-phil/example-causal-datasets/blob/4ba0565b8164b46063daf0f6fba6b94e5a99baf6/real/sachs/data/sachs.2005.continuous.txt)
  from `cmu-phil/example-causal-datasets` (CC0-1.0) at a pinned commit once, checks its SHA-256,
  and caches it, so the first run needs network access.

SHD follows the published summaries: PC, with any test, GES, XGES, HC, Tabu, exact search, CALM,
BOSS, and GRaSP against the truth's CPDAG, the LiNGAM methods, ANM, and PNL against the DAG,
MultiGroupDirectLiNGAM summed over its groups, and FCI and GFCI by endpoint SHD against the PAG of
the observed variables.
GIES is scored against the interventional essential graph of the true DAG under the rows' targets,
and CD-NOD against the CPDAG in which the domain index orients the edges it reaches. GIN's errors
are the latents it misses: the true latents whose indicators it does not return as one cluster.
RCD and BottomUpParceLiNGAM are scored by endpoint SHD against the latent
projection, the edges and pairs their benchmark scores: `i -> j` for a directed path from `i` to `j`
through hidden nodes only, and `i <-> j` for a pair that one hidden node reaches through hidden
nodes only. The time-series methods are scored in their summaries' units: PCMCI by the lag-resolved
SHD, PCMCI+ by endpoint SHD over the window of lags, LPCMCI by endpoint SHD against the true PAG of
the observed series, VARMA-LiNGAM and LongitudinalLiNGAM by the missing and extra cross-links, and
Granger by the missing and extra lagged edges. On each dataset, a method passes when its median SHD
over the draws is:

- below the empty graph's median SHD;
- within the method's tolerance of its baseline, in either direction.

`quality_baselines.json` holds the baselines, and in its `tolerances` block every tolerance with the
reason for its size. A median better than the baseline by more than the tolerance also fails, so an
intended change is recorded by a recapture. Every fit runs on the numpy backend with one worker and
one BLAS thread, so the host's accelerators and core count do not enter the result; the tolerances
absorb rounding differences between CPUs. On every pull request,
`tests/recovery/test_quality_gate.py` feeds each method the empty graph and a column-shifted true
graph, and the gate rejects both.

```shell
uv run python -m qa.quality_gate PC FCI
uv run python -m qa.quality_gate --capture  # re-measure; review the diff
uv run python -m qa.quality_gate --jobs 4   # fit four cells at a time
```

`--jobs N` fits N cells at a time in N worker processes, the slowest cells first. Each cell is
fit as with one job, on the numpy backend with one worker and one BLAS thread, so the results,
the order they print in, and the baselines `--capture` records do not change; the workflow runs
`--jobs 4` on the four vCPUs of its runner. The default is one job.

## Reference parity

Reference parity is a soft gate. Before any fit, the quality gate checks the published benchmark
summaries (`benchmarks/published/*/summary.json`) with `reference_parity.py`, which reads only those
files, so the reference packages need not be installed; it prints the flagged rows and does not fail
on them. Run on its own, `reference_parity.py` exits 1 on a flagged row: the release checklist runs
it offline and reviews every flag before the release. Each row pairs Andrey with one reference
package at one size. A summary's `parity_rule` sets how its rows are judged.

With `"parity_rule": "paired bootstrap"` (0.8.0 and later), a row is judged by D = SHD(Andrey) -
SHD(reference) on each of its paired datasets, the datasets both finished (`paired`). The
summary's builder records the mean of D (`mean_D`) and its percentile bootstrap 95% interval
(`bootstrap_95`, 10,000 resamples, NumPy's `default_rng(0)`), the interval a CI test's promotion
to supported also uses. A row with at least 10 paired datasets fails only when that interval lies
entirely above 0, so a difference within the noise of 20 datasets passes. A row with fewer than 10
paired datasets is reported, with its counts, and not gated. A randomized test's row is judged by
its `multi_draw` fields where it has them: each dataset fit several times per package (`draws`),
and D per dataset the mean over the draws.

A summary without `parity_rule` (0.1.0 to 0.7.0) keeps the count rule, which fails at any margin:

- where the row counts the series on which each side has fewer errors, when the reference has fewer
  errors on more series than Andrey;
- otherwise, when Andrey's errors (the row's mean or median SHD, or directed errors) are higher.

A row where either side has no result is skipped. A released summary never changes, so a method
re-measured against the same reference in a later summary is judged by the later rows only. Where
a summary has one entry per regime for a method and reference, a row's key names its `regime`.

A row of a reference whose result depends on the column order (`reference_order_dependent`) may
also hold Andrey's column-order mode on the same datasets (`andrey_column_shd_by_seed`,
`comparator_shd_by_seed`). It is context: the check prints the datasets on which each side has
fewer errors in that comparison beside the row, and never judges them.

`benchmarks/published/parity-allowlist.json` exempts a failing row only when it names the cause,
with a reason and a citation:

- `correct-by-paper`: Andrey follows the method's paper and the reference does not.
- `reference-defect`: the reference has a defect.
- `tie-break`: the two packages end at different local optima after an exact score tie that Andrey
  breaks by column order. Its `evidence` holds the row's two error values with the columns in
  reversed order (`"column_order": "reversed"`, `andrey_errors`, `comparator_errors`), where the
  row must pass.

Each entry pins the row's two error values (its two series counts, where the row has them, or each
side's mean SHD over the paired datasets under the paired bootstrap), so a re-run that changes the
row voids it, and an entry that matches no failing row fails the check.

```shell
uv run python -m qa.reference_parity  # exit 1 if any row is flagged
```

## Other tools

```shell
uv run python -m qa.capture_baselines --out-dir outputs/baseline-review
uv run --no-sync python qa/gpu_calibration_validate.py
```

The GPU check needs a driver-matched torch installation; see
[PyTorch for a GPU](../docs/docs/guides/getting-started.md#pytorch-for-a-gpu). `uv run --no-sync`
keeps that installation; plain `uv run` re-syncs the locked CPU build. Calibration results depend
on the machine and worker count.
