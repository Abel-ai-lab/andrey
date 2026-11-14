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
| `gnies_order.py` | Count how often a column or row order changes GnIES's result, and why. |
| `ci_calibration.py` | Measure how often each nonlinear CI test rejects, with and without a dependence. |
| `gpu_calibration_validate.py` | Check accelerator crossovers, calibration cache reuse, and dispatch. |

## Identity

A speed change must return the same answers. `identity.py` fits every supported method on the first
two draws of each of its quality-gate datasets and of one larger dataset, on the numpy backend with
one worker and one BLAS thread, and hashes each fit's full output: the graph, the edge weights, the
causal order, and every metadata value, such as a score or a p-value. Floats enter as their bytes,
so equal hashes mean equal bits; a second hash covers only the graphs' endpoint marks. The manifest
has one row per fit, keyed by method, dataset, seed, permutation, and worker count, under a header
naming the commit, the Python, NumPy, and SciPy versions, the loaded BLAS, the CPU, and the thread
settings. It also names the directory of the Andrey package the run imported and, when that
directory is in a git work tree, that tree's commit, so a manifest made with Andrey imported from
another tree through `PYTHONPATH` names the tree it ran.

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
  columns. The metadata lists of node indices, such as RCD's confounded pairs, are left out. ICP,
  IGSP, FCI-JCI, J-PCMCI+, CMB, and MMB-by-MMB are not refit: their data hold each row's
  environment in one column, the target of ICP, CMB, and MMB-by-MMB is the last variable, and
  J-PCMCI+'s contexts follow the system variables.
- `--workers N` refits PC, FCI, FCI-JCI, GFCI, CD-NOD, RSMAX2 (with PC's skeleton), GES, and PNL
  on `N` workers, with every GES pass, every skeleton pass of conditioning size two or more, and
  every possible-d-sep search sent to the worker pool.

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

## GnIES order

`gnies_order.py` measures how often reordering the columns or shuffling the rows changes GnIES's
graph or learned targets. It draws the GnIES lane's A design (`benchmarks/gnies_cpu.py`) at 6, 8,
10, and 12 variables, 100 and 1000 rows per environment, seeds 0-9 (80 models), fits each at the
defaults, refits it on three seeded column orders and three seeded row shuffles that keep each
row's environment, and maps the answers back. A tie the tie ranks decide is logged at DEBUG level
(`andrey.search.gies` for a step of the inner search, `andrey.search.gnies` for an outer step); the
summary counts the changes, those with such a record in one of the two fits, and those without. A
change without one points at an order dependence the tie ranks do not cover. With `--reference`,
gnies 0.3.3 is refitted on the same column orders, which shows whether its result depends on the
column order.

```shell
uv run python -m qa.gnies_order -o order.json
ANDREY_BENCH_GNIES_PYTHON=... uv run python -m qa.gnies_order --reference -o order.json
uv run python -m qa.gnies_order --summarize order-*.json
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
gate.

A push to the release pull request that changes only `docs/`, `site/`, and top-level Markdown
files (the CHANGELOG, the README) does not run the gate again. Nothing the gate reads is in those
paths: the gate reads `src/`, `benchmarks/` (the harness and the published summaries that
reference parity checks), `qa/`, `pyproject.toml`, and `uv.lock`. The first step of the job looks
for a passing run, on an earlier commit of the branch, that differs from this commit only in those
paths (`scripts/gate_reuse.py --ancestor`), and the job's other steps are skipped when it finds
one. The job still succeeds under its name, so the ruleset is satisfied, and its log and summary
name the run and its commit. A run reused this way was itself a pass on its own tree, so a chain of
such pushes traces back to a run that fitted the methods. A path outside those paths, a lookup that
fails, or no such run, runs the gate.

The gate checks PC, FCI, GFCI, CD-NOD, FCI-JCI, IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC, HITON-PC,
PCMB, CMB, GES, GIES, GnIES, IGSP, ICP, XGES, AGES, HC, Tabu, MMHC, RSMAX2, exact search, CALM,
NOTEARS, BOSS, GRaSP, BOSS-FCI, GRaSP-FCI, MMB-by-MMB, UT-IGSP, DirectLiNGAM, ICA-LiNGAM,
MultiGroupDirectLiNGAM, RCD, BottomUpParceLiNGAM, SCORE, NoGAM, CAM-UV, GIN, ANM, PNL, LOCI, PCMCI,
PCMCI+, LPCMCI, J-PCMCI+, VARMA-LiNGAM, LongitudinalLiNGAM, Granger, and DYNOTEARS on the datasets
whose assumptions each method needs, PC with each supported nonlinear conditional-independence test,
KCI, RCoT, CMIknn, and GPDC, on nonlinear data, and the targets UT-IGSP learns:

| Dataset | Draws | Methods |
|---|---|---|
| `linear_gauss_er` at `d = 20, 50` | 10 seeds each | PC, IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC, HITON-PC, PCMB, GES, XGES, HC, Tabu, MMHC, RSMAX2, BOSS, GRaSP; CMB at the last column |
| `linear_gauss_er` at `d = 10, 15` | 10 seeds each | exact search; CALM at `d = 10` |
| `notears_paper_standardized` at `d = 50`, `notears_paper_raw` at `d = 10`, `n = 1000` | 10 seeds each | NOTEARS |
| `ages_paper` with strong and weak shares `(0.1, 0.3)` and `(0.3, 0.3)`, `d = 10`, `n = 10000` | 10 seeds each | AGES |
| `latent_gauss_er` at `d = 20, 50` | 10 seeds each | FCI, GFCI, BOSS-FCI, GRaSP-FCI, MMB-by-MMB |
| `anm_er`, `nonmonotone_er` at `d = 5, 10`, `n = 50d` | 10 seeds each | PC with KCI; with RCoT at `d = 10`; with CMIknn and GPDC at `d = 5` |
| `interventional_gauss_er` at `d = 20, 50` | 10 seeds each | GIES |
| `noise_interventional_gauss_er` at `d = 10, 20`, four environments of `n = 1000` | 10 seeds each | GnIES |
| `unknown_target_shift_gauss_er`, `unknown_target_variance_gauss_er` at `d = 20`, six environments of `n = 1000` | 10 seeds each | UT-IGSP, its graph and its targets |
| `icp_do` at `d = 6, 10`, `n = 200` per environment, every other variable intervened | 10 seeds each | ICP |
| `known_targets_one_setting_gauss_er`, `known_targets_two_settings_gauss_er` at `d = 20`, `n = 1000` per environment | 10 seeds each | IGSP |
| `domain_shift_er` at `d = 20, 50`, `n = 2000` | 10 seeds each | CD-NOD |
| `fci_jci_s4`, `fci_jci_l10`: 4 and 10 variables, two regimes of `n = 500` | 10 seeds each | FCI-JCI |
| `multi_group_er` at `d = 10, 20`, three groups of `n = 1000` | 10 seeds each | MultiGroupDirectLiNGAM |
| `latent_measurement` at `d = 8, 12`, `n = 1000` | 10 seeds each | GIN |
| `lingam_sf` at `d = 20, 50` | 10 seeds each | DirectLiNGAM, ICA-LiNGAM |
| `latent_lingam_er` at `d = 10, 15` | 10 seeds each | RCD, BottomUpParceLiNGAM |
| `score_paper_standardized` at `d = 10` with `d` edges and at `d = 20` with `4 d`, `n = 1000` | 10 seeds each | SCORE |
| `score_paper_laplace_standardized` at `d = 10` with `d` edges, `score_paper_gumbel_standardized` at `d = 10` with `4 d`, `n = 1000` | 10 seeds each | NoGAM |
| `camuv_paper` at `d = 14`, `latent_anm_er_tenth` at `d = 22`, `n = 1000` | 10 seeds each | CAM-UV |
| `anm_pair` at `n = 1000` | 10 seeds | ANM |
| `pnl_pair` at `n = 1000` | 10 seeds | PNL |
| `lsnm_pair` at `n = 1000` | 10 seeds | LOCI |
| `pcmci_paper` at `d = 10`, `n = 500, 1000` | 10 seeds each | PCMCI |
| `pcmciplus_paper` at `(d, n) = (10, 500), (20, 1000)` | 10 seeds each | PCMCI+ |
| `lpcmci_paper` at `(d, n) = (5, 500), (10, 1000)` | 10 seeds each | LPCMCI |
| `jpcmciplus_all_m10_t100`, `jpcmciplus_half_m10_t100`: 5 system variables, 10 datasets of 100 steps | 10 seeds each | J-PCMCI+ |
| `varma_lingam_paper` at `d = 3, 5`, `n = 300` | 10 seeds each | VARMA-LiNGAM |
| `longitudinal_lingam_paper` at `(d, n, m) = (3, 20, 200), (5, 20, 500)` | 10 seeds each | LongitudinalLiNGAM |
| `var2_lagged` at `d = 5`, `n = 1000, 2000` | 10 seeds each | Granger |
| `dynotears_paper` at `d = 10, 20`, `n = 500` | 10 seeds each | DYNOTEARS |
| ASIA, linear-Gaussian | 10 seeds | PC, IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC, HITON-PC, PCMB, GES, XGES, HC, Tabu, MMHC, RSMAX2, BOSS, GRaSP, exact search, CALM, FCI, GFCI, BOSS-FCI, GRaSP-FCI; CMB and MMB-by-MMB at the last column |
| ASIA, linear with uniform noise | 10 seeds | DirectLiNGAM, ICA-LiNGAM |
| Sachs, measured | 1 | PC, FCI, GFCI, BOSS-FCI, GRaSP-FCI, IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC, HITON-PC, PCMB, GES, XGES, HC, Tabu, MMHC, RSMAX2, BOSS, GRaSP, exact search, DirectLiNGAM, ICA-LiNGAM |

- **Regimes.** `andrey.data` generates the published benchmark regimes with the benchmarks'
  arguments, at `n = 50d`; a test keeps the two in step. At the benchmarks' `n = 10d`, ICA-LiNGAM
  scores worse than the empty graph, so the check could not tell a broken build from a correct one.
  `latent_lingam_er` keeps its benchmark's 2,000 rows, with one node in five hidden; `d` counts
  the hidden nodes. From `d = 20`, BottomUpParceLiNGAM's median SHD is no lower than the empty
  graph's, and on ASIA's 1,000 rows RCD's is higher, so RCD and BottomUpParceLiNGAM run on this
  regime only.
- **Additive-noise orders.** SCORE's and NoGAM's published regime, drawn by their benchmark driver
  (`benchmarks/stein_cpu.py`) as cells of its comparison in their claims: the papers' simulation,
  Erdos-Renyi graphs with `d` edges or `4 d`, each link one draw of a Gaussian process over the
  node's parents, noise variances uniform on [0.4, 0.8], 1,000 rows, every column z-scored. SCORE's
  claim is Gaussian noise (`score_paper_standardized`), run at `d = 10` with `d` edges and at
  `d = 20` with `4 d`. NoGAM's is additive noise of any distribution, run with Laplace noise
  (`score_paper_laplace_standardized`) at `d = 10` with `d` edges and with Gumbel noise
  (`score_paper_gumbel_standardized`) at `d = 10` with `4 d`; its fit time grows with `d`, so
  `d = 20` is left out. Both run at their defaults.
- **Hidden variables, nonlinear.** CAM-UV's published regimes, drawn by its benchmark driver
  (`benchmarks/camuv_cpu.py`) at two cells in its claim where its errors are below the empty
  graph's: `camuv_paper`, the paper's simulation, 10 observed variables, two hidden common causes,
  and two hidden intermediates; and `latent_anm_er_tenth`, `anm_er`'s additive noise with one node
  in ten hidden, 20 observed. Both have 1,000 rows, and `d` counts the hidden nodes. A seed whose
  graph cannot hide its nodes is skipped and the next one taken, as the driver does.
- **Environments with hidden common causes.** FCI-JCI's published regimes, drawn by its benchmark
  driver (`benchmarks/fci_jci_cpu.py`) as its comparison's two in-claim cells of mechanism changes,
  the simulation of Mooij, Magliacane, and Claassen (2020): `fci_jci_s4`, 4 variables with edge and
  hidden-confounder probabilities 0.5, and `fci_jci_l10`, 10 variables with 0.25, each with one
  context variable that changes one variable's mechanism and 500 rows in each of its two regimes.
  The context variable, the environment, is the last column, and FCI-JCI runs at its defaults.
- **Nonlinear regimes.** `anm_er` adds to each variable a weighted sum of `z + sin(z)` over its
  parents, and `nonmonotone_er` of `cos(z)`, where a child is uncorrelated with its parent, so
  partial correlation misses the edge; both have Gaussian noise. PC with a nonlinear test costs far
  more per test, so they run at `d = 5` and `10`, `n = 50d`, and CMIknn and GPDC, at over half a
  minute per fit from `d = 10` on one core, at `d = 5` only. RCoT warns under 500 rows, so it runs
  at `d = 10` only. A test keeps both in step with the benchmark harness.
- **Smaller graphs.** Exact search scores every parent set, so its run time at least doubles with
  each variable, and the cost of a CALM optimizer step grows with the cube of the variable count;
  they run on `linear_gauss_er` at `d = 10` and `15`, and CALM on the smaller one only.
- **Linear models in the data's units.** NOTEARS's published regime, drawn by its benchmark driver
  (`benchmarks/notears_cpu.py`) as two of its comparison's cells: the simulation of Zheng et al.
  (2018), a linear model on an Erdos-Renyi graph with `2d` edges, weights from +/-[0.5, 2], Gaussian
  noise of unit variance, and 1,000 rows. NOTEARS's least-squares fit reads the columns' units, so
  one cell is standardized and one raw. `notears_paper_standardized` z-scores each column, the
  scale of the comparison's decision cells, and runs at 50 variables, the published size where the
  check has power: over the gate's seeds NOTEARS's median SHD is 18.5 against the empty graph's 20
  at 10 variables and 40 against 40 at 20. `notears_paper_raw` keeps the columns as generated,
  where the variances grow along the causal order and the fit reads the order from them, at 10
  variables. A fit at 50 standardized variables takes about half a minute on one core.
- **Linear Gaussian models with weak edges.** AGES's published regime, drawn by its benchmark
  driver (`benchmarks/ages_cpu.py`) as two of its comparison's cells: the simulation of Eigenmann,
  Nandy, and Maathuis (2017) on 10 variables with 10,000 rows, where each pair of variables holds a
  strong edge (weight magnitude 0.8 to 1.2) or a weak one (0.1 to 0.3). One cell has 10%
  strong and 30% weak edges, and one has 30% of each. AGES directs edges beyond the
  CPDAG, so the gate scores its graph against the true DAG, where an edge left undirected counts
  one.
- **Interventions, domains, groups, and indicators.** The published regimes of GIES, GnIES, CD-NOD,
  MultiGroupDirectLiNGAM, and GIN, drawn by the benchmark harness (`benchmarks/andrey_bench` and the
  benchmark drivers of GIES and GnIES): `interventional_gauss_er` at `n = 50d`, with intervention
  targets on four variables in ten and the rows split evenly between no intervention and each
  target; `noise_interventional_gauss_er`, the comparison's cell A1000 at 10 variables and its first
  ladder rung at 20, an observational environment and three that each multiply the noise variance of
  one variable, 1,000 rows each, with each row's environment beside the samples and the targets left
  for GnIES to learn; `domain_shift_er`, 2,000 rows from four domains, in which one variable in
  three shifts its mean; `multi_group_er`, three groups of 1,000 rows over one graph with weights
  drawn for each group; and `latent_measurement`, latents that each cause every later one, with
  three indicators each, where `d` counts the latents and the indicators. CD-NOD and
  MultiGroupDirectLiNGAM run through their benchmark adapters, which build the domain index and
  split the groups. GIN's number of tests grows combinatorially with the indicator count, so it runs
  at the two smaller sizes. A test keeps the domain, group, and measurement draws in step with the
  harness.
- **Several datasets with contexts.** J-PCMCI+'s published cells, drawn by its benchmark driver
  (`benchmarks/jpcmciplus_cpu.py`) from the model of the J-PCMCI+ authors' experiment script
  (Gunther, Ninad, and Runge 2023): 5 system variables with lagged links, 2 time and 2 space
  contexts, and 10 datasets of 100 steps, with every context observed and with one of each kind
  observed. A draw where a hidden context confounds an observed context and a system variable
  (Assumption 3 broken) is skipped and the next seed taken, as the driver's claimed rows leave it
  out. J-PCMCI+ runs with both dummies at the benchmark's claimed setting, `tau_max=2` and
  `alpha=0.05`.
- **Unknown targets.** UT-IGSP's published regime, drawn by its benchmark driver
  (`benchmarks/ut_igsp_cpu.py`) as two of its comparison's cells: 20 variables at mean degree 1.5,
  an observational environment and five settings of 1,000 rows, each setting with one target it is
  given and more it is not, a mean shift with one more per setting (`unknown_target_shift_gauss_er`)
  and a doubled noise standard deviation with three (`unknown_target_variance_gauss_er`). Each row's
  environment and the targets its setting is given sit beside the samples.
- **Known targets.** IGSP's published regime, drawn by its benchmark driver
  (`benchmarks/igsp_cpu.py`) as two of its comparison's cells, Wang et al. (2017), Figure 2, at 20
  variables: an observational environment and one setting of 10 targets
  (`known_targets_one_setting_gauss_er`) or two disjoint settings of 8
  (`known_targets_two_settings_gauss_er`), 1,000 rows each, every target cut from its causes and
  given. Each row's environment and its setting's targets sit beside the samples. IGSP runs at its
  defaults, where alpha, one over the observational rows, is `1e-3`, the level of the comparison's
  rows in these cells.
- **Pairs.** `anm_pair`, `pnl_pair`, and `lsnm_pair` are the published cause-effect pairs: two
  variables and one edge, in a direction the seed draws, at the benchmarks' 1,000-row size; a test
  keeps them in step with the benchmark drivers. The empty graph's SHD on a pair is 1, so ANM, PNL,
  and LOCI pass only when they name the true direction on most draws.
- **Time series.** The published time-series regimes, drawn by the benchmark harness
  (`benchmarks/andrey_bench`), which the gate imports; each method runs through its benchmark
  adapter with the published settings. The sizes are published sizes at which the method's median
  error is below the empty graph's: PCMCI's only at `d = 10`, and VARMA-LiNGAM's the two smallest,
  as its fit takes seconds. `longitudinal_lingam_paper` draws `m` units at `n` time points.
  Granger's published median error is above the empty graph's at every published size, so its
  gate draws 5 variables, where it is below. PCMCI and Granger run at their default
  `fdr_method`, `"fdr_bh"`. DYNOTEARS runs at its defaults, the paper's settings for 500 steps. A
  test keeps the Granger draws in step with the benchmark driver and the regime names with the
  harness.
- **Environments.** `icp_do` follows ICP's published regime of that name: an observational
  environment and environments of do-interventions `N(2, 0.2^2)`, at 6 or 10 variables and 200
  rows each. The data are `andrey.data`'s linear-Gaussian Erdos-Renyi graphs of mean degree 2,
  standardized, as in the regimes above. The published regime intervenes on 2 or 3 variables
  other than the target, where ICP's median error equals the empty set's; the gate intervenes on
  every variable but the target, which identifies the target's parents (Peters, Buhlmann and
  Meinshausen 2016, Theorem 2), so the check has power. Each seed draws its target among the
  variables with parents, and ICP runs at its defaults.
- **One variable's edges.** MMB-by-MMB answers the edges at one variable, so the gate fits it at the
  last column of each draw and scores the edges at that column alone. It runs on the draws of FCI's
  `latent_gauss_er` cells and on ASIA's. Sachs is left out: its one draw has two edges at its last
  column, jnk, and the fit finds one of them with a wrong mark, as FCI does, so it makes as many
  errors as the empty answer. The empty answer's median errors are the last column's median
  number of edges in the PAG, 1.5 on the `latent_gauss_er` cells and 2 on ASIA. Identity fits the
  same column, so it skips the column-order refits, as it does for ICP.
- **ASIA.** The 8-arc network of Lauritzen and Spiegelhalter (1988), sampled as a linear SCM with
  1,000 rows per seed. Its arcs are bnlearn's (`andrey.data.load_dataset("asia")`); the gate lists
  them in its own order, which fixes each seed's weights, and a test checks the two agree.
- **Sachs.** The 853 observational rows of Sachs et al. (2005), scored against the 17-arc consensus
  network, from `andrey.data.load_dataset("sachs")`. It downloads
  [`sachs.2005.continuous.txt`](https://github.com/cmu-phil/example-causal-datasets/blob/4ba0565b8164b46063daf0f6fba6b94e5a99baf6/real/sachs/data/sachs.2005.continuous.txt)
  from `cmu-phil/example-causal-datasets` (CC0-1.0) at a pinned commit once, checks its SHA-256,
  and caches it, so the first run needs network access. CMB leaves it out: at its last column,
  jnk, it finds at most one of the two edges, as PC does, so its SHD there equals the empty
  answer's.

SHD follows the published summaries: PC, with any test, IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC,
HITON-PC, PCMB, GES, XGES, HC, Tabu, MMHC, RSMAX2, exact search, CALM, BOSS, and GRaSP against the
truth's CPDAG, the LiNGAM methods, NOTEARS, AGES, ANM, PNL, SCORE, and NoGAM against the DAG,
MultiGroupDirectLiNGAM summed over its groups, and FCI, GFCI, BOSS-FCI, and GRaSP-FCI by endpoint
SHD against the PAG of the observed variables. CMB answers the last column, and counts the missing,
extra, and mismarked edges there against the truth's CPDAG. MMB-by-MMB answers the last column
too and counts the same edges there against the PAG: the true edges the answer lacks, the
answer's edges the PAG lacks, and the edges in both with a mark that differs. The empty answer
makes as many errors as the target has edges.
GIES is scored against the interventional essential graph of the true DAG under the rows' targets,
IGSP against the one under the settings' targets, GnIES and UT-IGSP against the one under the true
targets, and CD-NOD against the CPDAG in which the domain index orients the edges it reaches.
FCI-JCI is scored by endpoint SHD over the variables and the environment against its PAG at the true
distribution, the truth its benchmark driver builds. UT-IGSP's targets are scored apart, as
`UT-IGSP targets`, by the target error: the false plus the missed targets over the settings, against
the given targets alone in place of the empty graph. GIN's errors are the latents it misses: the
true latents whose indicators it does not return as one cluster. RCD and BottomUpParceLiNGAM are
scored by endpoint SHD against the latent projection, the edges and pairs their benchmark scores:
`i -> j` for a directed path from `i` to `j` through hidden nodes only, and `i <-> j` for a pair
that one hidden node reaches through hidden nodes only. The time-series methods are scored in their
summaries' units: PCMCI by the lag-resolved SHD, PCMCI+ by endpoint SHD over the window of lags,
J-PCMCI+ by endpoint SHD over the window of lags of the system variables, LPCMCI by endpoint SHD
against the true PAG of the observed series, VARMA-LiNGAM and LongitudinalLiNGAM by the missing and
extra cross-links, Granger by the missing and extra lagged edges, and DYNOTEARS by the edge SHD over
the window of lags. CAM-UV is scored by its directed edges against the edges between observed
variables that no hidden causal or backdoor path joins, the edges it can identify as direct: an edge
in one set and not the other counts 1, a reversed one 2. A method that estimates a set of variables
is scored by the set's false members plus its missed ones, where the empty set makes as many errors
as the true set has members: ICP by the target's parents. On each dataset, a method passes when its
median SHD over the draws is:

- below the empty graph's median SHD;
- within the method's tolerance of its baseline, in either direction.

`quality_baselines.json` holds the baselines, and in its `tolerances` block every tolerance with the
reason for its size. A median better than the baseline by more than the tolerance also fails, so an
intended change is recorded by a recapture. Every fit runs on the numpy backend with one worker and
one BLAS thread, so the host's accelerators and core count do not enter the result; the tolerances
absorb rounding differences between CPUs. On every pull request,
`tests/recovery/test_quality_gate.py` feeds each method the empty graph and a column-shifted true
graph, and UT-IGSP's targets check the given targets alone and the true targets moved one variable,
and the gate rejects both.

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

A setting whose first run is worse is a candidate, not a finding: judging a hundred settings at a
one-sided 2.5% gives about two or three worse ones even for identical methods. A row may therefore
hold a `replication`, the same setting rerun once on fresh seeds with the same build, with the
`seeds` it ran and the fields of a judged row (`paired`, `andrey_shd_mean`, `comparator_shd_mean`,
`mean_D`, `bootstrap_95`, `above_zero`). A row with a replication is judged by the replication's own
interval alone, so the setting counts as worse only if the rerun is worse as well. A replication
with fewer than 10 paired datasets is reported, not gated. A worse setting that replicates, or
that has no replication, fails unless the allowlist names its cause.

A method is judged in the setting it claims (`DEVELOPMENT.md`, "Test a method"). A row measured
outside that setting, for the method's **Limitations.** paragraph, carries `"claim": "out"`, on the
row or on its entry: the gate prints it with its numbers and does not fail it, whatever its
interval. A row with `"claim": "in"` or no `claim` is in claim and is judged as above, and a row's
own `claim` overrides its entry's. Any other value of `claim` is an error.

An entry's `metric` names the error its rows pair: `"shd"`, the default; `"target_error"`, the
number of intervention targets a method learned wrongly or missed (the size of the symmetric
difference of its targets and the true ones), for a method that learns its targets; or
`"direct_errors"`, the directed edges a method with hidden variables returns, against the edges
between observed variables that no unobserved causal or backdoor path joins. Its rows hold
`andrey_<metric>_mean` and `comparator_<metric>_mean`, or `andrey_shd_mean` and
`comparator_shd_mean` holding that error's means. Such a method has one entry per metric, each
judged by the rules here, and a row's key names its `metric`; any other value of `metric` is an
error. An entry with `"role": "context"` pairs Andrey with a comparator it is reported against and never
judged by: its rows are printed with their numbers and not gated. Two more fields gate an in-claim
row whatever its interval. Its `coverage` holds `andrey_short`, the datasets the reference finished
and Andrey did not under the time cap: above 0, the row fails at any number of paired datasets,
since a timeout never becomes a smaller mean, and so does a replication's. Its `floor` record
holds two bootstrap 95% intervals: `andrey_minus_empty`, of the mean of Andrey's error minus the
empty estimate's (the empty graph's error, or the target error of the targets the method was given
alone, or of no targets), and `reference_minus_empty`, of the reference's error minus the same. The
floor applies only where the reference's interval lies wholly below 0, where the reference itself
beats the empty estimate; there the row fails when Andrey's interval's upper end is not below 0.
Elsewhere the floor is printed with the row's numbers, as `FLOOR NOT APPLIED` where the row is
gated, and is not judged. The record's `applies` and `pass`, where given, must agree with the two
intervals, `pass` empty where the floor does not apply; a floor without both intervals, or a
`floor_95` or `target_floor_95` interval alone, is an error. A floor failure on a row with a
replication that records a floor counts only if the replication's floor fails too.

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
  breaks by column order, or by its tie ranks. Its `evidence` holds the row's two error values with
  the columns in reversed order (`"column_order": "reversed"`), or, for a method whose result does
  not depend on the column order, with its tie ranks reversed (`"tie_ranks": "reversed"`), beside
  `andrey_errors` and `comparator_errors`, where the row must pass.
- `trade-off`: the row is the cost of a default chosen for what it gains on other rows; the reason
  gives both.

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
