# Changelog

All notable changes to Andrey are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, Andrey is
in alpha, and any minor release may change the API. From 0.2.0, a release opens with Highlights
and, when a change breaks the API, Breaking, two sections Keep a Changelog does not define; each
change names its pull request.

## [Unreleased]

### Breaking

- `boss`, `grasp`, `ges`, `hc`, `tabu`, and `exact_search` pick their default score from the
  column types and warn once with a `DataTypeWarning`: BDeu when every column is categorical, and
  the conditional Gaussian BIC, `"local_score_cond_gauss"`, when some are, except for `tabu`, which
  has no mixed-data score. Pass `score_func="local_score_BIC_from_cov"` (`"local_score_BIC"` for
  `ges` and `exact_search`) to keep the linear-Gaussian BIC. Data with no categorical column keep it
  without a change. ([#237])

### Added

- `andrey.data.load_dataset("diabetes")`, the bnlearn diabetes network: 413 variables and 602
  arcs. ([#225])
- `boss` and `grasp` take `score_func="local_score_BDeu"` for categorical data, with the BDeu
  priors of `ges`, `hc`, `tabu`, and `exact_search`. ([#211])
- `boss` and `grasp` take `score_func="local_score_BIC_categorical"`, the multinomial BIC that
  `ges`, `hc`, `tabu`, and `exact_search` take for categorical data. ([#220])
- `pc`, `fci`, `gfci`, and `cdnod` take four experimental conditional-independence tests that
  detect nonlinear dependence: `indep_test="kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"`. ([#209])
- The classes behind them, `KCI`, `RCoT`, `CMIknn`, and `GPDC`, take their settings as arguments
  and have reference pages; `make_indep_test` passes settings to them, a variant registered with
  `register_indep_test` is selected by name, `CMIknn(k_cmi=...)` takes a neighbor count or a
  fraction of the rows, and `CMIknn.statistic` returns its conditional mutual information
  estimate. ([#209])
- `SampleSizeWarning`: RCoT warns when it conditions on fewer than 500 rows, the smallest sample
  size its reference evaluates. ([#209])
- `KCI(rank=...)`, a low-rank KCI for many rows: each kernel becomes an incomplete Cholesky
  factor of at most `rank` columns. ([#209])
- `andrey.data.sample_scm(functional="nonmonotone")`, additive noise through `cos`, whose effects
  partial correlation does not see. ([#209])
- KCI and CMIknn run on a CUDA GPU, KCI in float64 and CMIknn with the CPU's p-values, from the
  cutoffs `ANDREY_KCI_GPU_THRESHOLD` and `ANDREY_CMIKNN_GPU_THRESHOLD`. ([#209])
- `knowledge=` on `pc`, `cdnod`, `fci`, `gfci`, `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`,
  `grasp`, `exact_search`, and `direct_lingam` takes `andrey.Knowledge(required=, forbidden=,
  allowed=, tiers=)`: the edges the graph must have and must not have, the only pairs that may be
  adjacent, and tiers in time order; knowledge that contradicts itself or names a column the data
  lack raises a `ValueError`, and the CLI takes `knowledge=` as JSON or a path.
  `Knowledge.from_skeleton` turns a result's adjacencies into the allowed set. ([#214])
- `pc`, `fci`, `gfci`, and `cdnod` take `max_depth`, the largest conditioning set any test uses.
  ([#214])
- `andrey.config.tie_break`, `andrey.config(tie_break=...)`, and `ANDREY_TIE_BREAK` choose how
  the searches break ties: `"rank"`, the default, by a rank computed from each column's own data,
  or `"position"`, by column position. The new Column order and machines guide lists what each
  method keeps under a new column order and on another machine, and which methods follow the
  setting. Under `"position"`, `pc`, `cdnod`, `fci`, and `gfci` also fire their orientation rules
  one at a time in column order instead of in rounds, and `fci` and `gfci` default to
  `possible_dsep="fast"`. ([#210])

### Changed

- With `indep_test="fisherz"` or `"student_t"`, `pc`, `fci`, and `cdnod` reject copied columns
  before any test runs, even when no test meets the copies, and `pcmci`, `pcmciplus`, and `lpcmci`
  reject a series that repeats another at the same step or some steps back, naming its lagged
  value, such as `'a' at lag 1 repeats 'd'`. ([#206])
- A constant column raises one `ValueError` that names every constant column and says to drop
  them, `direct_lingam`, `ica_lingam`, and `multi_group_direct_lingam` say up front that they need
  at least `n_variables + 1` rows for their pruning step, and `andrey.core.orient.pdag2dag` says
  why a PDAG has no consistent extension, naming a chordless cycle of its undirected edges when it
  finds one. ([#207])
- `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="chisq"`, `"gsq"`, or `"cond_gauss"` return the
  same graphs and p-values in less time, because each test answers many queries at once. ([#208])

### Fixed

- `direct_lingam`, `multi_group_direct_lingam`, `varma_lingam`, and `longitudinal_lingam` treat
  causal-order candidates whose scores differ only by rounding as tied and pick among them by a
  rank computed from each column's own data, so two interchangeable columns, such as two balanced
  binary columns, no longer get a different order on another CPU or after reordering the columns.
  ([#205])
- `direct_lingam`, `multi_group_direct_lingam`, `varma_lingam`, and `longitudinal_lingam` score
  one column per group of copies, where a column copies another up to scale when it leaves at
  most `1e-12` of the other's variance unexplained. The other copies follow it directly in the
  causal order and each takes a single edge from it, so copies no longer come ahead of their
  causes and give the same graph on every CPU. ([#205])
- `exact_search` returns the same DAG of the equivalence class it finds on every CPU, with either
  `search_method`, and under any column order, because it orients the edges the class leaves
  reversible by a rank computed from each column's own data, so the DAG can differ from before but
  its class does not. ([#205])
- `pcmciplus` runs its tests on one BLAS thread, as `pcmci` and `lpcmci` do, so the host's BLAS
  thread count cannot change its p-values. ([#205])
- When several variables can come next in its causal order, `ica_lingam` picks one by a rank
  computed from each column's own data instead of by column position, so reordering the columns no
  longer changes that choice, and its order and graph can change. ([#205])
- `ges`, `gies`, `xges`, `calm`, `gin`, `bottom_up_parce_lingam`, `pcmci`, `pcmciplus`, and `lpcmci`
  break ties between equal score changes or test results by a rank computed from each column's own
  data instead of by column position; `calm` also assigns its random draws, and
  `bottom_up_parce_lingam` its regression predictors, in that rank order; `calm` chooses each
  Markov-blanket member by its partial correlation where p-values underflow to zero; and `pc` and
  `cdnod` orient colliders whose p-values agree to nine decimals together, so these choices no
  longer depend on the column order. ([#205])
- The nonlinear tests `"kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"` seed their random features,
  shuffles, and tie-breaks, and choose the variable KCI joins to the conditioning set and CMIknn
  shuffles, by a rank computed from each column's own data instead of by column position, so
  reordering the columns no longer changes their p-values or the graphs built from them.
  ([#213])
- The independent-change step of `cdnod` (`dependence_threshold`) visits the columns by a rank
  computed from each column's own data instead of by column position, so reordering the columns no
  longer changes the edges it orients, and on data where its tests compete its graph can change.
  ([#228])
- `"rcot"` samples the random Fourier features of `x` and `y` for each conditioning set, with
  frequencies and phases from a scrambled Halton sequence (quasi-Monte Carlo feature maps), so one
  pair's tests over different conditioning sets no longer share features and reject together, and
  PC with it keeps fewer false edges, misses slightly more true ones, and makes fewer errors in all
  on graphs of 10 to 50 variables. ([#248])
- Every method takes a DataFrame's column names as node labels in text, `str(name)`, so
  `categorical=["1"]` finds a column named `1`, which it reported missing before. Two names that
  are equal as text, such as `1` and `"1"`, raise a `ValueError` before the fit that names the
  label and its columns, also in `granger`, `rcd`, and `bottom_up_parce_lingam`, which raised after
  it. ([#261])
- `GraphStructure.from_numpy`, `from_edges`, and `from_networkx`, the `TemporalStructure`
  constructors, and `StructureOutput.load` hold labels as text and raise a `ValueError` on two
  labels that are equal as text: such a graph was already broken, as a NetworkX round trip merged
  its nodes, so the error replaces a broken result rather than a working one. `gin` checks its
  `labels` the same way and names its latents `L1`, `L2`, and so on, skipping a name an observed
  column has. ([#261])
- `andrey run` reads a quoted header name that holds a comma, and a first row that mixes numbers
  with text exits with `bad_data` naming one of each; a first row of numbers is data, so quote a
  header name that is a number. ([#261])
- `boss` and `grasp` with the linear-Gaussian BIC score a move in full wherever the fast update
  from its base parent set could be off by more than `1e-7`, a tenth of the tolerance they compare
  scores with, and no longer carry updated scores from one step to the next. On dense data with
  nearly collinear parent sets, where the update could be off by more than the tolerance, rounding
  error no longer decides a move, and `boss` can return a different graph. ([#271])

## [0.7.0] - 2026-10-19

### Highlights

- Permutation invariance: PC, FCI, GFCI, CD-NOD, hill climbing, tabu search, BOSS, GRaSP, and GIN
  return the same graph, up to relabeling its nodes, however the columns are ordered, and the data
  generators and the published networks shuffle their columns, so column order no longer reveals
  the causal order. The 0.7.0 post gives the measurements. ([#168], [#179], [#192], [#202],
  [#215], [#233], [#239])
- Seven more methods are supported: `gfci`, `cdnod`, `gies`, `exact_search`, `calm`,
  `multi_group_direct_lingam`, and `gin` have published benchmarks in the 0.7.0 release card and
  checks in the release quality gate, and no longer warn with `ExperimentalWarning`. ([#184])
- Faster GES: `ges` runs more than 60x faster than causal-learn at 50 to 150 variables, with the
  same graph on every dataset at those sizes. ([#169])

### Breaking

- The `numba` extra and backend are removed, because no method ran Numba code anymore:
  `andrey.config(backend="numba")` and `ANDREY_BACKEND=numba` raise a `ValueError`,
  `andrey.describe()` has no `numba` field, and `numpy` or the default `auto` runs the same code as
  before. ([#178])
- `camuv` and `andrey.irregular.camuv` are removed: on simulated nonlinear data with hidden common
  causes, its graphs had more errors than the empty graph at every size measured. ([#187])
- `cdnod` takes the domain index as `domain_index`, and `andrey run cdnod` as `--domain-index`;
  the names `c_indx` and `--c-indx` are gone. ([#187])

### Added

- `gies` learns from data that mixes observation with interventions: `interventions`
  (`--interventions` in the CLI) marks the variables set by an intervention in each row. ([#187])
- `cdnod` takes `dependence_threshold`, which runs CD-NOD's independent-change step (Huang et al.
  2020) to orient edges between variables that both change with the domain index; it is off by
  default. ([#187])
- `andrey.data.sortability.index_correlation` measures how much a dataset's column order reveals
  of its causal order: the rank correlation of each column's index with its causal depth. ([#202])
- `grasp` takes any `depth` of at least 1, the most tucks in one chain of moves, where it took
  only 3; smaller depths search less and run faster. ([#190])

### Changed

- `pc` and `cdnod` orient conflicting colliders by the p-value that separated each collider's pair,
  highest first, so their colliders no longer depend on the column order and their default output
  can change, and `collider_order="column"` keeps the column-order rule. ([#168], [#187])
- `fci` and `gfci` run their possible-d-separation step in the order of a rank computed from each
  column's own data, so reordering the columns no longer changes their PAG unless two columns are
  identical and their default output can change, and the new `possible_dsep="fast"` keeps the
  column order. ([#168], [#239])
- `andrey.data.sample_scm`, `sample_discrete`, `sample_mixed`, and `SCM.sample` shuffle the column
  order with the true graph, from the seed, so the order no longer follows how the graph was drawn
  (the hubs of a scale-free graph came first), and `shuffle=False` keeps the drawn order. ([#168])
- `andrey.data.load_dataset` lists a bnlearn network's columns in an order drawn from the seed,
  with `feature_names` and the graph's labels in the same order, because the network files list
  their variables close to causal order; `shuffle=False` keeps the file's order, and Sachs keeps
  its own. ([#202])
- Hill climbing and tabu search break exact score ties by a rank computed from each column's own
  data instead of by column position, so reordering the columns no longer changes the graph they
  find, and the same data give the same graph on other machines. ([#168], [#179])
- BOSS and GRaSP search the columns sorted by the same rank, so their seeded starting and visiting
  orders and their score ties follow each column's data instead of its position: reordering the
  columns no longer changes the graph they find, and a seed's graph can differ from 0.6.0's.
  ([#192])
- `ges`, `gfci`, and `xges` return the same graphs as 0.4.0 in less time. ([#169])
- `ANDREY_GES_PARALLEL_MIN_WORK` counts the work a pass redoes and its default stays 10000, so
  measure values calibrated before this change again with `andrey calibrate`. ([#169])
- `gies` runs its own search in one process, adding, removing, and turning edges, so on
  observational data its graph can differ from `ges`'s. ([#187])
- `gin` orders a latent only when its cluster passes the root test at `alpha`, and leaves the
  remaining latents unordered, listed in `metadata["unordered"]`, when no candidate passes.
  ([#187])
- `gin` builds each column's HSIC kernel matrix once per fit instead of once per test, so a fit
  on 1,000 rows runs more than 4x faster, and holds one such matrix per variable. ([#215])
- The `calm` reference explains each step of the fit, how to run the paper's longer schedule, and
  where the fit can stop at a local optimum. ([#187])
- When columns repeat others up to scale and shift, the score-based searches raise one `ValueError`
  that names every group of copies by column name and says to keep one column per group and read
  its edges as the group's, and `pc`, `fci`, `gfci`, and `cdnod` name the groups the same way when
  copies make a test's correlation matrix singular. ([#189])
- `andrey.viz.draw` lights a highlighted endpoint instead of ringing it: the mark and its half of
  the edge take the accent color over a soft accent glow, which keeps its proportions at any size
  and carries the class `highlight`. ([#229])

### Removed

- The benchmark explorer page is retired, and its address redirects to the release cards. ([#175])

### Fixed

- `fci` and `gfci` no longer turn every edge with a tail at one end and a circle at the other
  into an undirected edge; the rule that sets such tails now needs a third node. ([#168])
- In `fci` and `gfci`, rule R5 searches every uncovered circle path up to a fixed bound, rule R8
  checks both endpoint orders, and the possible-d-separation step tests every conditioning subset,
  so they remove false edges and orient endpoints they used to miss;
  `andrey.data.latent.marginal(..., "pag")` follows the same rules. ([#168])
- `ges`, `gies`, `gfci`, `xges`, `hc`, `tabu`, `boss`, and `grasp` return the same graph however
  small the data's scale, where squared covariances used to underflow and change the result.
  ([#169])
- These searches and `exact_search` raise a `ValueError` naming the columns when a column repeats
  another up to scale and shift given more than two rows or, with more rows than columns, is a
  linear function of other columns, instead of returning a graph from scores of a zero residual
  variance, which were infinite or finite but meaningless. ([#169])
- With no more rows than columns, these searches no longer pick a parent set of `n_samples - 1` or
  more variables, which fits its child exactly. ([#169])
- `gfci` follows the published algorithm: independence tests remove edges from the GES graph
  before the orientation rules run, so its PAG drops the extra edges GES adds around a hidden
  common cause. ([#187])
- `gin` tests each candidate for the next latent against every other remaining cluster, so a
  chain of three or more latents is no longer misordered. ([#187])
- `gin` pools its independence tests without capping any one p-value, so a strongly dependent
  group with many variables outside it is no longer accepted as a cluster. ([#187])
- `gin` runs each root test once per member of the clusters it compares, and searches the columns
  sorted by the rank hill climbing breaks ties by, so the latent order no longer depends on which
  indicator comes first, and reordering the columns no longer changes the clusters or the order.
  ([#215])
- `calm` raises `ValueError` when its loss stops being finite instead of returning NaN weights, and
  leaves the global torch random number generator untouched. ([#187])
- Converting or scoring a `dag` that holds an undirected edge raises `ValueError` instead of
  dropping the edge. ([#187])
- With the degenerate Gaussian score, on data where one variable's categories determine another's,
  `boss` no longer raises `ValueError` or `StopIteration` and no search picks parents by a wrong or
  minus-infinite score: collinear parents are fit by the columns that span them, and a node its
  parents determine scores `inf`, unavailable as under the BIC. ([#203])
- A DataFrame with reordered columns no longer fails on the torch backend, where torch rejected
  its array's negative stride. ([#216])
- `pc`, `cdnod`, `fci`, and `gfci` apply Meek's rules and FCI's R1-R10 in rounds that set
  together everything the rules imply on the graph as it stands, leaving an edge or endpoint two
  rules decide differently undirected or a circle where the rule reached first in column order used
  to decide, and `gfci` keeps the union of the smallest separating sets of a pair GES leaves
  non-adjacent instead of the first one found, so reordering the columns no longer changes what
  these rules orient, and on data where the rules conflict the graphs can change. ([#233])

## [0.6.0] - 2026-10-17

### Highlights

- Categorical and mixed data: PC, GES, hill climbing, and seven more methods take tests and scores
  for categorical columns and for tables that mix them with numbers. On categorical data, GES runs
  more than 50x faster than causal-learn at every size and returns the same graph on every
  dataset, and hill climbing runs more than 20x faster than pgmpy at every size. The 0.6.0 release
  card gives the numbers and each method's limits. ([#167])

### Added

- Categorical and mixed data: `pc`, `fci`, `gfci`, and `cdnod` take the chi-square, G-square,
  degenerate Gaussian, and conditional Gaussian tests (`indep_test="chisq"`, `"gsq"`,
  `"degen_gauss"`, `"cond_gauss"`), and `ges`, `gfci`, `hc`, `tabu`, `boss`, `grasp`, and
  `exact_search` take the matching scores. ([#167])
- `andrey.data.sample_discrete` and `sample_mixed` sample a discrete or a conditional linear
  Gaussian network with its true graph. ([#167])
- The categorical scores include the multinomial BIC beside BDeu on the BIC's `-2` log scale, read
  text, string, and `category` columns as categories, and `load_dataset(..., discrete=True)` samples
  ASIA, ALARM, HEPAR2, and ANDES from their own conditional probability tables (with Rui Zhu,
  [#94]). ([#167])

### Changed

- `hc` with a mixed-data score never adds a continuous -> categorical edge, which the conditional
  linear Gaussian model cannot express, and scores its candidate parent sets in batches. ([#167])

## [0.5.0] - 2026-10-16

### Highlights

- `pcmci`, `pcmciplus`, `lpcmci`, `granger`, `varma_lingam`, and `longitudinal_lingam` are
  supported. PCMCI+ runs more than 50x faster than tigramite at every size and PCMCI more than 20x
  at 10 variables, each returning the same graph as tigramite on every dataset. The 0.5.0 release
  card gives the numbers and each method's limits. ([#133])

### Added

- `pcmci`, `pcmciplus`, and `lpcmci`, supported time-series methods: PCMCI for lagged links,
  PCMCI+ for lagged and same-step links, and LPCMCI for a time-series PAG over series with
  unobserved variables. ([#133])
- `pc`, `fci`, `gfci`, and `cdnod` take `indep_test="student_t"`, a partial-correlation test with
  a Student-t p-value. ([#133])
- `granger`, supported lasso Granger causality for a multivariate time series: it returns a
  temporal structure with lags 1 to `maxlag` and the coefficients as weights, and does not depend
  on the units of any variable. ([#133])
- `varma_lingam` takes `max_iter`, the iteration limit of its likelihood fit. ([#133])

### Changed

- `varma_lingam` and `longitudinal_lingam` are supported and no longer warn with
  `ExperimentalWarning`. ([#133])
- `varma_lingam` with `prune=True` refits the weights it keeps by least squares, so they are no
  longer shrunk toward zero. ([#133])

## [0.4.1] - 2026-10-15

### Removed

### Breaking

- `andrey --llms` is removed: `andrey list` looks up the methods, `andrey config --help` the
  settings, and `andrey --skill` holds both in the agent guide. ([#122])

### Fixed

- `direct_lingam`, `multi_group_direct_lingam`, `var_lingam`, `varma_lingam`, and
  `longitudinal_lingam` now regress by exact least squares in the causal-order search, as the
  method's paper defines it, so on nearly tied candidates the causal order and the DAG can change
  and now match the paper's estimator. ([#127])

## [0.4.0] - 2026-10-15

### Highlights

- `anm` and `pnl` are supported. On cause-effect pairs, ANM names the same direction as
  causal-learn on every pair and runs about 2x faster; PNL runs up to 1,470x faster and names the
  true direction on more pairs. The 0.4.0 release card gives the numbers and each method's limits.
  ([#95])

### Added

- `anm`: a direction test for a pair of variables under the additive noise model, returning a
  two-node DAG and both p-values, as `pnl` does. ([#95])

### Changed

- With `alpha=None`, `pnl` breaks a tie between equal p-values by the HSIC statistics, which
  `metadata` now holds, instead of drawing no edge. ([#95])

## [0.3.0] - 2026-10-13

### Highlights

- `hc`, `tabu`, and `xges` are supported. On linear-Gaussian data, XGES returns the same graphs as
  xges and runs up to 140x faster; Tabu search and hill climbing return the same equivalence
  classes as bnlearn on most datasets and run up to 561x and 906x faster. The 0.3.0 release card
  gives the numbers and each method's limits. ([#64])

### Breaking

- `hc` runs in one process: its worker processes were slower at every measured size.
  `ANDREY_HC_PARALLEL_MIN_WORK` and `andrey calibrate --method hc` are removed. ([#56])

### Added

- `tabu`: hill climbing that keeps searching past a local optimum and returns the best DAG it
  visits, as a CPDAG. ([#56])
- `xges`: extremely greedy equivalence search, returning a CPDAG and its BIC score. ([#57])

### Changed

- `hc` returns the same graphs in less time. ([#56])
- `hc` sets no move limit by default (`max_iter=None`), so large graphs are searched to a local
  optimum. ([#56])

## [0.2.1] - 2026-10-12

### Fixed

- Result summaries show a long metadata list as its length instead of printing it in full. ([#59])

## [0.2.0] - 2026-10-12

### Highlights

- `rcd` and `bottom_up_parce_lingam` are supported: on data with hidden common causes they return
  the same graphs as lingam and run 1.7-10.8x faster. The 0.2.0 release card gives the numbers and
  each method's limits. ([#31])
- Small fits run several times faster. At 10 variables, PC and FCI take about a twelfth of the
  time, hill climbing an eighth, and GES two fifths. Larger fits are no slower, and graphs are
  unchanged. ([#33])

### Added

- Three methods for data with hidden common causes: `rcd`, `bottom_up_parce_lingam`, and the
  experimental `camuv`. Each returns a PAG over the observed variables. `rcd` and `camuv` list the
  pairs a hidden variable links in `metadata["confounded_pairs"]`; `bottom_up_parce_lingam` leaves
  the variables whose order such a cause hides unordered. ([#18], [#31], [#53])
- `hc` takes `max_iter`, its move limit, and warns with the new `andrey.SearchLimitWarning` when the
  search stops there. ([#14])

### Fixed

- Every method checks its data before the fit. Data with one dimension, fewer than 2 rows, `NaN` or
  infinity, or a constant column raises a `ValueError` that names the argument, instead of an error
  from NumPy or scikit-learn or a graph from data no method can use. ([#14])
- A single column gives a one-node graph in every method, and `cdnod` with one domain gives the PC
  graph. ([#14])
- Settings outside their range raise an error naming them: `n_lags`, `order`, `max_iter`, and
  `gin`'s `alpha` and `labels`. ([#14])
- `GraphStructure.from_numpy` rejects an unknown `kind` and a self-loop not marked as an arrowhead,
  and `andrey.viz.draw` points a `TemporalStructure` to `lag(k)` or `summary_graph()`. ([#14])
- `calm` no longer suggests a CUDA GPU, which it does not use. ([#14])

## [0.1.0] - 2026-10-08

First alpha release.

### Added

- 18 causal discovery methods in 7 families, each one function call from data to graph. PC, FCI,
  GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM are supported and benchmarked; the other 11 are
  experimental.
- One result type, `StructureOutput`, from every method; `multi_group_direct_lingam` returns one
  per group. Its graph is a `GraphStructure`: a DAG, a CPDAG, or a PAG, with nodes named by the
  data's columns, `oriented_edges()`, and export to NumPy, SciPy, NetworkX, and edge lists. For
  time series it is a `TemporalStructure`, which holds one such graph per lag.
- `andrey.metrics`: score a result against a known graph, with the metrics chosen by the kind of
  graph.
- A command line for scripts and agents: `andrey list`, `andrey run <method>`, `andrey config`,
  and `andrey calibrate`, with JSON output, machine-readable help, and an agent skill
  (`andrey --skill`).
- Optional backends: Numba JIT kernels, and PyTorch on a CUDA GPU; worker processes for GES, GIES,
  GFCI, and hill climbing; every setting is an `ANDREY_*` environment variable.
- `andrey.data`: the Sachs protein-signaling data and four bnlearn networks, downloaded on first
  use and checked by SHA-256.
- `andrey.viz`: SVG drawings of any graph, laid out by Graphviz when it is installed.
- Python 3.11 to 3.14. Linux is supported; macOS and Windows run it without a guarantee yet.

[Unreleased]: https://github.com/Abel-ai-lab/andrey/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Abel-ai-lab/andrey/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Abel-ai-lab/andrey/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Abel-ai-lab/andrey/releases/tag/v0.1.0
[#14]: https://github.com/Abel-ai-lab/andrey/pull/14
[#18]: https://github.com/Abel-ai-lab/andrey/pull/18
[#31]: https://github.com/Abel-ai-lab/andrey/pull/31
[#33]: https://github.com/Abel-ai-lab/andrey/pull/33
[#53]: https://github.com/Abel-ai-lab/andrey/pull/53
[#56]: https://github.com/Abel-ai-lab/andrey/pull/56
[#57]: https://github.com/Abel-ai-lab/andrey/pull/57
[#59]: https://github.com/Abel-ai-lab/andrey/pull/59
[#64]: https://github.com/Abel-ai-lab/andrey/pull/64
[#122]: https://github.com/Abel-ai-lab/andrey/pull/122
[#127]: https://github.com/Abel-ai-lab/andrey/pull/127
[#133]: https://github.com/Abel-ai-lab/andrey/pull/133
[#95]: https://github.com/Abel-ai-lab/andrey/pull/95
[#94]: https://github.com/Abel-ai-lab/andrey/pull/94
[#167]: https://github.com/Abel-ai-lab/andrey/pull/167
[#168]: https://github.com/Abel-ai-lab/andrey/pull/168
[#169]: https://github.com/Abel-ai-lab/andrey/pull/169
[#175]: https://github.com/Abel-ai-lab/andrey/pull/175
[#178]: https://github.com/Abel-ai-lab/andrey/pull/178
[#179]: https://github.com/Abel-ai-lab/andrey/pull/179
[#184]: https://github.com/Abel-ai-lab/andrey/pull/184
[#187]: https://github.com/Abel-ai-lab/andrey/pull/187
[#189]: https://github.com/Abel-ai-lab/andrey/pull/189
[#190]: https://github.com/Abel-ai-lab/andrey/pull/190
[#192]: https://github.com/Abel-ai-lab/andrey/pull/192
[#202]: https://github.com/Abel-ai-lab/andrey/pull/202
[#203]: https://github.com/Abel-ai-lab/andrey/pull/203
[#205]: https://github.com/Abel-ai-lab/andrey/pull/205
[#206]: https://github.com/Abel-ai-lab/andrey/pull/206
[#207]: https://github.com/Abel-ai-lab/andrey/pull/207
[#208]: https://github.com/Abel-ai-lab/andrey/pull/208
[#209]: https://github.com/Abel-ai-lab/andrey/pull/209
[#210]: https://github.com/Abel-ai-lab/andrey/pull/210
[#211]: https://github.com/Abel-ai-lab/andrey/pull/211
[#213]: https://github.com/Abel-ai-lab/andrey/pull/213
[#214]: https://github.com/Abel-ai-lab/andrey/pull/214
[#215]: https://github.com/Abel-ai-lab/andrey/pull/215
[#216]: https://github.com/Abel-ai-lab/andrey/pull/216
[#220]: https://github.com/Abel-ai-lab/andrey/pull/220
[#225]: https://github.com/Abel-ai-lab/andrey/pull/225
[#228]: https://github.com/Abel-ai-lab/andrey/pull/228
[#229]: https://github.com/Abel-ai-lab/andrey/pull/229
[#233]: https://github.com/Abel-ai-lab/andrey/pull/233
[#237]: https://github.com/Abel-ai-lab/andrey/pull/237
[#239]: https://github.com/Abel-ai-lab/andrey/pull/239
[#248]: https://github.com/Abel-ai-lab/andrey/pull/248
[#261]: https://github.com/Abel-ai-lab/andrey/pull/261
[#271]: https://github.com/Abel-ai-lab/andrey/pull/271
