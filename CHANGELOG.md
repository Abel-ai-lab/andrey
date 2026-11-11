# Changelog

All notable changes to Andrey are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, Andrey is
in alpha, and any minor release may change the API. From 0.2.0, a release opens with Highlights
and, when a change breaks the API, Breaking, and from 0.11.0 it closes with New contributors when
someone contributes for the first time, sections Keep a Changelog does not define; each change
names its pull request.

## [Unreleased]

### Highlights

- `score_matching` is SCORE (Rolland et al., 2022) and `nogam` is NoGAM (Montagna et al., 2023), for
  nonlinear data with additive noise. Each reads a causal order from the score of the data, the
  gradient of its log density, and prunes it to a DAG. Both are supported: against the SCORE
  authors' code and dodiscover on the simulation of their papers, neither made more errors in any
  setting, `nogam` made fewer in 16 of 24, and the median fits were 4 to 9 and 1.6 to 3 times
  faster. On sparse graphs, and on additive-noise data outside that simulation, `cam` is more
  accurate. The [0.17.0 release
  card](https://andrey.abel.ai/docs/benchmarks/0.17.0.html#score-and-nogam) gives the numbers.
  ([#872], [#874], [#879])
- `notears` is supported: against its authors' code on the paper's simulation with standardized
  columns, it made no more errors in any of the 24 settings. `dagma` stays experimental. ([#871])
- Missing data: `missing="em"` runs the alternating structural EM of Friedman (1997) and is
  supported in `ges`, `boss`, and `grasp`, as it already was in `notears`, `hc`, and `tabu`. Against
  Tetrad's EM covariance policy, none was less accurate in a setting that repeated on fresh
  datasets. A fit with `missing="em"` can return another graph than in 0.16.0 and takes about 1.5 to
  5 times as long. ([#873], [#887])
- Background knowledge: path and confounder claims are supported in 14 more methods, each in at
  least one part. On a graph with no consistent extension, where a fit dropped every path claim, a
  claim on a pair joined by an undirected edge now directs that edge. `mmhc` and `rsmax2` pass
  background knowledge to their MMPC or PC restrict phase. ([#870], [#880], [#892])
- `andrey.data` draws Student-t and Gaussian-mixture noise, functions from random Fourier features,
  three more post-nonlinear transforms, and measurement columns. ([#875])
- The Limitations of `grasp` say when one start is not enough, and those of `pc` say why
  `missing="mvpc"` loses to test-wise deletion on larger graphs. ([#891], [#893])

### Added

- `ages`, experimental, is aggregative greedy equivalence search (Eigenmann, Nandy, and Maathuis,
  2017) for linear-Gaussian data with weak edges. It computes the CPDAGs GES returns for every
  penalty from `penalty_discount` up and overlays their orientations on the CPDAG of that minimum
  penalty, so the graph keeps the GES skeleton and can direct more of its edges. It returns a
  partially directed graph tagged `cpdag` and does not depend on the columns' units or order.
  ([#894])
- `score_matching` is SCORE (Rolland et al., 2022) for nonlinear effects with additive Gaussian
  noise. It finds a causal order from the leaf up, each leaf the variable whose second derivative of
  the log density varies least over the rows, estimated by a Stein kernel estimator on the
  standardized columns, and prunes the complete DAG of the order with the smooth-term tests of
  `cam`. It returns a DAG and a causal order, takes `knowledge`, and does not depend on the columns'
  units or order. It is supported: against its authors' code on the paper's simulation at 10, 20,
  and 50 variables, z-scored and rescaled, its SHD tied in all 12 settings, its order was no less
  accurate in any setting that repeated on fresh datasets, and its median fit was 4 to 9 times
  faster. ([#872], [#879])
- `nogam` is NoGAM (Montagna et al., 2023) for nonlinear effects with additive noise of any
  distribution. It finds a causal order from the leaf up, each leaf the variable whose entry of the
  Stein score estimate is best predicted from the variable's kernel-ridge residual on the others,
  and prunes the complete DAG of the order as `score_matching` does. It returns a DAG and a causal
  order, takes `knowledge`, and does not depend on the columns' units or order. It is supported:
  against dodiscover on the paper's simulation at 10 and 20 variables under Gaussian, Laplace, and
  Gumbel noise, z-scored and rescaled, its SHD was lower in 16 of 24 settings, on the first datasets
  and again on fresh ones, and higher in none, and its median fit was 1.6 to 3 times faster.
  ([#874], [#879])
- `andrey.data` draws more kinds of data: Student-t and two-component Gaussian-mixture noise
  (`noise.student_t(df=)`, `noise.gaussian_mixture(weight=, means=, sds=)`), each of mean 0 and
  variance 1 like the other families; `functional.random_features(n_features=, length_scale=)`, an
  additive-noise model whose functions are approximate Gaussian-process draws by random Fourier
  features (Rahimi and Recht, 2007); `outer="cube"`, `"cube_root"`, and `"sigmoid"` on
  `functional.post_nonlinear`; and `measurements=measurement.noisy_children(...)` on `SCM.sample`
  and `sample_scm`, which adds for each selected variable a column that is it plus independent
  noise, a child of it in the true graph and shuffled with the other columns, with
  `measurements.parents` naming the column each one measures. Each option draws from its own stream
  and is off by default, so every seed drawn without them gives the data it gave before. ([#875])

### Changed

- `notears` is supported. Against its authors' code on the paper's simulation with standardized
  columns, Erdos-Renyi and scale-free graphs of 10 to 100 variables under Gaussian, exponential,
  and Gumbel noise, it made no more errors (SHD) in any of the 24 settings, tied in 23 and with
  fewer in 1 on the first datasets, and it made fewer errors than the empty graph in 15, where the
  authors' code did in 16. Its Limitations give the numbers and where it is weak: the scale-free
  graphs, where it was within 5% of the empty graph and below it in 3 of 12 settings; columns in
  the data's own units, where most of its accuracy can be read from the columns' variances; the
  cycle repair, which removed edges in 9 of its 20 fits on standardized columns at 20 variables and
  which both sides of the comparison share; and graphs that can differ between machines while the
  errors do not. `dagma` is measured the same way against its authors' package and stays
  experimental: it made fewer errors than the empty graph in none of the 24 settings, as its
  authors' package did, and its Limitations give the numbers. The evidence is on the
  [comparison page][nd-page]. ([#871])
- Path and confounder claims are supported in more methods: `required_paths` and `forbidden_paths`
  in `cdnod`, `xges`, `hc`, `grasp`, `exact_search`, and `multi_group_direct_lingam`,
  `forbidden_paths` in `tabu`, `mmhc`, and `rsmax2`, and all three parts in `fci_jci` and
  `grasp_fci`. With true claims none measured worse than the same fit without them, or than
  causal-learn, bnlearn, Tetrad, or lingam given the same knowledge, in a setting that repeated on
  fresh seeds; `xges`, `grasp`, `exact_search`, and `fci_jci` have no reference that takes the
  knowledge. `required_paths` stays experimental in `tabu`, `mmhc`, and `rsmax2`, where bnlearn
  given the required edges those paths imply made fewer errors, and `forbidden_paths` stays
  experimental in `iamb`, `inter_iamb`, and `mmpc`, where the graph found at 50 variables often has
  no consistent extension and most claims are dropped on it. The evidence is on the
  [comparison page][kp-paths]. ([#870], [#892])
- `missing="em"` is supported in `ges`, `boss`, and `grasp`. The benchmark lane `m2-tetrad` runs
  them at the standard BIC (`ebic_gamma=0.0`) beside Tetrad 8.0.0's FGES, BOSS, and GRaSP under its
  EM covariance policy, which estimates one covariance by EM before the search and is not structural
  EM, on 132 settings of 10 to 50 variables with values missing completely at random or at random.
  Against Tetrad's default, which counts every row as the sample size, none was less accurate in any
  setting: `ges` had fewer errors in 111 and tied in 21, `boss` in 100 and 32, and `grasp` in 101
  and 31. Against Tetrad's `MEAN_PAIRWISE` sample size, a stronger reference, `ges` had fewer errors
  in 90 and tied in 42, `boss` had fewer in 72, tied in 59, and had more in 1, and `grasp` had fewer
  in 68, tied in 61, and had more in 3; all four tied on fresh datasets. The 100-variable settings
  and the default `ebic_gamma=1.0` were not compared. In `xges`, `exact_search`, `dagma`, and `calm`
  it stays experimental, with no reference implementation to compare with. The [comparison
  page][em-tetrad] has the results. ([#873], [#887])
- `missing="em"` gives other results: it now runs the alternating structural EM of Friedman (1997)
  in every method that takes it (`ges`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp`,
  `notears`, `dagma`, and `calm`). Before each search after the first, 5 parameter-only EM steps
  refit the current graph's coefficients and noise variances; before, each round refit them once.
  With one refit per round, the first search, on the expected covariance under the model with no
  edge, returned many extra edges, the later rounds removed them slowly, and the graph was still
  changing after the 10 rounds; `boss` and `grasp` then each had more errors than Tetrad's
  `MEAN_PAIRWISE` reference in one setting on two sets of datasets, where `boss` now ties and
  `grasp` has fewer. On the three benchmark lanes with a reference, the alternating scheme had fewer
  errors than one refit per round in 64 settings and more in 16, which did not run again on fresh
  datasets, and the mean errors over the settings fell in every search. `hc` and `tabu` stay no less
  accurate than bnlearn's structural EM in every setting (fewer errors in 28 and 31 of 36, ties in
  the others), and `notears` than MissDAG (fewer in 23 of 24). The 10 rounds and the sample size of
  the BIC, every row, stay. `metadata["em"]` gains `parameter_steps` and `graph_changed`, whether
  each round changed the graph, and a fit whose last round changed it logs a warning: 2 to 6% of the
  benchmark's fits, and 20% of `grasp`'s. A fit runs 56 E-steps where it ran 11 and takes about 1.5
  to 5 times as long. The comparison page's tables of the lanes without a reference were not run
  again and say that they were measured with one refit per round. Results on complete data do not
  change. ([#887])
- The Limitations of `iamb`, `inter_iamb`, and `mmpc` say how path claims fare on a graph with no
  consistent extension, which stands for no DAG and which they found in 105 to 125 of 240 simulated
  datasets, and in 19 or 20 of 20 with 50 variables, 2,000 rows, and mean degree 4: how many claims
  direct an edge, are met by an edge the data directed, and are dropped, and why `forbidden_paths`
  is experimental there. The Limitations of
  `iamb` and `inter_iamb` also state how often the skeleton step separates a true edge as the
  blankets grow, and that required edges, which enlarge the blankets, cost about one true pair per
  graph at 50 variables and mean degree 4. ([#882], [#892])
- `required_paths` and `forbidden_paths` act on a graph with no consistent extension, which `pc`,
  `cdnod`, `iamb`, `inter_iamb`, and `mmpc` can return when their tests contradict each other on
  finite data. Such a graph stands for no DAG, and a fit dropped every path claim on it. A claim on
  a pair joined by an undirected edge now directs that edge, `a -> b` for a required path and
  `b -> a` for a forbidden one (Fang and He, 2020; Fang et al., 2025), and Meek's rules follow
  (Meek, 1995). A claim on a pair that is not adjacent, a claim against an edge the data directed,
  and claims that conflict with each other, by asking for both directions of one edge or by closing
  a directed cycle, are dropped and listed in `metadata["unmet_knowledge"]` as before, whatever the
  order of the claims or the columns. `metadata["edge_claims"]` is new and lists the claims applied
  to an edge. The result is not guaranteed to stand for a DAG, as the graph found did not, and
  applying the reading to such a graph is Andrey's own choice. A fit without path claims, and a fit
  whose graph has a consistent extension, return what they returned. On simulated linear-Gaussian
  data (10 to 50 variables, 240 datasets a method) the graph found had no consistent extension in
  49 datasets of `pc`, 71 of `cdnod`, and 105 to 125 of `iamb`, `inter_iamb`, and `mmpc`. Against
  the fits that dropped every claim there, with the same true claims, no setting of any of the five
  methods was farther from the true DAG or from the class of graphs the claims define, and 5 to 29
  of 36 were closer. Most claims are still dropped on such a graph, 76 to 81% of the required and
  96 to 97% of the forbidden paths, since most name a pair that is not adjacent. `required_paths`
  is now supported in `iamb`, `inter_iamb`, and `mmpc`: with true required paths none was farther
  from that class or from the true DAG than without them in any setting. `forbidden_paths` stays
  experimental in the three: with forbidden paths alone at 50 variables and mean degree 2, the
  CPDAG was 0.65 to 1.95 edges farther from the class in 3 or 4 of 6 settings over 40 datasets,
  though in no setting on two separate sets of 20. `pc` and `cdnod` keep their status. The
  [comparison page][kp-paths] has the results. ([#892])
- The Limitations and the `n_starts` entry of `grasp` say when one search is not enough: on
  simulated linear-Gaussian data with 50 variables, mean degree 4, and 500 rows at the standard BIC,
  one start's mean SHD over 40 datasets was 29.6 (median fit 1.3 s) against 18.8 with `n_starts=4`
  (5.1 s) and 16.3 with `n_starts=8` (10.4 s), because it stops at a much worse order on some
  datasets; at mean degree 2 one start is enough. `grasp` keeps its seeded random order of tucks,
  which the paper leaves open: trying them from the front of the order to its back gave no more
  accuracy for the time than more starts and took 9 to 20 times as long at 200 variables. ([#891])
- The Limitations of `pc` and the missing-data comparison page say why `missing="mvpc"` is behind
  test-wise deletion under quantile MNAR at 50 and 100 variables: the loss is true edges that the
  correction removes, because a corrected test reads a median of 13% and 19% of the rows the
  test-wise test reads when the parents of the missingness indicators have missing values
  themselves, and because Step 3 re-tests pairs that share only an indicator, which falsely detected
  indicator parents make candidates (44 of 79 and 13 of 57 re-tested pairs at 50 variables, all true
  edges). The regression's intercept, the direction a pair is tested in, the seed, orientation, and
  the fallback to the test-wise test are not the cause. Use `missing="testwise"` in these settings.
  ([#893])

### Fixed

- `mmhc`, and `rsmax2` with `restrict="mmpc"` or `"pc"`, pass background knowledge to the restrict
  phase, which reads it as `mmpc` and `pc` do on their own: the two ends of a required edge
  condition the tests of the other pairs, and a pair the knowledge makes non-adjacent is never a
  candidate. The restrict phase read the data alone, so a pair that only a required neighbor
  separates stayed a candidate and the score search could join it. On simulated linear-Gaussian
  graphs of mean degree 4 with 10 to 50 variables and required edges taken from the true graph,
  `mmhc` now makes 1.0 to 1.5 fewer errors against the true DAG on average (a missing, extra, or
  reversed edge counts 2, an undirected one 1), and `rsmax2(restrict="pc")` 0.9 to 1.9 fewer with
  20 and 50 variables; at mean degree 2 the change is 0.2 or less. `restrict="iamb"` and
  `"inter_iamb"` keep reading the data alone, with the pairs of the required edges added to their
  skeleton: given the knowledge they made 0.3 to 0.5 more errors on the same graphs with up to
  about four in ten of the true edges required, since larger blankets let the step that turns
  blankets into a skeleton separate more true pairs. Fits without knowledge are unchanged.
  ([#880])

## [0.16.0] - 2026-11-10

### Highlights

- Missing data: `andrey.data.missingness` simulates missing values whose missingness graph is known,
  and one argument, `missing`, names how each method handles them. `pc(missing="mvpc")` corrects PC
  where missingness depends on other variables (Tu et al., 2019), and `missing="em"` fits the BIC
  score searches, `notears`, `dagma`, and `calm` on incomplete data by structural EM.
  `missing="mvpc"` is supported, and so is `missing="em"` in `notears`, `hc`, and `tabu`: against
  the MVPC authors' code, MissDAG, and bnlearn's structural EM, none measured less accurate in any
  setting. In the other searches `missing="em"` is experimental, as no reference implementation
  exists to compare them with. ([#826], [#828], [#832], [#842], [#856])
- Background knowledge states causal paths and hidden common causes:
  `andrey.Knowledge(required_paths=, forbidden_paths=, confounded=)`. The CPDAG searches keep the
  DAGs of their class that meet the path claims, the PAG methods and `lpcmci` honor all three parts,
  and `direct_lingam` narrows its causal order. The parts are supported in `pc`, `ges`, `boss`,
  `fci`, `gfci`, `boss_fci`, `lpcmci`, and `direct_lingam`, where true claims measured no worse than
  the same fit without them and than reference packages given the same knowledge, and experimental
  in the other methods, which the benchmark does not run. A path claim says less than an edge and
  directs less: a known edge belongs in `required`. ([#829], [#835], [#854])
- `loci` decides a pair's direction under the location-scale noise model of Immer et al. (2023),
  whose noise scale changes with the cause, and is supported. Against its authors' code on 100 pairs
  in each of 26 settings, neither criterion made significantly more errors in any of the 39 judged
  comparisons, and fewer in 8 that repeated on fresh pairs, and the median fit was 40 to 912 times
  faster with the default criterion. The [0.16.0 release
  card](https://andrey.abel.ai/docs/benchmarks/0.16.0.html#loci) gives the numbers. ([#830], [#857])
- Drawings mark intervened variables as causal-discovery figures do: `andrey.viz.draw` puts a cross
  in the middle of each edge into a target, and `to_graphviz(structure, targets=)` draws the
  targets as squares. The graph types guide has a legend of every mark. ([#806])
- Every gamma-HSIC p-value stays precise far below 1e-16, so `anm` and `pnl` follow the larger
  p-value even where both used to round to 0; on 970 benchmark pairs no direction changed.
  ([#808])

### Added

- `loci` decides the direction of a variable pair under the location-scale noise model of Immer et
  al. (2023), whose noise scale changes with the cause. It fits a Gaussian whose mean and variance
  are spline functions of the proposed cause in each direction, and decides by an HSIC test of the
  scaled residual against the cause (`criterion="independence"`, the default) or by the larger
  likelihood (`criterion="likelihood"`). Like `anm` and `pnl`, it reads the rows where both columns
  are observed and takes `missing`. It joins the quality gate, and its docstring gives its measured
  accuracy, times, and memory. ([#830], [#857])
- `andrey.viz.to_graphviz(structure, targets=)` draws the intervened variables, named by label or
  by position, as `square` nodes, as `draw` does. ([#806])
- The graph types guide has a section, "How to read a causal graph", with a legend of every mark
  `draw` uses. ([#806])
- A release without its own art has a default release-card cover: the navy ground, the "Release
  card" masthead, the Andrey wordmark in place of a title, and the version, with no amber dot. Its
  tile among the earlier releases shows the version on navy, and a post without art has the same
  link preview, the wordmark and its release on navy. ([#809])
- `andrey.Knowledge` takes `required_paths` and `forbidden_paths`, the pairs `(cause, effect)` with
  and without a directed path from `cause` to `effect`, and `confounded`, the pairs with a hidden
  common cause. `pc`, `cdnod`, `ges`, `xges`, `hc`, `tabu`, `mmhc`, `rsmax2`, `boss`, `grasp`,
  `exact_search`, and `iamb`, `inter_iamb`, and `mmpc` with `structure="cpdag"` honor
  `required_paths` and `forbidden_paths`: the result is the MPDAG of the DAGs of
  the class found that meet them (Fang, Zhao, Liu, and He 2025), and
  `metadata["ancestor_clauses"]` lists the claims no MPDAG can show. Knowledge that contradicts
  itself raises `ValueError` before the search. A claim that no DAG of the class meets is dropped:
  the fit warns with the new `KnowledgeWarning`, naming the claims, and lists them in
  `metadata["unmet_knowledge"]`; when only some claims hold together, they are kept in order,
  `required_paths` and then `forbidden_paths`. `bootstrap` drops them from the averaged graph alike.
  The parts are supported in `pc`, `ges`, and `boss`, and experimental in the other CPDAG methods,
  which the benchmark does not run. A claim directs edges of the graph the search found, so where
  that graph is wrong, a true claim can direct an edge the wrong way: 1 to 11% of the edges true
  claims directed in the benchmark. A required path cannot restore an edge the search left out,
  which a required edge states. ([#829], [#850], [#854])
- `fci`, `fci_jci`, `gfci`, `boss_fci`, `grasp_fci`, and `lpcmci` honor `required_paths`,
  `forbidden_paths`, and `confounded`: on a pair the tests keep, a required path
  makes the edge `a -> b` and a forbidden path puts an arrowhead at the cause before Zhang's
  rules, no test removes a `confounded` pair, and a claim the PAG does not meet is dropped with a
  `KnowledgeWarning` and listed in `metadata["unmet_knowledge"]`; the marks and the checks assume
  no selection bias. `lpcmci` takes each claim with its lag. The three parts are supported in
  `fci`, `gfci`, `boss_fci`, and `lpcmci`, and experimental in `fci_jci` and `grasp_fci`, which
  the benchmark does not run. ([#835], [#850], [#854])
- `direct_lingam` and `multi_group_direct_lingam` honor `required_paths` and `forbidden_paths`
  as the prior knowledge of Shimizu et al. (2011, Section 3.4): the claims narrow
  the variables the causal order may place next, a forbidden path is no parent, and a claim the DAG
  does not meet is dropped with a `KnowledgeWarning` and listed in `metadata["unmet_knowledge"]`.
  The parts are supported in `direct_lingam` and experimental in `multi_group_direct_lingam`,
  which the benchmark does not run. ([#835], [#850], [#854])
- Each method's schema and `--help --json` list the knowledge parts it honors, and a method given
  another part raises `NotImplementedError`, naming the methods that honor it; so do `iamb`,
  `inter_iamb`, and `mmpc` given `required_paths` or `forbidden_paths` with a `target` or an
  undirected `structure`. ([#829])
- `andrey.data.missingness` draws missing values whose missingness graph is known: `mcar`, `mar`,
  `mnar`, and `self_masking` masks, in a logistic form (Muzellec et al., 2020) or the quantile form
  of the MVPC simulations (Tu et al., 2019), with `parents=` to place each indicator. `missing=` on
  `SCM.sample` and `sample_scm` draws the mask from the seed's own stream, so the same seed without
  it gives the complete data, and `CausalDataset.missing` holds the mask and the parents of each
  indicator; `missingness.draw` masks any complete data. ([#826])
- One package-wide argument, `missing`, on the thirteen methods that accept missing values: `pc`,
  `fci`, `cdnod`, `fci_jci`, `iamb`, `inter_iamb`, `mmpc`, `pcmci`, `pcmciplus`, `lpcmci`, `anm`,
  `pnl`, and `loci`. `missing="testwise"` is test-wise deletion, and `None`, the default, picks the
  method's default handling, `"testwise"`, so no result changes; a value a method does not list
  raises `ValueError`, and `metadata["missing"]` records the handling when the tests met missing
  values. ([#828])
- `pc(missing="mvpc")` runs Missing-Value PC (Tu et al., 2019) with the permutation-based
  correction: after the test-wise skeleton, it finds the parents of each missingness indicator and
  tests again, on virtual data that correct for them, the edges such an indicator could have kept.
  It takes `"fisherz"` or `"student_t"`, and `metadata["missingness_parents"]` and
  `metadata["corrected_edges"]` record what it found. A corrected test with too few rows where its
  variables and the parents of their missingness indicators are all observed is answered by the
  test-wise test, and `metadata["corrected_edges"]["uncorrected"]` lists its pair. It is supported:
  beside the MVPC authors' R code in the paper's setting, it was better in 28 of 32 cells, tied in
  4, and worse in none, and below the empty graph in all 32. It does not improve on test-wise
  deletion where each indicator has many weak parents, and there, at 100 variables and 10,000 rows,
  77 fits ran past 30 minutes in the search for the indicators' parents, which `max_depth` bounds.
  On a wider grid, with 50 or 100 variables and 20 or 30% of the values missing, it made 0.7 to 3.6
  more errors than test-wise deletion in 12 settings, on the first datasets and again on fresh ones.
  ([#832], [#856])
- `missing="em"` fits `ges`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, and `grasp` with
  `score_func="bic"`, and `notears`, `dagma`, and `calm`, on data with missing values by structural
  EM over the expected covariance (Friedman, 1997; Gao et al., 2022): under a linear-Gaussian model
  of the current graph, the E-step gives the expected covariance of the data, the method's own
  search runs on it over every row, and the model is fit again to the graph it returns, 10 times.
  `metadata["em"]` records the rounds and the observed-data log-likelihood after each, and a row
  that observes no value is dropped first. `missing=None` keeps refusing missing values, with a
  message that names `missing="em"` where EM would run. It is supported in `notears`, `hc`, and
  `tabu`: `notears` was better than MissDAG in 19 of 24 cells and tied in 5, and `hc` and `tabu`
  were better than bnlearn's `structural.em` in 29 and 32 of 36 cells and tied in 7 and 4. With 15
  variables and 100 rows, `hc` and `tabu` are often no better than the empty graph, below it in 11
  and 15 of 24 cells, as the plain search is on complete data of that size; with 20 variables and
  500 or 2,000 rows they are below it in 12 of 12 cells. In `ges`, `xges`, `exact_search`, `boss`,
  `grasp`, `dagma`, and `calm` it is experimental, as no reference implementation exists to compare
  them with. ([#842], [#856])
- The benchmark harness measures missing data: `missing_cpu.py` fits `pc(missing="mvpc")` beside
  the MVPC authors' R code, `notears(missing="em")` beside MissDAG, and `hc` and `tabu` with
  `missing="em"` beside bnlearn's `structural.em`, each with test-wise PC, listwise deletion, mean
  imputation, the complete data, and the empty graph, and `missing_summary.py` pairs the records
  into summary rows with a paired bootstrap. `benchmarks/comparisons/missing-data.md` presents the
  results, and `benchmarks/published/0.16.0/missing-data.json` holds the rows. ([#843], [#856])
- The benchmark harness checks the path and confounder claims of `Knowledge`: against the answer
  that listing every small graph gives, with d-separation in place of the tests, and on simulated
  data against the same fit without the claims, for `pc`, `ges`, `boss`, `fci`, `gfci`, `boss_fci`,
  `lpcmci`, and `direct_lingam` (`benchmarks/knowledge_paths.py`). A CPDAG fit is scored against the
  class the truth and its claims define, and by its distance to the true DAG. The same lanes run
  Tetrad, pcalg, tigramite, and lingam under the same knowledge, and `ges` with the plain BIC beside
  its default. ([#848], [#854])
- A Quick start notebook, `examples/quick_start.ipynb`, loads the Sachs protein measurements, fits
  PC, GES, FCI, and DirectLiNGAM, reads each result, and draws each graph. It leads the examples
  gallery and replaces `examples/ges_quickstart.ipynb`, whose page on the website redirects to it.
  ([#853])
- The roadmap lists the methods planned for its local causal discovery and very large causal
  discovery stages, each with its paper, an existing implementation and its license, and what the
  method adds; the local methods are grouped by what they learn. Stage 1 is "Major methods", very
  large causal discovery means more than 1,000 variables, and it comes before agentic causal
  discovery. ([#853])

### Changed

- `andrey.viz.draw` marks each edge with an arrowhead into an intervened variable by a cross in
  its middle, where it put a bar by the arrowhead, and marks a target that no arrowhead points
  into by its square alone, where it drew a short barred arrow from outside into it. ([#806])
- `fci`'s **Limitations** paragraph says that test errors can give results that no MAG fits,
  and the `pag` it returns is then not the PAG of any MAG: of 200 simulated datasets with 10
  variables, mean degree 3, and 2,000 rows, 36 fits with `"fisherz"` and 36 with `"normal_scores"`
  returned one. On 28 and 29 of them, the MAG that Zhang's construction builds from the `pag` is
  not ancestral, and a graph query such as `m_separated` raised `ValueError`. `bootstrap`'s Notes
  name a `pag` from which Zhang's construction builds no MAG among the fits its averaged graph
  passes over for the nearest valid resample. ([#807])
- `anm`, `pnl`, `rcd`, `bottom_up_parce_lingam`, `gin`, and `indep_test="kci"` without conditions
  compute their HSIC p-values from the gamma law's upper tail through its logarithm, as `resit` and
  `camuv` do, instead of as one minus the lower tail, which rounded every p-value below about 1e-16
  to 0. `anm` and `pnl` compare the logarithms of their two p-values, which keep their order where
  both p-values fall below about 1e-308 and read 0, and `bottom_up_parce_lingam` and `gin` pool a
  finite logarithm by Fisher's method where they pooled infinity or the smallest double. A result
  can therefore change where both p-values were below 1e-16 and the HSIC statistics decided; on
  970 pairs drawn as the ANM and PNL benchmarks draw theirs, from 200 to 16,000 rows, no direction
  changed. A statistic that rounding puts just below 0 gives a p-value of 1 in `rcd`,
  `bottom_up_parce_lingam`, and `gin`, where it gave `nan`. ([#808])
- The website's Examples page holds the notebooks and the demos that run in your browser, in
  sections: Quick start, Methods, Advanced, Interactive demos, and Misc. The sidebar groups the
  pages the same way, each group open on the Examples pages. The demos' pages moved from
  `/docs/demos/` to `/docs/examples/`, the navbar no longer has a Demos entry, and every old
  address redirects to its new one. ([#853])
- Each example notebook's title names its topic only, and "Drawing a recovered structure" is
  "Visualize a causal graph". ([#853])

### Fixed

- `jpcmciplus` raises `SingularCorrelationError` when a system variable has a copy among a
  test's conditions, such as a pulse every dataset shares that is not listed as a time
  context, with the note to list it as one; it dropped that variable's links without an
  error. ([#841])
- The J-PCMCI+ comparison's same-graph counts for 0.15.0 are recounted after a fix to the
  benchmark harness, which recorded each conflict of Andrey's arm as `o-o` where tigramite marks it
  `x-x`. On the comparison page, 7 of the 21 counts in the claim go up, and no claim changes.
  ([#821])
- `loci` keeps fitting when the Hessian of a Newton step is too ill-conditioned for a Cholesky
  factorization, as a fitted variance near zero can leave it: the step adds a growing multiple of
  the Hessian's diagonal (Nocedal and Wright 2006), where it raised `LinAlgError`. Both criteria
  raised on 11 of 2,600 benchmark pairs, all with Gumbel or Laplace noise; no other fit changes.
  ([#857])

## [0.15.0] - 2026-11-08

### Highlights

- `fci_jci` is supported and joins the quality gate. Against pcalg 2.7-12's FCI-JCI, the authors'
  implementation, on the JCI paper's simulation, it ties in all 12 comparisons and fits 4.0 to 4.6
  times faster at 4 variables and 72 to 79 times faster at 10 on one core (median); at 100
  variables, where pcalg finished no fit within 30 minutes, each `fci_jci` fit took under half a
  second. The [0.15.0 release card](https://andrey.abel.ai/docs/benchmarks/0.15.0.html) gives the
  numbers. ([#739], [#759], [#768])
- `boss_fci` is supported and joins the quality gate. Against Tetrad 7.6.10's BOSS-FCI, the
  authors' implementation, on the BOSS-FCI paper's simulation at 20 and 100 measured variables, it
  ties in 72 of 73 comparisons and is better in one, and fits 1.05 to 5.8 times faster at 20
  variables and 1.85 times faster at 100 on one core (median), as the
  [0.15.0 release card](https://andrey.abel.ai/docs/benchmarks/0.15.0.html) shows. ([#737], [#761],
  [#775])
- `grasp_fci` is supported and joins the quality gate. Against Tetrad 7.6.10's GRaSP-FCI on the
  same data, it ties in 72 of 73 comparisons and is better in one, and fits 1.16 to 5.8 times
  faster at 20 measured variables and 7.6 times faster at 100 on one core (median), as the
  [0.15.0 release card](https://andrey.abel.ai/docs/benchmarks/0.15.0.html) shows. ([#737], [#761],
  [#775])
- `jpcmciplus` is supported and joins the quality gate. Against tigramite 5.2.10.1's J-PCMCI+, the
  authors' implementation, on the model of the J-PCMCI+ paper's experiment script with 5 system
  variables and 5 to 200 datasets of 10 to 200 steps, it ties in all 28 comparisons and fits 14 to
  168 times faster on one core (median). The
  [0.15.0 release card](https://andrey.abel.ai/docs/benchmarks/0.15.0.html) gives the numbers.
  ([#796], [#803])
- `ges`, `hc`, `tabu`, `boss`, `grasp`, `gfci`, `boss_fci`, and `grasp_fci` take `env`, each row's
  environment, for data from several environments (experimental): the environment index
  joins the search as a variable that no data variable causes, which removes the edges that pooling
  adds and can orient more. They list the index's edges in `metadata["index_edges"]`; `ges`, `hc`,
  `tabu`, `boss`, and `grasp` also list the variables the environments change in
  `metadata["changing_variables"]`, as `cdnod` does, and `gfci`, `boss_fci`, and `grasp_fci` take
  `confounded_index`, as `fci_jci` does. Without `env`, every result is unchanged. ([#765], [#766])
- `andrey.data.load_dataset("sachs", conditions=True)` loads all nine of Sachs's conditions, 7,466
  cells, with `env`, each cell's condition, `conditions`, their names, and `targets`, the measured
  variables each reagent acts on (`None` for PMA and b2cAMP, whose targets are unknown), the inputs
  of every method that takes `env`; `gies` and `igsp`, which need every condition's targets, take
  the rows of the seven conditions whose targets are known. ([#719])
- `cdnod` reports what the environments change: `metadata["changing_variables"]` lists the
  variables whose causal module, the distribution given their causes, changes across them, and
  `metadata["index_edges"]` the environment index's edges into them. Their descendants change
  through them and are not listed. ([#720])

### Breaking

- The former names that 0.12.0 and 0.14.0 kept with an `andrey.DeprecatedArgumentWarning` no longer
  work, as those releases announced. A call that passes one raises `TypeError`, as for any argument
  a method does not take, and `pcmciplus(collider_rule="none")` raises `ValueError`, which lists the
  rules it takes; each message adds what the name or value is called now and the release that
  renamed it. `andrey run` and `andrey bootstrap` refuse the `key=value` names and the
  `--domain-index` and `--interventions` flags the same way, with what replaces each. `gies` raises
  `ValueError` for a 0/1 matrix as its second argument, `andrey.bootstrap` and `andrey.config` raise
  `TypeError` for `n_boot` and `num_workers`, and `andrey bootstrap` refuses `--n-boot`, each with
  the current name. `BootstrapOutput` no longer has `n_boot`, `andrey.config` no longer has a
  `num_workers` setting, and assigning a setting it does not have raises `AttributeError`; Andrey no
  longer reads `ANDREY_NUM_WORKERS`. The
  [upgrade guide](https://andrey.abel.ai/docs/guides/upgrading.html) gives the current calls.
  ([#723])

  | Method | Removed name or value | Name now |
  | --- | --- | --- |
  | `cdnod` | `domain_index`, `--domain-index` | `env`, `--env` |
  | `gies` | `interventions`, `--interventions`, or the 0/1 matrix as the second argument | `targets`, with `env` and `--env` on the command line |
  | `grasp` | `depth` | `grasp_depth` |
  | `pcmci` | `pc_alpha` | `pc1_alpha` |
  | `pcmciplus` | `collider_rule="none"` | `collider_rule="sepsets"` |
  | `granger` | `cv` | none: `alpha` sets the level of the test of each link |
  | `pc`, `fci`, `gfci`, `cdnod`, `iamb`, `inter_iamb`, `mmpc`, `mmhc`, `rsmax2`, `ges`, `gies`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp`, `pcmci`, `pcmciplus`, `lpcmci` | `max_categories` | none: declare the categorical columns with `categorical` or a column type |
  | `andrey.bootstrap`, `andrey bootstrap` | `n_boot`, `--n-boot` | `n_resamples`, `--n-resamples` |
  | `andrey.config` | `num_workers` | `n_jobs` |
  | environment | `ANDREY_NUM_WORKERS` | `ANDREY_N_JOBS` |

- `andrey.data.generate` and `andrey.data.Ensemble` are deprecated: they still work in this
  release but give an `andrey.DeprecatedArgumentWarning`, and they stop working in 0.16.0. A loop
  over `SCM.sample` draws the same datasets, as the
  [upgrade guide](https://andrey.abel.ai/docs/guides/upgrading.html) shows. ([#738])

### Added

- `andrey.fci_jci` learns a PAG from data collected in several environments while allowing hidden
  common causes, running FCI with the environment as a variable that no data variable causes (Joint
  Causal Inference, Mooij, Magliacane, and Claassen 2020): it takes `env` as `cdnod` does,
  `confounded_index=False` when the environment is set from outside, and lists the environment's
  edges in `metadata["index_edges"]`. ([#739])
- `jpcmciplus` learns the links of time series from several datasets of the same variables whose
  differences come from context variables, as J-PCMCI+ (Gunther, Ninad, and Runge 2023) does: it
  finds the links from the measured time and space contexts into the system variables, stands in for
  the hidden contexts with one-hot dummies of the dataset and the time step, and returns the
  temporal structure of `pcmciplus` over the system variables. `andrey.data.sample_context_series`
  draws the model the paper describes; here each context drives exactly one system variable, so no
  hidden context is a common cause of two. ([#796])
- `boss_fci` and `grasp_fci` learn a PAG that allows hidden common causes from the CPDAG of a BOSS
  or GRaSP search, removing with tests the adjacencies whose endpoints are independent given some
  set of other variables and orienting the rest with FCI's rules, without GFCI's
  possible-d-separation step, as Algorithm 5 of their paper does (Ramsey, Andrews, and Spirtes
  2025). ([#737], [#771])
- `gfci`, `boss_fci`, and `grasp_fci` take `edge_removal="triangles"`, which tests for removal only
  the edges of the score search's CPDAG that lie in a triangle of it, as Section 5 of the GFCI
  paper does (Ogarrio, Spirtes, and Ramsey 2016), while the default, `"all"`, tests every edge, as
  before. With `"triangles"`, `boss_fci` and `grasp_fci` follow the same rule as Tetrad 7.6.3, and
  against it on the BOSS-FCI paper's simulation, at the paper's settings and at Andrey's defaults,
  they tie in 72 and 71 of 73 settings and are better in the rest, with none worse. ([#774], [#781])
- `notears`, `dagma`, and `dynotears` take `standardize=True`, as `calm` does, which scales each
  column to unit variance before the fit, so that the fit does not read the columns' units.
  ([#738])
- `andrey.data.sample_mixed` takes `envs` and `model_seed`, as `SCM.sample` and `sample_discrete`
  do, and applies the interventions of a discrete variable to a categorical column and those of a
  continuous variable to a continuous one. ([#738])

### Changed

- `gies`, `gnies`, `icp`, `igsp`, and `ut_igsp` raise `ValueError` for an `inf` or `-inf` label in
  `env`, naming the first row that holds it, as `cdnod` does, instead of reading it as one more
  label. ([#764])
- `grasp` adds or removes a parent only when it changes the score by more than the `1e-6`
  tolerance, as `boss` does, so its graph can differ on data whose scores tie within it. ([#749])
- The `andrey` command needs Typer 0.12.4 or later. ([#755])
- `fci`, `fci_jci`, and `gfci` run their possible-d-separation step faster, with the same graphs and
  separating sets on any number of workers: a search of many subsets asks them in batched calls, the
  second endpoint's search leaves out the subsets the first endpoint's search found dependent, and
  with more than one worker a search of at least 250,000 subsets with the `fisherz`, `student_t`, or
  `normal_scores` test runs on the worker processes (`ANDREY_POSSIBLE_DSEP_PARALLEL_MIN_WORK` sets
  the cutoff). A fit with more than one worker warns once on the default when a search reaches a
  tenth of it, and `andrey calibrate --method possible_dsep` measures the value for the machine it
  runs on. On two-domain data with 80 to 100 variables, where the step takes most of the fit, the
  step ran about 1.5 times faster on one core and up to 6 and 12 times faster on 4 and 8 workers; a
  step of a few seconds gains little. ([#798])

### Fixed

- `andrey.bootstrap` no longer raises `IndexError` when a row-aligned argument names a column, such
  as `env="env"`; `andrey run` exits 2 with `invalid_params` for a parameter value the method
  rejects, such as `alpha=nan`; two first loads of one dataset at the same time both succeed;
  `grasp` on one variable reports its score; and `andrey.metrics.markov_equivalent` raises
  `ValueError` for a `digraph` instead of returning `True`. ([#711])
- `fci` and `gfci` with `indep_test="chisq"` or `"gsq"` return the same PAG when a column's
  categories are relabeled. ([#732])
- `fci` and `gfci` with `collider_rule="conservative"` or `"majority"` leave a triple ambiguous
  when no subset of either end's neighbors separates its ends, as `pc` and `cdnod` do and as
  Colombo and Maathuis (2014) define both rules, instead of making it a collider. ([#752])
- `calm`'s `metadata["h_final"]` is the acyclicity residual of the gates before the thresholds,
  the value its penalty rounds compare with `h_tol`, instead of that of the returned DAG, which was
  always `0`. ([#738])
- `andrey.viz.draw` outlines each label below a node in the palette's `paper` color, so an edge that
  passes under a label no longer crosses its letters. ([#791])
- The homepage speed panel shows FCI-JCI, IAMB, MMPC, ICP, and CAM-UV. A speed chart groups a
  method's comparisons only when they share a size and Andrey's times agree there, and draws
  Andrey at every size they measure, instead of also grouping comparisons that share no size,
  drawing Andrey from the first alone, and leaving the method off the panel. The panel's legend
  names each package once, without its version or setting, so a package keeps one marker shape per
  family; lingam's CAM-UV at its default set size stays on the 0.14.0 card. ([#797])

### New contributors

- [@HouZhu2001](https://github.com/HouZhu2001) made their first contribution in [#719].

## [0.14.0] - 2026-11-05

### Highlights

- `gnies` is supported and joins the quality gate. Against gnies 0.3.3, the authors' package, on
  simulated data in the paper's settings and the settings it claims, 10 variables, 10 to 1,000
  rows per environment, and the targets unknown, `gnies` is worse in none of the 21 settings, on
  the graph or on the targets, and makes fewer errors on the graph in 2 with 10 rows per
  environment and on the targets in 1 of them, on fresh datasets too. It fits about 34 times
  faster on one core (median). ([#665], [#694])
- `ut_igsp` is supported and joins the quality gate, which also checks the targets it learns.
  Against graphical-model-learning 0.1a8, the authors' library, in the paper's setting with every
  kind of intervention it claims, `ut_igsp` at its defaults is worse in none of the 112
  comparisons, on the graph or the targets, and better in 1 on fresh datasets; 17 more were lower
  on the first datasets and on those that chose the default, which count as ties. ([#668],
  [#694])
- `icp` is supported and joins the quality gate, scored by the errors in the target's parents.
  Against InvariantCausalPrediction 0.8 it accepts the same sets on all 5,220 paired fits with two
  environments, and every comparison ties; with three or more environments Andrey corrects the
  approximate test by the number of environments, as the paper does (an allowlisted
  correct-by-paper difference). ([#677], [#694])
- `camuv` is supported and joins the quality gate, scored by its directed edges against the edges
  it can identify as direct. Against lingam 1.13.0 and causal-learn 0.1.4.8 it ties all three
  references in each of the 9 judged settings, two of them after a rerun on fresh datasets, and
  fits 2.1 to 3.1 times faster than lingam and 7.4 to 13.5 times faster than causal-learn on one
  core (median). With one node in five hidden and 12 observed variables or fewer, no
  implementation makes fewer errors than the empty graph: at 8 observed and 2 hidden, `camuv` makes
  1.50 more errors than it on seeds 0 to 19 and 1.20 more on seeds 20 to 40, and the references
  1.85 to 1.90 and 1.20 to 1.30 more. ([#675], [#694])
- `dynotears` is supported and joins the quality gate. Against CausalNex 0.12.2, the authors'
  implementation, it ties in 32 of the 38 judged settings, makes fewer errors in 5, each replicated
  on fresh seeds, all with noise of mean 1, and more in one, exponential noise of mean 0 with 100
  variables and 50 time steps, the cost of centering the columns, the default because it fits the
  intercept the paper's loss leaves out (an allowlisted trade-off). ([#682], [#694])
- `igsp` is supported and joins the quality gate. Against graphical-model-learning 0.1a8, the
  authors' library, in the settings of its papers with every target given, `igsp` at its default
  level, one over the observational rows, is worse in none of the 114 comparisons and better in 2,
  on fresh datasets too. ([#692], [#694])

### Breaking

- These arguments have new names, so that one name, `env`, gives each row's environment in both
  methods; the former ones still work in this release, in Python and on the command line, but give
  an `andrey.DeprecatedArgumentWarning` that names the replacement, and they stop working in the
  next release. The [upgrade guide](https://andrey.abel.ai/docs/guides/upgrading.html) shows the
  calls before and after. ([#663])

  | Method | Former name | Name now |
  | --- | --- | --- |
  | `cdnod` | `domain_index`, `--domain-index` | `env`, `--env` |
  | `gies` | `interventions`, `--interventions`, or the 0/1 matrix as the second argument | `targets`, with `env` and `--env` on the command line |

### Added

- `andrey.data.SCM.do`, and `envs=` on `SCM.sample` and `andrey.data.sample_discrete`, simulate
  interventions: `andrey.data.interventions.hard` cuts a variable from its parents and sets it from
  outside, and `soft` changes its mechanism and keeps its parents; they return a `HardIntervention`
  and a `SoftIntervention`. A sample of several environments comes pooled with `env`, each row's
  environment, and `targets`, each environment's intervened columns, for the linear, nonlinear,
  location-scale, and discrete models. ([#659], [#674])
- `andrey.data` takes options for the simulations of interventional benchmarks, each off by default,
  so every sample at the defaults is unchanged: `soft(weights="redraw")` draws each parent weight
  anew from the functional's `weight_range`, `soft(noise_scale=(low, high))` multiplies the noise of
  each row by its own draw from `U[low, high]`, `model_seed=` on `SCM.sample` and `sample_discrete`
  keeps one model across datasets while `seed` draws the rows, `functional.linear(positive=True)`
  makes every weight positive, and `variance_range=` on the noise families draws each noise variance
  from a range. The provenance of a sample names `positive`, `variance_range`, and
  `heteroscedastic` when they are set. ([#690])
- `andrey.viz.draw(structure, targets=)` marks the intervened variables, named by label or by
  position: each is drawn as a square, and a bar crosses each arrowhead into it, the edges the
  intervention cuts; a target no arrowhead points into gets a short barred arrow from outside. The
  mark is drawn by shape, so it reads without color, and a latent target keeps its dashed ring.
  ([#707])
- `gies(data, env, targets=)` and `cdnod(data, env)` take each row's environment as labels, text
  or numbers, or as the name of a DataFrame column, and `gies`'s `targets` takes each
  environment's intervened variables as a mapping from the labels of `env`, as a column, or as the
  0/1 matrix, with `--env` and `targets=` in `andrey run` and `andrey bootstrap`, whose resamples
  for `gies` or text labels keep the same share of every environment's rows. ([#663])
- `cdnod`, `gies`, `gnies`, `icp`, `igsp`, and `ut_igsp` also take their data as each environment's
  rows apart, as gnies, the GnIES authors' package, and the UT-IGSP authors' library do: a list of
  DataFrames or 2-D arrays, each labeled by its position, or a dict from each label to its table,
  with `env` left out and `targets` a mapping from those labels. The rows are stacked in that order,
  so the fit is that of the stacked table with `env`, and `andrey.bootstrap` resamples within each
  environment, `cdnod`'s domains included. In Python, `cdnod`'s and `icp`'s `env` is optional for
  this; one table without it raises `TypeError` in `cdnod` and `ValueError` in `icp`, which needs
  two environments. The command line still needs `--env`. ([#676], [#702])
- `StructureOutput` metadata takes `int`, `float`, and `bool` keys in a nested mapping, such as the
  labels of `env` that key `ut_igsp`'s `metadata["targets"]`; `save` writes them as their JSON
  text, `"1"` for `1`. A nested mapping that mixes text and number keys, or a NumPy, tuple, or
  `None` key, raises `TypeError`. ([#680])
- `ut_igsp` learns from an observational environment and settings whose intervention targets are
  known in part or not at all (UT-IGSP, Squires, Wang, and Uhler 2020): it returns the
  interventional essential graph and, in `metadata["targets"]`, each setting's targets, known and
  learned, keyed by the labels of `env`, so they pass back as `targets`. It takes `env` and
  `targets` as `gies` does, with `None` in the mapping for a setting whose targets are unknown, and
  `invariance_test`, `"gaussian"` or `"indicator"`, the test of whether a variable's distribution
  given others changes. Against graphical-model-learning 0.1a8, the authors' library, on simulated
  data in the paper's setting with every kind of intervention it claims (20 variables, 5 settings,
  1,000 to 5,000 rows per setting, up to 3 unknown targets per setting), `ut_igsp` at its defaults,
  alpha one over the observational rows and 10 starts, is worse in none of the 112 comparisons, on
  the graph or the targets, and better in 1, its errors lower on the first datasets and on fresh
  ones; in 17 more they were lower on the first datasets and on those its default was chosen on,
  which are not fresh, so these count as ties. At the paper's alpha `1e-5` it is worse in none and
  better in none. A median fit on 20 variables takes 0.24 s, about 1.7 times the library's time on
  one core. ([#668], [#694])
- `igsp` learns the interventional essential graph from an observational environment and settings
  whose targets are all known (IGSP, Wang et al. 2017; Yang, Katcoff, and Uhler 2018), the same
  inputs as `gies`: it searches variable orders for the DAG with the fewest edges, then the fewest
  edges that the settings' invariance tests contradict. It takes the arguments of `ut_igsp`, with
  the same default `alpha=None`, one over the number of observational rows, and returns the level
  used in `metadata["alpha"]`. Against graphical-model-learning 0.1a8, the authors' library, on
  simulated data in the papers' settings (7 to 20 variables, perfect, inhibiting, and imperfect
  interventions, 1,000 to 100,000 rows per setting), `igsp` at its defaults, with the library at
  alpha `1e-3`, is worse in none of the 114 comparisons and better in 2, its errors lower on two
  sets of datasets; in 15 more they were lower on one set only, which count as ties. Alpha `1e-3`
  also has fewer errors in 17 on that set, as many as the default, and a rule written before the run
  gives that tie to one over the observational rows. At alpha `1e-3` with 10 starts, with the
  library at the same level, `igsp` is worse in none and better in 10, its errors lower on two sets
  of datasets, all in settings of two or more targets, where the library's test departs from the
  paper; against a copy of the library corrected to the paper it is worse in none and better in
  none. Interval ends that are floating-point residue of 0 are read as 0. A median fit takes 1.8
  times the library's time on one core at 1,000 rows per setting and 4.3 times at 10,000. ([#692])
- `andrey.gnies` learns from linear-Gaussian data pooled from two or more environments that differ
  in the noise variances of variables it is not told: it returns those variables, the targets, in
  `metadata["changing_variables"]`, with the interventional essential graph they orient (Gamella et
  al. 2022). Known targets go in `targets`. Against gnies 0.3.3, the authors' package, on simulated
  data in the paper's settings and the settings it claims, 10 variables, 10 to 1,000 rows per
  environment, and the targets unknown, `gnies` is worse in none of the 21 settings, on the graph
  or on the targets, and makes fewer errors on the graph in 2 with 10 rows per environment and on
  the targets in 1 of them, on fresh datasets too; at the standard BIC the two return the same graph
  and targets on 412 of 420 datasets. It fits about 34 times faster on one core (median) and
  reaches 100 variables, with a median SHD of 2.5 against the empty graph's 135 in a median 176 s.
  ([#665])
- `andrey.icp` finds the causal parents of one `target` from continuous data pooled from several
  environments that leave the target's mechanism alone (Peters, Buhlmann and Meinshausen 2016), by
  the paper's approximate or exact test, with each variable's p-value and confidence interval.
  Against InvariantCausalPrediction 0.8, the authors' package, on simulated data in the paper's
  settings, it accepts the same sets on all 5,220 paired fits with two environments, and it names a
  variable outside the true parents in at most 1.23% of fits at `alpha=0.05` in the settings it
  claims. ([#677])

### Changed

- `iamb`, `inter_iamb`, and `mmpc` with `indep_test="chisq"` or `"gsq"` run faster on categorical
  data, most of all `mmpc`'s default CPDAG search, with bit-identical outputs. ([#662])
- `andrey --help --json` lists a data input's former names under `aliases` again, such as
  `cdnod`'s `domain_index`. ([#663])
- `targets=` reads a text key that spells a number as the label of `env` it spells, in Python as on
  the command line, so a mapping read back from JSON passes as it is: `"1"` names the label `1`.
  Two keys that name one label, such as `1` and `"1"`, raise `ValueError` instead of the last one
  winning. ([#680])
- `andrey run` and `andrey bootstrap` read an `--env` of whole numbers as integers, so the labels in
  their output and messages read `1`, not `1.0`; the graphs do not change. ([#680])
- `gies` and `ut_igsp` prepare their rows faster, with bit-identical outputs. On one core, with 20
  variables, an observational environment, and 5 settings that each intervene on one variable,
  `gies` fits about 3 times as fast at 10,000 rows per environment and 1.8 to 2.2 times as fast at
  100,000, and `ut_igsp` about 1.2 times as fast. The ranks that break ties, which most methods
  compute, take about a quarter less time. ([#679])
- `camuv`'s Limitations report its benchmark against three references: lingam 1.13.0 and
  causal-learn 0.1.4.8, each run with sets of up to 3 variables, the paper's setting, and lingam at
  its default of 2, on simulated nonlinear additive data with and without hidden variables, 4 to 50
  observed variables and 500 to 2,000 rows, and on the paper's simulation. `camuv` ties all three
  in each of the 9 judged settings, two of them after a rerun on fresh datasets, and fits 2.1 to
  3.1 times faster than lingam and 7.4 to 13.5 times faster than causal-learn on one core
  (median). With one node in five hidden and 12 observed variables or fewer, no implementation
  makes fewer errors than the empty graph. ([#675])
- `dynotears`'s Limitations report its benchmark against CausalNex 0.12.2, the authors'
  implementation, on the structural vector autoregressions of the paper's design, with 5 to 100
  variables, 500 and 50 time steps, and Gaussian and exponential noise of mean 0 and 1: it ties in
  32 of the 38 judged settings, makes fewer errors in 5, all with noise of mean 1, whose mean its
  centering absorbs, and more in one, exponential noise of mean 0 with 100 variables and 50 time
  steps, where the centering costs about one time step of data. ([#682])
- The Limitations of `cam` and `resit` and the subsampling evidence in `bootstrap` count a setting
  whose paired interval contains 0 as a tie instead of saying the method made no more errors there.
  ([#687])
- For contributors, `qa/reference_parity.py` reads `"direct_errors"` as an entry's `metric`, the
  directed edges against the edges between observed variables that no hidden path joins, and the
  parity allowlist takes the cause `trade-off`: the row is the cost of a default chosen for what it
  gains on other rows. The check also judges a row's `floor` record, Andrey's and the reference's
  error minus the empty estimate's, on the first seeds: the floor applies only where the
  reference's interval lies wholly below 0, and there Andrey's must too; elsewhere it is printed
  and not judged. The record's `applies` and `pass` must agree with its intervals, a floor without
  the reference's interval is an error, and on a row with a replication a floor failure counts only
  if the replication's floor, where it records one, fails too. A tie-break's evidence reverses the
  column order or, for a method whose result does not depend on it, the tie ranks
  (`"tie_ranks": "reversed"`). ([#694])

### Fixed

- Copies of one column in other units, such as `x` and `2 * x + 1`, and categorical columns that
  group the rows alike under other labels break ties by column position, as identical columns do,
  so adding a constant to a copy or relabeling its categories no longer changes which column a tie
  goes to. ([#660])

## [0.13.0] - 2026-11-03

### Highlights

- `iamb` is supported and joins the quality gate. Against bnlearn 5.2.1 on simulated linear-Gaussian
  data with 20 to 200 variables, 1,000 and 5,000 rows, and mean degree 2 and 4, bnlearn makes
  significantly fewer errors in 1 of 80 judged comparisons, `iamb`'s CPDAG at 50 variables, 5,000
  rows, and mean degree 2, where bnlearn takes the first column among candidates whose p-values
  underflow to 0 and Andrey the candidate with the strongest test statistic; two more comparisons
  are worse once and tie on a rerun with fresh seeds. Two comparisons have 9 pairs because bnlearn's
  fits reached the 600 s cap. The median ratio of bnlearn's fit time to Andrey's for the CPDAG is
  4.6 on standardized data, 2.7 on unstandardized data, and 2.2 at 200 variables. ([#607])
- `mmpc` is supported and joins the quality gate, and the benchmark report adds the discrete
  networks asia, child, insurance, alarm, and hepar2 for `iamb`, `inter_iamb`, `mmpc`, `mmhc`, and
  `rsmax2`, Andrey's chi-square and G-square tests against bnlearn's `x2-adf` and `mi-adf`. On
  linear-Gaussian data `mmpc`'s skeleton at 200 variables, 5,000 rows, and mean degree 2 makes more
  errors than bnlearn's on two runs (2.7 more on average on the second), because Andrey admits the
  candidate with the strongest test statistic where bnlearn takes the first column, and its CPDAG at
  100 variables, 1,000 rows, and mean degree 2 is worse once (+0.70) and +0.56 on a rerun of 9
  pairs. On the networks, `iamb`'s moral graph of hepar2 at 5,000 rows is worse on two runs; the two
  packages run different discrete tests. Andrey is slower than bnlearn in 186 of the 220 network
  settings, in all 44 on hepar2 (bnlearn's fit time over Andrey's is 0.06 to 0.86), after the faster
  discrete tests. ([#626])
- `cam` is supported and joins the quality gate. Against CAM 1.0, the authors' R package, on
  simulated nonlinear additive-noise data with Gaussian or Gumbel noise, 10 to 100 variables, 1,000
  rows, and mean degree 2, `cam` makes no more errors in 14 of 16 judged settings and fewer in 2
  (Gumbel noise at 100 variables), returns the same graph on 278 of 320 datasets, and fits 6.5 to
  14.7 times faster on one core. CAM 1.0 caps a variable's parents at the smaller of the variables
  minus one and the rows over 20, 50 at 1,000 rows, which binds at 100 variables; Andrey sets no cap
  by default. Location-scale and linear-Gaussian data lie outside the model and are reported, not
  judged. ([#631])
- `resit` is supported and joins the quality gate. On simulated nonlinear additive-noise data with
  Gaussian or Gumbel noise, 5 to 20 variables, 1,000 rows, and mean degree 2, `resit` makes no more
  errors than the authors' R code in 12 of 12 judged settings (one after a rerun on fresh datasets)
  and fewer than lingam's implementation in 11 of 12 (no more in the other), returns the authors'
  graph on 87 of 220 datasets, and fits 3.2 to 4.6 times faster than the authors' code and 2.1 to
  2.9 times faster than lingam on one core. The accuracy claim holds up to 10 variables: at 20
  variables on Gaussian noise no implementation clearly beats the empty graph, so that range is for
  speed. The paper fixes no order for removing a variable's parents; Andrey's order is no worse than
  a random one on datasets not used before. Location-scale and linear-Gaussian data lie outside the
  model and are reported, not judged. ([#634])

### Added

- `andrey.data.sample_scm(functional="location_scale")` and
  `andrey.data.functional.location_scale(scale_range=)`, location-scale noise: each variable's
  noise scale is a function of its parents, with weights of its own in `params.scale_weights`,
  between the bounds of `scale_range`, by default those of the benchmark of Immer et al. (2023).
  ([#596])
- `andrey.data.load_dataset("child")` and `load_dataset("insurance")`, the bnlearn child network
  (20 variables, 25 arcs) and insurance network (27 variables, 52 arcs). ([#611])
- `cam`, the causal additive model of Buhlmann, Peters, and Ernest (2014), a functional causal
  model, learns a DAG and a causal order from nonlinear additive effects with the paper's
  settings: boosting preselects each variable's possible parents, a greedy search adds edges by the
  drop in log residual variance of additive spline fits, and smooth-term tests prune them.
  ([#597])
- `resit`, regression with subsequent independence test (Peters, Mooij, Janzing, and Scholkopf
  2014), a functional causal model, learns a DAG and a causal order from nonlinear effects with
  additive noise of any distribution, with the paper's settings: it takes as each next sink the
  variable whose additive-model residual an HSIC test finds most independent of the others, then
  removes each parent whose removal leaves the residual independent. ([#620])
- `camuv`, causal additive models with unobserved variables (Maeda and Shimizu 2021), a
  latent-variable method, learns the direct causes among the observed variables from nonlinear
  additive effects with the paper's settings: HSIC tests on the residuals of additive spline fits
  find each variable's direct causes over sets of up to 3 variables, and a pair that a hidden common
  cause or a causal path through a hidden variable joins, whose relation no regression can
  identify, comes back as `o-o` and in `metadata["latent_pairs"]`, and the edges it drops from
  directed cycles in `metadata["cycle_edges"]`; it is experimental. ([#623])
- `dynotears`, DYNOTEARS (Pamfil et al. 2020), a time-series method, learns a linear structural
  vector autoregression from one series, same-step and lagged effects with their weights, by least
  squares with an l1 penalty under the NOTEARS acyclicity constraint on the same-step weights, with
  the paper's settings: `lambda1=0.05`, `w_threshold=0.3`, and `lag_threshold=0.1` for 500 time
  steps, and `tau_max=1`; it is experimental. ([#639])

### Changed

- `direct_lingam`, `ica_lingam`, and `multi_group_direct_lingam` now belong to the family
  "Functional causal models", with `cam` (Glymour, Zhang, and Spirtes 2019): their `family` value
  in `andrey list` and in the method contracts of `andrey --help --json` is `"functional"` in place
  of `"lingam"`, and nothing else in the API changes. ([#597])
- The rank that breaks ties (`tie_break="rank"`, the default) no longer depends on the row order or
  on a column's units or sign: it is a hash of each column's relations to the other columns, and a
  new column order still relabels it; only columns that no row order, unit, or sign tells apart
  fall back to their values in row order. Results that a tie decides change in `hc`, `tabu`,
  `mmhc`, `rsmax2`, `boss`, `grasp`, `fci`, `gfci`, `ica_lingam`, and `varma_lingam`: the graph
  changed on 41 of the identity check's 233 draws, all in these methods, and their errors against
  the true graph on fresh draws were indistinguishable from those of a random ranking. Without
  changing a graph, the rank also moves `exact_search`'s `metadata["dag"]`, the member of the class
  it orients, and the last bits of the weights of `bottom_up_parce_lingam` and
  `longitudinal_lingam`, which order their regressions by it. No setting restores the former rank;
  `tie_break="position"` orders ties by column position, a different rule. ([#612])
- `iamb`, `inter_iamb`, and `mmpc` with `indep_test="chisq"` or `"gsq"` return the same graphs and
  p-values in less time, because the candidates of a grow step share one conditioning set, and the
  test now numbers that set's configurations once and counts the tables of all its candidates
  together, where each candidate's call numbered them again; `mmpc` reaches the same counts through
  the test's batch of statistics. On asia, child, insurance, alarm, and hepar2 with 1,000 and 5,000
  rows, the geometric mean of the speedup over the ten datasets was 2.4x (chisq) and 2.9x (gsq) for
  `iamb`, 2.2x and 2.6x for `inter_iamb`, and 1.6x and 1.8x for `mmpc`; `iamb` on hepar2 with 5,000
  rows fell from 12.3 s to 3.3 s with chisq and from 28.0 s to 5.5 s with gsq. The identity check's
  233 fits of every method but `calm`, 640 fits of the discrete methods on four networks, and
  86,163 recorded queries returned the same bits. ([#625])
- `iamb`, `inter_iamb`, and `mmpc` with `indep_test="gsq"` return the same graphs and p-values in
  less time: the test evaluates the degrees of freedom of all the tables of a scan in one pass,
  where each table shape evaluated its own, and reads the mean of a cell in a configuration of at
  most 128 rows from a table of 358 thousand floats (2.9 MB) that every test in the process shares
  and that fills as lookups need it. On asia, child, insurance, alarm, and hepar2 with 1,000 and
  5,000 rows, the geometric mean of the speedup over the ten datasets was 1.63x for `iamb`, 1.49x
  for `inter_iamb`, and 1.40x for `mmpc`, from 0.91x to 2.25x over the 30 settings; the three fits
  on asia with 5,000 rows were 3 to 7 ms slower, and `iamb` on hepar2 with 5,000 rows fell from 7.2
  to 4.8 seconds. Andrey stays slower than bnlearn's `mi-adf` in 28 of the 30 settings, taking 1.7
  to 10.7 times as long. The identity check's 239 fits of every method, 640 fits of the discrete
  methods on four networks, 68 on hepar2, and 86,163 recorded queries returned the same bits.
  ([#630])
- The reference-parity gate reports a published row marked `claim: "out"`, a setting measured
  outside the method's claim, with its numbers and does not fail it; a row without the field is
  judged as before. ([#631])
- Every method's docstring opens its Notes with a **When to use.** paragraph: the data, model,
  and scale the method is designed for, what it returns, and the method to use outside that
  setting. ([#633])

### Fixed

- `notears` rejects a `rho_max` at or below 1, the penalty weight its fit starts from, which ended
  the fit before its first solve and returned an empty graph without a warning. ([#639])
- `rcd` and `gin` no longer change their graph when a constant is added to every column. The kernel
  independence test they share computed squared distances as `x^2 + y^2 - 2xy`, which loses its
  digits far from 0. With `1e9` added to simulated data, `rcd` returned another graph on 5 of 6
  draws and `gin` on 6 of 6; now neither does on any. The statistic of a dependent pair fell from
  5.83 to 0.056 and its p-value rose from 0 to 0.005 at that offset. The test now takes the
  distances between mean-centered columns, and both methods center their columns first, as RCD and
  GIN assume zero-mean variables (Maeda and Shimizu 2020; Xie et al. 2020). On the identity check's
  239 draws no graph changed; the weights of `rcd` changed in the last bits on 4 of its 6, and every
  other method, `gin` included, is bit-identical. ([#619])
- `varma_lingam` no longer changes its graph when a constant is added to every column. Its
  likelihood fit takes a constant trend, so a column's mean is not part of the model, but a mean
  far from 0 stopped the fit's optimizer at its starting point. With `1e9` added to simulated
  data, and with `X * 3.7 + 1e6`, it returned another graph on 3 of 6 draws; now it does on none.
  The series is centered first, as the structural VAR model that VARMA-LiNGAM extends takes
  zero-mean series (Hyvarinen et al. 2010). On the identity check's 239 draws only the 6 of
  `varma_lingam` changed: 2 in their graph, with 1 and 2 fewer errors against the true graph, and 4
  in their weights; every other method is bit-identical. ([#629])

## [0.12.0] - 2026-10-30

### Highlights

- `chisq` and `gsq` hold their level in sparse tables: at 0.05 over 3 to 5 categories they rejected
  11.1% and 21.7% of true independences, and now 4.9% and 3.8%; at 5 categories with 3
  conditioning variables, 48% and 83%, now 3.2% and 0.8% (Tsamardinos and Borboudakis 2010).
  ([#574])
- `indep_test="normal_scores"`, a rank-based Fisher's z test, holds its level on skewed or
  heavy-tailed columns, where `"fisherz"` over-rejects; every constraint-based search takes it.
  ([#573])
- `andrey.markov_check` tests whether the data contradict a graph, such as a method's result
  (Ramsey, Andrews, and Spirtes 2024); `andrey markov-check` runs it from the command line. ([#571])
- `pcmci`, `pcmciplus`, and `granger` control the false discovery rate by default
  (`fdr_method="fdr_bh"`). ([#554], [#565], [#567])
- `bootstrap` subsamples half of the rows by default, and its averaged graph is a valid graph of
  the method's kind. ([#544], [#555], [#584])
- `GraphStructure` answers graph queries: `d_separated`, `m_separated`, `parents`, `ancestors`,
  `markov_blanket`, and more, by label or position. ([#569])
- `inter_iamb`, `mmhc`, and `rsmax2` are supported and join the quality gate: against bnlearn 5.2.1
  on simulated linear-Gaussian data with 20 to 100 variables, 1,000 and 5,000 rows, and mean degree
  2 and 4, bnlearn makes significantly fewer errors in none of the 72 comparisons, on the CPDAG,
  the skeleton, and the moral graph that `inter_iamb` returns. The median ratio of bnlearn's fit
  time to Andrey's is 3.8 for `inter_iamb`'s CPDAG, 6.6 for `mmhc`, and 7.8 for `rsmax2`. ([#581])

### Breaking

- These arguments, values, and settings have new names, so that one name means one thing across
  the package; the former ones still work in this release, in Python and on the command line, but
  give an `andrey.DeprecatedArgumentWarning` that names the replacement, and they stop working in
  the next release. The [upgrade guide](https://andrey.abel.ai/docs/guides/upgrading.html) shows
  the calls before and after. ([#551])

  | Method | Former name | Name now |
  | --- | --- | --- |
  | `grasp` | `depth` | `grasp_depth`, since Tetrad's `depth` is the conditioning-set cap, `max_depth` here |
  | `pcmci` | `pc_alpha` | `pc1_alpha`, the level of the PC1 step |
  | `pcmciplus` | `collider_rule="none"` | `collider_rule="sepsets"`, as in the other searches |
  | `andrey.bootstrap`, `andrey bootstrap` | `n_boot`, `--n-boot` | `n_resamples`, `--n-resamples`, as in SciPy |
  | `andrey.config` | `num_workers` | `n_jobs`, as in scikit-learn |
  | environment | `ANDREY_NUM_WORKERS` | `ANDREY_N_JOBS`; the former variable is read when the new one is unset |
  | `pc`, `fci`, `gfci`, `cdnod`, `iamb`, `inter_iamb`, `mmpc`, `mmhc`, `rsmax2`, `ges`, `gies`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp`, `pcmci`, `pcmciplus`, `lpcmci` | `max_categories` | none: values no longer declare a column's type, so it is ignored with a warning |

- No method reads a column's type from its values, whatever test or score you name. A mixed-data
  setting passed by name, such as `pc(df, indep_test="cond_gauss")`, read every column of at most
  `max_categories` distinct whole numbers as categorical; it now reads the columns you declare, by
  `categorical` or by a DataFrame's `category`, `bool`, or text type, as `indep_test=None` does, so
  a call that names the test the default recorded fits the same graph. An undeclared column of a
  few whole numbers reads as numbers, with the `DataTypeWarning` that it may hold category codes.
  For the former results, name those columns in `categorical`. ([#551])
- A declared categorical column behaves the same by either route. A test or score for continuous
  data, such as `"fisherz"` or `"bic"`, raises `ValueError` on a column of a pandas `category`,
  `bool`, or text type, as it did on one that `categorical` named, where it read the column as
  numbers with a warning; pass `categorical=[]` to read every column as numbers. A categorical test
  or score, such as `"chisq"` or `"bdeu"`, accepts `categorical`, where it raised, as it accepts
  the same columns by type, and reads every column as categorical. A `categorical` that is not a
  list raises `TypeError`, where it raised `ValueError` with a test or score for continuous or
  categorical data. ([#551])
- `ebic_gamma` takes values from 0 to 1, the range of the extended BIC (Chen and Chen 2008), where
  it took any value of at least 0, and a call that passes it with a score that does not add the
  extended-BIC term, any but `"bic"`, raises `ValueError`, where the score ignored it. ([#551])
- `andrey config --json` and `andrey.describe()` report the worker count as `n_jobs`,
  and `andrey calibrate` exports `ANDREY_N_JOBS`. ([#551])
- `andrey bootstrap --json` reports the number of resamples as `bootstrap.n_resamples`, where it
  reported `bootstrap.n_boot`. `BootstrapOutput.n_resamples` replaces `BootstrapOutput.n_boot`,
  which still works with an `andrey.DeprecatedArgumentWarning` and stops working in the next
  release. ([#551])
- `bootstrap` draws each resample as a subsample by default: `sample_fraction=0.5` of the rows,
  without replacement, instead of all the rows with replacement, so every default result changes
  (frequencies and averaged graph). A bootstrap resample repeats rows, so a test or a score sees
  more evidence than the resample holds; Banerjee, Andrews, and Kummerfeld (2025) find that
  subsampling half of the rows performs as well as the best correction. On linear data with 20 and
  50 variables and 1,000 and 5,000 rows, the averaged graph had no more errors (SHD) than the
  bootstrap's in any of 10 cells and fewer in 5, `ges` with 50 variables and 1,000 rows had 5.0
  fewer, and a fit takes 0.56 to 0.95 of the time. A time-series method on one series subsamples
  one stretch of consecutive rows, which had no more errors than the block bootstrap in any of 14
  cells of simulated series. Pass `resampling="bootstrap"` for the resampling of before; the moving
  block bootstrap's default block is now `max(ceil(n ** (1/3)), 10 * tau_max)` rows for a method
  with a `tau_max`, instead of `ceil(n ** (1/3))`. ([#544])
- `bootstrap`'s averaged graph gives each pair its most frequent state in the fits, no edge counted
  as one, where it gave each pair adjacent in at least `threshold` of the fits its most frequent
  edge. `threshold` now asks a pair to be adjacent in more than that share of the fits, or in all
  of them, and takes values from 0.5 to 1, where it took any share above 0. The vote is then made
  a valid graph of the method's kind by the published steps: the CPDAG (Chickering 2002) of a
  consistent DAG extension (Dor and Tarsi 1992) for a CPDAG method, the PAG of Zhang's (2008) MAG
  for a PAG method, and a DAG without a cycle for a DAG method, where the vote could be a partially
  directed graph labeled `cpdag`, a graph that is not a PAG, or a graph with a cycle. A DAG
  method's fits that split evenly between the two directions of an edge point it by the columns'
  `tie_break` ranks, so reordering the columns relabels the averaged graph, and
  `metadata["tie_break"]` records the rule. On linear
  data with 20 and 50 variables (`pc`, `ges`, `fci`, and `direct_lingam`; 24 cells of 20 datasets),
  the averaged graph had fewer errors (SHD) than before in 11 cells and more in none. When no valid
  graph of the kind holds the vote, the averaged graph is the resample graph nearest the others
  among the resamples that are valid graphs of the kind, Tetrad's `Median` of those, with an
  `andrey.AndreyWarning`; when no resample is valid either, the warning says so and
  `metadata["averaged_graph"]` is `"nearest_resample_invalid"`. Under `knowledge`, a PAG method's
  averaged graph is oriented as `fci` orients one: the knowledge's endpoints first, then Zhang's
  rules until no endpoint changes. ([#555], [#564], [#584])

- `granger` decides lagged links by a test at `alpha`, where it kept the nonzero coefficients of a
  cross-validated lasso. It fits each variable on the past `tau_max` steps of every variable by
  least squares and returns the links whose coefficient has a p-value of at most `alpha` (default
  0.05), after the correction `fdr_method` names: the F-test of that regressor (Granger 1969), the
  coefficient test of a vector autoregression. `lag_weights` hold the regression coefficients of the
  returned links, and `metadata["p_values"]` holds the p-value of every link, as `pcmci` does. On
  the quality gate's simulated series of 5 variables with 2 lags, the median errors (missing and
  extra lagged edges; the empty graph's is 28.5) fall from 13.5 to 6.5 at 1,000 steps and from 12.5
  to 4 at 2,000. With `fdr_method="none"`, at 80 or more variables with 1,000 steps the uncorrected
  tests return more errors than the empty graph, as the docstring's Limitations say. The lasso, its
  `cv` argument, and its worker threads are gone: `cv` is ignored with an
  `andrey.DeprecatedArgumentWarning` and stops being accepted in the next release. ([#556])

  | Method | Former argument | Now |
  | --- | --- | --- |
  | `granger` | `cv` | none: `alpha` sets the level of the test of each link |

- The former argument names and values that 0.11.0 kept for one release no longer work, as its
  changelog announced. A call that passes one raises `TypeError`, as for any argument a method does
  not take, and a former score name raises `ValueError`, which lists the scores the method takes;
  each message adds what the name or value is called now. `andrey run` and `andrey bootstrap`
  refuse the flags and the `key=value` names the same way, with the same hint, and
  the data inputs in `andrey --help --json` no longer list `aliases`. The
  [upgrade guide](https://andrey.abel.ai/docs/guides/upgrading.html) gives the current calls.
  ([#559])

  | Method | Removed name or value | Name now |
  | --- | --- | --- |
  | `ica_lingam` | `random_state` | `seed` |
  | `direct_lingam`, `multi_group_direct_lingam` | `random_state` | none: it set nothing |
  | `pcmci` | `alpha_level` | `alpha` |
  | `pcmciplus`, `lpcmci` | `pc_alpha` | `alpha` |
  | `rcd` | `ind_alpha` | `alpha` |
  | `calm` | `alpha` | `screen_alpha` |
  | `gfci`, `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp` | `lambda_value` | `penalty_discount` |
  | `gfci`, `ges`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func="local_score_BIC"`, `"local_score_BIC_from_cov"` | `score_func="bic"` |
  | `gfci`, `ges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func="local_score_BIC_categorical"` | `score_func="discrete_bic"` |
  | `gfci`, `ges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func="local_score_BDeu"` | `score_func="bdeu"` |
  | `gfci`, `ges`, `hc`, `exact_search`, `boss`, `grasp` | `score_func="local_score_cond_gauss"` | `score_func="cond_gauss"` |
  | `gfci`, `ges`, `hc`, `exact_search`, `boss`, `grasp` | `score_func="local_score_degen_gauss"` | `score_func="degen_gauss"` |
  | `granger` | `maxlag` | `tau_max` |
  | `longitudinal_lingam` | `n_lags` | `tau_max` |
  | `rcd` | `max_explanatory_vars` | `max_depth` |
  | `calm` | `max_outer` | `max_iter` |
  | `multi_group_direct_lingam` | `data_groups` | `data` |
  | `longitudinal_lingam` | `data_list` | `data` |
  | `andrey run`, `andrey bootstrap` | `--data-groups`, `--data-list` | `--data`, once per dataset |
  | `gin`, `calm` | `labels` | none: the labels come from a DataFrame's columns, or, in `andrey run`, from the data file's header |

- The default `fdr_method` of `pcmci`, `pcmciplus`, and `granger` is `"fdr_bh"`, where it was
  `"none"` in the pull requests that added it ([#554], [#565]), so a link is kept when its
  Benjamini-Hochberg adjusted p-value over all the lagged links is at most `alpha`. On simulated
  linear-Gaussian VAR series of 5 to 80 variables, 500 and 1,000 steps, and `tau_max` 1 and 2 (20
  cells of 20 series each), the correction had fewer errors (missing plus extra lagged links) than
  none in 16 cells for `pcmci` and 15 for `granger`, by the paired 95% bootstrap interval of the
  difference, the same in 3 and 4, and more in 1 and 1: 5 variables, 500 steps, and `tau_max=2`, by
  0.85 and 0.95. At 80 variables with 1,000 steps and `tau_max=2` the errors fell from 779 to 252
  (`pcmci`) and from 768 to 256 (`granger`), against the empty graph's 481: the extra links fell
  from 616 and 603 to 13, and the missed true links rose from 163 and 165 to 239 and 243.
  `pcmciplus`'s links do not change; its lagged `metadata["p_values"]` are the adjusted ones.
  `fdr_method="none"` returns `pcmci`'s results of 0.11.0 and `granger`'s uncorrected test. The
  quality gate's medians move: PCMCI's from 8 and 9.5 to 2 and 0.5, and Granger's from 5.5 and 3.5
  to 6.5 and 4. ([#567])
- `iamb`, `inter_iamb`, and `mmpc` take `structure`, the graph they return without a `target`.
  `iamb` and `inter_iamb` return a CPDAG by default, kind `cpdag`, where they returned the moral
  graph of the Markov blankets, as the R package bnlearn's `iamb` and `inter.iamb` return one by
  default; `structure="moral"` returns the 0.11.0 graph bit for bit. `mmpc` returns the skeleton by
  default, unchanged, as bnlearn's `mmpc` does, and `structure="cpdag"` returns its CPDAG. The CPDAG
  is the Grow-Shrink structure learner's (Margaritis 2003): each pair of the moral graph that a
  subset of the smaller blanket separates leaves, which gives the skeleton; each unshielded triple
  is a collider when its ends stay dependent given the middle node and every subset of the smaller
  blanket; and Meek's rules orient the rest. `structure="skeleton"` on `iamb` and `inter_iamb`
  returns the moral graph without the links between a child's parents. With a `target`, each returns
  the variable's set, as before. ([#572])

### Added

- `direct_lingam`, `multi_group_direct_lingam`, `varma_lingam`, and `longitudinal_lingam` take
  `measure="skew"` and `measure="tanh"`, the first-order approximations of the pairwise likelihood
  ratios of Hyvarinen and Smith (2013), beside the default `"pwling"`: `"tanh"` suits sparse noise,
  as for Laplace, and `"skew"` skewed noise, with either sign. Both read the standardized columns
  and cost one matrix product per step, so the order search took 0.1 to 0.3 of `"pwling"`'s time.
  On simulated data with 10 and 20 variables and 500 and 2,000 rows, `"pwling"` made no more
  errors (SHD) than either in any cell, and with uniform noise `"tanh"` orients backwards and
  `"skew"` at random, so `"pwling"` stays the default. ([#575])
- `GraphStructure` answers graph queries: `parents`, `children`, `ancestors`, `descendants`,
  `markov_blanket`, `d_separated`, and `m_separated`, each taking a node by label or by position.
  `d_separated` is d-separation on a DAG or a CPDAG; `m_separated` is m-separation on a MAG, a PAG,
  or a DAG, and graph separation on an undirected graph; each raises on the kinds the other one
  defines. A CPDAG or PAG answers as every member of its class does. `neighbors` and `endpoints`
  also take a label. ([#569])
- `pcmci` and `pcmciplus` take `fdr_method`, as tigramite's `run_pcmci` and `run_pcmciplus` do:
  `"fdr_bh"` replaces the p-values of the lagged links with their Benjamini-Hochberg adjusted
  values over all those links together. `pcmci` keeps a link when its adjusted p-value is at most
  `alpha`; `pcmciplus` decides its links in the skeleton phase and only reports the adjusted values
  in `metadata["p_values"]`. `"none"` keeps the p-values as tested, which returns the results
  these methods returned before. `lpcmci` takes none, as tigramite's does not. ([#554])
- `granger` takes `fdr_method`, as `pcmci` and `pcmciplus` do: `"fdr_bh"` replaces the p-values of
  the lagged links with their Benjamini-Hochberg adjusted values over all those links together,
  and a link is kept when its adjusted p-value is at most `alpha`; `"none"` keeps the p-values as
  tested. On the simulated series of 80 and 160 variables with 1,000 steps,
  where the uncorrected tests return more errors than the empty graph (median 762.5 against 494,
  and 2,855 against 973), `"fdr_bh"` returns fewer (266.5 and 546), at the cost of more missed
  true links. ([#565])
- `boss` and `grasp` take `n_starts`, the number of searches a fit runs, each later one from a
  random order drawn from `seed`, and return the best-scoring one; the default, 1, returns the
  graphs these methods returned before. ([#550])
- `hc` and `tabu` take `n_starts`, `n_perturbations`, and `seed`: each search after the first
  starts from the best DAG so far after `n_perturbations` random edge additions, removals, or
  reversals drawn from `seed`, by default as many as there are variables, and the fit returns the
  best-scoring one; the default, 1, returns the graphs these methods returned before. ([#552])
- `bootstrap` takes `resampling`, `"subsample"` or `"bootstrap"`, and `sample_fraction`, the share
  of rows in a subsample; for a time-series method on one series a subsample is one stretch of
  consecutive rows. `andrey bootstrap` takes `--resampling` and `--sample-fraction`, and the result
  records both. ([#544])
- `bootstrap` and `BootstrapOutput.from_fits` take `strict`: `True` raises `ValueError` when no
  valid graph of the method's kind holds the vote of the fits, instead of returning the resample
  graph nearest the others with a warning. `metadata["averaged_graph"]` records which graph came
  back, `"vote"`, `"nearest_resample"`, or `"nearest_resample_invalid"`. `andrey bootstrap` takes
  `--strict`, and its envelope records both. The example notebook `bootstrap.ipynb` shows how to
  read the edge frequencies and the averaged graph. ([#555])
- `calm`, `iamb`, `inter_iamb`, and `mmpc` take `knowledge` with all four parts, as the other
  searches do: a required edge is always in the result, a pair forbidden both ways or outside
  `allowed` never is, and `calm` also leaves out every forbidden edge and every edge into an earlier
  tier. ([#521])
- `pcmci`, `pcmciplus`, and `lpcmci` take every `indep_test` the other conditional-independence
  searches take: `"chisq"` and `"gsq"` for categorical series, `"degen_gauss"` and `"cond_gauss"`
  for mixed ones, and the nonlinear `"kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"`, each on the lagged
  values under test-wise deletion, with `categorical`, `max_categories`, and `seed`. Candidate
  conditions are ordered by each test's statistic, as in tigramite. `None` still picks from the
  declared column types. A test registered with `register_indep_test` needs only the single-query
  call, as in every other search with an `indep_test`, under missing values too: the family orders
  its candidates by p-value, smallest first. ([#515])
- `gies`, `xges`, and `gfci` take `max_parents`, a cap on every node's parents, as `ges` does: no
  DAG the returned graph represents gives a node more than that many parents, and a move that would
  give a node more is skipped. `gfci` caps its GES phase; the PAG its tests then orient can put more
  arrowheads at a node. Required edges of `knowledge` above the cap raise `ValueError`. The default,
  `None`, sets no cap and returns the graphs these methods returned before. ([#516])
- `gies` and `xges` take every score the other score searches take, `"bdeu"` and `"discrete_bic"`
  for categorical data and `"cond_gauss"` and `"degen_gauss"` for mixed data besides `"bic"`, with
  `categorical` and `max_categories`, and `score_func=None` picks from the declared column types as
  in the other score searches; `gies` scores each variable on the rows without an intervention on
  it, with BDeu as Cooper and Yoo (1999) define it for a mix of experimental and observational
  data. ([#520])
- `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`, `gfci`, `mmhc`, and `rsmax2`
  take `equivalent_sample_size`, the equivalent sample size of the BDeu score (Heckerman, Geiger and
  Chickering 1995; Buntine 1991): the number of rows its uniform prior over each variable's
  conditional distributions is worth. The default, `1.0`, is the weight the score used before, so
  every default result is unchanged. Only `score_func="bdeu"` reads it, so a call that passes it
  with another score, named or picked from the declared column types, raises `ValueError`; the
  result records the value a BDeu fit used as `metadata["equivalent_sample_size"]`. ([#553])
- `andrey.markov_check` tests whether the data contradict a DAG or a CPDAG, such as a method's
  result: it tests the independences the graph implies, each variable and every earlier non-parent
  given its parents, and rejects the graph when their p-values are not uniform on (0, 1) by the
  Anderson-Darling test (Ramsey, Andrews, and Spirtes 2024). `indep_test` is picked from the
  declared column types as in `pc`, and each test reads its own random half of the rows
  (`sample_fraction`), which keeps a true graph's rejection rate at 6% to 9% for `alpha=0.05`,
  against 9% to 21% on all rows. The result lists every tested independence with its p-value
  (`table()`), and `andrey markov-check GRAPH DATA` runs the check from the command line. The
  docstring gives the rates measured for missing, extra, and reversed edges and for categorical
  data. ([#571])
- `indep_test="normal_scores"` runs Fisher's z test on the normal scores of each column, a
  rank-based test for data whose columns are monotone transformations of Gaussian variables (a
  Gaussian copula): its p-values do not change under a strictly increasing transformation of a
  column, and it holds its level where `"fisherz"` over-rejects on skewed or heavy-tailed marginals.
  `pc`, `fci`, `gfci`, `cdnod`, `iamb`, `inter_iamb`, `mmpc`, `mmhc`, `rsmax2`, `pcmci`,
  `pcmciplus`, and `lpcmci` take it; on data with missing values, each test ranks within the rows it
  reads. It finds only monotone dependence. The default test is unchanged. ([#573])

### Changed

- A score search records the penalties its score used: `metadata["penalty_discount"]` for every
  score but `"bdeu"`, and `metadata["ebic_gamma"]` for `"bic"`. Every fit whose result
  `andrey.config`'s `tie_break` can change records the rule in `metadata["tie_break"]`, so the
  metadata names every setting a default or the configuration chose. ([#551])
- `andrey --help --json` gives each package-wide concept its template parameter, with its default
  and bounds, such as `alpha` in the open interval (0, 1), and each method that differs from it,
  with the reason, such as the `alpha` of 0.01 that `pcmciplus` and `rcd` take from their papers.
  ([#551])

### Fixed

- `rsmax2(restrict="iamb")` and `restrict="inter_iamb"` restrict the score search to the skeleton
  of `iamb` or `inter_iamb` (`structure="skeleton"`), as Friedman, Nachman and Pe'er (1999) and
  Tsamardinos, Brown and Aliferis (2006) restrict to the restrict algorithm's skeleton and as
  bnlearn's `rsmax2` does, where they searched the moral graph, which also joins the parents of a
  common child. The graphs of those two restricts change; `mmpc`, `pc`, and `mmhc` fit as before.
  On 15 simulated linear-Gaussian datasets of 20 to 100 variables the candidate pairs fell by 37%
  to 61%, and the mean SHD to the true CPDAG from 33.7 to 12.8, against 13.1 for bnlearn's
  `rsmax2(restrict="iamb")`. ([#577])
- `chisq` and `gsq` no longer reject true independences too often in sparse tables. They referred
  the statistic to `(r_x - 1) * (r_y - 1)` degrees of freedom per configuration, which rejected
  11.1% (`chisq`) and 21.7% (`gsq`) of true independences at level 0.05 over 3 to 5 categories,
  conditioning sets of 0 to 3 variables, and 500 to 5,000 rows, and 48% and 83% at 5 categories
  with 3 conditioning variables. The degrees of freedom are now the statistic's mean when `y` is
  shuffled within each configuration, as Tsamardinos and Borboudakis (2010) test with, worked out
  exactly; the rates are 4.9% and 3.8%, and 3.2% and 0.8%. The graphs of every search that runs
  these tests on discrete data can change, `gsq` finds fewer edges at small sizes, and `gsq` takes
  about three times as long. `pc` with `gsq`, the default for categorical data, recovered graphs
  no farther from the truth than with `chisq` in 16 settings and nearer in 3. ([#574])
- The `max_depth` of `pcmci` and `pcmciplus` says in `andrey --help --json` that it caps the PC1
  step, as their docstrings do, where it said every test. ([#551])
- Every `seed` says first that `None` is the configured seed, `ANDREY_SEED` or 0 when it is unset,
  not a fresh random one, so a fit repeats. ([#551])

## [0.11.0] - 2026-10-29

### Highlights

- Ready for real data: eleven methods accept missing values, each conditional-independence test
  reading the samples where its variables are observed; `andrey.bootstrap` gives each edge's
  frequency over refits on bootstrap samples; and 27 of the 34 methods take background knowledge.
  On simulated data with 20 variables, 1,000 samples, and a tenth of the values missing, dropping
  every incomplete sample keeps about 125, while PC with test-wise deletion keeps them all, with a
  median of 6.5 errors against 8.5. ([#396], [#410], [#443], [#453])
- The extended BIC is the default penalty of ten score searches, so a large graph keeps fewer false
  edges: over 299 simulated datasets, GES's median errors fall from 87 to 25. ([#445])

### Breaking

- This release renames some arguments and changes some defaults and outputs, so that methods that
  do the same thing share one name and one behavior. The four tables below list every such change.
  A former name or value in the first table still works in this release, in Python and in
  `andrey run` and `andrey bootstrap`, but gives an `andrey.DeprecatedArgumentWarning` that names
  its replacement, and it stops working in the next release. The
  [upgrade guide](https://andrey.abel.ai/docs/guides/upgrading.html) shows the common calls before
  and after. ([#496], [#499], [#501], [#505], [#509], [#510], [#526])

  These arguments and values have new names, and the former ones still work in this release:

  | Method | Former name | Name now |
  | --- | --- | --- |
  | `ica_lingam` | `random_state` | `seed` |
  | `direct_lingam`, `multi_group_direct_lingam` | `random_state` | none: it set nothing, and is ignored with a warning |
  | `pcmci` | `alpha_level` | `alpha` |
  | `pcmciplus`, `lpcmci` | `pc_alpha` | `alpha` |
  | `rcd` | `ind_alpha` | `alpha` |
  | `calm` | `alpha` | `screen_alpha` |
  | `gfci`, `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp` | `lambda_value` | `penalty_discount` |
  | `gfci`, `ges`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func="local_score_BIC"`, `"local_score_BIC_from_cov"` | `score_func="bic"` |
  | `gfci`, `ges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func="local_score_BIC_categorical"` | `score_func="discrete_bic"` |
  | `gfci`, `ges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func="local_score_BDeu"` | `score_func="bdeu"` |
  | `gfci`, `ges`, `hc`, `exact_search`, `boss`, `grasp` | `score_func="local_score_cond_gauss"` | `score_func="cond_gauss"` |
  | `gfci`, `ges`, `hc`, `exact_search`, `boss`, `grasp` | `score_func="local_score_degen_gauss"` | `score_func="degen_gauss"` |
  | `granger` | `maxlag` | `tau_max` |
  | `longitudinal_lingam` | `n_lags` | `tau_max` |
  | `rcd` | `max_explanatory_vars` | `max_depth` |
  | `calm` | `max_outer` | `max_iter` |
  | `multi_group_direct_lingam` | `data_groups` | `data` |
  | `longitudinal_lingam` | `data_list` | `data` |
  | `andrey run`, `andrey bootstrap` | `--data-groups`, `--data-list` | `--data`, once per dataset |
  | `gin`, `calm` | `labels` | none: ignored with a warning; the labels come from a DataFrame's columns, or, in `andrey run`, from the data file's header |

  These arguments have new names, and the former names no longer work: passing one raises
  `TypeError`.

  | Function | Former name | Name now |
  | --- | --- | --- |
  | `andrey.core.score`'s `BICScore`, `CategoricalBICScore`, `CGScore`, `DGScore`, `make_score`, `local_score_bic`, and `penalty_weight` | `lambda_value` | `penalty_discount` |
  | `andrey.bootstrap`, new in this release | `num_workers`, in pre-release builds | none: `andrey.config` or `ANDREY_NUM_WORKERS` sets the workers |
  | `lpcmci` | `max_depth`, in pre-release builds | `max_p_global` |

  These defaults have changed; the last column gives the setting that returns the former results:

  | Method | Argument | Before | Now | For the former results |
  | --- | --- | --- | --- | --- |
  | `pc`, `fci`, `gfci`, `cdnod` | `indep_test` | `"fisherz"` | `None`: `gsq` when every column is declared categorical, `cond_gauss` when some are, else `fisherz` | `indep_test="fisherz"` |
  | `gfci` | `score_func` | `"local_score_BIC"` | `None`: `bdeu` when every column is declared categorical, `cond_gauss` when some are, else `bic` | `score_func="bic"` |
  | `xges` | `score_func` | `"local_score_BIC"` | `None`, which is `bic` | `score_func="bic"` |
  | `ges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp` | `score_func` | `None`, which read a column of a few whole numbers as categorical | `None`, which reads only the declared column types | the score by name, or `categorical` |
  | `pcmci`, `pcmciplus`, `lpcmci` | `indep_test` | `"student_t"` | `None`: `fisherz` on numeric columns | `indep_test="student_t"` |
  | `granger` | `tau_max` (`maxlag`) | 2 | 1 | `tau_max=2` |
  | `bottom_up_parce_lingam` | `alpha` | 0.1 | 0.05 | `alpha=0.1` |
  | `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`, `mmhc`, `rsmax2` | `ebic_gamma`, new | the standard BIC | 1.0, the extended BIC ([#445]) | `ebic_gamma=0.0` |
  | `andrey.data.sample_scm`, `SCM.sample`, `andrey.data.generate` | `scale` | `"raw"` | `"standardize"` ([#411]) | `scale="raw"` |

  These outputs have changed:

  | Method | Before | Now |
  | --- | --- | --- |
  | `exact_search` | the DAG it found, kind `dag` | the CPDAG of its equivalence class, kind `cpdag`, with the DAG in `metadata["dag"]` |
  | `iamb`, `inter_iamb`, `mmpc`, new in this release | kind `cpdag`, in pre-release builds | kind `undirected`, with the same edges |
  | `andrey.data.latent.marginal(dag, "mag")` | kind `pag` | kind `mag`, with the same marks ([#418]) |
  | every method | `metadata["algorithm"]` a display name, such as `"PC"` or `"ICALiNGAM"` | the method's name, such as `"pc"` or `"ica_lingam"` |
  | every method with a test or a score | no record of a test or score that `None` picked | `metadata["indep_test"]` and `metadata["score_func"]`, the test and the score the fit used ([#524]) |
  | `anm`, `pnl` | `metadata["pval_forward"]`, `["pval_backward"]`, `["stat_forward"]`, and `["stat_backward"]` | `metadata["p_values"]` and `["statistics"]`, each with `forward` and `backward` |
  | `bottom_up_parce_lingam` | `metadata["causal_order"]` | `metadata["partial_order"]` |
  | `gin` | `metadata["causal_order"]` | `metadata["cluster_order"]` |
  | `pcmciplus` | `metadata["conflicts"]` as `[i, j]` pairs | `[i, 0, j]` links, as in `lpcmci` |
  | `calm`, `notears`, `dagma` | `metadata["n_edges"]` | removed: a count of the result |
  | `notears`, `dagma` | `metadata["robust"]` | removed: it repeated the argument |
  | `mmhc`, `rsmax2` | `metadata["restrict"]` and `["maximize"]` | removed: they repeated the arguments |
  | `multi_group_direct_lingam` | `metadata["n_groups"]` | removed: the number of results |
  | `gin` | `metadata["n_latents"]` | removed: the number of clusters |
  | every method | each method's own error type and wording; `NotImplementedError` for an unknown `indep_test`, `score_func`, or `measure` | `TypeError` for a value of the wrong type, `ValueError` for one out of range or not a choice, in one message that names the method, the argument, what it accepts, and the value |
  | `ica_lingam`'s FastICA, the LiNGAM methods' adaptive Lasso | scikit-learn's `ConvergenceWarning` | `andrey.SearchLimitWarning` |

- `andrey.data.sample_scm`, `SCM.sample`, and `andrey.data.generate` z-score every column by
  default, `scale="standardize"`, so no method can read the causal order off the variances the
  simulation happened to give; VAR and VARMA series take `"standardize"` too. Pass `scale="raw"`
  for the values these functions returned before. ([#411])
- `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`, `mmhc`, and `rsmax2` score
  continuous data with the extended BIC by default: the linear-Gaussian BIC adds
  `2 * ebic_gamma * log C(n_variables - 1, n_parents)` to each variable's score, with
  `ebic_gamma=1.0` (Chen and Chen 2008), a penalty that grows with the number of variables, so the
  graphs keep fewer false edges. On simulated linear-Gaussian data with 20 to 400 variables and 200
  to 4,000 rows, the mean errors (SHD) per true edge fell by 0.74 to 0.83 for GES, XGES, GIES, hill
  climbing, Tabu search, BOSS, and GRaSP, and by 0.10 for MMHC and RSMAX2, whose skeleton phase
  already leaves out most false edges; exact search's fell by 0.24 at 20 variables. GES's median
  errors fell from 87 to 25 over 299 datasets. `gfci` keeps `ebic_gamma=0.0`: its
  errors fell by only 0.06 per true edge, and rose on some graphs. Pass `ebic_gamma=0.0` for
  the graphs the others returned before. ([#445])
- Every method checks its arguments in one shared step before it reads the data: a value of the
  wrong type raises `TypeError`, and a value outside its range or not one of its choices raises
  `ValueError`, each with one message that names the method, the argument, what it accepts, and the
  value, such as `pc: alpha must be a number in (0, 1), got 1.5`. An unknown `indep_test`,
  `score_func`, or `measure` raised `NotImplementedError` before. ([#496])
- `tau_max` is the largest lag of every lagged method, and `granger`'s default lag is 1, as in the
  other methods, instead of 2; pass `tau_max=2` for the graphs it returned before. The
  `order=(p, q)` of `varma_lingam` is the ARMA order and keeps its name. ([#510])
- `calm` lists its parameters in its signature, with ranges checked in the one shared step, instead
  of taking `**params`; the names and defaults are the same except the two the first table
  renames. ([#510])
- `indep_test=None` and `score_func=None` are the defaults of every method with a test or a score,
  and pick from the column types you declare, with `categorical` or a DataFrame's `category`,
  `bool`, or text columns: `gsq` and `bdeu` when every column is categorical, `cond_gauss` when
  some are, else `fisherz` and `bic`. Values never decide: a column of a few whole numbers that you
  have not declared reads as numbers, where score searches switched to BDeu or the conditional
  Gaussian score before, and every method that reads columns as numbers warns about such columns
  with the same `DataTypeWarning`, the LiNGAM, time-series, continuous-optimization, and pairwise
  methods included. ([#501])
- `pcmci`, `pcmciplus`, and `lpcmci` default to `indep_test="fisherz"`, the default of every other
  method with a test; pass `indep_test="student_t"` for the results of before. ([#501])
- `bottom_up_parce_lingam` defaults to `alpha=0.05`, the level of the method's published
  experiments (Tashiro et al. 2014), instead of 0.1. ([#501])
- `exact_search` returns the CPDAG of the optimal equivalence class, as the other score searches
  do; `metadata["dag"]` keeps the DAG it found. `iamb`, `inter_iamb`, and `mmpc` return the new
  graph kind `undirected` instead of `cpdag`, with the same edges: their edges claim no direction,
  and a moral graph's do not even claim adjacency. `andrey.metrics` scores an undirected graph on
  its skeleton, so its `shd` is the skeleton SHD, and refuses to orient or convert it. ([#509])
- Metadata keys share one name and shape across methods, and each method's spec lists its keys
  (`output.metadata` in `andrey --help --json`). ([#509])

### Added

- `granger` takes `knowledge`: each variable's regression drops the lagged regressors the
  knowledge forbids and keeps the ones it requires unpenalized, and the lasso selects among the
  others. `varma_lingam` and `multi_group_direct_lingam` take `measure`, as `direct_lingam` and
  `longitudinal_lingam` do; `"pwling"` is the only measure. ([#510])
- `pc` and `cdnod` report `metadata["conflicts"]`, the edges two colliders ask to orient both ways.
  `bottom_up_parce_lingam` honors required edges in its `knowledge`, as the other LiNGAM methods
  do. ([#509])
- Experimental hybrid searches: `mmhc` learns MMPC's skeleton and then hill climbs within it,
  `rsmax2` pairs a restrict phase (`"mmpc"`, `"iamb"`, `"inter_iamb"`, or `"pc"`) with hill climbing
  or tabu search, and `iamb`, `inter_iamb`, and `mmpc` return one variable's Markov blanket or
  parents and children with `target=`, or every variable's set without it. ([#400])
- Experimental `notears` and `dagma`, continuous-optimization learners that fit a weighted DAG by
  minimizing a loss under a smooth acyclicity constraint, with NumPy and SciPy only: NOTEARS with
  the least-squares loss and the trace-exponential constraint, DAGMA with its log-determinant
  constraint and central path. Both take the papers' defaults, honor all four parts of `knowledge`,
  and fit on rank-ordered columns, so they draw no random numbers and do not depend on the column
  order. `robust=True` follows the first fit with the likelihood path of Deng et al. (2024): the
  Gaussian likelihood with one noise variance per variable and a quasi-MCP penalty, refit while the
  likelihood falls. ([#430])
- `andrey.bootstrap(method, data)` refits a method on bootstrap resamples of the rows and returns a
  `BootstrapOutput`: the share of fits in which each pair has each edge (`->`, `<-`, `--`, `<->`,
  `o->`, and the rest), a table of every edge that appeared, and an averaged graph that keeps the
  pairs adjacent in at least `threshold` of the fits. It takes every method but `gin`; a
  time-series method on one series resamples blocks of rows; the result is the same at any number of
  workers, which `andrey.config` sets, and a seeded method fits each resample with a seed derived
  from `seed`. `andrey bootstrap <method>` does the same from the command line. ([#396], [#499])
- `scale="unitless"` in `andrey.data.sample_scm` and `SCM.sample` draws a linear model's weights
  and noise so that every variable has variance 1 and its parameters depend only on its number of
  parents (Herman et al. 2025), so on average neither the variances nor the share of each variable
  the others explain follows the causal order; it takes up to 1,000 variables.
  `andrey.data.generate` takes `scale`. For contributors, the benchmark harness has unitless
  variants of its four linear regimes. ([#411])
- `pcmci`, `pcmciplus`, `lpcmci`, `varma_lingam`, `longitudinal_lingam`, `ica_lingam`,
  `multi_group_direct_lingam`, `rcd`, and `bottom_up_parce_lingam` (every part but required edges)
  take `knowledge`, where a time-series method reads `(cause, effect, lag)` as the link from
  `cause` at time `t - lag` to `effect` at time `t`, so you can rule links in or out at one lag.
  ([#410])
- `GraphStructure.to_dag`, `to_mag`, and `to_pag` turn a CPDAG into one DAG of its class, asking
  how to pick one when the class holds several, a DAG with hidden and selection variables into
  its maximal ancestral graph, and a DAG or MAG into its PAG; the new `"mag"` kind holds a MAG,
  which scores and draws as a PAG and reads and writes pcalg's `amat.pag`; and `to_dot()` writes
  any graph as Graphviz DOT text, which the graph-representation guide hands to DoWhy. ([#418])
- `pc`, `cdnod`, `fci`, and `gfci` take `collider_rule="conservative"` or `"majority"`, which decide
  each collider by a vote over the separating subsets of its ends' neighbors and list the triples
  the vote leaves undecided in `metadata["ambiguous_triples"]`. ([#397])
- `ebic_gamma`, the weight of the extended-BIC term, for every method that scores with the
  linear-Gaussian BIC: `ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`,
  `gfci`, `mmhc`, and `rsmax2`, and for `andrey.core.score.BICScore`, with the term itself in
  `andrey.core.score.ebic_penalty`. `0.0` is the standard BIC; the other scores ignore it. ([#445])
- `pc`, `fci`, `cdnod`, `iamb`, `inter_iamb`, `mmpc`, `pcmci`, `pcmciplus`, `lpcmci`, `anm`, and
  `pnl` accept missing values, a `NaN`, pandas' `NA`, or an empty, `NA`, or `N/A` field of a CSV
  file, each test reading only the rows where all of its variables are observed, `fisherz` and
  `student_t` computing each test's correlations directly from those rows, many tests at once, and
  `metadata["rows_tested"]` giving the fewest and the median rows a test read, while every other
  method refuses them with an error that names it and says what to do instead. ([#443], [#453])
- With more than one worker, `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="rcot"` run each
  large skeleton pass in worker processes, each conditioning set's queries in one worker, and
  return the same graphs and p-values as with one worker. `ANDREY_RCOT_PARALLEL_MIN_WORK` sets the
  least work, RCoT tests times rows, a pass needs to use them. The default, 2000000, was measured on
  an Intel Core i7-7820X running Linux with 4 workers; a fit with more than one worker warns once
  on it when a pass reaches a tenth of it. `andrey calibrate --method rcot` measures the value for
  your machine, and `andrey calibrate` now measures it with the others. ([#457])
- `lpcmci` and `pcmciplus` take `max_p_global`, tigramite's name for the most conditions a test
  takes beyond the default conditions, the parents of both ends, which also bounds the subsets of
  the majority vote; unlike `max_depth`, it does not count those parents. On `pcmciplus`,
  `max_depth` caps the lagged phase and `max_p_global` the same-step skeleton and collider phases.
  Their number of tests grows exponentially with the links at one node, so on many strongly
  related series an uncapped run may not finish; a cap bounds the number of tests. The default,
  `None`, keeps the uncapped search. ([#468], [#505])
- `max_depth`, the maximum conditioning-set size of every conditional-independence test, on every
  method that grows conditioning sets: `iamb`, `inter_iamb`, and `mmpc`, the restrict phase of
  `mmhc` and `rsmax2`, and the PC1 step of `pcmci` and `pcmciplus`, as on `pc`, `fci`, `gfci`, and
  `cdnod`. `max_parents`, a cap on every node's parents in every DAG the result represents, on
  `hc`, `tabu`, `exact_search`, `boss`, `grasp`, and the score phase of `mmhc` and `rsmax2`, as on
  `ges`. Both default to `None`, no cap, which returns the graphs these methods returned before.
  ([#505])
- `notears`, `dagma`, `calm`, and `varma_lingam` warn once with `andrey.SearchLimitWarning` when
  they stop on an iteration cap rather than on their own stopping rule, as `hc` and `tabu` do; each
  docstring names the cap's unit. ([#505])
- A fit larger than its method was measured to finish warns once per method with
  `andrey.BoundaryWarning`: when the data have more variables (for `anm` and `pnl`, more rows) than
  the most any mean degree finished within the caps of the newest benchmark's boundary lane, at the
  setting whose rows are nearest the data's, and that ladder ended on the time or memory cap, as
  exact search does past 24 variables at 2,000 rows. The message names the measured size and its
  setting and points to the method's **Limitations** paragraph and the release card; the sizes are
  generated from the published summary by `scripts/boundary_table.py`. ([#479])
- `andrey --help --json` gives each parameter its bounds (`minimum`, `maximum`,
  `exclusiveMinimum`, `exclusiveMaximum`), the package-wide `concept` it implements, and the former
  names (`aliases`) and values (`choice_aliases`) it accepts, and lists each concept, such as
  `alpha` or `max_depth`, with the name each method gives it; `andrey config --help --json` lists
  the concepts too. `andrey run` accepts a former name or value and rejects a value out of range
  before the fit. ([#492])
- `andrey.DeprecatedArgumentWarning`, the warning a renamed argument or value gives while its former
  name or value still works, for one release. ([#496])
- `tabu` takes `categorical`, `max_categories`, and the mixed-data scores `"cond_gauss"` and
  `"degen_gauss"`; `exact_search` takes `penalty_discount`; `gies` takes `score_func`. ([#501])
- `metadata["indep_test"]` and `metadata["score_func"]` name the test and the score a fit used, as
  the current short name, in every method with a test or a score: the one you passed, or, for
  `None`, the one picked from the declared column types, so a default choice is visible. A
  registered test is recorded under its registered name. ([#524])

### Changed

- `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="cmiknn"` return the same graphs and p-values
  in less time on CPUs without AVX512_ICL, such as the AMD EPYC 7713, because CMIknn selects each
  row's `k`-th neighbor distance in 32-bit integers, which NumPy sorts with SIMD on AVX2. On an
  EPYC 7713 a fit on 10 variables and 1,000 rows took 6.5 minutes instead of 22, and the quality
  gate's CMIknn fits took half as long. On CPUs with AVX512_ICL, such as the EPYC 9534, the
  selection is unchanged and the same fits take about 10% less time. ([#391])
- `andrey.data.latent.marginal`, `GraphStructure.to_mag`, and `to_pag` decide each pair of observed
  nodes by one moral-graph search from each node instead of a d-separation test per pair, and
  return the same graphs in less time. ([#481])
- `grasp` tucks only the edges of the current DAG and never takes a chain of tucks back to a DAG it
  has visited, as the GRaSP paper's search does, so it returns a different graph on some data,
  one that scores better about as often as worse. ([#442])
- `varma_lingam`'s **Limitations** paragraph and the column-order guide say that where its
  likelihood fit stops depends on the BLAS kernels the CPU runs, even when the fit converges, so
  the same series can give a different graph on another type of CPU. ([#451])
- `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="rcot"` return the same graphs and p-values in
  less time on large samples, because their skeleton search hands RCoT its queries in stacks, which
  RCoT runs one conditioning set at a time, building each set's random features once per stack.
  ([#455])
- `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="degen_gauss"` return the same graphs and
  p-values in less time, because their skeleton search hands the degenerate Gaussian test its
  queries in stacks, which it runs grouped by shape, with each conditioning set's regression
  computed once. ([#459])
- `fci` and `gfci` return the same graphs and p-values in less time, because their
  possible-d-separation step asks the test for each pair's conditioning sets in runs of 1, 2, 4, and
  up to 256 sets, cut at the first that separates the pair, when the test is `"fisherz"`,
  `"student_t"`, `"chisq"`, `"gsq"`, `"degen_gauss"`, or `"cond_gauss"`. ([#461])
- `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="chisq"` or `"gsq"` return the same graphs and
  p-values in less time, because the test's batched counts hold a table only for each conditioning
  configuration that occurs in the data, not for every possible one. ([#463])
- `lpcmci` returns the same graphs in less time on long searches, because a removal step builds a
  batch of conditioning sets as one array of column numbers, records its tests once per batch, and,
  without missing values, tests sets of more than three neighbors, which rarely repeat, without the
  result cache. ([#468])
- `andrey.data.latent.marginal(dag, "pag")` and `GraphStructure.to_pag` return the same graphs in
  less time on dense graphs with hidden variables, because the orientation rules R9 and R10, which
  `fci` and `gfci` share, make each of their path searches once per pass instead of once for each
  pair of nodes that needs it. ([#485])
- `ica_lingam`'s **Limitations** paragraph says that its causal order depends on the columns'
  scale, as its authors note, with one measurement on standardized data, and points to
  `direct_lingam`. ([#487])
- The **Limitations** paragraphs of `notears` and `dagma` say that the `robust=True` fit's graph
  changes when the columns are rescaled, even by powers of two, and that the penalized
  likelihood's global minimizers keep their graph under rescaling with a small enough penalty.
  ([#495])
- `boss`, `grasp`, and `ica_lingam` record the seed they used in `metadata["seed"]`, as `calm`
  does. ([#496])
- `pc`, `fci`, `gfci`, `cdnod`, `iamb`, `inter_iamb`, `mmpc`, `mmhc`, and `rsmax2` take `seed`,
  which they pass to the `rcot`, `cmiknn`, and `gpdc` tests, the only tests that draw random
  numbers. These searches followed `ANDREY_SEED` before, so a call that gives `seed` no longer
  depends on it. At the default seed the graphs are unchanged, and `metadata["seed"]` records the
  seed of a fit that uses one of those tests. ([#499])
- `print(out)` leaves out a metadata entry whose value is an empty list or dict, such as
  `conflicts: []`; the entry stays in `output.metadata`. ([#509])

### Fixed

- The adaptive Lasso that prunes the edges of `direct_lingam`, `ica_lingam`,
  `multi_group_direct_lingam`, `varma_lingam`, `longitudinal_lingam`, and `bottom_up_parce_lingam`
  keeps the same edges in any unit of the data. scikit-learn's LARS ended the Lasso path once its
  penalty fell to `2**-23` in the data's units, so data in small units lost edges (14 to 41
  percent of DirectLiNGAM's true edges at a standard deviation near `1e-3`, all of them near
  `1e-5`), standardized data could lose a parent of a nearly noiseless variable, and data with a
  standard deviation near `2**124` raised a LAPACK error. A fit whose path stops before the
  least-squares fit now runs again on the data divided by a power of two, and BIC picks its point
  on the whole path; a fit whose path reached the least-squares fit returns the same values as
  before. ([#478])
- The LiNGAM methods, `varma_lingam`, and `gin` fit their regressions, covariances, and
  time-series model on unit-scale columns and map the results back, so the units of the data no
  longer change their graphs; on standardized data the LiNGAM methods return the same values as
  before. ([#500])
- `varma_lingam`'s `max_iter` raises the likelihood fit's evaluation limit with it. The optimizer
  stopped after 15,000 function evaluations, which each iteration spends at about one per
  parameter, however large `max_iter` was and without a warning, so raising `max_iter` could not
  make a long fit converge. The limit is now 150 evaluations per iteration, the same 15,000 at the
  default `max_iter=100`, and a fit that ends on it without converging warns with
  `andrey.SearchLimitWarning` like one that ends on `max_iter`. Fits at the default `max_iter`
  return the same values as before. ([#505])
- `notears(robust=True)` and `dagma(robust=True)` floor each residual variance of the likelihood at
  `1e-12` of its column's variance instead of `1e-12` of the mean column variance, which could
  exceed the whole variance of a column in much smaller units than the others. On standardized
  data the fits return the same values as before. ([#517])
- Naming `indep_test="cond_gauss"` or `"degen_gauss"`, or the same scores, on a DataFrame with a
  pandas `category`, `bool`, or text column no longer raises a `DataTypeWarning` that the column
  was inferred as categorical: its type declares it. The warning stays for columns without such a
  type that hold at most `max_categories` distinct whole numbers, and names only those. ([#524])
- Every method that takes `required` keeps each required edge in its output. `granger` and
  `varma_lingam(prune=True)` left out a required lagged link whose least-squares weight came out
  exactly 0, such as a link into a series that does not vary over the fitted rows; the link is now
  in the graph with its weight 0, as `direct_lingam` and `multi_group_direct_lingam` already kept
  theirs. `longitudinal_lingam` builds its graph the same way. Fits without a zero weight on a
  required link return the same graphs as before. ([#524])

### New contributors

- [@petacube](https://github.com/petacube) made their first contribution in [#468].

## [0.10.0] - 2026-10-27

### Highlights

- How far each method goes: the 0.10.0 release card maps, for 26 methods, the problems each
  finishes within one hour and 28 GB on 4 cores with fewer errors than the empty graph, by
  variables, samples, and density, and each method's **Limitations** paragraph gives its measured
  boundary. At 2,000 samples and a mean degree of 2, BOSS stays more accurate than the empty graph
  up to 1,600 variables, GRaSP to 800, and PC to 400. ([#343], [#438], [#469])

### Changed

- Every method's **Limitations** paragraph says its measured sizes and times are a reference, not a
  hard limit: they change with the data generator, the density, the method's settings, and the
  machine. ([#367])
- The **Limitations** paragraph of every method but Granger gives its measured boundary: on each
  benchmark regime, the largest size through which its median errors stayed below the empty
  graph's, the largest size its fits finished within 1 h and 28 GB on 4 cores, what ended the
  measurement, and a median fit time. ([#469])
- For contributors, `python -m qa.quality_gate --jobs N` fits the quality gate's cells in N
  processes and prints the same results in the same order; the gate's workflow runs `--jobs 4` on
  its four vCPUs and logs the runner's CPU. A push to a release pull request that changes only
  `docs/`, `site/`, or top-level Markdown files reuses an earlier passing run instead of fitting the
  methods again. ([#431])
- For contributors, a production deploy of the site, from a `v*` tag or a manual run, waits
  for the repository owner's approval in the `production-site` environment, which
  administrators cannot bypass. ([#441])

### Removed

- `andrey.data.storage` (`save` and `load`) and the `data` extra, `andrey-core[data]`, which
  installed safetensors for them. Nothing in Andrey called them. ([#375])

### Fixed

- The worker pool starts when `TMPDIR` is on a file system that rejects the forkserver's socket,
  such as a cluster's parallel file system. `ges`, `gfci`, and the skeleton search of `pc`, `fci`,
  `gfci`, and `cdnod` with several workers raised `OSError: [Errno 95] Operation not supported`;
  the forkserver's socket now goes in `/dev/shm`, or `/tmp`, with a warning, and the workers start
  with `fork` or `spawn` when neither holds it. Results do not change. ([#415])
- `andrey.data.generate` with several workers starts them when `TMPDIR` rejects the forkserver's
  socket, as the search workers do; on Linux from Python 3.14 it raised
  `OSError: [Errno 95] Operation not supported`. ([#446])
- `categorical=[]` is accepted by every test and score. `pc`, `fci`, `gfci`, `cdnod`, `ges`, `hc`,
  `exact_search`, `boss`, and `grasp` raised a `ValueError` for it with a continuous or categorical
  test or score; an empty list now reads as `None` there. ([#395])
- Every score search takes both names of the linear-Gaussian BIC, `"local_score_BIC"` and
  `"local_score_BIC_from_cov"`, which build the same score. `ges`, `gfci`, `exact_search`, and
  `xges` rejected the second name, and `hc`, `tabu`, `boss`, and `grasp` the first; the defaults do
  not change. ([#394])
- For contributors, the benchmark harness's PCMCI, PCMCI+, and LPCMCI generators no longer fail at
  large sizes: when none of 1000 draws is stationary, as for PCMCI+ at 240 variables, they scale the
  first draw's lagged coefficients until it is, and the VARMA-LiNGAM generator does the same for an
  unstable draw. Every draw that was stationary is unchanged. ([#393])
- For contributors, the benchmark harness's longitudinal LiNGAM panels no longer diverge as the
  number of variables grows: the lag matrices of about 60% of the time points had norm 1 or more at
  3 variables and all of them from 5 on, and the panel's standard deviation reached 1e14 at 10
  variables and 1e90 at 40. A
  time point whose lag matrix through the same-step part has spectral norm 1 or more now has its
  lag-1 block scaled down until that norm is 0.95, so every step contracts. A time point already
  below 1 keeps its draw, and the panel's dataset identity changes. ([#467])
- For contributors, a release pull request into `main` fails its required `release-branch` check
  when `pyproject.toml`'s version is not the release branch's `<x.y.z>`. A post release keeps the
  package at `<x.y.z>`, and the check also fails one that changes anything under `src/` or whose
  `main` is at another version. A release's tag now submits the site to search engines once its
  production deploy check passes; a post release's tag waited for a version the site never
  shows and failed. ([#439])

## [0.9.0] - 2026-10-23

### Highlights

- The same answers in less time: PC, FCI, GFCI, CD-NOD, GES, XGES, BOSS, GRaSP, exact search,
  ICA-LiNGAM, LPCMCI, ANM, and PNL return the same output as 0.8.0, bit for bit, on every dataset of
  the identity check, with one and four workers and in five column orders. DirectLiNGAM's internal
  scores change in their last bits, with the same output on those datasets. ([#251], [#265], [#267],
  [#273], [#275], [#280], [#288], [#292], [#294], [#311], [#312], [#328], [#353], [#364], [#365],
  [#368], [#369])
- Worker processes for the constraint searches: with more than one worker, `pc`, `fci`, `gfci`, and
  `cdnod` run the larger passes of their skeleton search in worker processes and return the same
  graphs as with one, and `andrey calibrate` measures when a pass is large enough on your machine.
  ([#282], [#289], [#290])

### Added

- With more than one worker (`ANDREY_NUM_WORKERS` or `andrey.config(num_workers=...)`), `pc`,
  `fci`, `gfci`, and `cdnod` with `indep_test="fisherz"` or `"student_t"` run each skeleton pass
  over conditioning sets of two or more variables in worker processes once it holds enough tests,
  and return the same graphs and p-values as with one worker. ([#282], [#289])
- `ANDREY_SKELETON_PARALLEL_MIN_WORK` sets the fewest CI tests a conditioning-size pass of the
  `pc`, `fci`, `gfci`, and `cdnod` skeleton search needs to use the worker processes. The default,
  500000, was measured on an AMD EPYC 9534 server running Linux; a fit with more than one worker
  warns once on it when a pass reaches a tenth of it. `andrey calibrate --method skeleton`
  measures the value for your machine, and `andrey calibrate` now measures it with GES's. ([#290])
- For contributors, `qa/identity.py` hashes every supported method's full output, so a speed change
  can show that it returns the same bits at any worker count and column order, and
  `qa/profiler.py` shows where a fit spends its time. ([#262], [#269], [#277], [#284], [#286])
- `ges` takes `max_parents`, a cap on a node's parents: no graph the returned CPDAG represents
  gives a node more than that many parents, and some graph reaches each node's count. A move that
  would break the cap, in either phase, is skipped for the next best move, and
  a cap can stop GES short of the lowest-scoring graph. `None`, the default, sets no cap and keeps
  the forward gate of `n_variables / 2` and every result. ([#350], [#380])
- `GraphStructure.to_pcalg()` and `GraphStructure.from_pcalg(amat, kind)` convert to and from the
  adjacency matrix of R's pcalg, `amat.pag` for a PAG and `amat.cpdag` for a CPDAG or DAG, so
  exchanging a graph with R needs no transpose or code swap of your own. ([#351])
- For contributors, the benchmark harness compares Andrey's PC, GES, BOSS, GRaSP, and FCI with
  Tetrad 7.6.10's on continuous data (`benchmarks/tetrad_cpu.py`,
  `benchmarks/tetrad_fci_latent.py`), with the time of Tetrad's JVM reported apart from its search.
  ([#362])

### Changed

- `pc`, `fci`, `gfci`, and `cdnod` with `indep_test="fisherz"` or `"student_t"` return the same
  graphs and p-values in less time, because their skeleton search hands the test each pair's
  conditioning sets, and those of consecutive pairs, as one stack. ([#280], [#292])
- `pc`, `fci`, `gfci`, and `cdnod` return the same graphs and p-values with less memory, because
  their size-one pass builds its queries one stack at a time instead of listing every pair's
  conditioners first, which took about 4 GB of list pointers on a complete graph of 1,000
  variables. ([#346])
- `fci` and `gfci` return the same PAGs in less time: the possible-d-separation step computes one
  reach set per variable and builds its neighbor lists and reach sets once for each version of the
  graph, and GFCI reads each node's neighbors from lists built once and tests its separating sets
  in stacks, those of size 0 for every pair in one. ([#265], [#312], [#353])
- `ges` and `gfci` return the same graphs in less time, because GES keeps each reach set while a
  move cannot change it and rebuilds the CPDAG after each move faster. ([#328])
- `ges`, `gfci`, and `gies` return the same graphs in less time, because GES derives again only the
  edges of the nodes a move can reach instead of rebuilding the whole CPDAG after each move.
  ([#365])
- `xges` returns the same graphs in less time, because each branch of its extended search copies
  only the parts of the graph it changes. ([#311])
- `xges` enters numpy's floating-point error state once per run instead of once per scored trial,
  and raises `RuntimeError` instead of building a different graph if a branch of its extended
  search starts from a snapshot that no longer matches the graph; `ges` starts its reach cache from
  the empty graph's successor masks. The returned graphs are unchanged. ([#361])
- `lpcmci` returns the same graphs in less time, because it keeps each node's sorted neighbors, its
  parent, child, and neighbor sets, and its path steps for each version of the graph instead of
  building them again for every query. ([#364])
- `boss` and `grasp` return the same graphs in less time, because each grow-shrink tree keeps one
  tuple of candidate parents instead of copying the list on every trace, and the order sweeps keep
  the set of allowed parents between traces instead of building it for each one. ([#267], [#368])
- `exact_search` returns the same graphs in less time, because its linear-Gaussian BIC solves the
  normal equations of a size's parent sets in stacks of up to 4,096, and it finds each variable's
  best parent set within a candidate set from a table built once per fit instead of scanning its
  sorted parent sets. ([#273], [#369])
- `pnl` returns the same p-values and statistics in less time, because each variable's kernel is
  built once and in blocks, and its HSIC tests run on a CUDA GPU, in float64, from the cutoff
  `ANDREY_PNL_GPU_THRESHOLD`. ([#251])
- `pnl` runs fewer worker threads than asked for when their kernel matrices (three n x n float64
  matrices per thread) would hold more than 4 GiB, with a `PerformanceWarning`; `num_workers=-1`
  at 5,000 rows on 64 cores had asked for about 38 GB. With one worker the BLAS threads stay
  unlimited, so on a machine with several cores its p-values and statistics can differ in the last
  digits from those of several workers. ([#359])
- `anm` returns the same results with less memory, because it builds its kernel's distance matrix
  directly. ([#294])
- `anm` and the `kci` test without conditioning variables return the same p-values with less
  memory, because their HSIC test builds each kernel in blocks into one reused matrix, as `pnl`
  does. ([#363])
- `andrey calibrate` sets the skeleton search's cutoff one past the slowest pass it measured when
  the worker pool never won, because the search uses the pool for a pass of exactly the cutoff's
  size. ([#363])
- For contributors, `qa/identity.py --compare` fails when a count falls as well as when one rises
  (`--allow-fewer-queries` accepts a fall), counts the exact search's score evaluations, and
  refits `pnl` on workers. ([#363])
- `ica_lingam` returns the same results in less time, because its causal-order search drops each
  placed row with a mask. ([#275])
- `direct_lingam`, `multi_group_direct_lingam`, `varma_lingam`, and `longitudinal_lingam` compute
  each pair's residual entropy once per step instead of twice, so their internal scores change in
  the last bits. A near-tie between two candidates for the next variable in the causal order can
  then resolve the other way; on the identity check's datasets every output is the same as 0.8.0's.
  ([#288])
- `help(andrey)` opens with a quick start and points to `andrey --skill` and
  `andrey --help --json`, and the package summary says the import name is `andrey`, so an agent
  that has only run `pip install andrey-core` finds the command line and its skill. ([#347])

### Removed

- Names that only tests used: `andrey.core.backend.parallel_map`,
  `andrey.data.storage.save_collection` and `load_collection`,
  `andrey.temporal.LongitudinalLiNGAM.get_error_independence_p_values` and `residuals_`,
  `andrey.core.independence.hsic_gamma_test` and `measure_hsic_statistic`, and the bandwidth options
  of `hsic_test` and `hsic_statistic_test`, which no caller passed. ([#371])

## [0.8.0] - 2026-10-23

### Highlights

- Nonlinear independence tests: `pc`, `fci`, `gfci`, and `cdnod` support KCI, RCoT, CMIknn, and
  GPDC, four tests that detect nonlinear dependence that partial correlation misses. KCI and CMIknn
  also run on a CUDA GPU. The 0.8.0 post and release card give the measurements. ([#209], [#213],
  [#299])
- The same graph in any column order and on any CPU: DirectLiNGAM, ICA-LiNGAM, exact search, GES,
  GIES, XGES, CALM, GIN, ParceLiNGAM, the PCMCI methods, and the nonlinear tests break ties by
  each column's data rather than its position or rounding differences. ([#205], [#213], [#228])
- Faster categorical tests: `pc` with `indep_test="chisq"` or `"gsq"` runs 4-6x faster than in
  0.7.0 on the alarm, hepar2, and andes networks, with the same p-values and graphs. ([#208])
- Background knowledge for every search: all 13 searches accept required and forbidden edges,
  allowed adjacent pairs, and temporal tiers through `knowledge=`. Constraint-based methods also
  accept `max_depth`. ([#214])

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
- `pc`, `fci`, `gfci`, and `cdnod` take four conditional-independence tests that detect nonlinear
  dependence: `indep_test="kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"`. ([#209])
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
- `indep_test="kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"` are supported: on nonlinear data each
  makes fewer errors on average than its reference packages, causal-learn for KCI, causal-learn and
  pcalg for RCoT, and tigramite for CMIknn and GPDC, they have checks in the release quality gate,
  and they no longer warn with `ExperimentalWarning`. ([#299])

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
- `boss` and `grasp` with the linear-Gaussian BIC make every decision as full local scores make
  it: each fast update from a base parent set comes with a bound on its rounding error, and a
  comparison the bound leaves open is decided by full scores. Updates no longer build on earlier
  updates. On dense data with nearly collinear parent sets, where an update could be off by more
  than the `1e-6` tolerance, rounding error no longer decides a move, and `boss` can return a
  different graph. ([#271])
- `varma_lingam` and `longitudinal_lingam` fit on the columns sorted by a rank computed from each
  column's own data instead of in column order, so reordering the columns no longer changes their
  graphs, and on data where the order mattered their graphs and weights can change. ([#278])

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

[0.16.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.7.0...v0.8.0
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
[#251]: https://github.com/Abel-ai-lab/andrey/pull/251
[#261]: https://github.com/Abel-ai-lab/andrey/pull/261
[#262]: https://github.com/Abel-ai-lab/andrey/pull/262
[#265]: https://github.com/Abel-ai-lab/andrey/pull/265
[#267]: https://github.com/Abel-ai-lab/andrey/pull/267
[#269]: https://github.com/Abel-ai-lab/andrey/pull/269
[#271]: https://github.com/Abel-ai-lab/andrey/pull/271
[#273]: https://github.com/Abel-ai-lab/andrey/pull/273
[#275]: https://github.com/Abel-ai-lab/andrey/pull/275
[#277]: https://github.com/Abel-ai-lab/andrey/pull/277
[#278]: https://github.com/Abel-ai-lab/andrey/pull/278
[#280]: https://github.com/Abel-ai-lab/andrey/pull/280
[#282]: https://github.com/Abel-ai-lab/andrey/pull/282
[#284]: https://github.com/Abel-ai-lab/andrey/pull/284
[#286]: https://github.com/Abel-ai-lab/andrey/pull/286
[#288]: https://github.com/Abel-ai-lab/andrey/pull/288
[#289]: https://github.com/Abel-ai-lab/andrey/pull/289
[#290]: https://github.com/Abel-ai-lab/andrey/pull/290
[#292]: https://github.com/Abel-ai-lab/andrey/pull/292
[#294]: https://github.com/Abel-ai-lab/andrey/pull/294
[#299]: https://github.com/Abel-ai-lab/andrey/pull/299
[#311]: https://github.com/Abel-ai-lab/andrey/pull/311
[#312]: https://github.com/Abel-ai-lab/andrey/pull/312
[#328]: https://github.com/Abel-ai-lab/andrey/pull/328
[#343]: https://github.com/Abel-ai-lab/andrey/pull/343
[#346]: https://github.com/Abel-ai-lab/andrey/pull/346
[#347]: https://github.com/Abel-ai-lab/andrey/pull/347
[#350]: https://github.com/Abel-ai-lab/andrey/pull/350
[#351]: https://github.com/Abel-ai-lab/andrey/pull/351
[#353]: https://github.com/Abel-ai-lab/andrey/pull/353
[#359]: https://github.com/Abel-ai-lab/andrey/pull/359
[#361]: https://github.com/Abel-ai-lab/andrey/pull/361
[#362]: https://github.com/Abel-ai-lab/andrey/pull/362
[#363]: https://github.com/Abel-ai-lab/andrey/pull/363
[#364]: https://github.com/Abel-ai-lab/andrey/pull/364
[#365]: https://github.com/Abel-ai-lab/andrey/pull/365
[#367]: https://github.com/Abel-ai-lab/andrey/pull/367
[#368]: https://github.com/Abel-ai-lab/andrey/pull/368
[#369]: https://github.com/Abel-ai-lab/andrey/pull/369
[#371]: https://github.com/Abel-ai-lab/andrey/pull/371
[#375]: https://github.com/Abel-ai-lab/andrey/pull/375
[#380]: https://github.com/Abel-ai-lab/andrey/pull/380
[#391]: https://github.com/Abel-ai-lab/andrey/pull/391
[#393]: https://github.com/Abel-ai-lab/andrey/pull/393
[#394]: https://github.com/Abel-ai-lab/andrey/pull/394
[#395]: https://github.com/Abel-ai-lab/andrey/pull/395
[#396]: https://github.com/Abel-ai-lab/andrey/pull/396
[#397]: https://github.com/Abel-ai-lab/andrey/pull/397
[#400]: https://github.com/Abel-ai-lab/andrey/pull/400
[#410]: https://github.com/Abel-ai-lab/andrey/pull/410
[#411]: https://github.com/Abel-ai-lab/andrey/pull/411
[#415]: https://github.com/Abel-ai-lab/andrey/pull/415
[#418]: https://github.com/Abel-ai-lab/andrey/pull/418
[#430]: https://github.com/Abel-ai-lab/andrey/pull/430
[#431]: https://github.com/Abel-ai-lab/andrey/pull/431
[#438]: https://github.com/Abel-ai-lab/andrey/pull/438
[#439]: https://github.com/Abel-ai-lab/andrey/pull/439
[#441]: https://github.com/Abel-ai-lab/andrey/pull/441
[#442]: https://github.com/Abel-ai-lab/andrey/pull/442
[#443]: https://github.com/Abel-ai-lab/andrey/pull/443
[#445]: https://github.com/Abel-ai-lab/andrey/pull/445
[#446]: https://github.com/Abel-ai-lab/andrey/pull/446
[#451]: https://github.com/Abel-ai-lab/andrey/pull/451
[#453]: https://github.com/Abel-ai-lab/andrey/pull/453
[#455]: https://github.com/Abel-ai-lab/andrey/pull/455
[#457]: https://github.com/Abel-ai-lab/andrey/pull/457
[#459]: https://github.com/Abel-ai-lab/andrey/pull/459
[#461]: https://github.com/Abel-ai-lab/andrey/pull/461
[#463]: https://github.com/Abel-ai-lab/andrey/pull/463
[#467]: https://github.com/Abel-ai-lab/andrey/pull/467
[#468]: https://github.com/Abel-ai-lab/andrey/pull/468
[#469]: https://github.com/Abel-ai-lab/andrey/pull/469
[#478]: https://github.com/Abel-ai-lab/andrey/pull/478
[#479]: https://github.com/Abel-ai-lab/andrey/pull/479
[#481]: https://github.com/Abel-ai-lab/andrey/pull/481
[#485]: https://github.com/Abel-ai-lab/andrey/pull/485
[#487]: https://github.com/Abel-ai-lab/andrey/pull/487
[#492]: https://github.com/Abel-ai-lab/andrey/pull/492
[#495]: https://github.com/Abel-ai-lab/andrey/pull/495
[#496]: https://github.com/Abel-ai-lab/andrey/pull/496
[#499]: https://github.com/Abel-ai-lab/andrey/pull/499
[#500]: https://github.com/Abel-ai-lab/andrey/pull/500
[#501]: https://github.com/Abel-ai-lab/andrey/pull/501
[#505]: https://github.com/Abel-ai-lab/andrey/pull/505
[#509]: https://github.com/Abel-ai-lab/andrey/pull/509
[#510]: https://github.com/Abel-ai-lab/andrey/pull/510
[#515]: https://github.com/Abel-ai-lab/andrey/pull/515
[#516]: https://github.com/Abel-ai-lab/andrey/pull/516
[#517]: https://github.com/Abel-ai-lab/andrey/pull/517
[#520]: https://github.com/Abel-ai-lab/andrey/pull/520
[#521]: https://github.com/Abel-ai-lab/andrey/pull/521
[#524]: https://github.com/Abel-ai-lab/andrey/pull/524
[#526]: https://github.com/Abel-ai-lab/andrey/pull/526
[#544]: https://github.com/Abel-ai-lab/andrey/pull/544
[#550]: https://github.com/Abel-ai-lab/andrey/pull/550
[#551]: https://github.com/Abel-ai-lab/andrey/pull/551
[#552]: https://github.com/Abel-ai-lab/andrey/pull/552
[#553]: https://github.com/Abel-ai-lab/andrey/pull/553
[#554]: https://github.com/Abel-ai-lab/andrey/pull/554
[#555]: https://github.com/Abel-ai-lab/andrey/pull/555
[#556]: https://github.com/Abel-ai-lab/andrey/pull/556
[#559]: https://github.com/Abel-ai-lab/andrey/pull/559
[#564]: https://github.com/Abel-ai-lab/andrey/pull/564
[#565]: https://github.com/Abel-ai-lab/andrey/pull/565
[#567]: https://github.com/Abel-ai-lab/andrey/pull/567
[#569]: https://github.com/Abel-ai-lab/andrey/pull/569
[#571]: https://github.com/Abel-ai-lab/andrey/pull/571
[#572]: https://github.com/Abel-ai-lab/andrey/pull/572
[#573]: https://github.com/Abel-ai-lab/andrey/pull/573
[#574]: https://github.com/Abel-ai-lab/andrey/pull/574
[#575]: https://github.com/Abel-ai-lab/andrey/pull/575
[#577]: https://github.com/Abel-ai-lab/andrey/pull/577
[#581]: https://github.com/Abel-ai-lab/andrey/pull/581
[#584]: https://github.com/Abel-ai-lab/andrey/pull/584
[#596]: https://github.com/Abel-ai-lab/andrey/pull/596
[#597]: https://github.com/Abel-ai-lab/andrey/pull/597
[#607]: https://github.com/Abel-ai-lab/andrey/pull/607
[#611]: https://github.com/Abel-ai-lab/andrey/pull/611
[#612]: https://github.com/Abel-ai-lab/andrey/pull/612
[#619]: https://github.com/Abel-ai-lab/andrey/pull/619
[#620]: https://github.com/Abel-ai-lab/andrey/pull/620
[#623]: https://github.com/Abel-ai-lab/andrey/pull/623
[#625]: https://github.com/Abel-ai-lab/andrey/pull/625
[#626]: https://github.com/Abel-ai-lab/andrey/pull/626
[#629]: https://github.com/Abel-ai-lab/andrey/pull/629
[#630]: https://github.com/Abel-ai-lab/andrey/pull/630
[#631]: https://github.com/Abel-ai-lab/andrey/pull/631
[#633]: https://github.com/Abel-ai-lab/andrey/pull/633
[#634]: https://github.com/Abel-ai-lab/andrey/pull/634
[#639]: https://github.com/Abel-ai-lab/andrey/pull/639
[#659]: https://github.com/Abel-ai-lab/andrey/pull/659
[#660]: https://github.com/Abel-ai-lab/andrey/pull/660
[#662]: https://github.com/Abel-ai-lab/andrey/pull/662
[#663]: https://github.com/Abel-ai-lab/andrey/pull/663
[#665]: https://github.com/Abel-ai-lab/andrey/pull/665
[#668]: https://github.com/Abel-ai-lab/andrey/pull/668
[#674]: https://github.com/Abel-ai-lab/andrey/pull/674
[#675]: https://github.com/Abel-ai-lab/andrey/pull/675
[#676]: https://github.com/Abel-ai-lab/andrey/pull/676
[#677]: https://github.com/Abel-ai-lab/andrey/pull/677
[#679]: https://github.com/Abel-ai-lab/andrey/pull/679
[#680]: https://github.com/Abel-ai-lab/andrey/pull/680
[#682]: https://github.com/Abel-ai-lab/andrey/pull/682
[#687]: https://github.com/Abel-ai-lab/andrey/pull/687
[#690]: https://github.com/Abel-ai-lab/andrey/pull/690
[#692]: https://github.com/Abel-ai-lab/andrey/pull/692
[#694]: https://github.com/Abel-ai-lab/andrey/pull/694
[#702]: https://github.com/Abel-ai-lab/andrey/pull/702
[#707]: https://github.com/Abel-ai-lab/andrey/pull/707
[#711]: https://github.com/Abel-ai-lab/andrey/pull/711
[#719]: https://github.com/Abel-ai-lab/andrey/pull/719
[#720]: https://github.com/Abel-ai-lab/andrey/pull/720
[#723]: https://github.com/Abel-ai-lab/andrey/pull/723
[#732]: https://github.com/Abel-ai-lab/andrey/pull/732
[#737]: https://github.com/Abel-ai-lab/andrey/pull/737
[#738]: https://github.com/Abel-ai-lab/andrey/pull/738
[#739]: https://github.com/Abel-ai-lab/andrey/pull/739
[#749]: https://github.com/Abel-ai-lab/andrey/pull/749
[#752]: https://github.com/Abel-ai-lab/andrey/pull/752
[#755]: https://github.com/Abel-ai-lab/andrey/pull/755
[#759]: https://github.com/Abel-ai-lab/andrey/pull/759
[#761]: https://github.com/Abel-ai-lab/andrey/pull/761
[#764]: https://github.com/Abel-ai-lab/andrey/pull/764
[#765]: https://github.com/Abel-ai-lab/andrey/pull/765
[#766]: https://github.com/Abel-ai-lab/andrey/pull/766
[#768]: https://github.com/Abel-ai-lab/andrey/pull/768
[#771]: https://github.com/Abel-ai-lab/andrey/pull/771
[#774]: https://github.com/Abel-ai-lab/andrey/pull/774
[#775]: https://github.com/Abel-ai-lab/andrey/pull/775
[#781]: https://github.com/Abel-ai-lab/andrey/pull/781
[#791]: https://github.com/Abel-ai-lab/andrey/pull/791
[#796]: https://github.com/Abel-ai-lab/andrey/pull/796
[#797]: https://github.com/Abel-ai-lab/andrey/pull/797
[#798]: https://github.com/Abel-ai-lab/andrey/pull/798
[#803]: https://github.com/Abel-ai-lab/andrey/pull/803
[#806]: https://github.com/Abel-ai-lab/andrey/pull/806
[#807]: https://github.com/Abel-ai-lab/andrey/pull/807
[#808]: https://github.com/Abel-ai-lab/andrey/pull/808
[#809]: https://github.com/Abel-ai-lab/andrey/pull/809
[#821]: https://github.com/Abel-ai-lab/andrey/pull/821
[#826]: https://github.com/Abel-ai-lab/andrey/pull/826
[#828]: https://github.com/Abel-ai-lab/andrey/pull/828
[#829]: https://github.com/Abel-ai-lab/andrey/pull/829
[#830]: https://github.com/Abel-ai-lab/andrey/pull/830
[#832]: https://github.com/Abel-ai-lab/andrey/pull/832
[#835]: https://github.com/Abel-ai-lab/andrey/pull/835
[#841]: https://github.com/Abel-ai-lab/andrey/pull/841
[#842]: https://github.com/Abel-ai-lab/andrey/pull/842
[#843]: https://github.com/Abel-ai-lab/andrey/pull/843
[#848]: https://github.com/Abel-ai-lab/andrey/pull/848
[#850]: https://github.com/Abel-ai-lab/andrey/pull/850
[#853]: https://github.com/Abel-ai-lab/andrey/pull/853
[#854]: https://github.com/Abel-ai-lab/andrey/pull/854
[#856]: https://github.com/Abel-ai-lab/andrey/pull/856
[#857]: https://github.com/Abel-ai-lab/andrey/pull/857
[#870]: https://github.com/Abel-ai-lab/andrey/pull/870
[#871]: https://github.com/Abel-ai-lab/andrey/pull/871
[#872]: https://github.com/Abel-ai-lab/andrey/pull/872
[#873]: https://github.com/Abel-ai-lab/andrey/pull/873
[#874]: https://github.com/Abel-ai-lab/andrey/pull/874
[#875]: https://github.com/Abel-ai-lab/andrey/pull/875
[#879]: https://github.com/Abel-ai-lab/andrey/pull/879
[#880]: https://github.com/Abel-ai-lab/andrey/pull/880
[#882]: https://github.com/Abel-ai-lab/andrey/pull/882
[#887]: https://github.com/Abel-ai-lab/andrey/pull/887
[#891]: https://github.com/Abel-ai-lab/andrey/pull/891
[#892]: https://github.com/Abel-ai-lab/andrey/pull/892
[#893]: https://github.com/Abel-ai-lab/andrey/pull/893
[#894]: https://github.com/Abel-ai-lab/andrey/pull/894
[em-tetrad]: https://github.com/Abel-ai-lab/andrey/blob/main/benchmarks/comparisons/missing-data.md
[kp-paths]: https://github.com/Abel-ai-lab/andrey/blob/main/benchmarks/comparisons/knowledge-paths.md
[nd-page]: https://github.com/Abel-ai-lab/andrey/blob/main/benchmarks/comparisons/notears-dagma.md
