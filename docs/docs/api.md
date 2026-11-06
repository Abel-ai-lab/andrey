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

Each method takes a data matrix and returns a {py:class}`~andrey.StructureOutput`, with these
exceptions: `cdnod` and `fci_jci` also take `env`, each row's environment, `gies`, `igsp`, and
`ut_igsp` also take `env` with `targets`, the variables intervened on in each environment, `gnies`
takes `env` and learns the targets, `icp` also takes `env` and `target`, the variable whose parents
it finds, `longitudinal_lingam` takes a list of datasets, and `multi_group_direct_lingam` takes a
list of datasets and returns one result per dataset. The first argument of every method is `data`;
`andrey run` reads it from `--data`, which a method that takes a list of datasets needs once per
dataset. Each method is either supported or experimental; `andrey list` prints the status.

`env` holds one label per row, text or numbers, or names a column of a DataFrame `data`, which is
then not a variable; `andrey run` reads it from `--env`, a column of `--data` or a file. `gies`,
`igsp`, and `ut_igsp` take `targets` as a mapping from each label of `env` to a list of its
variables, by name or position, with `[]` for an observational environment; as the name of a column
whose cells give each row's variables, from whose distinct values `env` follows when you leave it
out; or as a 0/1 matrix shaped like `data`. A key that is not a label of `env` raises a
`ValueError`. `gies` and `igsp` need every environment's targets, so `env` without `targets`, and an
environment the mapping leaves out or gives `None`, raise one too; `ut_igsp` reads an environment
the mapping leaves out or gives `None` as one whose targets it learns, and still needs an
observational one. `gnies` takes the same forms of `targets`, but for the targets it knows: an
environment the mapping leaves out or gives `None` has unknown targets, which it learns, and
`targets=None`, the default, learns them all. When the labels of `env` are numbers, a text key
names the label it spells, so `"0"` names the label `0` and a mapping read back from JSON passes as
it is. `icp` takes `targets` in the same forms but reads only whether an environment intervened on
its `target`: it leaves such an environment's rows out, and keeps an environment whose targets are
unknown. `andrey run` takes a mapping as `targets=`, a JSON object inline or in a file, or as
`targets` in `--config`, and reads an `--env` of whole numbers as integers.

```python
out = andrey.gies(data, env, targets={"observed": [], "knockout": ["x"]})
out = andrey.gies(data, targets="target")  # a column naming each row's targets, "" if none
out = andrey.cdnod(data, "site")  # a column of labels; numbers such as a time step stay numbers
out.metadata["changing_variables"]  # the variables whose mechanism changes across the sites
out = andrey.gnies(data, "site")  # targets unknown: learned, in metadata["changing_variables"]
out = andrey.icp(data, env, target="y")  # the parents of y; its p-values in out.metadata
```

In Python, every method that takes `env` also takes each environment's rows apart as `data`, in
place of one table and `env`: a list of DataFrames or 2-D arrays, each labeled by its position
`0, 1, ...`, or a dict from each label to its table. `env` is then left out, and `targets` is a
mapping from those labels, or `None`. DataFrames need the same columns, in any order, which take
the first one's order, and a column of the `category` type in each keeps it, with every
DataFrame's categories; arrays need the same number of columns. The rows are stacked in the order
of the list or dict, so the fit is the one on the stacked table with `env` naming each row's
label.

```python
out = andrey.gies([observed, knockout], targets={0: [], 1: ["x"]})
out = andrey.ut_igsp({"control": control, "drug": drug}, targets={"control": [], "drug": None})
```

Every method checks its arguments first: a value of the wrong type raises `TypeError`, and a value
outside its range or not one of its choices raises `ValueError`, each with a message that names the
method, the argument, what it accepts, and the value, such as
`pc: alpha must be a number in (0, 1), got 1.5`. `andrey --help --json` lists each
argument's type, range, and choices. Where `indep_test` takes the shipped tests, it also takes the
ones you register with `register_indep_test`.

Every method checks its data before the fit: a numeric `(n_samples, n_variables)` matrix with at
least 2 rows, no infinity, no constant column, and, for a DataFrame, column names that differ as
text. A `NaN` is a missing value, which only the methods that delete rows per test accept (see
below). Each column name becomes its node's label as text, `str(name)`, so columns named
`1` and `"1"` would share the label `"1"`, and the `ValueError` names it with both positions.
Where an argument names columns, as `Knowledge` and `categorical` do, a string is a label and an
integer is a 0-based position: the column named `1` is `"1"`, and `1` is the second column.
A constant column holds no information about the other variables, so one `ValueError` names every
constant column and asks you to drop them; resampling or subsetting rows can make a column
constant. `direct_lingam`, `ica_lingam`, and `multi_group_direct_lingam` also need at least
`n_variables + 1` rows in each dataset, the fewest with which their pruning step can estimate the
edge weights. That pruning step, an adaptive Lasso in `direct_lingam`, `ica_lingam`,
`multi_group_direct_lingam`, `varma_lingam`, `longitudinal_lingam`, and `bottom_up_parce_lingam`,
picks the edges by BIC over the whole Lasso path, so for a given causal order it keeps the same
edges in any unit of the data. A step that would read the units only through a numerical shortcut,
such as a least-squares rank cutoff, the linear combination that `gin` tests, or the VARMA fit of
`varma_lingam`, runs on columns brought to unit scale, with its results mapped back to the data's
units. The causal order of `ica_lingam` still reads the magnitudes of its unmixing matrix, and the
least-squares scores of `notears`, `dagma`, and `dynotears` read the units, as their pages describe.
The categorical settings, `indep_test="chisq"` or `"gsq"` and `score_func="bdeu"` or
`"discrete_bic"`, read text, boolean, and `category` columns as categories and need whole numbers in
a column of numbers, each distinct value of a column one category. Every method that takes
`indep_test` or `score_func` defaults it to `None`, which picks from the column types you declare,
with `categorical` or a DataFrame's `category`, `bool`, or text columns: `"gsq"` and `"bdeu"` when
every column is categorical, `"cond_gauss"` when some are, and otherwise `"fisherz"` and the
linear-Gaussian BIC, `"bic"`; the result's `metadata["indep_test"]` and `metadata["score_func"]`
name the test and the score the fit used. Values never decide: a column of a few whole numbers reads
as numbers, with a `DataTypeWarning` that it may hold category codes, from every method that reads
columns as numbers, as [Data types](guides/getting-started.md#data-types) describes. The mixed-data
settings, `indep_test="degen_gauss"` or `"cond_gauss"` and `score_func="degen_gauss"` or
`"cond_gauss"`, read the categorical columns that `categorical` lists, or a DataFrame's `category`,
`bool`, and text columns, and never a column by its values; a categorical column of numbers needs
whole numbers. A test or score for continuous data, such as `"fisherz"` or `"bic"`, raises a
`ValueError` on a column declared categorical, by `categorical` or by its type. Methods that score
with BIC also reject a column that repeats another up to scale and shift or, given more rows than
columns, is a linear function of other columns. Copies of one variable cannot be told apart, so one
`ValueError` names every group of copies, such as `x2` and `x7` repeating `x1`, and asks you to keep
one column per group and read its edges as the group's; no option drops or merges the copies for
you. Every method that takes `indep_test`, with a partial-correlation test, `"fisherz"` or
`"student_t"`, rejects copies the same way before any test runs, and so do `icp` and `dynotears`.
`pcmci`, `pcmciplus`, and `lpcmci` with those tests also reject a series that repeats another up to
`2 * tau_max` steps back and name its lagged value, such as `'a' at lag 1 repeats 'd'`. `cam`,
`resit`, and `camuv` reject copies the same way, and a column of fewer than 3 distinct values, such
as a binary one, on which no smooth term can be fit; `resit` and `camuv` also need at least twice as
many rows as columns, and `camuv` rejects a column that is a linear function of other columns, such
as a total, as the BIC methods do. Other data raises a `ValueError` that names the argument and, for
a bad cell, its row and column. A single column gives a one-node graph.

% concept: missing
`pc`, `fci`, `cdnod`, `fci_jci`, `iamb`, `inter_iamb`, `mmpc`, `pcmci`, `pcmciplus`, and `lpcmci`
accept missing values, a `NaN` cell, pandas' `NA`, or an empty, `NA`, or `N/A` field of a CSV file:
each conditional-independence test reads only the rows where all of its variables are observed
(test-wise deletion), and so do the pair tests `anm` and `pnl`, which read the rows where both
columns are observed; `metadata["rows_tested"]` gives the fewest (`min`) and the median rows a test
read. A column with no observed value, or a constant one over its observed values, raises a
`ValueError` before the fit, and a test with fewer rows than it needs, `|S| + 3` for `"fisherz"` and
`"student_t"` and 2 for the other tests, raises one during it. `cdnod` and `fci_jci` need a complete
`env`. Every other method raises a `ValueError` that names it and says what to do instead, as
[Missing values](guides/getting-started.md#missing-values) describes; each method's Notes say why.

% concept: indep_test
`pc`, `fci`, `gfci`, `boss_fci`, `grasp_fci`, `cdnod`, `fci_jci`, `iamb`, `inter_iamb`, `mmpc`,
`mmhc`, `rsmax2`, `igsp`, `ut_igsp`, `pcmci`, `pcmciplus`, and `lpcmci` take the same `indep_test`
choices: `"fisherz"`, `"student_t"`, and `"normal_scores"` for continuous data, `"chisq"` and
`"gsq"` for categorical data, `"degen_gauss"` and `"cond_gauss"` for mixed data, and the nonlinear
tests `"kci"`, `"rcot"`, `"cmiknn"`, and `"gpdc"` ([Conditional-independence
tests](#conditional-independence-tests)). `None`, the default, picks from the declared column types,
`"fisherz"` for continuous data. `pcmci`, `pcmciplus`, and `lpcmci` run each test on the lagged
values of the series, and order the candidate conditions by the test's statistic, as tigramite does:
the partial correlation, the chi-square statistic, the likelihood ratio, CMIknn's conditional mutual
information estimate, or GPDC's distance correlation. A test you register that reports only p-values
is ordered by its p-value, smallest first.

% concept: knowledge
`pc`, `fci`, `gfci`, `boss_fci`, `grasp_fci`, `cdnod`, `fci_jci`, `iamb`, `inter_iamb`, `mmpc`,
`ges`, `gies`, `gnies`, `xges`, `hc`, `tabu`, `mmhc`, `rsmax2`, `boss`, `grasp`, `igsp`, `ut_igsp`,
`exact_search`, `calm`, `notears`, `dagma`, `cam`, `resit`, `direct_lingam`, `ica_lingam`,
`multi_group_direct_lingam`, `varma_lingam`, `longitudinal_lingam`, `pcmci`, `pcmciplus`, `lpcmci`,
`granger`, `dynotears`, `rcd`, `bottom_up_parce_lingam`, and `camuv` take `knowledge`, a
{py:class}`~andrey.Knowledge` of required and forbidden edges, the pairs allowed to be adjacent, and
temporal tiers; each method's `knowledge` parameter says how it honors them. Every required edge is
in the result, whatever its weight, test, or score says. For the time-series methods, an edge
`(cause, effect, lag)` is the link from `cause` at time `t - lag` to `effect` at time `t`.
[Background knowledge](guides/getting-started.md#background-knowledge) shows them in use.

% concept: tau_max
`pcmci`, `pcmciplus`, `lpcmci`, `granger`, `longitudinal_lingam`, and `dynotears` take `tau_max`,
the largest lag, at least 1, with a default of 1; for `longitudinal_lingam` a lag counts time points
of the panel. The ARMA order of VARMA-LiNGAM, `order=(p, q)`, is a different setting and keeps its
name.

% concept: measure
`direct_lingam`, `multi_group_direct_lingam`, `varma_lingam`, and `longitudinal_lingam` run
DirectLiNGAM's order search and take `measure`, how it picks the most exogenous variable at each
step, by an approximation of the pairwise likelihood ratios of Hyvarinen and Smith (2013).
`"pwling"`, the default, uses the entropies of each variable and of each regression residual.
`"tanh"` uses a correlation of the standardized pair through `tanh`, which suits sparse noise such
as Laplace, and `"skew"` one through the square after a sign correction, which suits skewed noise;
they cost less, and `direct_lingam`'s Notes and Limitations give where each fails.

% concept: confounded_index
`fci_jci` takes `confounded_index`, whether a hidden variable may cause both the environment index
and a data variable. `True`, the default (Mooij, Magliacane, and Claassen 2020's FCI-JCI1), leaves
the index end of an edge a circle until a rule orients it. `False` assumes the environment is set
from outside, as in a randomized experiment (FCI-JCI123), so every edge the index keeps points out
of it.

% concept: fdr_method
`pcmci`, `pcmciplus`, and `granger` take `fdr_method`, tigramite's name for the correction of the
lagged links' p-values for the number of links tested. `"fdr_bh"`, the default, replaces them with
their Benjamini-Hochberg adjusted values over all the lagged links together, which holds the false
discovery rate at or below the level when the tests are independent or positively dependent.
`"none"` keeps each p-value as tested, so about `alpha` of the absent links appear as links. `pcmci`
and `granger` keep a link when its adjusted p-value is at most `alpha`; `pcmciplus` decides its
links in the skeleton phase and reports the adjusted values in `metadata["p_values"]` without
changing them. LPCMCI takes none, as tigramite's LPCMCI does not.

% concept: max_depth
`pc`, `fci`, `gfci`, `boss_fci`, `grasp_fci`, `cdnod`, `fci_jci`, `iamb`, `inter_iamb`, `mmpc`,
`mmhc`, `rsmax2`, `pcmci`, `pcmciplus`, `rcd`, and `camuv` take `max_depth`, the maximum
conditioning-set size of their conditional-independence tests; `None`, the default, sets no cap.
`mmhc` and `rsmax2` cap their restrict phase and `pcmci` and `pcmciplus` their PC1 step; the
blankets of `iamb` and `inter_iamb` stop growing at the cap. `rcd`'s cap is the most variables it
regresses out together, and `camuv`'s the most variables a sink test adds together as a variable's
candidate parents, each with a default of 2, its paper's setting.

% concept: max_p_global
`lpcmci` and `pcmciplus` take `max_p_global`, tigramite's name for the most conditions a test takes
beyond the default conditions, the parents of both ends, which every test includes. `lpcmci` caps
all of its removal phases with it. `pcmciplus` splits its caps by phase: `max_depth` caps the
lagged phase, whose sets hold no default conditions, and `max_p_global` caps the same-step
skeleton and collider phases, whose sets add the lagged parents of both ends.

% concept: structure
`iamb`, `inter_iamb`, and `mmpc` take `structure`, the graph they return without a `target`.
`"cpdag"` is the CPDAG of the Grow-Shrink structure learner (Margaritis 2003): the skeleton, with
each unshielded triple tested for a collider and Meek's rules applied. `"skeleton"` is the same
adjacencies, undirected. `iamb` and `inter_iamb` default to `"cpdag"` and also take `"moral"`, the
moral graph, each variable joined to its Markov blanket, a child's other parents included. `mmpc`
defaults to `"skeleton"`, the graph its sets give; its parents-and-children sets hold no child's
other parents, so it has no moral graph. These are the defaults of the corresponding learners of
the R package bnlearn. With a `target`, each returns that variable's set, whatever `structure` is.

% concept: collider_rule
`pc`, `cdnod`, `fci`, `gfci`, `boss_fci`, `grasp_fci`, and `fci_jci` take `collider_rule`, which
decides whether the middle node of an unshielded triple `x - z - y` is a collider. `"sepsets"`, the
default, reads the separating set that removed the edge `x - y`. `"conservative"` (Ramsey, Zhang,
and Spirtes 2006) and `"majority"` (Colombo and Maathuis 2014) test `x` and `y` again given every
subset of either one's neighbors and make `z` a collider when none, or fewer than half, of the
separating subsets contain it. A triple the vote cannot decide is ambiguous: no rule orients through
it, and `metadata["ambiguous_triples"]` lists it. Each method's Notes give every outcome.
`pcmciplus` takes the same three rules for its same-step colliders, with `"majority"` as its
default, as in tigramite, since its orientations do not depend on the order of the variables.

% concept: max_parents
`ges`, `gies`, `gnies`, `xges`, `hc`, `tabu`, `exact_search`, `boss`, `grasp`, `mmhc`, `rsmax2`,
`gfci`, `boss_fci`, `grasp_fci`, `cam`, and `icp` take `max_parents`, a cap on every node's parents:
no DAG the returned graph represents gives a node more than that many parents. `None`, the default,
sets no cap. `ges`, `gies`, `gnies`, `xges`, `hc`, `tabu`, and the score phase of `mmhc` and
`rsmax2` take no move past the cap, `boss` and `grasp` grow no parent set past it, `exact_search`
scores only the parent sets within it, and `cam`'s order search adds no edge past it. `icp` tests
only the sets of at most that many variables, and its guarantee then needs the target to have no
more parents. `gfci`, `boss_fci`, and `grasp_fci` cap their GES, BOSS, or GRaSP phase; the PAG their
tests then orient can put more arrowheads at a node.

% concept: seed
`pc`, `fci`, `gfci`, `boss_fci`, `grasp_fci`, `cdnod`, `fci_jci`, `iamb`, `inter_iamb`, `mmpc`,
`mmhc`, `rsmax2`, `pcmci`, `pcmciplus`, `lpcmci`, `boss`, `grasp`, `igsp`, `ut_igsp`, `hc`, `tabu`,
`calm`, and `ica_lingam` take `seed`, a whole number from 0 to 2**32 - 1 that fixes their random
draws. `None`, the default, is the configured seed, `ANDREY_SEED` or 0 when it is unset, not a fresh
random one, so a call repeats exactly. Apart from `boss_fci` and `grasp_fci`, whose BOSS or GRaSP
phase visits the variables in a random order on every fit, the methods that test independence draw
random numbers only under `indep_test="rcot"`, `"cmiknn"`, or `"gpdc"`, `igsp` and `ut_igsp` also
with `n_starts` above 1, and `hc` and `tabu` only with `n_starts` above 1; otherwise `seed` changes
nothing, and `metadata["seed"]` is not set. A call that gives `seed` does not depend on
`ANDREY_SEED`. Every other method is deterministic and takes no seed.

% concept: n_starts
`boss`, `grasp`, `boss_fci`, `grasp_fci`, `igsp`, `ut_igsp`, `hc`, and `tabu` take `n_starts`, the
number of searches a fit runs: the first from the start the method describes, each later one from a
start drawn from `seed` and the search's index. `boss`, `grasp`, `boss_fci`, and `grasp_fci` start
it from a random order, `igsp` and `ut_igsp` from a minimum-degree order with its ties drawn from
`seed`, and `hc` and `tabu` from the best DAG so far after `n_perturbations` random edge changes.
The fit returns the best-scoring search; a later one replaces an earlier one only when it scores
better by more than `1e-6` (for `igsp`, fewer edges, then fewer I-contradicting edges; for
`ut_igsp`, a lower count of edges and targets), and `boss_fci` and `grasp_fci` run their tests on
the best search's CPDAG. `igsp` and `ut_igsp` run 10 by default. On one set of simulated datasets in
`igsp`'s papers' settings, at `alpha=1e-3`, 10 searches and 5 had a lower SHD than one in 21 and 19
of 87 groups and were worse in none, and tied with each other in every group; for `ut_igsp`, on 15
groups of simulated datasets, 12 of them in its paper's setting, 10 searches had fewer SHD errors
than 5 in 4 groups, 2 of them in that setting, and more in none. Each search builds its own caches
and drops the previous one's, so more starts take more time but little more memory. `1`, the default
for the others, runs one search.

% concept: n_perturbations
`hc` and `tabu` take `n_perturbations`, the number of random edge additions, removals, or
reversals made to the best DAG so far before each search after the first. Each change keeps the
knowledge, `max_parents`, the rule of the mixed-data scores, and acyclicity. `None`, the default,
uses the number of variables; with one start it changes nothing. With `n_starts` above 1,
`metadata["n_perturbations"]` records the number used.

% concept: n_jobs
No method takes the number of workers or the device as an argument. Set them for every call in a
block with `andrey.config(n_jobs=...)`, or with the `ANDREY_*` variables of
[Configuration](configuration.md); `bootstrap` follows them too.

% concept: penalty_discount
`ges`, `gies`, `gnies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`, `gfci`, `boss_fci`,
`grasp_fci`, `mmhc`, and `rsmax2` take `penalty_discount`, the multiplier on the BIC penalty per
parameter, the term of Ramsey et al. (2017): larger values give sparser graphs, and `1.0`, the
default, with `ebic_gamma=0.0` is the standard BIC. Every score but `"bdeu"` has this penalty, and
`metadata["penalty_discount"]` records the value the score used.

% concept: ebic_gamma
`ges`, `gies`, `gnies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`, `gfci`, `boss_fci`,
`grasp_fci`, `mmhc`, and `rsmax2` take `ebic_gamma`, the weight of the extended BIC, from 0 to 1
(Chen and Chen 2008): their linear-Gaussian BIC adds `2 * ebic_gamma * log C(n_variables - 1,
n_parents)` to each variable's score, a penalty that grows with the number of variables. `gfci`,
`boss_fci`, and `grasp_fci` default to `ebic_gamma=0.0`, the standard BIC, and the others to `1.0`;
{py:func}`~andrey.ges` describes the choice and {py:func}`~andrey.gfci` says why its default
differs. Only `score_func="bic"` adds the term, so passing `ebic_gamma` with another score, named or
picked by `None`, raises a `ValueError`; `metadata["ebic_gamma"]` records the value a `"bic"` fit
used. `gnies`, which has one score, always adds it.

% concept: equivalent_sample_size
`ges`, `gies`, `xges`, `hc`, `tabu`, `boss`, `grasp`, `exact_search`, `gfci`, `boss_fci`,
`grasp_fci`, `mmhc`, and `rsmax2` take `equivalent_sample_size`, the equivalent sample size of the
BDeu score (Heckerman, Geiger and Chickering 1995; Buntine 1991): the number of rows its uniform
prior over each variable's conditional distributions is worth, a positive number. `1.0`, the
default, is what the score used before the argument existed. A larger value weighs the prior more
against the data, and the learned graph can change with it (Silander, Kontkanen and Myllymaki 2007),
so compare fits at one value. Only `score_func="bdeu"` reads it, so passing `equivalent_sample_size`
with another score, named or picked by `None`, raises a `ValueError`;
`metadata["equivalent_sample_size"]` records the value a `"bdeu"` fit used.

```{eval-rst}
.. currentmodule:: andrey

.. autosummary::
   :toctree: api

   Knowledge
```

### Supported methods

The supported methods are benchmarked compared with other packages, and a quality gate checks
them against known graphs before each release. Supported means measured, not accurate on every
workload: the limits on every [release card](benchmarks.md), and each method's **Limitations.**
paragraph, show where a method scores worse than the empty graph or about as well. On the cards
before 0.11.0, which score with the standard BIC (`ebic_gamma=0.0`), hill climbing and tabu search
do so from 50 variables with ten rows per variable, and GES at 200 variables with ten rows per
variable.

```{eval-rst}
.. currentmodule:: andrey

.. autosummary::
   :toctree: api
   :nosignatures:

   pc
   fci
   gfci
   cdnod
   iamb
   inter_iamb
   mmpc
   icp
   ges
   gies
   gnies
   xges
   hc
   tabu
   exact_search
   calm
   mmhc
   rsmax2
   boss
   grasp
   igsp
   ut_igsp
   direct_lingam
   ica_lingam
   multi_group_direct_lingam
   cam
   resit
   camuv
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
   dynotears
```

CALM is a continuous-optimization method in the line of NOTEARS ({py:func}`~andrey.notears`,
{py:func}`~andrey.dagma`): it minimizes a Gaussian likelihood with L0 gates under a smooth
acyclicity constraint, within an estimated moral graph.

CAM, the causal additive model (Buhlmann, Peters, and Ernest 2014), fits each variable as a sum of
smooth functions of its parents plus noise, with the paper's settings: boosting preselects each
variable's possible parents, a greedy search adds the edge whose additive spline fit most lowers a
variable's log residual variance, and smooth-term tests at `alpha=0.001` prune the edges. It
returns a DAG. The [0.13.0](benchmarks/0.13.0.md#cam) release card measures it from 10 to 100
variables against CAM 1.0.

RESIT, regression with subsequent independence test (Peters, Mooij, Janzing, and Scholkopf 2014),
fits the same kind of model with the noise of any distribution, with the paper's settings: it finds
a causal order from the sink up, each sink the variable whose additive-model residual an HSIC test
finds most independent of the remaining variables, then removes each parent whose removal leaves
the residual independent of the variable's predecessors at `alpha=0.05`. It returns a DAG and a
causal order. The [0.13.0](benchmarks/0.13.0.md#resit) release card measures it from 5 to 20
variables against the authors' code and lingam; its accuracy claim holds up to 10 variables.

IAMB and Inter-IAMB learn a variable's Markov blanket, and MMPC its parents and children, given as
`target`; without one, they join a pair when each is in the other's set and return the graph
`structure` names: the CPDAG of the Grow-Shrink structure learner, the undirected skeleton, or, for
IAMB and Inter-IAMB, the moral graph. MMHC learns a skeleton with MMPC's
conditional-independence tests, then runs hill climbing that joins only the skeleton's pairs.
RSMAX2 generalizes it: its restrict phase keeps the pairs of the skeleton of MMPC, IAMB, or
Inter-IAMB, or of PC's skeleton, and its maximize phase runs hill climbing or tabu search among
them.

FCI, GFCI, BOSS-FCI, GRaSP-FCI, and FCI-JCI use uncovered paths for their R5, R9, and R10 tail
orientations: nodes two steps apart on each path must be non-adjacent. R5 considers every circle
path, up to a fixed search bound. R8 checks each pair of endpoints in both orders, and the rules run
in rounds, each setting together the marks every rule implies on the graph as it stands and leaving
a circle where two of them disagree, so neither depends on the node labels. The same rules determine
the oracle PAG returned by `andrey.data.latent.marginal(..., "pag")`. Their possible-d-separation
step visits the pairs and nodes by a rank computed from the data, so with
`possible_dsep="order_free"`, the default, their result does not depend on the column order unless
two columns are identical; `"fast"` visits them in column order and is the default under
`andrey.config(tie_break="position")`.

GnIES (Gamella et al. 2022) learns from linear-Gaussian data pooled from two or more environments
that share one set of weights and differ in the noise variances of some variables, its targets,
without being told which. It returns the interventional essential graph of the targets it learns,
which directs every edge at a target, and lists the targets in `metadata["changing_variables"]`: the
variables whose noise variance differs between some two environments. With `env`,
`metadata["environments"]` holds its labels, and
`targets=dict.fromkeys(out.metadata["environments"], out.metadata["changing_variables"])` refits the
graph under the learned targets. For each set of targets it tries, it runs the search of `gies` from
the empty graph, with the paper's likelihood: one noise variance per environment for a target and
one in all for the others. Known targets in `targets` stay in the set; with every environment's
targets known, one search runs. The [0.14.0](benchmarks/0.14.0.md#gnies) release card measures it
against gnies, the authors' package.

IGSP (Wang et al. 2017; Yang, Katcoff, and Uhler 2018) learns from an observational environment and
interventional settings whose intervention targets are all known, as GIES does, where a setting is
the rows of one environment of `env`: it searches variable orders for the DAG with the fewest edges,
then the fewest I-contradicting edges, the edges whose direction the settings' invariance tests
contradict. Its default level for both tests, `alpha=None`, is one over the number of observational
rows, as for UT-IGSP, `0.001` at 1,000 rows. Of the levels its authors swept, from `1e-4` to `0.05`,
`1e-3` had the fewest errors on one set of datasets simulated as in its papers; on fresh datasets,
against the authors' reference implementation at `1e-3`, it and one over the observational rows were
each worse in none of 114 comparisons and had fewer errors in 17, with interval ends that are
floating-point residue of 0 read as 0. It returns the interventional essential graph and the level
used in `metadata["alpha"]`. The [0.14.0](benchmarks/0.14.0.md#igsp) release card measures it
against graphical-model-learning, the authors' library.

UT-IGSP (Squires, Wang, and Uhler 2020) learns from an observational environment and settings whose
intervention targets are known in part or not at all: it searches variable orders for the DAG with
the fewest edges plus targets, where a setting's targets are its known ones and every variable whose
distribution given its parents differs from the observational one. Its default level for the
conditional-independence and invariance tests, `alpha=None`, is one over the number of observational
rows, `0.001` at 1,000 rows: on simulated data in the paper's setting, against the paper's reference
implementation at the paper's `1e-5`, it was worse in none of 112 comparisons and better in 1 on
fresh datasets, and lower in 17 more on the datasets that chose it. It returns the interventional
essential graph with each setting's targets in `metadata["targets"]`, keyed by the labels of `env`,
and the level used in `metadata["alpha"]`. The targets pass back as `targets`, where they are known:
`ut_igsp` may learn more, and `gies` refits the graph under exactly these. A setting where it found
no target maps to `[]`, which `targets` reads as observational. The
[0.14.0](benchmarks/0.14.0.md#ut-igsp) release card measures it against graphical-model-learning,
the authors' library.

ICP, invariant causal prediction (Peters, Buhlmann and Meinshausen 2016), finds the causal parents
of one `target` from continuous data pooled from several environments, such as observation and
experiments, that leave the target's mechanism alone. It tests every set of the other variables, or
every set of at most `max_parents`, for a regression on the target that is the same in every
environment, by the paper's approximate or exact test, and returns the variables in every accepted
set: in a linear model, they are causal parents with probability at least `1 - alpha`. Its
`metadata` holds each variable's p-value and confidence interval and the accepted sets. The
[0.14.0](benchmarks/0.14.0.md#icp) release card measures it against R's InvariantCausalPrediction.

CAM-UV, causal additive models with unobserved variables (Maeda and Shimizu 2021), fits CAM's
additive model where some causes are hidden, with the paper's settings: HSIC tests at `alpha=0.01`
on the residuals of additive spline fits find each variable's candidate parents over sets of up to 3
variables and then its parents, and mark `o-o` each pair that a hidden common cause or a causal path
through a hidden variable joins, whose relation no regression can identify. The
[0.14.0](benchmarks/0.14.0.md#cam-uv) release card measures it from 4 to 50 observed variables
against lingam and causal-learn.

DYNOTEARS (Pamfil et al. 2020) learns a linear structural vector autoregression from one time series
the way NOTEARS learns a DAG: the least-squares loss of each step on the same step and the `tau_max`
steps before it, with an l1 penalty, under the acyclicity constraint on the same-step weights alone,
since a lagged edge points forward in time. It keeps the paper's settings for 500 time steps:
`lambda1=0.05` on both blocks, and thresholds of 0.3 on the same-step weights and 0.1 on the lagged
ones; for 50 steps the paper used `lambda1=0.2` and `lag_threshold=0.2`. It returns a temporal
structure whose `lag_weights` are the coefficients in the data's units, with a `dag` at lag 0. The
[0.14.0](benchmarks/0.14.0.md#dynotears) release card measures it against CausalNex.

### Experimental methods

A new method ships as experimental: it passes recovery tests but has no published benchmark, its
API may change without deprecation, and it warns with {py:class}`~andrey.ExperimentalWarning` on
its first call in a process.

```{eval-rst}
.. currentmodule:: andrey

.. autosummary::
   :toctree: api
   :nosignatures:

   notears
   dagma
   boss_fci
   grasp_fci
   fci_jci
```

NOTEARS and DAGMA are continuous-optimization learners that fit a weighted DAG by minimizing a loss
under a smooth acyclicity constraint, with NumPy and SciPy only. `robust=True` follows the first fit
with a refit under the Gaussian likelihood, which does not depend on the scale of the columns
(Deng et al. 2024).

BOSS-FCI and GRaSP-FCI ({py:func}`~andrey.boss_fci`, {py:func}`~andrey.grasp_fci`) are GFCI with a
BOSS or GRaSP search in place of GES (Ramsey, Andrews, and Spirtes 2025): the search gives a CPDAG,
conditional-independence tests remove the adjacencies that hidden common causes explain, and FCI's
rules orient the rest into a PAG.

FCI-JCI (Mooij, Magliacane, and Claassen 2020) runs FCI on data pooled from several environments,
with the environment index as one more variable that no data variable causes, so hidden common
causes are allowed and the index removes the edges that pooling adds. It returns a PAG over the
data variables and lists the index's edges in `metadata["index_edges"]`; `confounded_index=False`
assumes no hidden variable causes both the index and a data variable, as when the environment is
set from outside.

A method becomes supported once it has:

- a benchmark campaign compared with another package, in a published summary and with a section
  in its release's [release card](benchmarks.md);
- recovery baselines in `tests/recovery/baselines/`;
- a place in the release quality gate, `qa/quality_gate.py`;
- an example notebook.

`tests/unit/test_spec_registry.py` checks each criterion for every supported method.

### Conditional-independence tests

{{ indep_test_methods }} take their conditional-independence test as `indep_test`:

| `indep_test` | Test | Status |
|---|---|---|
| `"fisherz"` | Partial correlation with Fisher's z-transform: linear effects, Gaussian noise | Supported |
| `"student_t"` | Partial correlation with a Student-t p-value, exact for Gaussian data at any size | Supported |
| `"normal_scores"` | Fisher's z-transform of the partial correlation of each column's normal scores: monotone effects, any marginals | Experimental |
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

On data with a `NaN`, {py:func}`~andrey.core.ci.make_indep_test` returns a
{py:class}`~andrey.core.ci_missing.TestwiseDeletion` of the test: each query reads the rows observed
in all of its columns. `fisherz` and `student_t` compute the query's correlations directly from
those rows, many queries at once, so the p-value is the one the test gives on that block up to
rounding in its last digits. Every other test is built on that block, its columns in data order,
and gives that test's p-value exactly; `normal_scores` therefore ranks each column within the rows
its query reads; a registered test works the same way, built anew for each
query.

```{eval-rst}
.. currentmodule:: andrey.core

.. autosummary::
   :toctree: api
   :nosignatures:

   ci.FisherZ
   ci.StudentT
   ci.NormalScores
   ci_kernel.KCI
   ci_kernel.RCoT
   ci_cmiknn.CMIknn
   ci_gpdc.GPDC
   ci.CITest
   ci_missing.TestwiseDeletion
   ci.register_indep_test
   ci.available_indep_tests
   ci.make_indep_test

.. currentmodule:: andrey
```

## Graph structures

`GraphStructure` reads and writes pcalg's `amat`, with `from_pcalg` and `to_pcalg`, and
writes Graphviz DOT text with `to_dot`, which needs no other package. It converts between graph
kinds: `to_dag` returns a DAG of a CPDAG's equivalence class, and when the class holds more than
one DAG, it asks you to choose how to pick one, with `tie_break="label"` or `seed`; `to_mag`
projects a DAG with hidden and selection variables onto its other nodes as a maximal ancestral
graph, of kind `mag`; and `to_pag` returns the PAG of a DAG or of a MAG, which `fci` would return
with an exact test. `andrey.metrics.to_cpdag` returns a DAG's CPDAG. A result reaches them through
its graph, as in `out.structure.to_dag(tie_break="label")`. It also answers graph queries:
`parents`, `children`, `ancestors`, `descendants`, `markov_blanket`, `d_separated` (a DAG or a
CPDAG), and `m_separated` (a MAG or a PAG, and a DAG, where the two agree); a CPDAG or PAG answers
as every member of its class does ([Query a graph](guides/graph-representation.md#query-a-graph)).
[Graph representation](guides/graph-representation.md) gives the conventions, and
[Hand a graph to DoWhy](guides/graph-representation.md#hand-a-graph-to-dowhy) passes a result to
DoWhy for effect estimation.

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
and the averaged graph, a valid graph of the method's kind. It takes a method's name, such as
`"pc"`, or the method itself, and the method's own arguments. Each resample is a subsample by
default: `sample_fraction` (0.5) of the rows, drawn without replacement. `resampling="bootstrap"`
draws as many rows as `data` has, with replacement, instead. The generator depends on `seed` and the
resample's number alone, so a seed gives the same result at any `n_jobs`, and a longer run begins
with the shorter run's fits. A method that takes a row-aligned input (`env`, or `targets` as a 0/1
matrix) resamples it with the rows, and a column that `env` or `targets` names travels with its
rows. When `env` holds text labels, or for the methods that take `targets` ({{ targets_names }}),
each resample takes the same share of every environment's rows, at least one, so no resample drops
an environment; the numeric index of `cdnod` and `fci_jci`, such as a time step, resamples the rows
as a whole. For the methods that take `targets` and return a CPDAG ({{ cpdag_targets_names }}), the
averaged graph is made a CPDAG without the targets, so an edge that only the interventions orient is
undirected there; the edge frequencies keep its direction. `multi_group_direct_lingam` resamples
each dataset on its own and returns one result per dataset; `longitudinal_lingam` resamples the
units, the same rows of every dataset. A time-series method on one series subsamples one stretch of
consecutive rows; with `resampling="bootstrap"` it resamples blocks of `block_length` consecutive
rows, by default at least ten times its `tau_max`. `gin` is the one method it does not take, since
its number of latent nodes differs from fit to fit.

{py:class}`~andrey.BootstrapOutput` holds the counts. `frequency("->")` is an `(n, n)` array whose
cell `[i, j]` is the share of fits with the edge `i -> j`, and the same for `"<-"`, `"--"`,
`"<->"`, `"o->"`, `"<-o"`, `"o-o"`, `"-o"`, `"o-"`, and `"absent"`; `table()` lists every edge that
appeared; `structure` is the averaged graph, and `average(threshold)` builds it at another
threshold without refitting. The averaged graph starts from a vote: each pair takes its most
frequent state, no edge counted as one, so it keeps its most frequent edge when more fits have that
edge than have no edge; `threshold` (0.5 to 1) asks the pair to be adjacent in more than that share
of the fits, or in all of them, and two edges that tie give `--` (`o-o` for a PAG). The vote is
then made a valid graph of the method's kind: a DAG with its edges and no cycle, the CPDAG of such
a DAG, or the PAG of a MAG built from it. A tie between the two directions of a DAG method's edge
points by the columns' `tie_break` ranks, so the graph relabels with the columns. When no such
graph exists, the averaged graph is the resample graph nearest the others among the resamples that
are valid graphs of the kind, with a warning, and `metadata["averaged_graph"]` says
`"nearest_resample"` instead of `"vote"`; when no resample is valid either, it says
`"nearest_resample_invalid"`. `strict=True` raises instead of both. A bootstrap resample repeats
rows, so a conditional-independence test sees more evidence than the resample holds; a subsample
does not, which is why we make it the default (the `bootstrap` docstring gives the measurements).
Either way, read a frequency as a ranking of the edges, not as a probability. `andrey bootstrap`
runs the same from the shell ([Command-line interface](#command-line-interface)).

```{eval-rst}
.. autosummary::
   :toctree: api

   bootstrap
   BootstrapOutput
```

## Markov check

`andrey.markov_check` asks whether data are consistent with a graph. A DAG implies independences:
in a topological order, each variable is independent of every earlier variable that is not its
parent, given its parents (the ordered local Markov property of Ramsey, Andrews, and Spirtes 2024).
A CPDAG is checked through a DAG of its class, which implies the same independences. The check runs
one conditional-independence test for each implied independence, `indep_test` as `pc` takes it, and
asks whether the p-values are uniform on (0, 1), as they are when every tested independence holds.
The graph is rejected when the Anderson-Darling test of uniformity gives a p-value of at most
`alpha`.

{py:class}`~andrey.MarkovCheckOutput` holds the verdict (`passed`, `verdict`), the test
(`uniformity_statistic`, `uniformity_p_value`), the `share_dependent` of p-values below `alpha`
(about `alpha` when the graph is right), and every tested independence with its p-value in
`table()`. A graph that implies no independence, a complete DAG, cannot be rejected. A PAG, a MAG,
an undirected graph, and a time-series result raise `ValueError`.

Check on rows the graph was not learned from: a graph fitted to the same rows can pass because the
fit made its independences true there. Split the rows, learn on one half, and check on the other.
Each test reads `sample_fraction` (0.5) of the rows, drawn at random for that test alone, so that
the tests do not all read the same rows; `seed` fixes the draws. In 1,000 simulated linear datasets
per cell, the check rejected the true graph at `alpha=0.05` in 6% to 9% of the datasets with the
default, and in 9% to 21% with `sample_fraction=1.0`, where the tests share every row. A graph with
one edge missing was rejected in 96% to 99% of the datasets at 1,000 rows and 72% to 82% at 200 rows
(98% to 100% and 89% to 93% on all rows). The `markov_check` docstring gives the measurements.

A pass does not show that the edges are right: a graph with extra edges also passes, because an
added edge removes an implied independence without making another false. Read the check with the
number of edges. To choose among graphs of the same data, take the one with the fewest edges among
those that pass, for example across a method's `alpha` values (the
[example notebook](examples/markov_check)). `andrey markov-check` runs the same from the shell
([Command-line interface](#command-line-interface)).

```{eval-rst}
.. autosummary::
   :toctree: api

   markov_check
   MarkovCheckOutput
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
`andrey.config.backend`, `andrey.config.n_jobs`, or `andrey.config.tie_break` to change the
default for every later call, or use `with andrey.config(backend=..., n_jobs=...,
tie_break=...):` for one block. [Configuration](configuration.md) lists the `ANDREY_*`
environment variables and the order in which the settings apply, and
[Column order and machines](guides/column-order.md) describes `tie_break`.

## Warnings

Every warning Andrey raises is an {py:class}`~andrey.AndreyWarning`, a `UserWarning`, and each
situation has one category, the one the table names, in every method. A warning never changes a
result. Each distinct message warns once per process, at the line in your code that
made the call; a `BoundaryWarning` warns once per method. A warning from a dependency keeps its own
category, except scikit-learn's `ConvergenceWarning` from FastICA and LARS, raised again as a
`SearchLimitWarning`.

| Category | When |
|---|---|
| {py:class}`~andrey.ExperimentalWarning` | The first call of an experimental method. |
| {py:class}`~andrey.PerformanceWarning` | A call that will run correctly but slowly: a PC conditioning pass of at least ten million CI tests, PNL on fewer worker threads than asked to stay within its memory, worker processes that start without the forkserver's local socket, or a search's worker-pool cutoff left at its default. |
| {py:class}`~andrey.BackendFallbackWarning` | A requested backend is unavailable, so the call runs on another one. |
| {py:class}`~andrey.SearchLimitWarning` | A fit stopped at its iteration cap, such as `max_iter`, rather than on its own stopping rule, or FastICA or LARS did not converge. |
| {py:class}`~andrey.SampleSizeWarning` | A test runs below its minimum rows, so it may reject a true independence more often than `alpha`: RCoT conditions on fewer than 500 rows. |
| {py:class}`~andrey.BoundaryWarning` | A fit larger than the largest size its method finished within the caps of the newest boundary lane, at the nearest measured setting. |
| {py:class}`~andrey.DeprecatedArgumentWarning` | A former argument name or value, which works for one more release as the current one. |
| {py:class}`~andrey.DataTypeWarning` | A method that reads columns as numbers met columns that look like category codes, or a mixed-data setting inferred categorical columns from their values or found none. |

A `BoundaryWarning` takes the setting whose rows, or rows per variable, are nearest the data's, and
the largest number of variables any measured mean degree finished there, since the data's density
is unknown; `anm` and `pnl` count rows instead. It warns when the data have more and that ladder
ended on the time or memory cap. Past a ladder that ended on accuracy, an error, or its last
planned size, the method was not measured, so it does not warn, and a method with no ladder never
does. The sizes are a reference, not a hard limit: they change with the data, the density, the
settings, and the machine. The method's **Limitations** paragraph and the release card give them,
and `warnings.filterwarnings("ignore", category=andrey.BoundaryWarning)` silences the warning.

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
   BoundaryWarning
   DeprecatedArgumentWarning
```

## Datasets

`andrey.data.load_dataset` loads a dataset with its reference graph:

| Name | Variables | Edges | Data |
|---|---|---|---|
| `sachs` | 11 | 17 | Measured: 853 observational cells, or 7,466 in nine conditions (Sachs et al., 2005) |
| `asia` | 8 | 8 | Simulated on the published network (bnlearn) |
| `child` | 20 | 25 | Simulated on the published network (bnlearn) |
| `insurance` | 27 | 52 | Simulated on the published network (bnlearn) |
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

`load_dataset("sachs", conditions=True)` loads all nine of Sachs's conditions instead, 7,466 cells:
the 853 observational cells and eight conditions in which a reagent stimulated or inhibited the
network. `env` gives each cell's condition as a code and `conditions` their names, and `targets` the
measured variables each reagent acts on, `None` for PMA and b2cAMP, whose targets are unknown. They
are the inputs of the methods that take `env`: `cdnod`, `fci_jci`, `gies`, `gnies`, `icp`, `igsp`,
and `ut_igsp`. `gies` and `igsp` need every condition's targets, so you pass them the rows of the
seven conditions whose targets are known; `ut_igsp` takes all nine and learns each condition's
targets beyond the declared ones:

```python
pooled = load_dataset("sachs", conditions=True)
andrey.cdnod(pooled.data, env=pooled.env)
andrey.ut_igsp(pooled.data, pooled.env, targets=pooled.targets)

known = {k: cols for k, cols in pooled.targets.items() if cols is not None}
rows = np.isin(pooled.env, list(known))
andrey.gies(pooled.data[rows], pooled.env[rows], targets=known)
```

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

`andrey.data.sample_scm` samples a structural causal model named by its graph, functional form, and
noise, and `andrey.data.SCM` builds one from those parts; both take a linear, nonlinear, or
time-series model. `functional="location_scale"` gives each variable noise whose scale depends on
its parents, the location-scale noise model of Immer et al. (2023); with `scale="rescale"` that
scale stays a smooth function of the parents at every depth of the graph.
`andrey.data.functional.linear(positive=True)` makes every weight positive, and
`variance_range=(lo, hi)` on a noise family draws each variable's noise variance from `U[lo, hi]`,
as the simulations of GnIES (Gamella et al. 2022) do.
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

   sample_scm
   SCM
   sample_discrete
   sample_mixed
   interventions.hard
   interventions.soft
   interventions.HardIntervention
   interventions.SoftIntervention
```

#### Scale

In linear data drawn with the usual weights, a variable's variance grows down the causal order, so
sorting the columns by variance recovers much of that order (var-sortability, Reisach et al., 2021).
The share of a variable's variance that the others explain, its $R^2$, grows down the order too, and
no rescaling of the columns changes it ($R^2$-sortability, Reisach et al., 2023). The `scale`
argument of `sample_scm` and `SCM.sample` sets how the values are scaled; the graph is the same in
every mode. Its default, `"standardize"`, z-scores every column, which works for every model at any
size, so no method can win by the variances. Pass `scale="raw"` for the values as generated.

| `scale` | Values | Var-sortability | $R^2$-sortability |
|---|---|---|---|
| `"standardize"` (default) | each column z-scored after generation | 0.5 | as in the raw data |
| `"unitless"` | weights and noise drawn so every variable has variance 1 (Herman et al., 2025) | about 0.5 | about 0.5 |
| `"rescale"` | each variable divided by its standard deviation as it is generated | 0.5 | about 0.5 on sparse graphs |
| `"raw"` | as generated | high | high |

`"unitless"` follows Algorithm 1 of Herman et al. (2025): a variable's weights and noise depend only
on its number of parents, and the ratio of a parent's share of its variance to the noise's has no
bound. It takes the linear functional, up to 1,000 variables, since it keeps their correlation
matrix. A sample's `report` holds its var-sortability and, up to 100 variables, its
$R^2$-sortability, and `andrey.data.sortability.varsortability` and `r2sortability` compute them
for any data and graph. On 30 random graphs of 20 variables at mean degree 2, with 1,000 rows,
$R^2$-sortability averages 0.74 raw or standardized, 0.52 with `"rescale"`, and 0.51 with
`"unitless"`.

#### Interventions

`SCM.do` returns the model under interventions, and its `sample` draws from it.
`andrey.data.interventions.hard` cuts a variable from its parents and sets it to a value, or draws
it from a Gaussian; `soft` keeps the parents and changes the mechanism: it shifts or rescales the
variable's noise, or rescales its parents' weights. `soft(noise_scale=(low, high))` multiplies the
noise of each row by its own draw from `U[low, high]`, the noise interventions of the ICP
simulations (Peters, Buhlmann and Meinshausen 2016, Appendix G). `soft(weights="redraw")` draws each
parent's weight anew, on its own, from the functional's own `weight_range`; the new weights of
Appendix G come from a range of their own, which `soft` does not take. A variable without parents
has no weights to rescale or draw anew. You name a variable by its column in the data `sample`
returns, after the column shuffle, and give values and shifts in the units of those data, whatever
the `scale`. The sample under `do` draws the noise the sample without interventions draws, so every
column the interventions do not reach is the same in the two.

`envs=` on `SCM.sample` and `sample_discrete` draws several environments into one dataset: one
mapping of interventions per environment, `{}` for none. Each environment draws its own noise, and
the rows come pooled, environment 0 first, with `env`, each row's environment, and `targets`, each
environment's intervened columns:

```python
import andrey.data as data
from andrey.data.interventions import hard, soft

scm = data.SCM(graph=data.graphs.erdos_renyi(d=20, avg_degree=2),
               functional=data.functional.linear(), noise=data.noise.gaussian())
ds = scm.sample(n=[2000, 400, 400], seed=0,
                envs=[{}, {3: hard(2.0, sd=0.2)}, {5: soft(shift=1.0), 8: soft(shift=1.0)}])
ds.env      # 2,000 zeros, then 400 ones and 400 twos
ds.targets  # {0: [], 1: [3], 2: [5, 8]}
```

`"standardize"` and `"rescale"` take their means, standard deviations, and divisors from the model
without interventions on the same noise and apply them to every environment, so a shift in one
environment stays a shift in the data. On discrete data, `hard(category)` or `hard(probs=...)`
gives a variable one distribution whatever its parents, and `soft()` draws its table anew. The true
graph is the graph without interventions.

`model_seed=` on `SCM.sample` and `sample_discrete` keeps one model across datasets. The model
comes from `model_seed`: the graph, the weights or tables, the noise scales, the column order, and
the weights or tables a `soft` intervention draws anew. Each environment draws its own, and `params`
holds those of the model without interventions. `seed` draws only the rows, so samples with one
`model_seed` and different seeds are datasets of one model, with the same true graph and `params`:

```python
datasets = [scm.sample(n=[500, 500], seed=s, model_seed=0, envs=[{}, {3: soft(weights="redraw")}])
            for s in range(10)]
```

## Metrics

`andrey.metrics` scores a result against a known graph. `score(output, truth)` chooses the metrics
from what the two structures carry (their kind, latent nodes, lags, an ordering, weights) and
returns them in one dictionary, with the choice under `family`. The other functions compute one
metric each. Both arguments can be a `StructureOutput`, a structure, or a directed adjacency
matrix, where a nonzero `A[i, j]` means `i -> j`, over the same variables in the same order. A
matrix whose directed edges form a cycle reads as a `digraph`. Wrap a matrix of endpoint marks with
`GraphStructure.from_numpy(M, kind=...)` first.

| `family` | When | `score` returns |
|---|---|---|
| `undirected` | either graph is `undirected` | the skeleton scores and `shd`, the skeleton SHD: the pairs adjacent in one graph and not in the other |
| `dag` | both graphs are DAGs | skeleton and arrowhead precision, recall, and F1; `shd`; `mec_shd` and `mec_arrowhead_f1`, scored through the CPDAGs; the causal-order scores when the result has an ordering; `coefficient_mae` when both carry weights |
| `cpdag` | either graph is a CPDAG | the skeleton and arrowhead scores and `shd`, with a DAG converted to its CPDAG first |
| `pag` | either graph is a PAG or a MAG | the skeleton and arrowhead scores and `shd`, plus `shd_endpoint` and the scores for bidirected pairs |
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
the nodes when pygraphviz, the `viz` extra, is installed, and a built-in layout otherwise. Latent
variables are hollow, dashed circles. `draw(structure, targets=["x"])` marks the variables an
experiment set: each is a square, and a bar crosses each edge into it. A hard intervention cuts
those edges, and a soft one keeps them; the marks are the same.

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
| `andrey bootstrap hc --data data.csv` | Edge frequencies over refits on resamples, and the averaged graph |
| `andrey markov-check pc.json data.csv` | The p-values of the independences a DAG or CPDAG implies, and whether they pass |
| `andrey --help --json` | Every method contract and the shared CLI contract |
| `andrey config` | Current settings, backend availability, the platform, and the optional extras |
| `andrey config --help` | Environment variables, defaults, effects, and precedence |
| `andrey calibrate` | Measured worker-pool cutoffs (GES, skeleton, RCoT) for this machine, as settings to export |
| `andrey --skill` | The agent guide (`SKILL.md`): methods, settings, how to run and read results |

In `--help --json`, each parameter carries its bounds as JSON Schema's `minimum` and `maximum`, or
`exclusiveMinimum` and `exclusiveMaximum`, the package-wide `concept` it implements, and the
former names (`aliases`) and values (`choice_aliases`) it still accepts. `andrey --help --json`
and `andrey config --help --json` list each concept, such as `alpha` or `max_depth`, with the name
each method gives it. `andrey run` rejects a value out of range before the fit.

`--help` selects introspection and `--json` selects the representation: help reads no data or
configuration file and never executes the method, while `andrey run hc --data data.csv --json` runs
hill climbing. Configuration reports list detected backends, not the backend an operation will use.

`andrey bootstrap <method>` takes the inputs, parameters, and options of `andrey run`, and
`--n-resamples`, `--seed`, `--threshold`, `--strict`, `--resampling`, `--sample-fraction`, and
`--block-length`.
Its result is a run's, with the averaged graph as `structure`, plus `bootstrap` (the settings) and
`frequencies` (each edge that appeared in a fit, most frequent first); see
[Bootstrap](#bootstrap).

`andrey markov-check <graph> <data>` takes a DAG or CPDAG from `andrey run <method> -o <file>` or
from a saved result, the data to check it against, and `--indep-test`, `--alpha`, `--categorical`,
`--sample-fraction`, and `--seed`. Its result has `markov_check` (`passed`, `verdict`, the
uniformity test, and `tests`, each independence with its p-value, capped at the 1,000 smallest) and
`run`; `status` is the run's, and the verdict is `markov_check.passed`. See
[Markov check](#markov-check).

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
