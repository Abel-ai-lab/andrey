---
orphan: true
name: standards
description: What Andrey requires of every method, how each requirement is checked, and where the
  exceptions are recorded.
meta:
  type: spec
---

# Standards

Every method in Andrey is held to the requirements below. Each section states a requirement, how we
check it, and where the exceptions and the gaps in the check are recorded. The tests that run on
every pull request cover every method; the quality gate, the reference comparisons, and the
benchmark campaigns cover the {{ n_supported }} supported ones ([Testing](ci.md)).

## The setting its paper claims

A method is judged in the setting its paper claims: the model, the data regime, and the scale. Its
docstring states that setting in a **When to use.** paragraph, and where it stops working in a
**Limitations.** paragraph.

**Check.** `tests/unit/test_spec_registry.py` fails when a docstring lacks either paragraph. The
recovery tests (`tests/recovery/`), the quality gate (`qa/quality_gate.py`), and the benchmark
campaigns draw their data from the claimed setting; review, not a test, checks that they do.

**Exceptions.** A benchmark row measured outside the claim carries `"claim": "out"` in its
published summary: its release card reports it, and no gate judges it.

## Agreement with reference implementations

Where the authors' code or a standard package implements a method, Andrey is never less accurate
than it on the same input in a setting that repeats on fresh data, and every difference has a cited
cause.

**Check.** The campaigns record each setting's paired difference in errors in
`benchmarks/published/<release>/summary.json`. From 0.8.0, `qa/reference_parity.py` counts a
setting as worse when the bootstrap 95% interval of that difference, over at least 10 paired
datasets, lies above 0; a setting rerun on fresh datasets is judged by the rerun. The release
quality gate prints each worse setting, and the release checklist reviews every one.

**Exceptions.** `benchmarks/published/parity-allowlist.json` gives each worse setting's cause, with
a citation: Andrey follows the paper and the reference does not, the reference has a defect, the
two break an exact tie differently, or a default trades this setting for others. A worse setting
does not fail the gate on its own, a setting with fewer than 10 paired datasets is reported and not
judged, and the summaries before 0.8.0 compare counts or means without an interval.

## Better than the empty graph

In its claimed setting, a method makes fewer errors than the empty graph, which has no edges.

**Check.** The quality gate passes a method on a dataset only when its median SHD is below the
empty graph's, and every supported method must pass before a release merges. Where a reference
makes significantly fewer errors than the empty graph, `qa/reference_parity.py` requires the same of
Andrey. A fit that does not finish in time never counts as an empty graph.

**Exceptions.** A method's **Limitations.** paragraph and its release card give the settings where
it does no better than the empty graph.

## How far each method runs

Each method's reach is measured: the largest number of variables, at a given number of rows and
density, at which it finishes within fixed time and memory caps and beats the empty graph.

**Check.** The [0.10.0 release card](benchmarks/0.10.0.md#the-boundary) measured it, with a 1-hour
and a 28 GB cap on 4 cores. `src/andrey/api/_boundary_table.py` holds the result, a fit past its
method's boundary warns with a {py:class}`~andrey.BoundaryWarning`, and
`tests/unit/test_boundary.py` fails when the table and the newest published boundary differ.

**Exceptions.** The map covers the methods Andrey had at 0.10.0. A method added since has no map and
no warning; its **Limitations.** paragraph gives the sizes its campaign measured.

## Column order

The order of the columns says nothing about the causal structure, so reordering them returns the
same graph with its nodes relabeled.

**Check.** `tests/unit/test_column_labels.py` refits every method whose registry entry says it
relabels on other orders of three columns and compares the edges by name. Ties between equal scores
go to a rank computed from the data, not to a column's position
([How Andrey breaks ties](guides/column-order.md#how-andrey-breaks-ties)).
`qa/identity.py --shuffle K` refits each supported method on `K` column orders of its quality-gate
datasets and lists every output that changes.

**Exceptions.** Each method's `column_order` in the registry records whether reordering relabels its
result, and the [column-order table](guides/column-order.md#each-method) prints it with the reason.
`qa/identity.py` runs by hand and does not refit ICP, IGSP, FCI-JCI, or J-PCMCI+; the last release
card that reports a run is [0.9.0](benchmarks/0.9.0.md)'s.

## Units and transforms

**Rescaled columns.** Rescaling a column, a change of its units, returns the same graph unless the
method's published algorithm reads the units. `tests/unit/test_column_transforms.py` fits every
method that takes a table with each column multiplied by its own power of two, from `2**-10` to
`2**10`, and again with environments and on mixed data where a method takes them, and requires the
same graph, mark for mark. The registry marks {{ reads_units_names }} as `reads_units`: each must
change its graph there instead, so the mark cannot go stale, and its page says how it reads the
units. `tests/unit/test_offset_invariance.py` shifts every column, or scales and shifts them all
alike, and requires the same graph from every method. Each check runs on one small dataset per kind
of input, so it covers the code those data reach.

**Benchmark data.** A benchmark must not reward a method for reading the simulation's units: in
linear data drawn with the usual weights, a variable's variance, and the share of it the other
variables explain, its $R^2$, grow along the causal order. The benchmarks' linear and nonlinear
regimes z-score every column, which sets var-sortability to 0.5, and the `_unitless` regimes, like
`andrey.data.sample_scm(scale="unitless")`, bring $R^2$-sortability close to 0.5 as well
([Scale](api.md#scale)). Every benchmark record carries its dataset's `varsortability` and
`r2_sortability`, and `tests/unit/test_data_sortability.py` checks both measures. Z-scoring leaves
$R^2$-sortability as in the raw data, and the unitless regimes and the controls that sort the
columns by variance or by $R^2$ appear only in the campaigns whose cards name them, such as the
0.17.0 comparisons of NOTEARS, DAGMA, SCORE, and NoGAM.

**Monotone transforms and category codes.** Under a rank-based test or score, a strictly increasing
transform of a column, such as a log, returns the same graph; under a categorical test or score, so
does recoding the categories. `tests/unit/test_core_ci.py` checks that the p-values of
`indep_test="normal_scores"` do not change under such transforms, `tests/unit/test_core_score.py`
that the scores of `score_func="normal_scores"` do not, and `tests/unit/test_core_discrete.py` that
`"chisq"`, `"gsq"`, BDeu, and the categorical BIC do not depend on the codes.
`tests/unit/test_column_transforms.py` refits methods with each column through its own increasing
map and requires the same graph: {{ rank_test_only_names }}, which read the data only through their
test, with `indep_test="normal_scores"`; {{ rank_score_only_names }}, which read it only through
their score, with `score_func="normal_scores"`; and {{ test_and_score_names }}, which take both,
with both, each method that takes the score on ten datasets. With their default score, the methods
that take both read the values, so a monotone transform can change their graph even under the
rank-based test. The file also refits every method that takes `"chisq"` or `"gsq"` with each
column's categories recoded in order. A method that scores categorical data without a test, such as
`ges` with `score_func="bdeu"`, is checked through its score alone. Both `"normal_scores"` options
are experimental, and the score takes no `env`.

## Faster paths, the same output

A faster path returns the same output as the path it replaces: the same graph, weights, causal
order, and metadata, bit for bit.

**Check.** On every pull request, `tests/unit/test_parallel_ges.py`, `test_parallel_skeleton.py`,
and `test_parallel_dsep.py` compare the worker pools of GES, the PC skeleton, and FCI's
possible-d-separation search with the serial search, and the PyTorch paths must match NumPy
([Testing](ci.md)). For a speed change, `qa/identity.py` hashes every supported method's full
output, floats as bytes, on the base commit and on the change; `--compare` fails on any changed
output or any rise in the number of queries a search makes.

**Exceptions.** For one speed change, `qa/identity.py` runs by hand. Each release runs it against
the release before, and [Release checks](#release-checks) gives the results. The [GPU](ci.md#gpus)
parity suite runs on our own machines, not in CI, and agrees with NumPy within a tolerance.

## Workers and machines

A fit returns the same result with any number of worker processes and on any machine.

**Check.** The pool tests above require identical results on one and on several workers, and the
recovery tests the same result on every run. `qa/identity.py --workers N` refits PC, FCI, FCI-JCI,
GFCI, CD-NOD, RSMAX2, GES, and PNL on `N` workers. Scores are compared after rounding, ties go to
ranks computed from the data, and most methods run their linear algebra on one BLAS thread, so the
host's thread count cannot change their result.

**Exceptions.** [Across machines](guides/column-order.md#across-machines) names the methods whose
result can change on another machine, and why. No check compares outputs across machines; the one
published measurement is ICA-LiNGAM's, on six BLAS kernels, in the same guide.

## Release checks

Before a release merges, `qa/identity.py` fits every supported method on the release and on the
release before it, on complete data with default arguments, and compares the outputs bit for bit.
The quality gate and the reference comparisons run on the same code.

| Release | Outputs against the release before | Quality gate | Reference comparisons |
|---|---|---|---|
| 0.17.1 | 49 methods, 1,963 outputs, none changed | 145 checks, 0 failed | 2,971 rows, 0 failed |
| 0.17.0 | 45 methods, 1,819 outputs, none changed; 4 methods new | 145 checks, 0 failed | 2,971 rows, 0 failed |
