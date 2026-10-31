---
name: column-order
description: What each method keeps when you reorder the columns or change machines, how Andrey
  breaks ties, and the setting that breaks them by column position.
meta:
  type: guide
---

# Column order and machines

The order of a dataset's columns says nothing about the causal structure, and neither does the
machine that runs the fit. Andrey aims to give the same graph however you order the columns and
whichever machine you use. This page says where that holds, where it does not, and how to trade it
for speed with the `tie_break` setting.

## How Andrey breaks ties

Many steps of a search choose among candidates that score the same: two edges whose additions
improve the score equally, or two variables that could come next in a causal order. Under a
score-equivalent score, adding `x -> y` or `y -> x` to an empty graph always ties, and each choice
changes the moves that follow. A rule that takes the first candidate by column position lets the
column order decide the graph, and many datasets list their variables close to causal order, so
that order would leak into the result.

By default, each column gets a rank computed from the data: a hash of how the column relates to
the others, through the rank correlations of continuous columns and through which rows share a
category in categorical ones. A tie goes to the candidate whose columns rank first. The rank stays
the same when you shuffle the rows, change a column's units, or flip its sign, and it moves with
its column, so reordering the columns relabels the result and changes nothing else. The rank is
arbitrary with respect to the causal structure.

Some columns are ones the data cannot tell apart in any row order, unit, or sign: a copy of another
column in other units or the other sign, or two columns whose rows also appear with the two
swapped. These rank by their values in row order instead, so for them a row shuffle can change
which ranks first, while reordering the columns still relabels the result.

Scores are compared at a fixed precision, so candidates whose scores differ only in their last
bits count as tied and go to the rank.

## Each method

What reordering the columns does to each method's result with the default settings. "Relabels"
means you get the same graph with its nodes in the new order. "Ties follow `tie_break`" marks the
methods the setting below changes: it replaces the rank that breaks their ties, for `pc` and `cdnod`
the p-value order of their colliders, for `pc`, `cdnod`, `fci`, and `gfci` the order-free rounds of
their orientation rules, and for `fci` and `gfci` the rank order of their possible-d-separation
step.

```{include} ../../_generated/column-order.md
```

The CI tests `kci`, `rcot`, `cmiknn`, and `gpdc` are not methods, so the table leaves them out.
They seed their random draws and order their variables by the same rank, so a p-value depends on
neither the argument order nor the column order, and `tie_break="position"` ranks their columns by
position instead. `pc`, `fci`, `gfci`, and `cdnod` follow the setting through them when
`indep_test` names one.

`bootstrap` is not a method either. Its resamples draw the same rows whatever the column order, so
its edge frequencies relabel like the method's. The averaged graph of a DAG method relabels too:
when the fits split evenly between the two directions of an edge, the columns' ranks point it, or
their positions under `tie_break="position"`. A CPDAG or a PAG needs no rank, as every order of the
tie gives the same graph.

Identical columns are the exception for every method: they rank the same, so position decides which
comes first, and the graph is the same up to swapping them. The score-based searches with the BIC
score, and `pc`, `fci`, and `cdnod` with the partial-correlation tests, reject columns that copy
another up to scale and shift, with a `ValueError` that names each group of copies; the LiNGAM
causal-order search keeps one column per group and places the other copies right after it.

## Across machines

Two machines can compute the same score with different last bits, because their CPUs run different
BLAS kernels or the same kernel on a different number of threads. The rounded comparison absorbs
these differences, and a column's rank comes from exact sums over the ranks of its values, which
they change only when they reorder two values. So a tie goes the same way on every machine, unless
two tied scores round to different values.

The thread count is fixed where it could matter. The skeleton searches of `pc`, `cdnod`, and `fci`,
and the tests of `iamb`, `inter_iamb`, `mmpc`, and the restrict phases of `mmhc` and `rsmax2`, with
the partial-correlation, mixed-data, RCoT, CMIknn, and GPDC tests, and with KCI below 2,000 rows,
the scores of `ges` (and so of `gfci`'s first phase), `gies`, `xges`, `hc`, `tabu`, `mmhc`, and
`rsmax2`, the fits of `notears`, `dagma`, `cam`, and `resit`, and the tests of `pcmci`,
`pcmciplus`, and `lpcmci` run on one BLAS thread, and so do their worker processes, so the host's
thread count cannot change their results. From 2,000 rows, KCI runs on every BLAS thread, where its
kernel products run faster; its p-values then repeat for a fixed thread count.

Two methods depend on more than ties. `calm` trains with PyTorch: its results repeat on one machine
with one torch version, and another platform can give another graph. When the likelihood fit of
`varma_lingam` stops before it converges, its graph can change with the versions of the numerical
libraries. Where the fit stops, even when it converges, also depends on the BLAS kernels the CPU
runs, so a series can give another graph on another type of CPU.

### ICA-LiNGAM

`ica_lingam` estimates an unmixing matrix with FastICA, which starts from a seeded random point and
iterates until it converges. With few rows per variable it converges slowly, and the last-bit
differences between machines, or a new column order, can lead it to another solution and so to
another causal order and graph; tighter FastICA tolerances do not prevent this. This comes from
FastICA's convergence, not from Andrey's implementation. With more rows per variable, FastICA
converges in fewer iterations and these differences shrink.

We fit simulated linear non-Gaussian data on scale-free graphs, with 20 and 50 variables and ten
seeds each, on six BLAS kernels: three on an Intel Core i7-7820X and three on an AMD EPYC 9534. From
100 rows per variable, every dataset gave the same graph on every kernel. At 30 rows per variable
some datasets with 50 variables differed, and at 10 rows every dataset did. Where several variables
can come next in the causal order, the rank picks one, as for the other methods.

## Ties by column position

`tie_break="position"` breaks the ties in the table's "Yes" rows by column position instead, the
rule of earlier releases. Set it for every later fit, for the fits in a block, or in the
environment, which also reaches `andrey run`:

```python
import andrey

andrey.config.tie_break = "position"  # every later fit

with andrey.config(tie_break="position"):  # the fits inside the block
    result = andrey.hc(data)
```

```shell
export ANDREY_TIE_BREAK=position
```

`andrey config` reports the setting in effect, and an unknown value raises `ValueError` naming
`rank` and `position`.

What it trades: ranking reads each column once per fit, so `position` saves little time next to the
search itself. The exception is `fci` and `gfci`, where the setting also orders the
possible-d-separation step. On simulated graphs, a fit with the order-free step took 0.99 to 1.05
times as long as one in column order at up to 200 variables and 0.78 to 1.42 times as long at 400
and 800. On some dense graphs from 400 variables, though, the order-free step's visiting order needs
more tests: one `gfci` dataset with 400 variables took 407 s, against 120 to 336 s in column order
over five column orders. `position` is not more or less accurate, since a tie is a choice the data
cannot make. Scores are still compared at the same rounding, so with a fixed column order you get
the same graph on every machine where the default does. But reordering the columns can change the
graph, and the column order can leak into it; `andrey.data.sortability.index_correlation` measures
how much a dataset's column order reveals of its causal order.

For `pc` and `cdnod`, the setting chooses the order in which conflicting colliders are oriented:
by the p-value that separated each pair under `rank`, in column order under `position`. Their
`collider_order` argument, `"p_value"` or `"column"`, overrides it for one call. Their
`collider_rule="conservative"` and `"majority"`, and those of `fci` and `gfci`, decide each
collider by a vote over the final skeleton's adjacencies, which does not read the column order.

The orientation rules of `pc`, `cdnod`, `fci`, and `gfci`, Meek's rules and FCI's R1-R10, run in
rounds under `rank`: each round sets together everything the rules imply on the graph as it stands,
and an edge or endpoint two rules decide differently stays undirected or a circle, so the column
order does not enter. Under `position` they fire one at a time in column order, and of two firings
that conflict the first decides. `gfci` then also records the first subset found as the separating
set of a pair its GES phase leaves non-adjacent, where `rank` records the union of the smallest
ones.

`fci` and `gfci` take the order of their possible-d-separation step from the setting too: their
`possible_dsep` argument, when the call gives none, is `"order_free"` under `rank`, which visits the
step by the rank, and `"fast"` under `position`, which visits it in column order. A `possible_dsep`
in the call overrides the setting.

The methods in the "No" rows have no rank to replace, and the setting does not change them.

## Comparisons with other packages

At the settings the benchmarks use, causal-learn's PC, FCI, BOSS, and GRaSP, pcalg's PC, and
tigramite's PC give results that depend on the column order. The benchmark tables compare such a
package with Andrey's order-free default. Where a benchmark also runs Andrey in its column-order
mode on the same datasets (for `pc` and `fci`, the search steps that `tie_break="position"` puts in
column order), the table shows that comparison beside the default, as context.
