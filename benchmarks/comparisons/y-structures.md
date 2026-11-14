# Y-structures

`y_structures_cpu.py` measures the ancestral claims of a method that works under selection bias. It
runs Andrey's `y_structures` and the authors' Y-structure code on the simulation of the method's
paper (Versteeg, Zhang, and Mooij, 2022, Section 4.2 and Appendix D) and scores each claim that `i`
is an ancestor of `j` against the model's ancestral pairs. A claim is ancestry, not an edge, so the
lane scores pairs and no graph. `y_structures_summary.py` judges Andrey's claims against the
authors' per cell, and the [release card][card] reports the result.

## Data

Each model is a linear-Gaussian structural model over `p` system variables and one context
variable, the last column (`andrey_bench/selection_models.py`):

- **Settings.** `fixed` is the 7-node graph of the paper's Figure 5(a): `C -> X1`, `C -> X4`,
  `X6 -> X4`, `X4 -> X5`, and `X3 -> X2`, selected on `X1 + X2`. `small` and `large` are random
  graphs with `p = 8` and `p = 16` system variables, drawn by the rule of the paper's Appendix D:
  each ordered pair of the `p + 1` nodes gets an edge with probability 0.15 or 0.09, in a causal
  order, and a graph without enough colliders is discarded. The context is a source with children,
  drawn uniformly. The selection variables, 1 or 3, come by one of two rules, `--rule`. `authors`,
  the default, is the rule of the authors' `simulate.R`: the first is drawn among the colliders
  and their descendants, the others among all other nodes, the context included, and a graph needs
  more than 3 or 5 colliders. `text` is the rule the paper's text describes: every one is a leaf
  among the colliders' descendants, and a graph needs at least 3 or 5 colliders. It is the harder
  setting, and the summary reports it as context.
- **Weights and noise.** Each edge has a weight with a random sign and a magnitude uniform on
  `[0.5, 1.5]`, and each column of weights is divided by
  `sqrt(1 + the sum of its squared weights from system variables)`, the rescaling of the authors'
  code against accumulating variance. Every noise term is standard Gaussian.
- **Two datasets per model.** `unbiased` has `n` plain rows. `biased` has `n` rows kept when the sum
  of the selection variables lies in `(2, 2.5)`, drawn in chunks until `n` are kept; a model whose
  sum rarely reaches the interval raises instead of returning fewer rows.
- **Rows and seeds.** 10,000 rows, as the authors' script runs them, and 20 models per setting by
  default; the release summary uses 40, seeds 0 to 19 judged and 20 to 39 the fresh ones. The
  model (graph, weights, column order) is a function of the setting and the seed, and the rows are a
  function of the model and `n`. The system columns come in an order drawn from the seed and the
  context stays last.

## Solutions

Every solution runs at `alpha = 0.01`, the value of the authors' `simulate.R`.

- `andrey.yst` and `andrey.yst_ext` run `andrey.y_structures` with `extended=False` and `True`,
  Fisher's z test (`indep_test="fisherz"`), on every column, the context included, as the authors'
  code reads them. The answer holds each claim's score and the fit's `metadata["n_tests"]`. An
  extended Y-structure whose second variable tests independent of the effect has score 0; the
  adapter writes it as the smallest positive number, so the lane counts the claim.
- `sbcd.yst` is the Y-structure and `sbcd.yst_ext` the extended Y-structure of the authors' code,
  `philipversteeg/sbcd` at a pinned commit (BSD 3-clause), read from disk and run unmodified.
  `benchmarks/start_sbcd_r.sh` downloads the repository, checks the archive's SHA-256, and installs
  the R packages it lists beside pcalg and RBGL. The adapter sources `R/wrapYStructures.R` and calls
  `ystructures` as `simulate.R` does, with Fisher's z test and the test level as the acceptance
  level. The function reads the number of variables and rows from the global environment, where the
  authors' script defines them, so the adapter assigns both before every call. The code counts no
  tests.
- The empty answer claims nothing, so its row is the base rate.

A fit that passes `--cap-wall-s` is a `timeout` with no scores.

## Verdicts

`y_structures_summary.py` takes the run folders as `RULE=GLOB`; the rule `authors` is judged and the
other is context. A cell is a setting, a dataset kind (with or without selection), and a form. Per
dataset it counts the claim errors, the false claims plus the true pairs not claimed, and takes 1
minus the average precision. The paired difference of Andrey's value and the authors' has a
percentile-bootstrap 95% interval over the datasets, and a cell is better or worse when the
interval lies wholly on that side of 0. A cell worse on the first seeds is rerun on the fresh seeds
and counts only if it is worse again. The floor has two forms: the claim errors against the empty
answer's, which a method passes where the authors' code does when its interval lies below 0, and
the precision of the pooled claims against the base rate, whose interval has to lie above 0.

## Measures

A claim `i -> j` among the system columns, with a score, is scored against the model's ancestral
pairs (`andrey_bench/local_lane.py`); pairs with the context and the diagonal are left out.

- `claims`, `tp`, and `fp`, `precision` and `recall`, which are undefined without a claim or a true
  pair, so the summary pools the counts over the models of a setting.
- `base_rate`, the share of ordered pairs that are ancestral: the precision of claiming every pair,
  which a method has to beat.
- `average_precision`, the area under the precision-recall curve of the ranking by score, ties as
  one threshold and the pairs that are not claimed last.

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with pcalg in the library, run
`start_sbcd_r.sh` and export what it prints, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/y_structures_cpu.py --out outputs/yst-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/y_structures_cpu.py --out outputs/yst --rule authors --seeds 40 \
  --warmup 1 --cores 1
$ANDREY_BENCH_PYTHON benchmarks/y_structures_cpu.py --out outputs/yst-text --rule text --seeds 40 \
  --warmup 1 --cores 1
$ANDREY_BENCH_PYTHON benchmarks/y_structures_summary.py authors=outputs/yst text=outputs/yst-text \
  --json cells.json
```

The release summary runs `--rule authors`, the default, and `--rule text` on seeds 0 to 39 with
`--warmup 1`. The other defaults are the three settings at 10,000 rows, 20 seeds from
`--first-seed 0`, one timed fit with no warm-up fit, and `--cap-wall-s` of 1800. `--part` and
`--parts` split the models between jobs, `--solutions` selects solutions, `--resume` continues a
run, and `--rescore` recomputes the scores from the stored claims without fitting. One JSON line per
fit goes to `records.jsonl`: the solution, the setting, the seed, whether the data are biased, the
time, the claims with their confidence scores, and the measures above.

## Adding a solution

A solution that claims pairs fills the contract of `local_lane.ClaimAdapter` and goes into
`ANDREY_ARMS` in the driver; nothing else changes. `fit(data, params)` receives the columns with the
context in the last one and returns `pack_answer(scores, n_tests)`: a `(d, d)` array in which
`scores[i, j] > 0` claims that `i` is an ancestor of `j`, the larger the score the more confident the
claim, and the number of tests, or `None`.

[card]: https://andrey.abel.ai/docs/benchmarks/0.18.0.html#y-structures
