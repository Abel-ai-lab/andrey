# FCI-JCI

`fci_jci_cpu.py` compares Andrey's `fci_jci` with `fci` from pcalg 2.7-12, the implementation of
FCI-JCI by the method's authors (Mooij, Magliacane, and Claassen 2020), on their simulation of data
from an observational environment and interventional environments, with hidden confounders. Tetrad's
FCI with the environment in its first knowledge tier, Andrey's `fci` on the pooled rows and on the
observational rows, and Andrey's `cdnod` run beside them as context: reported, never judged.

## Data

`andrey_bench.regime_models.jci_diagonal` draws each dataset as the authors' simulation script
(`simul_linear_stochastic.R`, acyclic) does, with the same parameters:

| Part | Draw |
|---|---|
| System variables | `p`, in a fixed order; each pair `i < j` gets the edge `i -> j` with probability `eps` |
| Hidden confounders | each unordered pair of system variables, with probability `eta`, gets a hidden variable of its own with exactly those two children |
| Context variables | `q` binary variables, each pointing into one system variable, its target; no two share a target when `q <= p` |
| Weights | uniform on `[0.5, 1.5]` in absolute value with a random sign, for every edge; each system variable's incoming weights are divided by the square root of one plus the sum of its squared system and confounder weights |
| Noise | standard normal |
| Design | `q + 1` regimes of the same number of rows: regime 0 observational, and in regime `k` context `k` is 1 and its target's noise is `N(1, 1)`; a perfect intervention also sets every weight into the target to 0 |

The paper's text describes the change as an offset of 1; the script also adds the context
variable's rescaled weight, so the shift is `1 + w / sqrt(s)`, which can be near 0. The script
produced the paper's figures, and we follow it. No context variable shares a hidden confounder
with a system variable, so JCI Assumptions 1 to 3 hold. The columns keep the script's order, as in
the paper; the column-order arm shows whether Andrey's result moves with it.

Andrey takes one environment variable. In the cells in claim the paper's simulation has one
context variable (`q = 1`), and the environment is that variable. In the merged cells the paper's
`q` context variables become one environment, the regime number `0` to `q`, which in the truth
points into every target.

## Cells

| Cell | `p` | `eps = eta` | Rows per regime | Context variables | Intervention | Claim |
|---|---|---|---|---|---|---|
| S4 | 4 | 0.5 | 500 | 1 | mechanism change | in |
| S4p | 4 | 0.5 | 500 | 1 | perfect | in |
| L10 | 10 | 0.25 | 500 | 1 | mechanism change | in |
| X100 | 100 | 0.02 | 100 | 1 | mechanism change | in, where pcalg finishes |
| S4m | 4 | 0.5 | 500 | 2, merged | mechanism change | out |
| L10m | 10 | 0.25 | 500 | 10, merged | mechanism change | out |
| X100m | 100 | 0.02 | 100 | 10, merged | mechanism change | out |
| L10n100, L10n2000 | 10 | 0.25 | 100, 2,000 | 1 | mechanism change | out |
| P20, P50 | 20, 50 | `2.25 / (p - 1)` | 500 | 1 | mechanism change | out |

In claim are the paper's three regimes (Sec. 5.2) with one context variable: small (`p = 4`,
`eps = eta = 0.5`, 500 rows per regime, the paper's sweep runs `q = 0` to 4) with mechanism changes
and with perfect interventions, larger (`p = 10`, `eps = eta = 0.25`, 500 rows, the paper's `q =
10`), and large (`p = 100`, `eps = eta = 0.02`, 100 rows, the paper's `q = 10`). Out of claim are
the paper's own `q` merged into one environment, which Fisher z reads as a number, other row
counts, and 20 and 50 variables at L10's expected degree.

Each cell runs 20 datasets, seeds 0 to 19, and P20 and P50 10. Both packages run at alpha 0.01, the
paper's, and 0.05, Andrey's default; X100m, P20, and P50 at 0.01 only.

## Settings

Both packages run the Fisher z test at the same alpha over the data and the environment as one more
column, with the same JCI assumptions:

| Andrey | pcalg | Assumptions |
|---|---|---|
| `fci_jci(X, env, alpha=a, indep_test="fisherz")`, the default `confounded_index=True` | `fci(..., jci = "1")` | 1: no system variable causes the environment |
| `confounded_index=False` | `fci(..., jci = "123")` | 1 to 3: also no system variable shares a hidden cause with the environment (3 says nothing with one context variable) |

pcalg runs `fci(list(C = cor(X), n = nrow(X)), indepTest = gaussCItest, alpha = a, type = "normal",
jci = ..., contextVars = <the environment's column>, selectionBias = FALSE)` with every other
argument at its default (the stable skeleton, the possible-d-separation step, no conservative or
majority rule), the settings of the paper's experiments (Sec. 5.1.2). `selectionBias = FALSE`
leaves out the rules for selection bias, R5 to R7, as JCI assumes none, and pcalg needs it for any
`jci` but `"0"`. `cor(X)` is part of the timed fit.

Where the two differ by design:

| | Andrey | pcalg |
|---|---|---|
| Adjacency search | the same result in any column order: columns ranked, separating sets the union of the smallest separating subsets | the stable skeleton in column order, keeping the first separating set found |
| Possible-d-separation sets | formed after the environment's marks are set | formed on pcalg's own skeleton and colliders |
| Test statistic | Fisher z from the data | Fisher z from `cor(X)` |

The context arms, each at both levels where the cell runs them:

| Arm | Cells | What it shows |
|---|---|---|
| Tetrad 7.6.10 `Fci`, the environment alone in the first tier, `excludeSelectionBias` | S4, L10, X100, P20, P50 | assumption 1 through background knowledge; one test per direction given the whole possible-d-separation set |
| causal-cmd 1.12.0 (tetrad-lib 7.6.3) FCI with the same tier | the same | the same knowledge with the rules for selection bias on, which its FCI cannot turn off |
| Andrey `fci` on the pooled rows, without the environment | S4, L10 | the paper's FCI-pooled |
| Andrey `fci` on the observational rows | S4, L10 | the paper's FCI-obs |
| Andrey `cdnod`, the regimes as domains | S4, L10 | assumes no hidden confounder; its undirected edges count against the PAG's circles |
| Andrey `fci_jci` with the columns in reverse order | L10 | the order the result depends on |
| Andrey `fci_jci` with `indep_test="cond_gauss"` | S4m, L10m | the environment read as categorical |

Tetrad's arms are scored against the truth of assumption 1. Tetrad 7.6.10's orientation rules R9
and R5 treat a covered path as uncovered and can add tails the truth does not have, a defect of
that version that 7.6.3 does not have; the page reports those tails as such. No arm is scored
against a truth Tetrad computes.

## The truth

The truth of each assumption set is FCI-JCI's PAG at the true distribution, over the data
variables and the environment, built as `andrey.data.latent.marginal` builds the plain PAG: the
adjacencies of the true MAG, the observed ancestors as separating sets, FCI-JCI's marks on the
environment's edges before the colliders, and the orientation rules without R5 to R7, the
discriminating-path rule answered by d-separation in the true DAG. Every dataset checks it against
the true MAG and the plain PAG: the same adjacencies, every mark that is not a circle the MAG's,
and every mark the plain PAG (no JCI assumption) sets kept. In the cells of 4 and 10 variables,
every dataset also checks that Andrey's FCI-JCI with d-separation in place of its tests returns the
same PAG, checks the plain PAG against pcalg's `dag2pag`, on the DAG numbered in a topological
order (its ancestor search reads parents among lower numbers only), and checks each truth against
pcalg's `fci` with `jci` and a d-separation oracle. That oracle answers by the moral graph of the
ancestral set and is checked against pcalg's `dsepTest` on 200 random queries per dataset. A
difference in any check stops the comparison.

## Scores

The parent scores every graph with one function, against the truth of the arm's assumptions.

| Metric | Meaning |
|---|---|
| `shd` | the endpoint SHD over the data variables and the environment: pairs whose two marks differ, so a missing or extra edge costs both (judged) |
| `shd_data` | the same over the data variables alone, the only scope of the arms without the environment |
| `adjacency_*`, `arrowhead_*`, `tail_*` | precision, recall, and F1; an arrowhead or a tail is an endpoint, found when the estimate has the same edge with the same mark at that node |
| `pdsep_max` | the largest possible-d-separation set an Andrey FCI fit forms, from an untimed repeat of the fit that must return the same graph |
| `pdsep_max_true` | the largest possible-d-separation set FCI-JCI forms on the true graph, on the dataset's line |
| `wall_s` | the timed fit, from the array in memory to the returned graph; for pcalg the transfer into R, `cor(X)`, and the call |
| empty | the empty graph's `shd` and `shd_data` |

A row is a cell, an assumption set, and a level. Per dataset, D is Andrey's `shd` minus pcalg's;
the row's interval is the percentile bootstrap 95% interval of the mean D over the paired datasets
(10,000 resamples). A row is judged only where pcalg beats the empty graph (its own interval
against the empty graph lies below 0) with at least 10 paired datasets. It is worse when the
interval lies above 0 and better when below, and either counts only if a rerun on seeds 100 to 119
agrees. A row also fails when Andrey timed out on a dataset pcalg finished, or when Andrey does not
beat the empty graph where pcalg does. A timeout is never scored as an empty graph.

## Results

Andrey's pre-release build of 6 November 2026 against pcalg 2.7-12 on R 4.6.1, one fit at a time on
one core of an AMD EPYC 7713 processor with one BLAS thread. Every number below is in
[`published/0.15.0/fci-jci.json`](../published/0.15.0/fci-jci.json), with the context arms and the
coverage of every row. All 200 truths passed their checks, the 140 in the cells of 4 and 10
variables against pcalg's too, and every Andrey fit finished. Refitted on the 0.15.0 release branch
of the same day, Andrey's two gated arms returned the same PAG on all 720 fits.

### In the claim

Means over the 20 datasets of each cell, seeds 0 to 19, of the endpoint SHD over the data variables
and the environment. JCI names the assumption sets, `confounded_index=True` with pcalg's `jci =
"1"` and `False` with `"123"`. D is Andrey's SHD minus pcalg's, with its 95% interval, and Same PAG
counts the datasets where the two returned the same PAG.

| Cell | JCI | alpha | Andrey | pcalg | Empty | D [95%] | Same PAG | Verdict |
|---|---|---|---|---|---|---|---|---|
| S4 | 1 | 0.01 | 5.95 | 5.9 | 12.5 | 0.05 [-0.3, 0.45] | 18 | tie |
| S4 | 123 | 0.01 | 5.9 | 5.8 | 12.5 | 0.1 [0, 0.3] | 19 | tie |
| S4 | 1 | 0.05 | 5.05 | 5.05 | 12.5 | 0 [-0.15, 0.15] | 18 | tie |
| S4 | 123 | 0.05 | 5 | 5.05 | 12.5 | -0.05 [-0.15, 0] | 18 | tie |
| S4p | 1 | 0.01 | 6.05 | 6.15 | 12.5 | -0.1 [-0.25, 0] | 18 | tie |
| S4p | 123 | 0.01 | 6.05 | 6.15 | 12.5 | -0.1 [-0.25, 0] | 18 | tie |
| S4p | 1 | 0.05 | 5.45 | 5.4 | 12.5 | 0.05 [-0.15, 0.3] | 18 | tie |
| S4p | 123 | 0.05 | 5.45 | 5.4 | 12.5 | 0.05 [-0.15, 0.3] | 18 | tie |
| L10 | 1 | 0.01 | 37.85 | 37.8 | 54.8 | 0.05 [-0.9, 0.95] | 4 | tie |
| L10 | 123 | 0.01 | 37.85 | 37.8 | 54.8 | 0.05 [-0.9, 0.95] | 4 | tie |
| L10 | 1 | 0.05 | 36.25 | 36.55 | 54.8 | -0.3 [-1.2, 0.65] | 6 | tie |
| L10 | 123 | 0.05 | 36.05 | 36.45 | 54.8 | -0.4 [-1.3, 0.6] | 6 | tie |

All 12 rows tie, so no row needed a rerun. X100, in the claim where pcalg finishes, has no row: on
the dry run's two datasets there, seeds 0 and 1, every pcalg fit, at both assumption sets and both
levels, ran past the 1,800 s cap, so the campaign did not run pcalg on X100. Andrey's X100 rows are
below, against the empty graph.

### Floors

pcalg's SHD was below the empty graph's in all 12 rows in the claim, so the floor applies in each,
and Andrey's was below it too: the upper end of its interval of SHD minus the empty graph's was
-5.35 or lower at 4 variables and -15.75 or lower at 10. Out of the claim the floor applied in the
18 rows where pcalg finished, and Andrey passed each. The 8 rows where pcalg did not run have no
floor; there Andrey's SHD was below the empty graph's on every dataset.

### Out of the claim

The merged cells, the other row counts, and P20, with pcalg:

| Cell | JCI | alpha | Paired | Andrey | pcalg | Empty | D [95%] | Same PAG |
|---|---|---|---|---|---|---|---|---|
| S4m | 1 | 0.01 | 20 | 7.75 | 8.05 | 14.6 | -0.3 [-0.7, 0.05] | 15 |
| S4m | 123 | 0.01 | 20 | 7.75 | 8.05 | 14.6 | -0.3 [-0.7, 0.05] | 15 |
| S4m | 1 | 0.05 | 20 | 7.95 | 8.1 | 14.6 | -0.15 [-0.3, 0] | 17 |
| S4m | 123 | 0.05 | 20 | 7.95 | 8 | 14.6 | -0.05 [-0.2, 0.1] | 17 |
| L10m | 1 | 0.01 | 20 | 49.1 | 49.3 | 71.5 | -0.2 [-1.55, 0.9] | 2 |
| L10m | 123 | 0.01 | 20 | 46.1 | 46.65 | 71.5 | -0.55 [-1.95, 0.6] | 2 |
| L10m | 1 | 0.05 | 20 | 48.85 | 49.1 | 71.5 | -0.25 [-1.2, 0.7] | 0 |
| L10m | 123 | 0.05 | 20 | 45.1 | 45.85 | 71.5 | -0.75 [-1.75, 0.25] | 0 |
| L10n100 | 1 | 0.01 | 20 | 44.65 | 45.1 | 54.8 | -0.45 [-1.2, 0.25] | 4 |
| L10n100 | 123 | 0.01 | 20 | 44.6 | 45.1 | 54.8 | -0.5 [-1.25, 0.15] | 4 |
| L10n100 | 1 | 0.05 | 20 | 43.1 | 43.35 | 54.8 | -0.25 [-1.2, 0.8] | 2 |
| L10n100 | 123 | 0.05 | 20 | 43.05 | 43.3 | 54.8 | -0.25 [-1.2, 0.8] | 2 |
| L10n2000 | 1 | 0.01 | 20 | 33.9 | 33.95 | 54.8 | -0.05 [-0.9, 0.75] | 5 |
| L10n2000 | 123 | 0.01 | 20 | 33.75 | 33.8 | 54.8 | -0.05 [-0.9, 0.75] | 5 |
| L10n2000 | 1 | 0.05 | 20 | 32.95 | 32.55 | 54.8 | 0.4 [-0.45, 1.25] | 4 |
| L10n2000 | 123 | 0.05 | 20 | 32.7 | 32.35 | 54.8 | 0.35 [-0.45, 1.15] | 4 |
| P20 | 1 | 0.01 | 4 | 54.75 | 59.25 | 87.5 | -4.5 [-8, -1] | 1 |
| P20 | 123 | 0.01 | 6 | 60.83 | 65.33 | 95.67 | -4.5 [-7, -2] | 1 |

No row with 20 paired datasets lies above or below 0. In P20, pcalg finished 4 and 6 of the 10
datasets within the cap, and Andrey all 10; on the datasets both finished, Andrey had fewer errors,
on too few pairs to judge.

Where pcalg did not run, Andrey's mean SHD against the empty graph's:

| Cell | JCI | alpha | Andrey | Empty | Andrey minus empty [95%] |
|---|---|---|---|---|---|
| X100 | 1 | 0.01 | 328.15 | 412.4 | -84.25 [-88.6, -79.85] |
| X100 | 123 | 0.01 | 328.1 | 412.4 | -84.3 [-88.65, -79.9] |
| X100 | 1 | 0.05 | 348.15 | 412.4 | -64.25 [-69.8, -58.25] |
| X100 | 123 | 0.05 | 347.7 | 412.4 | -64.7 [-70.3, -58.7] |
| X100m | 1 | 0.01 | 230.25 | 431 | -200.75 [-206.8, -195.05] |
| X100m | 123 | 0.01 | 228.45 | 431 | -202.55 [-208.65, -196.75] |
| P50 | 1 | 0.01 | 146.6 | 246.2 | -99.6 [-105.6, -94] |
| P50 | 123 | 0.01 | 146.5 | 246.2 | -99.7 [-105.7, -94.1] |

### Context

The arms that are not judged, as mean SHD over the data variables (`shd_data`) minus that of Andrey's
default arm on the same datasets, with its 95% interval, at alpha 0.01:

| Arm | S4 | L10 |
|---|---|---|
| Tetrad 7.6.10, the environment in the first tier | 0 [0, 0] | -0.1 [-0.9, 0.65] |
| causal-cmd 1.12.0, the same tier | 0 [0, 0] | -0.5 [-1.2, 0.15] |
| Andrey `fci` on the pooled rows | 0.45 [0, 0.85] | 0.75 [0.1, 1.4] |
| Andrey `fci` on the observational rows | 0.55 [-0.1, 1.25] | 3.4 [2.4, 4.35] |
| Andrey `cdnod`, the regimes as domains | 3.25 [2.25, 4.3] | 4.35 [2.9, 5.85] |
| Andrey `fci_jci`, the columns in reverse order | | 0 [0, 0] |

At L10, FCI on the pooled rows or on the observational rows alone made more errors over the data
variables than FCI-JCI, and so did CD-NOD, which assumes no hidden common cause. Reversing the
column order returned the same PAG on every dataset of L10, at both levels. In the merged cells at
alpha 0.01, reading the regime number as a category (`indep_test="cond_gauss"`) made fewer errors
over the data and the environment, 6.8 against 7.75 at S4m and 44.5 against 49.1 at L10m, while over
the data variables alone the difference's interval contains 0 (D 0.05 [-0.65, 0.85] and 1.05 [-0.15,
2.1]).

### Speed

pcalg's time is the transfer into R, `cor(X)`, and the call; Andrey's runs from the array in memory
to the returned graph. In the claim, Andrey's median fit took 0.004 to 0.005 s at 4 variables and
0.011 to 0.013 s at 10, pcalg's 0.017 to 0.021 s and 0.83 to 1.31 s: per row, the median of pcalg's
time over Andrey's was 4.0 to 4.6 at 4 variables and 72 to 79 at 10. Andrey's slowest fit in the
campaign took 0.48 s, at X100m. The largest possible-d-separation set Andrey's fits formed had 9
nodes; on the true graphs FCI-JCI forms sets of up to 30 nodes at 100 variables and 35 at P50.

## Run

```shell
benchmarks/start.sh
benchmarks/start_r.sh
export ANDREY_BENCH_TETRAD_GIN_JAR=$PWD/tetrad-lib-7.6.10-shaded.jar
export ANDREY_BENCH_TETRAD_JAR=$PWD/causal-cmd-1.12.0-jar-with-dependencies.jar
$ANDREY_BENCH_PYTHON benchmarks/fci_jci_cpu.py run --out outputs/fci-jci-S4 --cells S4 \
    --seeds 0-19 --check-pcalg
$ANDREY_BENCH_PYTHON benchmarks/fci_jci_cpu.py summarize --runs outputs/fci-jci-S4 \
    --out outputs/fci-jci-summary.json
```

`run` draws each dataset, builds its truth, and fits every arm of the cell on it, each fit in a
fresh interpreter on one core with one BLAS thread, after one untimed warm-up fit; pcalg takes the
warm-up only in the cells of 4 and 10 variables, where it cannot double a fit that may reach the
cap. A fit past its cap, 1,800 s with imports and warm-up, is a timeout.
`--check-pcalg` adds pcalg's checks of the truth; `--resume` continues a run; `--arms`, `--alphas`,
and `--seeds` take part of a cell. `summarize` pairs the records of one or more runs into the rows
above, and `--replications` adds each row's rerun.

## References

- Mooij, J. M., Magliacane, S., Claassen, T. (2020). Joint causal inference from multiple
  contexts. JMLR 21(99), 1-108.
- Spirtes, P., Glymour, C., Scheines, R. (2000). Causation, Prediction, and Search, 2nd ed. MIT
  Press.
- Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence
  of latent confounders and selection bias. Artificial Intelligence 172(16-17), 1873-1896.
- Kalisch, M., Machler, M., Colombo, D., Maathuis, M. H., Buhlmann, P. (2012). Causal inference
  using graphical models with the R package pcalg. Journal of Statistical Software 47(11), 1-26.
