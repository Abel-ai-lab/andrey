# FCI-JCI

`fci_jci_cpu.py` compares Andrey's `fci_jci` with `fci` from pcalg 2.7-12, the implementation of
FCI-JCI by the method's authors (Mooij, Magliacane, and Claassen 2020), on their simulation of data
pooled from an observational regime and interventional regimes, with hidden confounders. Tetrad's
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
