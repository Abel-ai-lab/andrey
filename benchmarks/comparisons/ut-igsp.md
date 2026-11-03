# UT-IGSP

`ut_igsp_cpu.py` compares Andrey's `ut_igsp` with `unknown_target_igsp`, the authors' library
(graphical-model-learning 0.1a8 with conditional-independence 0.1a6 and graphical-models 0.1a21),
on data that mixes an observational environment with interventional settings whose targets are
known in part or not at all. The library's IGSP and JCI-GSP, Andrey's `gies` with the designed
targets, and Andrey's `cdnod` run beside them as context: reported, never judged.

## Data

`andrey.data` draws each dataset in the setting of Squires, Wang, and Uhler (2020, Section 5.1): an
Erdos-Renyi DAG on 20 variables with mean degree 1.5 (exactly 15 edges, where the authors'
generator draws each pair with probability 1.5/19), weights uniform on `[0.25, 1]` in absolute
value with a random sign, standard normal noise in the generator's own units (`scale="raw"`), and
its columns in an order drawn from the seed. An observational environment and 5 settings have `n`
rows each. Each setting intervenes on one designed target, which the fit is told, and on `l`
off-targets, which it is not. The designed targets and the DAG depend on the seed alone, so the
cells of a block share them, and the cells at one seed and `n` share their observational rows.

| Kind | Intervention on each target | Source |
|---|---|---|
| `shift` | noise mean moved by 1, `soft(shift=1.0)` | the paper's text |
| `perfect` | parents cut, `N(1, 0.1)`, `hard(1.0, sd=0.1**0.5)` | the paper's text (variance 0.1) |
| `shift2` | noise mean moved by 2, `soft(shift=2.0)` | the authors' code |
| `perfect_code` | parents cut, `N(1, 0.01)`, `hard(1.0, sd=0.1)` | the authors' code (variance 0.01) |
| `variance` | noise standard deviation doubled, `soft(noise_scale=2.0)` | claimed, not reported |
| `weight` | parent weights halved, `soft(weight_scale=0.5)`, targets among variables with parents | claimed, not reported |

## Cells

In claim, where the paper claims the method works (perfect and imperfect interventions, targets
known in part or not at all) at the model class and regime it reports (linear Gaussian, 20
variables, 5 settings, at least 1,000 rows per setting), 20 datasets per cell, 5 draws per dataset:

| Arm | Kind | `l` | `n` per setting |
|---|---|---|---|
| G1 | `shift` | 0-3 | 1,000-5,000 |
| G2 | `perfect` | 0-3 | 1,000-5,000 |
| G3 | `shift2`, `perfect_code` | 0, 3 | 1,000 |
| G4 | `variance` | 0, 3 | 1,000, 5,000 |
| G5 | `shift`, G1's datasets with no targets given | 0, 3 | 1,000, 5,000 |
| G6 | `weight` | 0, 3 | 1,000, 5,000 |

Out of claim, measured for the method's limitations, one draw per dataset: small `n` (O1), weak
shifts of 0.25 and 0.5 (O2), a dense graph on 50 variables with 3 settings (O3), 50 and 100
variables (O4), additive-noise data with the Gaussian tests (O5), additive-noise data on 10
variables with kernel tests (O6), categorical data with the G-test, Andrey alone (O7), 1 and 2
settings (O8), and 2,000 observational rows with settings of 300 (O9).

Block A, on seeds of its own, chooses Andrey's defaults before the comparison: Andrey alone, `shift`
and `perfect` at `l` 0 and 3 and `n` 1,000, 3,000, and 5,000, with three out-of-claim cells
(`n` 300; 50 variables).

## Settings

Both packages test at alpha `1e-5`, the paper's level, with the Gaussian tests. Andrey runs 5
starts; the library runs its defaults: depth 4, 5 runs, the lexicographic objective. A draw seeds
Andrey's `seed` or the library's Python `random` and NumPy global generator with the draw number,
`0` to `4`. Where Andrey's shipped defaults differ from alpha `1e-5` and 5 starts, Andrey also runs
at them, and the library runs at Andrey's alpha as context.

JCI-GSP runs as a copy of the 0.1a8 `jci_gsp` with two call fixes: the shipped function passes
tuple pairs as fixed adjacencies, which the library's minimal-I-MAP builder rejects, and unpacks two
values from a search that returns one. The copy hands the builder frozensets and asks the search
for its summaries. Its tester is the authors' combined Gaussian test. In O6, the library's kernel
tests at their defaults regress on the conditioning set with a GAM and test the residuals with
HSIC: GAM-residual HSIC, not a conditional KCI.

Where Andrey's tests differ from the library's:

| | Andrey | Library |
|---|---|---|
| CI test | two-sided Fisher z at alpha | the start: a precision-matrix threshold; removals: two-sided at alpha |
| Coefficients and intercept | Chow F, upper tail at alpha / 2 | a Wald-type statistic with separate residual variances on `F(q, n1 + n2 - q)`, two-sided at alpha / 4 per tail, so close agreement also rejects |
| Residual variance | F on `(n_k - q_k, n_0 - q_0)`, two-sided at alpha / 2 | F on `(n1 - 1, n2 - 1)`, two-sided at alpha / 4 per tail |
| Objective | edges plus targets (the sum) | edges, then targets that test variant (lexicographic) |
| Returned targets | the known ones and the learned ones | the learned ones; scored with the known ones added, as the authors score them |
| Search | a depth-4 search that backtracks | the same for UT-IGSP; the library's IGSP does not backtrack at a dead end |

## Scores

The parent scores every package with one function. Each estimate is the interventional essential
graph of the returned graph under the targets the method returned or was given; the truth is that
of the generating DAG under the true targets.

| Metric | Meaning |
|---|---|
| `shd` | variable pairs whose endpoint marks differ from the truth's (judged) |
| `target_error` | false plus missed targets over the settings (judged) |
| `target_floor` | `target_error` of the given targets alone |
| `shd_true_targets`, `in_true_imec` | the estimate's DAG re-read under the true targets, and whether it is in the true class |
| `skeleton_shd` | adjacency differences |
| `empty_shd` | the truth's adjacent pairs: the empty graph's `shd` |
| `wall_s` | the timed fit: the tests' construction and the search |

A dataset's error per package is the mean over its draws. A cell is worse when the 95% percentile
bootstrap interval of the mean paired difference, Andrey minus the library over 20 datasets, lies
above 0, and counts only if a rerun on 20 fresh datasets is worse too. In claim, a cell also fails
when Andrey timed out on a dataset the library finished, when Andrey's `shd` is not below the empty
graph's, or when Andrey's targets are not better than the given targets alone where those miss
some. `qa/reference_parity.py` checks the published summary.

## Run

```shell
benchmarks/start_utigsp.sh
export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
$ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py check-truth --out outputs/utigsp-truth
$ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py run --arm G1 --out outputs/utigsp-G1 --workers 4
$ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py summarize --runs outputs/utigsp-G1 \
    --out outputs/utigsp-summary.json
```

`check-truth` compares the scorer's essential graph with graphical-models' on 100 random DAGs and
target families. `run` takes one arm, optionally some of its cells and seeds, and fits each
dataset's packages back to back, one child process per package, each pinned to one core with one
BLAS thread; `--resume` continues a run. A draw over its cap, 1,800 s (3,600 s at 100 variables
and in O6), is a timeout, never scored. `select --runs ...` applies block A's rule.

## Results

No campaign has been published.

## References

- Squires, C., Wang, Y., Uhler, C. (2020). Permutation-based causal structure learning with
  unknown intervention targets. UAI, PMLR 124, 1039-1048.
- Wang, Y., Solus, L., Yang, K., Uhler, C. (2017). Permutation-based causal inference algorithms
  with interventions. NeurIPS 30.
- Mooij, J. M., Magliacane, S., Claassen, T. (2020). Joint causal inference from multiple
  contexts. JMLR 21(99), 1-108.
