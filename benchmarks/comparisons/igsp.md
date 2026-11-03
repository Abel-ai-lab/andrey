# IGSP

`igsp_cpu.py` compares Andrey's `igsp` with `igsp` from the authors' library
(graphical-model-learning 0.1a8 with conditional-independence 0.1a6 and graphical-models 0.1a21) on
data that mixes an observational environment with interventional settings whose targets are known.
Where a setting has two or more targets, Andrey is also compared with a copy of the library whose
I-contradicting test follows Definition 4.2 of Yang, Katcoff, and Uhler (2018). Andrey's `gies`,
Andrey's `grasp` on the observational rows, and UT-IGSP with every target known (Andrey's and the
library's) run beside them as context: reported, never judged.

## Data

`andrey.data` draws every dataset in the generator's own units (`scale="raw"`), the papers'
intervention values being raw units, with its columns in an order drawn from the seed. An
observational environment and every setting have `n` rows each. The targets are column positions
drawn from the seed.

| Block | Source | Graph and weights | Settings | `n` per setting |
|---|---|---|---|---|
| W | Wang et al. (2017), Figure 2 | Erdos-Renyi, 10 or 20 variables, mean degree 1.5; `[0.1, 1]` | `K = 1` setting of 5 (10) targets, or 2 disjoint settings of 4 (8) | 1e3, 1e4, 1e5 |
| S | Wang et al. (2017), Figure 3 | their Figure 1 (left), 7 variables; `[c, 1]`, `c` 0.1, 0.25, 0.5 | one setting on `{4}`, `{5}`, or `{4, 5}` | 1e3, 1e4, 1e5 |
| Y6 | Yang et al. (2018), Figure 6 | Erdos-Renyi, 10 variables, mean degree 1.5; `[0.25, 1]` | `k` single-target settings, `k` 1-10 | 1e3, 1e4 |
| Y5 | Yang et al. (2018), Figure 5 | the same on 20 variables | every variable | 1e3, 1e4 |
| Y8 | Yang et al. (2018), Figure 8 | as Y6 | every pair of variables | 1e3, 1e4 |

Weights are uniform in absolute value with a random sign, noise is standard normal. The graph
generator draws exactly `round(0.75 p)` edges, where the authors' draws each pair with probability
`1.5 / (p - 1)`. In W and S each target is cut from its causes and given the noise `N(0, 0.2)`
(`hard(0.0, sd=0.2 ** 0.5)`). Figure 2's caption names single-node interventions; the paper's text
and the authors' data script use the target sets of the table, which decide. Block S takes the
paper's arcs (`1 -> 4`, `2 -> 4`, `3 -> 4`, `3 -> 5`, `7 -> 5`, `4 -> 6`) and the families its text
names; the authors' script swaps the labels 5 and 7. Block Y has three kinds of intervention:

| Kind | Intervention on each target |
|---|---|
| `perfect` | cut from its causes, noise `N(0, 1)` kept, `hard(0.0, sd=1.0)` |
| `inhibiting` | incoming weights divided by 10, `soft(weight_scale=0.1)` |
| `imperfect` | `perfect` on each row with probability 1/2, each target independently |

Yang et al. state neither `n` nor alpha for Block Y: both are our choices.

Under Yang's kinds, a target without causes keeps its observational distribution: the
intervention changes nothing there, so the target is null. Each setting's effective targets are
its targets less the null ones, and the effective targets define the truth every package is scored
against.

## Cells

In claim, where the papers claim the method works (linear Gaussian data, perfect, inhibiting, and
imperfect interventions with known targets, at their sizes): the 12 cells of W, 27 of S (one per
family, `c`, and `n`), and 48 of Y, 20 datasets per cell, 5 draws per dataset, at alpha `1e-4`,
`1e-3`, and `1e-2` (inside the authors' sweep), and at Andrey's default when it is a fourth value.

Out of claim, measured for the method's limitations, one draw per dataset: kernel tests on
additive-noise data (K), 50 to 400 variables (O1), 100 and 300 rows per setting (O2), mean degree 3
and 4 (O3), an unknown off-target in each setting (O4), additive-noise data with the Gaussian tests
(O5), categorical data with the G-test (O7), Laplace and uniform noise (O8), and an observational
environment 3 and 10 times larger than each setting (O9).

Andrey's defaults are chosen before the comparison, on seeds of their own, from alpha `1e-4`,
`5e-4`, `1e-3`, `5e-3`, `1e-2`, `5e-2` and from 1, 5, and 10 starts, over every in-claim cell, and
checked on a family outside the grid (perfect interventions on 8 of 20 variables, mean degree 3,
weights on `[0.5, 2]`).

## Settings

Both packages run their Gaussian tests at the same alpha for the CI and the invariance tests. The
library runs its defaults: depth 4, 5 runs from its threshold graph. A draw seeds Andrey's `seed`
or the library's Python `random` and NumPy global generator with the draw number, `0` to `4`.

Where Andrey follows the paper and the library does not:

| | Andrey | Library |
|---|---|---|
| I-contradicting arcs counted | every arc | only arcs its I-covered test passes |
| Visiting order | reversals of I-contradicting arcs first | every child in a random order |
| Start graph | minimal I-MAP by two-sided Fisher z at alpha | a precision-matrix threshold at one-sided alpha, so two-sided `2 alpha` |
| Search | a depth-4 search that backtracks at a dead end | at a dead end, the search does not return to the parent's other children |
| Multi-target settings | Definition 4.2 | Definition 4.2's conditions with their quantifiers inverted, so an arc out of a target is reversed |

Where the two differ by design:

| | Andrey | Library |
|---|---|---|
| Invariance test | Chow F for coefficients and intercept, upper tail at alpha / 2, and the residual-variance F, two-sided at alpha / 4 per tail | a Wald-type coefficient statistic with separate residual variances, two-sided at alpha / 4 per tail, so close agreement also rejects; the residual-variance F on `(n1 - 1, n2 - 1)` |
| Restarts | the restart-eligible child with the lowest objective | a random sparser child |
| Result of a run | the lowest objective among every DAG it scored | among the DAGs it entered |
| Settings with one target set | each its own setting | collapsed to the last |

The patched copy changes only the library's general branch of the I-contradicting test
(`patches/gml-igsp-def42.patch`): condition 1 holds when some conditioning set leaves the head
unchanged in every setting that targets the tail and not the head, and condition 2 when, for every
conditioning set, some setting that targets the head and not the tail changes the tail. Before any
fit, `validate-patch` checks it with every test answered by d-separation on 200 random DAGs and
target families: no arc of the true DAG is I-contradicting, every covered arc whose reversal
changes the class is I-contradicting in the reversed DAG, and on `0 -> 1` with the setting `{0, 2}`
it returns `0 -> 1` on 20 of 20 seeds.

## Scores

The parent scores every package with one function. Each package's output is reduced to a DAG, a
returned DAG as it is and an interventional essential graph through one of its consistent
extensions, and read as its interventional essential graph under the effective targets. The truth
is the generating DAG's under the same targets. `grasp`'s graph is scored as it is returned.

| Metric | Meaning |
|---|---|
| `shd` | variable pairs whose endpoint marks differ from the truth's (judged) |
| `shd_nominal` | the same under the targets as drawn |
| `skeleton_shd`, `orientation_errors` | adjacency differences, and mark differences among shared adjacencies |
| `iess_exact` | whether `shd` is 0 |
| `null_reversed`, `null_kept` | true arcs out of a null target the estimate directs against, or along, the truth |
| `dag_hamming` | pairs where a returned DAG differs from the true DAG |
| `empty_shd` | the truth's adjacent pairs: the empty graph's `shd` |
| `wall_s` | the timed fit: from the arrays in memory to the returned graph |

A dataset's error per package is the mean over its draws. A row is judged only where the library
beats the empty graph (the 95% percentile bootstrap interval of its mean paired difference with the
empty graph lies below 0); the others are reported. A judged row is worse when the interval of the
mean paired difference, Andrey minus the library, lies above 0, and counts only if a rerun on 20
fresh datasets is worse too. In claim, a row also fails when Andrey timed out on a dataset the
library finished, or when Andrey does not beat the empty graph. `qa/reference_parity.py` checks the
published summary.

IGSP's I-contradicting test marks a true arc out of a null target, so the search is pushed to
reverse it; GIES drops the target's own term in that setting and orients the arc from the same
inputs. `null_reversed` reports the effect.

## Run

```shell
benchmarks/start_utigsp.sh
export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
export ANDREY_BENCH_GML_DEF42_PYTHON=$PWD/benchmarks/.venv-gml-def42/bin/python
$ANDREY_BENCH_PYTHON benchmarks/igsp_cpu.py validate-patch --out outputs/igsp-patch
$ANDREY_BENCH_PYTHON benchmarks/igsp_cpu.py run --arm W --out outputs/igsp-W --workers 4
$ANDREY_BENCH_PYTHON benchmarks/igsp_cpu.py summarize --runs outputs/igsp-W \
    --out outputs/igsp-summary.json
```

`run` takes one arm, optionally some of its cells and seeds, and fits each dataset's packages back
to back, one child process per package, each pinned to one core with one BLAS thread; `--resume`
continues a run. A draw over its cap, 1,800 s in claim and 3,600 s out of it, is a timeout, never
scored. `run --block A` fits the in-claim cells on the seeds that choose Andrey's defaults, and
`select` applies the rule; `run --block C` reruns a cell on fresh seeds, and `summarize
--replications` adds those reruns to their rows.

## Results

No campaign has been published.

## References

- Wang, Y., Solus, L., Yang, K. D., Uhler, C. (2017). Permutation-based causal inference
  algorithms with interventions. NeurIPS 30.
- Yang, K. D., Katcoff, A., Uhler, C. (2018). Characterizing and learning equivalence classes of
  causal DAGs under interventions. ICML, PMLR 80, 5541-5550.
- Hauser, A., Buhlmann, P. (2012). Characterization and greedy learning of interventional Markov
  equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
- Squires, C., Wang, Y., Uhler, C. (2020). Permutation-based causal structure learning with
  unknown intervention targets. UAI, PMLR 124, 1039-1048.
