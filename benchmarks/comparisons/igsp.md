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
library runs its defaults: depth 4, 5 runs from its threshold graph. Andrey runs at its default of
10 starts and at 5, the library's number of runs; both are compared with the same library fits. A
draw seeds Andrey's `seed` or the library's Python `random` and NumPy global generator with the
draw number, `0` to `4`.

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

Andrey's pre-release build of 3 November 2026 against graphical-model-learning 0.1a8 and the
patched copy, every fit on one core of an AMD EPYC 7713 processor with one BLAS thread, the
packages of a dataset back to back on the same core. Every number below is in
[`published/0.14.0/igsp.json`](../published/0.14.0/igsp.json), with the context arms and every
row of every cell. In the tables, `p` is the number of variables, `n` the rows per setting, and D
Andrey's SHD minus the comparator's, the mean over the 20 datasets of a cell with its 95%
interval; each dataset's SHD is the mean over its 5 draws.

### In the claim

At its defaults, alpha `1e-3` and 10 starts, Andrey is worse than the library in none of the 114
comparisons (87 cells against the library, and the 27 cells with a setting of two or more targets
against the patched copy) and better in 11: its errors were lower on seeds 0 to 19 and again on a
rerun on seeds 6000 to 6019. All 11 are against the unpatched library, in cells with settings of
two or more targets (10 of W's 12 and Y8 `inhibiting` at n 10,000), where the library's test
inverts Definition 4.2; against the patched copy the same 11 cells tie. Seeds 6000 to 6019 are
those of the run that confirmed the default (see [Choosing the defaults](#choosing-the-defaults)),
which chose it as the candidate with the most comparisons below 0 there, 18 against 17, so they
favor below 0 for it. Another 4 were lower on seeds 0 to 19 and tied on the rerun, and the other
99 tie. Against the patched copy alone, Andrey is worse in none and better in none of 27.

With 5 starts, the library's number of runs, at the same level, Andrey is worse in none of the 114
comparisons: 13 were lower on seeds 0 to 19, with no rerun, and the other 101 tie.

At the default, 10 starts, seeds 0 to 19. The patched copy's columns are filled in the cells with
a setting of two or more targets.

| Cell | Empty graph | Andrey | Reference | D [95%] | Verdict | Patched copy | D [95%] | Verdict |
|---|---|---|---|---|---|---|---|---|
| W, K 1, p 10, n 1,000 | 8.00 | 2.37 | 3.05 | -0.68 [-1.60, +0.40] | tie | 2.33 | +0.04 [-0.59, +0.94] | tie |
| W, K 1, p 10, n 10,000 | 8.00 | 0.40 | 1.93 | -1.53 [-2.20, -0.88] | better | 0.31 | +0.09 [-0.10, +0.33] | tie |
| W, K 1, p 10, n 100,000 | 8.00 | 0.05 | 1.64 | -1.59 [-2.18, -1.04] | better | 0.26 | -0.21 [-0.47, 0.00] | tie |
| W, K 1, p 20, n 1,000 | 15.00 | 3.33 | 5.21 | -1.88 [-2.71, -1.07] | better | 3.24 | +0.09 [-0.46, +0.68] | tie |
| W, K 1, p 20, n 10,000 | 15.00 | 0.35 | 3.09 | -2.74 [-3.52, -2.00] | better | 0.41 | -0.06 [-0.16, 0.00] | tie |
| W, K 1, p 20, n 100,000 | 15.00 | 0.44 | 3.05 | -2.61 [-3.37, -1.90] | better | 0.57 | -0.13 [-0.27, -0.01] | lower on seeds 0-19 only |
| W, K 2, p 10, n 1,000 | 8.00 | 2.33 | 3.04 | -0.71 [-1.50, +0.29] | tie | 2.11 | +0.22 [-0.28, +1.02] | tie |
| W, K 2, p 10, n 10,000 | 8.00 | 0.25 | 2.02 | -1.77 [-2.38, -1.16] | better | 0.29 | -0.04 [-0.18, +0.08] | tie |
| W, K 2, p 10, n 100,000 | 8.00 | 0.05 | 1.89 | -1.84 [-2.32, -1.36] | better | 0.21 | -0.16 [-0.41, 0.00] | tie |
| W, K 2, p 20, n 1,000 | 15.00 | 3.01 | 4.59 | -1.58 [-2.22, -0.79] | better | 3.13 | -0.12 [-0.73, +0.53] | tie |
| W, K 2, p 20, n 10,000 | 15.00 | 0.75 | 2.59 | -1.84 [-2.60, -1.13] | better | 0.67 | +0.08 [-0.09, +0.28] | tie |
| W, K 2, p 20, n 100,000 | 15.00 | 0.45 | 2.85 | -2.40 [-3.08, -1.74] | better | 0.59 | -0.14 [-0.29, -0.01] | lower on seeds 0-19 only |
| S {4}, c 0.1, n 1,000 | 6.00 | 1.05 | 1.02 | +0.03 [-0.10, +0.17] | tie |  |  |  |
| S {4}, c 0.1, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.1, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.25, n 1,000 | 6.00 | 0.35 | 0.33 | +0.02 [0.00, +0.06] | tie |  |  |  |
| S {4}, c 0.25, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.25, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.5, n 1,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.5, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.5, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.1, n 1,000 | 6.00 | 0.85 | 0.83 | +0.02 [-0.03, +0.09] | tie |  |  |  |
| S {5}, c 0.1, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.1, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.25, n 1,000 | 6.00 | 0.20 | 0.16 | +0.04 [0.00, +0.12] | tie |  |  |  |
| S {5}, c 0.25, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.25, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.5, n 1,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.5, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.5, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4, 5}, c 0.1, n 1,000 | 6.00 | 0.75 | 0.97 | -0.22 [-0.44, -0.05] | lower on seeds 0-19 only | 0.74 | +0.01 [-0.03, +0.06] | tie |
| S {4, 5}, c 0.1, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.1, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.25, n 1,000 | 6.00 | 0.20 | 0.23 | -0.03 [-0.09, 0.00] | tie | 0.16 | +0.04 [0.00, +0.12] | tie |
| S {4, 5}, c 0.25, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.25, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.5, n 1,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.5, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.5, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| Y6 `perfect`, k 1, n 1,000 | 8.00 | 0.70 | 0.75 | -0.05 [-0.29, +0.16] | tie |  |  |  |
| Y6 `perfect`, k 1, n 10,000 | 8.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| Y6 `perfect`, k 2, n 1,000 | 8.00 | 0.70 | 0.75 | -0.05 [-0.31, +0.20] | tie |  |  |  |
| Y6 `perfect`, k 2, n 10,000 | 8.00 | 0.05 | 0.01 | +0.04 [0.00, +0.12] | tie |  |  |  |
| Y6 `perfect`, k 4, n 1,000 | 8.00 | 0.90 | 1.20 | -0.30 [-0.72, +0.13] | tie |  |  |  |
| Y6 `perfect`, k 4, n 10,000 | 8.00 | 0.25 | 0.38 | -0.13 [-0.35, +0.06] | tie |  |  |  |
| Y6 `perfect`, k 6, n 1,000 | 8.00 | 0.74 | 0.89 | -0.15 [-0.42, +0.13] | tie |  |  |  |
| Y6 `perfect`, k 6, n 10,000 | 8.00 | 0.20 | 0.22 | -0.02 [-0.16, +0.10] | tie |  |  |  |
| Y6 `perfect`, k 8, n 1,000 | 8.00 | 1.26 | 1.18 | +0.08 [-0.27, +0.42] | tie |  |  |  |
| Y6 `perfect`, k 8, n 10,000 | 8.00 | 0.50 | 0.54 | -0.04 [-0.30, +0.24] | tie |  |  |  |
| Y6 `perfect`, k 10, n 1,000 | 8.00 | 1.39 | 1.42 | -0.03 [-0.34, +0.27] | tie |  |  |  |
| Y6 `perfect`, k 10, n 10,000 | 8.00 | 1.03 | 0.93 | +0.10 [-0.12, +0.32] | tie |  |  |  |
| Y6 `inhibiting`, k 1, n 1,000 | 8.00 | 0.60 | 0.75 | -0.15 [-0.42, +0.10] | tie |  |  |  |
| Y6 `inhibiting`, k 1, n 10,000 | 8.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| Y6 `inhibiting`, k 2, n 1,000 | 8.00 | 0.62 | 0.73 | -0.11 [-0.32, +0.04] | tie |  |  |  |
| Y6 `inhibiting`, k 2, n 10,000 | 8.00 | 0.00 | 0.01 | -0.01 [-0.03, 0.00] | tie |  |  |  |
| Y6 `inhibiting`, k 4, n 1,000 | 8.00 | 0.93 | 1.20 | -0.27 [-0.61, +0.01] | tie |  |  |  |
| Y6 `inhibiting`, k 4, n 10,000 | 8.00 | 0.20 | 0.38 | -0.18 [-0.39, 0.00] | tie |  |  |  |
| Y6 `inhibiting`, k 6, n 1,000 | 8.00 | 0.78 | 0.93 | -0.15 [-0.39, +0.06] | tie |  |  |  |
| Y6 `inhibiting`, k 6, n 10,000 | 8.00 | 0.20 | 0.22 | -0.02 [-0.16, +0.13] | tie |  |  |  |
| Y6 `inhibiting`, k 8, n 1,000 | 8.00 | 0.91 | 1.13 | -0.22 [-0.50, +0.06] | tie |  |  |  |
| Y6 `inhibiting`, k 8, n 10,000 | 8.00 | 0.45 | 0.52 | -0.07 [-0.31, +0.20] | tie |  |  |  |
| Y6 `inhibiting`, k 10, n 1,000 | 8.00 | 1.37 | 1.41 | -0.04 [-0.34, +0.26] | tie |  |  |  |
| Y6 `inhibiting`, k 10, n 10,000 | 8.00 | 0.80 | 0.94 | -0.14 [-0.45, +0.15] | tie |  |  |  |
| Y6 `imperfect`, k 1, n 1,000 | 8.00 | 0.70 | 0.75 | -0.05 [-0.29, +0.16] | tie |  |  |  |
| Y6 `imperfect`, k 1, n 10,000 | 8.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| Y6 `imperfect`, k 2, n 1,000 | 8.00 | 0.72 | 0.75 | -0.03 [-0.28, +0.21] | tie |  |  |  |
| Y6 `imperfect`, k 2, n 10,000 | 8.00 | 0.02 | 0.01 | +0.01 [0.00, +0.03] | tie |  |  |  |
| Y6 `imperfect`, k 4, n 1,000 | 8.00 | 0.85 | 1.08 | -0.23 [-0.50, +0.01] | tie |  |  |  |
| Y6 `imperfect`, k 4, n 10,000 | 8.00 | 0.45 | 0.47 | -0.02 [-0.27, +0.24] | tie |  |  |  |
| Y6 `imperfect`, k 6, n 1,000 | 8.00 | 0.99 | 1.05 | -0.06 [-0.35, +0.25] | tie |  |  |  |
| Y6 `imperfect`, k 6, n 10,000 | 8.00 | 0.20 | 0.21 | -0.01 [-0.13, +0.11] | tie |  |  |  |
| Y6 `imperfect`, k 8, n 1,000 | 8.00 | 1.22 | 1.29 | -0.07 [-0.37, +0.24] | tie |  |  |  |
| Y6 `imperfect`, k 8, n 10,000 | 8.00 | 0.85 | 0.63 | +0.22 [-0.03, +0.51] | tie |  |  |  |
| Y6 `imperfect`, k 10, n 1,000 | 8.00 | 1.62 | 1.45 | +0.17 [-0.24, +0.61] | tie |  |  |  |
| Y6 `imperfect`, k 10, n 10,000 | 8.00 | 0.90 | 1.00 | -0.10 [-0.41, +0.19] | tie |  |  |  |
| Y5 `perfect`, n 1,000 | 15.00 | 2.04 | 2.03 | +0.01 [-0.32, +0.34] | tie |  |  |  |
| Y5 `perfect`, n 10,000 | 15.00 | 1.90 | 1.94 | -0.04 [-0.60, +0.49] | tie |  |  |  |
| Y5 `inhibiting`, n 1,000 | 15.00 | 2.10 | 2.20 | -0.10 [-0.66, +0.51] | tie |  |  |  |
| Y5 `inhibiting`, n 10,000 | 15.00 | 1.90 | 1.94 | -0.04 [-0.58, +0.44] | tie |  |  |  |
| Y5 `imperfect`, n 1,000 | 15.00 | 2.06 | 2.04 | +0.02 [-0.25, +0.29] | tie |  |  |  |
| Y5 `imperfect`, n 10,000 | 15.00 | 1.99 | 1.94 | +0.05 [-0.37, +0.50] | tie |  |  |  |
| Y8 `perfect`, n 1,000 | 8.00 | 1.53 | 1.55 | -0.02 [-0.34, +0.29] | tie | 1.40 | +0.13 [-0.20, +0.46] | tie |
| Y8 `perfect`, n 10,000 | 8.00 | 0.90 | 1.37 | -0.47 [-1.04, +0.01] | tie | 0.82 | +0.08 [-0.17, +0.34] | tie |
| Y8 `inhibiting`, n 1,000 | 8.00 | 1.23 | 1.55 | -0.32 [-0.63, -0.02] | lower on seeds 0-19 only | 1.39 | -0.16 [-0.44, +0.10] | tie |
| Y8 `inhibiting`, n 10,000 | 8.00 | 0.60 | 1.37 | -0.77 [-1.26, -0.34] | better | 0.82 | -0.22 [-0.45, +0.02] | tie |
| Y8 `imperfect`, n 1,000 | 8.00 | 1.47 | 1.52 | -0.05 [-0.40, +0.30] | tie | 1.46 | +0.01 [-0.32, +0.34] | tie |
| Y8 `imperfect`, n 10,000 | 8.00 | 0.90 | 1.29 | -0.39 [-0.92, +0.10] | tie | 0.86 | +0.04 [-0.24, +0.32] | tie |

At the default level with 5 starts, against the same library fits:

| Cell | Empty graph | Andrey | Reference | D [95%] | Verdict | Patched copy | D [95%] | Verdict |
|---|---|---|---|---|---|---|---|---|
| W, K 1, p 10, n 1,000 | 8.00 | 2.49 | 3.05 | -0.56 [-1.52, +0.60] | tie | 2.33 | +0.16 [-0.50, +1.12] | tie |
| W, K 1, p 10, n 10,000 | 8.00 | 0.55 | 1.93 | -1.38 [-2.12, -0.63] | lower, no rerun | 0.31 | +0.24 [0.00, +0.53] | tie |
| W, K 1, p 10, n 100,000 | 8.00 | 0.05 | 1.64 | -1.59 [-2.18, -1.04] | lower, no rerun | 0.26 | -0.21 [-0.47, 0.00] | tie |
| W, K 1, p 20, n 1,000 | 15.00 | 3.60 | 5.21 | -1.61 [-2.36, -0.88] | lower, no rerun | 3.24 | +0.36 [-0.10, +0.89] | tie |
| W, K 1, p 20, n 10,000 | 15.00 | 0.35 | 3.09 | -2.74 [-3.52, -2.00] | lower, no rerun | 0.41 | -0.06 [-0.16, 0.00] | tie |
| W, K 1, p 20, n 100,000 | 15.00 | 0.47 | 3.05 | -2.58 [-3.35, -1.85] | lower, no rerun | 0.57 | -0.10 [-0.25, +0.01] | tie |
| W, K 2, p 10, n 1,000 | 8.00 | 2.46 | 3.04 | -0.58 [-1.44, +0.45] | tie | 2.11 | +0.35 [-0.20, +1.18] | tie |
| W, K 2, p 10, n 10,000 | 8.00 | 0.25 | 2.02 | -1.77 [-2.38, -1.16] | lower, no rerun | 0.29 | -0.04 [-0.18, +0.08] | tie |
| W, K 2, p 10, n 100,000 | 8.00 | 0.05 | 1.89 | -1.84 [-2.32, -1.36] | lower, no rerun | 0.21 | -0.16 [-0.41, 0.00] | tie |
| W, K 2, p 20, n 1,000 | 15.00 | 3.16 | 4.59 | -1.43 [-1.99, -0.69] | lower, no rerun | 3.13 | +0.03 [-0.49, +0.62] | tie |
| W, K 2, p 20, n 10,000 | 15.00 | 0.75 | 2.59 | -1.84 [-2.60, -1.13] | lower, no rerun | 0.67 | +0.08 [-0.09, +0.28] | tie |
| W, K 2, p 20, n 100,000 | 15.00 | 0.47 | 2.85 | -2.38 [-3.06, -1.72] | lower, no rerun | 0.59 | -0.12 [-0.28, +0.01] | tie |
| S {4}, c 0.1, n 1,000 | 6.00 | 1.05 | 1.02 | +0.03 [-0.10, +0.17] | tie |  |  |  |
| S {4}, c 0.1, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.1, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.25, n 1,000 | 6.00 | 0.35 | 0.33 | +0.02 [0.00, +0.06] | tie |  |  |  |
| S {4}, c 0.25, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.25, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.5, n 1,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.5, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4}, c 0.5, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.1, n 1,000 | 6.00 | 0.85 | 0.83 | +0.02 [-0.03, +0.09] | tie |  |  |  |
| S {5}, c 0.1, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.1, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.25, n 1,000 | 6.00 | 0.20 | 0.16 | +0.04 [0.00, +0.12] | tie |  |  |  |
| S {5}, c 0.25, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.25, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.5, n 1,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.5, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {5}, c 0.5, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| S {4, 5}, c 0.1, n 1,000 | 6.00 | 0.75 | 0.97 | -0.22 [-0.44, -0.05] | lower, no rerun | 0.74 | +0.01 [-0.03, +0.06] | tie |
| S {4, 5}, c 0.1, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.1, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.25, n 1,000 | 6.00 | 0.20 | 0.23 | -0.03 [-0.09, 0.00] | tie | 0.16 | +0.04 [0.00, +0.12] | tie |
| S {4, 5}, c 0.25, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.25, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.5, n 1,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.5, n 10,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| S {4, 5}, c 0.5, n 100,000 | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 [0.00, 0.00] | tie |
| Y6 `perfect`, k 1, n 1,000 | 8.00 | 0.70 | 0.75 | -0.05 [-0.29, +0.16] | tie |  |  |  |
| Y6 `perfect`, k 1, n 10,000 | 8.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| Y6 `perfect`, k 2, n 1,000 | 8.00 | 0.70 | 0.75 | -0.05 [-0.31, +0.20] | tie |  |  |  |
| Y6 `perfect`, k 2, n 10,000 | 8.00 | 0.05 | 0.01 | +0.04 [0.00, +0.12] | tie |  |  |  |
| Y6 `perfect`, k 4, n 1,000 | 8.00 | 0.90 | 1.20 | -0.30 [-0.72, +0.13] | tie |  |  |  |
| Y6 `perfect`, k 4, n 10,000 | 8.00 | 0.37 | 0.38 | -0.01 [-0.31, +0.34] | tie |  |  |  |
| Y6 `perfect`, k 6, n 1,000 | 8.00 | 0.80 | 0.89 | -0.09 [-0.34, +0.16] | tie |  |  |  |
| Y6 `perfect`, k 6, n 10,000 | 8.00 | 0.20 | 0.22 | -0.02 [-0.16, +0.10] | tie |  |  |  |
| Y6 `perfect`, k 8, n 1,000 | 8.00 | 1.26 | 1.18 | +0.08 [-0.27, +0.42] | tie |  |  |  |
| Y6 `perfect`, k 8, n 10,000 | 8.00 | 0.56 | 0.54 | +0.02 [-0.26, +0.33] | tie |  |  |  |
| Y6 `perfect`, k 10, n 1,000 | 8.00 | 1.39 | 1.42 | -0.03 [-0.34, +0.27] | tie |  |  |  |
| Y6 `perfect`, k 10, n 10,000 | 8.00 | 1.03 | 0.93 | +0.10 [-0.12, +0.32] | tie |  |  |  |
| Y6 `inhibiting`, k 1, n 1,000 | 8.00 | 0.88 | 0.75 | +0.13 [-0.17, +0.51] | tie |  |  |  |
| Y6 `inhibiting`, k 1, n 10,000 | 8.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| Y6 `inhibiting`, k 2, n 1,000 | 8.00 | 0.84 | 0.73 | +0.11 [-0.20, +0.53] | tie |  |  |  |
| Y6 `inhibiting`, k 2, n 10,000 | 8.00 | 0.06 | 0.01 | +0.05 [-0.03, +0.18] | tie |  |  |  |
| Y6 `inhibiting`, k 4, n 1,000 | 8.00 | 0.93 | 1.20 | -0.27 [-0.61, +0.01] | tie |  |  |  |
| Y6 `inhibiting`, k 4, n 10,000 | 8.00 | 0.20 | 0.38 | -0.18 [-0.39, 0.00] | tie |  |  |  |
| Y6 `inhibiting`, k 6, n 1,000 | 8.00 | 0.80 | 0.93 | -0.13 [-0.39, +0.11] | tie |  |  |  |
| Y6 `inhibiting`, k 6, n 10,000 | 8.00 | 0.20 | 0.22 | -0.02 [-0.16, +0.13] | tie |  |  |  |
| Y6 `inhibiting`, k 8, n 1,000 | 8.00 | 0.91 | 1.13 | -0.22 [-0.50, +0.06] | tie |  |  |  |
| Y6 `inhibiting`, k 8, n 10,000 | 8.00 | 0.45 | 0.52 | -0.07 [-0.31, +0.20] | tie |  |  |  |
| Y6 `inhibiting`, k 10, n 1,000 | 8.00 | 1.37 | 1.41 | -0.04 [-0.34, +0.26] | tie |  |  |  |
| Y6 `inhibiting`, k 10, n 10,000 | 8.00 | 0.80 | 0.94 | -0.14 [-0.45, +0.15] | tie |  |  |  |
| Y6 `imperfect`, k 1, n 1,000 | 8.00 | 0.74 | 0.75 | -0.01 [-0.26, +0.22] | tie |  |  |  |
| Y6 `imperfect`, k 1, n 10,000 | 8.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |  |  |  |
| Y6 `imperfect`, k 2, n 1,000 | 8.00 | 0.97 | 0.75 | +0.22 [-0.10, +0.60] | tie |  |  |  |
| Y6 `imperfect`, k 2, n 10,000 | 8.00 | 0.08 | 0.01 | +0.07 [0.00, +0.20] | tie |  |  |  |
| Y6 `imperfect`, k 4, n 1,000 | 8.00 | 1.05 | 1.08 | -0.03 [-0.40, +0.39] | tie |  |  |  |
| Y6 `imperfect`, k 4, n 10,000 | 8.00 | 0.45 | 0.47 | -0.02 [-0.27, +0.24] | tie |  |  |  |
| Y6 `imperfect`, k 6, n 1,000 | 8.00 | 1.14 | 1.05 | +0.09 [-0.23, +0.45] | tie |  |  |  |
| Y6 `imperfect`, k 6, n 10,000 | 8.00 | 0.20 | 0.21 | -0.01 [-0.13, +0.11] | tie |  |  |  |
| Y6 `imperfect`, k 8, n 1,000 | 8.00 | 1.28 | 1.29 | -0.01 [-0.33, +0.33] | tie |  |  |  |
| Y6 `imperfect`, k 8, n 10,000 | 8.00 | 0.85 | 0.63 | +0.22 [-0.03, +0.51] | tie |  |  |  |
| Y6 `imperfect`, k 10, n 1,000 | 8.00 | 1.62 | 1.45 | +0.17 [-0.24, +0.61] | tie |  |  |  |
| Y6 `imperfect`, k 10, n 10,000 | 8.00 | 0.90 | 1.00 | -0.10 [-0.41, +0.19] | tie |  |  |  |
| Y5 `perfect`, n 1,000 | 15.00 | 2.04 | 2.03 | +0.01 [-0.32, +0.34] | tie |  |  |  |
| Y5 `perfect`, n 10,000 | 15.00 | 1.90 | 1.94 | -0.04 [-0.60, +0.49] | tie |  |  |  |
| Y5 `inhibiting`, n 1,000 | 15.00 | 2.24 | 2.20 | +0.04 [-0.49, +0.64] | tie |  |  |  |
| Y5 `inhibiting`, n 10,000 | 15.00 | 1.90 | 1.94 | -0.04 [-0.58, +0.44] | tie |  |  |  |
| Y5 `imperfect`, n 1,000 | 15.00 | 2.33 | 2.04 | +0.29 [-0.12, +0.79] | tie |  |  |  |
| Y5 `imperfect`, n 10,000 | 15.00 | 1.99 | 1.94 | +0.05 [-0.37, +0.50] | tie |  |  |  |
| Y8 `perfect`, n 1,000 | 8.00 | 1.57 | 1.55 | +0.02 [-0.29, +0.32] | tie | 1.40 | +0.17 [-0.15, +0.48] | tie |
| Y8 `perfect`, n 10,000 | 8.00 | 0.90 | 1.37 | -0.47 [-1.04, +0.01] | tie | 0.82 | +0.08 [-0.17, +0.34] | tie |
| Y8 `inhibiting`, n 1,000 | 8.00 | 1.23 | 1.55 | -0.32 [-0.63, -0.02] | lower, no rerun | 1.39 | -0.16 [-0.44, +0.10] | tie |
| Y8 `inhibiting`, n 10,000 | 8.00 | 0.60 | 1.37 | -0.77 [-1.26, -0.34] | lower, no rerun | 0.82 | -0.22 [-0.45, +0.02] | tie |
| Y8 `imperfect`, n 1,000 | 8.00 | 1.47 | 1.52 | -0.05 [-0.40, +0.30] | tie | 1.46 | +0.01 [-0.32, +0.34] | tie |
| Y8 `imperfect`, n 10,000 | 8.00 | 0.96 | 1.29 | -0.33 [-0.88, +0.18] | tie | 0.86 | +0.10 [-0.20, +0.40] | tie |

At the other two levels, counted over the same cells (worse and better count only where a rerun
agrees; "no rerun" marks rows lower on seeds 0 to 19 in cells that were not rerun):

| Level | Starts | Against | Worse | Higher on seeds 0-19 only | Better | Lower on seeds 0-19 only | Lower, no rerun | Tie |
|---|---|---|---|---|---|---|---|---|
| `1e-4` | 5 | reference | 0 | 5 | 0 | 0 | 13 | 69 |
| `1e-4` | 5 | patched copy | 1 | 1 | 0 | 0 | 0 | 25 |
| `1e-4` | 10 | reference | 0 | 2 | 1 | 0 | 14 | 70 |
| `1e-4` | 10 | patched copy | 0 | 0 | 0 | 0 | 0 | 27 |
| `1e-3` | 5 | reference | 0 | 0 | 0 | 0 | 13 | 74 |
| `1e-3` | 5 | patched copy | 0 | 0 | 0 | 0 | 0 | 27 |
| `1e-3` | 10 | reference | 0 | 0 | 11 | 2 | 0 | 74 |
| `1e-3` | 10 | patched copy | 0 | 0 | 0 | 2 | 0 | 25 |
| `1e-2` | 5 | reference | 0 | 0 | 1 | 0 | 28 | 58 |
| `1e-2` | 5 | patched copy | 0 | 0 | 0 | 1 | 13 | 13 |
| `1e-2` | 10 | reference | 0 | 0 | 1 | 1 | 34 | 51 |
| `1e-2` | 10 | patched copy | 0 | 0 | 0 | 1 | 13 | 13 |

### Replication

Every comparison whose interval lay above 0 on seeds 0 to 19, with its interval on fresh seeds
100 to 119, on which every package refit the cell. All nine are at alpha `1e-4` with 1,000 rows
per setting on 10 variables; none is at the default level or at `1e-2`.

| Cell | Level | Starts | Against | Seeds 0-19: Andrey, comparator, D [95%] | Seeds 100-119: Andrey, comparator, D [95%] | Verdict |
|---|---|---|---|---|---|---|
| W, K 2, p 10, n 1,000 | `1e-4` | 5 | patched copy | 2.51, 2.12, +0.39 [0.00, +0.93] | 3.07, 2.76, +0.31 [+0.01, +0.63] | worse |
| Y6 `inhibiting`, k 1, n 1,000 | `1e-4` | 5 | reference | 1.13, 0.65, +0.48 [+0.08, +0.98] | 1.59, 1.90, -0.31 [-0.86, +0.16] | higher on seeds 0-19 only |
| Y6 `imperfect`, k 2, n 1,000 | `1e-4` | 5 | reference | 1.02, 0.58, +0.44 [+0.10, +0.88] | 1.80, 1.91, -0.11 [-0.38, +0.16] | higher on seeds 0-19 only |
| Y6 `imperfect`, k 6, n 1,000 | `1e-4` | 5 | reference | 1.57, 1.19, +0.38 [+0.03, +0.79] | 2.08, 2.16, -0.08 [-0.41, +0.27] | higher on seeds 0-19 only |
| Y6 `imperfect`, k 8, n 1,000 | `1e-4` | 5 | reference | 1.53, 1.16, +0.37 [+0.09, +0.67] | 2.04, 2.40, -0.36 [-0.91, +0.08] | higher on seeds 0-19 only |
| Y6 `imperfect`, k 8, n 1,000 | `1e-4` | 10 | reference | 1.45, 1.16, +0.29 [+0.02, +0.58] | 1.92, 2.40, -0.48 [-1.00, -0.09] | higher on seeds 0-19 only |
| Y6 `imperfect`, k 10, n 1,000 | `1e-4` | 5 | reference | 1.93, 1.35, +0.58 [+0.22, +0.97] | 2.67, 2.83, -0.16 [-0.74, +0.30] | higher on seeds 0-19 only |
| Y6 `imperfect`, k 10, n 1,000 | `1e-4` | 10 | reference | 1.93, 1.35, +0.58 [+0.22, +0.97] | 2.54, 2.83, -0.29 [-0.86, +0.16] | higher on seeds 0-19 only |
| Y8 `perfect`, n 1,000 | `1e-4` | 5 | patched copy | 1.68, 1.28, +0.40 [+0.03, +0.77] | 2.18, 2.78, -0.60 [-1.23, -0.08] | higher on seeds 0-19 only |

The W row's lower end on seeds 0 to 19 is `4e-17`, floating-point residue of 0, read as above 0,
which sent the row to the rerun. On the rerun it was higher again, so it counts as worse: the one
replicated row of the campaign. The same rerun found 3 comparisons in these cells lower again,
which count as better at their levels (in the table above).

### The replicated row

At alpha `1e-4` with 5 starts, on W with 2 settings of 4 targets, 10 variables, and 1,000 rows per
setting, Andrey's SHD was higher than the patched copy's on both seed sets: 2.51 against 2.12 on
seeds 0 to 19 and 3.07 against 2.76 on seeds 100 to 119. The trace finds the cause in the first
start. Andrey's first start is the minimum-degree order with its ties broken by the column ranks,
the same in every draw, and a later start replaces its result only with a strictly lower
objective; the library draws all 5 starts at random in every draw. At this level and size the
objective often fails to single out the truth: the true DAG's objective equals the returned DAG's
in 42 and 26 of the 100 draws of the two seed sets, and is strictly higher in 45 and 65. Where
the first start lands on the worse member of such a tie, or in a worse region that the other 4
starts miss, all 5 of Andrey's draws repeat that result, while the library's 5 draws vary, and
the luck of the draws adds to the gap. With the ranks reversed the row ties on both seed sets
(+0.16 [-0.03, +0.37] and -0.17 [-0.47, +0.12]); with 20 random rankings it is higher on both for
none; with 30 draws per side instead of 5 it ties on both (+0.21 and +0.20). A smaller share
comes from a named difference, the library's one-sided start test, which keeps weak edges that
Andrey's level drops (switching it moves D from +0.39 to +0.29). Both packages count the same
I-contradicting edges on every DAG traced. The same cell ties at Andrey's default of 10 starts on
both seed sets (+0.13 [-0.07, +0.34] and +0.19 [-0.08, +0.47]), and against the unpatched library
the row ties on seeds 0 to 19 and is lower on seeds 100 to 119. The row awaits a decision before
IGSP can leave experimental status.

### Floors and coverage

In every cell, at every level and with both numbers of starts, Andrey's SHD was below the empty
graph's: the largest upper end of the interval of Andrey's SHD minus the empty graph's is -4.34,
over 522 rows. Every fit of Andrey and of the library in the claim finished within its cap, so no
row was filtered and no dataset is missing.

### Choosing the defaults

The defaults were chosen before this comparison, by Andrey alone on seeds 1000 to 1019, one draw,
over the 87 cells (Notes of `igsp`). Alpha `1e-3` won 123 comparisons and lost none against the
other levels of the authors' sweep: it was better than `1e-4` in 4 cells, `5e-3` in 16, `1e-2` in
20, and `5e-2` in 83, and tied `5e-4` in all 87. At `1e-3`, 10 starts and 5 tied each other in
every cell; against one start, 10 were better in 21 cells and 5 in 19.

A confirmation run, its rule written before any fit, then compared alpha `1e-3` with alpha one
over the observational rows, both at 10 starts, against the library at `1e-3` (and the patched
copy in the multi-target cells), on seeds 6000 to 6019 with 5 draws: worse in none of 114 for
either, below 0 in 18 for `1e-3` and in 17 for one over the observational rows. The rule kept the
candidate with the most comparisons below 0, `1e-3`.

### Other methods

Andrey's `igsp` at its defaults beside other methods on the same datasets, seeds 0 to 19, one
seed set and no rerun, never judged:

- Andrey's `gies`, from the same inputs: `igsp`'s SHD was higher in 35 of the 87 cells (all 6 of
  Y5 and of Y8, 6 of W's 12, 14 of Y6's 36, and 3 of S's 27), lower in 6 (Y6), and tied in 46.
  The mean differences in the 35 run from 0.35 to 2.63.
- Andrey's `ut_igsp`, told every target: `igsp`'s SHD was higher in 27 cells, from 0.38 to 1.95,
  and lower in none.
- The library's UT-IGSP told every target: higher in 14 cells and lower in 14.
- Andrey's `grasp` on the observational rows alone: lower in 54 cells and higher in none.
- The library with 10 runs, matching Andrey's starts: Andrey's SHD was lower in 13 cells and higher
  in none; against the patched copy with 10 runs, higher in 1 (Y8 `perfect` at n 1,000, +0.27
  [+0.01, +0.54]) and tied in 26.

IGSP's I-contradicting test marks a true arc out of a null target, so the search is pushed to
reverse it. Summed over the 87 cells, each dataset counted by its mean over draws, Andrey
reversed 823 of the 3,960 such arcs, the library 838, and Andrey's `gies`, which drops the
target's own term in that setting, 16.

### Out of the claim

One draw per dataset, 20 datasets per cell (10 at 50 to 400 variables), Andrey at its defaults,
with a cap of 3,600 s per fit. D, Andrey's SHD minus the library's, is given where the library
finished every dataset. The library has no run with an unknown off-target, with additive noise and
the Gaussian tests, or on categorical data, and none above 100 variables; Andrey's `gies` runs
beside it in every row but those with additive noise.

| Cell | Empty graph | Andrey | Reference | D [95%] | GIES | Andrey, median fit (s) | Reference, median fit (s) |
|---|---|---|---|---|---|---|---|
| Kernel tests, additive noise, n 200 | 8.00 | 5.00 | 3.80 | +1.20 [-0.20, +2.65] |  | 1.72 | 8.69 |
| Kernel tests, additive noise, n 500 | 8.00 | 1.15 | 1.90 | -0.75 [-2.05, +0.65] |  | 13.83 | 21.19 |
| 50 variables | 38.00 | 3.90 | 5.90 | -2.00 [-4.10, -0.10] | 0.20 | 1.56 | 4.28 |
| 100 variables | 75.00 | 15.30 | 17.00 (1 of 10 finished) |  | 0.10 | 32.57 | 519.27 |
| 200 variables | 150.00 | 42.90 |  |  | 0.30 | 726.06 |  |
| 400 variables | 300.00 | 0 of 10 finished |  |  |  |  |  |
| p 10, 4 settings, n 100 | 8.00 | 4.65 | 4.05 | +0.60 [+0.15, +1.05] | 0.75 | 0.02 | 0.01 |
| p 10, 4 settings, n 300 | 8.00 | 2.60 | 3.05 | -0.45 [-1.35, +0.35] | 0.05 | 0.02 | 0.01 |
| p 20, 8 settings, n 100 | 15.00 | 9.10 | 8.65 | +0.45 [-0.60, +1.55] | 0.30 | 0.10 | 0.03 |
| p 20, 8 settings, n 300 | 15.00 | 3.65 | 3.85 | -0.20 [-0.85, +0.45] | 0.00 | 0.09 | 0.03 |
| Mean degree 3 | 30.00 | 1.85 | 2.70 | -0.85 [-2.10, -0.15] | 0.20 | 0.38 | 0.09 |
| Mean degree 4 | 40.00 | 11.80 | 12.65 | -0.85 [-6.00, +4.35] | 7.90 | 0.39 | 0.09 |
| Unknown off-target, W's intervention | 15.00 | 0.85 |  |  | 2.70 | 0.16 |  |
| Unknown off-target, mean shift | 15.00 | 0.95 |  |  | 26.00 | 0.15 |  |
| Additive noise, Gaussian tests, n 200 | 8.00 | 2.05 |  |  |  | 0.02 |  |
| Additive noise, Gaussian tests, n 500 | 8.00 | 2.30 |  |  |  | 0.02 |  |
| Categorical, G-test, n 1,000 | 10.00 | 6.40 |  |  | 4.35 | 0.13 |  |
| Categorical, G-test, n 10,000 | 10.00 | 0.90 |  |  | 4.95 | 0.29 |  |
| Laplace noise, n 1,000 | 8.00 | 1.85 | 1.55 | +0.30 [-0.20, +0.85] | 0.20 | 0.03 | 0.01 |
| Laplace noise, n 10,000 | 8.00 | 0.05 | 0.05 | 0.00 [0.00, 0.00] | 0.00 | 0.12 | 0.02 |
| Uniform noise, n 1,000 | 8.00 | 1.20 | 1.05 | +0.15 [-0.10, +0.50] | 0.00 | 0.03 | 0.01 |
| Uniform noise, n 10,000 | 8.00 | 0.05 | 0.05 | 0.00 [0.00, 0.00] | 0.00 | 0.12 | 0.02 |
| S {4} and {5}, c 0.25, n 1,000, observational 1x | 6.00 | 0.20 | 0.20 | 0.00 [0.00, 0.00] | 0.15 | 0.01 | 0.01 |
| S {4} and {5}, c 0.25, n 1,000, observational 3x | 6.00 | 0.05 | 0.00 | +0.05 [0.00, +0.15] | 0.00 | 0.02 | 0.01 |
| S {4} and {5}, c 0.25, n 1,000, observational 10x | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.05 | 0.03 | 0.01 |
| S {4} and {5}, c 0.25, n 10,000, observational 1x | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.10 | 0.06 | 0.01 |
| S {4} and {5}, c 0.25, n 10,000, observational 3x | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.10 | 0.02 |
| S {4} and {5}, c 0.25, n 10,000, observational 10x | 6.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.27 | 0.05 |
| p 10, 4 `perfect` settings, n 1,000, observational 3x | 8.00 | 0.50 | 0.70 | -0.20 [-0.65, +0.10] | 0.05 | 0.03 | 0.01 |
| p 10, 4 `perfect` settings, n 1,000, observational 10x | 8.00 | 0.35 | 0.65 | -0.30 [-0.80, +0.05] | 0.15 | 0.05 | 0.02 |

On the off-target rows, Andrey's `ut_igsp`, told the targets that are known and learning the
others, reached SHD 0.15 and 0.05. The rows by number of variables stopped at 400, where no
Andrey fit finished within the cap. At 100 variables the library finished 1 of 10 datasets within
the cap, its finished fit peaking at 72 GB of memory against Andrey's 0.5 GB; at 50 variables its
fits peaked at up to 5.4 GB, Andrey's at 0.3 GB.

### Speed

Andrey's median fit took longer than the library's in 84 of the 87 cells at the default level, and
the ratio grows with the rows: its median over cells was 1.83 at 1,000 rows per setting, 4.29 at
10,000, and 3.48 at 100,000 (W and S only). Andrey's fit time grows with the total number of rows,
the library's much less. Where the search dominates, Andrey is faster: with the kernel tests 1.72 s
against 8.69 s at 200 rows and 13.83 s against 21.19 s at 500, and on 50 variables 1.56 s against
4.28 s. The ratio is per row the median over datasets of Andrey's median fit time over its draws
divided by the library's at 5 runs.

| Block | n | Rows | Andrey, median fit (s) | Reference, median fit (s) | Ratio, median [range] |
|---|---|---|---|---|---|
| S | 1,000 | 9 | 0.011 | 0.007 | 1.55 [0.74, 1.72] |
| S | 10,000 | 9 | 0.043 | 0.011 | 4.07 [1.69, 4.29] |
| S | 100,000 | 9 | 0.376 | 0.061 | 6.20 [2.84, 6.42] |
| W | 1,000 | 4 | 0.064 | 0.048 | 1.13 [1.06, 1.39] |
| W | 10,000 | 4 | 0.127 | 0.064 | 1.84 [1.60, 2.13] |
| W | 100,000 | 4 | 0.778 | 0.244 | 3.34 [3.17, 3.48] |
| Y5 | 1,000 | 3 | 0.197 | 0.058 | 3.27 [3.22, 3.42] |
| Y5 | 10,000 | 3 | 1.154 | 0.151 | 7.86 [7.72, 7.98] |
| Y6 | 1,000 | 18 | 0.038 | 0.016 | 2.28 [1.71, 2.87] |
| Y6 | 10,000 | 18 | 0.174 | 0.036 | 5.15 [3.04, 6.87] |
| Y8 | 1,000 | 3 | 0.238 | 0.186 | 1.29 [1.20, 1.34] |
| Y8 | 10,000 | 3 | 2.028 | 0.342 | 5.88 [5.62, 6.07] |

### Where the differences show

Of the named differences, the multi-target test accounts for the 11 comparisons better than the
unpatched library, since the patched copy ties in all 11, and the first start, with a smaller share
from the one-sided start test, for the replicated row.

## References

- Wang, Y., Solus, L., Yang, K. D., Uhler, C. (2017). Permutation-based causal inference
  algorithms with interventions. NeurIPS 30.
- Yang, K. D., Katcoff, A., Uhler, C. (2018). Characterizing and learning equivalence classes of
  causal DAGs under interventions. ICML, PMLR 80, 5541-5550.
- Hauser, A., Buhlmann, P. (2012). Characterization and greedy learning of interventional Markov
  equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
- Squires, C., Wang, Y., Uhler, C. (2020). Permutation-based causal structure learning with
  unknown intervention targets. UAI, PMLR 124, 1039-1048.
