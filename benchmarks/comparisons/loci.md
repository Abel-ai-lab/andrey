# LOCI direction test

`loci_pairs.py` compares Andrey's `loci` with the LOCI authors' code (Immer et al. 2023) on
cause-effect pairs, each criterion against the same criterion: `andrey.loci.independence.numpy`
against `immer.loci.independence` (LOCI-H, an HSIC test of the scaled residual) and
`andrey.loci.likelihood.numpy` against `immer.loci.likelihood` (LOCI-M, the larger likelihood).
Andrey's `anm`, `pnl`, and `cam` run on the same pairs as context: reported, never judged. Every
pair has two nodes and one edge, in a direction the seed draws, and every fit is scored by whether
it names that direction.

## Data

| Regime | Data | Rows |
|---|---|---|
| `lsnm_pair`, `lsnm_gumbel_pair`, `lsnm_laplace_pair` | the location-scale generator at the paper's scale range, Gaussian, Gumbel, or Laplace noise, z-scored | 500, 1,000, 2,000 |
| `lsnm_rescaled_pair`, `lsnm_gumbel_rescaled_pair`, `lsnm_laplace_rescaled_pair` | the same, rescaled so the cause has unit variance before the effect reads it | 500, 1,000, 2,000 |
| `anm_pair` | additive noise, Gaussian, z-scored: a constant noise scale | 500, 1,000, 2,000 |
| `tagasovska_{an,ans,ls,lss,mnu}_pair` | the synthetic sets of Tagasovska et al. (2020): AN, AN-s, LS, LS-s, MN-U, z-scored | 1,000 |

Every regime lies in LOCI's model, `y = f(x) + g(x) n` with the noise independent of a cause of
unbounded support, so the independence criterion is judged in all 26 cells. The likelihood criterion
assumes Gaussian noise: it is judged in the 13 cells with Gaussian noise and reported, not judged,
in the 13 with Gumbel, Laplace, or MN-U's uniform noise, which the run's `run_meta.json` marks. The
harness draws the Tagasovska et al. sets from that paper's description, not from its published pair
files; `andrey_bench/pair_models.py` lists every value the paper states and, with its source, every
constant it leaves open. Each cell holds 100 pairs, seeds 0 to 99.

## Settings

| Andrey | The authors' code |
|---|---|
| `loci(data)`, `criterion="independence"`, the default, with `alpha=None` | `loci(x, y, independence_test=True, neural_network=False)`, LOCI-H |
| `loci(data, criterion="likelihood")` | `loci(x, y, independence_test=False, neural_network=False)`, LOCI-M |

The authors' code is [aleximmer/loci](https://github.com/aleximmer/loci) at commit
`da61b8cdf95cbbc00fc42d351f89e64410f535f4`. It has no license. `benchmarks/start_loci.sh` downloads
the six files of its `causa` package that a fit imports at that commit, checks each one's SHA-256,
writes them to `benchmarks/.loci`, and builds `.venv-loci` from `requirements-loci.txt`; the
benchmark imports the code from there and copies none of it. Its adapter standardizes each column,
as the paper does, and calls the paper's estimator with spline features at the code's other
defaults, which the adapter's docstring lists. Andrey standardizes each column itself.

## Scores

Each record gets `direction_correct`, `direction_reversed`, and `direction_none`, one of them 1.0;
no fit of either package left a pair without a direction. Per pair, D is Andrey's wrong direction
minus the authors' code's, each 1 or 0, so a row's mean D is the difference of the two shares of
wrong directions and is below 0 where Andrey names more pairs right. A row is a cell and a
criterion. Its interval is the percentile bootstrap 95% interval of the mean D over its paired pairs
(10,000 resamples): the row is worse when the interval lies above 0 and better when below. Better
counts only if a rerun on fresh pairs, seeds 100 to 199, is better too; without one, the row is
published as no less accurate. The floor is a coin flip: a package is above it where the 95%
interval of its share of true directions lies above one half. A fit past its cap is never scored as
a guess.

## Results

Andrey's pre-release build of 9 November 2026 against the authors' code, Andrey on Python 3.13.11
and NumPy 2.5.4, the authors' code on Python 3.11 with torch 2.15.0+cpu, NumPy 1.26.4, and
scikit-learn 1.9.1, one fit at a time on one core with one BLAS thread: the judged cells on an AMD
EPYC 7713 processor and the size ladder on an AMD EPYC 9534. Every number below is in
[`published/0.16.0/loci.json`](../published/0.16.0/loci.json). Every fit of the authors' code
finished. On 11 of the 2,600 pairs, all with Gumbel or Laplace noise, the build the campaign ran
raised under both criteria: the Hessian of one Newton step was too ill-conditioned for a Cholesky
factorization. The released solver adds a growing multiple of the Hessian's diagonal there (Nocedal
and Wright 2006), and those pairs were fitted again with it, so the rows below hold the released
code's result on every pair. Refitted with the released solver, the other 1,178 fits of the six
cells with such a pair returned the same output to the last bit.

### Independence criterion

Pairs with the true direction, of 100, for `criterion="independence"` against LOCI-H. Rescaled marks
the cells whose cause has unit variance before the effect reads it.

| Pairs | Rows | Andrey | Authors' code | D [95%] | Published |
|---|---|---|---|---|---|
| Gaussian | 500 | 85 | 75 | -0.1 [-0.19, -0.01] | fewer errors |
| Gaussian | 1,000 | 94 | 87 | -0.07 [-0.13, -0.02] | fewer errors |
| Gaussian | 2,000 | 98 | 95 | -0.03 [-0.07, 0.01] | no less accurate |
| Gaussian, rescaled | 500 | 86 | 66 | -0.2 [-0.28, -0.12] | fewer errors |
| Gaussian, rescaled | 1,000 | 93 | 76 | -0.17 [-0.25, -0.09] | no less accurate |
| Gaussian, rescaled | 2,000 | 99 | 97 | -0.02 [-0.05, 0] | no less accurate |
| Laplace | 500 | 83 | 71 | -0.12 [-0.21, -0.04] | no less accurate |
| Laplace | 1,000 | 91 | 83 | -0.08 [-0.15, -0.02] | no less accurate |
| Laplace | 2,000 | 96 | 96 | 0 [-0.04, 0.04] | no less accurate |
| Laplace, rescaled | 500 | 82 | 69 | -0.13 [-0.21, -0.06] | no less accurate |
| Laplace, rescaled | 1,000 | 93 | 84 | -0.09 [-0.16, -0.03] | no less accurate |
| Laplace, rescaled | 2,000 | 98 | 93 | -0.05 [-0.1, -0.01] | no less accurate |
| Gumbel | 500 | 79 | 52 | -0.27 [-0.36, -0.19] | no less accurate |
| Gumbel | 1,000 | 95 | 79 | -0.16 [-0.25, -0.08] | no less accurate |
| Gumbel | 2,000 | 99 | 94 | -0.05 [-0.1, -0.01] | no less accurate |
| Gumbel, rescaled | 500 | 80 | 59 | -0.21 [-0.3, -0.12] | no less accurate |
| Gumbel, rescaled | 1,000 | 97 | 79 | -0.18 [-0.26, -0.1] | no less accurate |
| Gumbel, rescaled | 2,000 | 99 | 91 | -0.08 [-0.14, -0.03] | no less accurate |
| Additive, Gaussian | 500 | 83 | 81 | -0.02 [-0.09, 0.05] | no less accurate |
| Additive, Gaussian | 1,000 | 90 | 89 | -0.01 [-0.06, 0.04] | no less accurate |
| Additive, Gaussian | 2,000 | 98 | 98 | 0 [0, 0] | no less accurate |
| AN | 1,000 | 100 | 97 | -0.03 [-0.07, 0] | no less accurate |
| AN-s | 1,000 | 100 | 99 | -0.01 [-0.03, 0] | no less accurate |
| LS | 1,000 | 94 | 86 | -0.08 [-0.14, -0.02] | fewer errors |
| LS-s | 1,000 | 100 | 50 | -0.5 [-0.6, -0.4] | fewer errors |
| MN-U | 1,000 | 99 | 96 | -0.03 [-0.07, 0] | no less accurate |

No row is worse. On the first seeds 17 of the 26 intervals lie below 0 and 9 include it; Andrey
named the true direction on 79 to 100 pairs of 100 and the authors' code on 50 to 99.

### Likelihood criterion

The same for `criterion="likelihood"` against LOCI-M. The rows with Gumbel, Laplace, or MN-U's
uniform noise are outside the Gaussian model of this criterion and are not judged.

| Pairs | Rows | Andrey | Authors' code | D [95%] | Published |
|---|---|---|---|---|---|
| Gaussian | 500 | 100 | 91 | -0.09 [-0.15, -0.04] | fewer errors |
| Gaussian | 1,000 | 100 | 99 | -0.01 [-0.03, 0] | no less accurate |
| Gaussian | 2,000 | 100 | 100 | 0 [0, 0] | no less accurate |
| Gaussian, rescaled | 500 | 100 | 90 | -0.1 [-0.16, -0.05] | fewer errors |
| Gaussian, rescaled | 1,000 | 100 | 97 | -0.03 [-0.07, 0] | no less accurate |
| Gaussian, rescaled | 2,000 | 100 | 100 | 0 [0, 0] | no less accurate |
| Laplace | 500 | 37 | 31 | -0.06 [-0.15, 0.03] | not judged |
| Laplace | 1,000 | 44 | 35 | -0.09 [-0.15, -0.03] | not judged |
| Laplace | 2,000 | 59 | 50 | -0.09 [-0.16, -0.03] | not judged |
| Laplace, rescaled | 500 | 37 | 36 | -0.01 [-0.1, 0.08] | not judged |
| Laplace, rescaled | 1,000 | 47 | 37 | -0.1 [-0.17, -0.04] | not judged |
| Laplace, rescaled | 2,000 | 57 | 53 | -0.04 [-0.08, -0.01] | not judged |
| Gumbel | 500 | 35 | 31 | -0.04 [-0.11, 0.03] | not judged |
| Gumbel | 1,000 | 37 | 35 | -0.02 [-0.08, 0.03] | not judged |
| Gumbel | 2,000 | 35 | 35 | 0 [-0.03, 0.03] | not judged |
| Gumbel, rescaled | 500 | 35 | 33 | -0.02 [-0.08, 0.04] | not judged |
| Gumbel, rescaled | 1,000 | 38 | 37 | -0.01 [-0.05, 0.03] | not judged |
| Gumbel, rescaled | 2,000 | 35 | 34 | -0.01 [-0.04, 0.02] | not judged |
| Additive, Gaussian | 500 | 85 | 77 | -0.08 [-0.16, -0.01] | no less accurate |
| Additive, Gaussian | 1,000 | 93 | 87 | -0.06 [-0.11, -0.02] | no less accurate |
| Additive, Gaussian | 2,000 | 99 | 99 | 0 [0, 0] | no less accurate |
| AN | 1,000 | 100 | 96 | -0.04 [-0.08, -0.01] | no less accurate |
| AN-s | 1,000 | 100 | 99 | -0.01 [-0.03, 0] | no less accurate |
| LS | 1,000 | 100 | 92 | -0.08 [-0.14, -0.03] | fewer errors |
| LS-s | 1,000 | 100 | 96 | -0.04 [-0.08, -0.01] | no less accurate |
| MN-U | 1,000 | 100 | 96 | -0.04 [-0.08, -0.01] | not judged |

No row is worse. Of the 13 judged rows, 7 intervals lie below 0 on the first seeds and 6 include 0;
over all 26, 12 and 14. Under Gaussian noise Andrey named the true direction on every location-scale
pair. Under Gumbel noise both packages named fewer than half the pairs right at every size, and
under Laplace noise both stayed at or below one half: there the criterion is outside its model, in
the authors' code as in Andrey.

### Fresh seeds

The rows that came out better on the first seeds on location-scale pairs with Gaussian noise, the
setting both criteria claim, were fitted again on 100 fresh pairs, seeds 100 to 199, with both
packages. 8 of the 10 repeated, and only those are published as fewer errors; every other row, rerun
or not, is published as no less accurate. On 3 fresh pairs the authors' code did not start, a file
its environment reads was not found, and those pairs are left out of their rows.

| Pairs | Rows | Criterion | First seeds: D [95%] | Fresh pairs fitted by both | Andrey | Authors' code | D [95%] | Repeats |
|---|---|---|---|---|---|---|---|---|
| Gaussian | 500 | independence | -0.1 [-0.19, -0.01] | 100 | 90 | 67 | -0.23 [-0.32, -0.15] | yes |
| Gaussian | 1,000 | independence | -0.07 [-0.13, -0.02] | 98 | 95 | 85 | -0.1 [-0.17, -0.04] | yes |
| Gaussian, rescaled | 500 | independence | -0.2 [-0.28, -0.12] | 100 | 90 | 68 | -0.22 [-0.31, -0.14] | yes |
| Gaussian, rescaled | 1,000 | independence | -0.17 [-0.25, -0.09] | 99 | 95 | 89 | -0.06 [-0.12, 0] | no |
| LS | 1,000 | independence | -0.08 [-0.14, -0.02] | 100 | 96 | 86 | -0.1 [-0.17, -0.03] | yes |
| LS-s | 1,000 | independence | -0.5 [-0.6, -0.4] | 100 | 99 | 47 | -0.52 [-0.62, -0.42] | yes |
| Gaussian | 500 | likelihood | -0.09 [-0.15, -0.04] | 100 | 100 | 85 | -0.15 [-0.22, -0.08] | yes |
| Gaussian, rescaled | 500 | likelihood | -0.1 [-0.16, -0.05] | 100 | 100 | 92 | -0.08 [-0.14, -0.03] | yes |
| LS | 1,000 | likelihood | -0.08 [-0.14, -0.03] | 100 | 100 | 96 | -0.04 [-0.08, -0.01] | yes |
| LS-s | 1,000 | likelihood | -0.04 [-0.08, -0.01] | 100 | 100 | 97 | -0.03 [-0.07, 0] | no |

### Floors

With `criterion="independence"` Andrey is above the floor in all 26 cells; the authors' LOCI-H is
above it in 23 and at it in 3. With `criterion="likelihood"` Andrey is above it in 14 cells, every
cell with Gaussian noise and MN-U, at it in 4, and below it in 8; the authors' LOCI-M is above it in
14, at it in 2, and below it in 10. Every cell where the likelihood criterion is not above the floor
has Gumbel or Laplace noise.

### Context

Pairs with the true direction, of 100, for Andrey's two LOCI criteria and its other pairwise methods
on the same pairs, each at its defaults. ANM and PNL assume a noise scale that does not change with
the cause, and CAM is fitted here to two variables.

| Pairs | Rows | LOCI, independence | LOCI, likelihood | ANM | PNL | CAM |
|---|---|---|---|---|---|---|
| Gaussian | 500 | 85 | 100 | 10 | 44 | 57 |
| Gaussian | 1,000 | 94 | 100 | 9 | 45 | 48 |
| Gaussian | 2,000 | 98 | 100 | 8 | 42 | 45 |
| Gaussian, rescaled | 500 | 86 | 100 | 10 | 44 | 55 |
| Gaussian, rescaled | 1,000 | 93 | 100 | 9 | 44 | 49 |
| Gaussian, rescaled | 2,000 | 99 | 100 | 8 | 43 | 45 |
| Laplace | 500 | 83 | 37 | 76 | 85 | 68 |
| Laplace | 1,000 | 91 | 44 | 79 | 99 | 78 |
| Laplace | 2,000 | 96 | 59 | 88 | 100 | 84 |
| Laplace, rescaled | 500 | 82 | 37 | 76 | 84 | 66 |
| Laplace, rescaled | 1,000 | 93 | 47 | 78 | 98 | 76 |
| Laplace, rescaled | 2,000 | 98 | 57 | 87 | 100 | 84 |
| Gumbel | 500 | 79 | 35 | 30 | 39 | 47 |
| Gumbel | 1,000 | 95 | 37 | 25 | 42 | 45 |
| Gumbel | 2,000 | 99 | 35 | 28 | 36 | 46 |
| Gumbel, rescaled | 500 | 80 | 35 | 31 | 39 | 47 |
| Gumbel, rescaled | 1,000 | 97 | 38 | 25 | 43 | 45 |
| Gumbel, rescaled | 2,000 | 99 | 35 | 27 | 36 | 46 |
| Additive, Gaussian | 500 | 83 | 85 | 88 | 69 | 98 |
| Additive, Gaussian | 1,000 | 90 | 93 | 93 | 65 | 100 |
| Additive, Gaussian | 2,000 | 98 | 99 | 94 | 69 | 100 |
| AN | 1,000 | 100 | 100 | 100 | 100 | 100 |
| AN-s | 1,000 | 100 | 100 | 100 | 79 | 100 |
| LS | 1,000 | 94 | 100 | 61 | 100 | 100 |
| LS-s | 1,000 | 100 | 100 | 14 | 47 | 63 |
| MN-U | 1,000 | 99 | 100 | 2 | 39 | 92 |

### Speed

Median fit times in seconds, from the two columns in memory to the returned direction, and per row
the median over the pairs of the authors' time over Andrey's.

| Pairs | Rows | Independence: Andrey | LOCI-H | Ratio | Likelihood: Andrey | LOCI-M | Ratio |
|---|---|---|---|---|---|---|---|
| Gaussian | 500 | 0.07 | 60 | 852 | 0.030 | 60 | 1873 |
| Gaussian | 1,000 | 0.22 | 62 | 287 | 0.047 | 61 | 1236 |
| Gaussian | 2,000 | 0.82 | 68 | 82 | 0.081 | 65 | 764 |
| Gaussian, rescaled | 500 | 0.08 | 64 | 767 | 0.036 | 64 | 1736 |
| Gaussian, rescaled | 1,000 | 0.21 | 59 | 277 | 0.047 | 58 | 1232 |
| Gaussian, rescaled | 2,000 | 0.81 | 67 | 82 | 0.084 | 64 | 700 |
| Laplace | 500 | 0.08 | 67 | 831 | 0.038 | 67 | 1803 |
| Laplace | 1,000 | 0.22 | 65 | 285 | 0.054 | 64 | 1132 |
| Laplace | 2,000 | 0.85 | 67 | 79 | 0.104 | 64 | 614 |
| Laplace, rescaled | 500 | 0.08 | 67 | 825 | 0.036 | 67 | 1744 |
| Laplace, rescaled | 1,000 | 0.22 | 63 | 281 | 0.054 | 62 | 1112 |
| Laplace, rescaled | 2,000 | 0.84 | 67 | 81 | 0.096 | 64 | 639 |
| Gumbel | 500 | 0.07 | 64 | 879 | 0.032 | 64 | 1873 |
| Gumbel | 1,000 | 0.23 | 65 | 293 | 0.054 | 64 | 1241 |
| Gumbel | 2,000 | 0.80 | 71 | 86 | 0.086 | 67 | 737 |
| Gumbel, rescaled | 500 | 0.07 | 66 | 912 | 0.032 | 66 | 1948 |
| Gumbel, rescaled | 1,000 | 0.22 | 62 | 279 | 0.051 | 62 | 1213 |
| Gumbel, rescaled | 2,000 | 0.80 | 70 | 85 | 0.090 | 67 | 672 |
| Additive, Gaussian | 500 | 0.08 | 52 | 627 | 0.035 | 51 | 1432 |
| Additive, Gaussian | 1,000 | 0.23 | 38 | 170 | 0.045 | 37 | 830 |
| Additive, Gaussian | 2,000 | 0.82 | 31 | 40 | 0.078 | 28 | 376 |
| AN | 1,000 | 0.24 | 61 | 244 | 0.048 | 60 | 1217 |
| AN-s | 1,000 | 0.25 | 54 | 210 | 0.054 | 54 | 995 |
| LS | 1,000 | 0.24 | 70 | 286 | 0.050 | 69 | 1353 |
| LS-s | 1,000 | 0.23 | 77 | 334 | 0.056 | 76 | 1422 |
| MN-U | 1,000 | 0.23 | 41 | 159 | 0.049 | 41 | 787 |

Andrey's median fit with `criterion="independence"` took 0.07 to 0.08 s at 500 rows, 0.21 to 0.25 s
at 1,000, and 0.80 to 0.85 s at 2,000, against 31 to 77 s for LOCI-H: 40 to 912 times faster. With
`criterion="likelihood"` it took 0.03 to 0.10 s against 28 to 76 s for LOCI-M: 376 to 1,948 times
faster. Andrey's time under the independence criterion grows with the square of the rows, so its
lead narrows as the pairs grow.

### Size ladder

The first 20 pairs of `lsnm_pair` at more rows, each fit capped at 1 h and 28 GB. The authors' code
ran at 4,000 and 8,000 rows only: its HSIC multiplies matrices with one row and one column per
sample, so its time grows with the cube of the rows.

| Rows | Independence: Andrey, s | GB | True | LOCI-H, s | True | Likelihood: Andrey, s | True | LOCI-M, s | True |
|---|---|---|---|---|---|---|---|---|---|
| 4,000 | 3.93 | 0.8 | 20 | 86 | 19 | 0.14 | 20 | 61 | 20 |
| 8,000 | 15.34 | 2.7 | 20 | 257 | 19 | 0.27 | 20 | 84 | 19 |
| 16,000 | 59.66 | 10.0 | 20 | not run |  | 0.54 | 20 | not run |  |
| 32,000 | out of memory |  |  | not run |  | 1.37 | 20 | not run |  |

At 32,000 rows every fit with `criterion="independence"` ran past the 28 GB cap; the likelihood
criterion's memory stayed near 0.2 GB at every size.

## Run

```shell
benchmarks/start_loci.sh
export ANDREY_BENCH_LOCI_PYTHON=$PWD/benchmarks/.venv-loci/bin/python
export ANDREY_BENCH_LOCI=$PWD/benchmarks/.loci
export ANDREY_BENCH_CAM_PYTHON=$ANDREY_BENCH_PYTHON
$ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci-smoke --smoke
```

Andrey's `cam` runs in the interpreter `$ANDREY_BENCH_CAM_PYTHON` names, bench-env's for the build
under test. The defaults are 100 pairs per cell from seed 0, one repeat without warm-up, and a 1,800
s cap per fit. Fill the shared dataset store once with `--prepare-only`, then cut a campaign into
small runs with `--regimes`, `--sizes`, `--seeds`, and `--seed-start`, each with its own `--out`;
fresh pairs for a rerun start at seed 100:

```shell
export ANDREY_BENCH_DATA=$PWD/outputs/loci-data
$ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci-prep --prepare-only
$ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci-lsnm-1000 \
    --regimes lsnm_pair --sizes 1000 --seeds 20 --cores 1
```

The summary prints, per cell and solution, the median CPU time, the share of pairs with the true
direction, and, for each Andrey `loci` arm, `agree`, the share of pairs on which it and the authors'
arm of its criterion return the same graph. `--rescore` recomputes the direction columns without
fitting anything; pass the first run's flags.

## Known differences

- **P-values against raw statistics.** The paper says LOCI-H predicts from the p-values of the two
  HSIC tests. The authors' code compares the two biased HSIC statistics instead, backward minus
  forward. Andrey compares the p-values, through their logarithms, from a gamma approximation of
  each test's null distribution, and falls back to the statistics, each divided by its mean under
  independence, only when the p-values are equal. The two rules can name different directions from
  the same fits: on LS-s the authors' LOCI-H named the true direction on 50 of 100 pairs and Andrey
  on 100.
- **The paper's constants and the code's.** The paper's supplement gives the fit up to 100 steps;
  the code's default is at most 1,000. The code puts Gaussian priors of precision `1e-5` on both
  weight vectors, alternates a closed-form step in the mean's weights with ten L-BFGS steps in the
  precision's, and clamps the precision's linear predictor at `1e-7`. Andrey maximizes the same
  jointly concave likelihood with projected Newton steps, for at most 100 steps, under a prior of
  precision `1e-6`. Both use B-spline features of the cause with 25 knots, of degree 5.
- **One pair of the lane's tests.** On a location-scale pair of 400 rows that the lane's tests draw,
  the authors' LOCI-H names the reverse direction, its score -0.0084, while its LOCI-M and both of
  Andrey's criteria name the true one. The lane's gated test fits 300 rows of the same model, where
  every arm names the true direction.
- **The solver fix.** The build the campaign ran raised on 11 pairs (Results above). Their rows hold
  the released solver's fits; counted as wrong directions instead, they would move no row to worse.
- **The Tagasovska et al. sets.** The lane draws the five sets from the paper's description, so its
  pairs are not the 100 pairs per set that the paper's authors published, and its counts are not
  comparable with counts on those files.
- **Time.** The authors' HSIC builds several matrices with one row and one column per sample and
  multiplies them, so its time grows with the cube of the rows and its memory with their square; at
  several thousand rows it is the cost of a fit.

## References

- Immer, A., Schultheiss, C., Vogt, J. E., Scholkopf, B., Buhlmann, P., and Marx, A. (2023). On the
  identifiability and estimation of causal location-scale noise models. Proceedings of the 40th
  International Conference on Machine Learning (ICML), PMLR 202.
- Tagasovska, N., Chavez-Demoulin, V., and Vatter, T. (2020). Distinguishing cause from effect using
  quantiles: bivariate quantile causal discovery. Proceedings of the 37th International Conference
  on Machine Learning (ICML), PMLR 119, 9311-9323.
- Nocedal, J., and Wright, S. J. (2006). Numerical Optimization, 2nd edition. Springer.
