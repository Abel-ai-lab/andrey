# LOCI direction test

`loci_pairs.py` compares Andrey's `loci` with the authors' LOCI code (Immer et al. 2023) on
cause-effect pairs, each criterion against the same criterion: `andrey.loci.independence.numpy`
against `immer.loci.independence` (LOCI-H, an HSIC test of the scaled residual) and
`andrey.loci.likelihood.numpy` against `immer.loci.likelihood` (LOCI-M, the larger likelihood).
Andrey's `anm`, `pnl`, and `cam` run on the same pairs as context. Every pair has two nodes and one
edge, in a direction the seed draws, and every fit is scored by whether it names that direction.

| regime | data | rows |
|---|---|---|
| `lsnm_pair`, `lsnm_gumbel_pair`, `lsnm_laplace_pair` | the location-scale generator at the paper's scale range, Gaussian, Gumbel, or Laplace noise, z-scored | 500, 1000, 2000 |
| `lsnm_rescaled_pair`, `lsnm_gumbel_rescaled_pair`, `lsnm_laplace_rescaled_pair` | the same, rescaled so the cause has unit variance before the effect reads it | 500, 1000, 2000 |
| `anm_pair` | additive noise, Gaussian, z-scored: a constant noise scale | 500, 1000, 2000 |
| `tagasovska_{an,ans,ls,lss,mnu}_pair` | the synthetic sets of Tagasovska et al. (2020): AN, AN-s, LS, LS-s, MN-U, z-scored | 1000 |

Every regime lies in LOCI's model, `y = f(x) + g(x) n` with the noise independent of a cause of
unbounded support. The likelihood criterion assumes Gaussian noise, so it is misspecified on the
Gumbel, Laplace, and MN-U regimes; the run's `run_meta.json` marks them. The harness draws the
Tagasovska et al. sets from that paper's description; `andrey_bench/pair_models.py` lists every
value the paper states and, with its source, every constant it leaves open.

The authors' code has no license. `benchmarks/start_loci.sh` downloads the six files of its
`causa` package that a fit imports at a pinned commit, checks each one's SHA-256, writes them to
`benchmarks/.loci`, and builds `.venv-loci` from `requirements-loci.txt`; the benchmark imports the
code from there and copies none of it. Its adapter standardizes each column, as the paper does,
and calls `loci(x, y, independence_test=..., neural_network=False)`, the paper's estimator with
spline features, at the code's other defaults; the adapter's docstring lists them and the two
places where the code differs from the paper's text (at most 1,000 steps, not 100; LOCI-H compares
HSIC statistics, not p-values). Andrey's `cam` runs in the interpreter
`$ANDREY_BENCH_CAM_PYTHON` names, bench-env's for the build under test:

```shell
benchmarks/start_loci.sh
export ANDREY_BENCH_LOCI_PYTHON=$PWD/benchmarks/.venv-loci/bin/python
export ANDREY_BENCH_LOCI=$PWD/benchmarks/.loci
export ANDREY_BENCH_CAM_PYTHON=$ANDREY_BENCH_PYTHON
$ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci-smoke --smoke
```

The defaults are 100 pairs per cell from seed 0, one repeat without warm-up, and a 1,800 s cap per
fit. Fill the shared dataset store once with `--prepare-only`, then cut a campaign into small runs
with `--regimes`, `--sizes`, `--seeds`, and `--seed-start`, each with its own `--out`; fresh pairs
for a rerun start at seed 100:

```shell
export ANDREY_BENCH_DATA=$PWD/outputs/loci-data
$ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci-prep --prepare-only
$ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci-lsnm-1000 \
    --regimes lsnm_pair --sizes 1000 --seeds 20 --cores 1
```

Each record gets `direction_correct`, `direction_reversed`, and `direction_none`, one of them 1.0.
The summary prints, per cell and solution, the median CPU time, the share of pairs with the true
direction, and, for each Andrey `loci` arm, `agree`, the share of pairs on which it and the
authors' arm of its criterion return the same graph. `--rescore` recomputes the direction columns
without fitting anything; pass the first run's flags.

The authors' HSIC builds several matrices with one row and one column per sample and multiplies
them, so its time grows with the cube of the rows and its memory with their square; at several
thousand rows it is the cost of a fit.
