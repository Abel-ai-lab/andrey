# CAM and RESIT

`fcm_cpu.py` compares Andrey's CAM with CAM 1.0 and Andrey's RESIT with the authors' R code and
lingam's, with the empty graph as the floor. All of them return a DAG, scored against the true
DAG. The data are Erdos-Renyi graphs of mean degree 2 with 1000 rows, one core per fit, in eight
regimes: four generators, each z-scored (the default) and at raw scale (the `_raw` regimes).

| regime | generator | identifiable |
|---|---|---|
| `anm_er` | additive noise, Gaussian | yes |
| `anm_gumbel_er` | additive noise, Gumbel | yes |
| `lsnm_er` | location-scale noise | no: both methods assume constant noise scale |
| `linear_gauss_er` | linear, Gaussian | no: the control, which neither should orient |

The `andrey` fits run in the interpreter of the build under test, named by
`$ANDREY_BENCH_CAM_PYTHON` or `$ANDREY_BENCH_RESIT_PYTHON` (an environment with that build
installed); the driver's environment draws and scores the data and runs the references, so the build
need not be the driver's. `provenance.json` records the build's environment. Load R as
[The R lane](../README.md#the-r-lane) describes, then install the references:

```shell
benchmarks/start_cam_r.sh      # CAM 1.0 from the CRAN archive, SHA-256 checked
benchmarks/start_resit_r.sh    # the authors' code, SHA-256 checked; prints the export below
export ANDREY_BENCH_RESIT_R=$PWD/benchmarks/.resit-r/codeANM/code
```

The authors' code is supplementary material to Peters, Mooij, Janzing, and Scholkopf (2014) under a
two-clause BSD license. The benchmark `source()`s it from `$ANDREY_BENCH_RESIT_R`.

Every package runs at its documented defaults, and each difference is named in its adapter's
docstring rather than tuned away:

| | Andrey | reference |
|---|---|---|
| CAM | `alpha=0.001`, no parent cap | CAM 1.0 with `variableSel = TRUE, pruning = TRUE`; `maxNumParents = min(d - 1, round(n / 20))` |
| RESIT, regression | additive model, thin plate splines, GCV | authors: `mgcv::gam`, `k = 10`, GCV; lingam: random forest, `max_depth=4` |
| RESIT, `alpha` | 0.05 | authors: 0.05; lingam: 0.01 |
| RESIT, next sink | the largest HSIC p-value | authors: the same, on p-values not their logarithms; lingam: the smallest HSIC statistic |

A fit takes minutes at the larger sizes, so the driver defaults to one timed fit with no warm-up,
and `--seed-start`, `--seeds`, and `--regimes` cut a campaign into small runs with their own
`--out`; ratios pair inside one run. Fill the shared dataset store once with `--prepare-only`:

```shell
export ANDREY_BENCH_DATA=$PWD/outputs/fcm-data
$ANDREY_BENCH_PYTHON benchmarks/fcm_cpu.py --lane cam --out outputs/fcm-prep --prepare-only \
    --ladder 10 20 50 100 --seeds 20
$ANDREY_BENCH_PYTHON benchmarks/fcm_cpu.py --lane cam --out outputs/cam-anm-d20 \
    --regimes anm_er --ladder 20 --seeds 20 --cores 1 --cap-wall-s 1800
```

The published runs used `--ladder` values of 10, 20, 50, and 100 for CAM and 5, 10, and 20 for RESIT,
and 20 seeds per cell.
