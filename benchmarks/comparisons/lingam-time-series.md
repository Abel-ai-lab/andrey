# VARMA-LiNGAM and longitudinal LiNGAM

`lingam_ts.py --method varma` and `--method longitudinal` compare Andrey's VARMA-LiNGAM and
longitudinal LiNGAM with lingam's (`VARMALiNGAM`, `LongitudinalLiNGAM`), each on the simulation of
its paper, drawn by `andrey_bench/lingam_models.py`:

| regime | method | source | size |
|---|---|---|---|
| `varma_lingam_paper` | VARMA-LiNGAM | Hyvarinen et al. (2010), Sect. 5.4, plus one moving-average matrix | N x T, one series |
| `longitudinal_lingam_paper` | LongitudinalLiNGAM | Kadowaki et al. (2013), JSAI Sect. 6.1 | N x T, M units per time point |

`varma_lingam_paper` draws a same-step DAG with 0.8 N links (strictly lower-triangular before the
[column shuffle](../README.md#column-order)) and a lag-1 matrix with 2 N links, the paper's 40% of
entries at N = 5 held per variable as N grows, weights 0.05 to 0.5 in magnitude with a random sign,
and super-Gaussian noise. The paper's model has no moving-average part, so the series adds one
moving-average matrix drawn like the lag-1 one. `longitudinal_lingam_paper` draws, at every time
point, a dense same-step DAG with weights 0.5 to 1.5 in magnitude and a dense lag-1 matrix with
weights up to 0.5, a hidden common cause at the first time point, and non-Gaussian noise; Bach and
Jordan's (2002) densities that the paper names are specified in full for 5 of the 18, which are the
ones used. A time point whose lag matrix through the same-step part, `(I - B(t,t))^-1 B(t,t-1)`, has
spectral norm 1 or more (about 60% of the time points at N = 3 and every one from N = 5 on) has its
lag-1 matrix scaled down until that norm is 0.95, so every step contracts and the panel's scale
stays bounded as T grows; a time point below 1 keeps its draw. A panel's data has shape `(T, M, N)`,
stored with `d = N`, `n = M`, and `times = T`.

Both packages run the same settings: VARMA-LiNGAM with `order=(1, 1)`, no order search, and the lag
blocks pruned by adaptive Lasso (lingam's defaults are an order search by BIC and pruning);
LongitudinalLiNGAM with one lag and the pairwise likelihood-ratio measure. Every estimate is a
coefficient stack in the `[effect, cause]` layout both packages report
(`andrey_bench/coefficients.py`), scored by `scoring.coefficient_scores`:

| columns | what they score |
|---|---|
| `precision`, `recall`, `f1` | the directed cross-links of the same-step and lag-1 blocks, a link counted when its lag, cause, and effect match; `instantaneous_*` and `lagged_*` split them |
| `instantaneous_rmse`, `lagged_rmse`, `ma_rmse` | the root mean square error of the blocks' weights, the papers' accuracy measure |

Two solutions return the same graph when the same weights are nonzero (`structure_hash`). lingam
1.13.0, the release with both estimators, installs on Python 3.12, so its fits run in an environment
of their own:

```shell
benchmarks/start_lingam_ts.sh
export ANDREY_BENCH_LINGAM_TS_PYTHON=$PWD/benchmarks/.venv-lingam-ts/bin/python
$ANDREY_BENCH_PYTHON benchmarks/lingam_ts.py --method longitudinal --out outputs/longitudinal \
    --ladder 3 5 10 --lengths 20 50 --samples 200 1000 --seeds 10
```

Where the two VARMA-LiNGAM graphs differ, two causes remain. lingam's pruning builds the lagged
columns with `np.roll`, so its first regression row pairs `x(0)` with `x(T-1)` and `e(T-1)`; Andrey
starts the regressions at step `max(p, q)`. And the likelihood fit stops at 100 iterations, the
default of both, usually before it converges, so the same statsmodels call returns different
estimates under the different NumPy and SciPy versions of the two environments.
