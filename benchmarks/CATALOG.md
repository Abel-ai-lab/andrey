# Benchmark catalog

Every comparison the harness runs, and every release card's published numbers.
[`README.md`](README.md) covers setup, running a driver, and reading a result, which every
comparison shares.

## Comparisons

One driver per comparison. A page in [`comparisons/`](comparisons/) gives a comparison's data,
settings, and outputs where they go beyond the README; the release cards column names the cards
whose Reproduce table lists the driver.

| Comparison | Driver | Andrey | Against | Release cards |
|---|---|---|---|---|
| The template driver | `benchmark.py` | PC | causal-learn | 0.1.0, 0.2.0, 0.9.0 |
| GES | `ges_cpu.py` | GES, serial and parallel | causal-learn | 0.1.0, 0.2.0, 0.7.0, 0.9.0 |
| [GES parent limit](comparisons/ges-parent-limit.md) | `ges_cpu.py`, `xges_ges.py` | GES with `max_parents` | causal-learn, XGES | |
| [XGES](comparisons/xges.md) | `xges_cpu.py` | XGES, with and without the extended search; GES | xges | 0.3.0, 0.9.0 |
| [Tabu search](comparisons/tabu.md) | `tabu_cpu.py` | Tabu search, hill climbing | bnlearn | 0.3.0, 0.9.0 |
| [GIES](comparisons/gies.md) | `gies_cpu.py` | GIES | pcalg | 0.7.0, 0.9.0 |
| [Exact search](comparisons/exact-search.md) | `exact_cpu.py` | Exact search | causal-learn | 0.7.0, 0.9.0 |
| [CALM](comparisons/calm.md) | `calm_cpu.py` | CALM | causal-learn | 0.7.0, 0.9.0 |
| BOSS and GRaSP | `boss_grasp_cpu.py`, `boss_grasp_reach.py` | BOSS, GRaSP | causal-learn | 0.1.0, 0.9.0 |
| [Tetrad on continuous data](comparisons/tetrad-continuous.md) | `tetrad_cpu.py`, `tetrad_fci_latent.py` | PC, GES, BOSS, GRaSP, FCI | Tetrad (PC, FGES, BOSS, GRaSP, FCI) | |
| R packages | `r_packages.py` | PC, FCI, GES, hill climbing, Tabu search | pcalg, bnlearn | 0.2.0 |
| [FCI collider configurations](comparisons/fci-colliders.md) | `fci_latent.py` | FCI, both collider rules | causal-learn | 0.1.0, 0.2.0, 0.9.0 |
| [GFCI](comparisons/gfci.md) | `gfci_latent.py` | GFCI | Tetrad | 0.7.0, 0.9.0 |
| [Hidden-confounder methods](comparisons/hidden-confounder-methods.md) | `confounded_cpu.py` | RCD, BottomUpParceLiNGAM | lingam | |
| [Domains, groups, and latent indicators](comparisons/domains-groups-latent.md) | `cdnod_cpu.py`, `multi_group_cpu.py`, `gin_cpu.py` | CD-NOD, MultiGroupDirectLiNGAM, GIN | causal-learn, lingam, Tetrad | 0.7.0, 0.9.0 |
| [DirectLiNGAM and ICA-LiNGAM](comparisons/lingam.md) | `lingam_cpu.py`, `lingam_gpu.py` | DirectLiNGAM (CPU and GPU), ICA-LiNGAM | causal-learn | 0.1.0, 0.9.0 |
| [ANM direction test](comparisons/anm.md) | `anm_pairs.py` | ANM | causal-learn | 0.4.0, 0.9.0 |
| [CAM and RESIT](comparisons/cam-resit.md) | `fcm_cpu.py` | CAM, RESIT | CAM 1.0 (R), the authors' R code for RESIT, lingam | 0.13.0 |
| [PNL direction test](comparisons/pnl.md) | `pnl_pairs.py` | PNL | causal-learn | 0.4.0, 0.9.0 |
| [Categorical data](comparisons/categorical.md) | `discrete_cpu.py` | PC, GES, BOSS, GRaSP, hill climbing | causal-learn, bnlearn, pgmpy | 0.6.0, 0.8.0 |
| [Mixed data](comparisons/mixed.md) | `mixed_cpu.py` | PC, GES, hill climbing | Tetrad, bnlearn | 0.6.0 |
| [Nonlinear conditional-independence tests](comparisons/nonlinear-ci.md) | `nonlinear_ci_cpu.py`, `nonlinear_ci_queries.py` | PC and FCI with KCI, RCoT, CMIknn, GPDC | causal-learn, tigramite, pcalg with RCIT | 0.8.0 |
| [PCMCI on lagged series](comparisons/pcmci.md) | `pcmci_cpu.py` | PCMCI | tigramite | 0.5.0, 0.9.0 |
| [PCMCI+ and LPCMCI](comparisons/pcmciplus-lpcmci.md) | `pcmci_cpu.py` | PCMCI+, LPCMCI | tigramite | 0.5.0, 0.9.0 |
| [Granger causality](comparisons/granger.md) | `granger_var.py` | Granger | causal-learn | 0.5.0, 0.9.0 |
| [VARMA-LiNGAM and longitudinal LiNGAM](comparisons/lingam-time-series.md) | `lingam_ts.py` | VARMA-LiNGAM, LongitudinalLiNGAM | lingam | 0.5.0, 0.9.0 |
| [Cross-link F1 of a time-series run](comparisons/cross-link-f1.md) | `rescore_cross_links.py`, `rescore_window_pag.py` | rescores a finished run | | |
| Public datasets | `public_datasets.py` | every method on Sachs and the bnlearn networks | every package | 0.1.0 |
| [Dispatch thresholds](comparisons/dispatch-thresholds.md) | | Andrey's own execution paths | | |

`start.sh` builds the benchmark environment; `start_r.sh`, `start_cam_r.sh`, `start_resit_r.sh`,
`start_xges.sh`, `start_tigramite.sh`, `start_lingam.sh`, and `start_lingam_ts.sh` build the
separate environments some comparisons need ([setup](README.md#setup)).

## Release cards

Each release card publishes its numbers from a summary in `published/<release>/`. Two more folders
hold measurements no card of their own presents: `published/0.2.0-small-fits/`, the faster small
fits the 0.2.0 card links, and `published/0.4.1/`, DirectLiNGAM's causal-order search in the 0.4.1
fix.

| Release | Release card | Summary |
|---|---|---|
| 0.12.0 | [Calibration](../docs/docs/benchmarks/0.12.0.md) | [`published/0.12.0/summary.json`](published/0.12.0/summary.json) |
| 0.11.0 | [Ready for real data](../docs/docs/benchmarks/0.11.0.md) | [`published/0.11.0/summary.json`](published/0.11.0/summary.json) |
| 0.10.0 | [The boundary](../docs/docs/benchmarks/0.10.0.md) | [`published/0.10.0/summary.json`](published/0.10.0/summary.json) |
| 0.9.0 | [Acceleration](../docs/docs/benchmarks/0.9.0.md) | [`published/0.9.0/summary.json`](published/0.9.0/summary.json) |
| 0.8.0 | [Nonlinear tests](../docs/docs/benchmarks/0.8.0.md) | [`published/0.8.0/summary.json`](published/0.8.0/summary.json) |
| 0.7.0 | [Permutation invariance](../docs/docs/benchmarks/0.7.0.md) | [`published/0.7.0/summary.json`](published/0.7.0/summary.json) |
| 0.6.0 | [Mixed and categorical data](../docs/docs/benchmarks/0.6.0.md) | [`published/0.6.0/summary.json`](published/0.6.0/summary.json) |
| 0.5.0 | [Temporal causal discovery](../docs/docs/benchmarks/0.5.0.md) | [`published/0.5.0/summary.json`](published/0.5.0/summary.json) |
| 0.4.0 | [Pairwise direction](../docs/docs/benchmarks/0.4.0.md) | [`published/0.4.0/summary.json`](published/0.4.0/summary.json) |
| 0.3.0 | [Score-based search](../docs/docs/benchmarks/0.3.0.md) | [`published/0.3.0/summary.json`](published/0.3.0/summary.json) |
| 0.2.0 | [Hidden common causes](../docs/docs/benchmarks/0.2.0.md) | [`published/0.2.0/summary.json`](published/0.2.0/summary.json) |
| 0.1.0 | [The alpha benchmarks](../docs/docs/benchmarks/0.1.0.md) | [`published/0.1.0/summary.json`](published/0.1.0/summary.json) |
