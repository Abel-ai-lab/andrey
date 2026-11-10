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
| [GnIES](comparisons/gnies.md) | `gnies_cpu.py`, `gnies_cases.py` | GnIES | gnies | 0.14.0 |
| [ICP](comparisons/icp.md) | `icp_cpu.py`, `icp_correction.py` | ICP, both tests | InvariantCausalPrediction (R) | 0.14.0 |
| [UT-IGSP](comparisons/ut-igsp.md) | `ut_igsp_cpu.py`, `ut_igsp_cases.py` | UT-IGSP; GIES and CD-NOD as context | graphical-model-learning (UT-IGSP; IGSP and JCI-GSP as context) | 0.14.0 |
| [IGSP](comparisons/igsp.md) | `igsp_cpu.py`, `igsp_cases.py` | IGSP; GIES, GRaSP, and UT-IGSP as context | graphical-model-learning (IGSP, and a copy patched to Definition 4.2; UT-IGSP as context) | 0.14.0 |
| [Exact search](comparisons/exact-search.md) | `exact_cpu.py` | Exact search | causal-learn | 0.7.0, 0.9.0 |
| [CALM](comparisons/calm.md) | `calm_cpu.py` | CALM | causal-learn | 0.7.0, 0.9.0 |
| BOSS and GRaSP | `boss_grasp_cpu.py`, `boss_grasp_reach.py` | BOSS, GRaSP | causal-learn | 0.1.0, 0.9.0 |
| [Tetrad on continuous data](comparisons/tetrad-continuous.md) | `tetrad_cpu.py`, `tetrad_fci_latent.py` | PC, GES, BOSS, GRaSP, FCI | Tetrad (PC, FGES, BOSS, GRaSP, FCI) | |
| R packages | `r_packages.py` | PC, FCI, GES, hill climbing, Tabu search | pcalg, bnlearn | 0.2.0 |
| [FCI collider configurations](comparisons/fci-colliders.md) | `fci_latent.py` | FCI, both collider rules | causal-learn | 0.1.0, 0.2.0, 0.9.0 |
| [GFCI](comparisons/gfci.md) | `gfci_latent.py` | GFCI | Tetrad | 0.7.0, 0.9.0 |
| [BOSS-FCI and GRaSP-FCI](comparisons/boss-fci-grasp-fci.md) | `boss_grasp_fci_cpu.py` | BOSS-FCI, GRaSP-FCI, each also with `edge_removal="triangles"`; GFCI and FCI as context | Tetrad (BOSS-FCI, GRaSP-FCI; GFCI and FCI as context); pcalg for the truth | 0.15.0 |
| [FCI-JCI](comparisons/fci-jci.md) | `fci_jci_cpu.py` | FCI-JCI; FCI and CD-NOD as context | pcalg (FCI-JCI); Tetrad (FCI with the environment in the first tier) as context | 0.15.0 |
| [Hidden-confounder methods](comparisons/hidden-confounder-methods.md) | `confounded_cpu.py` | RCD, BottomUpParceLiNGAM | lingam | |
| [CAM-UV](comparisons/camuv.md) | `camuv_cpu.py` | CAM-UV | lingam, causal-learn | 0.14.0 |
| [Domains, groups, and latent indicators](comparisons/domains-groups-latent.md) | `cdnod_cpu.py`, `multi_group_cpu.py`, `gin_cpu.py` | CD-NOD, MultiGroupDirectLiNGAM, GIN | causal-learn, lingam, Tetrad | 0.7.0, 0.9.0 |
| [DirectLiNGAM and ICA-LiNGAM](comparisons/lingam.md) | `lingam_cpu.py`, `lingam_gpu.py` | DirectLiNGAM (CPU and GPU), ICA-LiNGAM | causal-learn | 0.1.0, 0.9.0 |
| [ANM direction test](comparisons/anm.md) | `anm_pairs.py` | ANM | causal-learn | 0.4.0, 0.9.0 |
| [CAM and RESIT](comparisons/cam-resit.md) | `fcm_cpu.py` | CAM, RESIT | CAM 1.0 (R), the authors' R code for RESIT, lingam | 0.13.0 |
| Stein-score screen | `stein_cpu.py` | CAM as context; no Andrey SCORE or NoGAM | the authors' SCORE, NoGAM (dodiscover), CausalDisco (R2-SortnRegress) | |
| NOTEARS and DAGMA | `notears_cpu.py`, `notears_summary.py` | NOTEARS, DAGMA, each also with `standardize=True` and `robust=True` | the authors' NOTEARS code, the authors' dagma; CausalDisco (R2-SortnRegress, Var-SortnRegress) as controls | |
| [PNL direction test](comparisons/pnl.md) | `pnl_pairs.py` | PNL | causal-learn | 0.4.0, 0.9.0 |
| [LOCI direction test](comparisons/loci.md) | `loci_pairs.py` | LOCI, both criteria; ANM, PNL, and CAM as context | the authors' LOCI code (spline features, both criteria) | 0.16.0 |
| [Categorical data](comparisons/categorical.md) | `discrete_cpu.py` | PC, GES, BOSS, GRaSP, hill climbing | causal-learn, bnlearn, pgmpy | 0.6.0, 0.8.0 |
| [Missing data](comparisons/missing-data.md) | `missing_cpu.py` | PC with MVPC; GES, XGES, BOSS, GRaSP, hill climbing, Tabu search, exact search, NOTEARS, DAGMA, and CALM with structural EM; test-wise PC, listwise deletion, and mean imputation | the MVPC authors' R code, MissDAG, bnlearn's `structural.em` | 0.16.0 |
| [Path and confounder claims](comparisons/knowledge-paths.md) | `knowledge_paths.py` | The 22 methods that honor them, from PC to multi-group DirectLiNGAM, with `required_paths`, `forbidden_paths`, and `confounded`, against the same fit without them | enumeration of every small graph; Tetrad, pcalg, tigramite, lingam, bnlearn, and causal-learn given the same knowledge | |
| [Mixed data](comparisons/mixed.md) | `mixed_cpu.py` | PC, GES, hill climbing | Tetrad, bnlearn | 0.6.0 |
| [Nonlinear conditional-independence tests](comparisons/nonlinear-ci.md) | `nonlinear_ci_cpu.py`, `nonlinear_ci_queries.py` | PC and FCI with KCI, RCoT, CMIknn, GPDC | causal-learn, tigramite, pcalg with RCIT | 0.8.0 |
| [PCMCI on lagged series](comparisons/pcmci.md) | `pcmci_cpu.py` | PCMCI | tigramite | 0.5.0, 0.9.0 |
| [PCMCI+ and LPCMCI](comparisons/pcmciplus-lpcmci.md) | `pcmci_cpu.py` | PCMCI+, LPCMCI | tigramite | 0.5.0, 0.9.0 |
| [J-PCMCI+](comparisons/jpcmciplus.md) | `jpcmciplus_cpu.py` | J-PCMCI+; PCMCI+ on the pooled datasets as context | tigramite (J-PCMCI+, with a fix) | 0.15.0 |
| [Granger causality](comparisons/granger.md) | `granger_var.py` | Granger | causal-learn | 0.5.0, 0.9.0 |
| [VARMA-LiNGAM and longitudinal LiNGAM](comparisons/lingam-time-series.md) | `lingam_ts.py` | VARMA-LiNGAM, LongitudinalLiNGAM | lingam | 0.5.0, 0.9.0 |
| [DYNOTEARS](comparisons/dynotears.md) | `dynotears_ts.py` | DYNOTEARS | CausalNex | 0.14.0 |
| [Cross-link F1 of a time-series run](comparisons/cross-link-f1.md) | `rescore_cross_links.py`, `rescore_window_pag.py` | rescores a finished run | | |
| Public datasets | `public_datasets.py` | every method on Sachs and the bnlearn networks | every package | 0.1.0 |
| [Dispatch thresholds](comparisons/dispatch-thresholds.md) | | Andrey's own execution paths | | |

`start.sh` builds the benchmark environment; `start_r.sh`, `start_cam_r.sh`, `start_resit_r.sh`,
`start_mvpc_r.sh`, `start_missdag.sh`, `start_xges.sh`, `start_tigramite.sh`, `start_lingam.sh`,
`start_lingam_ts.sh`, `start_causalnex.sh`, `start_utigsp.sh`, `start_loci.sh`, `start_stein.sh`,
and `start_notears.sh` build the separate environments some comparisons need
([setup](README.md#setup)).

## Release cards

Each release card publishes its numbers from a summary in `published/<release>/`. Two more folders
hold measurements no card of their own presents: `published/0.2.0-small-fits/`, the faster small
fits the 0.2.0 card links, and `published/0.4.1/`, DirectLiNGAM's causal-order search in the 0.4.1
fix. The 0.14.0 summary draws its rows from the lane files
[`published/0.14.0/gnies.json`](published/0.14.0/gnies.json), GnIES with gnies,
[`published/0.14.0/ut-igsp.json`](published/0.14.0/ut-igsp.json), UT-IGSP with
graphical-model-learning, [`published/0.14.0/igsp.json`](published/0.14.0/igsp.json), IGSP with
graphical-model-learning, [`published/0.14.0/icp.json`](published/0.14.0/icp.json), ICP with
InvariantCausalPrediction, [`published/0.14.0/camuv.json`](published/0.14.0/camuv.json), CAM-UV with
lingam and causal-learn, and [`published/0.14.0/dynotears.json`](published/0.14.0/dynotears.json),
DYNOTEARS with CausalNex; their comparison pages, [GnIES](comparisons/gnies.md),
[UT-IGSP](comparisons/ut-igsp.md), [IGSP](comparisons/igsp.md), [ICP](comparisons/icp.md),
[CAM-UV](comparisons/camuv.md), and [DYNOTEARS](comparisons/dynotears.md), present them. Some
summary fields come from the runs behind a lane file rather than from the file itself: ICP's rows
split by size (`icp.json` holds each setting's sizes pooled), GnIES's intervals outside its claim,
the speed context (ratio quartiles, per-row fit times, median errors), and the regime, coverage, and
seed notes of GnIES, UT-IGSP, and IGSP. The 0.15.0 summary draws FCI-JCI's rows from
[`published/0.15.0/fci-jci.json`](published/0.15.0/fci-jci.json), FCI-JCI with pcalg, which its
[comparison page](comparisons/fci-jci.md) presents; its median errors, fit times, and the counts and
means of its floor intervals come from the runs behind the file. BOSS-FCI's and GRaSP-FCI's rows
come from [`published/0.15.0/boss-fci-grasp-fci.json`](published/0.15.0/boss-fci-grasp-fci.json),
both hybrids with Tetrad, which their [comparison page](comparisons/boss-fci-grasp-fci.md) presents,
with the median errors and fit times from the runs behind the file. J-PCMCI+'s rows come from
[`published/0.15.0/jpcmciplus.json`](published/0.15.0/jpcmciplus.json), J-PCMCI+ with tigramite,
which its [comparison page](comparisons/jpcmciplus.md) presents, with the median errors, fit times,
link F1, and the counts and means of its floor intervals from the runs behind the file. The 0.16.0
summary draws LOCI's rows from [`published/0.16.0/loci.json`](published/0.16.0/loci.json), LOCI with
its authors' code, which its [comparison page](comparisons/loci.md) presents, with the size ladder
and the other pairwise methods on the same pairs. The path and confounder claims are in no release
summary: [`published/0.16.0/knowledge-paths.json`](published/0.16.0/knowledge-paths.json) holds the
rows of the first eight methods and
[`published/0.17.0/knowledge-paths.json`](published/0.17.0/knowledge-paths.json) those of the other
14, which their [comparison page](comparisons/knowledge-paths.md) presents. The missing-data lanes'
rows are in [`published/0.16.0/missing-data.json`](published/0.16.0/missing-data.json), with their
[comparison page](comparisons/missing-data.md).

| Release | Release card | Summary |
|---|---|---|
| 0.16.0 | [0.16.0 release card](../docs/docs/benchmarks/0.16.0.md) | [`published/0.16.0/summary.json`](published/0.16.0/summary.json) |
| 0.15.0 | [Multi-domain data](../docs/docs/benchmarks/0.15.0.md) | [`published/0.15.0/summary.json`](published/0.15.0/summary.json) |
| 0.14.0 | [Interventional data](../docs/docs/benchmarks/0.14.0.md) | [`published/0.14.0/summary.json`](published/0.14.0/summary.json) |
| 0.13.0 | [Functional causal models](../docs/docs/benchmarks/0.13.0.md) | [`published/0.13.0/summary.json`](published/0.13.0/summary.json) |
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
