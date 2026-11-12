# Roadmap

Andrey's mission is to make causal discovery more accessible, scalable, and accurate, with native
support for agents. This roadmap lays out the planned work in a rough order; the stages overlap.

Missing a method or a feature you need? Open a
[feature request](https://github.com/Abel-ai-lab/andrey/issues/new/choose).

## 1. Major methods

Andrey covers the main families of causal discovery methods: constraint-based, score-based,
permutation-based, functional causal models, temporal causal discovery, latent variables, and
pairwise direction. Each method ships as experimental and becomes supported once its benchmark
against a reference package is published. The stage ends when every method planned in the table
below is supported. Every method takes continuous data, and many take categorical or mixed data and
missing values; the [API reference](https://andrey.abel.ai/docs/api.html) says which.

<!-- ROADMAP:START -->
<!-- Generated from andrey.spec by apps/build_launch.py; do not edit. -->

| Family | Supported | Planned |
|---|---|---|
| Constraint-based | [`pc`](https://andrey.abel.ai/docs/api/andrey.pc.html), [`fci`](https://andrey.abel.ai/docs/api/andrey.fci.html), [`gfci`](https://andrey.abel.ai/docs/api/andrey.gfci.html), [`cdnod`](https://andrey.abel.ai/docs/api/andrey.cdnod.html), [`fci_jci`](https://andrey.abel.ai/docs/api/andrey.fci_jci.html), [`icp`](https://andrey.abel.ai/docs/api/andrey.icp.html), [`iamb`](https://andrey.abel.ai/docs/api/andrey.iamb.html), [`inter_iamb`](https://andrey.abel.ai/docs/api/andrey.inter_iamb.html), [`mmpc`](https://andrey.abel.ai/docs/api/andrey.mmpc.html) | [`gs`](https://andrey.abel.ai/docs/api/andrey.gs.html) (experimental), [`fast_iamb`](https://andrey.abel.ai/docs/api/andrey.fast_iamb.html) (experimental), [`hiton_pc`](https://andrey.abel.ai/docs/api/andrey.hiton_pc.html) (experimental) |
| Score-based | [`ges`](https://andrey.abel.ai/docs/api/andrey.ges.html), [`gies`](https://andrey.abel.ai/docs/api/andrey.gies.html), [`gnies`](https://andrey.abel.ai/docs/api/andrey.gnies.html), [`xges`](https://andrey.abel.ai/docs/api/andrey.xges.html), [`ages`](https://andrey.abel.ai/docs/api/andrey.ages.html), [`hc`](https://andrey.abel.ai/docs/api/andrey.hc.html), [`tabu`](https://andrey.abel.ai/docs/api/andrey.tabu.html), [`exact_search`](https://andrey.abel.ai/docs/api/andrey.exact_search.html), [`calm`](https://andrey.abel.ai/docs/api/andrey.calm.html), [`notears`](https://andrey.abel.ai/docs/api/andrey.notears.html), [`mmhc`](https://andrey.abel.ai/docs/api/andrey.mmhc.html), [`rsmax2`](https://andrey.abel.ai/docs/api/andrey.rsmax2.html) | [`dagma`](https://andrey.abel.ai/docs/api/andrey.dagma.html) (experimental) |
| Permutation-based | [`boss`](https://andrey.abel.ai/docs/api/andrey.boss.html), [`grasp`](https://andrey.abel.ai/docs/api/andrey.grasp.html), [`boss_fci`](https://andrey.abel.ai/docs/api/andrey.boss_fci.html), [`grasp_fci`](https://andrey.abel.ai/docs/api/andrey.grasp_fci.html), [`igsp`](https://andrey.abel.ai/docs/api/andrey.igsp.html), [`ut_igsp`](https://andrey.abel.ai/docs/api/andrey.ut_igsp.html) |  |
| Functional causal models | [`direct_lingam`](https://andrey.abel.ai/docs/api/andrey.direct_lingam.html), [`ica_lingam`](https://andrey.abel.ai/docs/api/andrey.ica_lingam.html), [`multi_group_direct_lingam`](https://andrey.abel.ai/docs/api/andrey.multi_group_direct_lingam.html), [`cam`](https://andrey.abel.ai/docs/api/andrey.cam.html), [`resit`](https://andrey.abel.ai/docs/api/andrey.resit.html), [`score_matching`](https://andrey.abel.ai/docs/api/andrey.score_matching.html), [`nogam`](https://andrey.abel.ai/docs/api/andrey.nogam.html) |  |
| Temporal causal discovery | [`varma_lingam`](https://andrey.abel.ai/docs/api/andrey.varma_lingam.html), [`longitudinal_lingam`](https://andrey.abel.ai/docs/api/andrey.longitudinal_lingam.html), [`pcmci`](https://andrey.abel.ai/docs/api/andrey.pcmci.html), [`pcmciplus`](https://andrey.abel.ai/docs/api/andrey.pcmciplus.html), [`jpcmciplus`](https://andrey.abel.ai/docs/api/andrey.jpcmciplus.html), [`lpcmci`](https://andrey.abel.ai/docs/api/andrey.lpcmci.html), [`granger`](https://andrey.abel.ai/docs/api/andrey.granger.html), [`dynotears`](https://andrey.abel.ai/docs/api/andrey.dynotears.html) |  |
| Latent variables | [`gin`](https://andrey.abel.ai/docs/api/andrey.gin.html), [`rcd`](https://andrey.abel.ai/docs/api/andrey.rcd.html), [`bottom_up_parce_lingam`](https://andrey.abel.ai/docs/api/andrey.bottom_up_parce_lingam.html), [`camuv`](https://andrey.abel.ai/docs/api/andrey.camuv.html) | FOFC |
| Pairwise direction | [`anm`](https://andrey.abel.ai/docs/api/andrey.anm.html), [`pnl`](https://andrey.abel.ai/docs/api/andrey.pnl.html), [`loci`](https://andrey.abel.ai/docs/api/andrey.loci.html) |  |

<!-- ROADMAP:END -->

## 2. Interventional data

Experiments reveal causal directions that observational data cannot. GIES, GnIES, IGSP, UT-IGSP,
and ICP learn from interventional data, alone or mixed with observational data, and are benchmarked
on interventions simulated by `andrey.data`, data with a known answer. FCI-JCI, BOSS-FCI, and
GRaSP-FCI learn from data from several environments, the score-based and permutation searches take
an environment index, and J-PCMCI+ learns from time series of several datasets, with Sachs's nine
conditions as a real test set.

## 3. Local causal discovery

Local causal discovery methods search around one variable at a time: its Markov blanket, its
direct causes and effects, or the variables to adjust for. Most return that local structure
without learning the whole graph, and some put local searches together into a whole graph. Andrey
already has `iamb` and `inter_iamb`, which learn a variable's Markov blanket, and `mmpc`, which
learns its parents and children. We plan the methods below in this rough order, with no promised
dates, grouped by what they learn.

### Markov blankets, and parents and children

| Method | Paper | Code | What it adds |
|---|---|---|---|
| Grow-Shrink (GS) | Margaritis and Thrun, [Bayesian Network Induction via Local Neighborhoods](https://proceedings.neurips.cc/paper/1999/hash/5d79099fcdf499f12b79770834c0164a-Abstract.html), Advances in Neural Information Processing Systems 12, 1999 | No authors' code found; a port, `gs` in [bnlearn](https://www.bnlearn.com/documentation/man/constraint.html) (R, GPL-2 or later) | The grow phase and shrink phase of a Markov blanket search, the scheme IAMB builds on. |
| Fast-IAMB | Yaramakala and Margaritis, [Speculative Markov Blanket Discovery for Optimal Feature Selection](https://doi.org/10.1109/ICDM.2005.134), IEEE International Conference on Data Mining, 2005 | No authors' code found; a port, `fast.iamb` in [bnlearn](https://www.bnlearn.com/documentation/man/constraint.html) (R, GPL-2 or later) | Adds several variables in one growing step, so it runs fewer tests than IAMB. |
| HITON-PC and HITON-MB | Aliferis, Statnikov, Tsamardinos, Mani, and Koutsoukos, [Local Causal and Markov Blanket Induction for Causal Discovery and Feature Selection for Classification Part I: Algorithms and Empirical Evaluation](https://jmlr.org/papers/v11/aliferis10a.html), Journal of Machine Learning Research 11:171-234, 2010 | `alg_HITON_PC` and `alg_HITON_MB` in [Causal Explorer](https://github.com/mensxmachina/CausalExplorer_1.5) (compiled MATLAB, non-commercial academic license) | Parents and children found by adding and removing candidates in turn, as MMPC's relative, and a Markov blanket built from those sets. |
| MMMB | Tsamardinos, Aliferis, and Statnikov, [Time and sample efficient discovery of Markov blankets and direct causal relations](https://doi.org/10.1145/956750.956838), ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 673-678, 2003 | No authors' code found; a port, [`MMMB.py`](https://github.com/wt-hu/pyCausalFS/blob/master/pyCausalFS/CBD/MBs/MMMB/MMMB.py) in pyCausalFS (Python, no standard license) | A Markov blanket in two steps: the parents and children with MMPC, then the other parents of the children. |
| PCMB | Peña, Nilsson, Björkegren, and Tegnér, [Towards scalable and data efficient learning of Markov boundaries](https://doi.org/10.1016/j.ijar.2006.06.008), International Journal of Approximate Reasoning 45(2):211-232, 2007 | No authors' code found; a port, [`PCMB`](https://github.com/wt-hu/pyCausalFS/tree/master/pyCausalFS/CBD/MBs/PCMB) in pyCausalFS (Python, no standard license) | A Markov blanket built from parents-and-children searches, which condition on smaller sets than IAMB and so need less data. |

### The direct causes and direct effects of a target

| Method | Paper | Code | What it adds |
|---|---|---|---|
| MB-by-MB | Wang, Zhou, Zhao, and Geng, [Discovering and orienting the edges connected to a target variable in a DAG via a sequential local learning approach](https://doi.org/10.1016/j.csda.2014.03.003), Computational Statistics & Data Analysis 77:252-266, 2014 | No authors' code found; a port, [`MBbyMB.py`](https://github.com/wt-hu/pyCausalFS/blob/master/pyCausalFS/LSL/MBs/MBbyMB.py) in pyCausalFS (Python, no standard license) | Local orientation: it learns Markov blankets outward from the target, one variable at a time, until the target's direct causes and direct effects are told apart. |
| CMB | Gao and Ji, [Local Causal Discovery of Direct Causes and Effects](https://proceedings.neurips.cc/paper/2015/hash/fcdf25d6e191893e705819b177cddea0-Abstract.html), Advances in Neural Information Processing Systems 28, 2015 | No authors' code found; a port, [`CMB`](https://github.com/wt-hu/pyCausalFS/tree/master/pyCausalFS/LSL/MBs/CMB) in pyCausalFS (Python, no standard license) | The direct causes and the direct effects of a target, told apart without learning the whole graph. |
| Local discovery with cycles | Dai, Ng, Zheng, Gao, and Zhang, [Local Causal Discovery with Linear non-Gaussian Cyclic Models](https://proceedings.mlr.press/v238/dai24a.html), International Conference on Artificial Intelligence and Statistics, PMLR 238:154-162, 2024 | The authors' [Local-LiNG-Discovery](https://github.com/MarkDana/Local-LiNG-Discovery) (Python, MIT) | The structure around a target from its Markov blanket when effects are linear, noise is non-Gaussian, and feedback cycles are allowed. |
| LocPC-CDE | Loranchet and Assaad, [Local Markov Equivalence for PC-style Local Causal Discovery and Identification of Controlled Direct Effects](https://proceedings.mlr.press/v323/loranchet26a.html), Conference on Causal Learning and Reasoning, PMLR 323:84-118, 2026 | [pyCIPHOD](https://github.com/CIPHOD/pyCIPHOD), the authors' package (Python, MIT) | A PC-style search that stays near a target and learns only as much structure as it takes to decide whether a controlled direct effect on the target can be identified. |

### Adjustment sets for the effect of a treatment on an outcome

| Method | Paper | Code | What it adds |
|---|---|---|---|
| Covariate selection rules | Entner, Hoyer, and Spirtes, [Data-driven covariate selection for nonparametric estimation of causal effects](https://proceedings.mlr.press/v31/entner13a.html), International Conference on Artificial Intelligence and Statistics, PMLR 31:256-264, 2013 | No code found | Two rules of independence tests that decide whether a treatment affects an outcome and, when it does, find an adjustment set among the variables measured before the treatment. |
| LDECC | Gupta, Childers, and Lipton, [Local Causal Discovery for Estimating Causal Effects](https://proceedings.mlr.press/v213/gupta23b.html), Conference on Causal Learning and Reasoning, PMLR 213:408-447, 2023 | The authors' [local-causal-discovery](https://github.com/acmi-lab/local-causal-discovery) (Python, no license stated) | Local discovery for effect estimation: it orients the parents of a treatment from the structure around it, which gives the possible values of the treatment's effect. |
| LDP | Maasch, Pan, Gupta, Kuleshov, Gan, and Wang, [Local Discovery by Partitioning: Polynomial-Time Causal Discovery Around Exposure-Outcome Pairs](https://proceedings.mlr.press/v244/maasch24a.html) ([arXiv](https://arxiv.org/abs/2310.17816)), Conference on Uncertainty in Artificial Intelligence, PMLR 244:2350-2382, 2024 | The authors' [ldp](https://github.com/jmaasch/ldp) (Python, no license stated) | A valid adjustment set for an exposure and an outcome, with hidden common causes allowed, in a number of tests that grows with the square of the number of variables. |
| LD3 | Maasch, Gan, Chen, Orfanoudaki, Akpinar, and Wang, [Local Causal Discovery for Structural Evidence of Direct Discrimination](https://doi.org/10.1609/aaai.v39i18.34130), Proceedings of the AAAI Conference on Artificial Intelligence 39(18):19349-19357, 2025 | The authors' [LD3](https://github.com/jmaasch/LD3) (Python, no license stated) | The parents of an outcome, in a number of tests that grows in proportion to the number of variables, with hidden common causes allowed when every parent of the outcome is measured. The parents give a valid adjustment set for the direct effect of an exposure on the outcome. |
| SNAP | Schubert, Claassen, and Magliacane, [SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph](https://proceedings.mlr.press/v258/schubert25a.html), International Conference on Artificial Intelligence and Statistics, PMLR 258:3340-3348, 2025 | The authors' [snap](https://github.com/Matyasch/snap) (Python, BSD-2-Clause) | For effects among a few target variables: it removes, step by step, the variables that are not ancestors of any target, so the tests that remain are few and small. |
| LOAD | Schubert, Claassen, and Magliacane, [Local Causal Discovery for Statistically Efficient Causal Inference](https://proceedings.mlr.press/v300/schubert26a.html), International Conference on Artificial Intelligence and Statistics, PMLR 300:1990-1998, 2026 | The authors' [load](https://github.com/Matyasch/load) (Python, BSD-2-Clause) | A local search around a treatment and an outcome that returns the adjustment set giving the most precise estimate of the effect, where other local methods return any valid set. |

### Hidden variables and selection bias

| Method | Paper | Code | What it adds |
|---|---|---|---|
| MMB-by-MMB | Xie, Li, Wu, Zeng, Liu, and Geng, [Local Causal Structure Learning in the Presence of Latent Variables](https://proceedings.mlr.press/v235/xie24f.html), International Conference on Machine Learning, PMLR 235:54511-54530, 2024 | The authors' [MMB-by-MMB](https://github.com/fengxie009/MMB-by-MMB) (Python, no license stated) | MB-by-MB with hidden variables allowed: the direct causes and direct effects of a target, learned outward one Markov blanket at a time. |
| Y-structures under selection bias | Versteeg, Mooij, and Zhang, [Local Constraint-Based Causal Discovery under Selection Bias](https://proceedings.mlr.press/v177/versteeg22a.html), Conference on Causal Learning and Reasoning, PMLR 177:840-860, 2022 | The authors' [sbcd](https://github.com/philipversteeg/sbcd) (R, BSD-3-Clause) | A pattern of independences among four variables that shows one causes another even when the sample was selected and hidden common causes exist. |

### Whole graphs from local searches

| Method | Paper | Code | What it adds |
|---|---|---|---|
| LHTS and NHTS | Hiremath, Maasch, Gao, Ghosal, and Gan, [Hybrid Top-Down Global Causal Discovery with Local Search for Linear and Nonlinear Additive Noise Models](https://doi.org/10.52202/079017-4231), Advances in Neural Information Processing Systems 37, pages 133147-133179, 2024 | The authors' [hybrid-discovery](https://github.com/Sujai1/hybrid-discovery) (Python, no license stated) | A whole graph built from local searches: a top-down causal order found from ancestor relations, for linear and for nonlinear effects with additive noise, then spurious edges removed with tests on small local sets. |

## 4. Very large causal discovery

Very large causal discovery means learning graphs of more than 1,000 variables. We use sparsity and
parallel hardware to scale past that size, toward millions of variables, starting with FGES, a GES
built for large sparse graphs. Andrey already runs the skeleton search of PC and of the methods
built on it, and GES, on parallel CPU workers. We plan the methods below in this rough order, with
no promised dates.

| Method | Paper | Code | What it adds |
|---|---|---|---|
| FGES | Ramsey, Glymour, Sanchez-Romero, and Glymour, [A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images](https://doi.org/10.1007/s41060-016-0032-z), International Journal of Data Science and Analytics 3:121-129, 2017 | [`Fges.java`](https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Fges.java) in Tetrad, the authors' package (Java, GPL-3.0) | GES arranged for large sparse graphs: it caches scores, computes them in parallel, and can skip pairs of uncorrelated variables. |
| ARGES | Nandy, Hauser, and Maathuis, [High-dimensional consistency in score-based and hybrid structure learning](https://doi.org/10.1214/17-AOS1654), The Annals of Statistics 46(6A):3151-3183, 2018 | `ges(adaptive = ...)` in [pcalg](https://cran.r-project.org/package=pcalg), the authors' package (R, GPL-2 or later) | GES restricted to an estimated skeleton, with the restriction relaxed as the search goes, which keeps the search consistent on sparse high-dimensional data. |
| cuPC | Zarebavani, Jafarinejad, Hashemi, and Salehkaleybar, [cuPC: CUDA-Based Parallel PC Algorithm for Causal Structure Learning on GPU](https://doi.org/10.1109/TPDS.2019.2939126), IEEE Transactions on Parallel and Distributed Systems 31(3):530-542, 2020 | The authors' [cupc](https://github.com/LIS-Laboratory/cupc) (CUDA and R, GPL-3.0) | The skeleton search of PC on a GPU: the tests of one conditioning-set size run in parallel on Gaussian data. |
| GPU-accelerated PC for discrete data | Hagedorn and Huegle, [GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data](https://doi.org/10.1137/1.9781611976700.5), SIAM International Conference on Data Mining, pages 37-45, 2021 | `DiscretePC` in [gpucsl](https://github.com/hpi-epic/gpucsl), the authors' package (Python and CUDA, MIT) | The same GPU skeleton search for categorical data. |
| FCI+ | Claassen, Mooij, and Heskes, [Learning Sparse Causal Models is not NP-hard](https://arxiv.org/abs/1309.6824), Conference on Uncertainty in Artificial Intelligence, 2013 | `fciPlus` in [pcalg](https://cran.r-project.org/package=pcalg), written with one of the authors (R, GPL-2 or later) | A search that allows hidden variables and runs a number of tests polynomial in the number of variables when the graph is sparse. |
| Causal graph partitioning | Shah, DePavia, Hudson, Foster, and Stevens, [Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning](https://openreview.net/forum?id=FecsgPCOHk), Transactions on Machine Learning Research, 2025 | The authors' [causal_discovery_via_partitioning](https://github.com/shahashka/causal_discovery_via_partitioning) (Python, no license stated) | Divide and conquer: it partitions the variables with a graph of candidate edges, learns each part on its own, and merges the results. |

## 5. Agentic causal discovery

Agents that run causal discovery end to end, and ensembles of methods for real-world analysis.

## 6. More platforms

Full support on macOS and Windows, and Apple GPUs.
