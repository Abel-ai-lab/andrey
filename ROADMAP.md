# Roadmap

Andrey's mission is to make causal discovery more accessible, scalable, and accurate, with native
support for agents. This roadmap lays out the planned work in a rough order; the stages overlap.

Missing a method or a feature you need? Open a
[feature request](https://github.com/Abel-ai-lab/andrey/issues/new/choose).

## 1. Every major causal discovery method, benchmarked

Each method ships as experimental and becomes supported once its benchmarks against a reference
package are published. The stage ends when every method planned below is supported. Every method
takes continuous data, and many take categorical or mixed data and missing values; the [API
reference](https://andrey.abel.ai/docs/api.html) says which.

<!-- ROADMAP:START -->
<!-- Generated from andrey.spec by apps/build_launch.py; do not edit. -->

| Family | Supported | Planned |
|---|---|---|
| Constraint-based | [`pc`](https://andrey.abel.ai/docs/api/andrey.pc.html), [`fci`](https://andrey.abel.ai/docs/api/andrey.fci.html), [`gfci`](https://andrey.abel.ai/docs/api/andrey.gfci.html), [`cdnod`](https://andrey.abel.ai/docs/api/andrey.cdnod.html), [`fci_jci`](https://andrey.abel.ai/docs/api/andrey.fci_jci.html), [`icp`](https://andrey.abel.ai/docs/api/andrey.icp.html), [`iamb`](https://andrey.abel.ai/docs/api/andrey.iamb.html), [`inter_iamb`](https://andrey.abel.ai/docs/api/andrey.inter_iamb.html), [`mmpc`](https://andrey.abel.ai/docs/api/andrey.mmpc.html) |  |
| Score-based | [`ges`](https://andrey.abel.ai/docs/api/andrey.ges.html), [`gies`](https://andrey.abel.ai/docs/api/andrey.gies.html), [`gnies`](https://andrey.abel.ai/docs/api/andrey.gnies.html), [`xges`](https://andrey.abel.ai/docs/api/andrey.xges.html), [`hc`](https://andrey.abel.ai/docs/api/andrey.hc.html), [`tabu`](https://andrey.abel.ai/docs/api/andrey.tabu.html), [`exact_search`](https://andrey.abel.ai/docs/api/andrey.exact_search.html), [`calm`](https://andrey.abel.ai/docs/api/andrey.calm.html), [`mmhc`](https://andrey.abel.ai/docs/api/andrey.mmhc.html), [`rsmax2`](https://andrey.abel.ai/docs/api/andrey.rsmax2.html) | [`notears`](https://andrey.abel.ai/docs/api/andrey.notears.html) (experimental), [`dagma`](https://andrey.abel.ai/docs/api/andrey.dagma.html) (experimental) |
| Permutation-based | [`boss`](https://andrey.abel.ai/docs/api/andrey.boss.html), [`grasp`](https://andrey.abel.ai/docs/api/andrey.grasp.html), [`boss_fci`](https://andrey.abel.ai/docs/api/andrey.boss_fci.html), [`grasp_fci`](https://andrey.abel.ai/docs/api/andrey.grasp_fci.html), [`igsp`](https://andrey.abel.ai/docs/api/andrey.igsp.html), [`ut_igsp`](https://andrey.abel.ai/docs/api/andrey.ut_igsp.html) |  |
| Functional causal models | [`direct_lingam`](https://andrey.abel.ai/docs/api/andrey.direct_lingam.html), [`ica_lingam`](https://andrey.abel.ai/docs/api/andrey.ica_lingam.html), [`multi_group_direct_lingam`](https://andrey.abel.ai/docs/api/andrey.multi_group_direct_lingam.html), [`cam`](https://andrey.abel.ai/docs/api/andrey.cam.html), [`resit`](https://andrey.abel.ai/docs/api/andrey.resit.html) |  |
| Temporal causal discovery | [`varma_lingam`](https://andrey.abel.ai/docs/api/andrey.varma_lingam.html), [`longitudinal_lingam`](https://andrey.abel.ai/docs/api/andrey.longitudinal_lingam.html), [`pcmci`](https://andrey.abel.ai/docs/api/andrey.pcmci.html), [`pcmciplus`](https://andrey.abel.ai/docs/api/andrey.pcmciplus.html), [`jpcmciplus`](https://andrey.abel.ai/docs/api/andrey.jpcmciplus.html), [`lpcmci`](https://andrey.abel.ai/docs/api/andrey.lpcmci.html), [`granger`](https://andrey.abel.ai/docs/api/andrey.granger.html), [`dynotears`](https://andrey.abel.ai/docs/api/andrey.dynotears.html) |  |
| Latent variables | [`gin`](https://andrey.abel.ai/docs/api/andrey.gin.html), [`rcd`](https://andrey.abel.ai/docs/api/andrey.rcd.html), [`bottom_up_parce_lingam`](https://andrey.abel.ai/docs/api/andrey.bottom_up_parce_lingam.html), [`camuv`](https://andrey.abel.ai/docs/api/andrey.camuv.html) | FOFC |
| Pairwise direction | [`anm`](https://andrey.abel.ai/docs/api/andrey.anm.html), [`pnl`](https://andrey.abel.ai/docs/api/andrey.pnl.html) | [`loci`](https://andrey.abel.ai/docs/api/andrey.loci.html) (experimental) |

<!-- ROADMAP:END -->

## 2. Interventional data

Experiments reveal causal directions that observational data cannot. GIES, GnIES, IGSP, UT-IGSP,
and ICP learn from interventional data, alone or mixed with observational data, and are benchmarked
on interventions simulated by `andrey.data`, data with a known answer. FCI-JCI, BOSS-FCI, and
GRaSP-FCI learn from data from several environments, the score-based and permutation searches take
an environment index, and J-PCMCI+ learns from time series of several datasets, with Sachs's nine
conditions as a real test set.

## 3. Local causal discovery

Local causal discovery learns the structure around one variable, such as its direct causes and
effects or its Markov blanket, without learning the whole graph. Andrey already has `iamb` and
`inter_iamb`, which learn a variable's Markov blanket, and `mmpc`, which learns its parents and
children. We plan the methods below in this rough order, with no promised dates.

| Method | Paper | Code | What it adds |
|---|---|---|---|
| Grow-Shrink (GS) | Margaritis and Thrun, [Bayesian Network Induction via Local Neighborhoods](https://proceedings.neurips.cc/paper/1999/hash/5d79099fcdf499f12b79770834c0164a-Abstract.html), Advances in Neural Information Processing Systems 12, 1999 | No authors' code found; a port, `gs` in [bnlearn](https://www.bnlearn.com/documentation/man/constraint.html) (R, GPL-2 or later) | The grow phase and shrink phase of a Markov blanket search, the scheme IAMB builds on. |
| Fast-IAMB | Yaramakala and Margaritis, [Speculative Markov Blanket Discovery for Optimal Feature Selection](https://doi.org/10.1109/ICDM.2005.134), IEEE International Conference on Data Mining, 2005 | No authors' code found; a port, `fast.iamb` in [bnlearn](https://www.bnlearn.com/documentation/man/constraint.html) (R, GPL-2 or later) | Adds several variables in one growing step, so it runs fewer tests than IAMB. |
| HITON-PC and HITON-MB | Aliferis, Statnikov, Tsamardinos, Mani, and Koutsoukos, [Local Causal and Markov Blanket Induction for Causal Discovery and Feature Selection for Classification Part I: Algorithms and Empirical Evaluation](https://jmlr.org/papers/v11/aliferis10a.html), Journal of Machine Learning Research 11:171-234, 2010 | `alg_HITON_PC` and `alg_HITON_MB` in [Causal Explorer](https://github.com/mensxmachina/CausalExplorer_1.5) (compiled MATLAB, non-commercial academic license) | Parents and children found by adding and removing candidates in turn, as MMPC's relative, and a Markov blanket built from those sets. |
| PCMB | Peña, Nilsson, Björkegren, and Tegnér, [Towards scalable and data efficient learning of Markov boundaries](https://doi.org/10.1016/j.ijar.2006.06.008), International Journal of Approximate Reasoning 45(2):211-232, 2007 | No authors' code found; a port, [`PCMB`](https://github.com/wt-hu/pyCausalFS/tree/master/pyCausalFS/CBD/MBs/PCMB) in pyCausalFS (Python, no standard license) | A Markov blanket built from parents-and-children searches, which condition on smaller sets than IAMB and so need less data. |
| MB-by-MB | Wang, Zhou, Zhao, and Geng, [Discovering and orienting the edges connected to a target variable in a DAG via a sequential local learning approach](https://doi.org/10.1016/j.csda.2014.03.003), Computational Statistics & Data Analysis 77:252-266, 2014 | No authors' code found; a port, [`MBbyMB.py`](https://github.com/wt-hu/pyCausalFS/blob/master/pyCausalFS/LSL/MBs/MBbyMB.py) in pyCausalFS (Python, no standard license) | Local orientation: it learns Markov blankets outward from the target, one variable at a time, until the target's direct causes and direct effects are told apart. |
| LDECC | Gupta, Childers, and Lipton, [Local Causal Discovery for Estimating Causal Effects](https://proceedings.mlr.press/v213/gupta23b.html), Conference on Causal Learning and Reasoning, PMLR 213:408-447, 2023 | The authors' [local-causal-discovery](https://github.com/acmi-lab/local-causal-discovery) (Python, no license stated) | Local discovery for effect estimation: it orients the parents of a treatment from the structure around it, which gives the possible values of the treatment's effect. |
| LDP | Maasch, Pan, Gupta, Kuleshov, Gan, and Wang, [Local Discovery by Partitioning: Polynomial-Time Causal Discovery Around Exposure-Outcome Pairs](https://proceedings.mlr.press/v244/maasch24a.html), Conference on Uncertainty in Artificial Intelligence, PMLR 244:2350-2382, 2024 | The authors' [ldp](https://github.com/jmaasch/ldp) (Python, no license stated) | A valid adjustment set for an exposure and an outcome, with hidden common causes allowed, in a number of tests that grows with the square of the number of variables. |

## 4. Agentic causal discovery

Agents that run causal discovery end to end, and ensembles of methods for real-world analysis.

## 5. Very large causal discovery

Use sparsity and parallel hardware to scale causal discovery to millions of variables, starting
with FGES, a GES built for large sparse graphs. Andrey already runs the skeleton search of PC and of
the methods built on it, and GES, on parallel CPU workers. We plan the methods below in this rough
order, with no promised dates.

| Method | Paper | Code | What it adds |
|---|---|---|---|
| FGES | Ramsey, Glymour, Sanchez-Romero, and Glymour, [A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images](https://doi.org/10.1007/s41060-016-0032-z), International Journal of Data Science and Analytics 3:121-129, 2017 | [`Fges.java`](https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Fges.java) in Tetrad, the authors' package (Java, GPL-3.0) | GES arranged for large sparse graphs: it caches scores, computes them in parallel, and can skip pairs of uncorrelated variables. |
| ARGES | Nandy, Hauser, and Maathuis, [High-dimensional consistency in score-based and hybrid structure learning](https://doi.org/10.1214/17-AOS1654), The Annals of Statistics 46(6A):3151-3183, 2018 | `ges(adaptive = ...)` in [pcalg](https://cran.r-project.org/package=pcalg), the authors' package (R, GPL-2 or later) | GES restricted to an estimated skeleton, with the restriction relaxed as the search goes, which keeps the search consistent on sparse high-dimensional data. |
| cuPC | Zarebavani, Jafarinejad, Hashemi, and Salehkaleybar, [cuPC: CUDA-Based Parallel PC Algorithm for Causal Structure Learning on GPU](https://doi.org/10.1109/TPDS.2019.2939126), IEEE Transactions on Parallel and Distributed Systems 31(3):530-542, 2020 | The authors' [cupc](https://github.com/LIS-Laboratory/cupc) (CUDA and R, GPL-3.0) | The skeleton search of PC on a GPU: the tests of one conditioning-set size run in parallel on Gaussian data. |
| GPU-accelerated PC for discrete data | Hagedorn and Huegle, [GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data](https://doi.org/10.1137/1.9781611976700.5), SIAM International Conference on Data Mining, pages 37-45, 2021 | `DiscretePC` in [gpucsl](https://github.com/hpi-epic/gpucsl), the authors' package (Python and CUDA, MIT) | The same GPU skeleton search for categorical data. |
| FCI+ | Claassen, Mooij, and Heskes, [Learning Sparse Causal Models is not NP-hard](https://arxiv.org/abs/1309.6824), Conference on Uncertainty in Artificial Intelligence, 2013 | `fciPlus` in [pcalg](https://cran.r-project.org/package=pcalg), written with one of the authors (R, GPL-2 or later) | A search that allows hidden variables and runs a number of tests polynomial in the number of variables when the graph is sparse. |
| Causal graph partitioning | Shah, DePavia, Hudson, Foster, and Stevens, [Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning](https://openreview.net/forum?id=FecsgPCOHk), Transactions on Machine Learning Research, 2025 | The authors' [causal_discovery_via_partitioning](https://github.com/shahashka/causal_discovery_via_partitioning) (Python, no license stated) | Divide and conquer: it partitions the variables with a graph of candidate edges, learns each part on its own, and merges the results. |

## 6. More platforms

Full support on macOS and Windows, and Apple GPUs.
