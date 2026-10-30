# Roadmap

Andrey's mission is to make causal discovery more accessible, scalable, and accurate, with native
support for agents. This roadmap lays out the planned work, in order.

Missing a method or a feature you need? Open a
[feature request](https://github.com/Abel-ai-lab/andrey/issues/new/choose).

## 1. Every major causal discovery method, benchmarked

Each method ships as experimental and becomes supported once its benchmarks against a reference
package are published. Every method takes continuous data, and many take categorical or mixed data
and missing values; the [API reference](https://andrey.abel.ai/docs/api.html) says which.

<!-- ROADMAP:START -->
<!-- Generated from andrey.spec by apps/build_launch.py; do not edit. -->

| Family | Supported | Planned |
|---|---|---|
| Constraint-based | [`pc`](https://andrey.abel.ai/docs/api/andrey.pc.html), [`fci`](https://andrey.abel.ai/docs/api/andrey.fci.html), [`gfci`](https://andrey.abel.ai/docs/api/andrey.gfci.html), [`cdnod`](https://andrey.abel.ai/docs/api/andrey.cdnod.html), [`inter_iamb`](https://andrey.abel.ai/docs/api/andrey.inter_iamb.html) | [`iamb`](https://andrey.abel.ai/docs/api/andrey.iamb.html) (experimental), [`mmpc`](https://andrey.abel.ai/docs/api/andrey.mmpc.html) (experimental), JCI |
| Score-based | [`ges`](https://andrey.abel.ai/docs/api/andrey.ges.html), [`gies`](https://andrey.abel.ai/docs/api/andrey.gies.html), [`xges`](https://andrey.abel.ai/docs/api/andrey.xges.html), [`hc`](https://andrey.abel.ai/docs/api/andrey.hc.html), [`tabu`](https://andrey.abel.ai/docs/api/andrey.tabu.html), [`exact_search`](https://andrey.abel.ai/docs/api/andrey.exact_search.html), [`calm`](https://andrey.abel.ai/docs/api/andrey.calm.html), [`mmhc`](https://andrey.abel.ai/docs/api/andrey.mmhc.html), [`rsmax2`](https://andrey.abel.ai/docs/api/andrey.rsmax2.html) | [`notears`](https://andrey.abel.ai/docs/api/andrey.notears.html) (experimental), [`dagma`](https://andrey.abel.ai/docs/api/andrey.dagma.html) (experimental), CAM, FGES, NOTEARS-MLP, GraN-DAG |
| Permutation-based | [`boss`](https://andrey.abel.ai/docs/api/andrey.boss.html), [`grasp`](https://andrey.abel.ai/docs/api/andrey.grasp.html) | UT-IGSP |
| Linear non-Gaussian | [`direct_lingam`](https://andrey.abel.ai/docs/api/andrey.direct_lingam.html), [`ica_lingam`](https://andrey.abel.ai/docs/api/andrey.ica_lingam.html), [`multi_group_direct_lingam`](https://andrey.abel.ai/docs/api/andrey.multi_group_direct_lingam.html) |  |
| Temporal causal discovery | [`varma_lingam`](https://andrey.abel.ai/docs/api/andrey.varma_lingam.html), [`longitudinal_lingam`](https://andrey.abel.ai/docs/api/andrey.longitudinal_lingam.html), [`pcmci`](https://andrey.abel.ai/docs/api/andrey.pcmci.html), [`pcmciplus`](https://andrey.abel.ai/docs/api/andrey.pcmciplus.html), [`lpcmci`](https://andrey.abel.ai/docs/api/andrey.lpcmci.html), [`granger`](https://andrey.abel.ai/docs/api/andrey.granger.html) | DYNOTEARS, J-PCMCI+ |
| Latent variables | [`gin`](https://andrey.abel.ai/docs/api/andrey.gin.html), [`rcd`](https://andrey.abel.ai/docs/api/andrey.rcd.html), [`bottom_up_parce_lingam`](https://andrey.abel.ai/docs/api/andrey.bottom_up_parce_lingam.html) | CAM-UV, FOFC |
| Pairwise direction | [`anm`](https://andrey.abel.ai/docs/api/andrey.anm.html), [`pnl`](https://andrey.abel.ai/docs/api/andrey.pnl.html) | RESIT |

<!-- ROADMAP:END -->

## 2. Local causal discovery

Local causal discovery learns the structure around one variable, such as its direct causes and
effects or its Markov blanket, without learning the whole graph.

## 3. Agentic causal discovery

Agents that run causal discovery end to end, and ensembles of methods for real-world analysis.

## 4. Very large causal discovery

Use sparsity to scale causal discovery to millions of variables.

## 5. More platforms

Full support on macOS and Windows, and Apple GPUs.

## Alongside every stage

Fixes and speed for the supported methods, and more benchmarks, in patch releases.
