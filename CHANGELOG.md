# Changelog

All notable changes to Andrey are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, Andrey is
in alpha, and any minor release may change the API.

## [0.1.0] - 2026-10-08

First alpha release.

### Added

- 18 causal discovery methods in 7 families, each one function call from data to graph. PC, FCI,
  GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM are supported and benchmarked; the other 11 are
  experimental.
- One result type, `StructureOutput`, from every method; `multi_group_direct_lingam` returns one
  per group. Its graph is a `GraphStructure`: a DAG, a CPDAG, or a PAG, with nodes named by the
  data's columns, `oriented_edges()`, and export to NumPy, SciPy, NetworkX, and edge lists. For
  time series it is a `TemporalStructure`, which holds one such graph per lag.
- `andrey.metrics`: score a result against a known graph, with the metrics chosen by the kind of
  graph.
- A command line for scripts and agents: `andrey list`, `andrey run <method>`, `andrey config`,
  and `andrey calibrate`, with JSON output, machine-readable help, and an agent skill
  (`andrey --skill`).
- Optional backends: Numba JIT kernels, and PyTorch on a CUDA GPU; worker processes for GES, GIES,
  GFCI, and hill climbing; every setting is an `ANDREY_*` environment variable.
- `andrey.data`: the Sachs protein-signaling data and four bnlearn networks, downloaded on first
  use and checked by SHA-256.
- `andrey.viz`: SVG drawings of any graph, laid out by Graphviz when it is installed.
- Python 3.11 to 3.14. Linux is supported; macOS and Windows run it without a guarantee yet.

[0.1.0]: https://github.com/Abel-ai-lab/andrey/releases/tag/v0.1.0
