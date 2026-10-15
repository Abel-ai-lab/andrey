# Changelog

All notable changes to Andrey are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, Andrey is
in alpha, and any minor release may change the API. From 0.2.0, a release opens with Highlights
and, when a change breaks the API, Breaking, two sections Keep a Changelog does not define; each
change names its pull request.

## [Unreleased]

### Removed

- `andrey --llms`: `andrey list` looks up the methods, `andrey config --help` the settings, and
  `andrey --skill` holds both in the agent guide. ([#122])

## [0.4.0] - 2026-10-15

### Highlights

- `anm` and `pnl` are supported. On cause-effect pairs, ANM names the same direction as
  causal-learn on every pair and runs about 2x faster; PNL runs up to 1,470x faster and names the
  true direction on more pairs. The 0.4.0 release card gives the numbers and each method's limits.
  ([#95])

### Added

- `anm`: a direction test for a pair of variables under the additive noise model, returning a
  two-node DAG and both p-values, as `pnl` does. ([#95])

### Changed

- With `alpha=None`, `pnl` breaks a tie between equal p-values by the HSIC statistics, which
  `metadata` now holds, instead of drawing no edge. ([#95])

## [0.3.0] - 2026-10-13

### Highlights

- `hc`, `tabu`, and `xges` are supported. On linear-Gaussian data, XGES returns the same graphs as
  xges and runs up to 140x faster; Tabu search and hill climbing return the same equivalence
  classes as bnlearn on most datasets and run up to 561x and 906x faster. The 0.3.0 release card
  gives the numbers and each method's limits. ([#64])

### Breaking

- `hc` runs in one process: its worker processes were slower at every measured size.
  `ANDREY_HC_PARALLEL_MIN_WORK` and `andrey calibrate --method hc` are removed. ([#56])

### Added

- `tabu`: hill climbing that keeps searching past a local optimum and returns the best DAG it
  visits, as a CPDAG. ([#56])
- `xges`: extremely greedy equivalence search, returning a CPDAG and its BIC score. ([#57])

### Changed

- `hc` returns the same graphs in less time. ([#56])
- `hc` sets no move limit by default (`max_iter=None`), so large graphs are searched to a local
  optimum. ([#56])

## [0.2.1] - 2026-10-12

### Fixed

- Result summaries show a long metadata list as its length instead of printing it in full. ([#59])

## [0.2.0] - 2026-10-12

### Highlights

- `rcd` and `bottom_up_parce_lingam` are supported: on data with hidden common causes they return
  the same graphs as lingam and run 1.7-10.8x faster. The 0.2.0 release card gives the numbers and
  each method's limits. ([#31])
- Small fits run several times faster. At 10 variables, PC and FCI take about a twelfth of the
  time, hill climbing an eighth, and GES two fifths. Larger fits are no slower, and graphs are
  unchanged. ([#33])

### Added

- Three methods for data with hidden common causes: `rcd`, `bottom_up_parce_lingam`, and the
  experimental `camuv`. Each returns a PAG over the observed variables. `rcd` and `camuv` list the
  pairs a hidden variable links in `metadata["confounded_pairs"]`; `bottom_up_parce_lingam` leaves
  the variables whose order such a cause hides unordered. ([#18], [#31], [#53])
- `hc` takes `max_iter`, its move limit, and warns with the new `andrey.SearchLimitWarning` when the
  search stops there. ([#14])

### Fixed

- Every method checks its data before the fit. Data with one dimension, fewer than 2 rows, `NaN` or
  infinity, or a constant column raises a `ValueError` that names the argument, instead of an error
  from NumPy or scikit-learn or a graph from data no method can use. ([#14])
- A single column gives a one-node graph in every method, and `cdnod` with one domain gives the PC
  graph. ([#14])
- Settings outside their range raise an error naming them: `n_lags`, `order`, `max_iter`, and
  `gin`'s `alpha` and `labels`. ([#14])
- `GraphStructure.from_numpy` rejects an unknown `kind` and a self-loop not marked as an arrowhead,
  and `andrey.viz.draw` points a `TemporalStructure` to `lag(k)` or `summary_graph()`. ([#14])
- `calm` no longer suggests a CUDA GPU, which it does not use. ([#14])

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

[Unreleased]: https://github.com/Abel-ai-lab/andrey/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Abel-ai-lab/andrey/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Abel-ai-lab/andrey/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Abel-ai-lab/andrey/releases/tag/v0.1.0
[#14]: https://github.com/Abel-ai-lab/andrey/pull/14
[#18]: https://github.com/Abel-ai-lab/andrey/pull/18
[#31]: https://github.com/Abel-ai-lab/andrey/pull/31
[#33]: https://github.com/Abel-ai-lab/andrey/pull/33
[#53]: https://github.com/Abel-ai-lab/andrey/pull/53
[#56]: https://github.com/Abel-ai-lab/andrey/pull/56
[#57]: https://github.com/Abel-ai-lab/andrey/pull/57
[#59]: https://github.com/Abel-ai-lab/andrey/pull/59
[#64]: https://github.com/Abel-ai-lab/andrey/pull/64
[#122]: https://github.com/Abel-ai-lab/andrey/pull/122
[#95]: https://github.com/Abel-ai-lab/andrey/pull/95
