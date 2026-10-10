---
name: apps
description: Run the demo app locally and regenerate the README benchmark block.
meta:
  type: guide
  status: active
---

# Andrey apps

`live-discovery/` contains the Gradio Race, Benchmarks, Explore, and Learn tabs.
`build_assets.py` and `compare.py` also write the interactive examples in the documentation's
Examples gallery. Notebooks live in `examples/`. None of these ship in the wheel or sdist.

## Run locally

```shell
uv sync --group docs
uv run --group docs --with-requirements apps/live-discovery/requirements.txt python apps/build_assets.py
uv run --with-requirements apps/live-discovery/requirements.txt python apps/live-discovery/app.py
```

The apps' packages are listed in `live-discovery/requirements.txt`, which the hosted app also
installs. uv adds them in a temporary environment on top of the project's, so they stay out of
`uv.lock`.

The app uses the homepage's chart renderer and
`benchmarks/published/alpha-2026-09/summary.json`. Generated files go in the ignored
`apps/live-discovery/data/` directory. A missing ratio means a package has no time there (it
passed its time cap) or no datasets could be paired.

Workers import and warm each package before accepting fits. Fits run sequentially with one numeric
thread; each has a 60-second deadline. A timeout kills and reaps the worker's process group, and
the next request starts a new worker. A failed fit reports its error message and keeps the worker.
Startup has a separate 120-second deadline. Workers send protocol messages on a private pipe, so
library output on stdout cannot corrupt them.

PC and DirectLiNGAM use structural SHD. FCI uses endpoint SHD against the latent-projected truth
PAG. A timing ratio is shown whenever both fits finish; each estimate's SHD is shown beside it.
gCastle is offered only if its worker imports and warms successfully.

CSV uploads require a UTF-8 header and numeric, finite values: at most 5 MB, 50 columns, and
20,000 rows, with at least three more rows than columns. Constant, duplicate, and perfectly
correlated columns are rejected.

## Compare on one dataset

The gallery's comparison computes Andrey's graphs at build time and reads causal-learn's from
`compare-causal-learn.json`, so the docs build does not need causal-learn. The recording holds the
SHA-256 of the data it was made on; the build refuses it once the dataset changes. Record it again
with causal-learn installed:

```shell
uv run --group docs --with-requirements apps/live-discovery/requirements.txt python apps/compare.py --record
```

## README benchmark block

Regenerate or check the README's methods and performance blocks. Both modes also fail when the
hand-written pitch does not state the summary's headline and the registry's method count:

```shell
uv run --group docs python apps/build_launch.py --readme
uv run --group docs python apps/build_launch.py --check-readme
```

The README's images are in `.github/readme/`: the speedup chart in both themes, which `--readme`
writes from the summary and `--check-readme` checks, and the banner, which only `--banner` redraws
(it needs the docs group's renderer). The README is also the package-index description, so every
link is absolute, and the package build points each image at the release tag on GitHub. The
recording is not linked.

## Checks

```shell
uv run --group docs --with-requirements apps/live-discovery/requirements.txt pytest \
  tests/unit/test_demo_safeguards.py \
  tests/unit/test_meek_trace.py tests/unit/test_pc_trace.py tests/unit/test_launch_material.py \
  tests/unit/test_compare_example.py
uv run --group docs python -m pytest docs/test_built_site.py
```
