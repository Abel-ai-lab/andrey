# Contributing to Andrey

This page covers reporting a problem, setting up, and what a pull request needs.
[`DEVELOPMENT.md`](https://github.com/Abel-ai-lab/andrey/blob/main/DEVELOPMENT.md) explains how the
code fits together and how to make common changes, such as adding a method;
[`docs/DEVELOPMENT.md`](https://github.com/Abel-ai-lab/andrey/blob/main/docs/DEVELOPMENT.md)
covers the website.

## Report a problem

Open an issue with the bug or feature form in the
[issue tracker](https://github.com/Abel-ai-lab/andrey/issues/new/choose). For a wrong result,
include the call, the shape of the data, and the output of `andrey config`.

## Set up

The project uses [`uv`](https://docs.astral.sh/uv/). An editable install with the dev tooling:

```shell
git clone https://github.com/Abel-ai-lab/andrey
cd andrey
uv sync                       # base + dev group (ruff, ty, pytest)
uv sync --extra numba         # optional: CPU JIT for GES reachability, up to 64 variables
uv sync --extra torch         # optional: the torch backend (CPU by default; see below)
```

The published wheel is torch-free, and `uv sync --extra torch` installs the CPU build of torch:
`uv sync` resolves one lock for every machine, so it cannot pick a build for the local GPU. For a
GPU build, replace torch after syncing. `--torch-backend=auto` reads the installed driver and picks
the matching wheel:

```shell
uv sync --extra torch
uv pip install --reinstall --torch-backend=auto torch
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The GPU build lives only in the active `.venv`; a later `uv sync` restores the locked CPU build, so
run with `uv run --no-sync` until then.

## Before a pull request

Run the checks CI runs. The optional methods' tests need the extras:

```shell
uv sync --all-extras
uv run --no-sync ruff format --check .    # formatting
uv run --no-sync ruff check .             # lint
uv run --no-sync ty check                 # types
uv run --no-sync pytest tests             # every test, including the drift guards
```

`prek install` (or `pre-commit install`) adds the hooks in `.pre-commit-config.yaml`, including
gitleaks, to this checkout, so they run on every commit; `prek run --all-files` runs them once.

## Pull requests

- One change per pull request, with the tests that show it works.
- When the public API changes, update its docstrings and the API reference
  (`docs/docs/code/index.md`) in the same pull request.
- Everything in `andrey.__all__` is public and documented. Names that start with `_` are internal
  and may change without notice; do not import them from outside the package.
- Code, data, and assets copied from another project keep their license notice, as
  `site/brand/fonts/OFL.txt` does for the site's fonts.
