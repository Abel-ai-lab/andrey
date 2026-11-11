# Contributing to Andrey

Thank you for helping. This page lists the ways you can help, then walks through one contribution
from a fork to a merged pull request. You do not need to know the code base to start.
[`DEVELOPMENT.md`](https://github.com/Abel-ai-lab/Andrey/blob/main/DEVELOPMENT.md) explains how the
code fits together, how to make common changes such as adding a method, and how a release is put
together;
[`docs/DEVELOPMENT.md`](https://github.com/Abel-ai-lab/Andrey/blob/main/docs/DEVELOPMENT.md)
covers the website.

## Ways to help

Every item here is useful, and the first ones take minutes:

- **Tell us where it fails.** Run a method on your data, and open an issue when a result looks
  wrong, a call is confusing, or an error message does not say what to do.
- **Improve a page.** Fix a typo, rewrite a sentence you had to read twice, or add the example a
  guide is missing. The guides are Markdown files under `docs/docs/`.
- **Add an example.** A notebook under `examples/` that applies a method to a public dataset and
  says what the result means.
- **Add a reference to a comparison.** Our benchmarks compare each method with its authors' code
  or a standard package. If you know another implementation, tell us, or add its arm to the
  comparison ([`benchmarks/README.md`](https://github.com/Abel-ai-lab/Andrey/blob/main/benchmarks/README.md)).
- **Pick up an open issue.** The issues labeled
  [`good first issue`](https://github.com/Abel-ai-lab/Andrey/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
  are small and say what done looks like;
  [`help wanted`](https://github.com/Abel-ai-lab/Andrey/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)
  marks larger ones.
- **Propose a method.** See [Propose a method](#propose-a-method) below.

For anything larger than a small fix, open an issue first and say you would like to work on it. We
agree on the scope there, and your pull request then closes that issue. This saves you from
writing code that does not fit.

## Report a problem

Open an issue with the bug or feature form in the
[issue tracker](https://github.com/Abel-ai-lab/Andrey/issues/new/choose). For a wrong result,
include:

- the call you made;
- the shape of the data (rows, columns, and the column types);
- what you expected and what you got;
- the output of `andrey config`, which lists your versions and settings.

## Propose a method

Andrey implements methods that are validated and peer-reviewed; it does not develop new ones. A new
method is developed in its own repository, with its evidence and a paper, and joins Andrey after
that. To propose one, open an issue with the
[method form](https://github.com/Abel-ai-lab/Andrey/issues/new?template=method.yml) and give its
spec:

- the peer-reviewed paper that defines the method;
- the existing implementations, the authors' code first, with their licenses;
- the setting the paper claims: the data, the assumptions, and the sizes;
- the evidence the paper reports, and the task Andrey cannot do today without the method.

A method joins Andrey as experimental and becomes supported once it is compared with its authors'
implementation in that setting
([`DEVELOPMENT.md`](https://github.com/Abel-ai-lab/Andrey/blob/main/DEVELOPMENT.md#test-a-method)).

## Make a change, step by step

### 1. Fork and clone

Fork [Abel-ai-lab/Andrey](https://github.com/Abel-ai-lab/Andrey) to your account with the **Fork**
button, then clone your fork and add the original repository as `upstream`:

```shell
git clone https://github.com/<your-username>/Andrey.git
cd Andrey
git remote add upstream https://github.com/Abel-ai-lab/Andrey.git
```

### 2. Install

The project uses [`uv`](https://docs.astral.sh/uv/). One command creates the environment with
Andrey installed in editable mode and the development tools (`ruff`, `ty`, `pytest`):

```shell
uv sync
uv run python -c "import andrey; print(andrey.__version__)"
```

### 3. Create a branch

Work for the next release is collected on a branch named `release/<x.y.z>`, not on `main`. Start
your branch from it:

```shell
git fetch upstream
git branch -r --list "upstream/release/*"      # the release branch, for example upstream/release/0.18.0
git switch -c fix-readme-typo upstream/release/<x.y.z>
```

If two release branches are listed, use the one with the higher version, or ask in the issue.

### 4. Make the change

- Keep one change per pull request, with the tests that show it works.
- When the public API changes, update its docstrings and the API reference (`docs/docs/api.md`) in
  the same pull request.
- Everything in `andrey.__all__` is public and documented. Names that start with `_` are internal
  and may change without notice; do not import them from outside the package.
- Code, data, and assets copied from another project keep their license notice, as
  `site/brand/fonts/OFL.txt` does for the site's fonts.

[`DEVELOPMENT.md`](https://github.com/Abel-ai-lab/Andrey/blob/main/DEVELOPMENT.md#common-changes)
has a recipe for each common change.

### 5. Run the checks

These are the checks a pull request must pass, so running them first saves a round trip:

```shell
uv sync --all-extras                      # the optional methods' tests need the extras
uv run --no-sync ruff format --check .    # formatting (`ruff format .` fixes it)
uv run --no-sync ruff check .             # lint
uv run --no-sync ty check                 # types
uv run --no-sync pytest tests             # every test
```

For a change to one page or one function, the tests next to it are enough to start with, for
example `uv run pytest tests/unit -k <name>`; everything runs again before the merge.

`prek install` (or `pre-commit install`) adds the hooks in `.pre-commit-config.yaml` to your
checkout, so formatting, lint, and a scan for secrets run on every commit.

### 6. Open the pull request

Push your branch to your fork and open a pull request against the release branch you started from:

```shell
git push -u origin fix-readme-typo
```

On GitHub, choose **Compare & pull request**, set the base to `release/<x.y.z>`, and fill in the
two headings of the form:

- **What and why**: what the change does and the issue it closes, for example `Closes #123`.
- **How it was checked**: the tests you ran, the docs you built, or the results you compared.

A fix or a feature also adds one bullet under `## [Unreleased]` in `CHANGELOG.md`. If you are
unsure how to write it, leave it out and say so; a maintainer adds it. A maintainer also sets the
label and the milestone.

### 7. Review and merge

A maintainer reviews the pull request and runs the full checks on it. When it is approved, we
squash-merge it into the release branch, and it ships with that release. Your first merged pull
request is credited under New contributors in the release notes.

## The GPU build of PyTorch

You need this only to work on the GPU paths. The published wheel is torch-free, and
`uv sync --extra torch` installs the CPU build of torch: `uv sync` resolves one lock for every
machine, so it cannot pick a build for the local GPU. For a GPU build, replace torch after syncing.
`--torch-backend=auto` reads the installed driver and picks the matching wheel:

```shell
uv sync --extra torch
uv pip install --reinstall --torch-backend=auto torch
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The GPU build lives only in the active `.venv`; a later `uv sync` restores the locked CPU build, so
run with `uv run --no-sync` until then.
