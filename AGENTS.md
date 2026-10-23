# Agent Instructions

This file is the operating manual for the **Andrey** repository. [`DEVELOPMENT.md`](DEVELOPMENT.md)
maps the code and the common changes, [`DESIGN.md`](DESIGN.md) sets the rules for any visual, and
[`docs/docs/ci.md`](docs/docs/ci.md) describes CI. When this file and any skill overlap, this file
wins.

## Project context

Andrey is an independent causal-discovery package built for CPU-parallel and GPU execution.
Tests compare against fixed cases, ground truth, and Andrey's own recorded outputs. Comparisons
with other packages and older Andrey builds belong in `benchmarks/`.

## Coding style

Follow KISS, YAGNI, Locality-of-Behavior, Fail-fast, in that order. "Make it work, make it
right, make it fast." For algorithm implementations:

- Every algorithm slice passes fixed-case, ground-truth, and recorded-output recovery checks.
  Performance claims use benchmark campaigns with recorded build and machine details.
- **No `print()` in the package** - use `logging`.
- Self-documenting names (`pc`, `StructureOutput`, `Structure`, `adjacency`) that match the
  surrounding code.
- Use `|` instead of `Optional`/`Union` in type annotations; type hints and docstrings on the
  public surface.
- Visuals take their colors from `andrey.viz` and follow [`DESIGN.md`](DESIGN.md); never hardcode
  a hex value.

## Documentation

- Prefer editing an existing page over adding one. Every page under `docs/docs/` sets a
  `description` in its front matter (its search snippet and link preview); a page added, renamed,
  or removed updates its folder's `index.md` in the same change.
- When the public API changes, update the docstrings, `docs/docs/api.md`, and the affected
  guides in the same change.
- A blog post (`docs/blog/<yyyymmdd>-<name>.md`, named for its date) also sets ABlog's
  `blogpost`, `date`, and `author`, a `description`, which the blog index shows, and `tags`: its
  kind (`Release` or `Engineering`) first, then one or more topics (a method family as the README's
  method table names it, `Speed`, or the post's subject). Its `cover` names an art in
  `site/brand/art-<cover>.jpg`, its subject on the right; the link preview's banner adds the title
  and the release. A centered `art-<cover>-center.jpg`, when there is one, is the post's cover and
  thumbnail instead. A published post that is renamed keeps its old address with `redirect`.

## Documentation style

Concise and simple; describe current behavior only - no "this PR adds ..." commentary or issue
numbers in prose. The CHANGELOG is the exception: each entry ends with a link to its pull request.
Everything, code names included, is written in US English: color, behavior, neighbor, center.
Code, docstrings, comments, and the developer docs (`AGENTS.md`,
`DEVELOPMENT.md`, `CONTRIBUTING.md`) are written in the third person. Pages written for readers
(the README, the blog, the homepage, and the pages under `docs/docs/`) keep their voice: they
address the reader as "you" and speak for the project as "we". In code and tests, describe what
the code does, not how it came to be: no development-process narration, no plan step or decision
labels, and no reviewer or session attribution. Cite method papers in `References` sections.
Wrap Markdown at 100 characters. Use `->` not arrow unicode. Backticks for commands, code, and file
paths. A package name is code only inside code: `import andrey`, the `andrey` command,
`pip install andrey-core`, `numpy>=2`. Everywhere else every package, Andrey included, is a plain
name: Andrey, NumPy, lingam, pcalg. Respect `.markdownlint.yaml`.

## Workflow

One issue -> one PR, each checked by CI. A PR targets its release's branch, `release/<x.y.z>`; the
release PR, from that branch to `main`, is the only PR that merges to `main`, and the tag follows
it. `uv` manages the environment, `ruff` lints and formats, and `prek` runs the commit hooks
(`DEVELOPMENT.md` has the commands).
