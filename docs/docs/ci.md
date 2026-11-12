---
orphan: true
name: ci
description: What each Andrey release is tested on - Linux, macOS, Windows, and GPUs - and how to
  run the GPU checks yourself.
meta:
  type: spec
---

# Testing

Every pull request, from a fork too, runs the full test suite on Linux with every extra, on Python
3.11 and 3.14, and the recovery tests without the extras. One that changes the code also builds and
installs the wheel, and one that changes the code, the documentation, or the notebooks builds this
website and reruns every notebook, which must reproduce the outputs the website shows. NumPy in
float64 is the reference: the PyTorch paths must match its results and repeat exactly.

Each release's pull request into `main` runs a quality gate that fits every supported method
({{ n_supported }}) and the supported nonlinear tests to data with known graphs, and many of them
to the Sachs data too, and the release merges only when every method passes. A push that changes
only the documentation, the website's files, or a top-level Markdown file such as the CHANGELOG
reuses the last passing run instead of fitting the methods again. The publish reuses a run when the
tagged commit has the same files as the commit the gate checked, and runs the gate again when it
does not.

The quality gate also reads the published benchmarks and flags any row where the reference package
makes significantly fewer errors: the paired 95% interval of the difference in errors, over at
least 10 datasets, lies above 0. A row is exempt when a cited source shows that Andrey follows the
method's paper or that the reference has a defect. Unlike a method that does not pass, a flag does
not fail the gate; the release checklist reviews every flag.

| Platform | What is tested |
|---|---|
| Linux x86_64 | everything above |
| macOS arm64 | on demand: install, import, the command line, PC and GES fits, worker processes, and the unit tests with PyTorch on the CPU |
| Windows x86_64 | on demand: install, import, the command line, PC and GES fits, and worker processes, on the CPU |

## GPUs

No hosted runner has a GPU, so the CUDA paths are checked on our own machines: a parity suite
compares the GPU with NumPy, in float64 to about `1e-9`. To run it on a machine with a CUDA GPU and
a [GPU build of PyTorch](guides/getting-started.md#pytorch-for-a-gpu):

```shell
ANDREY_DEVICE=cuda ANDREY_REQUIRE_GPU=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync pytest tests/gpu -ra
```

`ANDREY_REQUIRE_GPU=1` makes the suite fail, instead of skip, when no GPU is present. Apple GPUs are
not supported in the alpha; with `ANDREY_DEVICE=mps` the suite runs on one, in float32 to `1e-3`.
