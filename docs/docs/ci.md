---
orphan: true
name: ci
description: What each Andrey release is tested on - Linux, macOS, Windows, and GPUs - and how to
  run the GPU checks yourself.
meta:
  type: spec
---

# Testing

Every pull request runs the full test suite on Linux, on Python 3.11 and 3.14, with and without
the optional extras. It also builds and installs the wheel, builds this website, and reruns every
notebook, which must reproduce the outputs the website shows. NumPy in float64 is the reference:
the Numba and PyTorch paths must match its results and repeat exactly.

Before each release, a quality gate fits the nine supported methods to data with known graphs,
seven of them to the Sachs data too, and the release is published only when every method passes.
Every pull request also checks the published benchmarks: a method fails when its reference package
has fewer errors at any size, unless a cited source shows that Andrey follows the method's paper or
that the reference has a defect.

| Platform | What is tested |
|---|---|
| Linux x86_64 | everything above, on every pull request |
| macOS arm64 | install, import, the command line, PC and GES fits, worker processes, and the unit tests with PyTorch on the CPU |
| Windows x86_64 | install, import, the command line, PC and GES fits, and worker processes, on the CPU |

## GPUs

No hosted runner has a GPU, so the CUDA and MPS paths are checked on our own machines: a parity
suite compares each device with NumPy, CUDA in float64 to about `1e-9` and MPS in float32 to
`1e-3`. To run it on a machine with a GPU and a
[GPU build of PyTorch](guides/getting-started.md#pytorch-for-a-gpu):

```shell
ANDREY_DEVICE=cuda ANDREY_REQUIRE_GPU=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/gpu -ra
```

`ANDREY_REQUIRE_GPU=1` makes the suite fail, instead of skip, when no GPU is present. Use
`ANDREY_DEVICE=mps` on an Apple GPU.
