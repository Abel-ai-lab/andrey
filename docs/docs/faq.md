---
name: faq
description: Frequently asked questions about Andrey - install, methods, speed, data, the command line, and the project.
meta:
  type: guide
  tags: [faq]
---

# FAQ

## Getting started

### What is Andrey?

A Python package that learns causal graphs from observational data. Its {{ method_count }} include
PC, FCI, GES, BOSS, GRaSP, and the LiNGAM family. Most methods take a NumPy array or a pandas
DataFrame and return a `StructureOutput`; `multi_group_direct_lingam` and `longitudinal_lingam` take
a list of datasets, and the first returns one result per group. Some methods can spread their work
over more CPU cores or a GPU.

### How do I install it?

The package on PyPI is `andrey-core`, and it imports as `andrey`. It needs Python 3.11 or later;
CI tests 3.11 to 3.14.

```shell
uv add andrey-core
```

[Getting started](guides/getting-started.md#install) covers pip, the optional extras, the
development version, and building from source.

### Which operating systems does it support?

| | Linux | macOS | Windows |
|---|:---:|:---:|:---:|
| Installs and runs on the CPU | ✓ | ✓ | ✓ |
| Full test suite | ✓ | ✗ | ✗ |
| GPU speedups (CUDA / MPS) | ✓ | ✗ | ✗ |

Linux is fully supported; macOS and Windows can run Andrey, with no guarantee yet. On macOS and
Windows, CI checks the install, the command line, and small fits on the CPU, and on macOS the unit
tests; the rest of the test suite there is planned.

### What is the quickest way to try it?

Run the [quickstart](guides/getting-started.md#learn-a-graph), try a [demo](demos/index) in your
browser, or open one of the [example notebooks](examples/index) in Colab. Each notebook installs
Andrey in its first cell.

## Methods

### Which methods are there?

{{ n_supported }} are supported: {{ supported_methods }}. {{ n_experimental }} more are experimental,
including time-series, hidden-variable, and multi-dataset methods. The
[API reference](code/index.md#structure-learning-methods) lists them all, and `andrey list` prints
them with their status.

### What is the difference between supported and experimental?

A supported method is benchmarked and checked on known graphs before each release. An experimental
method has recovery tests but no published benchmark. It warns with `andrey.ExperimentalWarning` the
first time it runs in a process.

### Which method should I start with?

For continuous data with no hidden variables, start with `pc` or `ges`; both return a CPDAG. Use
`fci` when hidden variables may confound the data; it returns a PAG. Use `direct_lingam` when the
relationships are linear and the noise is not Gaussian; it returns a DAG with edge weights.

### Are the classic methods accurate on very large graphs?

Not always. In the benchmarks, which use sparse simulated graphs with ten samples per variable, FCI
at 800 variables returns graphs with no fewer errors than the empty graph in both Andrey and
causal-learn, so there the limit is the algorithm's, not the software's. PC's graphs
in gCastle are worse than the empty graph at 200 and 400 variables, while PC in Andrey and
causal-learn stays below it. The benchmarks' [large graphs](benchmarks.md#large-graphs) section
lists every case, including Andrey's GES at 200 variables with the default penalty.

### Why does a result differ from another package's?

Methods can break ties in a different order, use different defaults, or treat the column order
differently; FCI, for one, depends on column order. The [benchmarks](benchmarks.md) score every
package on the same true graphs, so compare accuracy there rather than edge by edge.

## Speed and hardware

### How does it compare with other packages?

The [benchmarks](benchmarks.md) show Andrey side by side with causal-learn, gCastle, and lingam
on the same data and machines: fit time, error counts, and where Andrey is not faster. The
[homepage](https://andrey.abel.ai/#compare) compares their interfaces and methods.

### Why is it faster?

It batches independent tests and scores into array operations, reuses work across steps, skips
work that cannot change the result, and uses compiled or GPU code where that helps.
[Why Andrey is fast](../blog/why-andrey-is-fast.md) explains the four ideas.

### Do I need a GPU?

No. The default install uses NumPy on the CPU. The `torch` extra adds a PyTorch backend, which
CALM needs. With a CUDA GPU, DirectLiNGAM's entropy computations move to the GPU once they are
large enough to repay the transfer; correlation matrices move too when you set
`ANDREY_BACKEND=cuda`. Apple GPUs are not supported in the alpha.

### What do the optional extras add?

- `numba`: compiled CPU code for GES's path checks.
- `torch`: the PyTorch backend for GPUs, and CALM.
- `viz`: Matplotlib chart styling (`andrey.viz.mpl`), and Graphviz layouts for drawn graphs.
- `data`: saving and loading generated datasets (`andrey.data`).
- `all`: all four.

### Can it use more CPU cores?

GES, GIES, GFCI (in its GES phase), and hill climbing can spread each pass over worker processes:
set `ANDREY_NUM_WORKERS` above 1. GES's parallel result is identical to its serial one. A script
that does this must put its entry point under `if __name__ == "__main__":`, because starting the
workers re-imports the script;
[Getting started](guides/getting-started.md#backends-and-parallelism) shows how. A pass uses the
workers only above a work cutoff. GES's default suits one fit per process, and HC's sits between
the values for one fit and for many; for a process that runs many fits, `andrey calibrate` measures
the cutoffs on your machine and prints the settings.

## Data

### What input does it take?

A NumPy array or a pandas DataFrame, with samples in rows and variables in columns. A DataFrame's
column names become the graph's node labels. From the command line, a CSV or TSV file (its header
row names the nodes) or a `.npy` file.

### Does it handle discrete or mixed data?

Not yet; it is part of stage 1 of the [roadmap](roadmap.md). Every method assumes continuous data:
PC and FCI test independence with Fisher's z; GES, BOSS, and GRaSP score with the linear-Gaussian
BIC; and the LiNGAM methods assume continuous non-Gaussian noise.

### Time series and several datasets?

`varma_lingam` and `longitudinal_lingam` learn graphs over time lags. `multi_group_direct_lingam`
fits several datasets that share one causal order, and `cdnod` takes data from several domains.
All four are experimental.

## Command line and agents

### How do I run a method from the shell?

```shell
andrey list                        # every method, with its status
andrey run pc --help               # its inputs and parameters
andrey run pc --data your_data.csv alpha=0.01
```

The [command-line reference](code/index.md#command-line-interface) has every command and flag.

### How does an agent use it?

The `andrey` command never prompts. `andrey run` and `andrey list` print JSON when a program
reads their output, and every command takes `--json`. `andrey run <method> --help --json` gives a
method's parameters as JSON Schema without running it, and `andrey --skill` prints a skill an agent
can install, for example `andrey --skill > ~/.claude/skills/andrey/SKILL.md`.

## The project

### How stable is it?

Andrey is in alpha: any release may change the API, including the supported methods. Pin the
version you depend on, and read the [changelog](changelog.md) before upgrading. The public API is
what the [API reference](code/index.md) documents; names that start with an underscore are private
and can change in any release.

### What is planned next?

The [roadmap](roadmap.md) lists the planned work in order, starting with every major causal
discovery method, benchmarked. The [roadmap issue](https://github.com/Abel-ai-lab/andrey/issues/1)
on GitHub tracks the progress of each stage.

### What is the license?

The Apache License 2.0.

### Who builds it?

[Abel AI Lab](https://abel.ai/).

### Where does the name "Andrey" come from?

From two mathematicians named Andrey whose ideas causal discovery still runs on.

Andrey Markov (1856-1922) studied chains of events in which the next step depends only on the
present one. Causal discovery borrows his idea: once you know a variable's direct causes, nothing
else except its own effects tells you more about it. That assumption, the Markov condition, is where
PC, GES, and FCI start.

Andrey Kolmogorov (1903-1987) set down the axioms of probability we still use, and later measured
how complex an object is by the length of the shortest program that produces it. Causal discovery
uses that idea to tell cause from effect: a cause and the mechanism that turns it into its effect
should carry no information about each other.

We named the package for both, and built it to make their ideas fast enough to use every day.

### How do I report a bug or ask for a method?

Open an issue on [GitHub](https://github.com/Abel-ai-lab/andrey/issues/new).

### How do I cite it?

Cite the software with this BibTeX entry; the repository's
[`CITATION.cff`](https://github.com/Abel-ai-lab/andrey/blob/main/CITATION.cff) has the same
citation.

```bibtex
@software{andrey,
  author = {{Abel AI Lab}},
  title = {Andrey: A very fast causal discovery package},
  year = {2026},
  version = {0.1.0},
  url = {https://andrey.abel.ai},
}
```
