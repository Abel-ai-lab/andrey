---
name: introducing-andrey
description: "Introducing Andrey, a very fast causal discovery package for Python."
meta:
  type: post
blogpost: true
date: 2026-10-08
author: Shu Wan
version: 0.1.0
cover: pair
cover_caption: "Andrey Kolmogorov and Andrey Markov."
---

# Introducing Andrey

We're happy to introduce Andrey, a very fast causal discovery package for Python.

Andrey is named after Andrey Markov and Andrey Kolmogorov, whose ideas causal discovery still runs
on ([where the name comes from](../docs/faq.md#where-does-the-name-andrey-come-from)).

Most data analysis tells you which variables move together. Causal discovery learns from data which
ones drive which, and draws them as a graph of causes. As AI systems take on more analysis and make
decisions of their own, they need to know what causes what.

For a long time, scale has held causal discovery back. The methods are well understood, but popular
Python packages run them slowly. {{ slow_fit }}. At that speed, trying a new idea takes time, and
data with thousands of variables is out of reach.

We built Andrey to make causal discovery fast and easy to use. It runs the same methods faster
({{ speed_short }}).

- **Fast**: {{ speed_claim }}
- **Just as accurate**: the same errors or fewer in {{ as_good }} comparisons with other packages.
- **One API**: {{ method_count }}, each one function call.
- **Agent-ready**: a command line that answers in JSON, and a skill for coding agents.

```{raw} html
:file: ../_generated/blog/speedup.html
```

[Why Andrey is fast](why-andrey-is-fast.md) explains the four ideas behind the speed. Andrey
is free and open source, and so are its [benchmarks](../docs/benchmarks.md), which you can rerun
yourself.

## What's next

We plan to add more methods, including more for time series. We also plan to add local causal
discovery, which finds the structure around one variable, such as its direct causes and effects or
its Markov blanket, without learning the whole graph. And we want to use sparsity to scale causal
discovery to millions of variables.

## Try Andrey

```{code-block} shell
:class: window

uv add andrey-core
```

Then run PC on a real dataset:

```{literalinclude} ../../site/example.py
:language: python
:caption: example.py
```

```{literalinclude} ../_generated/blog/example-output.txt
:language: text
:caption: example.py
```

## Help us build it

Andrey {{ release }} is in alpha, the API may change between releases.

When you try it on your own data,
[tell us what breaks](https://github.com/Abel-ai-lab/andrey/issues/new) or which method you need.
We'd welcome a pull request too. The
[contributing guide](https://github.com/Abel-ai-lab/andrey/blob/main/CONTRIBUTING.md) explains how
to set up.

---

- [Getting started](../docs/guides/getting-started.md)
- [Examples](../docs/examples/index.md)
- [Demos](../docs/demos/index.md)
- [Benchmarks](../docs/benchmarks.md)
- [FAQ](../docs/faq.md)
- [GitHub](https://github.com/Abel-ai-lab/andrey)
