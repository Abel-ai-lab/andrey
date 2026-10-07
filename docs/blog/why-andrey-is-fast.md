---
name: why-andrey-is-fast
description: "The four ideas behind Andrey's speed, each shown as a short animation."
meta:
  type: post
blogpost: true
date: 2026-10-08
author: Shu Wan
version: 0.1.0
cover: graph
---

# Why Andrey is fast

Most of the time in causal discovery goes to two jobs: testing whether variables are independent
(PC and FCI) and scoring candidate graphs (GES and BOSS). We speed up both with four ideas,
easiest first. Each animation runs the same work in two lanes: Andrey's, and one labeled Others
that does it one step at a time. They are illustrations of the ideas, not measurements; the
[benchmarks](../docs/benchmarks.md) have the measurements.

## 1. Do many things at once

**Batch and parallelize.** Within one round of PC, each independence test stands alone, and within
one pass of GES, so does each candidate move. Andrey runs them as one batched array operation,
thousands at a time, with the same results as one at a time.

```{raw} html
:file: ../_generated/why-fast/parallel.html
```

## 2. Never compute twice

**Cache and update incrementally.** Andrey computes the correlation matrix once, up front, and
caches it. Each test reads the few entries it needs instead of recomputing them, and GES memoizes
its local scores the same way.

```{raw} html
:file: ../_generated/why-fast/reuse.html
```

## 3. Skip what can't matter

**Prune the search early.** Tests given zero or one other variable have a closed form, so a quick
pass settles most edges before any matrix is inverted. GES rules out invalid candidates with bit
masks before scoring them.

```{raw} html
:file: ../_generated/why-fast/skip.html
```

## 4. Use the hardware you have

**Hardware-aware acceleration.** With the optional extras, Numba JIT-compiles GES's path checks, and
a CUDA GPU computes large entropy tables many cells at a time, once they are large enough to repay
the transfer. Correlation matrices move to the GPU too when you set `ANDREY_BACKEND=cuda`.

```{raw} html
:file: ../_generated/why-fast/hardware.html
```
