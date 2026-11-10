---
name: compare
description: Compare Andrey's and causal-learn's graphs with the true graph on one dataset.
meta:
  type: guide
html_theme.sidebar_secondary.remove: true
---

# Compare on one dataset

Andrey and causal-learn each recover a graph from the same observations, drawn from a known true
graph. Every edge is colored against the truth: correct, missing, extra, or pointing the wrong way.
Under each graph are its structural Hamming distance (SHD) and the F1 scores of its skeleton and its
arrowheads. Choose PC or GES above the graphs.

```{raw} html
<iframe class="example-frame" src="../../_static/examples/compare.html" title="Compare on one dataset"
  sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" loading="lazy"></iframe>
```
