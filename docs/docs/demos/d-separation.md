---
name: d-separation
description: Pick two variables and a conditioning set, and see which paths are open.
meta:
  type: guide
html_theme.sidebar_secondary.remove: true
---

# d-separation explorer

Two variables are d-separated given a set when every path between them is blocked. Choose two
variables and what to condition on: conditioning blocks a chain or a fork, but opens a path through
a collider when the collider or one of its descendants is in the set. Each answer comes from
Andrey's d-separation oracle.

```{raw} html
<iframe class="example-frame" src="../../_static/examples/d-separation.html" title="d-separation explorer"
  sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" loading="lazy"></iframe>
```
