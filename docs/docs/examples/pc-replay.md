---
name: pc-replay
description: PC's steps replayed beside its pseudocode, from the trace Andrey's PC records as it runs.
meta:
  type: guide
html_theme.sidebar_secondary.remove: true
---

# PC, step by step

PC removes edges by conditional-independence tests, orients colliders, then applies Meek's rules.
Each step below was recorded by Andrey's PC as it ran: the pseudocode marks the current line, and
the graph and its endpoint-mark matrix show the state. The 4-variable example, rain and a sprinkler
wetting the grass, plays one test per step; the 40-variable graph plays one depth per step.

```{raw} html
<iframe class="example-frame" src="../../_static/examples/pc.html" title="PC step by step"
  sandbox="allow-scripts" loading="lazy"></iframe>
```
