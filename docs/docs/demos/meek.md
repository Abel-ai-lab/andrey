---
name: meek
description: Meek's orientation rules, replayed one step at a time from the orientation engine.
meta:
  type: guide
html_theme.sidebar_secondary.remove: true
---

# Meek's rules

After the skeleton and the colliders are found, Meek's rules orient more edges without creating a
new collider or a directed cycle. Each step below is a rule the orientation engine fired on a small
input; a ring marks the arrowhead the step adds.

```{raw} html
<iframe class="example-frame" src="../../_static/examples/meek.html" title="Meek's rules, one step at a time"
  sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" loading="lazy"></iframe>
```
