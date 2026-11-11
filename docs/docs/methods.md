---
name: methods
description: Every Andrey method by family, with the data it reads, the graph it returns, and its
  status. Start from your data to find the method to begin with.
meta:
  type: guide
---

# Methods

Andrey has {{ n_supported }} supported methods and {{ experimental_count }}. Start from the kind of
data you have to find the method to begin with, or look a method up in its family below. Each name
opens its entry in the [API reference](api.md).

(your-data)=

## Start from your data

```{raw} html
:file: ../_generated/methods-start.html
```

How we pick the method to start with: in the first family that reads your data, among the
supported methods whose graph beats the empty graph at a size every compared package ran, the one
compared with the most packages, then the one with the largest such size, then the first in the
registry. A method that needs your kind of data, such as one that requires each row's environment,
comes before one that only accepts it. A method reads missing values when it deletes rows test by
test, or fills them by structural EM where EM is supported for it.

(by-family)=

## Every method by family

```{include} ../_generated/methods-families.md
```
