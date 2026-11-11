---
name: methods
description: Every Andrey method by family, with the data it reads, the graph it returns, and its
  status. Start from your data to find the method to begin with.
meta:
  type: guide
---

# Methods

Choose what your data look like to see the methods that read it, the one to start with, and the
code that runs it. Below, every method sits with its family: the kinds of data it reads, the graph
it returns, whether it is supported or experimental, and the release cards that compare it with
other packages. Each name opens its entry in the [API reference](api.md).

(your-data)=

## What does your data look like?

```{raw} html
:file: ../_generated/methods-picker.html
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
