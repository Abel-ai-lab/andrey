# GES parent limit

`andrey.ges(max_parents=k)` caps the parents a node has in any DAG of the returned CPDAG's class.
The limit of the reference packages is a gate: causal-learn's `maxP` stops a node that already
holds more than `maxP` directed parents from gaining new ones, and one move can still leave a node
above it. The Andrey GES adapter therefore reads `max_parents` through a private keyword of
`andrey.search.ges.ges`, `_parent_rule`, whose `"gate"` applies the limit as that gate and whose
default, `"cap"`, is the public behavior. A lane sets `parent_rule="gate"` on its Andrey arms only
when it limits the reference's parents too, so both packages apply one rule: `ges_cpu.py`,
`xges_ges.py`, `public_datasets.py`, and the BDeu pair of `discrete_cpu.py` run causal-learn with
the same limit. A lane that limits only Andrey, such as the R lane (pcalg has no parent limit),
keeps the cap, and an arm with `max_parents=None` has no limit under either rule. `params` records
the rule next to the limit.
