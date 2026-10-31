# FCI collider configurations

`fci_latent.py --include-majority` measures both Andrey collider rules alongside causal-learn FCI
and the empty-graph baseline. The default rule keeps the name `andrey.fci.numpy`; the optional rule
uses `andrey.fci.majority.numpy`. Each record includes `collider_rule` in its parameters. The same
datasets and resource limits apply to both configurations, with wall-clock timing and recovery
metrics recorded separately. Agreement with reference implementations is reported as evidence;
exact agreement is not a correctness requirement.

```shell
$ANDREY_BENCH_PYTHON benchmarks/fci_latent.py --out outputs/fci-colliders \
    --ladder 6 8 --seeds 3 --repeats 3 --include-majority
```

PAG recovery scores use the projection of the known generating DAG. That projection shares
orientation helpers with Andrey FCI, so it does not independently validate those helpers. The column-order
count is documented in [`qa/README.md`](../../qa/README.md#fci-column-order).
