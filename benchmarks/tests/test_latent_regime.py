"""Check hidden-column removal and PAG scoring for every estimate in the latent regime.

Runs ``materialize -> subprocess fit -> to_structure -> score`` on one tiny dataset. The dataset has
one hidden node. The empty graph returns a CPDAG, which also requires projection of the truth to a
PAG.
"""

from __future__ import annotations

from andrey_bench import datasets
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.contracts import Status
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch


def test_fci_and_the_empty_graph_score_against_the_observed_pag(tmp_path):
    key = DatasetKey(
        topology="er",
        functional="linear",
        noise="gaussian",
        standardize="standardized",
        density=datasets.CANONICAL_DENSITY,
        d=10,
        n=100,
        seed=0,
        latents=datasets.latents_for("latent_gauss_er", 10),
    )
    result = run_batch(
        tmp_path, [AndreyFCI(), BaselineEmpty()], [key], machine_id="tiny-cpu", cap_wall_s=240.0
    )
    assert not result.errors, result.errors
    by_name = {r.name: r for r in result.records}
    for r in result.records:
        assert r.status == Status.OK.value, f"{r.name}: {r.error_text}"
        assert r.latents == 1
        assert r.metrics["family"] == "pag"
        assert r.varsortability is not None
    # The empty graph misses every edge of the observed PAG; FCI finds some of them.
    assert by_name["andrey.fci.numpy"].metrics["shd"] < by_name["baseline.empty"].metrics["shd"]
