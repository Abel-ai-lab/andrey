"""The Y-structures lane judged per cell: paired verdicts against the authors' code, the replication
a worse cell needs, the floors, and the rows of the release summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import y_structures_summary as ys  # noqa: E402

SEEDS = range(40)
POSITIVES, PAIRS = 6, 56


def _record(solution: str, seed: int, tp: int, fp: int, selection: str = "biased") -> dict:
    claims = [[i, i + 1, 1.0] for i in range(tp + fp)]
    return {
        "setting": "large",
        "n": 10000,
        "seed": seed,
        "selection": selection,
        "d": 8,
        "acceptance": 0.1,
        "solution": solution,
        "status": "ok",
        "wall_s": 0.5,
        "n_tests": 12 if solution.startswith("andrey") else None,
        "claims": claims,
        "scores": {
            "claims": tp + fp,
            "tp": tp,
            "fp": fp,
            "positives": POSITIVES,
            "pairs": PAIRS,
            "average_precision": 0.4 if tp else 0.1,
        },
    }


def _run(tmp_path: Path, extra_fp: dict[int, int] | None = None, tp: int = 3, fp: int = 1) -> str:
    """Both sides claim ``tp`` true and ``fp`` false pairs per dataset; Andrey's false claims are
    ``extra_fp[seed]`` more where given."""
    extra_fp = extra_fp or {}
    lines = []
    for seed in SEEDS:
        lines.append(_record("andrey.yst", seed, tp, fp + extra_fp.get(seed, 0)))
        lines.append(_record("sbcd.yst", seed, tp, fp))
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return f"authors={run}"


def _cell(arg: str) -> dict:
    rows = ys.summarize(ys.load([arg]), range(20), range(20, 40))
    assert [(r["setting"], r["selection"], r["form"]) for r in rows] == [("large", "biased", "yst")]
    return rows[0]


def test_equal_claims_tie_and_the_floors_pass(tmp_path):
    row = _cell(_run(tmp_path))
    assert row["result"] == "tie" and row["first"]["same_claims"] == 20
    assert row["first"]["claim_error"]["mean_D"] == 0 and row["first"]["paired"] == 20
    # Errors minus the empty answer's are fp - tp = -2 per dataset, for both sides.
    assert row["floor"]["andrey_minus_empty"]["mean"] == -2.0
    assert (
        row["floor"]["andrey_minus_empty"]["pass"] and row["floor"]["reference_minus_empty"]["pass"]
    )
    # Pooled precision 3/4 against the base rate 6/56.
    floor = row["precision_floor"]
    assert floor["precision"] == 0.75 and floor["base_rate"] == round(6 / 56, 4) and floor["pass"]


def test_a_cell_worse_on_both_seed_sets_is_worse(tmp_path):
    row = _cell(_run(tmp_path, {seed: 2 for seed in SEEDS}))
    assert row["first"]["claim_error"]["verdict"] == "worse" and row["result"] == "worse"
    assert row["fresh"]["claim_error"]["verdict"] == "worse"


def test_a_cell_worse_on_the_first_seeds_only_is_a_tie(tmp_path):
    row = _cell(_run(tmp_path, {seed: 2 for seed in range(20)}))
    assert row["first"]["claim_error"]["verdict"] == "worse"
    assert row["fresh"]["claim_error"]["verdict"] == "tie" and row["result"] == "tie"


def test_the_floor_fails_when_the_claims_are_mostly_false(tmp_path):
    row = _cell(_run(tmp_path, tp=1, fp=9))
    assert not row["precision_floor"]["pass"]
    assert not row["floor"]["andrey_minus_empty"]["pass"]


def test_summary_rows_carry_what_the_parity_check_reads(tmp_path):
    row = _cell(_run(tmp_path, {seed: 1 for seed in range(20)}))
    [out] = ys.summary_rows([row], "error", "20-39")
    assert out["d"] == 8 and out["n"] == 10000 and out["paired"] == 20
    assert out["andrey_claim_error_mean"] > out["comparator_claim_error_mean"]
    assert out["above_zero"] is True and out["verdict"] == "worse"
    assert out["replication"]["above_zero"] is False and out["replication"]["seeds"] == "20-39"
    assert out["floor"]["applies"] and out["floor"]["pass"]
    assert out["coverage"] == {"datasets": 20, "andrey_short": 0, "reference_short": 0}
    [ap] = ys.summary_rows([row], "ap", "20-39")
    assert "andrey_claim_ap_loss_mean" in ap and ap["verdict"] == "tie" and "replication" not in ap
