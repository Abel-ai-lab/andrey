"""The MMB-by-MMB lane's judge: per-dataset measures, paired verdicts, replication, and the rows the
release summary and the parity gate read."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmb_by_mmb_summary as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _scores(correct: int, est: int, true: int, shd: int) -> dict:
    return {
        "est_edges": est,
        "true_edges": true,
        "correct_edges": correct,
        "shd": shd,
        "empty_shd": true,
    }


def _record(solution, seed, target, correct, est, true=4, status="ok", n=1000, tests=10):
    shd = (est - correct) + (true - correct)
    return {
        "network": "alarm",
        "n": n,
        "seed": seed,
        "d_observed": 33,
        "solution": solution,
        "target": target,
        "repeat": 0,
        "status": status,
        "wall_s": 0.5,
        "n_tests": tests,
        "scores": {} if status != "ok" else {str(target): _scores(correct, est, true, shd)},
    }


def _write(tmp_path: Path, records: list[dict]) -> str:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(run)


def test_an_empty_answer_counts_precision_one_distance_one_and_f1_zero():
    m = S.target_measures(_scores(0, 0, 3, 3))
    assert m["distance"] == 1.0 and m["f1"] == 0.0 and m["errors"] == m["empty_errors"] == 3


def test_a_dataset_counts_only_when_every_fit_finished_and_skips_targets_with_no_true_edge():
    ok = [_record("a", 0, 1, 2, 2), _record("a", 0, 2, 0, 0, true=0)]
    m = S.dataset(ok)
    assert m["targets"] == ["1"] and m["f1"] == pytest.approx(2 * 2 / 6)
    assert S.dataset([*ok, _record("a", 0, 3, 0, 0, status="timeout")]) is None


def _cell(tmp_path, andrey_correct, ref_correct, seeds=range(20)):
    records = []
    for seed in seeds:
        for t in (1, 2):
            records.append(_record(S.ANDREY, seed, t, andrey_correct(seed), 4, tests=100))
            records.append(_record(S.REFERENCE, seed, t, ref_correct(seed), 4))
    return S.load([_write(tmp_path, records)])


def test_a_cell_worse_on_the_first_seeds_counts_only_when_the_fresh_seeds_agree(tmp_path):
    runs = _cell(tmp_path, lambda s: 2 if s < 20 else 3 + s % 2, lambda s: 3, range(40))
    [row] = S.summarize(runs, range(20), range(20, 40), None)
    assert row["judged"] and row["first"]["f1"]["verdict"] == "worse"
    assert row["fresh"]["f1"]["verdict"] != "worse" and row["result"] == "tie"
    runs = _cell(tmp_path, lambda s: 2, lambda s: 3, range(40))
    [row] = S.summarize(runs, range(20), range(20, 40), None)
    assert row["result"] == "worse"


def test_the_summary_rows_carry_errors_the_parity_gate_judges(tmp_path):
    from qa import reference_parity

    runs = _cell(tmp_path, lambda s: 3, lambda s: 4, range(40))
    rows = S.summarize(runs, range(20), range(20, 40), None)
    for measure, metric in S.SUMMARY_METRICS.items():
        [summary] = S.summary_rows(rows, measure, "20-39")
        assert summary[f"andrey_{metric}_mean"] > summary[f"comparator_{metric}_mean"]
        assert summary["above_zero"] and summary["replication"]["above_zero"]
        assert summary["floor"]["applies"] and summary["floor"]["pass"]
        entry = {"method": "MMB-by-MMB", "metric": metric, "comparator": "authors", "rows": [summary]}
        data = {"parity_rule": "paired bootstrap", "methods": [entry]}
        [parsed] = reference_parity.paired_rows("test", data)
        assert parsed.failed


def test_the_gate_knows_both_edge_metrics():
    from qa import reference_parity

    assert set(S.SUMMARY_METRICS.values()) <= set(reference_parity.METRICS)
    assert (ROOT / "qa" / "reference_parity.py").exists()
