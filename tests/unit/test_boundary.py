"""A fit larger than its method's measured boundary warns once per method, and no other fit does."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import andrey
import scripts.boundary_table as generator
from andrey.api import _boundary, _boundary_table
from andrey.api._boundary import CAP_STOPS, Ladder, measured, past_boundary, warn_past_boundary
from andrey.core import backend
from andrey.spec import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ("anm", "pnl")


def _capped() -> list[str]:
    """Methods with a ladder that ended on a cap: the ones whose facades can warn."""
    return sorted(
        m for m, ladders in measured().items() if any(lad.stop in CAP_STOPS for lad in ladders)
    )


def _shape(ladder: Ladder, size: int) -> tuple[int, int]:
    """Data of ``size`` at ``ladder``'s setting: its columns, or its rows on a rows ladder."""
    if ladder.unit == "rows":
        return size, 2
    return (ladder.rows * size if ladder.per_variable else ladder.rows), size


def _settings(method: str) -> dict[tuple, list[Ladder]]:
    """A method's ladders by setting: the same rows, or rows per variable."""
    settings: dict[tuple, list[Ladder]] = {}
    for ladder in measured()[method]:
        settings.setdefault((ladder.rows, ladder.per_variable), []).append(ladder)
    return settings


@pytest.mark.parametrize("method", _capped())
def test_warns_only_above_the_most_any_density_finished_at_a_capped_setting(method):
    for ladders in _settings(method).values():
        limit = max(lad.last_through for lad in ladders if lad.last_through is not None)
        reached = [lad for lad in ladders if lad.last_through == limit]
        if any(lad.stop in CAP_STOPS for lad in reached):
            assert past_boundary(method, _shape(reached[0], limit)) is None
            ladder, _ = past_boundary(method, _shape(reached[0], limit + 1))
            assert (ladder.last_through, ladder.stop in CAP_STOPS) == (limit, True)
        else:  # past an accuracy, error, or last-rung stop the method was not measured
            assert past_boundary(method, _shape(reached[0], 10 * limit + 1)) is None


@pytest.mark.parametrize("method", PAIRS)
def test_the_pair_methods_measure_rows(method):
    assert {ladder.unit for ladder in measured()[method]} == {"rows"}
    limit = max(ladder.last_through for ladder in measured()[method])
    assert past_boundary(method, (limit + 1, 2)) is not None
    assert past_boundary(method, (limit, 2)) is None


def test_methods_with_no_capped_ladder_never_warn():
    others = set(andrey.api.__all__) - set(_capped())
    assert "granger" in others  # no ladder
    for method in others:
        for shape in ((10**7, 10**6), (10**7, 2)):
            assert past_boundary(method, shape) is None
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                warn_past_boundary(method, shape)


def _fake(monkeypatch, *ladders: Ladder, methods: tuple[str, ...] = ("method",)) -> None:
    monkeypatch.setattr(_boundary, "measured", lambda: dict.fromkeys(methods, ladders))


def test_the_nearest_setting_is_nearest_in_rows_on_a_log_scale(monkeypatch):
    _fake(
        monkeypatch,
        Ladder("500 rows", 500, False, 2, "variables", 20, 20, "time"),
        Ladder("10,000 rows", 10000, False, 2, "variables", 100, 100, "time"),
    )
    assert past_boundary("method", (2000, 21)) is not None  # 4 times 500, 5 times under 10,000
    assert past_boundary("method", (2500, 21)) is None
    assert past_boundary("method", (2500, 101)) is not None


def test_a_density_that_finished_more_without_a_cap_stop_lifts_the_limit(monkeypatch):
    _fake(
        monkeypatch,
        Ladder("mean degree 8", 2000, False, 8, "variables", 20, 20, "time"),
        Ladder("mean degree 2", 2000, False, 2, "variables", 40, 40, "accuracy"),
    )
    assert past_boundary("method", (2000, 30)) is None
    assert past_boundary("method", (2000, 41)) is None


def test_warns_once_per_method_whatever_the_size(monkeypatch):
    methods = ("first", "second")
    ladder = Ladder("2,000 rows", 2000, False, 2, "variables", 20, 20, "time")
    _fake(monkeypatch, ladder, methods=methods)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for method in methods:
            for size in (21, 40):
                warn_past_boundary(method, (2000, size))
    assert [w.category for w in caught] == [andrey.BoundaryWarning] * len(methods)
    assert all(f"andrey.{m} " in str(w.message) for m, w in zip(methods, caught, strict=True))


def test_the_warning_points_to_a_release_card():
    assert (ROOT / "docs" / "docs" / "benchmarks" / f"{_boundary_table.RELEASE}.md").is_file()


def test_the_table_is_generated_from_the_newest_boundary_lane():
    expected = generator.table(json.loads(generator.newest().read_text()))
    assert {name: getattr(_boundary_table, name) for name in expected} == expected


class _Checked(Exception):
    """Raised by the check, so the fit does not run."""


def _call(method: str, data: np.ndarray) -> object:
    """Fit ``method`` on ``data``: a domain index for cdnod, two or three copies for a panel."""
    if method == "cdnod":
        return andrey.cdnod(data, domain_index=np.arange(len(data)) % 2)
    if method in ("multi_group_direct_lingam", "longitudinal_lingam"):
        return getattr(andrey, method)([data, data + 1, data * 2])
    return getattr(andrey, method)(data)


@pytest.mark.parametrize("method", sorted(REGISTRY))
def test_every_method_checks_its_data_against_the_boundary(method, monkeypatch):
    if method == "calm" and backend.torch() is None:
        pytest.skip("calm needs torch")
    checked = []

    def check(name, shape):
        checked.append((name, tuple(shape)))
        raise _Checked

    monkeypatch.setattr(_boundary, "past_boundary", check)
    data = np.random.default_rng(0).standard_normal((60, 2 if method in PAIRS else 4))
    with pytest.raises(_Checked):
        _call(method, data)
    assert checked == [(method, data.shape)]
    data[:, 1] = 1.0
    with pytest.raises(ValueError, match="constant"):
        _call(method, data)
    assert len(checked) == 1
