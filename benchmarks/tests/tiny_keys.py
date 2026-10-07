"""The datasets the harness is validated on end-to-end, before any benchmark fires.

Four keys: the canonical slice (ER · linear · gaussian · standardized, degree 2) at ``d ∈ {6, 10}``
× ``seed ∈ {0, 1}``, all at ``n = 200``. Small enough that the whole suite runs in seconds, which is
the only way an acceptance gate gets used.

A literal list, not a product of axes: the tasks a test asserts against have to be readable in the
test, and a key's ``digest`` is the ``dataset_id`` every record carries, so writing them out is what makes
a drift in identity visible as a diff here.
"""

from __future__ import annotations

from andrey_bench.datasets import DatasetKey

_CANONICAL = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": 2.0,
    "n": 200,
}

TINY_KEYS: list[DatasetKey] = [
    DatasetKey(**_CANONICAL, d=6, seed=0),
    DatasetKey(**_CANONICAL, d=6, seed=1),
    DatasetKey(**_CANONICAL, d=10, seed=0),
    DatasetKey(**_CANONICAL, d=10, seed=1),
]
