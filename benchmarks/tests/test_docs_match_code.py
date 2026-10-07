"""The flags the README documents are the flags `benchmark.py` accepts.

A renamed flag breaks every documented command with nothing failing, and the person who finds out is
a newcomer typing what the README told them to. This is the one part of the docs that is a set
comparison rather than prose, so it is the one part worth a test.

Everything else about the docs — whether a number is right, whether an explanation helps — is read
by a person.
"""

from __future__ import annotations

import re
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def test_the_readme_documents_exactly_the_flags_that_exist():
    readme = (BENCH / "README.md").read_text()
    benchmark = (BENCH / "benchmark.py").read_text()

    documented = set(re.findall(r"`(--[a-z][a-z-]*)`", readme))
    accepted = set(re.findall(r'add_argument\(\s*"(--[a-z][a-z-]*)"', benchmark))

    assert not documented - accepted, (
        f"README documents flags benchmark.py does not accept: {sorted(documented - accepted)}"
    )
    assert not accepted - documented, (
        f"benchmark.py accepts undocumented flags: {sorted(accepted - documented)}"
    )
