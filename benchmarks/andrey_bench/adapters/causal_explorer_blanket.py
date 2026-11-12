"""Causal Explorer's HITON-PC, HITON-MB, and MMMB: the authors' reference of the set-learner lane.

Causal Explorer 1.5 is the MATLAB toolbox of the authors of HITON-PC, HITON-MB, and MMMB
(Aliferis et al. 2003, 2010; Tsamardinos et al. 2003), distributed as P-code under a non-commercial
single-user license in ``mensxmachina/CausalExplorer_1.5`` (commit :data:`COMMIT`). It is a
measurement reference only: ``benchmarks/start_causal_explorer.sh`` clones the repository at that
commit into the directory ``$ANDREY_BENCH_CAUSAL_EXPLORER`` names (``benchmarks/.causal-explorer``
by default), and the adapter calls it from there through MATLAB and copies none of it. Without
``matlab`` on the ``PATH`` or without the toolbox, :func:`available` is false and the lane leaves
the arm out, so nothing about it runs in CI.

A fit writes the data and the settings to a new temporary directory, starts ``matlab -batch`` once
with one computational thread, and calls the toolbox's function for every target there (the runner
is :data:`RUNNER`, written beside the data). It answers for targets only; the toolbox learns no
whole graph from these functions. Its wall time includes MATLAB's start, so it is not a speed
measure. The toolbox reports no test count.

The settings the lane passes, and how each reaches the toolbox:

- **Data.** Each categorical column is coded ``0..k-1`` over its observed values, and
  ``domain_counts`` holds each ``k``, the number of levels Andrey's tests read. Continuous data go
  as they are, with ``domain_counts = []``.
- **Test.** ``gsq`` is the toolbox's ``'g2'``, its G-square test, which takes the empty cells out
  of the degrees of freedom and runs a test only with at least five rows per cell of its table.
  Andrey's ``gsq`` takes the mean of the statistic under a shuffle as the degrees of freedom, and
  HITON's tests in Andrey run with the same five rows per cell. ``fisherz`` is ``'z'``, which the
  toolbox's manual takes only with a categorical target for HITON; MMMB reads categorical data
  only. ``chisq`` has no counterpart.
- **Significance level.** ``alpha`` is the ``threshold``, a p-value, for every function.
- **Conditioning sets.** ``max_depth`` is HITON's last argument, the largest conditioning set, and
  MMMB's ``maxK``; ``None`` (no cap) passes ``d - 1``. MMMB also caps the size by its own rule of
  :data:`MMMB_CELL_ROWS` rows per cell (``epc``), which has no off switch.
- **Symmetry.** HITON-PC's sets are kept to the pairs in each other's set, the AND rule of Andrey's
  ``hiton_pc``, so the adapter runs HITON-PC for every variable and keeps ``x`` in ``t``'s set when
  ``t`` is in ``x``'s (``symmetry=False`` returns the toolbox's own sets). HITON-MB and MMMB give
  the toolbox's blanket as it is.

The toolbox's HITON functions date from its 2005 release and are not documented as the
semi-interleaved variant of the 2010 paper; the lane compares them as they are.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.blanket_lane import (
    NO_COUNT,
    STRUCTURE_PARAM,
    TARGETS_PARAM,
    answer_structure,
    pack_answer,
)

#: Names the directory holding the toolbox.
TOOLBOX_ENV = "ANDREY_BENCH_CAUSAL_EXPLORER"

#: The pinned commit of ``mensxmachina/CausalExplorer_1.5``; ``start_causal_explorer.sh`` fetches it.
COMMIT = "205a31f65577339ffb9a465ecc9fef0d4b04ca76"

#: The kind of set each function answers with, by the toolbox's name.
FUNCTIONS: dict[str, str] = {"HITON_PC": "pc", "HITON_MB": "mb", "MMMB": "mb"}

#: The function that runs the algorithm of each learner of Andrey, by the learner's name.
COUNTERPART: dict[str, str] = {"hiton_pc": "HITON_PC", "hiton_mb": "HITON_MB", "mmmb": "MMMB"}

#: The toolbox's statistic for each of Andrey's tests, by function.
TESTS: dict[str, dict[str, str]] = {
    "HITON_PC": {"gsq": "g2", "fisherz": "z"},
    "HITON_MB": {"gsq": "g2", "fisherz": "z"},
    "MMMB": {"gsq": "g2"},
}

#: MMMB's ``epc``: the least rows per cell of a table for its test to run, the toolbox's default.
MMMB_CELL_ROWS = 5

#: The longest a fit may run MATLAB, seconds; the lane's own wall cap is usually shorter.
TIMEOUT_S = 7200.0

_DEFAULT_ALPHA = 0.05

#: The MATLAB runner: one call of the function per target, one line each in ``out.txt``.
RUNNER = r"""function andrey_ce_run(toolbox, io)
% One Causal Explorer function for each target in io/config.json; writes io/out.txt.
addpath(fullfile(toolbox, 'causal_explorer'));
cfg = jsondecode(fileread(fullfile(io, 'config.json')));
data = load(fullfile(io, 'data.txt'), '-ascii');
dc = cfg.domain_counts;
if isempty(dc), dc = []; else, dc = dc(:)'; end
fid = fopen(fullfile(io, 'out.txt'), 'w');
fprintf(fid, 'matlab\t%s\n', version);
for t = cfg.targets(:)'
  tic;
  try
    if strcmp(cfg.function, 'MMMB')
      options = struct('threshold', cfg.threshold, 'epc', cfg.epc, 'maxK', cfg.max_k, ...
                       'use_card_lim', 0, 'max_card', 0);
      found = Causal_Explorer('MMMB', data, t, dc, 'MMPC', options);
    else
      found = Causal_Explorer(cfg.function, data, t, dc, cfg.statistic, cfg.threshold, cfg.max_k);
    end
    fprintf(fid, '%d\tok\t%.6f\t%s\n', t, toc, strtrim(sprintf('%d ', found)));
  catch err
    fprintf(fid, '%d\terror\t%.6f\t%s\n', t, toc, strrep(err.message, newline, ' '));
  end
end
fclose(fid);
end
"""


def toolbox_dir() -> Path | None:
    """The toolbox's directory, or ``None`` when ``$ANDREY_BENCH_CAUSAL_EXPLORER`` names none."""
    raw = os.environ.get(TOOLBOX_ENV, "").strip()
    if not raw:
        return None
    directory = Path(raw)
    return directory if (directory / "causal_explorer" / "Causal_Explorer.p").is_file() else None


def available() -> bool:
    """Whether a fit can run: ``matlab`` on the ``PATH`` and the toolbox in its directory."""
    return shutil.which("matlab") is not None and toolbox_dir() is not None


def environment() -> dict[str, Any]:
    """What a run records about the arm: the commit, the toolbox's directory, and MATLAB's path."""
    found = toolbox_dir()
    return {
        "commit": COMMIT,
        "toolbox": None if found is None else str(found),
        "matlab": shutil.which("matlab"),
    }


def encode(data: np.ndarray, discrete: bool) -> tuple[np.ndarray, list[int]]:
    """The data as the toolbox reads it, and ``domain_counts``.

    A categorical column becomes the codes ``0..k-1`` of its observed values in ascending order,
    and its count is ``k``; continuous data stay as they are, with no counts.
    """
    X = np.asarray(data, dtype=np.float64)
    if not discrete:
        return X, []
    codes = np.empty(X.shape, dtype=np.int64)
    counts = []
    for j in range(X.shape[1]):
        levels, codes[:, j] = np.unique(X[:, j], return_inverse=True)
        counts.append(int(levels.size))
    return codes, counts


def parse_output(text: str, d: int) -> dict[int, list[int]]:
    """Each target's set, 0-based, from the runner's ``out.txt``; an error line raises."""
    sets: dict[int, list[int]] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if fields[0] == "matlab" or not line.strip():
            continue
        target, status = int(fields[0]) - 1, fields[1]
        rest = fields[3] if len(fields) > 3 else ""
        if status != "ok":
            raise RuntimeError(f"Causal Explorer failed at target {target}: {rest}")
        members = sorted(int(v) - 1 for v in rest.split())
        if any(not 0 <= m < d or m == target for m in members):
            raise RuntimeError(f"Causal Explorer gave target {target} the set {members}")
        sets[target] = members
    return sets


def and_rule(sets: Mapping[int, list[int]], targets: list[int]) -> dict[int, list[int]]:
    """Each target's set kept to the members whose own set holds the target."""
    return {t: [x for x in sets[t] if t in sets.get(x, ())] for t in targets}


class CausalExplorerBlanket:
    """Solution adapter for one function of :data:`FUNCTIONS`, at one setting.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter` for a run that answers for targets
    (``params["targets"]``). ``symmetry`` applies the AND rule to HITON-PC's sets (see the module
    docstring).
    """

    package: str = "causal-explorer"
    package_version: str = f"1.5+{COMMIT[:7]}"
    backend: str = "matlab"
    mode: str = "serial"

    def __init__(
        self,
        function: str,
        *,
        indep_test: str,
        alpha: float = _DEFAULT_ALPHA,
        max_depth: int | None = None,
        symmetry: bool = True,
    ) -> None:
        if function not in FUNCTIONS:
            raise ValueError(f"function must be one of {sorted(FUNCTIONS)}, got {function!r}")
        if indep_test not in TESTS[function]:
            raise ValueError(
                f"{function} takes the tests {sorted(TESTS[function])}, got {indep_test!r}"
            )
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth is None or at least 0, got {max_depth}")
        self.function = function
        self.kind = FUNCTIONS[function]
        self.indep_test = indep_test
        self.alpha = alpha
        self.max_depth = max_depth
        self.symmetry = symmetry and function == "HITON_PC"
        self.name = "causal-explorer." + function.lower().replace("_", "-")
        if function == "HITON_PC" and not symmetry:
            self.name += ".raw"
        self.algorithm = function
        self.output_type = "undirected"

    def setup(self) -> None:
        """Fail before timing when MATLAB or the toolbox is absent."""
        if not available():
            raise RuntimeError(
                f"Causal Explorer needs matlab on the PATH and ${TOOLBOX_ENV}; run "
                "benchmarks/start_causal_explorer.sh"
            )

    def params(self) -> dict[str, Any]:
        """Every setting of the run, in Andrey's names, and the toolbox's statistic."""
        return {
            "indep_test": self.indep_test,
            "statistic": TESTS[self.function][self.indep_test],
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "kind": self.kind,
            "symmetry": self.symmetry,
            STRUCTURE_PARAM: None,
        }

    def config(self, d: int, targets: list[int], counts: list[int]) -> dict[str, Any]:
        """The runner's ``config.json``: the function, its settings, and 1-based targets."""
        max_depth = self.max_depth
        return {
            "function": self.function,
            "statistic": TESTS[self.function][self.indep_test],
            "threshold": float(self.alpha),
            "max_k": d - 1 if max_depth is None else int(max_depth),
            "epc": MMMB_CELL_ROWS,
            "domain_counts": counts,
            "targets": [int(t) + 1 for t in targets],
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the function for each target; return
        :func:`~andrey_bench.blanket_lane.pack_answer`'s array, with no test counts. The only timed
        region, MATLAB's start included."""
        targets = params.get(TARGETS_PARAM)
        if targets is None:
            raise ValueError("Causal Explorer answers for targets only; pass params['targets']")
        toolbox, matlab = toolbox_dir(), shutil.which("matlab")
        if toolbox is None or matlab is None:
            raise RuntimeError(f"Causal Explorer needs matlab on the PATH and ${TOOLBOX_ENV}")
        targets = [int(t) for t in targets]
        X, counts = encode(data, discrete=self.indep_test != "fisherz")
        d = X.shape[1]
        asked = list(range(d)) if self.symmetry else targets
        with tempfile.TemporaryDirectory(prefix="andrey-ce-") as raw:
            io = Path(raw)
            np.savetxt(io / "data.txt", X, fmt="%d" if counts else "%.17g")
            (io / "config.json").write_text(json.dumps(self.config(d, asked, counts)))
            (io / "andrey_ce_run.m").write_text(RUNNER)
            call = f"addpath('{io}'); andrey_ce_run('{toolbox}', '{io}')"
            done = subprocess.run(
                [matlab, "-singleCompThread", "-batch", call],
                cwd=io,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                check=False,
            )
            out = io / "out.txt"
            if done.returncode != 0 or not out.is_file():
                tail = (done.stderr or done.stdout)[-2000:]
                raise RuntimeError(f"matlab exited with {done.returncode}: {tail}")
            sets = parse_output(out.read_text(), d)
        if self.symmetry:
            sets = and_rule(sets, targets)
        answer = np.zeros((d, d), dtype=np.int8)
        for t in targets:
            answer[t, sets[t]] = 1
        return pack_answer(answer, np.full(d, NO_COUNT, dtype=np.int64), None)

    def to_structure(self, native: np.ndarray) -> Any:
        """The answer as a ``GraphStructure``; untimed. See ``answer_structure``."""
        return answer_structure(native, None)


__all__ = [
    "COMMIT",
    "COUNTERPART",
    "FUNCTIONS",
    "MMMB_CELL_ROWS",
    "TESTS",
    "CausalExplorerBlanket",
    "and_rule",
    "available",
    "encode",
    "environment",
    "parse_output",
]
