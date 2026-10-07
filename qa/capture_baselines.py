"""Record Andrey outputs as regression baselines in ``tests/recovery/baselines/``.

Each runner mirrors its regression test and uses the shared output projection.
Capture runs on demand; committed fixtures are checked in CI.

Recapturing rewrites committed values, so a diff is the point: read it before committing. Numeric
output only; no host or hardware identifier is recorded.

Usage::

    python -m qa.capture_baselines                        # all algorithms, in place
    python -m qa.capture_baselines --algorithms PC GES
    python -m qa.capture_baselines --out-dir /tmp/check     # write elsewhere to diff first
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import BASELINES_DIR, SCHEMA_VERSION
from tests.recovery.helpers.project import project_output

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Runners mirror the Andrey calls made by the regression tests ------------------


def _facade(name: str, **kwargs):
    def run(X: np.ndarray, algorithm: str) -> dict:
        import andrey

        return project_output(getattr(andrey, name)(X, **kwargs), algorithm)

    return run


def _run_cdnod(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    return project_output(andrey.cdnod(X, datagen.domain_index("gauss_5v")), algorithm)


def _run_varlingam(X: np.ndarray, algorithm: str) -> dict:
    from andrey.api._temporal import _adapt_var_lingam
    from andrey.lingam.var import var_lingam

    order, adj = var_lingam(X)
    model = SimpleNamespace(adjacency_matrices_=adj, causal_order_=order)
    return project_output(_adapt_var_lingam(model), algorithm)


def _run_varmalingam(X: np.ndarray, algorithm: str) -> dict:
    from andrey.temporal.varma import varma_lingam

    order, psis, omegas = varma_lingam(X, order=(1, 1), prune=False)
    return {"causal_order": list(order), "psis": np.asarray(psis), "omegas": np.asarray(omegas)}


def _run_granger(X: np.ndarray, algorithm: str) -> dict:
    from andrey.temporal.granger import granger_lasso

    coeff = np.asarray(granger_lasso(X), dtype=np.float64)
    return {"coeff": coeff, "adj": (coeff != 0.0).astype(int)}


def _run_multi_group_direct_lingam(groups: list[np.ndarray], algorithm: str) -> dict:
    import andrey

    return project_output(andrey.multi_group_direct_lingam(groups), algorithm)


# algorithm -> (runner, case id). Single-input methods take one matrix.
ALGORITHMS = {
    "DirectLiNGAM": (_facade("direct_lingam"), "lingam_5v_uniform"),
    "ICALiNGAM": (_facade("ica_lingam"), "lingam_5v_uniform"),
    "PC": (_facade("pc"), "gauss_5v"),
    "FCI": (_facade("fci"), "gauss_5v"),
    "GFCI": (_facade("gfci", lambda_value=2.0), "gauss_5v"),
    "CDNOD": (_run_cdnod, "gauss_5v"),
    "GES": (_facade("ges", lambda_value=2.0), "gauss_5v"),
    "GIES": (_facade("gies", lambda_value=2.0), "gauss_5v"),
    "HC": (_facade("hc"), "gauss_5v"),
    "ExactSearch": (_facade("exact_search"), "gauss_5v"),
    "BOSS": (_facade("boss", lambda_value=2.0), "gauss_5v"),
    "GRaSP": (_facade("grasp", lambda_value=2.0), "gauss_5v"),
    "VARLiNGAM": (_run_varlingam, "var_4v_stable"),
    "VARMALiNGAM": (_run_varmalingam, "varma_3v"),
    "Granger": (_run_granger, "var_4v_stable"),
}

# Multi-group methods take a list of matrices (runner receives ``generate_groups(case)``).
GROUP_ALGORITHMS = {
    "MultiGroupDirectLiNGAM": (_run_multi_group_direct_lingam, "mgdl_2groups"),
}


def _jsonable(value):
    """Convert numpy arrays and scalars to plain JSON types, recursing into containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def capture(args: argparse.Namespace) -> None:
    # Baselines are captured at seed 0; an exported ANDREY_SEED would flow into the seeded searches
    # (GRaSP, BOSS) and record a different graph. tests/recovery/conftest.py strips it the same way.
    os.environ.pop("ANDREY_SEED", None)

    out_root = Path(args.out_dir) if args.out_dir else BASELINES_DIR
    algorithms = args.algorithms or (list(ALGORITHMS) + list(GROUP_ALGORITHMS))
    for name in algorithms:
        # Branch so each runner is called with its concrete input type (one matrix vs a list).
        if name in GROUP_ALGORITHMS:
            group_runner, case = GROUP_ALGORITHMS[name]
            groups = datagen.generate_groups(case)
            input_block = {
                "case": case,
                "seeds": [group["seed"] for group in datagen.GROUP_CASES[case]["groups"]],
                "shapes": [list(X.shape) for X in groups],
                "sha256": datagen.input_sha256(case),
            }
            output = group_runner(groups, name)
        else:
            runner, case = ALGORITHMS[name]
            X = datagen.generate(case)
            input_block = {
                "case": case,
                "seed": datagen.CASES[case]["seed"],
                "shape": list(X.shape),
                "sha256": datagen.input_sha256(case),
            }
            output = runner(X, name)

        doc = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": name,
            "case": case,
            "input": input_block,
            "output": _jsonable(output),
        }

        out_path = out_root / name / f"{case}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
        rel = out_path.relative_to(_REPO_ROOT) if out_root is BASELINES_DIR else out_path
        print(f"  wrote {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write here instead of tests/recovery/baselines (to diff first)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        choices=list(ALGORITHMS) + list(GROUP_ALGORITHMS),
        default=None,
    )
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
