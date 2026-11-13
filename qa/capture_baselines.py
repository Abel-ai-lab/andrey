"""Record Andrey outputs as regression baselines in ``tests/recovery/baselines/``.

``tests/recovery/test_regression_baselines.py`` fits each recorded output with its runner here,
which uses the shared output projection.
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

import numpy as np

import andrey
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import BASELINES_DIR, SCHEMA_VERSION
from tests.recovery.helpers.project import project_output

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Runners --------------------------------------------------------------------------------------


def _facade(name: str, **kwargs):
    def run(X: np.ndarray, algorithm: str) -> dict:
        import andrey

        return project_output(getattr(andrey, name)(X, **kwargs), algorithm)

    return run


def _run_cdnod(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    return project_output(andrey.cdnod(X, datagen.domain_index("gauss_5v")), algorithm)


def _run_fci_jci(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    shifted, index = datagen.domain_shift("gauss_5v", (0,))
    return project_output(andrey.fci_jci(shifted, index), algorithm)


def _run_gnies(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    env = datagen.environment_index("gauss_5v_noise_shift")
    return project_output(andrey.gnies(X, env), algorithm)


def _run_icp(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    target = int(datagen.column_of("gauss_5v")[3])  # the column of x3, whose parents are x1, x2
    return project_output(andrey.icp(X, datagen.domain_index("gauss_5v"), target=target), algorithm)


def _run_ut_igsp(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    env = datagen.environment_index("gauss_5v_noise_shift")
    return project_output(andrey.ut_igsp(X, env, targets={0: [], 1: None, 2: None}), algorithm)


def _run_ci_test(X: np.ndarray, algorithm: str) -> dict:
    from tests.recovery.helpers.ci_tests import recorded_pvalues

    return recorded_pvalues(algorithm, X, "nonmonotone_4v")


def _run_multi_group_direct_lingam(groups: list[np.ndarray], algorithm: str) -> dict:
    import andrey

    return project_output(andrey.multi_group_direct_lingam(groups), algorithm)


def _run_igsp(X: np.ndarray, algorithm: str) -> dict:
    import andrey

    case = "gauss_5v_noise_shift"
    column = list(datagen.column_order(case)).index
    targets = {0: [], 1: [column(3)], 2: [column(4)]}
    return project_output(
        andrey.igsp(X, datagen.environment_index(case), targets=targets), algorithm
    )


# algorithm -> (runner, case id). Single-input methods take one matrix.
ALGORITHMS = {
    "DirectLiNGAM": (_facade("direct_lingam"), "lingam_5v_uniform"),
    "ICALiNGAM": (_facade("ica_lingam"), "lingam_5v_uniform"),
    "PC": (_facade("pc"), "gauss_5v"),
    "FCI": (_facade("fci"), "gauss_5v"),
    "GFCI": (_facade("gfci", penalty_discount=2.0), "gauss_5v"),
    "BOSSFCI": (_facade("boss_fci"), "gauss_5v"),
    "GRaSPFCI": (_facade("grasp_fci"), "gauss_5v"),
    "CDNOD": (_run_cdnod, "gauss_5v"),
    "FCIJCI": (_run_fci_jci, "gauss_5v"),
    "ICP": (_run_icp, "gauss_5v"),
    "GES": (_facade("ges", penalty_discount=2.0), "gauss_5v"),
    "GIES": (_facade("gies", penalty_discount=2.0), "gauss_5v"),
    "GnIES": (_run_gnies, "gauss_5v_noise_shift"),
    "IGSP": (_run_igsp, "gauss_5v_noise_shift"),
    "XGES": (_facade("xges"), "gauss_5v"),
    "AGES": (_facade("ages"), "weak_triangle_3v"),
    "HC": (_facade("hc"), "gauss_5v"),
    "Tabu": (_facade("tabu"), "gauss_5v"),
    "IAMB": (_facade("iamb"), "gauss_5v"),
    "InterIAMB": (_facade("inter_iamb"), "gauss_5v"),
    "GS": (_facade("gs"), "gauss_5v"),
    "FastIAMB": (_facade("fast_iamb"), "gauss_5v"),
    "MMPC": (_facade("mmpc", structure="cpdag"), "gauss_5v"),
    "HITONPC": (_facade("hiton_pc", structure="cpdag"), "gauss_5v"),
    "MMHC": (_facade("mmhc"), "gauss_5v"),
    "RSMAX2": (_facade("rsmax2", restrict="pc", maximize="tabu"), "gauss_5v"),
    "ExactSearch": (_facade("exact_search"), "gauss_5v"),
    "BOSS": (_facade("boss", penalty_discount=2.0), "gauss_5v"),
    "GRaSP": (_facade("grasp", penalty_discount=2.0), "gauss_5v"),
    "UT-IGSP": (_run_ut_igsp, "gauss_5v_noise_shift"),
    "CALM": (_facade("calm"), "gauss_tree_5v"),
    "NOTEARS": (_facade("notears"), "gauss_5v"),
    "CAM": (_facade("cam"), "nonmonotone_4v"),
    "VARMALiNGAM": (_facade("varma_lingam", order=(1, 1), prune=False), "varma_3v"),
    "VARMALiNGAMPruned": (_facade("varma_lingam", order=(1, 1), prune=True), "varma_3v"),
    "Granger": (_facade("granger", tau_max=2, fdr_method="none"), "var_4v_stable"),
    "DYNOTEARS": (_facade("dynotears"), "svar_4v_gauss"),
    "RCD": (_facade("rcd"), "confounded_fork_4v"),
    "BottomUpParceLiNGAM": (_facade("bottom_up_parce_lingam", alpha=0.01), "confounded_fork_4v"),
    "ANM": (_facade("anm"), "anm_pair_cubic"),
    "PNL": (_facade("pnl"), "anm_pair_cubic"),
    "LOCI": (_facade("loci"), "lsnm_pair"),
    "GIN": (_facade("gin"), "measurement_6v"),
    "RESIT": (_facade("resit"), "nonmonotone_4v"),
    "ScoreMatching": (_facade("score_matching"), "rational_chain_4v"),
    "NoGAM": (_facade("nogam"), "rational_chain_4v"),
    "CAMUV": (_facade("camuv"), "nonlinear_fork_4v"),
    "PCMCI": (_facade("pcmci", tau_max=2, indep_test="student_t"), "var_3v_lag2"),
    "PCMCIplus": (_facade("pcmciplus", tau_max=2, indep_test="student_t"), "var_4v_stable"),
    "LPCMCI": (_facade("lpcmci", tau_max=2, indep_test="student_t"), "var_6v_hidden"),
    "KCI": (_run_ci_test, "nonmonotone_4v"),
    "RCoT": (_run_ci_test, "nonmonotone_4v"),
    "CMIknn": (_run_ci_test, "nonmonotone_4v"),
    "GPDC": (_run_ci_test, "nonmonotone_4v"),
}

# Multi-group, panel, and multi-dataset methods take a list of matrices (runner receives
# ``generate_groups(case)``).
GROUP_ALGORITHMS = {
    "MultiGroupDirectLiNGAM": (_run_multi_group_direct_lingam, "mgdl_2groups"),
    "LongitudinalLiNGAM": (_facade("longitudinal_lingam"), "panel_4v"),
    "JPCMCIplus": (_facade("jpcmciplus", tau_max=1), "context_var_4v"),
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
    # Baselines record NumPy numerics whatever the ANDREY_* variables pin; the recovery suite
    # compares under the same pin (tests/recovery/conftest.py).
    with andrey.config(backend="numpy"):
        for name in algorithms:
            # Branch so each runner is called with its concrete input type (one matrix vs a list).
            if name in GROUP_ALGORITHMS:
                group_runner, case = GROUP_ALGORITHMS[name]
                groups = datagen.generate_groups(case)
                spec = datagen.GROUP_CASES[case]
                seeds = [g["seed"] for g in spec["groups"]] if "groups" in spec else [spec["seed"]]
                input_block = {
                    "case": case,
                    "seeds": seeds,
                    "column_order": datagen.column_order(case).tolist(),
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
                    "column_order": datagen.column_order(case).tolist(),
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
