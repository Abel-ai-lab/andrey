"""Shared execution pipeline for benchmarks and validation.

:func:`run_batch` materializes datasets, fits solutions, and scores estimates against truth.
:func:`to_dataframe` flattens the resulting records for ``runs.parquet``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from andrey import GraphStructure
from andrey_bench import datasets, rundir, scoring
from andrey_bench.contracts import RunRecord, SolutionAdapter, Status, environment_id, run_id
from andrey_bench.datasets import DatasetKey
from andrey_bench.oracle import OracleGES
from andrey_bench.runner import (
    available_cores,
    check_interpreters,
    planned_env_fields,
    run_task,
    split_cores,
)


def _truth(marks: np.ndarray, kind: str = "dag", latents: int = 0) -> GraphStructure:
    """Convert endpoint marks for the true graph into a ``GraphStructure``.

    Flag the last ``latents`` nodes as hidden. Scoring then projects the truth onto the observed
    nodes present in the data.
    """
    marks = np.asarray(marks)
    types = datasets.latent_node_types(len(marks), latents)
    return GraphStructure.from_numpy(marks, kind=kind, node_types=types)


def assemble_record(
    key: DatasetKey,
    adapter: SolutionAdapter,
    repeat: int,
    measures: Mapping[str, Any],
    metrics: Mapping[str, float] | None,
    sort: Mapping[str, float] | None,
    *,
    machine_id: str,
    cores: int,
) -> RunRecord:
    """Build a ``RunRecord`` from dataset, solution, and runner outputs.

    ``run_id`` identifies the dataset, solution, repeat, and execution environment. Solution
    parameters are serialized with sorted keys.
    """
    # Required keys: missing values indicate runner/record schema drift.
    env = {
        "backend": adapter.backend,
        "mode": adapter.mode,
        "dtype": measures["dtype"],
        "device": measures["env_device"],
        "cap_mem_mb": measures["cap_mem_mb"],
        "cap_wall_s": measures["cap_wall_s"],
        "warmup": measures["warmup"],
        "machine_id": machine_id,
        "threads": measures["env_threads"],
        "num_workers": measures["env_num_workers"],
        "ges_min_work": measures["env_ges_min_work"],
        "hc_min_work": measures["env_hc_min_work"],
    }
    return RunRecord(
        run_id=run_id(dataset_id=key.digest, name=adapter.name, repeat=repeat, env=env),
        dataset_id=key.digest,
        name=adapter.name,
        algorithm=adapter.algorithm,
        package=adapter.package,
        package_version=getattr(adapter, "package_version", ""),
        backend=adapter.backend,
        mode=adapter.mode,
        output_type=adapter.output_type,
        params=json.dumps(dict(adapter.params()), sort_keys=True),
        machine_id=machine_id,
        cores=cores,
        device=measures["env_device"],
        dtype=measures["dtype"],
        topology=key.topology,
        functional=key.functional,
        noise=key.noise,
        standardize=key.standardize,
        density=key.density,
        d=key.d,
        n=key.n,
        data_seed=key.seed,
        repeat=repeat,
        status=measures.get("status", Status.OK.value),
        latents=key.latents,
        wall_s=measures.get("wall_s"),
        cpu_s=measures.get("cpu_s"),
        child_cpu_s=measures.get("child_cpu_s"),
        warmup_s=measures.get("warmup_s"),
        setup_s=measures.get("setup_s"),
        peak_rss_mb=measures.get("peak_rss_mb"),
        child_peak_rss_mb=measures.get("child_peak_rss_mb"),
        gpu_peak_alloc_mb=measures.get("gpu_peak_alloc_mb"),
        cap_wall_s=measures.get("cap_wall_s"),
        cap_mem_mb=measures.get("cap_mem_mb"),
        mem_route=measures.get("mem_route"),
        tree_peak_mem_mb=measures.get("tree_peak_mem_mb"),
        scheduler=measures.get("scheduler", ""),
        oom_source=measures.get("oom_source"),
        error_text=(measures.get("error_text") or measures.get("stderr")),
        exit_code=measures.get("exit_code"),
        elapsed_at_kill_s=measures.get("elapsed_at_kill_s"),
        output_hash=measures.get("output_hash"),
        structure_hash=measures.get("structure_hash"),
        ges_min_work=measures.get("env_ges_min_work"),
        hc_min_work=measures.get("env_hc_min_work"),
        varsortability=(sort or {}).get("varsortability"),
        r2_sortability=(sort or {}).get("r2_sortability"),
        # Child-observed, so they stay None when the child never reported (timeout / oom / error).
        device_id=measures.get("device_id"),
        threads=measures.get("threads"),
        num_workers=measures.get("num_workers"),
        warmup=measures.get("warmup"),
        metrics=dict(metrics or {}),
    )


def _unit_environment(
    adapter: SolutionAdapter,
    *,
    dtype: str,
    cap_mem_mb: float | None,
    cap_wall_s: float,
    warmup: int,
    machine_id: str,
    threads: int,
    num_workers: int,
    extra_env: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Return the planned execution environment used to identify a unit."""
    fields = planned_env_fields(
        adapter, threads=threads, num_workers=num_workers, extra_env=extra_env
    )
    return {
        "backend": adapter.backend,
        "mode": adapter.mode,
        "dtype": dtype,
        "device": fields["env_device"],
        "cap_mem_mb": cap_mem_mb,
        "cap_wall_s": cap_wall_s,
        "warmup": warmup,
        "machine_id": machine_id,
        "threads": fields["env_threads"],
        "num_workers": fields["env_num_workers"],
        "ges_min_work": fields["env_ges_min_work"],
        "hc_min_work": fields["env_hc_min_work"],
    }


@dataclass
class BatchResult:
    records: list[RunRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Units whose part file already existed. Only ever non-zero when ``records_dir`` is set.
    skipped: int = 0


def run_batch(
    out_dir: str | Path,
    adapters: Sequence[SolutionAdapter],
    keys: Sequence[DatasetKey],
    *,
    machine_id: str,
    dtype: str = "float64",
    cap_wall_s: float = 120.0,
    cap_mem_mb: float | None = None,
    mem_route: str | None = None,
    repeats: int = 1,
    warmup: int = 0,
    cores: int | None = None,
    extra_env: Mapping[str, str] | None = None,
    records_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> BatchResult:
    """Fit every adapter on every dataset and return one record per repeat.

    Datasets are materialized once and estimates scored against generated truth. Failed, timed-out,
    and OOM fits remain records; launch failures are added to ``errors``.

    ``cores`` sets the per-task budget, split by adapter mode. ``cap_wall_s`` and ``cap_mem_mb``
    set resource limits; ``extra_env`` is forwarded to child processes.

    With ``records_dir``, completed dataset/solution units are written incrementally and existing
    units skipped on resume. ``run_dir`` is the run's ``--out``, where estimated graphs are stored.
    ``out_dir`` may be shared through ``ANDREY_BENCH_DATA``; keeping graphs in ``run_dir`` prevents
    collisions because ``run_id`` excludes the build.
    """
    out_dir = Path(out_dir)
    records_dir = Path(records_dir) if records_dir is not None else None
    cores = cores or available_cores()
    # An adapter defined in a driver pickles as `__main__.Name` and cannot be rebuilt in the child,
    # so every one of its fits returns an error record. Checked before the first dataset is
    # materialized, so no measurement is taken first.
    for adapter in adapters:
        if type(adapter).__module__ == "__main__":
            raise SystemExit(
                f"{type(adapter).__name__} is defined in the driver, so the worker cannot "
                "unpickle it and every fit would fail without saying why. Move it into an "
                "importable module under benchmarks/."
            )
    # Every solution's interpreter, resolved before anything is measured. A solution that names an
    # environment of its own fails here if it was never built - otherwise the loop below records one
    # error per unit and the ladder runs to the end before anyone learns the campaign's competitor
    # was never measured.
    check_interpreters(adapters)
    registry = datasets.materialize(keys, out_dir)
    result = BatchResult()

    for key in keys:
        data, marks = datasets.load(out_dir, key, dtype=dtype)
        # The true-graph kind as datasets.materialize recorded it, so scoring never guesses the
        # truth's family.
        truth = _truth(marks, registry["keys"][key.label]["graph_kind"], key.latents)
        sort = scoring.sortability(data, truth)

        for adapter in adapters:
            params = dict(adapter.params())
            if isinstance(adapter, OracleGES):
                # Truth is runtime input for the oracle, not solution configuration.
                params.update(OracleGES.truth_params(truth))
            # Equal total cores across solutions, split per the adapter's mode.
            threads, num_workers = split_cores(cores, adapter.mode)

            part = None
            if records_dir is not None:
                unit_env = environment_id(
                    _unit_environment(
                        adapter,
                        dtype=dtype,
                        cap_mem_mb=cap_mem_mb,
                        cap_wall_s=cap_wall_s,
                        warmup=warmup,
                        machine_id=machine_id,
                        threads=threads,
                        num_workers=num_workers,
                        extra_env=extra_env,
                    )
                )
                part = rundir.part_path(records_dir, key.digest, adapter.name, unit_env)
                if part.exists():
                    result.skipped += 1
                    continue

            try:
                # Hand the child the materialized file path (serialized once by materialize) rather
                # than the in-memory array. The runner never re-serializes per repeat. The
                # parent keeps its own `data` (loaded above) for sortability/truth scoring.
                measures_list = run_task(
                    adapter,
                    datasets.data_path(out_dir, key),
                    params,
                    cap_wall_s=cap_wall_s,
                    cap_mem_mb=cap_mem_mb,
                    mem_route=mem_route,
                    repeats=repeats,
                    warmup=warmup,
                    threads=threads,
                    num_workers=num_workers,
                    dtype=dtype,
                    extra_env=extra_env,
                )
            except Exception as exc:  # nothing was measured, so there is nothing to keep
                result.errors.append(f"{adapter.name} @ {key.label}: {type(exc).__name__}: {exc}")
                continue

            unit_records: list[RunRecord] = []
            for repeat, measures in enumerate(measures_list):
                metrics, est = None, None
                try:
                    if (
                        measures.get("status") == Status.OK.value
                        and measures.get("adj") is not None
                    ):
                        est = adapter.to_structure(measures["adj"])
                        metrics = scoring.score_result(est, truth)
                        measures = {**measures, "structure_hash": scoring.structure_hash(est)}
                except Exception as exc:
                    # Preserve parse/scoring failures as error records.
                    result.errors.append(
                        f"{adapter.name} @ {key.label} repeat {repeat}: {type(exc).__name__}: {exc}"
                    )
                    metrics, measures = (
                        None,
                        {
                            **measures,
                            "status": Status.ERROR.value,
                            "error_text": f"{type(exc).__name__}: {exc}",
                        },
                    )
                record = assemble_record(
                    key,
                    adapter,
                    repeat,
                    measures,
                    metrics,
                    sort,
                    machine_id=machine_id,
                    cores=cores,
                )
                unit_records.append(record)
                # Keep the graph alongside the numbers scored off it: nothing else records what a
                # solution returned.
                if est is not None and run_dir is not None:
                    rundir.write_structure(run_dir, record.run_id, est)

            if part is not None and unit_records:
                # Part-file identity must match the environment encoded in its records.
                carried = unit_records[0].run_id.rsplit(":", 1)[-1]
                if carried != unit_env:
                    raise RuntimeError(
                        f"{adapter.name} @ {key.label}: part named for environment {unit_env}, "
                        f"records carry {carried}; the pre-flight environment has drifted from "
                        "the runner's"
                    )
                rundir.write_part(part, unit_records)
            result.records.extend(unit_records)
    return result


# Every RunRecord scalar field in declaration order, excluding ``metrics`` (flattened below).
_RUNRECORD_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(RunRecord) if f.name != "metrics"
)

# Canonical metric columns from ``RunRecord.metrics`` (see ``scoring.score_result``). Families that
# do not define a metric, such as ``mec_shd`` for CPDAG estimates, receive NaN.
_METRIC_COLUMNS: tuple[str, ...] = (
    "family",
    "shd",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "arrowhead_precision",
    "arrowhead_recall",
    "arrowhead_f1",
    "mec_shd",
    "mec_arrowhead_f1",
)


def to_dataframe(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Flatten ``RunRecord`` objects into the benchmark table.

    Scalar fields become columns and metric dictionaries are expanded into
    metric columns. Missing metrics are filled with NaN, and non-canonical
    metrics are preserved as additional columns.

    """
    rows: list[dict[str, Any]] = []
    extra_metric_keys: set[str] = set()
    for record in records:
        flat = record.as_dict()
        metrics = flat.pop("metrics", None) or {}
        extra_metric_keys.update(k for k in metrics if k not in _METRIC_COLUMNS)
        rows.append({**flat, **metrics})

    columns = list(_RUNRECORD_FIELDS) + list(_METRIC_COLUMNS) + sorted(extra_metric_keys)
    return pd.DataFrame.from_records(rows).reindex(columns=columns)
