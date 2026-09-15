#!/usr/bin/env python3
"""Plan, submit, and audit the G1 canonical shared-orbital A/B gate.

The immutable plan always describes six fresh-process jobs in this order:
water2 legacy/resident, water4 legacy/resident, and water8 legacy/resident.
Execution is intentionally split.  The first receipt may contain only the
water2/water4 gate.  A second water8 receipt is accepted only after a
fail-closed analysis proves that both smaller cases passed the numerical,
resource, residency, transfer, source, orbital, and topology contracts.

``plan`` and ``submit`` are previews unless an explicit output or ``--execute``
is supplied.  Plans, analyses, and receipts are always created exclusively;
an existing path is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable, Sequence

try:
    from . import compare_canonical_checkpoints as _checkpoint_comparator
except ImportError:  # direct execution from this directory
    _module_root = str(Path(__file__).resolve().parent)
    if _module_root not in sys.path:
        sys.path.insert(0, _module_root)
    import compare_canonical_checkpoints as _checkpoint_comparator


SCHEMA = "gpu4pyscf.water8.g1-ab-analysis.v2"
PLAN_SCHEMA = "gpu4pyscf.water8.g1-ab-plan.v2"
SUBMISSION_SCHEMA = "gpu4pyscf.water8.g1-ab-submission.v2"
NODE = "compute-1-6"
CASES = ("water2-tz", "water4-tz", "water8-tz")
GATE_CASES = CASES[:2]
METHODS = ("canonical_legacy", "canonical")
ENERGY_LIMIT_EH = 1.0e-8
RESIDUAL_LIMIT = 1.0e-6
AMPLITUDE_LIMIT = 1.0e-6
HBM_LIMIT_GIB = 72.0
RSS_LIMIT_GIB = 110.0
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCY_PATTERN = re.compile(r"^afterok:[1-9][0-9]*(?::[1-9][0-9]*)*$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _plan_sha256(plan: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(plan).encode("ascii")).hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _json_write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain one JSON object")
    return value


def _read_json_evidence(
    path: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = _read_json(resolved, label)
    return payload, {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "payload_sha256": _payload_sha256(payload),
    }


def _preflight_output_absent(path: Path, label: str) -> None:
    """Reject an existing output before an irreversible external action."""

    resolved = path.expanduser()
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite {label} {resolved}")


def _source_digest(source_root: Path, task_root: Path) -> str:
    source = source_root.expanduser().resolve()
    task = task_root.expanduser().resolve()
    try:
        relative = source.relative_to(task / "snapshots")
    except ValueError as exc:
        raise ValueError(
            "source root must be below <task-root>/snapshots/<sha256>/source"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[1] != "source":
        raise ValueError(
            "source root must equal <task-root>/snapshots/<sha256>/source"
        )
    digest = relative.parts[0]
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("source snapshot directory must be a lowercase SHA-256")
    return digest


def _case_token(case_id: str) -> str:
    if case_id not in CASES:
        raise ValueError(f"unsupported G1 case {case_id!r}")
    return case_id.removesuffix("-tz")


def _job_id(case_id: str, method: str) -> str:
    token = _case_token(case_id)
    return f"{token}-legacy" if method == "canonical_legacy" else f"{token}-resident"


def _repeat_id(run_id: str, case_id: str, method: str) -> str:
    token = _case_token(case_id)
    route = "legacy-a" if method == "canonical_legacy" else "resident-b"
    return f"{run_id}-{token}-{route}"


def make_plan(
    *,
    task_root: Path,
    source_root: Path,
    run_id: str,
    initial_dependency: str | None = None,
) -> dict[str, Any]:
    """Return the fixed six-job, shared-orbital G1 A/B plan."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run id must start with an alphanumeric character and contain "
            "only letters, digits, dot, underscore, or hyphen"
        )
    if initial_dependency is not None and DEPENDENCY_PATTERN.fullmatch(
        initial_dependency
    ) is None:
        raise ValueError(
            "initial dependency must have Slurm form afterok:<jobid>[:<jobid>...]"
        )
    task = task_root.expanduser().resolve()
    source = source_root.expanduser().resolve()
    digest = _source_digest(source, task)
    result_root = task / "results" / "g1-ab" / run_id
    timing_root = result_root / "timing"
    orbital_root = result_root / "orbitals"
    launcher = (
        source / "benchmarks" / "cc" / "a100_water8" /
        "run_mtu_benchmark.sbatch"
    )
    common = {
        "CCSD_TASK_ROOT": str(task),
        "CCSD_SOURCE_ROOT": str(source),
        "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate",
        "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
    }
    jobs: list[dict[str, Any]] = []
    predecessor: str | None = None
    for case_id in CASES:
        token = _case_token(case_id)
        artifact = orbital_root / f"{case_id}.npz"
        for method in METHODS:
            job_id = _job_id(case_id, method)
            repeat = _repeat_id(run_id, case_id, method)
            exports = {
                **common,
                "CASE": case_id,
                "METHOD": method,
                "REPEAT": repeat,
                "BENCHMARK_OUTPUT_DIR": str(timing_root),
            }
            if method == "canonical_legacy":
                exports["BENCHMARK_ORBITAL_ARTIFACT_OUT"] = str(artifact)
                role = "A: canonical legacy orbital producer and timing oracle"
            else:
                exports["BENCHMARK_ORBITAL_ARTIFACT_IN"] = str(artifact)
                role = "B: GPU-resident canonical consumer and timing candidate"
            expected_stem = f"{case_id}__{method}__r{repeat}"
            outputs = [
                str(timing_root / f"{expected_stem}.json"),
                str(timing_root / f"{expected_stem}.checkpoint.npz"),
            ]
            if method == "canonical_legacy":
                outputs.append(str(artifact))
            jobs.append({
                "id": job_id,
                "case_id": case_id,
                "method": method,
                "phase": "small-case-gate" if case_id in GATE_CASES else "water8",
                "ordinal": len(jobs),
                "launcher": str(launcher),
                "slurm_options": [f"--nodelist={NODE}"],
                "depends_on": [] if predecessor is None else [predecessor],
                "exports": exports,
                "expected_outputs": outputs,
                "role": role,
            })
            predecessor = job_id

    for job in jobs:
        for key, value in job["exports"].items():
            if any(character in value for character in ",\n\r"):
                raise ValueError(
                    f"job {job['id']!r} export {key!r} contains an unsafe "
                    "sbatch character"
                )

    return {
        "schema": PLAN_SCHEMA,
        "run_id": run_id,
        "task_root": str(task),
        "source_root": str(source),
        "source_tree_sha256": digest,
        "node": NODE,
        "initial_slurm_dependency": initial_dependency,
        "case_order": list(CASES),
        "method_order": list(METHODS),
        "result_root": str(result_root),
        "timing_result_root": str(timing_root),
        "policies": {
            "shared_orbitals": (
                "each legacy A process writes and reloads one exact canonical "
                "orbital artifact; its resident B process loads that artifact"
            ),
            "serialization": (
                "all six jobs form one logical afterok chain on compute-1-6"
            ),
            "two_stage_submission": (
                "only water2/water4 may execute initially; a passing analysis "
                "and its executed gate receipt are required to submit water8"
            ),
            "timing": (
                "fresh process; complete post-HF preprocessing, residual, and "
                "normal checkpoint are included"
            ),
        },
        "limits": {
            "correlation_energy_delta_eh": ENERGY_LIMIT_EH,
            "total_energy_delta_eh": ENERGY_LIMIT_EH,
            "residual_delta": RESIDUAL_LIMIT,
            "amplitude_max_abs": AMPLITUDE_LIMIT,
            "amplitude_l2": AMPLITUDE_LIMIT,
            "hbm_gib": HBM_LIMIT_GIB,
            "host_rss_gib": RSS_LIMIT_GIB,
        },
        "jobs": jobs,
    }


def _validate_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported G1 A/B plan schema")
    required = ("task_root", "source_root", "run_id")
    if any(not isinstance(value.get(key), str) for key in required):
        raise ValueError("G1 A/B plan identity is incomplete")
    expected = make_plan(
        task_root=Path(value["task_root"]),
        source_root=Path(value["source_root"]),
        run_id=value["run_id"],
        initial_dependency=value.get("initial_slurm_dependency"),
    )
    if value != expected:
        raise ValueError("G1 A/B plan differs from the fixed generated contract")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _at(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _record_path(record: dict[str, Any]) -> Path | None:
    value = record.get("_g1_ab_input_path")
    return Path(value) if isinstance(value, str) and value else None


def _load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for path in sorted(set(item.resolve() for item in files)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read JSON record {path}: {exc}") from exc
        if not isinstance(value, dict):
            continue
        if (
            value.get("case_id") in CASES
            and value.get("method") in METHODS
            and "post_hf_seconds" in value
        ):
            value = dict(value)
            value["_g1_ab_input_path"] = str(path)
            records.append(value)
    return records


def _resolve_checkpoint(record: dict[str, Any]) -> Path:
    json_path = _record_path(record)
    if json_path is None:
        raise ValueError("benchmark input path is unavailable")
    metadata = record.get("checkpoint")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata is missing")
    declared = metadata.get("path")
    candidates: list[Path] = []
    if isinstance(declared, str) and declared:
        candidates.append(Path(declared))
        candidates.append(json_path.parent / Path(declared).name)
    candidates.append(json_path.with_suffix(".checkpoint.npz"))
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError(
        f"checkpoint is unavailable beside synchronized record {json_path}"
    )


def _source_gate(record: dict[str, Any], expected: str) -> dict[str, Any]:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    snapshot = source.get("snapshot") if isinstance(source.get("snapshot"), dict) else {}
    manifest = snapshot.get("manifest") if isinstance(snapshot.get("manifest"), dict) else {}
    provenance = (
        manifest.get("local_provenance")
        if isinstance(manifest.get("local_provenance"), dict) else {}
    )
    checks = {
        "tree_sha256": source.get("tree_sha256") == expected,
        "tree_sha256_at_end": source.get("tree_sha256_at_end") == expected,
        "stable_during_run": source.get("stable_during_run") is True,
        "formal_performance_eligible": source.get("formal_performance_eligible") is True,
        "snapshot_tree_sha256": snapshot.get("tree_sha256") == expected,
        "snapshot_tree_sha256_at_end": snapshot.get("tree_sha256_at_end") == expected,
        "snapshot_stable": snapshot.get("stable_during_run") is True,
        "snapshot_read_only": snapshot.get("read_only_at_end") is True,
        "runtime_binaries_valid": snapshot.get("runtime_binaries_valid_at_end") is True,
        "deployment_profile": provenance.get("deployment_profile") == "candidate",
    }
    return {"passed": all(checks.values()), "checks": checks}


def _hardware_gate(record: dict[str, Any]) -> dict[str, Any]:
    gpu = record.get("gpu") if isinstance(record.get("gpu"), dict) else {}
    device = gpu.get("device_0") if isinstance(gpu.get("device_0"), dict) else {}
    pcie = record.get("pcie") if isinstance(record.get("pcie"), dict) else {}
    visible = pcie.get("visible_gpus") if isinstance(pcie.get("visible_gpus"), list) else []
    link = visible[0] if len(visible) == 1 and isinstance(visible[0], dict) else {}
    environment = (
        record.get("thread_environment")
        if isinstance(record.get("thread_environment"), dict) else {}
    )
    total_memory = _number(device.get("totalGlobalMem"))
    checks = {
        "node": record.get("host") == NODE,
        "cpu_model": "AMD EPYC 7513" in str(record.get("cpu_model", "")),
        "physical8_affinity": record.get("affinity") == list(range(24, 32)),
        "thread_count": all(
            str(environment.get(name, "")) == "8"
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "single_visible_gpu": gpu.get("device_count") == 1 and len(visible) == 1,
        "gpu_model": "A100-SXM4-80GB" in str(device.get("name", "")),
        "gpu_memory": (
            total_memory is not None
            and 79 * 1024**3 <= total_memory <= 82 * 1024**3
        ),
        "numa_node": str(link.get("numa_node", "")) == "3"
            and str(environment.get("GPU4PYSCF_NUMA", "")) == "3",
        "pcie_gen4": str(link.get("pcie_gen_current", "")) == "4",
        "pcie_x16": str(link.get("pcie_width_current", "")) == "16",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "host": record.get("host"),
            "affinity": record.get("affinity"),
            "gpu": device.get("name"),
            "pcie": link,
        },
    }


def _orbital_gate(
    legacy: dict[str, Any], resident: dict[str, Any], expected_source: str
) -> dict[str, Any]:
    left = legacy.get("canonical_orbitals")
    right = resident.get("canonical_orbitals")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {"passed": False, "checks": {"metadata_present": False}}
    checks: dict[str, bool] = {
        "metadata_present": True,
        "producer_mode": left.get("mode") == "written-and-reloaded",
        "consumer_mode": right.get("mode") == "loaded",
        "producer_applied": left.get("applied_to_mean_field") is True,
        "consumer_applied": right.get("applied_to_mean_field") is True,
        "outside_post_hf": (
            left.get("included_in_post_hf") is False
            and right.get("included_in_post_hf") is False
        ),
    }
    for field in (
        "schema", "artifact_path", "artifact_sha256", "orbital_fingerprint",
        "identity", "producer",
    ):
        checks[f"same_{field}"] = left.get(field) == right.get(field)
    sha = left.get("artifact_sha256")
    checks["artifact_sha256"] = (
        isinstance(sha, str) and SHA256_PATTERN.fullmatch(sha) is not None
    )
    identity = left.get("identity") if isinstance(left.get("identity"), dict) else {}
    checks["identity_case"] = identity.get("case_id") == legacy.get("case_id")
    producer = left.get("producer") if isinstance(left.get("producer"), dict) else {}
    checks["producer_source"] = producer.get("tree_sha256") == expected_source
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "artifact_sha256": sha,
        "orbital_fingerprint": left.get("orbital_fingerprint"),
    }


def _resource_gate(record: dict[str, Any]) -> dict[str, Any]:
    hbm_mib = _number(_at(record, "hbm.peak_process_MiB"))
    if hbm_mib is None:
        hbm_mib = _number(_at(record, "hbm.peak_process_mib"))
    hbm_gib = hbm_mib / 1024.0 if hbm_mib is not None else None
    rss_gib = _number(record.get("peak_host_RSS_GiB"))
    if rss_gib is None:
        rss_gib = _number(record.get("peak_host_rss_gib"))
    checks = {
        "hbm_measured": hbm_gib is not None,
        "hbm_within_limit": hbm_gib is not None and hbm_gib <= HBM_LIMIT_GIB,
        "rss_measured": rss_gib is not None,
        "rss_within_limit": rss_gib is not None and rss_gib <= RSS_LIMIT_GIB,
        "hbm_samples": (_number(_at(record, "hbm.sample_count")) or 0) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "hbm_gib": hbm_gib,
        "hbm_limit_gib": HBM_LIMIT_GIB,
        "rss_gib": rss_gib,
        "rss_limit_gib": RSS_LIMIT_GIB,
    }


def _transfer_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    transfers = _at(record, "run_metrics.transfers")
    if not isinstance(transfers, dict):
        return {"valid": False, "reason": "run_metrics.transfers is missing"}
    total_bytes = _number(transfers.get("total_bytes"))
    total_count = _number(transfers.get("total_transfers"))
    by_kind = transfers.get("by_kind") if isinstance(transfers.get("by_kind"), dict) else {}
    parsed: dict[str, dict[str, int]] = {}
    for kind in ("h2d", "d2h"):
        payload = by_kind.get(kind) if isinstance(by_kind.get(kind), dict) else {}
        byte_count = _number(payload.get("bytes"))
        operation_count = _number(payload.get("count"))
        if byte_count is None or operation_count is None:
            return {"valid": False, "reason": f"{kind} transfer totals are missing"}
        parsed[kind] = {"bytes": int(byte_count), "count": int(operation_count)}
    if total_bytes is None or total_count is None:
        return {"valid": False, "reason": "aggregate transfer totals are missing"}
    summed_bytes = sum(item["bytes"] for item in parsed.values())
    summed_count = sum(item["count"] for item in parsed.values())
    consistent = int(total_bytes) == summed_bytes and int(total_count) == summed_count
    by_operation = (
        transfers.get("by_operation")
        if isinstance(transfers.get("by_operation"), dict) else {}
    )
    host_staging_bytes = 0
    host_staging_count = 0
    operation_ledger_complete = True
    for kind in ("h2d", "d2h"):
        operations = by_operation.get(kind)
        if not isinstance(operations, dict):
            operation_ledger_complete = False
            continue
        for name, payload in operations.items():
            if "host-staging" not in str(name):
                continue
            if not isinstance(payload, dict):
                operation_ledger_complete = False
                continue
            byte_count = _number(payload.get("bytes"))
            count = _number(payload.get("count"))
            if byte_count is None or count is None:
                operation_ledger_complete = False
                continue
            host_staging_bytes += int(byte_count)
            host_staging_count += int(count)
    return {
        "valid": consistent and operation_ledger_complete,
        "aggregate_consistent": consistent,
        "operation_ledger_complete": operation_ledger_complete,
        "total_bytes": int(total_bytes),
        "total_transfers": int(total_count),
        "by_kind": parsed,
        "host_staging_bytes": host_staging_bytes,
        "host_staging_transfers": host_staging_count,
    }


def _residency_and_transfer_gate(
    legacy: dict[str, Any], resident: dict[str, Any]
) -> dict[str, Any]:
    baseline = _transfer_snapshot(legacy)
    candidate = _transfer_snapshot(resident)
    counters = _at(resident, "run_metrics.counters")
    counters_present = isinstance(counters, dict)
    counters = counters if counters_present else {}
    fallback_observed = _number(counters.get("direct_host_staging_fallbacks"))
    # RunMetrics is sparse: event counters that never fire are omitted rather
    # than serialized with zero.  Treat omission as zero only when the counter
    # map itself and the complete per-operation transfer ledger are present.
    fallback_count = 0.0 if fallback_observed is None else fallback_observed
    cycles = _number(resident.get("cc_iterations"))
    resident_iterations = _number(counters.get("resident_iterations"))
    direct_resident_updates = _number(
        counters.get("direct_resident_staging_iterations")
    )
    final_residual_updates = _number(
        counters.get("resident_final_residual_updates")
    )
    staging_release_events = _number(
        counters.get("final_residual_staging_release_events")
    )
    released_staging_bytes = _number(
        counters.get("final_residual_released_staging_bytes")
    )
    temporary_bound_bytes = _number(
        counters.get("final_residual_temporary_bound_bytes")
    )
    synchronized_driver_hbm_peak_bytes = _number(
        counters.get("synchronized_driver_hbm_peak_bytes")
    )
    resident_update_hbm_checkpoint_count = _number(
        counters.get("resident_update_hbm_checkpoint_count")
    )
    residual = (
        resident.get("residual")
        if isinstance(resident.get("residual"), dict) else {}
    )
    operations = _at(resident, "run_metrics.transfers.by_operation")
    operations = operations if isinstance(operations, dict) else {}
    h2d_operations = (
        operations.get("h2d")
        if isinstance(operations.get("h2d"), dict) else {}
    )
    d2h_operations = (
        operations.get("d2h")
        if isinstance(operations.get("d2h"), dict) else {}
    )

    def exact_operation(
        ledger: dict[str, Any], name: str, *, byte_count: int | None = None
    ) -> bool:
        payload = ledger.get(name)
        if not isinstance(payload, dict) or _number(payload.get("count")) != 1:
            return False
        return (
            byte_count is None
            or _number(payload.get("bytes")) == byte_count
        )

    static_uploads = (
        "resident-static-fock-upload",
        "resident-static-mo-energy-upload",
        "resident-static-mo-coeff-upload",
        "resident-static-oooo-upload",
        "resident-static-ovoo-upload",
        "resident-static-oovv-upload",
        "resident-static-ovvo-upload",
    )
    residual_scalar_downloads = (
        "final-residual-singles-equation-norm-download",
        "final-residual-doubles-equation-norm-download",
        "final-residual-singles-jacobi-update-norm-download",
        "final-residual-doubles-jacobi-update-norm-download",
        "final-residual-full-equation-norm-download",
        "final-residual-full-jacobi-update-norm-download",
    )
    def parse_hbm_checkpoints(value: Any) -> tuple[bool, list[str], list[float]]:
        names: list[str] = []
        driver_peaks: list[float] = []
        valid = isinstance(value, list)
        if not valid:
            return False, names, driver_peaks
        for checkpoint in value:
            if not isinstance(checkpoint, dict):
                valid = False
                break
            name = checkpoint.get("name")
            driver_used = _number(checkpoint.get("driver_used_bytes"))
            driver_total = _number(checkpoint.get("driver_total_bytes"))
            pool_used = _number(checkpoint.get("cupy_pool_used_bytes"))
            pool_total = _number(checkpoint.get("cupy_pool_total_bytes"))
            if (
                not isinstance(name, str)
                or driver_used is None
                or driver_total is None
                or pool_used is None
                or pool_total is None
                or not 0 <= driver_used <= driver_total
                or not 0 <= pool_used <= pool_total <= driver_used
                or checkpoint.get("timing_semantics")
                    != "cuda-synchronized-instantaneous"
                or checkpoint.get("measurement_scope")
                    != "exclusive-slurm-device-global"
                or checkpoint.get("source")
                    != "cudaMemGetInfo-and-cupy-default-memory-pool"
            ):
                valid = False
                break
            names.append(name)
            driver_peaks.append(driver_used)
        return valid, names, driver_peaks

    synchronized_hbm_checkpoints = residual.get(
        "synchronized_hbm_checkpoints"
    )
    final_residual_hbm_peak = _number(residual.get(
        "final_residual_synchronized_driver_hbm_peak_bytes"
    ))
    whole_stage_hbm_peak = _number(residual.get(
        "whole_stage_synchronized_driver_hbm_peak_bytes"
    ))
    (
        checkpoints_valid,
        checkpoint_names,
        checkpoint_driver_peaks,
    ) = parse_hbm_checkpoints(synchronized_hbm_checkpoints)
    all_hbm_checkpoints = _at(
        resident, "run_metrics.metadata.synchronized_hbm_checkpoints"
    )
    (
        all_checkpoints_valid,
        all_checkpoint_names,
        all_checkpoint_driver_peaks,
    ) = parse_hbm_checkpoints(all_hbm_checkpoints)
    active_nvir = _number(residual.get("active_nvir"))
    residual_block_size = _number(residual.get("doubles_block_size"))
    expected_block_checkpoint_names = None
    if (
        active_nvir is not None
        and residual_block_size is not None
        and active_nvir >= 1
        and residual_block_size >= 1
    ):
        block_size_integer = int(residual_block_size)
        active_nvir_integer = int(active_nvir)
        expected_block_checkpoint_names = [
            f"final-residual-block-{a0}-{min(active_nvir_integer, a0 + block_size_integer)}-live"
            for a0 in range(0, active_nvir_integer, block_size_integer)
        ]
    expected_update_checkpoint_names = [
        "final-residual-internal-direct-transform-live",
        "resident-update-direct-output-live",
        "final-residual-internal-direct-output-live",
        "final-residual-internal-residual_fock_intermediates",
        "final-residual-internal-residual_oooo_ladder",
        "final-residual-internal-residual_linear_ov",
        "final-residual-internal-residual_wVooV_ring",
        "final-residual-internal-residual_fock_dressing",
        "final-residual-internal-residual_wVOov_ring",
        "final-residual-internal-residual_singles_doubles_coupling",
        "final-residual-internal-residual_one_body_doubles",
        "final-residual-internal-residual_denominator_and_symmetry",
        "final-residual-update-live",
    ]
    synchronized_checkpoint_set_complete = (
        checkpoints_valid
        and expected_block_checkpoint_names is not None
        and checkpoint_names == [
            *expected_update_checkpoint_names,
            *expected_block_checkpoint_names,
        ]
    )
    expected_resident_update_peak_count = (
        None if cycles is None else int(cycles) + 1
    )
    resident_update_peak_count = all_checkpoint_names.count(
        "resident-update-direct-output-live"
    )
    resident_update_peaks_complete = (
        all_checkpoints_valid
        and expected_resident_update_peak_count is not None
        and resident_update_peak_count == expected_resident_update_peak_count
        and resident_update_hbm_checkpoint_count
            == expected_resident_update_peak_count
    )
    checks = {
        "legacy_transfer_ledger": baseline.get("valid") is True,
        "resident_transfer_ledger": candidate.get("valid") is True,
        "event_counter_ledger_present": counters_present,
        "no_host_staging_fallback": counters_present and fallback_count == 0.0,
        "no_host_staging_transfer": (
            candidate.get("host_staging_bytes") == 0
            and candidate.get("host_staging_transfers") == 0
        ),
        "all_iterations_resident": (
            cycles is not None
            and resident_iterations is not None
            and resident_iterations == cycles
        ),
        "final_residual_update_resident": final_residual_updates == 1,
        "direct_staging_covers_iterations_and_final_residual": (
            cycles is not None
            and direct_resident_updates is not None
            and direct_resident_updates == cycles + 1
        ),
        "static_eri_uploaded_once": all(
            exact_operation(h2d_operations, name) for name in static_uploads
        ),
        "final_residual_norm_scalars_accounted": all(
            exact_operation(d2h_operations, name, byte_count=8)
            for name in residual_scalar_downloads
        ),
        "final_residual_gpu_blockwise": (
            residual.get("evaluation_backend") == "gpu-blockwise-fp64"
            and _number(residual.get("doubles_block_size")) is not None
            and _number(residual.get("doubles_block_size")) >= 1
        ),
        "final_residual_workspace_bounded": (
            temporary_bound_bytes is not None
            and 0 < temporary_bound_bytes <= 3 * 128 * 1024**2
            and _number(residual.get("temporary_workspace_bound_bytes"))
                == temporary_bound_bytes
        ),
        "final_residual_staging_released": (
            staging_release_events == 1
            and released_staging_bytes is not None
            and released_staging_bytes > 0
            and _number(residual.get("released_staging_bytes"))
                == released_staging_bytes
        ),
        "synchronized_hbm_checkpoints_complete": (
            synchronized_checkpoint_set_complete
            and resident_update_peaks_complete
        ),
        "synchronized_hbm_within_budget": (
            synchronized_checkpoint_set_complete
            and resident_update_peaks_complete
            and synchronized_driver_hbm_peak_bytes is not None
            and synchronized_driver_hbm_peak_bytes
                == max(all_checkpoint_driver_peaks)
            and final_residual_hbm_peak
                == max(checkpoint_driver_peaks)
            and whole_stage_hbm_peak
                == synchronized_driver_hbm_peak_bytes
            and synchronized_driver_hbm_peak_bytes <= 72 * 1024**3
        ),
        "total_transfer_not_regressed": (
            baseline.get("valid") is True
            and candidate.get("valid") is True
            and candidate["total_bytes"] <= baseline["total_bytes"]
            and candidate["total_transfers"] <= baseline["total_transfers"]
        ),
        "h2d_not_regressed": (
            baseline.get("valid") is True
            and candidate.get("valid") is True
            and candidate["by_kind"]["h2d"]["bytes"]
                <= baseline["by_kind"]["h2d"]["bytes"]
            and candidate["by_kind"]["h2d"]["count"]
                <= baseline["by_kind"]["h2d"]["count"]
        ),
        "d2h_not_regressed": (
            baseline.get("valid") is True
            and candidate.get("valid") is True
            and candidate["by_kind"]["d2h"]["bytes"]
                <= baseline["by_kind"]["d2h"]["bytes"]
            and candidate["by_kind"]["d2h"]["count"]
                <= baseline["by_kind"]["d2h"]["count"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "legacy": baseline,
        "resident": candidate,
        "direct_host_staging_fallbacks": fallback_count,
        "direct_host_staging_counter_serialized": fallback_observed is not None,
        "resident_iterations": resident_iterations,
        "direct_resident_updates": direct_resident_updates,
        "final_residual_updates": final_residual_updates,
        "final_residual_temporary_bound_bytes": temporary_bound_bytes,
        "final_residual_released_staging_bytes": released_staging_bytes,
        "synchronized_driver_hbm_peak_bytes": (
            synchronized_driver_hbm_peak_bytes),
        "final_residual_synchronized_driver_hbm_peak_bytes": (
            final_residual_hbm_peak),
        "whole_stage_synchronized_driver_hbm_peak_bytes": (
            whole_stage_hbm_peak),
        "resident_update_hbm_checkpoint_count": (
            resident_update_hbm_checkpoint_count),
        "cc_iterations": cycles,
    }


def _expected_relative_path(plan: dict[str, Any], value: str) -> Path:
    try:
        relative = Path(value).relative_to(Path(plan["result_root"]))
    except ValueError as exc:
        raise ValueError(
            f"planned output is outside the G1 result root: {value}"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"planned output has an unsafe relative path: {value}")
    return relative


def _local_result_root(local_json: Path, relative_json: Path) -> Path:
    resolved = local_json.expanduser().resolve()
    if len(resolved.parts) < len(relative_json.parts):
        raise ValueError("local JSON path is shorter than its planned relative path")
    if tuple(resolved.parts[-len(relative_json.parts):]) != relative_json.parts:
        raise ValueError(
            "local input JSON does not preserve the planned result-tree suffix "
            f"{relative_json}"
        )
    root = resolved
    for _ in relative_json.parts:
        root = root.parent
    return root


def _record_receipt_gate(
    record: dict[str, Any],
    job: dict[str, Any],
    receipt_job: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Bind one parsed result and its files to one actual submitted job."""

    expected_outputs = job["expected_outputs"]
    expected_json = expected_outputs[0]
    expected_checkpoint = expected_outputs[1]
    local_json = _record_path(record)
    checks: dict[str, bool] = {
        "case_id": record.get("case_id") == receipt_job["case_id"],
        "method": record.get("method") == receipt_job["method"],
        "repeat": record.get("repeat") == receipt_job["repeat"],
        "record_output": record.get("output") == expected_json,
    }
    slurm = record.get("slurm") if isinstance(record.get("slurm"), dict) else {}
    checks.update({
        "slurm_job_id": str(slurm.get("SLURM_JOB_ID", ""))
            == receipt_job["job_id"],
        "slurm_job_name": slurm.get("SLURM_JOB_NAME") == job["id"],
        "slurm_node": slurm.get("SLURM_JOB_NODELIST") == NODE,
        "slurm_partition": slurm.get("SLURM_JOB_PARTITION") == "mrigpu",
    })
    local_root: Path | None = None
    actual_checkpoint: Path | None = None
    actual_artifact: Path | None = None
    path_error: str | None = None
    try:
        if local_json is None or not local_json.is_file():
            raise ValueError("local input JSON is not a regular file")
        relative_json = _expected_relative_path(plan, expected_json)
        local_root = _local_result_root(local_json, relative_json)
        checks["local_input_json"] = (
            local_json.resolve() == (local_root / relative_json).resolve()
        )

        checkpoint = (
            record.get("checkpoint")
            if isinstance(record.get("checkpoint"), dict) else {}
        )
        checks["checkpoint_declared_path"] = (
            checkpoint.get("path") == expected_checkpoint
        )
        relative_checkpoint = _expected_relative_path(
            plan, expected_checkpoint
        )
        expected_local_checkpoint = (local_root / relative_checkpoint).resolve()
        actual_checkpoint = _resolve_checkpoint(record).resolve()
        checks["checkpoint_local_path"] = (
            actual_checkpoint == expected_local_checkpoint
        )
        declared_checkpoint_sha = checkpoint.get("sha256")
        checks["checkpoint_declared_sha256"] = (
            isinstance(declared_checkpoint_sha, str)
            and SHA256_PATTERN.fullmatch(declared_checkpoint_sha) is not None
        )
        checks["checkpoint_actual_sha256"] = (
            checks["checkpoint_declared_sha256"]
            and _file_sha256(actual_checkpoint) == declared_checkpoint_sha
        )

        if job["method"] == "canonical_legacy":
            expected_artifact = expected_outputs[2]
            orbitals = (
                record.get("canonical_orbitals")
                if isinstance(record.get("canonical_orbitals"), dict) else {}
            )
            checks["orbital_artifact_declared_path"] = (
                orbitals.get("artifact_path") == expected_artifact
            )
            relative_artifact = _expected_relative_path(plan, expected_artifact)
            actual_artifact = (local_root / relative_artifact).resolve()
            checks["orbital_artifact_local_path"] = actual_artifact.is_file()
            declared_artifact_sha = orbitals.get("artifact_sha256")
            checks["orbital_artifact_declared_sha256"] = (
                isinstance(declared_artifact_sha, str)
                and SHA256_PATTERN.fullmatch(declared_artifact_sha) is not None
            )
            checks["orbital_artifact_actual_sha256"] = (
                checks["orbital_artifact_local_path"]
                and checks["orbital_artifact_declared_sha256"]
                and _file_sha256(actual_artifact) == declared_artifact_sha
            )
    except (ValueError, OSError) as exc:
        path_error = str(exc)
        checks["local_file_binding"] = False
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "logical_job": job["id"],
        "slurm_job_id": receipt_job["job_id"],
        "expected_outputs": list(expected_outputs),
        "local_input_json": str(local_json) if local_json else None,
        "local_result_root": str(local_root) if local_root else None,
        "local_checkpoint": (
            str(actual_checkpoint) if actual_checkpoint else None
        ),
        "local_orbital_artifact": (
            str(actual_artifact) if actual_artifact else None
        ),
        "error": path_error,
    }


def _pair_row(
    case_id: str,
    legacy_matches: list[dict[str, Any]],
    resident_matches: list[dict[str, Any]],
    expected_source: str,
    plan: dict[str, Any],
    legacy_job: dict[str, Any],
    resident_job: dict[str, Any],
    receipt_jobs: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if receipt_jobs is None:
        reasons.append(f"executed receipt is missing for {case_id}")
    if len(legacy_matches) != 1:
        reasons.append(
            f"expected exactly one canonical_legacy record, found {len(legacy_matches)}"
        )
    if len(resident_matches) != 1:
        reasons.append(
            f"expected exactly one canonical resident record, found {len(resident_matches)}"
        )
    row: dict[str, Any] = {
        "case_id": case_id,
        "complete": not reasons,
        "passed": False,
        "reasons": reasons,
    }
    if reasons:
        return row
    legacy, resident = legacy_matches[0], resident_matches[0]
    left_path, right_path = _record_path(legacy), _record_path(resident)
    row["records"] = {
        "legacy": str(left_path) if left_path else None,
        "resident": str(right_path) if right_path else None,
    }
    source = {
        "legacy": _source_gate(legacy, expected_source),
        "resident": _source_gate(resident, expected_source),
    }
    hardware = {
        "legacy": _hardware_gate(legacy),
        "resident": _hardware_gate(resident),
    }
    orbitals = _orbital_gate(legacy, resident, expected_source)
    resources = _resource_gate(resident)
    residency = _residency_and_transfer_gate(legacy, resident)
    receipt_binding = {
        "legacy": _record_receipt_gate(
            legacy, legacy_job, receipt_jobs[legacy_job["id"]], plan
        ),
        "resident": _record_receipt_gate(
            resident, resident_job, receipt_jobs[resident_job["id"]], plan
        ),
    }
    legacy_corr = _number(legacy.get("e_corr"))
    resident_corr = _number(resident.get("e_corr"))
    legacy_total = _number(legacy.get("e_tot"))
    resident_total = _number(resident.get("e_tot"))
    corr_delta = (
        abs(legacy_corr - resident_corr)
        if legacy_corr is not None and resident_corr is not None else None
    )
    total_delta = (
        abs(legacy_total - resident_total)
        if legacy_total is not None and resident_total is not None else None
    )
    energy = {
        "correlation_delta_eh": corr_delta,
        "total_delta_eh": total_delta,
        "tolerance_eh": ENERGY_LIMIT_EH,
        "passed": (
            corr_delta is not None and corr_delta <= ENERGY_LIMIT_EH
            and total_delta is not None and total_delta <= ENERGY_LIMIT_EH
        ),
    }
    residual_values = {
        "legacy_equation_norm": _number(_at(legacy, "residual.equation_norm")),
        "resident_equation_norm": _number(_at(resident, "residual.equation_norm")),
        "legacy_jacobi_update_norm": _number(
            _at(legacy, "residual.jacobi_update_norm")
        ),
        "resident_jacobi_update_norm": _number(
            _at(resident, "residual.jacobi_update_norm")
        ),
    }
    residual = {
        **residual_values,
        "tolerance": RESIDUAL_LIMIT,
        "passed": all(
            value is not None and value <= RESIDUAL_LIMIT
            for value in residual_values.values()
        ),
    }
    legacy_seconds = _number(legacy.get("post_hf_seconds"))
    resident_seconds = _number(resident.get("post_hf_seconds"))
    timing = {
        "legacy_post_hf_seconds": legacy_seconds,
        "resident_post_hf_seconds": resident_seconds,
        "speedup": (
            legacy_seconds / resident_seconds
            if legacy_seconds is not None and resident_seconds is not None
            and resident_seconds > 0 else None
        ),
        "acceptance_gate": False,
    }
    try:
        if left_path is None or right_path is None:
            raise ValueError("benchmark input paths are unavailable")
        checkpoint = _checkpoint_comparator.compare_checkpoints(
            left_path,
            right_path,
            left_checkpoint=_resolve_checkpoint(legacy),
            right_checkpoint=_resolve_checkpoint(resident),
            energy_tol=ENERGY_LIMIT_EH,
            residual_tol=RESIDUAL_LIMIT,
            amplitude_max_tol=AMPLITUDE_LIMIT,
            amplitude_l2_tol=AMPLITUDE_LIMIT,
        )
        checkpoint_passed = checkpoint.get("passed") is True
        if not checkpoint_passed:
            reasons.append("strict checkpoint energy/residual/amplitude comparison failed")
    except (ValueError, OSError) as exc:
        checkpoint = {"passed": False, "error": str(exc)}
        checkpoint_passed = False
        reasons.append(f"strict checkpoint comparison could not be completed: {exc}")

    gates = {
        "source": all(item["passed"] for item in source.values()),
        "hardware": all(item["passed"] for item in hardware.values()),
        "shared_orbitals": orbitals["passed"],
        "energy": energy["passed"],
        "absolute_residual_and_update": residual["passed"],
        "checkpoint": checkpoint_passed,
        "resident_resources": resources["passed"],
        "residency_and_transfers": residency["passed"],
        "receipt_and_files": all(
            item["passed"] for item in receipt_binding.values()
        ),
    }
    for label, passed in gates.items():
        if not passed and not any(label in reason for reason in reasons):
            reasons.append(f"{label} gate failed")
    row.update({
        "source": source,
        "hardware": hardware,
        "canonical_orbitals": orbitals,
        "energy": energy,
        "absolute_residual_and_update": residual,
        "checkpoint_comparison": checkpoint,
        "resident_resources": resources,
        "residency_and_transfers": residency,
        "receipt_and_files": receipt_binding,
        "timing": timing,
        "gates": gates,
        "passed": all(gates.values()),
        "reasons": reasons,
    })
    return row


def analyze_g1_ab(
    records: Iterable[dict[str, Any]],
    plan: dict[str, Any],
    *,
    gate_receipt: dict[str, Any],
    gate_receipt_evidence: dict[str, Any],
    water8_receipt: dict[str, Any] | None = None,
    water8_receipt_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit records bound to exact planned and receipted Slurm jobs."""

    plan = _validate_plan(plan)
    gate_jobs = _validate_submission_receipt(gate_receipt, plan, "gate")
    gate_summary = _receipt_summary(
        gate_receipt, gate_receipt_evidence, plan, "gate"
    )
    water8_jobs: dict[str, dict[str, Any]] | None = None
    water8_summary: dict[str, Any] | None = None
    if water8_receipt is not None:
        if water8_receipt_evidence is None:
            raise ValueError("water8 receipt file evidence is missing")
        water8_jobs = _validate_submission_receipt(
            water8_receipt, plan, "water8"
        )
        water8_summary = _receipt_summary(
            water8_receipt, water8_receipt_evidence, plan, "water8"
        )
    elif water8_receipt_evidence is not None:
        raise ValueError("water8 receipt evidence was supplied without a receipt")
    inputs = list(records)
    expected_repeats = {
        (job["case_id"], job["method"]): job["exports"]["REPEAT"]
        for job in plan["jobs"]
    }
    rows: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    planned_jobs = {
        (job["case_id"], job["method"]): job for job in plan["jobs"]
    }
    for case_id in CASES:
        matches: dict[str, list[dict[str, Any]]] = {}
        for method in METHODS:
            repeat = expected_repeats[(case_id, method)]
            matches[method] = [
                record for record in inputs
                if record.get("case_id") == case_id
                and record.get("method") == method
                and record.get("repeat") == repeat
            ]
            for record in matches[method]:
                path = record.get("_g1_ab_input_path")
                if isinstance(path, str):
                    selected_paths.add(path)
        receipt_jobs = gate_jobs if case_id in GATE_CASES else water8_jobs
        rows.append(_pair_row(
            case_id,
            matches["canonical_legacy"],
            matches["canonical"],
            plan["source_tree_sha256"],
            plan,
            planned_jobs[(case_id, "canonical_legacy")],
            planned_jobs[(case_id, "canonical")],
            receipt_jobs,
        ))
    by_case = {row["case_id"]: row for row in rows}
    ready_for_water8 = all(by_case[case]["passed"] for case in GATE_CASES)
    water8_complete = by_case["water8-tz"]["complete"]
    final_passed = ready_for_water8 and by_case["water8-tz"]["passed"]
    ignored = max(0, len(inputs) - len(selected_paths))
    return {
        "schema": SCHEMA,
        "run_id": plan["run_id"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_sha256": _plan_sha256(plan),
        "node": NODE,
        "limits": plan["limits"],
        "records_considered": len(inputs),
        "records_selected": len(selected_paths),
        "records_ignored": ignored,
        "receipts": {
            "gate": gate_summary,
            "water8": water8_summary,
        },
        "rows": rows,
        "decision": {
            "small_case_gate_passed": ready_for_water8,
            "ready_for_water8": ready_for_water8,
            "water8_complete": water8_complete,
            "final_g1_passed": final_passed,
            "water8_submission_policy": (
                "requires ready_for_water8=true plus an executed small-case receipt"
            ),
        },
    }


def _validate_submission_receipt(
    receipt: dict[str, Any], plan: dict[str, Any], stage: str
) -> dict[str, dict[str, Any]]:
    if receipt.get("schema") != SUBMISSION_SCHEMA:
        raise ValueError("unsupported prior G1 A/B receipt schema")
    if receipt.get("executed") is not True or receipt.get("stage") != stage:
        raise ValueError(f"prior receipt is not an executed {stage!r} submission")
    if receipt.get("run_id") != plan["run_id"]:
        raise ValueError("prior receipt run id differs from the plan")
    if receipt.get("source_tree_sha256") != plan["source_tree_sha256"]:
        raise ValueError("prior receipt source digest differs from the plan")
    if receipt.get("plan_sha256") != _plan_sha256(plan):
        raise ValueError("prior receipt plan digest differs from the plan")
    expected_jobs = [
        job for job in plan["jobs"]
        if (job["phase"] == "small-case-gate") == (stage == "gate")
    ]
    jobs = receipt.get("jobs")
    if (
        not isinstance(jobs, list)
        or [item.get("job") for item in jobs]
            != [job["id"] for job in expected_jobs]
    ):
        raise ValueError("prior receipt does not contain the exact expected jobs")
    result: dict[str, dict[str, Any]] = {}
    seen_slurm_ids: set[str] = set()
    for item, expected in zip(jobs, expected_jobs):
        if not isinstance(item, dict):
            raise ValueError("prior receipt contains a malformed job entry")
        job_id = item.get("job_id")
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError("prior receipt contains a non-numeric Slurm job id")
        if job_id in seen_slurm_ids:
            raise ValueError("prior receipt reuses a Slurm job id")
        seen_slurm_ids.add(job_id)
        required = {
            "job": expected["id"],
            "case_id": expected["case_id"],
            "method": expected["method"],
            "repeat": expected["exports"]["REPEAT"],
            "expected_outputs": expected["expected_outputs"],
        }
        for key, expected_value in required.items():
            if item.get(key) != expected_value:
                raise ValueError(
                    f"receipt job {expected['id']!r} has incorrect {key}"
                )
        result[expected["id"]] = item
    return result


def _receipt_summary(
    receipt: dict[str, Any],
    evidence: dict[str, Any],
    plan: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    jobs = _validate_submission_receipt(receipt, plan, stage)
    payload_sha256 = _payload_sha256(receipt)
    evidence_payload_sha = evidence.get("payload_sha256")
    file_sha256 = evidence.get("file_sha256")
    path = evidence.get("path")
    if evidence_payload_sha != payload_sha256:
        raise ValueError(f"{stage} receipt payload evidence does not match")
    if (
        not isinstance(file_sha256, str)
        or SHA256_PATTERN.fullmatch(file_sha256) is None
        or not isinstance(path, str)
        or not path
    ):
        raise ValueError(f"{stage} receipt file evidence is incomplete")
    return {
        "stage": stage,
        "path": path,
        "file_sha256": file_sha256,
        "payload_sha256": payload_sha256,
        "plan_sha256": receipt["plan_sha256"],
        "jobs": [
            {
                "logical_job": job_id,
                "slurm_job_id": item["job_id"],
                "case_id": item["case_id"],
                "method": item["method"],
                "repeat": item["repeat"],
                "expected_outputs": list(item["expected_outputs"]),
            }
            for job_id, item in jobs.items()
        ],
    }


def _receipt_content_identity(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the portable identity of a receipt, excluding display path."""

    return {
        key: summary.get(key)
        for key in (
            "stage", "file_sha256", "payload_sha256", "plan_sha256", "jobs"
        )
    }


def _validate_gate_analysis(
    analysis: dict[str, Any],
    plan: dict[str, Any],
    prior_receipt: dict[str, Any],
    prior_receipt_evidence: dict[str, Any],
) -> None:
    if analysis.get("schema") != SCHEMA:
        raise ValueError("unsupported G1 A/B gate analysis schema")
    for key, expected in (
        ("run_id", plan["run_id"]),
        ("source_tree_sha256", plan["source_tree_sha256"]),
        ("plan_sha256", _plan_sha256(plan)),
    ):
        if analysis.get(key) != expected:
            raise ValueError(f"gate analysis {key} differs from the plan")
    decision = analysis.get("decision")
    if not isinstance(decision, dict) or decision.get("ready_for_water8") is not True:
        raise ValueError("gate analysis does not authorize water8")
    rows = analysis.get("rows")
    if not isinstance(rows, list):
        raise ValueError("gate analysis rows are missing")
    by_case = {
        row.get("case_id"): row
        for row in rows if isinstance(row, dict)
    }
    if any(
        not isinstance(by_case.get(case), dict)
        or by_case[case].get("passed") is not True
        for case in GATE_CASES
    ):
        raise ValueError("water2 and water4 rows did not both pass")
    receipt_summary = _receipt_summary(
        prior_receipt, prior_receipt_evidence, plan, "gate"
    )
    receipts = analysis.get("receipts")
    analysis_gate = (
        receipts.get("gate") if isinstance(receipts, dict) else None
    )
    if (
        not isinstance(analysis_gate, dict)
        or _receipt_content_identity(analysis_gate)
            != _receipt_content_identity(receipt_summary)
    ):
        raise ValueError(
            "gate analysis is not bound to the supplied prior gate receipt"
        )


def _stage_jobs(plan: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "all":
        return list(plan["jobs"])
    phase = "small-case-gate" if stage == "gate" else "water8"
    return [job for job in plan["jobs"] if job["phase"] == phase]


def _sbatch_argv(
    job: dict[str, Any],
    submitted: dict[str, str],
    *,
    initial_dependency: str | None,
    terminal_dependencies: frozenset[str] = frozenset(),
) -> list[str]:
    # A separately validated gate analysis proves that every small-case job
    # reached its terminal successful state before the water8 stage is
    # submitted.  Some Slurm installations reject a *new* afterok edge to a
    # job that has already aged out of the controller's active-job table.  Do
    # not recreate those redundant edges; dependencies created within the
    # current stage (water8 resident -> water8 legacy) remain mandatory.
    dependencies = [
        submitted[item]
        for item in job["depends_on"]
        if item not in terminal_dependencies
    ]
    exports = ["ALL"] + [
        f"{key}={value}" for key, value in sorted(job["exports"].items())
    ]
    argv = ["sbatch", "--parsable", f"--job-name={job['id']}"]
    argv.extend(job["slurm_options"])
    if dependencies:
        argv.append("--dependency=afterok:" + ":".join(dependencies))
    elif initial_dependency is not None:
        argv.append("--dependency=" + initial_dependency)
    argv.extend(["--export=" + ",".join(exports), job["launcher"]])
    return argv


def submit_plan(
    plan: dict[str, Any],
    *,
    execute: bool,
    stage: str = "all",
    gate_analysis: dict[str, Any] | None = None,
    prior_receipt: dict[str, Any] | None = None,
    prior_receipt_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preview or submit a validated stage in dependency order."""

    plan = _validate_plan(plan)
    if stage not in {"all", "gate", "water8"}:
        raise ValueError("stage must be all, gate, or water8")
    if execute and stage == "all":
        raise ValueError(
            "the six-job plan cannot execute in one step; submit gate, analyze, "
            "then submit water8"
        )
    if stage != "water8" and (
        gate_analysis is not None
        or prior_receipt is not None
        or prior_receipt_evidence is not None
    ):
        raise ValueError("gate analysis and prior receipt apply only to water8")
    submitted: dict[str, str] = {}
    terminal_dependencies: frozenset[str] = frozenset()
    if stage == "water8":
        if execute and (
            gate_analysis is None
            or prior_receipt is None
            or prior_receipt_evidence is None
        ):
            raise ValueError(
                "water8 execution requires a passing gate analysis and gate receipt"
            )
        if prior_receipt is not None:
            if prior_receipt_evidence is None:
                raise ValueError("prior gate receipt file evidence is missing")
            prior_jobs = _validate_submission_receipt(
                prior_receipt, plan, "gate"
            )
            submitted.update({
                job_id: item["job_id"] for job_id, item in prior_jobs.items()
            })
        else:
            submitted[_job_id("water4-tz", "canonical")] = (
                f"<{_job_id('water4-tz', 'canonical')}>"
            )
        if gate_analysis is not None:
            if prior_receipt is None or prior_receipt_evidence is None:
                raise ValueError(
                    "gate analysis validation requires the prior gate receipt"
                )
            _validate_gate_analysis(
                gate_analysis, plan, prior_receipt, prior_receipt_evidence
            )
            terminal_dependencies = frozenset(prior_jobs)
    jobs = _stage_jobs(plan, stage)
    commands: list[dict[str, Any]] = []
    for job in jobs:
        argv = _sbatch_argv(
            job,
            submitted,
            initial_dependency=(
                plan["initial_slurm_dependency"] if job["ordinal"] == 0 else None
            ),
            terminal_dependencies=terminal_dependencies,
        )
        if execute:
            output = subprocess.check_output(argv, text=True).strip()
            job_id = output.split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(
                    f"sbatch returned an invalid job id for {job['id']}: {output!r}"
                )
        else:
            job_id = f"<{job['id']}>"
        submitted[job["id"]] = job_id
        commands.append({
            "job": job["id"],
            "job_id": job_id,
            "case_id": job["case_id"],
            "method": job["method"],
            "repeat": job["exports"]["REPEAT"],
            "expected_outputs": list(job["expected_outputs"]),
            "argv": argv,
            "shell_preview": shlex.join(argv),
        })
    return {
        "schema": SUBMISSION_SCHEMA,
        "executed": execute,
        "stage": stage,
        "run_id": plan["run_id"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_sha256": _plan_sha256(plan),
        "jobs": commands,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write the fixed six-job plan")
    plan.add_argument("--task-root", type=Path, required=True)
    plan.add_argument("--source-root", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument(
        "--initial-dependency",
        default=None,
        help="optional Slurm dependency, for example afterok:62434",
    )
    plan.add_argument("--output", type=Path, default=None)

    analyze = subparsers.add_parser(
        "analyze", help="audit synchronized legacy/resident benchmark pairs"
    )
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--records", type=Path, nargs="+", required=True)
    analyze.add_argument(
        "--gate-receipt", type=Path, required=True,
        help="executed water2/water4 submission receipt",
    )
    analyze.add_argument(
        "--water8-receipt", type=Path,
        help="executed water8 submission receipt, required for final analysis",
    )
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument(
        "--require-water8", action="store_true",
        help="return success only when the final water8 pair also passes",
    )

    submit = subparsers.add_parser(
        "submit", help="preview a plan stage, or call sbatch with --execute"
    )
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--stage", choices=("all", "gate", "water8"), default="all")
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--gate-analysis", type=Path)
    submit.add_argument("--prior-receipt", type=Path)
    submit.add_argument(
        "--receipt", type=Path, default=None,
        help="required with --execute; written once after submissions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        payload = make_plan(
            task_root=args.task_root,
            source_root=args.source_root,
            run_id=args.run_id,
            initial_dependency=args.initial_dependency,
        )
        if args.output is None:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _json_write_once(args.output, payload)
            print(args.output)
        return 0
    if args.command == "analyze":
        plan = _read_json(args.plan, "G1 A/B plan")
        gate_receipt, gate_evidence = _read_json_evidence(
            args.gate_receipt, "G1 A/B gate receipt"
        )
        if args.require_water8 and args.water8_receipt is None:
            raise ValueError("--require-water8 requires --water8-receipt")
        if args.water8_receipt is None:
            water8_receipt = None
            water8_evidence = None
        else:
            water8_receipt, water8_evidence = _read_json_evidence(
                args.water8_receipt, "G1 A/B water8 receipt"
            )
        payload = analyze_g1_ab(
            _load_records(args.records),
            plan,
            gate_receipt=gate_receipt,
            gate_receipt_evidence=gate_evidence,
            water8_receipt=water8_receipt,
            water8_receipt_evidence=water8_evidence,
        )
        _json_write_once(args.output, payload)
        print(json.dumps(payload["decision"], indent=2, sort_keys=True))
        passed = (
            payload["decision"]["final_g1_passed"]
            if args.require_water8 else payload["decision"]["ready_for_water8"]
        )
        return 0 if passed else 1

    plan = _read_json(args.plan, "G1 A/B plan")
    if args.execute and args.receipt is None:
        raise ValueError("--execute requires --receipt")
    if not args.execute and args.receipt is not None:
        raise ValueError("--receipt is used only with --execute")
    if args.execute and args.receipt is not None:
        # This check deliberately happens before reading validation evidence or
        # entering submit_plan, which is the only code path that can call
        # sbatch.  Thus an existing receipt always means zero submissions.
        _preflight_output_absent(args.receipt, "submission receipt")
    gate_analysis = (
        _read_json(args.gate_analysis, "G1 A/B gate analysis")
        if args.gate_analysis is not None else None
    )
    if args.prior_receipt is None:
        prior_receipt = None
        prior_receipt_evidence = None
    else:
        prior_receipt, prior_receipt_evidence = _read_json_evidence(
            args.prior_receipt, "G1 A/B prior receipt"
        )
    payload = submit_plan(
        plan,
        execute=args.execute,
        stage=args.stage,
        gate_analysis=gate_analysis,
        prior_receipt=prior_receipt,
        prior_receipt_evidence=prior_receipt_evidence,
    )
    if args.receipt is not None:
        _json_write_once(args.receipt, payload)
        print(args.receipt)
    else:
        for item in payload["jobs"]:
            print(item["shell_preview"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
